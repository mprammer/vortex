#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Single entry point that (re)creates the OnPair compression benchmark.

For every column in ``columns.COLUMNS`` and every ``bits × chunk × threshold``
cell this:

  1. builds the Rust ``onpair-chunk-bench`` binary (release),
  2. ensures the source parquet exists (generating TPC-H locally),
  3. compresses the sampled column into Vortex files (one OnPair dictionary per
     chunk) and verifies the string round-trip,
  4. aggregates the per-cell JSON into a markdown table + ``summary.json``.

The Rust binary parallelises chunk compression internally. Independent columns
are processed concurrently with ``--jobs``.

Usage::

    python benchmarks/onpair-bench/run.py                  # full default run
    python benchmarks/onpair-bench/run.py --sample-bytes 50_000_000  # quick
    python benchmarks/onpair-bench/run.py --bits 12 --chunk-mb 1,10  # subset
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from columns import AMAZON_URL, COLUMNS, DATA_DIR, REPO_ROOT, SRC_DIR, Column

# Columns whose bench process exited non-zero. Consulted by the final exit status so an
# all-failed run cannot look like a clean run with an empty matrix.
COLUMN_FAILURES: list[str] = []

OUT_ROOT = DATA_DIR / "onpair-bench"
BIN = "onpair-chunk-bench"
MB = 1 << 20
GIB = 1 << 30


def available_cores() -> int:
    return os.cpu_count() or 1


def clean_outputs() -> None:
    """Remove generated OnPair benchmark output files.

    Source parquet caches under `onpair-bench-src` are intentionally preserved.
    """
    if OUT_ROOT.exists():
        print(f"==> removing {OUT_ROOT}", file=sys.stderr)
        shutil.rmtree(OUT_ROOT)
    else:
        print(f"==> nothing to clean at {OUT_ROOT}", file=sys.stderr)


def build_binary(release: bool, cuda: bool) -> Path:
    profile = ["--release"] if release else []
    features = ["--features", "cuda"] if cuda else []
    print(
        f"==> building {BIN} ({'release' if release else 'dev'}"
        f"{', cuda' if cuda else ''})",
        file=sys.stderr,
    )
    subprocess.run(
        ["cargo", "build", *profile, "-p", "vortex-bench", "--bin", BIN, *features],
        cwd=REPO_ROOT,
        check=True,
    )
    target = "release" if release else "debug"
    return REPO_ROOT / "target" / target / BIN


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` (atomically via a .part file). Uses curl/wget if
    available, else urllib — so it works without extra deps. For huggingface.co
    URLs, an HF_TOKEN in the environment is sent as a Bearer header (scoped to HF
    hosts only) for faster, rate-limit-free downloads of the gated/large datasets."""
    import os
    import shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"==> downloading {url}\n        -> {dest}", file=sys.stderr)
    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    auth = hf_tok if (hf_tok and "huggingface.co" in url) else None
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--fail", "-o", str(tmp)]
        if auth:
            cmd += ["-H", f"Authorization: Bearer {auth}"]
        subprocess.run(cmd + [url], check=True)
    elif shutil.which("wget"):
        cmd = ["wget", "-O", str(tmp)]
        if auth:
            cmd += ["--header", f"Authorization: Bearer {auth}"]
        subprocess.run(cmd + [url], check=True)
    else:
        import urllib.request
        req = urllib.request.Request(url)
        if auth:
            req.add_header("Authorization", f"Bearer {auth}")
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    tmp.rename(dest)


def text_to_parquet(src: Path, dest: Path, column: str) -> None:
    """Convert a newline-delimited text file to a one-column parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"==> converting {src} -> {dest}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8", errors="replace") as f:
        values = [line.rstrip("\n\r") for line in f]
    table = pa.table({column: pa.array(values, type=pa.string())})
    tmp = dest.with_suffix(dest.suffix + ".part")
    pq.write_table(table, tmp)
    tmp.rename(dest)


# Amazon-Reviews-2023 review-text byte cap (run.py samples 1 GB; ~1.2 GB gives headroom).
AMAZON_CAP_BYTES = int(os.environ.get("AMAZON_CAP_BYTES", 1_200_000_000))


def amazon_to_parquet(col: Column) -> Path:
    """Stream an Amazon-Reviews-2023 category's raw review JSONL (McAuley Lab) and write
    its `text` field, up to AMAZON_CAP_BYTES, to a one-column parquet cache. `HF_TOKEN`
    (if set) avoids anonymous rate limits; `AMAZON_URL_OVERRIDE` points at a mirror. The
    corpus is non-redistributable, so it is materialized on-box and never committed."""
    import gzip
    import json
    import urllib.request

    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = col.cache_path()
    url = os.environ.get("AMAZON_URL_OVERRIDE") or AMAZON_URL.format(category=col.category)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> streaming Amazon '{col.category}' text (cap {AMAZON_CAP_BYTES} B)\n"
          f"        {url}\n        -> {dest}", file=sys.stderr)
    hdr = {"User-Agent": "vortex-bench/onpair"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        hdr["Authorization"] = "Bearer " + tok
    texts: list[str] = []
    total = 0
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req) as resp:
        stream = gzip.GzipFile(fileobj=resp) if url.endswith(".gz") else resp
        for line in stream:
            try:
                t = json.loads(line).get("text")
            except Exception:
                continue
            if not t:
                continue
            texts.append(t)
            total += len(t.encode("utf-8"))
            if total >= AMAZON_CAP_BYTES:
                break
    tmp = dest.with_suffix(dest.suffix + ".part")
    pq.write_table(pa.table({col.column: pa.array(texts, type=pa.string())}), tmp)
    tmp.rename(dest)
    print(f"==> wrote {len(texts)} reviews ~{total} B -> {dest}", file=sys.stderr)
    return dest


# Byte cap for the streamed corpora (parquet_stream / jsonl). run.py samples 1 GB per
# column, so ~1.5 GB of extracted text leaves headroom without pulling whole shards.
STREAM_CAP_BYTES = int(os.environ.get("STREAM_CAP_BYTES", 1_500_000_000))

HTTP_TIMEOUT_SECONDS = float(os.environ.get("ONPAIR_HTTP_TIMEOUT_SECONDS", "60"))
HTTP_MAX_ATTEMPTS = int(os.environ.get("ONPAIR_HTTP_MAX_ATTEMPTS", "4"))
HTTP_RETRY_BASE_SECONDS = float(os.environ.get("ONPAIR_HTTP_RETRY_BASE_SECONDS", "0.5"))
_RETRYABLE_HTTP_STATUS = frozenset({403, 408, 429, 500, 502, 503, 504})
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urllib.parse.urlsplit(url)
    port = parsed.port
    if port is None:
        port = {"http": 80, "https": 443}.get(parsed.scheme.lower())
    return parsed.scheme.lower(), parsed.hostname, port


class _ScopedAuthorizationRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Preserve ordinary headers across redirects, but never credentials cross-origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _origin(req.full_url) != _origin(newurl):
            redirected.remove_header("Authorization")
        return redirected


_HTTP_OPENER = urllib.request.build_opener(_ScopedAuthorizationRedirectHandler())


class _HttpProtocolError(OSError):
    """A successful HTTP response violated the range-reader contract."""


class _RetryableHttpReadError(OSError):
    """A response ended before its declared body was available."""


def _retry_delay(attempt: int) -> float:
    ceiling = HTTP_RETRY_BASE_SECONDS * (2 ** attempt)
    return random.uniform(ceiling / 2, ceiling)


def _request_with_retry(
    request: urllib.request.Request,
    consume,
    *,
    opener=None,
    sleep=time.sleep,
    max_attempts: int = HTTP_MAX_ATTEMPTS,
):
    """Open and consume one request, retrying only transient transport/status failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    open_request = opener or _HTTP_OPENER.open
    for attempt in range(max_attempts):
        try:
            with open_request(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return consume(response)
        except urllib.error.HTTPError as error:
            error.close()
            if error.code not in _RETRYABLE_HTTP_STATUS or attempt + 1 == max_attempts:
                raise
        except (
            _RetryableHttpReadError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            ConnectionError,
            TimeoutError,
        ):
            if attempt + 1 == max_attempts:
                raise
        sleep(_retry_delay(attempt))
    raise AssertionError("retry loop exhausted without returning or raising")


class _HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests.

    Exists so a multi-GB remote parquet can be read row group by row group: pyarrow
    seeks to the footer, then to the row groups it wants, and only those byte ranges
    cross the network. Every request starts from the canonical URL so an expiring CDN
    redirect can refresh, while a strong ETag pins all reads to the object inspected by
    HEAD."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        *,
        opener=None,
        sleep=time.sleep,
        max_attempts: int = HTTP_MAX_ATTEMPTS,
    ):
        self._url, self._headers, self._pos = url, dict(headers or {}), 0
        self._headers["Accept-Encoding"] = "identity"
        self._opener = opener
        self._sleep = sleep
        self._max_attempts = max_attempts

        def inspect_head(response):
            if response.status != 200:
                raise _HttpProtocolError(f"{url} HEAD returned status {response.status}, expected 200")
            content_length = response.headers.get("Content-Length")
            if content_length is None:
                raise _HttpProtocolError(f"{url} HEAD omitted Content-Length")
            try:
                size = int(content_length)
            except (TypeError, ValueError) as error:
                raise _HttpProtocolError(
                    f"{url} HEAD returned invalid Content-Length {content_length!r}"
                ) from error
            if size < 0:
                raise _HttpProtocolError(f"{url} HEAD returned negative Content-Length {size}")
            etag = response.headers.get("ETag")
            if (
                not etag
                or etag.startswith("W/")
                or not (etag.startswith('"') and etag.endswith('"'))
            ):
                raise _HttpProtocolError(f"{url} requires a strong ETag, got {etag!r}")
            return size, etag

        head = urllib.request.Request(url, method="HEAD", headers=self._headers)
        self._size, self._etag = _request_with_retry(
            head,
            inspect_head,
            opener=self._opener,
            sleep=self._sleep,
            max_attempts=self._max_attempts,
        )

    @property
    def source_metadata(self) -> dict[str, str | int]:
        return {"url": self._url, "etag": self._etag, "content_length": self._size}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            base = 0
        elif whence == io.SEEK_CUR:
            base = self._pos
        elif whence == io.SEEK_END:
            base = self._size
        else:
            raise ValueError(f"invalid whence: {whence}")
        position = base + offset
        if position < 0:
            raise ValueError(f"negative seek position: {position}")
        self._pos = position
        return self._pos

    def readinto(self, b) -> int:
        view = memoryview(b)
        if view.readonly:
            raise TypeError("readinto() argument must be writable")
        try:
            view = view.cast("B")
        except TypeError as error:
            raise TypeError("readinto() argument must be a contiguous buffer") from error
        n = min(view.nbytes, max(0, self._size - self._pos))
        if n <= 0:
            return 0
        start = self._pos
        end = start + n - 1
        h = dict(self._headers)
        h["Range"] = f"bytes={start}-{end}"
        h["If-Match"] = self._etag

        def read_range(response):
            if response.status != 206:
                raise _HttpProtocolError(
                    f"{self._url} ignored a Range request (status {response.status}); "
                    "refusing to download the whole object")
            content_range = response.headers.get("Content-Range")
            match = _CONTENT_RANGE_RE.fullmatch(content_range or "")
            actual_range = tuple(int(value) for value in match.groups()) if match else None
            expected_range = (start, end, self._size)
            if actual_range != expected_range:
                raise _HttpProtocolError(
                    f"{self._url} returned Content-Range {content_range!r}, expected "
                    f"'bytes {start}-{end}/{self._size}'"
                )
            response_etag = response.headers.get("ETag")
            if response_etag != self._etag:
                raise _HttpProtocolError(
                    f"{self._url} changed ETag from {self._etag!r} to {response_etag!r}"
                )
            content_encoding = response.headers.get("Content-Encoding")
            if content_encoding not in (None, "identity"):
                raise _HttpProtocolError(
                    f"{self._url} encoded a byte-range response as {content_encoding!r}"
                )
            response_length = response.headers.get("Content-Length")
            if response_length is not None:
                try:
                    declared_length = int(response_length)
                except (TypeError, ValueError) as error:
                    raise _HttpProtocolError(
                        f"{self._url} returned invalid Content-Length {response_length!r}"
                    ) from error
                if declared_length != n:
                    raise _HttpProtocolError(
                        f"{self._url} declared {declared_length} bytes for a {n}-byte range"
                    )
            data = response.read(n + 1)
            if len(data) != n:
                raise _RetryableHttpReadError(
                    f"{self._url} returned {len(data)} bytes for range {start}-{end}, expected {n}"
                )
            return data

        request = urllib.request.Request(self._url, headers=h)
        data = _request_with_retry(
            request,
            read_range,
            opener=self._opener,
            sleep=self._sleep,
            max_attempts=self._max_attempts,
        )
        view[:n] = data
        self._pos += n
        return n


def _stream_cache_identity(col: Column) -> dict:
    urls = list(col.urls) or ([col.url] if col.url else [])
    return {
        "loader_version": 2,
        "dataset_id": col.dataset_id,
        "kind": col.kind,
        "columns": list(col.siblings) or [col.column],
        "json_paths": [list(item) for item in col.json_paths],
        "cap_bytes": col.cap_bytes or STREAM_CAP_BYTES,
        "source_revision": col.source_revision,
        "urls": urls,
    }


def _cache_manifest_path(dest: Path) -> Path:
    return Path(f"{dest}.manifest.json")


def _schema_description(schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _stream_cache_status(dest: Path, identity: dict) -> tuple[bool, str]:
    import pyarrow.parquet as pq

    manifest_path = _cache_manifest_path(dest)
    if not dest.is_file():
        return False, "parquet is absent"
    if not manifest_path.is_file():
        return False, "manifest is absent"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"manifest cannot be read: {error}"
    if manifest.get("identity") != identity:
        return False, "manifest identity differs"
    try:
        parquet = pq.ParquetFile(dest)
        metadata = parquet.metadata
        schema = parquet.schema_arrow
        stat_size = dest.stat().st_size
    except (OSError, ValueError) as error:
        return False, f"parquet footer cannot be read: {error}"
    if metadata.num_rows != manifest.get("row_count"):
        return False, "parquet row count differs from manifest"
    if stat_size != manifest.get("parquet_bytes"):
        return False, "parquet file size differs from manifest"
    if _schema_description(schema) != manifest.get("schema"):
        return False, "parquet schema differs from manifest"
    utf8_bytes = manifest.get("utf8_bytes")
    expected_columns = identity["columns"]
    if not isinstance(utf8_bytes, dict) or set(utf8_bytes) != set(expected_columns):
        return False, "manifest UTF-8 byte totals do not cover the selected columns"
    if any(not isinstance(utf8_bytes[name], int) or utf8_bytes[name] < 0 for name in expected_columns):
        return False, "manifest has an invalid UTF-8 byte total"
    return True, "identity and parquet metadata match"


@contextlib.contextmanager
def _stream_cache_lock(dest: Path):
    import fcntl

    lock_path = Path(f"{dest}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _publish_stream_cache(table, dest: Path, identity: dict, details: dict) -> dict:
    import pyarrow.parquet as pq

    if table.num_rows == 0:
        raise OSError(f"refusing to publish an empty streamed cache at {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest.parent, prefix=f".{dest.name}.", suffix=".part", delete=False
    ) as tmp_file:
        tmp = Path(tmp_file.name)
    manifest_path = _cache_manifest_path(dest)
    with tempfile.NamedTemporaryFile(
        dir=dest.parent,
        prefix=f".{manifest_path.name}.",
        suffix=".part",
        delete=False,
    ) as manifest_file:
        manifest_tmp = Path(manifest_file.name)
    try:
        pq.write_table(table, tmp)
        parquet = pq.ParquetFile(tmp)
        if parquet.metadata.num_rows != table.num_rows:
            raise OSError(
                f"temporary cache row count changed: {parquet.metadata.num_rows} != {table.num_rows}"
            )
        manifest = {
            "identity": identity,
            "row_count": table.num_rows,
            "parquet_bytes": tmp.stat().st_size,
            "schema": _schema_description(parquet.schema_arrow),
            **details,
        }
        with open(manifest_tmp, "w", encoding="utf-8") as output:
            output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, dest)
        os.replace(manifest_tmp, manifest_path)
        return manifest
    finally:
        tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _utf8_payload_bytes(array) -> int:
    import pyarrow.compute as pc

    total = pc.sum(pc.binary_length(array)).as_py()
    return int(total or 0)


def parquet_stream_to_parquet(col: Column, identity: dict | None = None) -> Path:
    """Pull row groups from a remote parquet over HTTP ranges until `cap_bytes` of the
    dataset's columns have accumulated, then write them to the shared cache. Used for
    shards far larger than the ~1 GB the benchmark samples."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = col.cache_path()
    identity = identity or _stream_cache_identity(col)
    cap = col.cap_bytes or STREAM_CAP_BYTES
    cols = list(col.siblings) or [col.column]
    dest.parent.mkdir(parents=True, exist_ok=True)
    hdr = {"User-Agent": "vortex-bench/onpair"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok and "huggingface.co" in (col.url or ""):
        hdr["Authorization"] = "Bearer " + tok
    urls = list(col.urls) or [col.url]
    print(f"==> streaming remote parquet columns {cols} from {len(urls)} shard(s) "
          f"(cap {cap} B)\n        {urls[0]}\n        -> {dest}", file=sys.stderr)

    # The cap tracks the widest column. This preserves the campaign's existing staging
    # contract; the pair-level sweep selects only CodeSearchNet whole_func_string and
    # FineWeb2 text, and both clear the benchmark's 1 GB payload sample.
    batches, per_col, total, sources = [], {c: 0 for c in cols}, 0, []
    stop_reason = "all_sources_exhausted"
    for si, url in enumerate(urls, 1):
        remote = _HttpRangeFile(url, hdr)
        sources.append(remote.source_metadata)
        pf = pq.ParquetFile(remote)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=cols)
            batches.append(t)
            for c in cols:
                per_col[c] += _utf8_payload_bytes(t.column(c))
            total = max(per_col.values())
            print(f"    shard {si}/{len(urls)} row group {rg + 1}/{pf.num_row_groups}: "
                  f"{total} B", file=sys.stderr)
            if total >= cap:
                stop_reason = "cap_reached"
                break
        if total >= cap:
            break
    if not batches:
        raise OSError(f"no row groups were read from streamed source(s) for {col.dataset_id}")
    table = pa.concat_tables(batches)
    _publish_stream_cache(
        table,
        dest,
        identity,
        {
            "sources": sources,
            "utf8_bytes": per_col,
            "stop_reason": stop_reason,
            "shards_read": len(sources),
        },
    )
    print(f"==> wrote {table.num_rows} rows ~{total} B -> {dest}", file=sys.stderr)
    return dest


def jsonl_to_parquet(col: Column, identity: dict | None = None) -> Path:
    """Stream gzipped JSON-lines URLs in order, extract a dotted path per output column,
    and write them all to the shared cache. Stops at the byte cap or when the URLs run
    out; a URL that 404s ends the stream rather than failing the run, so a corpus that
    has lost a shard still yields a (smaller, reported) sample."""
    import gzip
    import json

    import pyarrow as pa

    dest = col.cache_path()
    identity = identity or _stream_cache_identity(col)
    cap = col.cap_bytes or STREAM_CAP_BYTES
    paths = dict(col.json_paths)
    dest.parent.mkdir(parents=True, exist_ok=True)
    hdr = {"User-Agent": "vortex-bench/onpair"}
    print(f"==> streaming {len(col.urls)} JSON-lines shards for columns {list(paths)} "
          f"(cap {cap} B)\n        {col.urls[0]} ...\n        -> {dest}", file=sys.stderr)

    out: dict[str, list[str]] = {c: [] for c in paths}
    per_col = {c: 0 for c in paths}
    total, shards, sources = 0, 0, []
    stop_reason = "all_sources_exhausted"
    for url in col.urls:
        req = urllib.request.Request(url, headers=hdr)

        def read_shard(response):
            shard_out: dict[str, list[str]] = {c: [] for c in paths}
            shard_bytes = {c: 0 for c in paths}
            with gzip.GzipFile(fileobj=response) as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{url}:{line_number}: malformed JSON") from error
                    for name, path in paths.items():
                        value = rec
                        for part in path.split("."):
                            value = value.get(part) if isinstance(value, dict) else None
                        text = value if isinstance(value, str) else ""
                        shard_out[name].append(text)
                        shard_bytes[name] += len(text.encode("utf-8"))
            return shard_out, shard_bytes, {
                "url": url,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_length": response.headers.get("Content-Length"),
            }

        try:
            shard_out, shard_bytes, source = _request_with_retry(req, read_shard)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    {url}: HTTP 404, source sequence exhausted", file=sys.stderr)
                stop_reason = "terminal_404"
                break
            raise
        shards += 1
        sources.append(source)
        for name in paths:
            out[name].extend(shard_out[name])
            per_col[name] += shard_bytes[name]
        total = max(per_col.values())
        print(f"    shard {shards}/{len(col.urls)}: {total} B", file=sys.stderr)
        if total >= cap:
            stop_reason = "cap_reached"
            break
    if shards == 0:
        raise OSError(f"no JSONL shards were read for {col.dataset_id}")
    table = pa.table({c: pa.array(v, type=pa.string()) for c, v in out.items()})
    _publish_stream_cache(
        table,
        dest,
        identity,
        {
            "sources": sources,
            "utf8_bytes": per_col,
            "stop_reason": stop_reason,
            "shards_read": shards,
        },
    )
    print(f"==> wrote {table.num_rows} rows from {shards} shards ~{total} B -> {dest}",
          file=sys.stderr)
    return dest


def ensure_parquet(binary: Path, col: Column) -> Path:
    path = col.parquet_path()
    if col.kind in {"parquet_stream", "jsonl"} and path == col.cache_path():
        identity = _stream_cache_identity(col)
        with _stream_cache_lock(path):
            valid, reason = _stream_cache_status(path, identity)
            if valid:
                print(f"==> streamed cache hit: {path} ({reason})", file=sys.stderr)
                return path
            if path.exists() or _cache_manifest_path(path).exists():
                print(f"==> rebuilding streamed cache {path}: {reason}", file=sys.stderr)
            if col.kind == "parquet_stream":
                return parquet_stream_to_parquet(col, identity)
            return jsonl_to_parquet(col, identity)
    if path.exists():
        return path
    if col.kind == "tpch":
        # Generates *all* TPC-H tables (one file each) into the sf dir; the Rust
        # side is idempotent so repeated calls for sibling columns are no-ops.
        out_dir = col.tpch_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"==> generating TPC-H sf={col.scale_factor} tables", file=sys.stderr)
        subprocess.run(
            [str(binary), "gen-tpch", "--sf", str(col.scale_factor), "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            check=True,
        )
        return path
    if col.kind == "tpcds":
        # DuckDB dsdgen → one parquet per table under <tpcds_dir>/parquet/.
        out_dir = col.tpcds_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"==> generating TPC-DS sf={col.scale_factor} tables (duckdb dsdgen)", file=sys.stderr)
        subprocess.run(
            [str(binary), "gen-tpcds", "--sf", str(col.scale_factor), "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            check=True,
        )
        return path
    if col.kind == "parquet" and col.url:
        download(col.url, col.cache_path())
        return col.cache_path()
    if col.kind == "text" and col.url:
        raw_path = SRC_DIR / col.dataset_id / "raw" / f"{col.column}.txt"
        if not raw_path.exists():
            download(col.url, raw_path)
        text_to_parquet(raw_path, col.cache_path(), col.column)
        return col.cache_path()
    if col.kind == "amazon":
        return amazon_to_parquet(col)
    if col.kind == "synthetic":
        # Deterministic in-pipeline generation (seed 123) via the Rust binary;
        # no external source. Idempotent on the Rust side.
        cache = col.cache_path()
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"==> generating synthetic URL corpus ({col.rows} rows)", file=sys.stderr)
        subprocess.run(
            [str(binary), "gen-synth-urls", "--rows", str(col.rows), "--out", str(cache)],
            cwd=REPO_ROOT,
            check=True,
        )
        return cache
    raise FileNotFoundError(
        f"parquet for {col.dataset_id}/{col.column} not found at {path} "
        f"and no download url configured"
    )


_STRING_ARROW_TYPES = ("string", "utf8", "large_string", "large_utf8")


def column_is_string(parquet: Path, column: str) -> bool:
    """True iff `column` exists in `parquet` and is a (any-width) string type."""
    import pyarrow.parquet as pq

    try:
        field = pq.read_schema(parquet).field(column)
    except KeyError:
        return False
    return any(t in str(field.type).lower() for t in _STRING_ARROW_TYPES)


def run_column(binary: Path, col: Column, args) -> list[dict]:
    parquet = ensure_parquet(binary, col)
    chunk_bytes = ",".join(str(int(mb * MB)) for mb in args.chunk_mb)
    bits = ",".join(str(b) for b in args.bits)
    thresholds = ",".join(str(t) for t in args.threshold)
    print(f"==> running {col.dataset_id}/{col.column}", file=sys.stderr)
    proc = subprocess.run(
        [
            str(binary), "run",
            "--parquet", str(parquet),
            "--column", col.column,
            "--dataset-id", col.dataset_id,
            "--bits", bits,
            "--chunk-bytes", chunk_bytes,
            "--threshold", thresholds,
            "--codec", args.codec,
            "--sample-bytes", str(args.sample_bytes),
            "--file-target-bytes", str(int(args.file_target_mb * MB)),
            "--out-dir", str(OUT_ROOT),
            *(
                [
                    "--gpu-decode",
                    "--gpu-iters", str(args.gpu_iters),
                    *(["--gpu-validate"] if args.gpu_validate else []),
                    *(["--gpu-kernels", args.gpu_kernels] if args.gpu_kernels else []),
                ]
                if args.gpu_decode
                else []
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        print(f"!! {col.dataset_id}/{col.column} FAILED:\n{tail}", file=sys.stderr)
        # Record the failure. Returning a bare [] made a failed column indistinguishable
        # from a column that produced no cells, so a run in which EVERY column failed still
        # wrote summaries and exited 0 -- reporting success with no data.
        COLUMN_FAILURES.append(f"{col.dataset_id}/{col.column}: exit {proc.returncode}")
        return []
    rows = json.loads(proc.stdout)
    if col.kind in {"parquet_stream", "jsonl"} and col.parquet_path() == col.cache_path():
        try:
            source_cache = json.loads(_cache_manifest_path(col.cache_path()).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise OSError(f"validated source-cache manifest disappeared for {col.dataset_id}") from error
        for row in rows:
            row["source_cache"] = source_cache
    return rows


def fmt_bytes(n: int) -> str:
    for unit, size in (("GiB", GIB), ("MiB", MB), ("KiB", 1 << 10)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n} B"


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "dataset", "column", "bits", "thr", "chunk", "rows", "uniq", "uniq%",
        "chunks", "sample", "in-mem", "on-disk", "str→codec×", "str→files×",
        "enc GiB/s", "dec GiB/s", "gpu auto", "gpu best", "ok", "onpair",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        uniq_pct = 100.0 * r["unique_count"] / r["rows"] if r["rows"] else 0.0
        gpu = r.get("gpu")
        gpu_auto = ""
        gpu_best = ""
        if gpu:
            if gpu.get("auto_kernel") is not None:
                gpu_auto = f"{gpu['auto_kernel']} {gpu['auto_decode_gib_s']:.1f}"
            gpu_best = f"{gpu['best_kernel']} {gpu['best_decode_gib_s']:.1f}"
            if gpu.get("validated"):
                status = "ok" if gpu.get("verified") else "bad"
                gpu_best = f"{gpu_best} ({status})"
        lines.append("| " + " | ".join([
            r["dataset_id"], r["column"], str(r["bits"]), f"{r['threshold']:.2f}",
            fmt_bytes(r["chunk_bytes"]), f"{r['rows']:,}", f"{r['unique_count']:,}",
            f"{uniq_pct:.1f}%", str(r["n_chunks"]),
            fmt_bytes(r["sample_bytes"]), fmt_bytes(r["in_memory_bytes"]),
            fmt_bytes(r["on_disk_bytes"]), f"{r['mem_ratio']:.2f}",
            f"{r['disk_ratio']:.2f}", f"{r['encode_gib_s']:.2f}",
            f"{r['decode_gib_s']:.2f}", gpu_auto, gpu_best, "✓" if r["verified"] else "✗",
            "✓" if r["onpair_only"] else "✗",
        ]) + " |")
    return "\n".join(lines)


def pivot_table(rows: list[dict]) -> str:
    """Per-column compression (str→codec×) across every param combo:
    dict width (bits) × block size (chunk)."""
    # Stable param-combo column order.
    combos = sorted({(r["bits"], r["chunk_bytes"]) for r in rows})

    def combo_label(bits, chunk):
        return f"b{bits}/{fmt_bytes(chunk).split()[0]}"

    val = {}  # (dataset,column) -> {combo -> ratio}
    uniq = {}
    for r in rows:
        k = (r["dataset_id"], r["column"])
        val.setdefault(k, {})[(r["bits"], r["chunk_bytes"])] = r["mem_ratio"]
        uniq[k] = 100.0 * r["unique_count"] / r["rows"] if r["rows"] else 0.0

    headers = ["dataset/column", "uniq%"] + [combo_label(*c) for c in combos]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for k in sorted(val, key=lambda k: (k[0], k[1])):
        cells = [f"{val[k].get(c, float('nan')):.2f}" for c in combos]
        lines.append("| " + " | ".join([f"{k[0]}/{k[1]}", f"{uniq[k]:.1f}%"] + cells) + " |")
    return "\n".join(lines)


def consolidated_summary(rows: list[dict], args, pivot: str, full_table: str) -> str:
    """Everything in one place: params, datasets, verification, the per-column
    pivot, key findings, and the full per-cell table."""
    from collections import Counter

    datasets = Counter(r["dataset_id"] for r in rows)
    cols = len({(r["dataset_id"], r["column"]) for r in rows})
    fails = [
        r for r in rows
        if (
            not r["verified"]
            or (r.get("codec", "onpair") == "onpair" and not r["onpair_only"])
            or (r.get("gpu", {}).get("validated") and not r["gpu"].get("verified"))
        )
    ]

    # Best param per column (max str→codec×).
    best = {}
    for r in rows:
        k = (r["dataset_id"], r["column"])
        if k not in best or r["mem_ratio"] > best[k]["mem_ratio"]:
            best[k] = r

    lines = [
        "# OnPair chunked-array compression — benchmark summary",
        "",
        "Each string column is OnPair-compressed per chunk (one dictionary each), "
        "every OnPair child is BtrBlocks-compressed, and the chunks are written to "
        "real `.vortex` files that preserve the OnPair encoding. The offset children "
        "use the smaller of BtrBlocks-only and delta+BtrBlocks. All cells are round-trip verified.",
        "",
        "## Run parameters",
        f"- dict widths (bits): `{args.bits}`",
        f"- block sizes (uncompressed): `{[f'{m:g}MB' for m in args.chunk_mb]}`",
        f"- training threshold: `{args.threshold}`",
        f"- raw sample cap: `{args.sample_bytes:,}` bytes  |  file target: `{args.file_target_mb:g} MB`",
        f"- GPU kernel-only decode: `{'on' if args.gpu_decode else 'off'}`"
        + (f" ({args.gpu_iters} timed iterations)" if args.gpu_decode else ""),
        f"- GPU kernel selection: `{args.gpu_kernels or 'all registered kernels'}`",
        f"- GPU byte validation: `{'on' if args.gpu_validate else 'off'}`",
        "",
        "## Coverage",
        f"- **{len(rows)} cells**, **{cols} columns**, datasets: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(datasets.items())),
        f"- round-trip + OnPair-only failures: **{len(fails)}**",
        "",
        "## str→codec× per column × (dict-width / block)",
        "",
        pivot,
        "",
        "## Best configuration per column",
        "",
        "| dataset/column | uniq% | best str→codec× | bits | block |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in sorted(best, key=lambda k: best[k]["mem_ratio"]):
        r = best[k]
        up = 100.0 * r["unique_count"] / r["rows"] if r["rows"] else 0.0
        lines.append(f"| {k[0]}/{k[1]} | {up:.1f}% | {r['mem_ratio']:.2f}× | "
                     f"{r['bits']} | {fmt_bytes(r['chunk_bytes'])} |")
    lines += ["", "## All cells", "", full_table, ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bits", type=lambda s: [int(x) for x in s.split(",")],
                   default=[12, 16])
    p.add_argument("--chunk-mb", type=lambda s: [float(x) for x in s.split(",")],
                   default=[1, 10, 100, 1000], help="per-chunk uncompressed MB budgets")
    p.add_argument("--threshold", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.2],
                   help="OnPair dynamic frequency threshold (use 0.2; 0.5 was "
                        "evaluated and dropped)")
    p.add_argument("--sample-bytes", type=int, default=1_000_000_000)
    p.add_argument("--file-target-mb", type=float, default=200.0)
    p.add_argument("--gpu-decode", action="store_true",
                   help="also time CUDA kernel-only OnPair decompression for all applicable kernels")
    p.add_argument("--gpu-iters", type=int, default=10,
                   help="timed CUDA iterations per kernel when --gpu-decode is set")
    p.add_argument("--gpu-validate", action="store_true",
                   help="copy GPU output back and compare every applicable kernel against CPU bytes")
    p.add_argument("--gpu-kernels", default=None,
                   help="exact comma-separated kernel allowlist; use tpt-matched for the ten-way control")
    p.add_argument("--jobs", type=int, default=0,
                   help="columns to run concurrently (default: all available CPU cores)")
    p.add_argument("--codec", choices=["onpair", "fsst12"], default="onpair",
                   help="stored codec; fsst12 ignores --bits/--threshold (one configuration), "
                        "is GPU-only, and implies --gpu-decode --gpu-validate")
    p.add_argument("--dev", action="store_true", help="dev build instead of release")
    p.add_argument("--datasets", type=lambda s: {x.strip() for x in s.split(",")},
                   default=None,
                   help="only these dataset ids (comma-separated), e.g. tpch-sf10,fineweb")
    p.add_argument("--columns", type=lambda s: {x.strip() for x in s.split(",")},
                   default=None,
                   help="only these column names (comma-separated), e.g. l_comment,text")
    p.add_argument("--allow-missing-inputs", action="store_true",
                   help="skip requested datasets/columns whose parquet or string column is missing")
    p.add_argument("--list", action="store_true",
                   help="list available dataset/column pairs and exit")
    p.add_argument("--clean", action="store_true",
                   help="delete generated OnPair benchmark .vortex files and summaries, then exit")
    args = p.parse_args()

    # FSST-12 exists on the GPU path only: there is no .vortex round-trip for it, so the
    # kernel byte-exactness check IS its correctness result. Imply both flags rather than
    # letting a missing one surface as an error deep in the Rust cell, and because the CUDA
    # feature selection below keys on gpu_decode.
    if args.codec == "fsst12":
        if not args.gpu_decode or not args.gpu_validate:
            print("--codec fsst12 implies --gpu-decode --gpu-validate; enabling both",
                  file=sys.stderr)
        args.gpu_decode = True
        args.gpu_validate = True

    if args.clean:
        clean_outputs()
        return 0

    if args.gpu_validate and not args.gpu_decode:
        print("--gpu-validate requires --gpu-decode", file=sys.stderr)
        return 1

    if args.list:
        for c in COLUMNS:
            print(f"{c.dataset_id}\t{c.column}")
        return 0

    # Restrict to the requested datasets / columns (both filters are AND-ed).
    columns = [c for c in COLUMNS
               if (args.datasets is None or c.dataset_id in args.datasets)
               and (args.columns is None or c.column in args.columns)]
    missing_filters = []
    if args.datasets is not None:
        missing_filters.extend(
            f"dataset {name!r}" for name in sorted(args.datasets - {c.dataset_id for c in columns})
        )
    if args.columns is not None:
        missing_filters.extend(
            f"column {name!r}" for name in sorted(args.columns - {c.column for c in columns})
        )
    if missing_filters:
        print("unmatched selection(s): " + ", ".join(missing_filters), file=sys.stderr)
        return 1
    if not columns:
        print("no columns match the given --datasets/--columns filters", file=sys.stderr)
        return 1

    binary = build_binary(release=not args.dev, cuda=args.gpu_decode)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    # Ensure each source parquet exists up front (sequentially) so concurrent
    # columns never race on generation, then keep only columns that are present
    # and string-typed.
    selected: list[Column] = []
    unavailable: list[str] = []
    for col in columns:
        try:
            parquet = ensure_parquet(binary, col)
        except FileNotFoundError as e:
            # External datasets (ClickBench/FineWeb/book-reviews) aren't
            # auto-downloaded; skip any whose source parquet is absent so the
            # run still completes on whatever data is present (TPC-H always
            # generates locally).
            print(f"-- skip {col.dataset_id}/{col.column}: {e}", file=sys.stderr)
            unavailable.append(f"{col.dataset_id}/{col.column}: {e}")
            continue
        if column_is_string(parquet, col.column):
            selected.append(col)
        else:
            print(f"-- skip {col.dataset_id}/{col.column} (missing or non-string)",
                  file=sys.stderr)
            unavailable.append(f"{col.dataset_id}/{col.column}: missing or non-string")
    if unavailable and not args.allow_missing_inputs:
        print("requested benchmark inputs are unavailable; refusing a partial campaign:",
              file=sys.stderr)
        for item in unavailable:
            print(f"   - {item}", file=sys.stderr)
        print("pass --allow-missing-inputs to opt into a partial campaign", file=sys.stderr)
        return 1
    print(f"==> {len(selected)}/{len(columns)} columns selected", file=sys.stderr)

    jobs = args.jobs if args.jobs > 0 else available_cores()
    print(f"==> running with {jobs} column worker(s)", file=sys.stderr)

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(run_column, binary, c, args): c for c in selected}
            for fut in as_completed(futs):
                results.extend(fut.result())
    else:
        for col in selected:
            results.extend(run_column(binary, col, args))

    results.sort(key=lambda r: (r["dataset_id"], r["column"], r["bits"],
                                r["threshold"], r["chunk_bytes"]))

    summary_json = OUT_ROOT / "summary.json"
    summary_md = OUT_ROOT / "summary.md"
    pivot_md = OUT_ROOT / "summary_pivot.md"
    consolidated = OUT_ROOT / "SUMMARY.md"
    summary_json.write_text(json.dumps(results, indent=2))
    table = markdown_table(results)
    pivot = pivot_table(results)
    summary_md.write_text(table + "\n")
    pivot_md.write_text("# str→codec× per column × (dict-width / block)\n\n"
                        + pivot + "\n")
    consolidated.write_text(consolidated_summary(results, args, pivot, table))

    print("\n" + pivot)
    print(f"\nWrote {consolidated} (everything in one place)\n"
          f"Wrote {summary_json}\nWrote {summary_md}\nWrote {pivot_md}", file=sys.stderr)

    # `onpair_only` asserts the stored column is purely OnPair-encoded. It is meaningless
    # for a non-Vortex codec, and FSST-12 sets it false BY CONSTRUCTION -- applying it to
    # every codec classified every successful FSST-12 cell as a failure.
    failures = [
        r for r in results
        if (
            not r["verified"]
            or (r.get("codec", "onpair") == "onpair" and not r["onpair_only"])
            or (r.get("gpu", {}).get("validated") and not r["gpu"].get("verified"))
        )
    ]
    if COLUMN_FAILURES:
        print("\n!! columns whose bench process failed:", file=sys.stderr)
        for f in COLUMN_FAILURES:
            print(f"   - {f}", file=sys.stderr)
    if not results:
        print("\n!! no cells were produced; treating as failure", file=sys.stderr)
    if failures or COLUMN_FAILURES or not results:
        if failures:
            print(f"\n{len(failures)} cell(s) FAILED round-trip / onpair-only / GPU validation check",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
