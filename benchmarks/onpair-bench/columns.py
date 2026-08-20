# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Registry of (dataset, column) sources for the OnPair compression benchmark.

Adding a new column is a one-line append to ``COLUMNS``.

Source kinds:

* ``tpch``    — generated locally via the Rust ``gen-tpch`` subcommand (requested
                table only, one parquet file).
* ``parquet`` — an external parquet. Give a download ``url`` (fetched into a
                repo-relative cache on first use) and/or ``local`` paths to
                reuse if already present. Nothing is hard-required: a column
                with no resolvable source is skipped by ``run.py``.
* ``text``    — a newline-delimited raw text file converted to a single-column
                parquet cache by ``run.py``.
* ``amazon``  — an Amazon-Reviews-2023 review-text category (McAuley Lab); run.py
                streams the raw ``.jsonl`` and writes its ``text`` field to a
                one-column parquet cache (capped ~1.2 GB).

All generated/downloaded data lives **under the repo** (``vortex-bench/data``),
resolved relative to this file — so the benchmark works from any checkout
location with no absolute paths required.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# Repo-relative roots (resolved from this file's location — no absolute paths).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "vortex-bench" / "data"
SRC_DIR = DATA_DIR / "onpair-bench-src"

# Download URLs for the external corpora. These were the live canonical locations as of
# 2026-06 (the paper's measurement window); each dataset's landing page is noted so a
# reader can find the current location if a direct link has since moved. This is a
# reproducibility harness, not an archive: we record best-effort provenance for the data,
# not a frozen copy of it.
CLICKBENCH_URL = "https://datasets.clickhouse.com/hits_compatible/hits.parquet"
#   landing page: https://github.com/ClickHouse/ClickBench  (the "hits" dataset)
FINEWEB_URL = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb/"
               "resolve/v1.4.0/sample/10BT/000_00000.parquet")
#   landing page: https://huggingface.co/datasets/HuggingFaceFW/fineweb
# Wikipedia (English, 2023-11-01 snapshot) — long encyclopaedic free text.
# One ~420 MB parquet shard; columns id/url/title/text.
WIKIPEDIA_URL = ("https://huggingface.co/datasets/wikimedia/wikipedia/"
                 "resolve/main/20231101.en/train-00000-of-00041.parquet")
# Shard 0 alone yields only 0.70 GB of `text`. Measured, not estimated: the completed locked
# sweeps recorded sample_bytes=703,062,420 against the 1e9 they asked for, so wikipedia was
# quietly a 0.70 GB cell while clickbench/URL and l_comment both filled 1e9. Three shards clear
# 1.15 GB with margin (0.70 + ~0.58 + ~0.55 GB at shard sizes 420/351/329 MB).
#
# This CHANGES wikipedia's sampled volume, so wikipedia numbers from the 2026-08-19 screening
# runs (0.70 GB) are NOT comparable with anything measured through WIKIPEDIA_URLS.
WIKIPEDIA_URLS = [("https://huggingface.co/datasets/wikimedia/wikipedia/"
                   f"resolve/main/20231101.en/train-{i:05d}-of-00041.parquet")
                  for i in (0, 1, 2)]
#   landing page: https://huggingface.co/datasets/wikimedia/wikipedia
# Public BI benchmark. Data and schemas both come from the same fork that
# `vortex-bench/scripts/fetch_public_bi_schemas_and_queries.sh` already pins, so the GPU bench
# and the Rust benches read one source of truth. The data URLs are the ones listed in each
# workbook's `data-urls.txt`; the upstream CWI mirror is gone (404) and the
# `public-bi-benchmark` S3 bucket refuses anonymous reads (403), so this R2 bucket is the only
# fetchable copy. Verified 2026-08-19: HTTP 200, `BZh9` magic, sizes matching local copies.
# OnPair's training-shuffle seed, pinned 2026-08-19. See `Column.training_seed` for why, and for
# what pinning invalidates. Kept distinct from ONPAIR_SHUFFLE_SEED (kernel variant ORDER, set by
# the launchers) -- these are different knobs and conflating them would hide one behind the other.
ONPAIR_TRAINING_SEED = 20260819

# Loghub (Zhu et al., "Loghub: A Large Collection of System Log Datasets", arXiv:2008.06448) --
# real system logs, the one regime at multi-GB scale that reaches the long-token end of the axis.
# Profiled 2026-08-19: application logs cluster at 55-63% short tokens because timestamps, thread
# names, IPs and IDs carry per-line entropy, while Windows CBS reaches 38% / 21% because its lines
# are dominated by long, exactly-repeated servicing-component paths. Windows is the corpus's
# OnPair-12 low anchor and nothing else measured at >=1 GB comes near it.
#
# `.tar.gz` members stream, so a 1.15 GB sample of the 26 GB Windows.log costs ~66 MB of
# transfer (measured) -- cheaper than fetching HDFS_v1.zip whole.
LOGHUB_BASE = "https://zenodo.org/records/8196385/files"
#   landing page: https://github.com/logpai/loghub

# CodeParrot clean (Tunstall et al. / HF `codeparrot/codeparrot-clean`) -- deduplicated GitHub
# Python source. A distinct content domain from prose, URLs, logs and relational text, and the
# most training-stable column in the corpus (frac_le8 sigma 0.08 over ten draws). 54 jsonl.gz
# shards of ~0.25 GB; the `codeparrot-clean-valid` split people usually cite is shard 54 of this
# same series, and is too small on its own for a 1 GB sample.
# NOTE on cost: `jsonl_to_parquet` checks its byte cap BETWEEN shards, not within one, and each
# shard yields ~1.02 GB of `content`. So a 1.15 GB cap reads two shards (~2.04 GB extracted,
# ~0.5 GB transferred) rather than stopping mid-shard. Three URLs are listed so a shard that
# 404s does not sink the column. Dropping the cap to 1.0e9 would read exactly one shard, but
# leaves only a 1.9% margin over the 1 GB the bench samples -- not worth 0.25 GB of transfer.
CODEPARROT_URLS = [("https://huggingface.co/datasets/codeparrot/codeparrot-clean/"
                    f"resolve/main/file-{i:012d}.json.gz") for i in (1, 2, 3)]
#   landing page: https://huggingface.co/datasets/codeparrot/codeparrot-clean

PBI_DATA_BASE = "https://pub-334c2a12c9bf46f3b8464a8718df8cae.r2.dev"
PBI_SCHEMA_BASE = ("https://raw.githubusercontent.com/vortex-data/"
                   "public_bi_benchmark/master/benchmark")
#   landing page: https://github.com/vortex-data/public_bi_benchmark
DBTEXT_URL_BASE = "https://raw.githubusercontent.com/cwida/fsst/master/paper/dbtext"
#   landing page: https://github.com/cwida/fsst  (the FSST paper's dbtext corpus)
# Amazon-Reviews-2023 (McAuley Lab, UCSD): per-category raw review JSONL; run.py streams
# the `text` field. Non-redistributable corpus — materialized on-box, never committed.
#   landing page: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
AMAZON_URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
              "resolve/main/raw/review_categories/{category}.jsonl")
# CodeSearchNet (Husain et al. 2019) — source code beside its docstrings. Function text is
# highly repetitive at the token level (indentation runs, keywords, identifier prefixes),
# which is the regime OnPair targets; the docstring column is ordinary prose drawn from the
# same rows. The all-languages split (Go, Java, JavaScript, PHP, Python, Ruby) is used
# rather than Python alone: the per-language splits top out around 375 MB of function text,
# too little to fill the 1 GB chunk the throughput cells decode. Both train shards are
# listed because the first alone yields 833,059,459 UTF-8 payload bytes of
# `whole_func_string`, just short. Pin the Hub revision so every box resolves the same files.
CODESEARCHNET_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
_CSN_BASE = ("https://huggingface.co/datasets/code-search-net/code_search_net/"
             f"resolve/{CODESEARCHNET_REVISION}/all/")
CODESEARCHNET_URLS = [_CSN_BASE + f"train-0000{i}-of-00002.parquet" for i in (0, 1)]
#   landing page: https://huggingface.co/datasets/code-search-net/code_search_net
# FineWeb2 Mandarin (cmn_Hani) — the multibyte-UTF-8 case. Every shard is ~4.8 GB, far more
# than the ~1 GB the bench samples, so this uses `parquet_stream`: row groups arrive over
# HTTP range requests until the byte cap is hit, and the shard is never fetched whole.
FINEWEB2_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
FINEWEB2_ZH_URL = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/"
                   f"resolve/{FINEWEB2_REVISION}/data/cmn_Hani/train/000_00000.parquet")
#   landing page: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
# GH Archive — public GitHub events as hourly gzipped JSON-lines, starting 2024-10-01
# (the day the 2026-05 dataset survey used). One day yields only ~110 MB per extracted
# column, well short of the 1 GB the throughput cells decode, so ONPAIR_GHARCHIVE_DAYS
# sets how many consecutive days to stream; the loader stops early once the widest column
# hits its byte cap. Each hour is ~82 MB compressed, so raising this costs staging time.
GHARCHIVE_URL = "https://data.gharchive.org/{date}-{hour}.json.gz"
#   landing page: https://www.gharchive.org/
GHARCHIVE_START = date(2024, 10, 1)
GHARCHIVE_DAYS = int(os.environ.get("ONPAIR_GHARCHIVE_DAYS", "10"))


def _gharchive_urls() -> list[str]:
    return [
        GHARCHIVE_URL.format(date=(GHARCHIVE_START + timedelta(days=d)).isoformat(), hour=h)
        for d in range(GHARCHIVE_DAYS)
        for h in range(24)
    ]

# Optional pre-existing local copies to reuse instead of downloading. Point
# ONPAIR_LOCAL_<DATASET> (e.g. ONPAIR_LOCAL_CLICKBENCH, ONPAIR_LOCAL_BOOK_REVIEWS)
# at an absolute parquet path to skip that dataset's download; ignored if unset or
# absent. Lets a reproducer reuse data already on the box without editing this file.
def _local(dataset_id: str) -> list[Path]:
    env = os.environ.get("ONPAIR_LOCAL_" + dataset_id.upper().replace("-", "_"))
    return [Path(env)] if env else []


@dataclass
class Column:
    """One column to benchmark."""

    dataset_id: str
    column: str
    kind: str  # "tpch" | "tpcds" | "parquet" | "parquet_stream" | "text" | "amazon"
    #          | "jsonl" | "synthetic"
    # tpch / tpcds
    scale_factor: float = 10.0
    table: str = "lineitem"
    # parquet
    url: str | None = None
    cache: str | None = None  # filename under SRC_DIR/<dataset_id>/
    local: list[Path] = field(default_factory=list)
    # synthetic
    rows: int = 10_000_000  # row count for the `synthetic` generator
    # synthetic: distinct filler path segments, widening the OnPair dictionary. 0 keeps the
    # original seed-123 corpus byte-for-byte; positive values are the selector ladder.
    synth_vocab: int = 0
    # OnPair's training-shuffle seed, for EVERY column -- not the synthetic ladder only, which is
    # what this field used to be. Upstream `TrainingConfig::seed` is a std::optional and its
    # comment is explicit: "nullopt -> non-deterministic. Set for reproducible compression (same
    # dictionary across runs)." The shim only forwards a nonzero value, so the old default of 0
    # left it nullopt and every materialization trained a DIFFERENT dictionary.
    #
    # That was not theoretical. On 2026-08-19 an NCU capture set (materialized 08-18) and a
    # coarsening sweep (materialized 08-19) disagreed about the best K on 4 of 8 OnPair-16 cells
    # while agreeing 8/8 at OnPair-12, and the two runs' compressed sizes for the same column and
    # preset differed by 1.8-2.6% -- different dictionaries, same code. OnPair-16 is hit harder
    # because 65,536 entries leave far more room for training to diverge than 4,096 do.
    #
    # CONSEQUENCE OF PINNING: `training_seed != 0` adds `_seed<N>` to the cell directory name, so
    # this bypasses every existing materialization and re-creates it. That is the point -- the seed
    # becomes visible provenance in the path -- but it means numbers measured before this landed
    # were taken on dictionaries that cannot be reproduced, and must be re-measured to be citable
    # as reproducible.
    training_seed: int = ONPAIR_TRAINING_SEED
    # amazon
    category: str | None = None  # HF Amazon-Reviews-2023 category, e.g. "Books"
    # parquet_stream / jsonl: every column of the dataset shares one cache file, so the
    # loader must materialize the whole set on first touch. `siblings` is that set; the
    # column that gets there first writes the cache for all of them.
    siblings: tuple[str, ...] = ()
    # jsonl: dotted path into each record, per column name (e.g. "actor.login").
    json_paths: tuple[tuple[str, str], ...] = ()
    # jsonl: the URLs to stream, in order, until `cap_bytes` is reached.
    urls: tuple[str, ...] = ()
    # parquet_stream / jsonl: stop once this many extracted UTF-8 bytes have accumulated.
    # 0 means the loader's own default.
    cap_bytes: int = 0
    # Immutable source revision when the remote registry supports one. This is repeated
    # in the cache manifest even when it is also embedded in the resolve URL.
    source_revision: str | None = None
    # loghub: the log file's name inside the archive. Explicit rather than derived from the URL,
    # because the two do not correspond -- HDFS_v1.zip contains HDFS.log, not HDFS_v1.log.
    member: str | None = None
    # loghub: an fnmatch pattern selecting MANY members, concatenated in archive order (tar, a
    # forward-only stream) or sorted order (zip) until the cap. Mutually exclusive with `member`.
    #
    # Two Loghub systems are not shipped as a single log file: Spark is 3,852 per-container logs
    # across 194 applications, Android_v2 is 78 logcat captures. Concatenating those is
    # PARTITIONING, not the re-extraction duplication that disqualified Public BI -- verified
    # 2026-08-20 by line-set intersection over Android's four largest members, which share 0.0%
    # of their lines. Public BI's numbered files were the same rows in a different order (97.7%
    # mutual containment), so concatenating them doubled every value's frequency and inflated
    # exactly the ratios this corpus exists to measure.
    members: str | None = None

    def tpch_dir(self) -> Path:
        return SRC_DIR / f"tpch_sf{_scale_factor_tag(self.scale_factor)}"

    def tpcds_dir(self) -> Path:
        # Matches the path TpcDsBenchmark/generate_tpcds use.
        return DATA_DIR / "tpcds" / _scale_factor_tag(self.scale_factor)

    def cache_path(self) -> Path:
        return SRC_DIR / self.dataset_id / (self.cache or f"{self.column}.parquet")

    def parquet_path(self) -> Path:
        """Resolved source: TPC-H/TPC-DS generated path, an existing local copy,
        or the repo-relative cache (download target)."""
        if self.kind == "tpch":
            return self.tpch_dir() / "parquet" / f"{self.table}_0.parquet"
        if self.kind == "tpcds":
            return self.tpcds_dir() / "parquet" / f"{self.table}.parquet"
        if self.kind in ("parquet", "parquet_stream", "text", "amazon", "jsonl", "pbi", "loghub"):
            for p in self.local:
                if Path(p).exists():
                    return Path(p)
            return self.cache_path()
        if self.kind == "synthetic":
            return self.cache_path()
        raise ValueError(f"unknown source kind {self.kind!r}")


def _scale_factor_tag(scale_factor: float) -> str:
    """Round-trip-safe path component for a TPC scale factor."""
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError(f"scale factor must be finite and positive, got {scale_factor!r}")
    text = str(scale_factor)
    # Keep the established `tpch_sf10` paths, but never truncate a fractional factor: int(26.3)
    # used to alias sf26 and could silently reuse a parquet generated for the wrong row count.
    return text[:-2] if text.endswith(".0") else text


def _parquet_cols(dataset_id, columns, *, url, cache):
    return [
        Column(dataset_id=dataset_id, column=c, kind="parquet", url=url, cache=cache,
               local=_local(dataset_id))
        for c in columns
    ]


def _stream_cols(dataset_id, columns, *, urls, cache, cap_bytes=0, source_revision=None):
    """Columns served by a byte-capped remote-parquet read (`parquet_stream`). `urls` are
    read in order until the widest column reaches the cap, so a corpus whose per-shard
    yield falls short of the sampled chunk can span shards."""
    sib = tuple(columns)
    us = (urls,) if isinstance(urls, str) else tuple(urls)
    return [
        Column(dataset_id=dataset_id, column=c, kind="parquet_stream", url=us[0], urls=us,
               cache=cache, siblings=sib, cap_bytes=cap_bytes,
               source_revision=source_revision, local=_local(dataset_id))
        for c in columns
    ]


def _pbi_cols(dataset_id, columns, *, workbook, tables, cache, cap_bytes=0):
    """Public BI columns, concatenated across a workbook's numbered parts until every column
    reaches `cap_bytes`. `tables` is the part order; only parts whose `.table.sql` column list
    matches the first part's are read, so a workbook whose parts are actually different tables
    stops rather than extracting the wrong column position."""
    sib = tuple(columns)
    us = tuple(f"{PBI_DATA_BASE}/{workbook}/{t}.csv.bz2" for t in tables)
    return [
        Column(dataset_id=dataset_id, column=c, kind="pbi", url=us[0], urls=us,
               cache=cache, siblings=sib, cap_bytes=cap_bytes, local=_local(dataset_id))
        for c in columns
    ]


def _loghub_cols(dataset_id, column, *, archive, member=None, members=None, cap_bytes=0):
    """A Loghub system's logs as a single string column, one log line per value.

    `member` names one `.log` inside the archive -- explicitly, because it does not follow the
    archive name (HDFS_v1.zip contains HDFS.log). `members` is an fnmatch pattern for the systems
    shipped as many files (Spark, Android), concatenated until the cap. Exactly one of the two."""
    return [Column(dataset_id=dataset_id, column=column, kind="loghub",
                   url=f"{LOGHUB_BASE}/{archive}?download=1",
                   urls=(f"{LOGHUB_BASE}/{archive}?download=1",),
                   member=member, members=members, cache=f"{dataset_id}.parquet",
                   cap_bytes=cap_bytes, local=_local(dataset_id))]


def _jsonl_cols(dataset_id, paths, *, urls, cache, cap_bytes=0):
    """Columns extracted from a gzipped JSON-lines stream. `paths` maps each output
    column name to a dotted path into the record."""
    sib = tuple(paths)
    jp = tuple(paths.items())
    return [
        Column(dataset_id=dataset_id, column=c, kind="jsonl", cache=cache, siblings=sib,
               json_paths=jp, urls=tuple(urls), cap_bytes=cap_bytes, local=_local(dataset_id))
        for c in paths
    ]


def _dbtext_cols(columns):
    return [
        Column(dataset_id="dbtext", column=c, kind="text",
               url=f"{DBTEXT_URL_BASE}/{c}", cache=f"{c}.parquet")
        for c in columns
    ]


# Every TPC-H string column, by table. Single-char / few-value columns
# (returnflag, linestatus, orderstatus, mktsegment, brand, ...) are intentionally
# included: they show OnPair *expanding* low-cardinality data, where a value
# dictionary wins.
_TPCH_STR_COLS: dict[str, list[str]] = {
    "region": ["r_name", "r_comment"],
    "nation": ["n_name", "n_comment"],
    "supplier": ["s_name", "s_address", "s_phone", "s_comment"],
    "customer": ["c_name", "c_address", "c_phone", "c_mktsegment", "c_comment"],
    "part": ["p_name", "p_mfgr", "p_brand", "p_type", "p_container", "p_comment"],
    "partsupp": ["ps_comment"],
    "orders": ["o_orderstatus", "o_orderpriority", "o_clerk", "o_comment"],
    "lineitem": ["l_returnflag", "l_linestatus", "l_shipinstruct", "l_shipmode", "l_comment"],
}

# A representative spread of TPC-DS string columns across cardinalities.
_TPCDS_STR_COLS: dict[str, list[str]] = {
    "item": ["i_item_desc", "i_product_name", "i_brand", "i_class", "i_category"],
    "customer": ["c_email_address", "c_first_name", "c_last_name",
                 "c_birth_country", "c_preferred_cust_flag"],
    "customer_address": ["ca_street_name", "ca_city", "ca_zip", "ca_state", "ca_country"],
}

_DBTEXT_COLS = [
    "hex", "yago", "email", "wiki", "uuid", "urls2", "urls",
    "firstname", "lastname", "city", "credentials", "street", "movies",
    "faust", "hamlet", "chinese", "japanese", "wikipedia",
    "genome", "location",
    "c_name", "l_comment", "ps_comment",
]

# Per-column scale factors for the generated arm of the corpus, measured 2026-08-20 at sf1 and
# confirmed at sf30. Every corpus column must supply the 1 GB the methodology samples, and TPC-H's
# string columns differ by ~40x in payload per scale factor, so one shared factor cannot do it:
#
#   column           payload/SF   SF for ~1 GB   what it contributes
#   l_comment          159 MB          15        pseudo-text (FSST dbtext lineage)
#   l_shipinstruct      72 MB          15        enum, 4 values (scale-invariant)
#   ps_comment          74 MB          15        pseudo-text at 124 ch -- the length endpoint
#   o_clerk           22.5 MB          45        templated identifier, capacity-bound
#   c_address          3.8 MB         263        random characters; OnPair EXPANDS it (0.95x)
#
# lineitem and partsupp share sf15, so three generations cover five columns. Generating all eight
# tables at sf263 would mean a 1.6-billion-row lineitem that nothing reads -- hence the per-table
# selection in `gen-tpch`.
_TPCH_CORPUS_SF: dict[float, dict[str, list[str]]] = {
    15.0: {"lineitem": ["l_comment", "l_shipinstruct"], "partsupp": ["ps_comment"]},
    45.0: {"orders": ["o_clerk"]},
    263.0: {"customer": ["c_address"]},
}

# The benchmark registry. Append a one-line `Column(...)` to grow the suite.
COLUMNS: list[Column] = [
    *(
        Column(dataset_id="tpch-sf10", column=c, kind="tpch", scale_factor=10.0, table=t)
        for t, cols in _TPCH_STR_COLS.items()
        for c in cols
    ),
    # The generated corpus arm, each at the factor its column needs. Kept ALONGSIDE the sf10 set
    # rather than replacing it: sf10 entries carry every committed measurement (including the
    # l_comment cells the 2026-08-19 locked legs hold), and changing their identity would silently
    # orphan that data.
    *(
        Column(dataset_id=f"tpch-sf{_scale_factor_tag(sf)}", column=c, kind="tpch",
               scale_factor=sf, table=t)
        for sf, tables in _TPCH_CORPUS_SF.items()
        for t, cols in tables.items()
        for c in cols
    ),
    *(
        Column(dataset_id="tpcds-sf10", column=c, kind="tpcds", scale_factor=10.0, table=t)
        for t, cols in _TPCDS_STR_COLS.items()
        for c in cols
    ),
    # ClickBench "hits" — long high-cardinality URLs/titles vs short categoricals.
    # `OriginalURL` joins URL/Title/Referer as a fourth multi-GB column in the same file:
    # once hits.parquet is on the box, another column costs sweep time and nothing else.
    # PageCharset has 1,176,679,095 UTF-8 payload bytes over 10 distinct values:
    # l_shipinstruct's tiny-dictionary regime on real data. SearchPhrase is 86.8% empty
    # strings and has only 739,636,342 payload bytes — the low-fill case, not a
    # like-for-like throughput cell.
    *_parquet_cols("clickbench", ["URL", "Title", "Referer", "OriginalURL", "PageCharset",
                                  "SearchPhrase", "MobilePhoneModel"],
                   url=CLICKBENCH_URL, cache="hits.parquet"),
    # FineWeb 10BT sample — long free text + URLs + low-cardinality categoricals.
    *_parquet_cols("fineweb", ["text", "url", "file_path", "dump", "language"],
                   url=FINEWEB_URL, cache="fineweb_10BT_000.parquet"),
    # Wikipedia (en, 2023-11-01) — long encyclopaedic free text, titles, URLs. Spans three
    # shards via parquet_stream because one shard is 0.70 GB, short of the 1 GB the sweep samples.
    *_stream_cols("wikipedia", ["text", "title", "url"],
                  urls=WIKIPEDIA_URLS, cache="wikipedia_20231101_en_000.parquet",
                  cap_bytes=1_150_000_000),
    # Public BI: the long-token end of the corpus, which real data had to cover because the
    # spread previously came from dbgen output. Part counts are sized from measured per-part
    # yield per SINGLE table: psc_code_description 0.52 GB, naics_name 0.37, co_name 0.29,
    # Subsector 0.60, "Transaction ID" 0.75.
    #
    # ONE TABLE EACH, DELIBERATELY. The numbered files in a Public BI workbook are NOT
    # partitions -- they are re-extractions of the same workbook in different row order.
    # Measured 2026-08-19 on CommonGovernment_1 vs _2 over 400k-row windows: rows are 99.8%
    # unique, mutual containment is 97.7% / 97.6%, and only 0.6% align row-for-row. Decisive
    # confirmation on the 2-file case via raincloud's merged copy of RealEstate1, where row
    # 19,531,359 (exactly N/2 of 39,062,718) is byte-identical to row 0.
    #
    # So concatenating them duplicates the data and doubles every value's frequency, which
    # inflates compression ratio -- and deduplicating is not neutral either, because it flattens
    # the true frequency distribution. Either way the statistics stop describing the source
    # column. Consequence: no Public BI column here can reach 1 GB, which is why NONE of them is
    # in the paper's ten-column corpus. They are kept registered as valid single-table columns.
    #
    # (Upstream note: raincloud's `public_bi_merge` handler treats these files as partitions and
    # concatenates them without dedup, so its `bi-*` tables are duplicated by their file count --
    # CommonGovernment ~13x. Worth reporting; it distorts exactly the ratios that corpus exists
    # to measure.)
    *_pbi_cols("publicbi-commongovernment",
               ["psc_code_description", "naics_name", "co_name"],
               workbook="CommonGovernment", tables=["CommonGovernment_3"],
               cache="pbi_commongovernment.parquet", cap_bytes=1_150_000_000),
    *_pbi_cols("publicbi-generico", ["Subsector"],
               workbook="Generico", tables=["Generico_5"],
               cache="pbi_generico.parquet", cap_bytes=1_150_000_000),
    *_pbi_cols("publicbi-realestate1", ["Transaction ID"],
               workbook="RealEstate1", tables=["RealEstate1_2"],
               cache="pbi_realestate1.parquet", cap_bytes=1_150_000_000),
    # Loghub: five logging systems, spanning the short-token axis from 38% to 89% at OnPair-12.
    # Windows is the low anchor (38% / 21%).
    *_loghub_cols("loghub-windows", "line", archive="Windows.tar.gz",
                  member="Windows.log", cap_bytes=1_150_000_000),
    *_loghub_cols("loghub-thunderbird", "line", archive="Thunderbird.tar.gz",
                  member="Thunderbird.log", cap_bytes=1_150_000_000),
    *_loghub_cols("loghub-hdfs", "line", archive="HDFS_v1.zip",
                  member="HDFS.log", cap_bytes=1_150_000_000),
    # Spark: 55% / 44%, measured 2026-08-20 on a 64 MiB sample accumulated across containers.
    # It takes the rung `l_comment` used to hold, and takes it with REAL data -- l_comment sits
    # at 57% because dbgen draws from a 3,775-word list with 1,060x reuse per word, against
    # 66,898 words at 60x reuse for real relational text. Spark was previously excluded for
    # "sitting on l_comment's rung", which was the wrong reason: that rung is a grammar artifact.
    # 3,852 container logs over 194 applications, 2.94 GB uncompressed, so `members`.
    *_loghub_cols("loghub-spark", "line", archive="Spark.tar.gz",
                  members="*.log", cap_bytes=1_150_000_000),
    # Android: 89% / 46%, the largest preset gap of any real column, and a domain nothing else in
    # the corpus covers (mobile logcat). 78 members, 3.62 GB uncompressed. The `duplicate_type1/`
    # directory names refer to duplicate BUG REPORTS -- it is a duplicate-detection dataset -- not
    # duplicated log content: the four largest members share 0.0% of their lines (measured).
    # Fetched whole (445 MB) because zip keeps its directory at the end and cannot be streamed.
    *_loghub_cols("loghub-android", "line", archive="Android_v2.zip",
                  members="*.log", cap_bytes=1_150_000_000),
    # CodeParrot: source code as its own domain. Uses the existing jsonl loader.
    *_jsonl_cols("codeparrot", {"content": "content"},
                 urls=CODEPARROT_URLS, cache="codeparrot_clean.parquet",
                 cap_bytes=1_150_000_000),
    # FSST paper's dbtext corpus: 23 raw text columns under cwida/fsst.
    *_dbtext_cols(_DBTEXT_COLS),
    # OnPair paper's book-reviews corpus (single `text` column). Reproduced on-box
    # from Amazon-Reviews-2023 "Books" (McAuley Lab), streamed by run.py.
    Column(dataset_id="book-reviews", column="text", kind="amazon", category="Books",
           cache="book_reviews.parquet", local=_local("book-reviews")),
    # Two further Amazon-Reviews-2023 categories with contrasting token profiles:
    # Movies_and_TV (long narrative review prose) and Electronics (short product text).
    Column(dataset_id="amazon-movies", column="text", kind="amazon", category="Movies_and_TV",
           cache="amazon_movies.parquet", local=_local("amazon-movies")),
    Column(dataset_id="amazon-electronics", column="text", kind="amazon", category="Electronics",
           cache="amazon_electronics.parquet", local=_local("amazon-electronics")),
    # CodeSearchNet all-languages split: whole functions, their docstrings, and the
    # repository paths they came from — source code beside prose in one corpus.
    *_stream_cols("codesearchnet",
                  ["whole_func_string", "func_documentation_string",
                   "func_code_url", "repository_name"],
                  urls=CODESEARCHNET_URLS, cache="code_search_net_all_train.parquet",
                  source_revision=CODESEARCHNET_REVISION),
    # FineWeb2 Mandarin: multibyte UTF-8, where the share of tokens under 8 bytes — the
    # selector's main input — is far lower than in any of the Latin-script corpora.
    *_stream_cols("fineweb2-zh", ["text", "url"],
                  urls=FINEWEB2_ZH_URL, cache="fineweb2_cmn_Hani_000.parquet",
                  source_revision=FINEWEB2_REVISION),
    # GH Archive 2024-10-01: machine-generated event records. `type` is a ~20-value
    # categorical, `repo.name` and `actor.login` are high-cardinality identifiers.
    # Catalogued but deliberately OUT of the campaign sweep: its widest column is only
    # ~4.8 MB per hourly shard, so filling a 1 GB chunk means streaming ~10 days (~20 GB
    # of downloads and JSON parsing) on every ephemeral box, for columns that stay short.
    *_jsonl_cols("gharchive",
                 {"type": "type", "actor_login": "actor.login", "repo_name": "repo.name"},
                 urls=_gharchive_urls(),
                 cache=f"gharchive_2024-10-01_{GHARCHIVE_DAYS}d.parquet"),
    # Synthetic ClickBench-style URL corpus (deterministic, seed 123) — the
    # micro-benchmark workload, regenerated in-pipeline via `gen-synth-urls`
    # (no external source). Column name `url` matches the paper's synthetic row.
    Column(dataset_id="synthetic", column="url", kind="synthetic",
           cache="synthetic_urls.parquet"),
    # Dictionary-cardinality ladder. The measured corpus clusters near 870, 4096 and 65536
    # dictionary entries with nothing in between, so any selector threshold placed inside
    # those gaps classifies no observation and cannot be fitted -- the Ada rule's 2048 sits
    # in exactly such a gap, which is why 1024 and 2048 score identically. These candidate
    # rungs are intended to put observations in the gap so the threshold can be fitted.
    # Same base URL stream, plus one fixed-width path segment selected by an independent
    # RNG from `synth_vocab` distinct strings. One-eighth of rows use the diverse tail and
    # the rest use a common sentinel, holding diversity frequency fixed while pool size
    # changes. The rungs are candidates from CPU preflight; `gpu.dict_entries_max`, not the
    # vocabulary value or `dict_bytes`, decides which cells actually straddle the gate.
    # Cache names include the generator revision so corpora from the earlier
    # variable-width/single-RNG design cannot be silently reused.
    *(
        # training_seed=1 is retained deliberately, NOT updated to ONPAIR_TRAINING_SEED: the
        # selector-ladder rungs were measured at seed 1 and are only comparable to each other.
        # These three synthetic columns were cut from the corpus on 2026-08-19 in favour of real
        # columns covering the same regimes, so this is a frozen record rather than live config.
        Column(dataset_id=f"synthdict-{v}", column="url", kind="synthetic",
               synth_vocab=v, training_seed=1,
               cache=f"synthetic_urls_fixed3_sparse8_v2_vocab{v}.parquet")
        for v in (512, 1024, 2048, 3072, 6144, 16384)
    ),
]
