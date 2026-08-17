# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Registry of (dataset, column) sources for the OnPair compression benchmark.

Adding a new column is a one-line append to ``COLUMNS``.

Source kinds:

* ``tpch``    — generated locally via the Rust ``gen-tpch`` subcommand (all
                tables, one parquet file each).
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
#   landing page: https://huggingface.co/datasets/wikimedia/wikipedia
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
# listed because the first alone yields 0.86 GB of `whole_func_string`, just short.
_CSN_BASE = ("https://huggingface.co/datasets/code-search-net/code_search_net/"
             "resolve/main/all/")
CODESEARCHNET_URLS = [_CSN_BASE + f"train-0000{i}-of-00002.parquet" for i in (0, 1)]
#   landing page: https://huggingface.co/datasets/code-search-net/code_search_net
# FineWeb2 Mandarin (cmn_Hani) — the multibyte-UTF-8 case. Every shard is ~4.8 GB, far more
# than the ~1 GB the bench samples, so this uses `parquet_stream`: row groups arrive over
# HTTP range requests until the byte cap is hit, and the shard is never fetched whole.
FINEWEB2_ZH_URL = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/"
                   "resolve/main/data/cmn_Hani/train/000_00000.parquet")
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

    def tpch_dir(self) -> Path:
        return SRC_DIR / f"tpch_sf{int(self.scale_factor)}"

    def tpcds_dir(self) -> Path:
        # Matches the path TpcDsBenchmark/generate_tpcds use.
        return DATA_DIR / "tpcds" / f"{int(self.scale_factor)}"

    def cache_path(self) -> Path:
        return SRC_DIR / self.dataset_id / (self.cache or f"{self.column}.parquet")

    def parquet_path(self) -> Path:
        """Resolved source: TPC-H/TPC-DS generated path, an existing local copy,
        or the repo-relative cache (download target)."""
        if self.kind == "tpch":
            return self.tpch_dir() / "parquet" / f"{self.table}_0.parquet"
        if self.kind == "tpcds":
            return self.tpcds_dir() / "parquet" / f"{self.table}.parquet"
        if self.kind in ("parquet", "parquet_stream", "text", "amazon", "jsonl"):
            for p in self.local:
                if Path(p).exists():
                    return Path(p)
            return self.cache_path()
        if self.kind == "synthetic":
            return self.cache_path()
        raise ValueError(f"unknown source kind {self.kind!r}")


def _parquet_cols(dataset_id, columns, *, url, cache):
    return [
        Column(dataset_id=dataset_id, column=c, kind="parquet", url=url, cache=cache,
               local=_local(dataset_id))
        for c in columns
    ]


def _stream_cols(dataset_id, columns, *, urls, cache, cap_bytes=0):
    """Columns served by a byte-capped remote-parquet read (`parquet_stream`). `urls` are
    read in order until the widest column reaches the cap, so a corpus whose per-shard
    yield falls short of the sampled chunk can span shards."""
    sib = tuple(columns)
    us = (urls,) if isinstance(urls, str) else tuple(urls)
    return [
        Column(dataset_id=dataset_id, column=c, kind="parquet_stream", url=us[0], urls=us,
               cache=cache, siblings=sib, cap_bytes=cap_bytes, local=_local(dataset_id))
        for c in columns
    ]


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

# The benchmark registry. Append a one-line `Column(...)` to grow the suite.
COLUMNS: list[Column] = [
    *(
        Column(dataset_id="tpch-sf10", column=c, kind="tpch", scale_factor=10.0, table=t)
        for t, cols in _TPCH_STR_COLS.items()
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
    # PageCharset is 0.96 GB over 4 distinct values: l_shipinstruct's tiny-dictionary
    # regime on real data. SearchPhrase is 86.8% empty strings — the low-fill case, not
    # a like-for-like throughput cell.
    *_parquet_cols("clickbench", ["URL", "Title", "Referer", "OriginalURL", "PageCharset",
                                  "SearchPhrase", "MobilePhoneModel"],
                   url=CLICKBENCH_URL, cache="hits.parquet"),
    # FineWeb 10BT sample — long free text + URLs + low-cardinality categoricals.
    *_parquet_cols("fineweb", ["text", "url", "file_path", "dump", "language"],
                   url=FINEWEB_URL, cache="fineweb_10BT_000.parquet"),
    # Wikipedia (en, 2023-11-01) — long encyclopaedic free text, titles, URLs.
    *_parquet_cols("wikipedia", ["text", "title", "url"],
                   url=WIKIPEDIA_URL, cache="wikipedia_20231101_en_000.parquet"),
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
    # CodeSearchNet (Python split): whole functions, their docstrings, and the repository
    # paths they came from — source code beside prose in one corpus.
    *_stream_cols("codesearchnet",
                  ["whole_func_string", "func_documentation_string",
                   "func_code_url", "repository_name"],
                  urls=CODESEARCHNET_URLS, cache="code_search_net_all_train.parquet"),
    # FineWeb2 Mandarin: multibyte UTF-8, where the share of tokens under 8 bytes — the
    # selector's main input — is far lower than in any of the Latin-script corpora.
    *_stream_cols("fineweb2-zh", ["text", "url"],
                  urls=FINEWEB2_ZH_URL, cache="fineweb2_cmn_Hani_000.parquet"),
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
]
