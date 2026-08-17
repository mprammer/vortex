#!/usr/bin/env python3
"""Regression tests for streamed-corpus cache identity and publication."""

from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("onpair_run", BENCH / "run.py")
assert spec and spec.loader
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
Column = run.Column


class BodyResponse(io.BytesIO):
    def __init__(self, data, *, headers=None):
        super().__init__(data)
        self.status = 200
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class ScriptedOpener:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def jsonl_column(dest: Path) -> Column:
    column = Column(
        dataset_id="events",
        column="kind",
        kind="jsonl",
        cache=dest.name,
        siblings=("kind",),
        json_paths=(("kind", "type"),),
        urls=("https://origin/events.json.gz",),
        cap_bytes=100,
    )
    column.cache_path = lambda: dest
    return column


class StreamCacheTests(unittest.TestCase):
    def test_identity_includes_cap_ordered_urls_and_revision(self):
        column = Column(
            dataset_id="code",
            column="text",
            kind="parquet_stream",
            url="https://origin/a",
            urls=("https://origin/a", "https://origin/b"),
            siblings=("text",),
            cap_bytes=100,
            source_revision="rev-a",
        )
        identity = run._stream_cache_identity(column)
        for changed in [
            dataclasses.replace(column, cap_bytes=101),
            dataclasses.replace(column, urls=("https://origin/b", "https://origin/a")),
            dataclasses.replace(column, source_revision="rev-b"),
        ]:
            self.assertNotEqual(identity, run._stream_cache_identity(changed))

    def test_published_cache_is_accepted_only_for_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "cache.parquet"
            column = Column(
                dataset_id="code",
                column="text",
                kind="parquet_stream",
                url="https://origin/a",
                urls=("https://origin/a",),
                siblings=("text",),
                cap_bytes=100,
                source_revision="rev-a",
            )
            identity = run._stream_cache_identity(column)
            run._publish_stream_cache(
                pa.table({"text": ["alpha", "beta"]}),
                dest,
                identity,
                {
                    "sources": [{"url": "https://origin/a", "etag": '"a"'}],
                    "utf8_bytes": {"text": 9},
                    "stop_reason": "all_sources_exhausted",
                    "shards_read": 1,
                },
            )
            self.assertEqual(run._stream_cache_status(dest, identity), (True, "identity and parquet metadata match"))
            changed = {**identity, "cap_bytes": 101}
            self.assertEqual(run._stream_cache_status(dest, changed), (False, "manifest identity differs"))


class JsonlLoaderTests(unittest.TestCase):
    def test_429_is_retried_before_cache_publication(self):
        compressed = gzip.compress(b'{"type":"PushEvent"}\n{"type":"IssueEvent"}\n')
        rate_limit_body = io.BytesIO(b"rate limited")
        rate_limit = urllib.error.HTTPError(
            "https://origin/events.json.gz", 429, "rate limited", {}, rate_limit_body
        )
        response = BodyResponse(
            compressed,
            headers={"ETag": '"events-a"', "Content-Length": str(len(compressed))},
        )
        scripted = ScriptedOpener([rate_limit, response])
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "events.parquet"
            column = jsonl_column(dest)
            with (
                patch.object(run._HTTP_OPENER, "open", scripted),
                patch.object(run, "_retry_delay", return_value=0),
            ):
                run.jsonl_to_parquet(column)
            self.assertTrue(dest.is_file())
            manifest = json.loads(run._cache_manifest_path(dest).read_text())
            self.assertEqual(manifest["row_count"], 2)
            self.assertEqual(manifest["utf8_bytes"], {"kind": 19})
            self.assertEqual(len(scripted.requests), 2)
            self.assertTrue(rate_limit_body.closed)

    def test_exhausted_403_and_malformed_json_publish_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, outcomes, expected in [
                (
                    "forbidden",
                    [
                        urllib.error.HTTPError(
                            "https://origin/events.json.gz", 403, "forbidden", {}, io.BytesIO()
                        )
                        for _ in range(run.HTTP_MAX_ATTEMPTS)
                    ],
                    urllib.error.HTTPError,
                ),
                (
                    "malformed",
                    [BodyResponse(gzip.compress(b"not-json\n"))],
                    ValueError,
                ),
            ]:
                with self.subTest(name=name):
                    dest = Path(directory) / f"{name}.parquet"
                    scripted = ScriptedOpener(outcomes)
                    with (
                        patch.object(run._HTTP_OPENER, "open", scripted),
                        patch.object(run, "_retry_delay", return_value=0),
                        self.assertRaises(expected),
                    ):
                        run.jsonl_to_parquet(jsonl_column(dest))
                    self.assertFalse(dest.exists())
                    self.assertFalse(run._cache_manifest_path(dest).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
