#!/usr/bin/env python3
"""Regression tests for the DE input dumpers.

WHAT THESE PROTECT. The hardware Decompression Engine is fed from `dump_columns.py`, and its
compression ratio and decode rate are both computed against what this file writes. Two properties
have to hold or those numbers are measured on a different experiment than every other technique:

  1. The framed dumper samples the SAME ROWS as the Rust `Sample` -- which stops before the first
     row whose bytes would cross the cap, and never splits one. Truncating instead produced a
     stream that framed correctly and agreed on payload bytes, so the byte check in de_stage.py
     accepted it while the row set differed.
  2. Nulls are refused rather than silently dropped, since this framing carries no validity.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("dump_columns", BENCH / "dump_columns.py")
assert spec and spec.loader
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


def rust_sample(parquet, column, cap):
    """onpair_bench.rs build_sample, and de_stage.py's onpair_sample: (payload_bytes, rows)."""
    total = 0
    rows = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64 * 1024, columns=[column]):
        for value in batch.column(0).to_pylist():
            size = len(value.encode("utf-8")) if value is not None else 0
            if total + size > cap:
                return total, rows
            total += size
            rows += 1
    return total, rows


def parse_framed(path):
    """Read back a u32-length-prefixed stream as a list of values."""
    out = []
    raw = Path(path).read_bytes()
    i = 0
    while i < len(raw):
        (n,) = struct.unpack_from("<I", raw, i)
        i += 4
        out.append(raw[i:i + n])
        i += n
    return out


class DumpColumnPrefixedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self._real_resolve = dc.resolve_parquet

    def write(self, values):
        path = self.dir / "col.parquet"
        pq.write_table(pa.table({"c": values}), path)
        dc.resolve_parquet = lambda ds, col, _p=path: _p
        self.addCleanup(setattr, dc, "resolve_parquet", self._real_resolve)
        return path

    def test_matches_the_rust_sample_at_every_cap(self):
        # The case that motivated this: at cap 3 the Rust sampler keeps "abc" and the empty row,
        # while a truncating dumper kept one row and still reported three payload bytes.
        parquet = self.write(["abc", "", "d", "efgh"])
        out = str(self.dir / "out.bin")
        for cap in (0, 1, 2, 3, 4, 7, 8, 100):
            want_bytes, want_rows = rust_sample(parquet, "c", cap)
            payload, fbytes, rows = dc.dump_column_prefixed("ds", "c", out, cap)
            self.assertEqual((payload, rows), (want_bytes, want_rows), "cap=%d" % cap)
            self.assertEqual(fbytes, payload + 4 * rows, "cap=%d: file is payload plus framing" % cap)

    def test_never_splits_a_row(self):
        parquet = self.write(["aaaa", "bbbb"])
        out = str(self.dir / "out.bin")
        # Cap falls inside the second value: it must be dropped whole, not truncated to 2 bytes.
        payload, _fbytes, rows = dc.dump_column_prefixed("ds", "c", out, 6)
        self.assertEqual((payload, rows), (4, 1))
        self.assertEqual(parse_framed(out), [b"aaaa"])
        self.assertEqual(rust_sample(parquet, "c", 6), (4, 1))

    def test_round_trips_the_values_it_kept(self):
        self.write(["alpha", "", "gamma", "d"])
        out = str(self.dir / "out.bin")
        payload, _fbytes, rows = dc.dump_column_prefixed("ds", "c", out, 1 << 20)
        got = parse_framed(out)
        self.assertEqual(got, [b"alpha", b"", b"gamma", b"d"])
        self.assertEqual((payload, rows), (11, 4))

    def test_a_first_value_longer_than_the_cap_yields_an_empty_sample(self):
        # Called with a small cap directly from the CLI, this used to write one truncated record
        # holding a possibly-invalid UTF-8 prefix, where Rust samples nothing at all.
        self.write(["abcdef"])
        out = str(self.dir / "out.bin")
        self.assertEqual(dc.dump_column_prefixed("ds", "c", out, 3), (0, 0, 0))
        self.assertEqual(parse_framed(out), [])

    def test_nulls_are_refused(self):
        self.write(["a", None, "b"])
        out = str(self.dir / "out.bin")
        with self.assertRaises(ValueError) as e:
            dc.dump_column_prefixed("ds", "c", out, 1 << 20)
        self.assertIn("null", str(e.exception))


class MainArgumentTest(unittest.TestCase):
    def test_length_prefix_flag_is_parsed_before_the_positional_cap(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        pq.write_table(pa.table({"c": ["abc"]}), d / "col.parquet")
        real = dc.resolve_parquet
        dc.resolve_parquet = lambda ds, col, _p=d / "col.parquet": _p
        self.addCleanup(setattr, dc, "resolve_parquet", real)
        # `int("--length-prefix")` used to raise here, before the flag was stripped.
        rc = dc.main(["dump_columns.py", "ds", "c", str(d / "o.bin"), "--length-prefix"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
