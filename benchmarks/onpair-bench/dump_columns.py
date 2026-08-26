# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Materialize the raw decoded bytes of a benchmark column to a flat file.

Single source of truth for "give me the exact bytes OnPair decodes for (dataset,
column)": concatenate the column's UTF-8 string values sequentially, no
separators, up to a byte cap (mirrors ``onpair_bench.rs``: ``raw_bytes +=
value.len()``). The (dataset, column) -> source-parquet resolution comes from the
``columns.py`` registry (``Column.parquet_path()``) — the SAME resolver ``run.py``
uses — so nothing here hardcodes a parquet path that can drift from the registry.

Consumers: the nvCOMP hardware-DE bench (``nvcomp_hw_bench.cu``) and software
bench (``nvcomp_sw_bench.cu``) both take a raw byte file as input; feeding them
from this dumper guarantees they compress the IDENTICAL bytes OnPair decodes, on
the same box — a true same-bytes comparison.

CLI:
    python3 dump_columns.py <dataset_id> <column> <out_path> [cap_bytes]

Importable:
    from dump_columns import dump_column, resolve_parquet
    n = dump_column("tpch-sf10", "l_comment", "/tmp/x.bin")   # bytes written

Exit codes (CLI): 0 wrote bytes; 3 source parquet missing (caller should SKIP);
2 usage / unknown column. A missing source is a normal "not materialized yet"
condition, distinct from a real error, so callers can skip cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

import columns as _columns

# run.py's default --sample-bytes: the OnPair sample cap. Same bytes, same cap.
DEFAULT_CAP = 1_000_000_000

# (dataset_id, column) -> Column, built once from the registry.
_BY_KEY = {(c.dataset_id, c.column): c for c in _columns.COLUMNS}


def resolve_parquet(dataset_id: str, column: str) -> Path:
    """Resolve (dataset, column) to its source parquet via the registry.

    Raises KeyError if the pair is not in the registry.
    """
    col = _BY_KEY.get((dataset_id, column))
    if col is None:
        raise KeyError(f"{dataset_id}/{column} not in the columns.py registry")
    return col.parquet_path()


def dump_column(dataset_id: str, column: str, out_path: str, cap: int = DEFAULT_CAP) -> int:
    """Write the column's concatenated UTF-8 bytes (to ``cap``) to ``out_path``.

    Returns the number of bytes written. Raises KeyError (unknown column) or
    FileNotFoundError (source parquet not materialized).
    """
    parquet = resolve_parquet(dataset_id, column)
    if not parquet.exists():
        raise FileNotFoundError(f"source parquet missing: {parquet}")
    tmp = out_path + ".part"
    total = 0
    # Batch-stream so a 1 GB cap never materializes the whole column in memory.
    with open(tmp, "wb") as out:
        for b in pq.ParquetFile(parquet).iter_batches(batch_size=1 << 20, columns=[column]):
            chunk = "".join(v for v in b.column(0).to_pylist() if v is not None).encode("utf-8")
            if total + len(chunk) > cap:
                chunk = chunk[: cap - total]
            out.write(chunk)
            total += len(chunk)
            if total >= cap:
                break
    Path(tmp).replace(out_path)
    return total


def dump_column_prefixed(dataset_id: str, column: str, out_path: str,
                         cap: int = DEFAULT_CAP) -> tuple[int, int]:
    """Write the column as a u32-length-prefixed record stream. Returns (payload, file) bytes.

    WHY. The flat dump above loses row boundaries, so a compressor fed from it is measured on a
    byte stream rows cannot be recovered from -- while nvCOMP Zstd prefixes every string with a u32
    length ("the same as what Parquet does") and OnPair stores row-to-code offsets. That put the
    hardware DE on a basis no other technique used. This dumper closes it: same bytes, same cap on
    the PAYLOAD, plus the row structure every other technique already pays for.

    The cap counts payload only, so the file is larger than `cap` by 4 bytes per row. The bench must
    be told the payload size (NVCOMP_PAYLOAD_BYTES) or it will treat prefix bytes as decoded output.
    """
    import struct

    parquet = resolve_parquet(dataset_id, column)
    if not parquet.exists():
        raise FileNotFoundError(f"source parquet missing: {parquet}")
    tmp = out_path + ".part"
    payload = 0
    written = 0
    with open(tmp, "wb") as out:
        for b in pq.ParquetFile(parquet).iter_batches(batch_size=1 << 20, columns=[column]):
            for v in b.column(0).to_pylist():
                if v is None:
                    continue
                raw = v.encode("utf-8")
                if payload + len(raw) > cap:
                    raw = raw[: cap - payload]
                    if not raw:
                        break
                out.write(struct.pack("<I", len(raw)))
                out.write(raw)
                payload += len(raw)
                written += 4 + len(raw)
                if payload >= cap:
                    break
            if payload >= cap:
                break
    Path(tmp).replace(out_path)
    return payload, written


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("usage: dump_columns.py <dataset_id> <column> <out_path> [cap_bytes]", file=sys.stderr)
        return 2
    ds, col, out = argv[1], argv[2], argv[3]
    cap = int(argv[4]) if len(argv) > 4 else DEFAULT_CAP
    if "--length-prefix" in argv:
        argv = [a for a in argv if a != "--length-prefix"]
        try:
            payload, fbytes = dump_column_prefixed(ds, col, out, cap)
        except KeyError:
            print(f"unknown column: {ds}/{col}", file=sys.stderr)
            return 2
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 3
        print(f"{payload} {fbytes}")
        return 0
    try:
        n = dump_column(ds, col, out, cap)
    except KeyError as e:
        print(f"dump_columns: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"dump_columns: SKIP {ds}/{col}: {e}", file=sys.stderr)
        return 3
    print(f"dump_columns: {ds}/{col} -> {out} ({n} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
