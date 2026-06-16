#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Python (PyO3) path for the Vortex Arrow boundary microbenchmark (experiment E4).

Reads the file produced by `ffi-boundary-bench gen` and runs the same consume workload as the
native Rust runner (sum the ``value`` column), so the difference is the Rust->CPython Arrow
C-stream boundary cost. Requires the ``vortex-data`` wheel to be built/installed, e.g.:

    uv run --reinstall-package vortex-data python runners/python_runner.py --file /tmp/boundary.vortex
"""

import argparse
import time

import vortex as vx


def bench(path: str, iters: int, warmup: int, batch_size: int | None) -> None:
    times: list[float] = []
    rows = 0
    batches = 0
    for it in range(iters + warmup):
        scan = vx.open(path).scan(batch_size=batch_size) if batch_size else vx.open(path).scan()
        reader = scan.to_arrow()  # pyarrow.RecordBatchReader, backed by the Arrow C stream
        t0 = time.perf_counter()
        rows = 0
        batches = 0
        # Minimal consume: iterating forces decode + the Arrow C-stream import, without a per-row
        # compute that would differ by language and swamp the boundary cost we are isolating.
        for batch in reader:
            rows += batch.num_rows
            batches += 1
        dt = time.perf_counter() - t0
        if it >= warmup:
            times.append(dt)

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[min(int(len(times) * 0.95), len(times) - 1)]
    print(f"python  rows={rows} batches={batches}")
    print(f"  p50={p50 * 1e3:.3f}ms  p95={p95 * 1e3:.3f}ms")
    print(f"  {rows / p50 / 1e6:.1f} Mrows/s  {p50 / max(batches, 1) * 1e9:.0f} ns/batch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True, help="Vortex file produced by `ffi-boundary-bench gen`")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None, help="rows per batch (for the per-batch sweep)")
    args = ap.parse_args()
    bench(args.file, args.iters, args.warmup, args.batch_size)
