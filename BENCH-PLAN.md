<!--
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->
# Batched small-row-group decode bench (`batch_decode`)

## What this is for

The FastPair paper reports its small columns (`dbtext`, TPC-H `s_comment`; 1–6 MB) only
"for completeness": a single such column launches under a fifth of one wave of thread
blocks and reaches ~3% occupancy, so it is latency/launch-bound, not throughput-bound
(§6.4, `sec:eval:caveat`). A reviewer's fair objection: *a real file is not one small
column — it is hundreds of small row-groups, each independently OnPair-compressed with its
own dictionary.* This bench answers whether decoding a realistic file's worth of small
row-groups (hundreds of 1–6 MB groups, ~1 GB total) in one launch — or a small number of
streams / a CUDA graph — recovers throughput-bound rates.

It compares four ways to launch that work over the **same** staged inputs, all validated
byte-exact against a CPU reference, all reported as aggregate GB/s (decoded output bytes /
min-over-iters time, decimal `1e9`):

| mode | what it does | isolates |
|---|---|---|
| `sequential` | N row-groups as N sequential kernel launches (the shipped per-chunk path) | the per-launch fixed cost — the caveat's cause |
| `streams` | the same N launches round-robined over K CUDA streams | filling the device by overlapping many small grids |
| `graph` | the same N launches captured once into a CUDA graph, replayed | killing per-launch **host-submit** overhead (still N grids) |
| `multidict` | **one** launch spanning **every** row-group, via a per-warp-batch dictionary indirection | filling the device with a single grid sized by the whole file |

## Why the corpus must be synthesized (scoping)

The paper's *actual* small columns are tiny in aggregate: `dbtext` + TPC-H `s_comment`
total ~16 MB ≈ 64 warp-batches, which cannot fill a 150-SM B300 even batched. So the bench
does **not** decode those columns as-is. It decodes a **synthesized realistic corpus**: a
large column sliced into hundreds of 1–6 MB row-groups, **each independently
OnPair-compressed with its own dictionary**, totaling ~1 GB. The existing harness already
does exactly this slicing — `--chunk-mb N` compresses each ~N-MB chunk with its own
dictionary (`run_cell` → one `OnPairArray` per chunk). Setting `--chunk-mb 1` over a ~1 GB
sample of ClickBench `URL` yields ~1000 independently-dicted row-groups; `--chunk-mb 4`
yields ~250. That is the corpus. (The real small columns can optionally be concatenated
into the same dump for a "real-tail" variant, but the headline corpus is the sliced large
column, so per-row-group size is a controlled sweep knob rather than an artifact of which
tiny columns happen to exist.)

## Input preparation

The row-group dump is produced by the existing Rust decode path, env-gated, mirroring
`ONPAIR_DUMP_E2E`. A new `ONPAIR_DUMP_BATCH=<path>` env var:

- truncates `<path>` once per cell (in `run_gpu_kernel_bench`), then
- **appends** one self-describing record per row-group (in `stage_gpu_chunk`, right after
  the `ONPAIR_DUMP_E2E` block), in chunk order.

Record format (repeated until EOF; no top-level header, append-friendly):

```
"RGB1" | total_tokens:u64 | dict_size:u32 | max_token:u32
      | codes(u16 LE, total_tokens) | lens(u8, dict_size)
      | dict_padded(u8, dict_size*max_token)
```

This is the `E2E1` body, once per row-group. `max_token` is `vortex_onpair::MAX_TOKEN_SIZE`
(16). The bench derives `dict_s8`, per-token offsets, per-128-token-batch chunk offsets, the
concatenated multi-dict layout, and the CPU reference from these — nothing else is needed.

**Scope a dump run to a single cell** so the append stream is neither interleaved by
concurrent columns (`run.py --jobs` spawns one process per column) nor clobbered by a second
cell: one `--datasets`, one `--columns`, one `--bits`, one `--chunk-mb`, one `--threshold`.

Example (produce the 4-MB-row-group, ~1 GB corpus from ClickBench `URL` at dict-16):

```bash
ONPAIR_DUMP_BATCH=/tmp/clickbench_url_rg4.batchbin \
  python benchmarks/onpair-bench/run.py --gpu-decode \
    --datasets clickbench --columns URL \
    --bits 16 --chunk-mb 4 --threshold 0.2 \
    --sample-bytes 1000000000 --jobs 1
```

`--gpu-decode` triggers the CUDA staging path (where the dump lives); the run also times the
kernel family as usual, but only the dump is needed here. One `.batchbin` file per cell.

## Build (on a GPU box)

Standalone, exactly like `e2e_scan.cu` — the bench `#include`s the two kernel `.cu` files by
relative path, so it must be built from `benchmarks/onpair-bench/`:

```bash
cd benchmarks/onpair-bench
nvcc -O3 -arch=native -std=c++17 batch_decode.cu -o batch_decode
```

The input-prep side (the `ONPAIR_DUMP_BATCH` dump) needs a `--features cuda` build of the
Rust harness, which `run.py --gpu-decode` does automatically (see
`benchmarks/onpair-bench/README.md` §Prerequisites: CUDA ≥ 12.x, `nvcc` on `PATH`,
`-arch=native` targets the present device).

### Review status (2026-07-17)

The two `.cu` and the Rust dump edit passed a 4-lens gauntlet (gpt-5.6-terra @ xhigh) plus a
gpt-5.6-sol @ xhigh meta-review. Fixes applied before the box: 16-aligned per-row-group output
bases (the shared-buffer `uint4 __stcs` were otherwise misaligned in sequential/streams/graph);
the RGB1 writer now drops the dictionary's 16-byte staging pad (the multi-record dump otherwise
desynced on the next magic); reader ABI/bounds gates (`max_token==16`, code/lens/empty-record
rejection) mirroring `e2e_scan.cu`; streams timing moved off host `cudaStreamSynchronize` to a
non-blocking-worker + start/completion-event topology; nonzero validation poison; a grid-packing
confound metric; propagated dump I/O errors. See the review digest in the campaign runbook.

### Still requires the box (no CUDA toolchain on the dev Mac)

- **Neither `.cu` was compiled**; the Rust dump edit sits in `#[cfg(feature="cuda")]` and was
  not type-checked. First on-box step: `nvcc` build + one byte-exact multi-record smoke cell
  (full and short final batches) requiring all four `_ok=true`.
- **CUDA-graph replay** — confirm the instantiated graph replays and validates byte-exact on
  the target driver (the 3-arg `cudaGraphInstantiate` and stream-0 replay are API-valid).
- **Multi-stream overlap** — the workers are now `cudaStreamNonBlocking` with an explicit
  ev0→worker→completion-event→stream-0 topology; confirm in Nsight Systems that the K grids
  actually overlap rather than serialize.
- **Occupancy** — the §6.4 rewrite's achieved-occupancy claim is NOT derivable from the bench
  JSON; collect it per mode with Nsight Compute, e.g.
  `ncu -k regex:onpair_shmem --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./batch_decode <dump>`
  (compare the multidict one-grid launch against a single sequential row-group launch), or drop
  the occupancy number from the claim. Fill in measured achieved-occupancy and GB/s from the run.

## Run matrix

Per-row-group-size sweep at a fixed ~1 GB total, ClickBench `URL` (long, high-cardinality —
the paper's representative throughput column), B300 as the headline device:

| row-group size | `--chunk-mb` | ≈ #row-groups @ 1 GB | dump file |
|---|---|---|---|
| 1 MB | 1 | ~1000 | `clickbench_url_rg1.batchbin` |
| 2 MB | 2 | ~500 | `clickbench_url_rg2.batchbin` |
| 4 MB | 4 | ~250 | `clickbench_url_rg4.batchbin` |
| 6 MB | 6 | ~167 | `clickbench_url_rg6.batchbin` |

For each dump, run all four modes in one invocation:

```bash
./batch_decode <dump.batchbin> [iters=100] [streams=8]
```

The bench emits one JSON object with all four modes' rates. Sweep `streams` (e.g. 4, 8, 16,
32) on one dump to find the K where streams saturates. The headline comparison is
`sequential` (launch-bound baseline) vs `multidict` (one grid) vs the best of
`streams`/`graph`. Optionally add the real small-column tail (`dbtext`, `s_comment`) as an
extra dump to show the identical mechanism on the genuine data, not only the sliced corpus.

Reproduce across GPUs (A100, L40S, H100, B300) by building and running on each; `-arch=native`
retargets automatically.

## Expected outputs (JSON schema)

`batch_decode` prints one JSON object to stdout (diagnostics to stderr), e.g.:

```json
{
  "gpu": "NVIDIA B300",
  "sm": "10.0",
  "kernel": "onpair_shmem_4tpt_split8read+multidict",
  "n_row_groups": 250,
  "total_tokens": 167000000,
  "total_batches": 1310000,
  "total_decoded_bytes": 1000000000,
  "total_dict_entries": 1024000,
  "mean_rg_mb": 4.0,
  "min_rg_bytes": 3900000,
  "max_rg_bytes": 4100000,
  "iters": 100,
  "n_streams": 8,
  "sequential_ok": true, "streams_ok": true, "graph_ok": true, "multidict_ok": true,
  "sequential_ms": 0.0, "streams_ms": 0.0, "graph_ms": 0.0, "multidict_ms": 0.0,
  "sequential_ns_iters": [ ... ], "streams_ns_iters": [ ... ],
  "graph_ns_iters": [ ... ], "multidict_ns_iters": [ ... ],
  "sequential_gbps": 0.0, "streams_gbps": 0.0, "graph_gbps": 0.0, "multidict_gbps": 0.0,
  "multidict_speedup_over_sequential": 0.0
}
```

- `*_ms` is the min over `iters` timed iterations (CUDA events), per the paper's estimator.
- `*_ns_iters` is the raw per-iteration nanoseconds (all iters), so any reduction (min,
  median, dispersion) is recomputable at figure-generation — matching the harness's gold
  per-iteration provenance convention.
- `*_gbps` = `total_decoded_bytes / (ms/1e3) / 1e9`, aggregated across all row-groups.
- exit code 0 iff **all four** modes are byte-exact; non-zero otherwise.

## Validation story

Byte-exactness, not just self-consistency:

1. The bench builds a CPU reference by decoding every row-group independently (token →
   `dict_padded[code]`), concatenated in row-group order — the same oracle style as
   `e2e_scan.cu`.
2. Each of the four modes decodes into the same global output buffer layout (`sequential`,
   `streams`, `graph` write per-row-group slices with 0-based chunk offsets — byte-identical
   to the real per-chunk harness path; `multidict` writes the same global offsets via the
   per-batch metadata). Each mode's device output is copied back and `memcmp`'d against the
   CPU reference; `*_ok` reports the result and the process exits non-zero on any failure.
3. The `multidict` kernel's decode/scan/drain is copied verbatim from the shipped
   split-read kernel; only the addressing (per-batch token bound, per-row-group dict base,
   per-batch output offset) differs, so a byte-exact `multidict_ok` confirms the indirection
   is correct while the decode remains the shipped one.

## The paper claim this supports (§6.4 rewrite)

Current §6.4 (`sec:eval:caveat`) concedes the small columns are launch-bound and reports
them "only for completeness." With this result, that caveat can become a *resolved* point
rather than an apology. Proposed rewrite (fill in the measured numbers from the run):

> A single 1–6 MB column is latency/launch-bound: it launches under a fifth of one wave of
> thread blocks and reaches roughly 3% achieved occupancy, so its gather latency cannot be
> hidden and the fixed launch cost cannot amortize. But a column of that size is not how a
> file is decoded. A file is hundreds of such row-groups, each independently compressed with
> its own dictionary, and they need not be decoded one launch at a time. Decoding a
> realistic file — ~1 GB of ClickBench `URL` sliced into N independently-dicted row-groups
> of 1–6 MB — recovers the throughput-bound regime: a single launch that spans every
> row-group (a per-warp-batch dictionary indirection over the shipped decode kernel, output
> byte-identical) sustains **X GB/s**, versus **Y GB/s** for the same work as N sequential
> launches — a **Z×** recovery — and reaches **W% occupancy** against the ~3% of the
> one-column launch. Overlapping the N launches across K streams, or replaying them from a
> CUDA graph, recovers most of the same rate without a kernel change, isolating the penalty
> as per-launch fixed cost rather than anything intrinsic to small row-groups. The small
> columns are launch-bound only when decoded one at a time; at file granularity they are
> throughput-bound like the rest.

This turns the weakest cell in the evaluation into a demonstration that the request-reducing
decode's throughput regime is reached at *file* granularity, which is the granularity that
matters in practice.
