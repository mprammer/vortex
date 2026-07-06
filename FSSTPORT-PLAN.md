# FSST recipe port — design, run plan, risks

Porting the FastPair/OnPair GPU **decode recipe** — fixed-stride table + warp
exclusive-scan of lengths + staged-then-aligned 16-B drain — to **unmodified
FSST-compressed data**. This is the *generality* experiment: show the recipe is
a codec-agnostic decode pattern, benchmarkable on an A100 against GSST's
published **191 GB/s** (same chip, same codec, same column: TPC-H `l_comment`)
and against this tree's thread-per-string `fsst.cu`.

Worktree: `~/repos/vortex@fsst-port`  •  branch: `mp/fsst-recipe`
(based on `mp/onpair-synth-corpus`). Committed locally; **not pushed**.

## Deliverables

| File | Role |
|---|---|
| `vortex-cuda/kernels/src/fsst_4tpt.cu` | the recipe port (heavily commented) |
| `benchmarks/onpair-bench/fsst_scan.cu` | standalone bench: naive vs. recipe, byte-exact validation, min-of-N CUDA-event timing, JSON |
| `vortex-cuda/benches/vortex_fsst_real_data.rs` | `+ FSST_DUMP` hook: encodes a column with the tree's FSST encoder and writes the `FST1` blob the standalone reads |
| `FSSTPORT-PLAN.md` | this file |

## What the recipe maps to, and the two real differences

The kernel body is `onpair_shmem_4tpt.cu` reused nearly verbatim: one warp per
128-unit batch, 4 units per lane, warp scan of lengths → per-unit output offset,
staged per-lane byte writes into a shared scratch shifted to the global cursor's
alignment, cooperative head/body(16-B `__stcs`)/tail drain. "Token" becomes
"code". Two things differ, and they are the whole substance of the port:

### 1. Symbol table lives in SHARED memory (the interesting contrast)

OnPair's dictionary is large (up to `dict_size × 16 B`, often KB–MB) and is left
**L1/L2-resident** — the request traffic to it *is* the paper's cache-request-
bound story. FSST's symbol table is at most **255 symbols × 8 B ≈ 2 KB**, so it
fits comfortably in shared memory. The port stages it **once per block**; every
`symbols[code]` lookup is then a shared-memory bank read instead of an L1
request. Same recipe, but the codec's tiny table takes the lookup *off* the
cache-request path entirely. This is the point worth making in the paper: the
recipe's table-residency knob adapts to the codec.

(Note: the tree's own `fsst.cu` measured staging the table into shared as ~3%
*slower* for its thread-per-string design — because each thread reads the table
directly and L1 already holds it. The recipe is different: 32 lanes hammer the
table concurrently every batch, so the bank-read path should pay off. Whether it
actually does is an **A100 measurement**, not a settled claim — see risks.)

### 2. Escapes make the code→byte map position-dependent (escape handling)

FSST codes are **1 byte**, except code **255 = ESCAPE**, meaning "the next input
byte is a literal". So a code is 1 input byte, or 2 for an escape. Critically,
escape detection is **inherently sequential**: you cannot classify input byte *i*
as a code vs. an escaped literal without parsing forward from a known boundary
(an escaped literal may itself be the value 255). A lane therefore cannot
random-access "its" codes the way OnPair does (`chunk*128 + lane + k*32`, valid
only because OnPair codes are fixed-width).

**Chosen design — host-precomputed per-lane input offsets + per-lane serial
parse of a contiguous code run.** This is the option the task flagged as
"simplest correct", and it keeps the warp-scan/drain structure intact:

- Each lane owns a **contiguous** run of `FSST_CODES_PER_LANE = 4` codes
  (`[batch*128 + L*4, +4)`), and **parses them serially**, branching on 255.
- The host precomputes `group_in_off[b*32 + L]` = the input byte offset of that
  run's first code. This is the FSST analog of OnPair's `chunk_offsets`, but at
  **4-code (one-lane) granularity** rather than 128-code (one-warp), because
  escape boundaries can't be rediscovered in-warp — each lane needs its own
  start. It is host-side setup (like `chunk_offsets`) and is **not timed**.
- Because lanes now own contiguous runs, codes concatenate **lane-major**
  (code index `= lane*4 + k`), not OnPair's k-major interleave. So the scan is
  *one* warp scan over the per-lane 4-code total (for the lane base) plus a tiny
  local 4-length prefix — actually **cheaper** than OnPair's four scans.

Why this over the alternatives:
- *On-device intra-warp boundary discovery* (host gives only per-128 offsets,
  device finds per-lane starts): impossible without serializing the warp, since
  escape classification is sequential. Rejected.
- *Per-thread variable code count over a fixed input-byte run*: makes the
  warp-scan bookkeeping (which lane emits how many codes) data-dependent and
  divergent, breaking the clean fixed-4 scan. Rejected in favour of fixed codes
  per lane + host offsets. Correctness and recipe fidelity beat peak cleverness
  here (explicit task guidance).

The cost is that `group_in_off` is dense: `~total_codes/4` `u32`s ≈ `total_codes`
bytes of side data — larger than OnPair's `chunk_offsets`. For the paper's ~1 GB
`l_comment` (~0.5 GB compressed, a few hundred M codes) that is a few hundred MB
resident alongside the ~1 GB output — fine on an A100, but a reviewer will note
it is comparable to the compressed input. Framing: this is a *recipe-generality
demonstration*, not a production FSST on-disk layout. A production design would
store periodic sync points (per-256 codes, say) and do a short serial walk per
lane; call that out as future work, don't hide it. `CODES_PER_LANE` is a
`#define` if we want to trade offset density for per-lane serial work later.

Byte-exactness is by construction (a code emits exactly `len` bytes of its
symbol, or 1 literal byte; the scan lays them out in code order; the drain copies
shared→global unchanged) and is **validated every run** by the standalone bench
against a CPU reference decode, with a first-mismatch byte index reported on
failure.

## Data flow

```
vortex_fsst_real_data.rs  --FSST_DUMP-->  <base>.<col>.fsstbin  (FST1 blob:
   train compressor on the column, compress each string, dump
   symbols/symbol_lengths/codes_bytes/codes_offsets/output_offsets)
                                   |
                                   v
fsst_scan.cu <dump> [iters]:
   host walk of codes_bytes  ->  total_codes, CPU reference decode (oracle),
                                 group_in_off (per-4-code), batch_out_off (per-128)
   drive fsst_u64 (naive, per-string)     -> validate vs oracle
   drive fsst_4tpt (recipe)               -> validate vs oracle
   min-of-N CUDA events each; JSON {naive_gbps, recipe_gbps, recipe_vs_gsst, ...}
```

The blob dumps the natural FSST arrays; the standalone derives the recipe's
`group_in_off`/`batch_out_off` on the host (exactly as `e2e_scan.cu` derives
`dict_s8`/`chunk_offsets` from the OnPair dump).

## What could NOT be verified locally

**This Mac has no CUDA compiler (`nvcc` absent) and no NVIDIA GPU.** Therefore:

- `fsst_4tpt.cu` and `fsst_scan.cu` were **not compiled** — no `nvcc` syntax
  check, no PTX, no run. Reasoned correct by construction and by close
  structural parity with the shipped, verified `onpair_shmem_4tpt.cu` and
  `fsst.cu`, but treat "compiles" as unverified.
- The `vortex_fsst_real_data.rs` dump hook was **not `cargo check`ed**: the
  `vortex-cuda` crate's `build.rs` invokes `nvcc`, so the crate cannot build on
  this machine. The Rust was written against the confirmed `fsst-rs` 0.5.10 API
  (`Compressor::{symbol_table, symbol_lengths, compress}`, `Symbol::to_u64`) and
  the existing bench's Arrow-column plumbing, but treat it as unbuilt.
- No throughput numbers exist yet. Every GB/s figure is produced on the A100.

First action on the A100 box: `nvcc -O3 -arch=native -std=c++17 fsst_scan.cu -o
fsst_scan` and fix whatever the compiler flags before trusting anything.

## A100 run plan

Target: TPC-H `l_comment` at ~1 GB decoded (paper's headline column; matches the
GSST comparison), plus the FSST paper's `dbtext` corpus for breadth.

1. **Get data.** `l_comment`: `benchmarks/onpair-bench/run.py` generates TPC-H
   SF10 (`lineitem` → ~1.6 GB raw `l_comment`); or cap to ~1 GB. `dbtext`:
   `run.py` fetches `cwida/fsst` `paper/dbtext` text columns.
2. **Dump the blob** (CPU-side, on the box):
   ```
   FSST_DUMP=/tmp/fsst \
   ONPAIR_DATA_PATH=<...>/tpch_sf10/lineitem.parquet \
   cargo bench -p vortex-cuda --bench vortex_fsst_real_data
   # -> /tmp/fsst.l_comment.fsstbin (and one file per eligible string column)
   ```
3. **Build + run the standalone:**
   ```
   cd benchmarks/onpair-bench
   nvcc -O3 -arch=native -std=c++17 fsst_scan.cu -o fsst_scan
   ./fsst_scan /tmp/fsst.l_comment.fsstbin 100   # JSON on stdout
   ```
   Confirm `naive_ok` and `recipe_ok` are `true` FIRST (byte-exact), then read
   `recipe_gbps`, `naive_gbps`, `recipe_speedup_over_naive`, `recipe_vs_gsst`.
4. **Compare.** `recipe_gbps` vs **GSST 191 GB/s** (both are decoded-bytes/s on
   an A100 over `l_comment`) and vs `naive_gbps` (this tree's `fsst.cu`). Repeat
   for the `dbtext` blobs.
5. **If the recipe wins**, capture the shared-vs-L1 table contrast with NCU
   (`long_scoreboard`, shared-load throughput) to substantiate the "table off
   the cache-request path" claim rather than asserting it.

Note on fairness: report decode GB/s = `decoded_bytes / kernel_ms`, with the
offset tables pre-staged in HBM and un-timed — exactly how OnPair times its
kernel with `chunk_offsets` pre-staged. Disclose `group_in_off`'s size (see
above) so the comparison is honest.

## Risk list — where the port most likely has bugs

1. **Intra-warp byte ordering.** The switch from OnPair's k-major interleave to
   lane-major contiguous runs is the highest-risk change. If the scan's
   `lane_base + local_prefix` is off, output is scrambled but *plausible-looking*
   (right length, wrong order). The byte-exact validation catches it; the
   `recipe_first_mismatch` index localizes it. **Check this first.**
2. **Escape at a lane/batch boundary.** A lane's 4-code run may end on an escape
   whose literal is the run's last consumed input byte — fine (parse reads
   `codes_bytes[s+1]` within the same contiguous stream). But verify the host
   `group_in_off` walk and the device walk advance identically (`+2` on 255,
   `+1` otherwise) so a lane starts exactly where the previous lane's codes end.
   A drift here corrupts from that lane on.
3. **`total_codes` vs input bytes.** `total_codes` is the code *count*, not
   `codes_bytes.len()` (escapes make them differ). The host walk computes it;
   an off-by-one in the active-slot guard (`gidx < total_codes`) would drop or
   duplicate the last few codes of the final batch.
4. **Offset-array padding / sizing.** `group_in_off` padded to
   `num_batches*32 + 1`, `batch_out_off` to `num_batches + 1`. An
   under-allocation is an OOB device read on the final partial batch.
5. **`u32` input offsets.** `group_in_off` is `u32`; the bench asserts
   `num_input_bytes < 2^32`. A column with >4 GB *compressed* would need `u64`
   offsets — not the ~1 GB target, but a footgun for larger runs.
6. **Shared-memory budget.** 16 warps × 1088 B scratch + 2 KB table ≈ 20 KB/
   block, under the 48 KB default — but if `FSST_WARPS_PER_BLOCK_MAX` or the
   block size is raised, re-check against the SM limit (and `__launch_bounds__`).
7. **Shared table actually a win?** The premise (staging pays off for the
   concurrent-lane recipe even though it didn't for thread-per-string `fsst.cu`)
   is a hypothesis. If `recipe_gbps` disappoints, try the L1-resident-table
   variant (read `symbols[code]` from global directly) as an A/B — the code
   change is one line.
8. **Symbol length edge cases.** The all-zero symbol is length 1 (a 0x00 byte),
   and `symbol_lengths[255]` is unused (escape). The dump writes `num_symbols`
   lengths and the standalone zero-pads to 256; a mismatch between the encoder's
   `symbol_lengths()` order and `symbol_table()` order would corrupt decode.
   Validation catches it.
