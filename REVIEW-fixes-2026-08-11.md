# Fix-focused review of `35bd4af17`

This review is limited to `HEAD~1..HEAD` and to whether the round-three fixes do what
they claim. The original nine-item must-fix list was taken from gauntlet run
`cli-gauntlet-98202264c7a6`. No CUDA compilation or execution was possible because this
machine has no `nvcc`.

## Findings

### High — experimental eligibility still missed `_split8read_ldcs` (corrected)

`vortex-bench/src/onpair_bench.rs:1226-1284` registers
`onpair_shmem_4tpt_split8read_ldcs` in the experiment group, and
`vortex-cuda/kernels/src/README-experiments-0810.md:9` describes it as an experiment to
promote only if it wins. The committed predicate at `onpair_bench.rs:2641-2643` matched
only `_lookback`, `_stcsedge`, and `_hilo`. Therefore an unvalidated `_ldcs` result could
still become `best_kernel`.

I corrected this by also matching the specific `_split8read_ldcs` name. The narrower
substring is intentional: the shipped `onpair_shmem_4tpt_ldcs` kernel must remain eligible.
All currently registered experiment rows are now covered, including the three
`report_as` look-back aliases, and no shipped row matches the predicate.

### High — the failed-H2D sentinel silently corrupted JSON (corrected)

The committed `None => f64::NAN` path fed directly into the result writers at
`vortex-bench/src/onpair_bench.rs:879` and
`vortex-bench/src/bin/onpair-chunk-bench.rs:181,199`. This workspace uses
`serde_json 1.0.149`, which serializes non-finite floats as JSON `null`. That is silent
data corruption here: the same result type required an `f64`, so reading the written
`null` back into `GpuCellResult` would fail.

I corrected `h2d_gib_s` and `whole_decompress_gib_s` to `Option<f64>` with
`serde(default)` and omission for `None`, and added a diagnostic for an H2D measurement
error or non-positive result (`onpair_bench.rs:215-223,1633-1661,1729-1736`). Previously
written numeric values deserialize as `Some(value)`; missing fields deserialize as
`None`. No in-repository consumer compares either field.

### Medium — original must-fix 7 was not touched

`vortex-bench/src/onpair_bench.rs:798` still uses `w.flush().ok()`. A final buffered-write
failure can therefore truncate `ONPAIR_OFFSET_COST` output while the benchmark reports
success. This is the original must-fix 7 and remains a **code gap**. It should propagate
the flush error with the same path context used for each `writeln!`.

### Medium — original must-fix 8 remains only partially documented

The round-three appendix was added, but the canonical manifest header and table remain
internally contradictory:

- `README-experiments-0810.md:3-5` says the fifth kernel is unregistered, while line 13
  says look-back is registered and the Rust registry contains four look-back geometries.
- The table still lists both deregistered `_bounds` probes as if they were the experiment
  set and omits the registered `_stcsedge` variant entirely.
- Consequently `_stcsedge` has no question or retirement/promotion criterion in the
  canonical table, and the look-back aliases are not inventoried there.

The appendix does not repair those stale statements. Original must-fix 8 remains a
**documentation gap**.

### Medium — original must-fix 4 remains deliberately open

There is still no CUDA compile-smoke or differential test for look-back or streaming-edge
behavior. The source itself says it has never been compiled or run and that
`--gpu-validate` is the only byte-exact check
(`onpair_shmem_4tpt_split8read_lookback.cu:73-75` and
`README-experiments-0810.md:196-199`). The requested boundary, width, interleaving,
epoch/concurrency, and head/body/tail cases are absent. This is a **code/test gap**.

### Low — ticket-ring hardening is claimed but not implemented

This was should-fix 10 rather than one of the original nine must-fixes, but the commit
claims to fix it. The Rust and CUDA values currently agree at 16,384, and the CUDA
`static_assert` correctly proves that the CUDA value is a power of two. They are still two
independent constants, however, and there is no host startup agreement assertion.
`onpair_shmem_4tpt_split8read_lookback.cu:105-107` incorrectly says such an assertion
exists, while `README-experiments-0810.md:185` incorrectly calls the value canonical.
Cross-language drift therefore remains a code-hardening gap plus a false documentation
claim.

## Per-fix audit

### 1. Epoch scoping — pass, with a stale dead declaration

`fused_epoch` sits on the same `GpuOnPairChunk` that owns `fused_ticket`,
`fused_part_agg`, `fused_part_inc`, and `fused_part_flag`, and every staged chunk initializes
its own atomic. It is therefore scoped to the scratch owner rather than to a benchmark
cell or process.

The `fetch_update` closure still fails closed: it uses `checked_add(1)` and updates only
when `next < FUSED_TICKET_SLOTS`. At exhaustion the atomic remains pinned at 16,383;
retries cannot wrap it. `fetch_update` is one atomic read-modify-write operation, so all
successful calls on the same chunk occupy a single modification order and return distinct
previous values. `Ordering::Relaxed` is sufficient for uniqueness; concurrent callers
cannot receive the same epoch.

No path reads the old global. `FUSED_EPOCH` does remain as an unused declaration at
`onpair_bench.rs:1092`; it should be deleted to avoid a stale CUDA-feature warning and
misleading documentation, but it has no behavioral effect. This uniqueness check does not
assert that simultaneous launches sharing the payload arrays are otherwise safe; that was
the separate should-fix 11, not the requested epoch-allocation check.

### 2. Dynamic shared memory — pass at source level

- Host `ONPAIR_WARP_BUF_BYTES` and CUDA `WARP_BUF_BYTES` are both 2,080 bytes.
- The four registrations launch 16, 4, 2, or 1 warps, and the same `block_warps` value
  determines both `blockDim.x` and `cfg.shared_mem_bytes`. Requested sizes are therefore
  33,280, 8,320, 4,160, and 2,080 bytes.
- `s_buf_all` has 16-byte alignment, and 2,080 is a multiple of 16, so every per-warp
  slice begins aligned. The shift applied to `s_buf` equals `out_start mod 16`; adding
  `head_pre` makes both `s_buf + off` and `output_bytes + out_start + off` 16-byte aligned
  for every `uint4` body load/store. The dictionary staging reads are unchanged.
- One warp can emit at most `128 * 16 = 2,048` bytes. The alignment shift is at most 15,
  so the highest staged/drained byte is at per-slice offset 2,062, below 2,080. The last
  `uint4` load is also bounded by `body_chunks`, so it ends no later than
  `warp_total - 1`. No registered width requests less dynamic memory than the kernel can
  index.
- `s_warp_tot` remains a fixed 16-element `uint64_t` array (128 bytes), bounded by the
  512-thread launch contract. Together with `s_blk` and `s_block_excl`, static shared
  usage remains tiny; even the largest dynamic request stays below the ordinary 48 KiB
  per-block limit.
- The PTX symbol is referenced only by the four `FusedPositions` registrations, and all
  four traverse the host branch that sets dynamic shared memory. No other kernel or launch
  path shares the symbol.

These are source-level conclusions only; CUDA syntax, resource accounting, and runtime
behavior remain uncompiled here.

### 3. Experimental eligibility — failed as committed, corrected

The three look-back aliases all contain `_lookback`, so filtering the serialized report
label does not miss them. `_stcsedge` and `_hilo` have no aliases and are also matched.
The committed predicate nevertheless missed the registered `_split8read_ldcs` experiment.
After the correction, the exact current experiment set is covered and the shipped
`onpair_shmem_4tpt_ldcs` name is not.

### 4. `kernel_symbol` — pass

There are exactly two `GpuKernelResult` construction sites, for inapplicable and timed
rows (`onpair_bench.rs:1651-1661,1676-1686` in the commit), and both set
`kernel_symbol = variant.name`. The reporting label still comes from `report_name()`.
`#[serde(default)]` makes a missing field deserialize as an empty string, so old result
JSON remains readable. The empty value cannot reconstruct historical symbol identity, but
it does not break deserialization.

### 5. H2D and nvCOMP restoration — failed as committed, corrected

There was no downstream numeric comparison of the NaN, but direct JSON serialization was
enough to make the committed fix wrong. The working-tree correction now represents
unavailability explicitly and emits no non-finite metric.

The nvCOMP-ZSTD block is fully restored. Extracting the block from `let
(nvcomp_zstd_hw, nvcomp_zstd)` up to `Ok(GpuCellResult` produced the same SHA-256 in
`HEAD~1`, `HEAD`, and the corrected worktree:

`b2b487428d31fcf3232990c253f8560ae8058d1b9a8475b202e450c1ef7285ab`

The zero-context commit diff has no other unexplained deletion: every removed code line is
the old H2D calculation, fixed-size shared buffer, global epoch call, or launch-config
binding that its replacement supersedes. Nothing else from the parent is missing.

## Original nine must-fix ledger

“By commit” describes `35bd4af17` before the two corrections made during this review.

| # | Item | Status by commit | Gap type | Working tree after review |
|---:|---|---|---|---|
| 1 | Process-global epoch exhaustion | Addressed | — | Addressed; unused global declaration remains |
| 2 | Width sweep resource confound | Addressed | — | Addressed at source level |
| 3 | Unverified experiments in headline results | **Unaddressed**: `_split8read_ldcs` escaped | Code | Corrected |
| 4 | Executable CUDA coverage | **Unaddressed** | Code/test | Still unaddressed |
| 5 | Unapproved plan-item replacement | Addressed through the permitted documentation alternative | — | Addressed |
| 6 | Invalid H2D end-to-end rate | **Unaddressed**: NaN became JSON `null` | Code/schema | Corrected |
| 7 | Silently discarded offset-output flush error | **Unaddressed** | Code | Still unaddressed |
| 8 | Contradictory experiment manifest | **Unaddressed** | Documentation | Still unaddressed |
| 9 | Ambiguous serialized kernel identity | Addressed | — | Addressed |

Thus, the commit itself leaves must-fixes **3, 4, 6, 7, and 8** unaddressed. The review
corrections close **3** and **6**; the remaining working-tree gaps are **4** (code/test),
**7** (code), and **8** (documentation).

## Verification

- `git diff --check` — pass.
- `cargo test -p vortex-bench --lib` — pass, 59 tests, including all three FSST-12 tests.
- `cargo clippy -p vortex-bench --all-targets` — attempted. The first run could not fetch
  `onpair_cpp` in the sandbox; the network-enabled retry reached `vortex-bench` and failed
  on 17 pre-existing lints in the unchanged `ONPAIR_OFFSET_COST` block, beginning at
  `onpair_bench.rs:716`. None is introduced by the reviewed diff.
- CUDA build/runtime/differential checks — not run; `nvcc` is unavailable.
