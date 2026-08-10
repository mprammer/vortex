# Experiment kernels, 2026-08-10 (branch `mp/onpair-expts-0810`)

Forked from the frozen harness `9b4714c2a`. All four mirror the shipped
512-thread `split8read` operating point (`chunk_size` 128, `block_warps` 16).

| kernel | question | retirement |
|---|---|---|
| `onpair_shmem_4tpt_split8read_ldcs` | Does the `__ldcs` streaming hint on the read-once codes stream still have headroom once `split8read` has already halved the hot dict footprint? The existing `onpair_shmem_4tpt_ldcs` applies the hint to the stride-16 base, where it is worth ~1% in geomean over 167 committed cells and −5.6% on the L40S. | promote if it wins, else delete |
| `onpair_shmem_4tpt_split8read_hilo` | Does a disjoint stride-8 `dict_hi` (64 KB read working set vs 96 KB) beat the shipped kernel's duplicated low half? | promote if it wins, else delete |
| `onpair_shmem_4tpt_split8read_bounds` | Does any drain store leave the region the host allotted, `[chunk_offsets[c], chunk_offsets[c+1])`? Tests the 2026-08-05 overrun claim. | delete once E-A is settled either way |
| `onpair_shmem_4tpt_split8read_bounds_faultinj` | Control. Identical but claims one byte more than it writes, so it MUST trap. | delete with the above |

## Why the bounds probe checks what it checks

Every store predicate in the drain is derived from `warp_total`, so checking a
store against `warp_total` is a tautology that can never fire — a clean run would
prove nothing. The probe therefore checks each store, in absolute output
coordinates, against the **independently supplied** region
`[chunk_offsets[chunk], chunk_offsets[chunk+1])`, and separately checks that the
allotment equals the total the warp drains. `chunk_offsets` carries a terminal
sentinel (`total_chunks + 1` entries), so `chunk+1` is valid for every active
chunk including the last.

`__trap()` rather than `assert()`, so the check survives `-DNDEBUG`.

**Reading the result requires both kernels.** `_bounds` clean AND `_bounds_faultinj`
trapping is a refutation. `_bounds` clean AND `_faultinj` also clean means the
instrument is broken and the run is void.

## Known gaps (accepted, not fixed)

- **No automated byte-exact regression suite** over the boundary cases (token
  counts 1/127/128/129/256, dict lengths 0/8/9/16, partial terminal chunks). The
  harness's `--gpu-validate` byte-exact check against the CPU reference is the only
  coverage. Adequate for an experiment, not for promotion to the shipped set.
- **The variants are near-complete forks** of the shipped kernel rather than a
  shared body with load/diagnostic hooks, so a later fix to the scan, staging, or
  drain can silently diverge. Accepted deliberately: factoring the shipped kernel
  is a change to the thing being measured, which is exactly what an experiment
  branch must not do.

---

## E-A is BLOCKED (2026-08-10) — two review rounds killed two designs

`--depth max` gauntlet, runs `cli-gauntlet-0fd3f381815c` then `cli-gauntlet-b0b7042a27d4`
(verdict: reject). Both probe kernels are **deregistered** from `onpair_bench.rs`; the
sources remain for the redesign.

**Round 1.** Checking each store against `warp_total` was a tautology: every store
predicate is *derived* from `warp_total`, so `rel + width <= warp_total` holds by
construction at all three sites. A clean run would have "refuted" the claim while
proving nothing.

**Round 2.** Switching the bound to `[chunk_offsets[c], chunk_offsets[c+1])` did not
fix it, because the probe also traps unless `region_end - out_start == warp_total`.
Any execution that survives that check has made the two quantities equal, so the
per-store checks reduce to the round-1 tautology again.

The general lesson: **inside the kernel, the drain's write extent and the offsets'
allotment are the same quantity computed the same way.** No in-kernel predicate over
them can be independent. The overrun Joe describes could only arise from the offsets
disagreeing with the lengths, which is a host-side data property, not a kernel one.

Two further defects found in round 2, both real:

- **The fault-injection control passes vacuously.** The widened check sits in the
  tail branch, so inputs with no tail (128 one-byte tokens; a single 16-byte token)
  complete without trapping. A control that can silently not fire is worse than none.
- **`__trap()` poisons the process-wide CUDA context.** The harness shares one
  context across all kernels, so an expected trap aborts the benchmark before results
  are written *and* compromises every later kernel in the run. This is why the probes
  are deregistered rather than merely left unused: including them in a sweep risks the
  whole run's results, not just their own.

### Redesign specification

Report through a buffer, never a trap, and verify on the host:

1. Kernel records, per chunk, the minimum and maximum absolute output address it
   actually wrote, into a device array (`uint64 [2 * n_chunks]`). No predicates, no
   traps, no early exits.
2. Host checks, against its own `chunk_offsets`: (a) each chunk's recorded interval
   equals `[chunk_offsets[c], chunk_offsets[c+1])`, and (b) the intervals are pairwise
   disjoint and cover the output exactly once.

This is genuinely independent because the comparison happens outside the kernel
against data the kernel never reads, and it makes the control trivial: perturb one
recorded bound and the host check must fail.

Cost: one extra buffer argument and its `KernelLayout`, plus a host verification pass.
Also worth checking the offsets table itself against the output allocation in release
builds; today that invariant is only a `debug_assert_eq!` in `chunk_offsets()`.
