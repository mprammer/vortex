# Experiment kernels, 2026-08-10 (branch `mp/onpair-expts-0810`)

Forked from the frozen harness `9b4714c2a`. The first four mirror the shipped
512-thread `split8read` operating point (`chunk_size` 128, `block_warps` 16); the
fifth has its own launch contract and is not registered.

| kernel | question | retirement |
|---|---|---|
| `onpair_shmem_4tpt_split8read_ldcs` | Does the `__ldcs` streaming hint on the read-once codes stream still have headroom once `split8read` has already halved the hot dict footprint? The existing `onpair_shmem_4tpt_ldcs` applies the hint to the stride-16 base, where it is worth ~1% in geomean over 167 committed cells and −5.6% on the L40S. | promote if it wins, else delete |
| `onpair_shmem_4tpt_split8read_hilo` | Does a disjoint stride-8 `dict_hi` (64 KB read working set vs 96 KB) beat the shipped kernel's duplicated low half? | promote if it wins, else delete |
| `onpair_shmem_4tpt_split8read_bounds` | Does any drain store leave the region the host allotted, `[chunk_offsets[c], chunk_offsets[c+1])`? Tests the 2026-08-05 overrun claim. | delete once the drain-bounds question is settled either way |
| `onpair_shmem_4tpt_split8read_bounds_faultinj` | Control. Identical but claims one byte more than it writes, so it MUST trap. | delete with the above |
| `onpair_shmem_4tpt_split8read_lookback` | "Bucket chain": can a batch's output position be produced DURING decode by decoupled look-back, instead of read from the sidecar or regenerated in a separate pass? This is the third option on the cursor decision and the one that threatens the paper's stored-vs-regenerated finding. **Registered 2026-08-11; never compiled, never run.** | warp-wide look-back done; promote after a byte-exact differential test, else delete kernel, layout arm, registry entry and the four `fused_*` scratch fields |

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

## The drain-bounds probe is BLOCKED (2026-08-10) — two review rounds killed two designs

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


---

## The look-back draft is REJECTED as written (gauntlet `cli-gauntlet-9e1f2c8dc41f`)

Fixed since the review:

- **Payload race (the real bug).** `part_value` was reused for both the `A`
  aggregate and the `P` inclusive prefix, so a successor could observe `A`, then
  read a value the owner had *since overwritten* with its prefix, and double-count
  every block before it. Split into two write-once arrays `part_agg` / `part_inc`,
  so the payload a flag refers to is immutable.
- **Memory model.** `volatile` + `__threadfence()` does not establish inter-block
  happens-before, and `__threadfence_block()` in the spin was the wrong scope
  entirely. Flags now use device-scope `st.release.gpu` / `ld.acquire.gpu`.
- **Launch contract** (block shape, descriptor capacity = `gridDim.x`, ticket and
  flag zeroing per launch, grid must cover the input) is now stated in the header.
  The kernel cannot check any of it.

Still outstanding, and why no number from this kernel means anything yet:

- No caller, no compilation evidence, no differential test against the shipped
  kernel, no scratch lifecycle implementation.
- The look-back is serial in one thread where CUB uses a warp ballot over 32
  predecessors. **A win despite that scaffolding would be informative; a loss would
  establish nothing** about whether an optimised fused look-back is competitive.
  Do not let a slow draft close this question.
- Timing must report kernel-only *and* reset-inclusive cost, since zeroing the
  descriptors is work the stored-offsets path never pays.


---

## Fused positioning, second review (2026-08-11)

Two rounds of adversarial review. Round one found the payload race; round two, after
the warp-wide rewrite, found a worse one:

- **The poll was not collective.** `__all_sync` with the full 32-lane mask sat inside
  the branch taken only by lanes whose predecessor index is >= 0, so lanes past the
  left end of the grid never reached it — undefined behaviour on every window that
  straddles the start. Restructured so all lanes reach the barrier, with off-end lanes
  trivially ready.
- **Epoch tags truncated to 30 bits while readers compared 32.** Any epoch past 2^30
  would have matched nothing and spun forever. Readers now compare the same 30 bits.
- **The host-computed ticket base was fragile** — wrong if several chunks shared a
  buffer, if grid widths differed, or if a launch failed with the counter advanced.
  Replaced by a ticket ring indexed by epoch: one slot per launch, never reused,
  zeroed once, so no launch resets anything and no base exists to get wrong.
- Descriptors were sized for a two-warp grid while the variant launches sixteen, an
  8x over-allocation. Right-sized, with the dispatch still bailing if a grid outgrows
  them.
- `lb_*` scratch names collided with `lb_meta`, which means *length bucket*. Renamed
  `fused_*`.

Verified by the reviewers, and unchanged: lane 0 is the closest predecessor, `__ffs`
selects the nearest published prefix, successive windows neither double-count nor omit
a block, every block that claims an id publishes before any early return, and the
decode and drain half is byte-equivalent to the shipped kernel once the output base is
right.

Outstanding, deliberately: the look-back waits for all 32 descriptors in a window
before consuming an already-visible closest prefix. That is head-of-line latency a
production implementation would not pay, so a **loss does not settle the question**.
