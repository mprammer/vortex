// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// EXPERIMENT (2026-08-11). Reviewed twice, registered, NOT YET COMPILED OR RUN.
//
// "Bucket chain" / single-pass decode: obtain each batch's output position DURING
// the decode instead of reading it from the stored offset sidecar or regenerating
// it in a separate pass.
//
// Why this matters to the paper, not just to the kernel: Section "Updating the
// Cursor" measures storing positions against regenerating them, and finds storing
// wins by 14-19%. That 14-19% is against regeneration as a SEPARATE PASS. This
// kernel is the third option — regenerate, fused into the decode — which pays
// neither the sidecar's storage nor a second pass over the codes. If it lands near
// memcpy, the stored-vs-regenerated finding inverts and the write-time-metadata
// contribution loses its measured justification. That is the reviewer question
// this exists to answer.
//
// THE ALGORITHM IS NOT A NAIVE CHAIN. Joe's 2026-08-06 sketch — warp N spins until
// warp N-1 publishes its total — serializes and can deadlock: a 1 GB column is
// ~1.4M chunks against O(1000) resident warps, so most predecessors are not
// scheduled when their successors start waiting. The standard fix is Merrill &
// Garland's DECOUPLED LOOK-BACK (what CUB's DeviceScan uses, and what this paper
// already cites for the warp scan):
//
//   1. A block claims a DYNAMIC id by atomicAdd on a ticket counter, so a block
//      holding id N is guaranteed to have arrived before any block holding id > N.
//      Predecessors therefore always exist and make progress; blockIdx ordering is
//      NOT relied on, because the hardware gives no such guarantee.
//   2. The block publishes its own aggregate with flag A, then walks backwards:
//      an A neighbour contributes its aggregate and the walk continues; a P
//      neighbour contributes an inclusive prefix and the walk STOPS. This is what
//      decouples it — a block never waits for the whole chain, only until it meets
//      someone who already knows their prefix.
//   3. Having its exclusive prefix, the block publishes its own inclusive prefix
//      with flag P, unblocking its successors.
//
// EACH CODE IS READ EXACTLY ONCE. c[4], l[4] and lo[4] are loaded into registers
// before the look-back and are still live afterwards for the emit. That is the
// whole point of fusing: a separate offsets pass reads the code stream twice.
//
// Kernel signature differs from every other variant: there is NO chunk_offsets
// argument. The three scratch buffers replace it.
//
// LAUNCH CONTRACT — the caller MUST honour all of it; the kernel cannot check it:
//   - 1-D block, threads a multiple of 32, 32..512 (the warp mask, the `warps`
//     computation and the per-warp totals array assume this).
//   - dynamic shared memory of exactly `warps * WARP_BUF_BYTES` bytes.
//   - NOTE for the width sweep: __launch_bounds__(512, 2) is compile-time and therefore
//     identical at every width. Register allocation is tuned for 512 threads, so narrow
//     widths are not independently tuned. The sweep is an end-to-end geometry
//     comparison, NOT an isolated measurement of the block-wide stall.
//   - part_agg, part_inc and part_flag each hold at least gridDim.x entries. They
//     are indexed by the DYNAMIC block id, which ranges over gridDim.x, NOT by a
//     chunk count. A short buffer corrupts memory.
//   - `ticket` an array of at least LB_TICKET_SLOTS u32, zeroed once at allocation.
//     Each launch consumes slot `epoch % LB_TICKET_SLOTS`, so epochs must not wrap
//     the ring while an earlier launch's slot is still live. Nothing is reset.
//   - `epoch` STRICTLY INCREASING across launches that share the descriptor arrays,
//     and never reused. Descriptors do NOT need clearing: a flag whose tagged epoch
//     differs from the current one reads as LB_X. The arrays need zeroing exactly
//     once, at allocation, so that epoch 0 sees no false positives.
//   - Scratch must be launch-private, or the epoch must be advanced under the same
//     synchronisation that orders the launches.
//   - The grid must cover total_tokens exactly: there is no grid-stride loop, so an
//     undersized grid silently drops input.
//
// BEFORE A NUMBER FROM THIS KERNEL MEANS ANYTHING:
//   - Never compiled, never run. The byte-exact check against the CPU reference
//     (--gpu-validate) is the differential test and has not been performed.
//   - The whole BLOCK stalls on warp 0's look-back while the other warps wait at a
//     barrier. This cannot be removed by emitting earlier: the scratch base is
//     shifted by the output base's alignment precisely so both the shared read and
//     the global store in the drain body can be 16-byte wide, and an unaligned uint4
//     shared load is undefined. Quantified instead by sweeping block width; see the
//     _lookback_w{1,2,4} variants.
//   - Timing is warmed kernel-only. Nothing is reset per launch, so it hides no
//     clearing pass — but it also excludes one-time scratch allocation, and it
//     excludes what the stored-offsets path pays OUTSIDE decode (building the
//     sidecar at write time, storing it, reading it back). A fair answer to
//     "should we store positions" needs both sides' out-of-kernel costs stated.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 16u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif
#define WARP_BUF_BYTES 2080u

// Partition-descriptor states, stored in the low 2 bits of a flag word whose high
// 30 bits carry the LAUNCH EPOCH. A flag whose epoch differs from the current one is
// a leftover from a previous launch and reads as LB_X. That removes the requirement
// to clear gridDim.x descriptors before every launch, which would otherwise be real
// work the stored-offsets path never pays and would confound the comparison. Ticket
// counters are likewise zeroed once; the host assigns a fresh ring slot per launch.
#define LB_X 0u  // nothing published yet (or stale epoch)
#define LB_A 1u  // aggregate available: this block's own total, prefix unknown
#define LB_P 2u  // inclusive prefix available: everything up to and including it
// Ticket ring depth. Power of two; one slot per launch, never reused. MUST equal
// FUSED_TICKET_SLOTS in vortex-bench/src/onpair_bench.rs; the host asserts agreement at
// startup because a silent mismatch would alias live slots.
#define LB_TICKET_SLOTS 16384u
static_assert((LB_TICKET_SLOTS & (LB_TICKET_SLOTS - 1u)) == 0u,
              "ticket ring must be a power of two: the kernel indexes it with a mask");
#define LB_EPOCH_MASK 0x3fffffffu
#define LB_TAG(epoch, st) ((((epoch) & LB_EPOCH_MASK) << 2) | (st))
#define LB_EPOCH_OF(w) ((w) >> 2)
#define LB_STATE_OF(w) ((w) & 3u)

// Device-scope release store / acquire load. volatile + __threadfence() does NOT
// establish inter-block happens-before, and __threadfence_block() is the wrong
// scope entirely: a successor can consume a flag without its payload.
__device__ __forceinline__ void lb_store_flag_release(uint32_t *p, uint32_t v) {
    asm volatile("st.release.gpu.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t lb_load_flag_acquire(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ inline uint32_t warp_inclusive_scan_u32_lb(uint32_t x, int lane) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t y = __shfl_up_sync(mask, x, offset);
        if (lane >= offset) {
            x += y;
        }
    }
    return x;
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_split8read_lookback(
    const uint16_t *__restrict codes, const uint8_t *__restrict dict_s8,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens,
    uint32_t *__restrict ticket, uint64_t *__restrict part_agg,
    uint64_t *__restrict part_inc, uint32_t *__restrict part_flag,
    uint32_t epoch) {
    // Compare tags against the same 30 bits LB_TAG stores, or an epoch past 2^30
    // would match nothing and spin forever.
    const uint32_t epoch_tag = epoch & LB_EPOCH_MASK;
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t warps = blockDim.x >> 5;

    // ---- 1. claim a dynamic block id -------------------------------------------
    __shared__ uint32_t s_blk;
    // Bounded by the launch contract: warps <= WARPS_PER_BLOCK_MAX. Small and fixed, so
    // it does not distort the width sweep the way the staging buffer did.
    __shared__ uint64_t s_warp_tot[WARPS_PER_BLOCK_MAX];
    __shared__ uint64_t s_block_excl;
    if (threadIdx.x == 0) {
        // Each launch gets its OWN ticket slot, indexed by epoch, from a ring
        // zeroed once at allocation. So no slot is ever reused, nothing needs
        // resetting, and a base is unnecessary — which also removes the ways a
        // host-computed base could be wrong (several chunks sharing a buffer,
        // unequal grid widths, a failed launch leaving the counter advanced).
        s_blk = atomicAdd(&ticket[epoch & (LB_TICKET_SLOTS - 1u)], 1u);
    }
    __syncthreads();
    const uint32_t blk = s_blk;

    // ---- 2. load this lane's four (code, len) pairs into REGISTERS -------------
    // These stay live through the look-back so the code stream is read once.
    const uint64_t chunk = (uint64_t)blk * (uint64_t)warps + (uint64_t)warp_id;
    const uint64_t base_i = chunk * 128u + (uint64_t)lane;
    uint2 lo[4];
    uint32_t c[4];
    uint32_t l[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        if (i < total_tokens) {
            const uint32_t code = (uint32_t)codes[i];
            c[k] = code;
            lo[k] = *reinterpret_cast<const uint2 *>(dict_s8 + (size_t)code * 8u);
            l[k] = (uint32_t)lens[code];
        } else {
            c[k] = 0u;
            lo[k] = make_uint2(0u, 0u);
            l[k] = 0u;
        }
    }

    // ---- 3. intra-warp scan, then intra-block totals ---------------------------
    uint32_t excl[4];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_lb(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc_base;
    if (lane == 0) {
        s_warp_tot[warp_id] = (uint64_t)warp_total;
    }
    __syncthreads();

    // ---- 4. decoupled look-back, warp-wide -------------------------------------
    // Warp 0 inspects 32 predecessors per step, as CUB does. A serial walk would
    // make a SLOW result uninterpretable: it could not distinguish "fused
    // positioning is uncompetitive" from "this look-back is a toy".
    if (warp_id == 0u) {
        uint64_t block_total = 0;
        for (uint32_t w = 0; w < warps; ++w) {
            block_total += s_warp_tot[w];
        }

        uint64_t exclusive = 0;
        if (blk == 0u) {
            if (lane == 0) {
                // Both payloads are written before the release store, so any
                // observer that sees the flag also sees the values.
                part_agg[0] = block_total;
                part_inc[0] = block_total;
                lb_store_flag_release(&part_flag[0], LB_TAG(epoch_tag, LB_P));
            }
        } else {
            if (lane == 0) {
                part_agg[blk] = block_total;
                lb_store_flag_release(&part_flag[blk], LB_TAG(epoch_tag, LB_A));
            }
            __syncwarp();

            // Window of 32 predecessors, closest first: lane 0 inspects blk-1.
            int64_t window_end = (int64_t)blk;  // exclusive
            for (;;) {
                const int64_t idx = window_end - 1 - (int64_t)lane;
                const bool off_end = (idx < 0);
                uint32_t st = LB_P;  // off-end lanes hold a known empty prefix
                uint64_t v = 0;
                if (!off_end) {
                    const uint32_t w = lb_load_flag_acquire(&part_flag[idx]);
                    st = (LB_EPOCH_OF(w) == epoch_tag) ? LB_STATE_OF(w) : LB_X;
                    if (st != LB_X) {
                        v = (st == LB_P) ? part_inc[idx] : part_agg[idx];
                    }
                }

                // Ballots are collective: every lane named in `mask` reaches both,
                // including off-end lanes. Lane 0 is the nearest predecessor, so
                // __ffs gives the nearest lane in each class.
                const unsigned xmask = __ballot_sync(mask, st == LB_X);
                const unsigned pmask = __ballot_sync(mask, st == LB_P);
                const int firstX = (xmask == 0u) ? 32 : (__ffs((int)xmask) - 1);
                const int firstP = (pmask == 0u) ? 32 : (__ffs((int)pmask) - 1);

                // Only descriptors NEARER than the first unpublished one are usable
                // this iteration. Consuming a visible prefix the moment it is nearer
                // than any hole is the point: waiting for all 32 descriptors before
                // using an already-visible prefix is head-of-line latency.
                const int usable = (firstP < firstX) ? firstP : (firstX - 1);
                uint64_t contrib = ((int)lane <= usable) ? v : 0;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    contrib += __shfl_down_sync(mask, contrib, off);
                }
                contrib = __shfl_sync(mask, contrib, 0);
                exclusive += contrib;

                if (firstP < firstX) {
                    break;  // consumed an inclusive prefix: the walk is done
                }
                // Otherwise everything usable was an aggregate. Slide past exactly
                // what we consumed and re-read; a hole becomes lane 0 next time,
                // which is a spin on that one descriptor rather than on all 32.
                window_end -= (firstX == 32) ? 32 : firstX;
            }

            if (lane == 0) {
                part_inc[blk] = exclusive + block_total;
                lb_store_flag_release(&part_flag[blk], LB_TAG(epoch_tag, LB_P));
            }
        }
        if (lane == 0) {
            s_block_excl = exclusive;
        }
    }
    __syncthreads();

    // ---- 5. this warp's output base = block prefix + intra-block warp prefix ----
    uint64_t out_start = s_block_excl;
    for (uint32_t w = 0; w < warp_id; ++w) {
        out_start += s_warp_tot[w];
    }
    if (chunk * 128u >= total_tokens) {
        return;
    }

    // ---- 6. from here identical to the shipped kernel ---------------------------
    // Dynamic, so a narrow block reserves only what it uses. With a fixed
    // WARPS_PER_BLOCK_MAX array every width reserved ~33 KiB and the width sweep moved
    // occupancy for reasons unrelated to the stall it is supposed to measure. The host
    // passes warps * WARP_BUF_BYTES.
    extern __shared__ __align__(16) uint8_t s_buf_all[];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t len = l[k];
        if (len == 0u) {
            continue;
        }
        const uint32_t base = excl[k];
        const uint8_t *lob = reinterpret_cast<const uint8_t *>(&lo[k]);
        const uint32_t nlo = len < 8u ? len : 8u;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < (int)nlo) {
                s_buf[base + j] = lob[j];
            }
        }
        if (len > 8u) {
            const uint2 hi =
                *reinterpret_cast<const uint2 *>(dict_padded + (size_t)c[k] * 16u + 8u);
            const uint8_t *hib = reinterpret_cast<const uint8_t *>(&hi);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (8 + j < (int)len) {
                    s_buf[base + 8 + j] = hib[j];
                }
            }
        }
    }
    __syncwarp();

    const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
    if ((uint32_t)lane < head) {
        output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
    }
    if (head >= warp_total) {
        return;
    }
    const uint32_t body_chunks = (warp_total - head) >> 4;
    for (uint32_t k = lane; k < body_chunks; k += 32u) {
        const uint32_t off = head + k * 16u;
        const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + off);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), v);
    }
    const uint32_t tail_start = head + (body_chunks << 4);
    if ((uint32_t)lane < warp_total - tail_start) {
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] =
            s_buf[tail_start + lane];
    }
}
