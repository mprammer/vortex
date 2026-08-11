// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// DRAFT (2026-08-10), NOT REVIEWED, NOT COMPILED, NOT CORRECTNESS-TESTED.
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
//     computation and the fixed 16-warp shared arrays all assume this).
//   - part_agg, part_inc and part_flag each hold at least gridDim.x entries. They
//     are indexed by the DYNAMIC block id, which ranges over gridDim.x, NOT by a
//     chunk count. A short buffer corrupts memory.
//   - ticket == 0 and every used part_flag entry == LB_X before EVERY launch. A
//     retained ticket skips work, indexes out of bounds, or waits forever on a
//     descriptor no live block will publish. Scratch must be launch-private, or
//     synchronised across streams.
//   - The grid must cover total_tokens exactly: there is no grid-stride loop, so an
//     undersized grid silently drops input.
//
// BLOCKERS BEFORE ANY NUMBER FROM THIS KERNEL MEANS ANYTHING (gauntlet
// cli-gauntlet-9e1f2c8dc41f, verdict reject — these are NOT cosmetic):
//   - Never compiled, never run, no differential test against the shipped kernel,
//     no caller. build.rs compiles every .cu in this directory, so its presence
//     here is not evidence that it works.
//   - The look-back is serial in ONE thread; CUB inspects 32 predecessors per step
//     with a warp ballot. This draft's RATE is therefore not the achievable rate.
//     A WIN despite this scaffolding would be informative; a LOSS would establish
//     nothing about whether an optimised fused look-back is competitive. That
//     asymmetry is the whole reason to be careful with this kernel's numbers.
//   - Timing must report both kernel-only and reset-inclusive cost, since zeroing
//     the descriptors is real work the stored-offsets path does not pay.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 16u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif
#define WARP_BUF_BYTES 2080u

// Partition-descriptor flags.
#define LB_X 0u  // invalid: this block has published nothing yet
#define LB_A 1u  // aggregate available: this block's own total, prefix unknown
#define LB_P 2u  // inclusive prefix available: everything up to and including it

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
    uint64_t *__restrict part_inc, uint32_t *__restrict part_flag) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t warps = blockDim.x >> 5;

    // ---- 1. claim a dynamic block id -------------------------------------------
    __shared__ uint32_t s_blk;
    __shared__ uint64_t s_warp_tot[WARPS_PER_BLOCK_MAX];
    __shared__ uint64_t s_block_excl;
    if (threadIdx.x == 0) {
        s_blk = atomicAdd(ticket, 1u);
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

    // ---- 4. decoupled look-back, performed by thread 0 -------------------------
    if (threadIdx.x == 0) {
        uint64_t block_total = 0;
        for (uint32_t w = 0; w < warps; ++w) {
            block_total += s_warp_tot[w];
        }

        uint64_t exclusive = 0;
        if (blk == 0u) {
            // Head: aggregate IS the inclusive prefix. Both payloads written before
            // the release store, so any observer that sees the flag sees the values.
            part_agg[0] = block_total;
            part_inc[0] = block_total;
            lb_store_flag_release(&part_flag[0], LB_P);
        } else {
            part_agg[blk] = block_total;
            lb_store_flag_release(&part_flag[blk], LB_A);

            uint32_t i = blk - 1u;
            for (;;) {
                uint32_t f = lb_load_flag_acquire(&part_flag[i]);
                while (f == LB_X) {
                    f = lb_load_flag_acquire(&part_flag[i]);
                }
                // Each payload is written exactly once and never mutated, so the
                // flag we observed cannot go stale against the value we read.
                if (f == LB_P) {
                    exclusive += part_inc[i];
                    break;
                }
                exclusive += part_agg[i];
                if (i == 0u) {
                    break;
                }
                --i;
            }
            part_inc[blk] = exclusive + block_total;
            lb_store_flag_release(&part_flag[blk], LB_P);
        }
        s_block_excl = exclusive;
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
    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
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
