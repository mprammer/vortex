// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

// E-A (2026-08-10): `split8read` instrumented to test Joe's 2026-08-05 claim that a
// warp's drain writes up to 16 bytes past its exclusive output region and clobbers
// the next warp's first byte.
//
// Byte-exactness alone cannot refute the claim: an overwrite that lands BEFORE the
// neighbour writes its own byte is repaired by the neighbour and is invisible in
// the output. This variant therefore checks, at the point of issue, that every
// store lies inside [out_start, out_start + warp_total), AND that consecutive
// chunk_offsets differ by exactly the total the warp drains. Those two together
// cover both ways adjacent regions could overlap: a store running past the end, or
// offsets that allot less than a warp writes.
//
// Identical signature to the shipped kernel, so it needs no host plumbing: a
// violation device-printf's the offending (site, chunk, offset, width, total) and
// trips an assert, surfacing as a CUDA error. A clean run over the corpus is the
// refutation; any trip is the reproduction.
//
// Instrumented, not shipped. The guards are on the hot path and this kernel's rate
// is NOT a decode measurement.
//
// OnPair decompress — 4 tokens/thread, split-read dictionary.
//
// Baseline `onpair_shmem_4tpt` is L1/TEX-cache-request bound on the per-token
// 16-byte `uint4` gather into the 64 KB padded dict, where the dict L1 hit rate
// is only ~31% (the 64 KB dict thrashes against the streaming codes/output).
//
// Most tokens are short (mean dict len ~6). This variant reads the common case
// from the **32 KB** `dict_s8` array (first 8 bytes/entry, `uint2`) and only
// touches the 64 KB `dict_padded` for the rare `len > 8` tokens. Halving the
// hot dict working set aims to raise the dict L1 hit rate, cutting L2 sectors
// and L1/TEX-request pressure. As a bonus, holding `uint2 lo[4]` (32 B) instead
// of `uint4 t[4]` (64 B) lowers register pressure.
//
// Identical scan/drain to `onpair_shmem_4tpt`; only the token-byte source
// changes.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 16u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_s8rb(uint32_t x, int lane) {
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


// A store at `out_start + rel` of `width` bytes stays inside this warp's exclusive
// region iff `rel + width <= warp_total`. Site ids: 1 head, 2 body, 3 tail.
__device__ inline void s8rb_check(uint64_t chunk, uint32_t rel, uint32_t width,
                                  uint32_t warp_total, int site) {
    if ((uint64_t)rel + (uint64_t)width > (uint64_t)warp_total) {
        printf("E-A VIOLATION site=%d chunk=%llu rel=%u width=%u warp_total=%u\n",
               site, (unsigned long long)chunk, rel, width, warp_total);
        assert(false && "drain store outside the warp's exclusive output region");
    }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_split8read_bounds(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_s8, const uint8_t *__restrict dict_padded,
    const uint8_t *__restrict lens, uint8_t *__restrict output_bytes,
    uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * 128u >= total_tokens) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

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

    uint32_t excl[4];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_s8rb(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc_base;

    const uint64_t out_start = chunk_offsets[chunk];
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
            // Rare path: high bytes from the full padded dict.
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

    // Second half of the test: the region this warp was ALLOTTED must equal the
    // total it is about to drain. If chunk_offsets allocate less than a warp
    // writes, every store can be in-range above and the regions still overlap.
    if (lane == 0 && (chunk + 1u) * 128u < total_tokens) {
        const uint64_t allotted = chunk_offsets[chunk + 1u] - out_start;
        if (allotted != (uint64_t)warp_total) {
            printf("E-A VIOLATION site=0 chunk=%llu allotted=%llu warp_total=%u\n",
                   (unsigned long long)chunk, (unsigned long long)allotted, warp_total);
            assert(false && "chunk_offsets stride disagrees with the drained total");
        }
    }

    const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
    if ((uint32_t)lane < head) {
        s8rb_check(chunk, (uint32_t)lane, 1u, warp_total, 1);
        output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
    }
    if (head >= warp_total) {
        return;
    }

    const uint32_t body_chunks = (warp_total - head) >> 4;
    for (uint32_t k = lane; k < body_chunks; k += 32u) {
        const uint32_t off = head + k * 16u;
        const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + off);
        s8rb_check(chunk, off, 16u, warp_total, 2);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), v);
    }

    const uint32_t tail_start = head + (body_chunks << 4);
    if ((uint32_t)lane < warp_total - tail_start) {
        s8rb_check(chunk, tail_start + (uint32_t)lane, 1u, warp_total, 3);
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] =
            s_buf[tail_start + lane];
    }
}
