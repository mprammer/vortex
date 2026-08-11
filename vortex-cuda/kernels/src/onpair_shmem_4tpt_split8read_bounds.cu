// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

// Drain bounds probe (2026-08-10): `split8read` instrumented to test Joe's 2026-08-05 claim that a
// warp's drain writes past its exclusive output region and clobbers the next
// warp's first byte.
//
// Byte-exactness cannot refute the claim on its own: an overwrite landing BEFORE
// the neighbour writes its own byte is repaired by the neighbour and is invisible
// in the output.
//
// The check must use a bound INDEPENDENT of the drain. Every store predicate here
// is derived from `warp_total`, so checking a store against `warp_total` is a
// tautology that can never fire, and a clean run would prove nothing. Instead each
// store is checked, in absolute output coordinates, against the region the host
// allotted: [chunk_offsets[chunk], chunk_offsets[chunk+1]). The allotment itself is
// separately checked to equal the total this warp drains, which covers the other
// way regions could overlap: offsets that allot less than a warp writes.
//
// Same signature as the shipped kernel, so no host plumbing. A clean run over the
// corpus is the refutation; a trap is the reproduction. The companion
// `..._bounds_faultinj` kernel is identical but deliberately overruns by one byte,
// and MUST trap: it is the control proving this instrument can fail.
//
// Instrumented, not shipped. The guards are on the hot path; this kernel's rate is
// NOT a decode measurement.
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

__device__ inline uint32_t warp_inclusive_scan_u32_8read_bounds(uint32_t x, int lane) {
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


// The store predicates in this kernel are all derived from `warp_total`, so
// comparing a store back against `warp_total` is a tautology and can never fire.
// The independent bound is the region the HOST allotted this warp:
// [chunk_offsets[chunk], chunk_offsets[chunk+1]). Every store is checked against
// that, in absolute output coordinates. `__trap()` rather than `assert()` so the
// check survives -DNDEBUG. Site ids: 0 allotment, 1 head, 2 body, 3 tail.
__device__ inline void s8rb_check(uint64_t chunk, uint64_t addr, uint32_t width,
                                  uint64_t lo, uint64_t hi, int site) {
    if (addr < lo || addr + (uint64_t)width > hi) {
        printf("DRAIN-BOUNDS VIOLATION site=%d chunk=%llu addr=%llu width=%u region=[%llu,%llu)\n",
               site, (unsigned long long)chunk, (unsigned long long)addr, width,
               (unsigned long long)lo, (unsigned long long)hi);
        __trap();
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
        const uint32_t incl = warp_inclusive_scan_u32_8read_bounds(l[k], lane);
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

    // chunk_offsets has total_chunks+1 entries (terminal sentinel), so
    // chunk_offsets[chunk+1] is valid for EVERY active chunk including the last.
    const uint64_t region_end = chunk_offsets[chunk + 1u];
    if (lane == 0 && region_end - out_start != (uint64_t)warp_total) {
        printf("DRAIN-BOUNDS VIOLATION site=0 chunk=%llu allotted=%llu warp_total=%u\n",
               (unsigned long long)chunk,
               (unsigned long long)(region_end - out_start), warp_total);
        __trap();
    }

    const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
    if ((uint32_t)lane < head) {
        s8rb_check(chunk, out_start + (uint64_t)lane, 1u, out_start, region_end, 1);
        output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
    }
    if (head >= warp_total) {
        return;
    }

    const uint32_t body_chunks = (warp_total - head) >> 4;
    for (uint32_t k = lane; k < body_chunks; k += 32u) {
        const uint32_t off = head + k * 16u;
        const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + off);
        s8rb_check(chunk, out_start + (uint64_t)off, 16u, out_start, region_end, 2);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), v);
    }

    const uint32_t tail_start = head + (body_chunks << 4);
    if ((uint32_t)lane < warp_total - tail_start) {
        s8rb_check(chunk, out_start + (uint64_t)tail_start + (uint64_t)lane, 1u,
                   out_start, region_end, 3);
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] =
            s_buf[tail_start + lane];
    }
}
