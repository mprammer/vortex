// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// E-B (2026-08-10): `split8read` with a TRUE disjoint hi/lo dictionary split.
//
// The shipped kernel reads the low eight bytes from the stride-8 `dict_s8` and,
// for `len > 8`, the high eight from `dict_padded + code*16 + 8` — a stride-16
// table whose low half duplicates `dict_s8`. Device footprint is therefore
// 32 KB + 64 KB = 96 KB at 4,096 entries. This variant reads the high half from a
// dedicated stride-8 `dict_hi` (bytes 8..16 of each entry), so the two tables are
// disjoint and total 64 KB. L1 capacity is exactly the resource `split8read`
// buys, so removing the redundancy should compound the mechanism.
//
// Counter-hypothesis on record: the high path is exercised so rarely that most of
// `dict_padded` never becomes L1-resident, so this may recover nothing. A null
// result is the publishable answer.
//
// Byte-exact with the shipped kernel by construction: same bytes, different
// address arithmetic on the rare path only.
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

__device__ inline uint32_t warp_inclusive_scan_u32_s8rhilo(uint32_t x, int lane) {
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

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_split8read_hilo(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_s8, const uint8_t *__restrict dict_hi,
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
        const uint32_t incl = warp_inclusive_scan_u32_s8rhilo(l[k], lane);
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
            // Rare path: high bytes from the disjoint stride-8 high table.
            const uint2 hi =
                *reinterpret_cast<const uint2 *>(dict_hi + (size_t)c[k] * 8u);
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
