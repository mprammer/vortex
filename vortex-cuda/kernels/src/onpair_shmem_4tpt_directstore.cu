// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// Byte-exact drain counterfactual for `onpair_shmem_4tpt`.
//
// This retains the production kernel's 128-token warp chunks, four tokens per
// lane, stride-16 dictionary gather, and four warp prefix scans. Unlike the
// production kernel, it skips shared-memory staging and writes each token's
// true-length byte range directly to its prefix-sum-derived output offset.
// The deliberately narrow, generally uncoalesced global stores isolate the
// benefit of the staged aligned drain.

#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif

__device__ inline uint32_t warp_inclusive_scan_u32_4tpt_directstore(uint32_t x, int lane) {
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

__device__ inline void emit_token_directstore(uint8_t *output_bytes, uint64_t base,
                                              const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            output_bytes[base + (uint64_t)j] = tb[j];
        }
    }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_directstore(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * 128u >= total_tokens) {
        return;
    }

    // Lane handles tokens i_k = chunk*128 + lane + k*32 for k in 0..4.
    // Layout: all 32 lanes' i0 first (32 tokens), then all i1 (32 tokens),
    // then i2, i3 — keeps in-warp scan/excl arithmetic linear in lane id.
    const uint64_t base_i = chunk * 128u + (uint64_t)lane;
    uint4 t[4];
    uint32_t l[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        const bool active = (i < total_tokens);
        if (active) {
            const uint32_t c = (uint32_t)codes[i];
            t[k] = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)c * 16u);
            l[k] = (uint32_t)lens[c];
        } else {
            t[k] = make_uint4(0u, 0u, 0u, 0u);
            l[k] = 0u;
        }
    }

    uint32_t excl[4];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_4tpt_directstore(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }

    const uint64_t out_start = chunk_offsets[chunk];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (l[k] > 0u) {
            emit_token_directstore(output_bytes, out_start + (uint64_t)excl[k], t[k], l[k]);
        }
    }
}
