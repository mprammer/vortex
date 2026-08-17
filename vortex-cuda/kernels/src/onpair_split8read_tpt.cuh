// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Configurable-TPT form of the shipped split8read decoder. The matched
// comparison wrappers all use eight warps/block and launch_bounds(256, 4).

#ifndef TOKENS_PER_THREAD
#error "TOKENS_PER_THREAD must be defined"
#endif
#ifndef ONPAIR_KERNEL_NAME
#error "ONPAIR_KERNEL_NAME must be defined"
#endif
#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 8u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#endif

#define TOKENS_PER_WARP (TOKENS_PER_THREAD * 32u)
#define WARP_BUF_BYTES  (TOKENS_PER_WARP * 16u + 32u)

__device__ inline uint32_t warp_inclusive_scan_u32_split8(uint32_t value, int lane) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(mask, value, offset);
        if (lane >= offset) {
            value += other;
        }
    }
    return value;
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void ONPAIR_KERNEL_NAME(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_s8, const uint8_t *__restrict dict_padded,
    const uint8_t *__restrict lens, uint8_t *__restrict output_bytes,
    uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * TOKENS_PER_WARP >= total_tokens) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    const uint64_t base_i = chunk * TOKENS_PER_WARP + (uint64_t)lane;
    uint2 lo[TOKENS_PER_THREAD];
    uint32_t code[TOKENS_PER_THREAD];
    uint32_t len[TOKENS_PER_THREAD];
#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        if (i < total_tokens) {
            code[k] = (uint32_t)codes[i];
            lo[k] = *reinterpret_cast<const uint2 *>(dict_s8 + (size_t)code[k] * 8u);
            len[k] = (uint32_t)lens[code[k]];
        } else {
            code[k] = 0u;
            lo[k] = make_uint2(0u, 0u);
            len[k] = 0u;
        }
    }

    uint32_t excl[TOKENS_PER_THREAD];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_split8(len[k], lane);
        excl[k] = acc_base + incl - len[k];
        acc_base += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc_base;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
        const uint32_t token_len = len[k];
        if (token_len == 0u) {
            continue;
        }
        const uint32_t base = excl[k];
        const uint8_t *low_bytes = reinterpret_cast<const uint8_t *>(&lo[k]);
        const uint32_t low_len = token_len < 8u ? token_len : 8u;
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            if (byte < (int)low_len) {
                s_buf[base + (uint32_t)byte] = low_bytes[byte];
            }
        }
        if (token_len > 8u) {
            const uint2 high = *reinterpret_cast<const uint2 *>(
                dict_padded + (size_t)code[k] * 16u + 8u);
            const uint8_t *high_bytes = reinterpret_cast<const uint8_t *>(&high);
#pragma unroll
            for (int byte = 0; byte < 8; ++byte) {
                if (8 + byte < (int)token_len) {
                    s_buf[base + 8u + (uint32_t)byte] = high_bytes[byte];
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
    for (uint32_t k = (uint32_t)lane; k < body_chunks; k += 32u) {
        const uint32_t offset = head + k * 16u;
        const uint4 value = *reinterpret_cast<const uint4 *>(s_buf + offset);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + offset), value);
    }

    const uint32_t tail_start = head + (body_chunks << 4);
    if ((uint32_t)lane < warp_total - tail_start) {
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] =
            s_buf[tail_start + lane];
    }
}
