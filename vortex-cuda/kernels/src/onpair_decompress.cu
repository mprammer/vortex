// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Universal OnPair decoder for the original u16 code stream and a CPU-flattened
// dictionary. Each warp owns 128 tokens. The dictionary is split into dense
// 8-byte low and high planes, and two (length - 1) values share each byte of
// `packed_lens`. Short entries only read the low plane; long entries request a
// high-plane load through a dense per-warp queue. The first high-half load
// overlaps the independent low-byte shared-memory stores.

#define WARPS_PER_BLOCK_MAX 8u
#define WARP_BUF_BYTES      2080u
#define REQUESTS_PER_WARP   128u

__device__ inline uint64_t warp_scan_four_u16_prefixes(uint64_t x, int lane) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint64_t y = __shfl_up_sync(mask, x, offset);
        if (lane >= offset) {
            x += y;
        }
    }
    return x;
}

// OnPair token lengths are in [1, 16]. Encoding length - 1 lets all values,
// including 16, fit in four bits. Even codes occupy the low nibble.
__device__ inline uint32_t unpack_length(const uint8_t *__restrict packed_lens, uint32_t code) {
    const uint32_t packed = (uint32_t)packed_lens[code >> 1u];
    const uint32_t shift = (code & 1u) << 2u;
    return ((packed >> shift) & 0xfu) + 1u;
}

// Layout: code[15:0], shared destination[27:16], high_length-1[30:28].
__device__ inline uint32_t pack_high_request(uint32_t code, uint32_t destination, uint32_t high_length) {
    return code | (destination << 16u) | ((high_length - 1u) << 28u);
}

__device__ inline void
emit_high_bytes(uint8_t *s_buf, uint32_t destination, uint32_t high_length, const uint2 &high) {
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&high);
#pragma unroll
    for (int byte = 0; byte < 8; ++byte) {
        if (byte < (int)high_length) {
            s_buf[destination + (uint32_t)byte] = bytes[byte];
        }
    }
}

extern "C" __global__ __launch_bounds__(256,
                                        4) void onpair_decompress(const uint16_t *__restrict codes,
                                                                  const uint64_t *__restrict chunk_offsets,
                                                                  const uint8_t *__restrict dict_s8_lo,
                                                                  const uint8_t *__restrict dict_s8_hi,
                                                                  const uint8_t *__restrict packed_lens,
                                                                  uint8_t *__restrict output_bytes,
                                                                  uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk = (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * 128u >= total_tokens) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    __shared__ __align__(16) uint32_t s_requests[WARPS_PER_BLOCK_MAX][REQUESTS_PER_WARP];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];
    uint32_t *requests = s_requests[warp_id];

    const uint64_t base_i = chunk * 128u + (uint64_t)lane;
    uint2 lo[4];
    uint32_t code[4];
    uint32_t len[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        if (i < total_tokens) {
            code[k] = (uint32_t)codes[i];
            lo[k] = *reinterpret_cast<const uint2 *>(dict_s8_lo + (size_t)code[k] * 8u);
            len[k] = unpack_length(packed_lens, code[k]);
        } else {
            code[k] = 0u;
            lo[k] = make_uint2(0u, 0u);
            len[k] = 0u;
        }
    }

    constexpr uint64_t field_mask = 0xffffull;
    static_assert(32u * 16u <= field_mask, "packed fields must hold a full plane prefix");
    uint64_t packed =
        (uint64_t)len[0] | ((uint64_t)len[1] << 16u) | ((uint64_t)len[2] << 32u) | ((uint64_t)len[3] << 48u);
    packed = warp_scan_four_u16_prefixes(packed, lane);
    const uint64_t packed_totals = __shfl_sync(mask, packed, 31);

    uint32_t excl[4];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t shift = (uint32_t)k * 16u;
        const uint32_t incl = (uint32_t)((packed >> shift) & field_mask);
        const uint32_t plane_total = (uint32_t)((packed_totals >> shift) & field_mask);
        excl[k] = acc_base + incl - len[k];
        acc_base += plane_total;
    }
    const uint32_t warp_total = acc_base;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

    // Build the identical plane-major request stream first, so dense lane N
    // still owns request N and the first high gather can be issued early.
    uint32_t high_count = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const bool needs_high = len[k] > 8u;
        const uint32_t needs_mask = __ballot_sync(mask, needs_high);
        const uint32_t lower_lanes = lane == 0 ? 0u : ((1u << (uint32_t)lane) - 1u);
        const uint32_t rank = __popc(needs_mask & lower_lanes);
        if (needs_high) {
            requests[high_count + rank] = pack_high_request(code[k], excl[k] + 8u, len[k] - 8u);
        }
        high_count += __popc(needs_mask);
    }
    __syncwarp();

    const bool first_active = (uint32_t)lane < high_count;
    uint32_t first_destination = 0u;
    uint32_t first_high_length = 0u;
    uint2 first_high = make_uint2(0u, 0u);
    if (first_active) {
        const uint32_t request = requests[lane];
        const uint32_t selected_code = request & 0xffffu;
        first_destination = (request >> 16u) & 0xfffu;
        first_high_length = (request >> 28u) + 1u;
        first_high = *reinterpret_cast<const uint2 *>(dict_s8_hi + (size_t)selected_code * 8u);
    }

    // The first high value is deliberately not consumed until all owners have
    // emitted their low bytes, creating independent instructions after LDG.
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t low_length = len[k] < 8u ? len[k] : 8u;
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&lo[k]);
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            if (byte < (int)low_length) {
                s_buf[excl[k] + (uint32_t)byte] = bytes[byte];
            }
        }
    }

    if (first_active) {
        emit_high_bytes(s_buf, first_destination, first_high_length, first_high);
    }

    // The remaining three rounds retain the baseline's dense queue drain.
#pragma unroll
    for (uint32_t round = 1u; round < 4u; ++round) {
        const uint32_t request_idx = (uint32_t)lane + round * 32u;
        if (request_idx < high_count) {
            const uint32_t request = requests[request_idx];
            const uint32_t selected_code = request & 0xffffu;
            const uint32_t destination = (request >> 16u) & 0xfffu;
            const uint32_t high_length = (request >> 28u) + 1u;
            const uint2 high = *reinterpret_cast<const uint2 *>(dict_s8_hi + (size_t)selected_code * 8u);
            emit_high_bytes(s_buf, destination, high_length, high);
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
        const uint32_t off = head + k * 16u;
        const uint4 value = *reinterpret_cast<const uint4 *>(s_buf + off);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), value);
    }

    const uint32_t tail_start = head + (body_chunks << 4);
    if ((uint32_t)lane < warp_total - tail_start) {
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] = s_buf[tail_start + lane];
    }
}
