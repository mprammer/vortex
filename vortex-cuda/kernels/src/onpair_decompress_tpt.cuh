// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Configurable-TPT OnPair decoder for the original u16 code stream and a
// CPU-flattened dictionary. Each warp owns TOKENS_PER_WARP tokens. The
// dictionary has dense 8-byte low/high planes and packed four-bit lengths.

#ifndef TOKENS_PER_THREAD
#error "TOKENS_PER_THREAD must be defined"
#endif
#ifndef ONPAIR_KERNEL_NAME
#error "ONPAIR_KERNEL_NAME must be defined"
#endif
// The block size is ONE definition, and both the launch bound and the shared-memory
// allocation derive from it. They were previously independent -- __launch_bounds__(256, 4)
// in each variant and a hardcoded WARPS_PER_BLOCK_MAX of 8 here -- so a variant that changed
// its block size without changing the other silently either over-allocated shared memory for
// warps it never launches, or, in the dangerous direction, indexed s_buf_all and s_requests
// past their end. Derive both and the failure mode cannot occur.
#ifndef ONPAIR_BLOCK_THREADS
#define ONPAIR_BLOCK_THREADS 256u
#endif
#ifndef ONPAIR_MIN_BLOCKS
#define ONPAIR_MIN_BLOCKS 4u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(ONPAIR_BLOCK_THREADS, ONPAIR_MIN_BLOCKS)
#endif
// W, the low-plane width in bytes. 8 reads a dense 8-byte plane and sends the tail of long
// tokens through the request queue; 16 reads one padded 16-byte entry and has no high path at
// all, so the queue, the hoist and the second gather disappear together.
//
// The two arms carry the SAME dictionary footprint, which is what makes this a clean control:
// at 4096 entries W=8 is 32768 (lo) + 32768 (hi) + 2048 (nibble lengths) = 67584 B, and W=16
// is 65536 (padded) + 2048 = 67584 B. Cache residency is therefore held fixed by construction
// rather than argued away, which the superseded split8read-vs-stride16 comparison could not do
// (100 KB against 68 KB). What is left varying is one wide access against one narrow access
// plus a conditional second.
//
// Only 8 and 16 are meaningful: the load must be a native vector width, and at 4 bytes almost
// every token takes the long path (measured mean token length is 7.8 to 11.8 B), which is the
// superseded split4read and never beat baseline.
#ifndef ONPAIR_LOW_PLANE_BYTES
#define ONPAIR_LOW_PLANE_BYTES 8u
#endif
#if ONPAIR_LOW_PLANE_BYTES != 8u && ONPAIR_LOW_PLANE_BYTES != 16u
#error "ONPAIR_LOW_PLANE_BYTES must be 8 or 16"
#endif

// H, how many rounds of high-plane tokens are HELD -- issued before the low-byte emit and
// resolved after it. This is the hoist already in the shipped decoder at H=1: it is what leaves
// independent work behind the LDG. H>1 keeps more loads live across the emit, costing registers.
// Meaningless at W=16, where there is no high plane.
//
// Named H, not C: the paper already uses C1/C2/C3 for the three design challenges.
//
// H only does anything when the warp's queue is deeper than 32*(H-1). On a column where almost
// every token is short -- wikipedia at OnPair-12 has frac_le8 = 0.981, about 3.6 long tokens per
// K=6 warp -- round 2 can never fire, so H>1 there measures extra live registers and predicates
// rather than extra early loads. Sweep H on deep-queue columns (low frac_le8), not shallow ones.
#ifndef ONPAIR_HELD_HIGH
#define ONPAIR_HELD_HIGH 1u
#endif
#if ONPAIR_HELD_HIGH < 1u || ONPAIR_HELD_HIGH > TOKENS_PER_THREAD
#error "ONPAIR_HELD_HIGH must be in [1, TOKENS_PER_THREAD]"
#endif

// K <= 8 is a HARD limit of the request encoding, not a preference. pack_high_request puts the
// staging destination in bits 27:16, twelve bits, max 4095. For N = 32K tokens the largest
// destination is 16(N-1)+8 = 16N-8, so K=8 gives 4088 and fits; K=9 reaches 4600, overlaps the
// high-length field, and silently corrupts requests. Widening the destination to thirteen bits
// (bits 28:16, moving the three-bit high length to 31:29 and using the currently unused bit 31)
// would raise the limit to K<=16.
#if TOKENS_PER_THREAD > 8u
#error "TOKENS_PER_THREAD > 8 overflows the 12-bit destination in pack_high_request"
#endif
// The kernel indexes shared state by warp and assumes whole warps.
#if (ONPAIR_BLOCK_THREADS % 32u) != 0u
#error "ONPAIR_BLOCK_THREADS must be a multiple of 32"
#endif

#if ONPAIR_LOW_PLANE_BYTES == 16u
#define ONPAIR_LO_VEC uint4
#else
#define ONPAIR_LO_VEC uint2
#endif

#define WARPS_PER_BLOCK_MAX (ONPAIR_BLOCK_THREADS / 32u)
#define TOKENS_PER_WARP     (TOKENS_PER_THREAD * 32u)
#define WARP_BUF_BYTES      (TOKENS_PER_WARP * 16u + 32u)
#define REQUESTS_PER_WARP   TOKENS_PER_WARP

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

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void ONPAIR_KERNEL_NAME(const uint16_t *__restrict codes,
                                                                   const uint64_t *__restrict chunk_offsets,
                                                                   const uint8_t *__restrict dict_s8_lo,
                                                                   const uint8_t *__restrict dict_s8_hi,
                                                                   const uint8_t *__restrict packed_lens,
                                                                   uint8_t *__restrict output_bytes,
                                                                   uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    // Every shared index below is relative to warp_id, so a block launched wider than the
    // kernel was compiled for would run off the end of s_buf_all and s_requests and corrupt
    // another warp's staging silently. The host is expected to launch ONPAIR_BLOCK_THREADS;
    // this makes a mismatch drop work rather than produce wrong bytes, and the host-side
    // check on MAX_THREADS_PER_BLOCK is what turns it into a loud failure.
    if (warp_id >= WARPS_PER_BLOCK_MAX) {
        return;
    }
    const uint64_t chunk = (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * TOKENS_PER_WARP >= total_tokens) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
#if ONPAIR_LOW_PLANE_BYTES == 8u
    // At W=16 there is no high plane, so the request queue does not exist and its shared
    // memory is not allocated. That keeps the W comparison honest on occupancy as well as on
    // dictionary footprint.
    __shared__ __align__(16) uint32_t s_requests[WARPS_PER_BLOCK_MAX][REQUESTS_PER_WARP];
    uint32_t *requests = s_requests[warp_id];
#endif
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    const uint64_t base_i = chunk * TOKENS_PER_WARP + (uint64_t)lane;
    ONPAIR_LO_VEC lo[TOKENS_PER_THREAD];
    uint32_t code[TOKENS_PER_THREAD];
    uint32_t len[TOKENS_PER_THREAD];
#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        if (i < total_tokens) {
            code[k] = (uint32_t)codes[i];
            // At W=16 the host binds the padded stride-16 table to this pointer, so one wide
            // load covers the whole token and no second gather is ever needed.
            lo[k] = *reinterpret_cast<const ONPAIR_LO_VEC *>(
                dict_s8_lo + (size_t)code[k] * (size_t)ONPAIR_LOW_PLANE_BYTES);
            len[k] = unpack_length(packed_lens, code[k]);
        } else {
            code[k] = 0u;
#if ONPAIR_LOW_PLANE_BYTES == 16u
            lo[k] = make_uint4(0u, 0u, 0u, 0u);
#else
            lo[k] = make_uint2(0u, 0u);
#endif
            len[k] = 0u;
        }
    }

    constexpr uint64_t field_mask = 0xffffull;
    static_assert(32u * 16u <= field_mask, "packed fields must hold a full plane prefix");
    uint32_t excl[TOKENS_PER_THREAD];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int group = 0; group < (int)TOKENS_PER_THREAD; group += 4) {
        uint64_t packed = 0u;
#pragma unroll
        for (int field = 0; field < 4; ++field) {
            const int k = group + field;
            if (k < (int)TOKENS_PER_THREAD) {
                packed |= (uint64_t)len[k] << ((uint32_t)field * 16u);
            }
        }
        packed = warp_scan_four_u16_prefixes(packed, lane);
        const uint64_t packed_totals = __shfl_sync(mask, packed, 31);
#pragma unroll
        for (int field = 0; field < 4; ++field) {
            const int k = group + field;
            if (k < (int)TOKENS_PER_THREAD) {
                const uint32_t shift = (uint32_t)field * 16u;
                const uint32_t incl = (uint32_t)((packed >> shift) & field_mask);
                const uint32_t plane_total = (uint32_t)((packed_totals >> shift) & field_mask);
                excl[k] = acc_base + incl - len[k];
                acc_base += plane_total;
            }
        }
    }
    const uint32_t warp_total = acc_base;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#if ONPAIR_LOW_PLANE_BYTES == 8u
    // Build the identical plane-major request stream first, so dense lane N
    // still owns request N and the first high gather can be issued early.
    uint32_t high_count = 0u;
#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
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

    // Hold the first H rounds of high-plane loads now and resolve them after the low-byte
    // emit. H=1 is the shipped decoder; larger H keeps more loads live across the emit.
    uint32_t hoist_destination[ONPAIR_HELD_HIGH];
    uint32_t hoist_length[ONPAIR_HELD_HIGH];
    uint2 hoist_high[ONPAIR_HELD_HIGH];
    bool hoist_active[ONPAIR_HELD_HIGH];
#pragma unroll
    for (uint32_t h = 0u; h < ONPAIR_HELD_HIGH; ++h) {
        const uint32_t idx = (uint32_t)lane + h * 32u;
        hoist_active[h] = idx < high_count;
        hoist_destination[h] = 0u;
        hoist_length[h] = 0u;
        hoist_high[h] = make_uint2(0u, 0u);
        if (hoist_active[h]) {
            const uint32_t request = requests[idx];
            const uint32_t selected_code = request & 0xffffu;
            hoist_destination[h] = (request >> 16u) & 0xfffu;
            hoist_length[h] = (request >> 28u) + 1u;
            hoist_high[h] = *reinterpret_cast<const uint2 *>(dict_s8_hi + (size_t)selected_code * 8u);
        }
    }
#endif

    // The first high value is deliberately not consumed until all owners have
    // emitted their low bytes, creating independent instructions after LDG.
#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
        // At W=16 the single wide load already holds the whole token, so this writes all of
        // it and nothing is left for a high path.
        const uint32_t low_length =
            len[k] < ONPAIR_LOW_PLANE_BYTES ? len[k] : (uint32_t)ONPAIR_LOW_PLANE_BYTES;
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&lo[k]);
#pragma unroll
        for (int byte = 0; byte < (int)ONPAIR_LOW_PLANE_BYTES; ++byte) {
            if (byte < (int)low_length) {
                s_buf[excl[k] + (uint32_t)byte] = bytes[byte];
            }
        }
    }

#if ONPAIR_LOW_PLANE_BYTES == 8u
#pragma unroll
    for (uint32_t h = 0u; h < ONPAIR_HELD_HIGH; ++h) {
        if (hoist_active[h]) {
            emit_high_bytes(s_buf, hoist_destination[h], hoist_length[h], hoist_high[h]);
        }
    }

    // The remaining rounds retain the baseline's dense queue drain.
#pragma unroll
    for (uint32_t round = ONPAIR_HELD_HIGH; round < TOKENS_PER_THREAD; ++round) {
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
#endif
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
