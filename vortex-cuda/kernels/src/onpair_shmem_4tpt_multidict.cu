// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress -- multi-dictionary, one-grid variant of
// `onpair_shmem_4tpt_split8read`.
//
// Motivation: the shipped kernel family takes ONE dictionary per launch, so a
// file made of hundreds of small row-groups (each independently OnPair-compressed
// with its OWN dictionary) can only be decoded as one launch per row-group. A
// 1--6 MB row-group launches under a fifth of one wave of thread blocks and
// reaches ~3% occupancy -- the "launch-bound small column" regime the paper
// reports only for completeness. This variant dissolves that: ONE launch spans
// EVERY row-group, so the grid is sized by the whole file, not one row-group,
// and the device fills.
//
// The decode of each 128-token warp-batch is byte-for-byte identical to
// `onpair_shmem_4tpt_split8read`; only the *addressing* changes. Per warp-batch
// the host supplies (rather than deriving `chunk = blockIdx*wpb + warp` and a
// single global dictionary):
//   batch_tok_off[b] -- global index into the concatenated `codes` of this
//                       batch's first token,
//   batch_out_off[b] -- global byte offset into the concatenated `output_bytes`,
//   batch_ntok[b]    -- tokens in this batch (1..128; the last batch of a
//                       row-group is short),
//   batch_rg[b]      -- the batch's row-group id.
// A per-row-group dict-pointer table `rg_dict_base[rg]` gives the row-group's
// base ENTRY offset into the three concatenated dictionary blobs (dict_s8,
// dict_padded, lens all hold `dict_size` entries per row-group, so one base
// serves all three: byte offset = base*8 into dict_s8, base*16 into dict_padded,
// base*1 into lens). A dictionary code `c` in row-group `rg` therefore resolves
// to entry `rg_dict_base[rg] + c`.
//
// Diff vs split8read is ~30 lines: the chunk math and the `i < total_tokens`
// bound become a per-batch `i < ntok` bound with a `tok0` base, the three dict
// bases are rebased per batch, and `out_start` comes from `batch_out_off`. The
// warp scan and the head/body/tail aligned drain are untouched.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 16u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif
#ifndef WARP_BUF_BYTES
#define WARP_BUF_BYTES 2080u
#endif

__device__ inline uint32_t warp_inclusive_scan_u32_md(uint32_t x, int lane) {
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

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_multidict(
    const uint16_t *__restrict codes, const uint8_t *__restrict dict_s8,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes,
    const uint64_t *__restrict batch_tok_off,
    const uint64_t *__restrict batch_out_off,
    const uint32_t *__restrict batch_ntok, const uint32_t *__restrict batch_rg,
    const uint64_t *__restrict rg_dict_base, uint64_t n_batches) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t batch =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (batch >= n_batches) {
        return;
    }

    // Per-batch addressing: rebase codes/dict/output onto this batch's
    // row-group. Everything below is identical to split8read once the bases and
    // the per-batch token count `ntok` are in hand.
    const uint32_t ntok = batch_ntok[batch];
    const uint32_t rg = batch_rg[batch];
    const uint64_t tok0 = batch_tok_off[batch];
    const uint64_t dbase = rg_dict_base[rg];  // entry base into the dict blobs

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    uint2 lo[4];
    uint32_t c[4];
    uint32_t l[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t i = (uint32_t)lane + (uint32_t)(k * 32);
        if (i < ntok) {
            const uint32_t code = (uint32_t)codes[tok0 + (uint64_t)i];
            c[k] = code;
            lo[k] = *reinterpret_cast<const uint2 *>(dict_s8 + (dbase + code) * 8u);
            l[k] = (uint32_t)lens[dbase + code];
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
        const uint32_t incl = warp_inclusive_scan_u32_md(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc_base;

    const uint64_t out_start = batch_out_off[batch];
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
            // Rare path: high bytes from the full padded dict (same row-group base).
            const uint2 hi = *reinterpret_cast<const uint2 *>(
                dict_padded + (dbase + c[k]) * 16u + 8u);
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
