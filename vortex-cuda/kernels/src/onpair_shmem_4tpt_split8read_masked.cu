// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// OnPair decompress — masked LATE-MATERIALIZATION variant of
// `onpair_shmem_4tpt_split8read` (the shipped decoder). This is the ONE new
// kernel for the offsets trade-off experiment (see the paper's late-vs-early
// materialization story). The shipped dense path is NOT modified; this is an
// additive variant.
//
// It answers Joe's "stored offsets break on filter/slice" objection to OP4
// (store chunk_offsets at write time). Late-materialization keeps the stored
// offsets valid under an upstream filter: we decode the DENSE chunk using the
// SAME stored `chunk_offsets`, but skip *resolving* (dict gather + byte emit)
// the masked-out tokens. Nothing is compacted across chunks — each chunk still
// begins at its stored `chunk_offsets[chunk]`, so the write-time offsets stay
// valid. Within a chunk, the existing warp scan compacts the survivors to the
// front (a masked token contributes output length 0), so the coalesced drain
// writes exactly the survivor bytes.
//
// ── The ONLY differences vs onpair_shmem_4tpt_split8read (line-by-line): ──
//   1. one added parameter `sel_mask` (a packed 128-bit-per-chunk selection
//      bitmask: 4 x u32 words per 128-token chunk; word k covers local token
//      indices [k*32, k*32+32), bit `lane` selects this warp's token k);
//   2. the load loop reads the 4 mask words once per warp, and for a masked-out
//      token forces length l[k]=0 and SKIPS the request-bound `dict_s8` gather
//      + the `lens[code]` LUT read. The dense `codes[i]` read is KEPT (the
//      "read everything, resolve nothing" floor is dense code read + mask read).
//   3. EVERYTHING downstream (the inclusive scan, `out_start`, the emit loop,
//      the staged drain) is byte-identical to the base. Because a masked token
//      has l[k]=0, the base's `if (len == 0u) continue;` already skips its
//      gather+emit, and the scan already gives it zero output bytes — so the
//      survivors land compacted and byte-exact with zero new drain logic.
//   4. one added output `survivor_len` (u32/chunk): the per-chunk survivor byte
//      count (== the drain `warp_total` the warp scan already computes). This is
//      the per-CHUNK "survivor-length directory": together with a per-ROW prefix
//      directory it lets the downstream row-level scan (offsets_tradeoff.cu's
//      row_scan_latemat) locate each survivor row's per-chunk segments by pure
//      lookup — middle/last-segment extents come from survivor_len — with no
//      codes/lens reread. Decode and scan stay SEPARATE kernels; fusing them is
//      the production/future-work endpoint (it breaks leg composition and is
//      operator-specific).
//
// Byte-exactness (verified by construction; confirmed on-box by
// offsets_tradeoff.cu's validation mode): the bytes emitted for a survivor token
// are `dict[code][0:len]`, in warp/token order, exactly as the dense kernel
// emits them; only the *masked* tokens are absent, and the stored `out_start`
// per chunk is unchanged. So masked_out, read as (per-chunk survivor runs
// anchored at the stored offsets), equals dense_out filtered to the survivor
// tokens.
//
// At m=100% (all mask bits set) the kernel does IDENTICAL work to dense
// split8read PLUS only the 16-byte mask read + per-token predication — this is
// the apparatus-tax cell, and its throughput must be <= dense (never faster; a
// faster result is a bug).

// Pull in the shipped base kernel unchanged: this gives us the exact
// `onpair_shmem_4tpt_split8read` symbol for side-by-side comparison, plus the
// shared macros (WARPS_PER_BLOCK_MAX, ONPAIR_LAUNCH_BOUNDS, WARP_BUF_BYTES) and
// the `warp_inclusive_scan_u32_s8r` device helper — defined exactly once here.
#include "onpair_shmem_4tpt_split8read.cu"

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_split8read_masked(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_s8, const uint8_t *__restrict dict_padded,
    const uint8_t *__restrict lens, uint8_t *__restrict output_bytes,
    uint64_t total_tokens, const uint32_t *__restrict sel_mask,
    uint32_t *__restrict survivor_len) {
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

    // (NEW) Read the 128-bit selection mask for this 128-token chunk. Four u32
    // words; word k covers this warp's token k for every lane. All 32 lanes read
    // the SAME four consecutive words (a single broadcast 16-byte transaction) —
    // this is the only added global traffic vs the dense kernel.
    uint32_t selw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        selw[k] = sel_mask[(size_t)chunk * 4u + (uint32_t)k];
    }

    const uint64_t base_i = chunk * 128u + (uint64_t)lane;
    uint2 lo[4];
    uint32_t c[4];
    uint32_t l[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        const bool sel = ((selw[k] >> (unsigned)lane) & 1u) != 0u;  // (NEW) survivor?
        if (i < total_tokens) {
            const uint32_t code = (uint32_t)codes[i];  // dense code read — kept (the floor)
            c[k] = code;
            if (sel) {
                // Survivor: identical to the dense load — the request-bound gather.
                lo[k] = *reinterpret_cast<const uint2 *>(dict_s8 + (size_t)code * 8u);
                l[k] = (uint32_t)lens[code];
            } else {
                // (NEW) Masked out: length 0 so the scan compacts it away, and
                // SKIP the request-bound dict_s8 gather + the lens LUT read.
                lo[k] = make_uint2(0u, 0u);
                l[k] = 0u;
            }
        } else {
            c[k] = 0u;
            lo[k] = make_uint2(0u, 0u);
            l[k] = 0u;
        }
    }

    // ── From here down: byte-identical to onpair_shmem_4tpt_split8read. ──
    uint32_t excl[4];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_s8r(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc_base;

    // (NEW) Publish the per-chunk survivor-length directory (uniform across the warp).
    // The downstream row-level scan (row_scan_latemat) uses survivor_len[chunk] to get
    // the extent of a survivor row's middle/last per-chunk segments by lookup, so its
    // gapped bytes are located without rereading codes/lens.
    if (lane == 0) {
        survivor_len[chunk] = warp_total;
    }

    const uint64_t out_start = chunk_offsets[chunk];  // STORED dense offset — unchanged ("stays valid")
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t len = l[k];
        if (len == 0u) {  // masked-out (len==0) tokens fall out here: no gather, no emit
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
