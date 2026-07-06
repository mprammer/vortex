// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// FSST decode via the OnPair recipe — 4 codes per thread, 128 codes per warp.
//
// This is the *generality* port: it applies OnPair's fixed-stride-table +
// warp-scan + staged-then-aligned-drain recipe to UNMODIFIED FSST-compressed
// data. The point is to show the recipe is a codec-agnostic decode pattern,
// not something bespoke to OnPair. It is meant to be benchmarked on an A100
// against GSST's published 191 GB/s on the same chip / codec / column
// (TPC-H l_comment) and against this tree's thread-per-string `fsst.cu`.
//
// ─────────────────────────────────────────────────────────────────────────
// Recipe mapping (OnPair `onpair_shmem_4tpt.cu`  →  FSST here)
// ─────────────────────────────────────────────────────────────────────────
//   OnPair                                   FSST
//   ------                                   ----
//   one warp per 128 *tokens*                one warp per 128 *codes*
//   2-byte (u16) fixed-width code            1-byte code, 255 = ESCAPE
//   len = lens[code]  (side table)           len = symbol_lengths[code], or 1
//                                              for an escaped literal
//   16-byte vector load from a fixed-stride  8-byte symbol read from a
//     dict table (dict is LARGE → L1/L2)       256-entry table STAGED IN
//                                              SHARED MEM (table is ≤2 KB)
//   warp exclusive scan of token lengths     warp scan of code lengths
//   staged per-lane byte writes → shared     same
//   cooperative aligned 16-B drain           same (verbatim)
//
// Two things differ from OnPair and are the whole substance of the port:
//
// (1) TABLE IN SHARED MEMORY.  OnPair's dictionary can be many KB–MB and is
//     left resident in L1/L2 (that residency, and the request traffic to it,
//     is the paper's central "cache-request bound" story). FSST's symbol
//     table is at most 255 symbols × 8 B ≈ 2 KB, so it fits comfortably in
//     shared memory. We stage it ONCE per block; every `symbols[code]` lookup
//     is then a shared-memory bank read rather than an L1 request. This is the
//     interesting contrast: same recipe, but the table moves off the cache
//     request path entirely because the codec's table is tiny.
//
// (2) ESCAPES MAKE THE CODE→BYTE MAP POSITION-DEPENDENT.  In OnPair a code is
//     a fixed 2 bytes, so lane L's k-th token is at a computable input
//     position (`chunk*128 + lane + k*32`) with no parsing. In FSST a code is
//     1 byte UNLESS it is 255 (ESCAPE), in which case it consumes an extra
//     input byte (the literal). Worse, escape detection is inherently
//     sequential: you cannot classify input byte i as a code vs. an escaped
//     literal without parsing forward from a known boundary (an escaped
//     literal may itself be the byte value 255). So a lane cannot random-access
//     "its" codes — it must serially walk a CONTIGUOUS run of input bytes from
//     a known offset.
//
//     We handle this exactly the way OnPair handles output offsets with
//     `chunk_offsets`: the HOST precomputes the per-lane input byte offset in
//     `group_in_off`. Each lane then owns a contiguous run of 4 codes and
//     parses them serially (branch on 255). This keeps the warp-scan / drain
//     structure identical; only the code→length step gains a 2-way branch and
//     the intra-warp code ordering changes (see the scan comment below).
//
//     `group_in_off` is the FSST analog of OnPair's `chunk_offsets`, but at
//     4-code (one lane) granularity rather than 128-code (one warp)
//     granularity — because escape boundaries can't be rediscovered in-warp,
//     each lane needs its own start. It is host-side setup (like
//     `chunk_offsets`) and is NOT part of the timed decode.
//
// Byte-exact by construction: a code emits exactly `len` bytes of its symbol
// (or 1 literal byte for an escape); the warp scan lays those out in code
// order; the drain copies shared→global unchanged. The standalone bench
// validates every run against a CPU reference decode.

#ifndef FSST_WARPS_PER_BLOCK_MAX
#define FSST_WARPS_PER_BLOCK_MAX 16u
#endif
#ifndef FSST_LAUNCH_BOUNDS
// Scratch is half OnPair's (8 B/code vs 16 B/token), so register/occupancy
// pressure is lower; 512 threads × 2 resident blocks matches the OnPair build.
#define FSST_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif

// Each warp buffers up to 128 codes × 8 B = 1024 B of decoded bytes, plus up
// to 15 B of head-alignment slack and up to 15 B of drain over-read room for
// the trailing uint4. Round up to a 16-B multiple.
#define FSST_WARP_BUF_BYTES 1088u

// The escape code: 255 means "the next input byte is a literal".
#define FSST_ESCAPE_CODE 255u

// Codes handled per lane (recipe-faithful "4 tokens per thread").
#define FSST_CODES_PER_LANE 4u
// Codes handled per warp (32 lanes × FSST_CODES_PER_LANE).
#define FSST_CODES_PER_BATCH (32u * FSST_CODES_PER_LANE)

static_assert(FSST_WARP_BUF_BYTES >= FSST_CODES_PER_BATCH * 8u + 32u,
              "warp scratch too small for 128 codes × 8 B + alignment slack");

// Inclusive prefix scan across a warp (5 shuffles). Verbatim from the OnPair
// recipe; renamed to avoid ODR clashes when this file is #included alongside
// an OnPair kernel in a standalone bench translation unit.
__device__ inline uint32_t fsst_warp_inclusive_scan_u32(uint32_t x, int lane) {
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

// Stage a decoded symbol's bytes into the shared scratch at byte offset `base`.
// A symbol is ≤ 8 bytes held little-endian in `sym` (byte j = (sym >> 8j)).
// We write EXACTLY `len` bytes — consecutive codes pack tightly in the scratch,
// so writing the full 8 bytes would clobber the neighbouring code's bytes that
// another lane is writing concurrently (before the __syncwarp). Mirrors
// OnPair's `emit_token`, narrowed from 16 to 8 bytes (FSST symbols ≤ 8 B).
__device__ inline void fsst_emit_symbol(uint8_t *s_buf, uint32_t base,
                                        uint64_t sym, uint32_t len) {
    const uint8_t *sb = reinterpret_cast<const uint8_t *>(&sym);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = sb[j];
        }
    }
}

// FSST decode kernel.
//
//   codes_bytes     flat FSST code stream (all strings concatenated; a symbol
//                   never spans a string boundary and an escape's literal is
//                   in the same string, so the concatenation decodes as the
//                   concatenation of the per-string outputs).
//   group_in_off    input byte offset of each 4-code group. Group index for
//                   (batch b, lane L) is b*32 + L; entry = input offset of code
//                   b*128 + L*4. Length ≥ num_batches*32 + 1 (host pads the tail
//                   with the input length so fully-inactive lanes read in-bounds).
//   batch_out_off   output byte offset of each 128-code batch (the FSST analog
//                   of OnPair's chunk_offsets). Length num_batches + 1.
//   symbols         256-entry symbol table, u64 little-endian, one symbol per
//                   code (< 255). Entry 255 is unused (escape). ≤ 2 KB.
//   symbol_lengths  256-entry per-code byte length (1..8). Entry 255 unused.
//   output_bytes    decoded output; base assumed 16-aligned (cudaMalloc is).
//   total_codes     number of codes (NOT input bytes; an escape is one code
//                   that consumes two input bytes).
extern "C" __global__ FSST_LAUNCH_BOUNDS void fsst_4tpt(
    const uint8_t *__restrict__ codes_bytes,
    const uint32_t *__restrict__ group_in_off,
    const uint64_t *__restrict__ batch_out_off,
    const uint64_t *__restrict__ symbols,
    const uint8_t *__restrict__ symbol_lengths,
    uint8_t *__restrict__ output_bytes, uint64_t total_codes) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t warps_per_block = blockDim.x >> 5;

    // ── Stage the symbol table into shared memory, ONCE per block ──
    // Every thread in the block participates (must reach the __syncthreads),
    // so this precedes the per-warp out-of-range early-out below.
    __shared__ uint64_t s_symbols[256];
    __shared__ uint8_t s_symlen[256];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
        s_symbols[i] = symbols[i];
        s_symlen[i] = symbol_lengths[i];
    }
    __syncthreads();

    const uint64_t batch =
        (uint64_t)blockIdx.x * (uint64_t)warps_per_block + (uint64_t)warp_id;
    if (batch * (uint64_t)FSST_CODES_PER_BATCH >= total_codes) {
        return;
    }

    __shared__ __align__(16)
        uint8_t s_buf_all[FSST_WARPS_PER_BLOCK_MAX * FSST_WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * FSST_WARP_BUF_BYTES];

    // ── Serial parse of this lane's contiguous 4-code run ──
    // Lane L owns codes [batch*128 + L*4, +4). `group_in_off` gives the input
    // byte offset of the first; we walk forward, branching on the escape code.
    const uint64_t group = batch * 32u + (uint64_t)lane;
    uint32_t s = group_in_off[group];  // input byte cursor for this lane
    const uint64_t code0 = batch * (uint64_t)FSST_CODES_PER_BATCH +
                           (uint64_t)lane * (uint64_t)FSST_CODES_PER_LANE;

    uint64_t sym[FSST_CODES_PER_LANE];
    uint32_t len[FSST_CODES_PER_LANE];
#pragma unroll
    for (uint32_t k = 0; k < FSST_CODES_PER_LANE; ++k) {
        const uint64_t gidx = code0 + (uint64_t)k;
        if (gidx < total_codes) {
            const uint8_t code = codes_bytes[s];
            if (code == FSST_ESCAPE_CODE) {
                // Escape: the following input byte is emitted verbatim.
                sym[k] = (uint64_t)codes_bytes[s + 1];
                len[k] = 1u;
                s += 2u;
            } else {
                // Regular code: one shared-memory symbol lookup.
                sym[k] = s_symbols[code];
                len[k] = (uint32_t)s_symlen[code];
                s += 1u;
            }
        } else {
            // Past the end of the stream (partial final batch): inactive slot.
            sym[k] = 0u;
            len[k] = 0u;
        }
    }

    // ── Warp scan → per-code output offset within the warp's region ──
    // NOTE the ordering differs from OnPair. OnPair interleaves lanes
    // (token index = lane + k*32) and scans k-major. Here each lane owns a
    // CONTIGUOUS 4-code run (code index = lane*4 + k), forced by the sequential
    // escape parse, so codes concatenate LANE-major: all of lane 0's codes,
    // then all of lane 1's, … The offset of code (lane,k) is therefore
    //   (Σ lengths of every code in lanes < lane) + (Σ this lane's len[0..k]).
    // One warp scan over the per-lane total gives the first term; the second is
    // a tiny local prefix. This is actually cheaper than OnPair's four scans.
    uint32_t lane_total = len[0] + len[1] + len[2] + len[3];
    const uint32_t incl = fsst_warp_inclusive_scan_u32(lane_total, lane);
    const uint32_t lane_base = incl - lane_total;            // exclusive prefix
    const uint32_t warp_total = __shfl_sync(mask, incl, 31); // total warp bytes

    uint32_t excl[FSST_CODES_PER_LANE];
    uint32_t acc = lane_base;
#pragma unroll
    for (uint32_t k = 0; k < FSST_CODES_PER_LANE; ++k) {
        excl[k] = acc;
        acc += len[k];
    }

    // ── Head alignment + staged emit into shared scratch ──
    // Shift the scratch base so that s_buf + head is 16-aligned in shared
    // memory, matching the global cursor's alignment for the aligned body drain.
    const uint64_t out_start = batch_out_off[batch];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (uint32_t k = 0; k < FSST_CODES_PER_LANE; ++k) {
        if (len[k] > 0u) {
            fsst_emit_symbol(s_buf, excl[k], sym[k], len[k]);
        }
    }
    __syncwarp();

    // ── Cooperative aligned drain (head / body / tail) — verbatim OnPair ──
    const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
    if ((uint32_t)lane < head) {
        output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
    }
    if (head >= warp_total) {
        return;
    }

    const uint32_t body_chunks = (warp_total - head) >> 4;
    for (uint32_t c = lane; c < body_chunks; c += 32u) {
        const uint32_t off = head + c * 16u;
        const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + off);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), v);
    }

    const uint32_t tail_start = head + (body_chunks << 4);
    if ((uint32_t)lane < warp_total - tail_start) {
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] =
            s_buf[tail_start + lane];
    }
}
