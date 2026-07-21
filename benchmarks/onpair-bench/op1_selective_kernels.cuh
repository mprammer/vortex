// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// op1_selective_kernels.cuh -- OP1 EARLY-materialization SELECTIVE decode kernels,
// shared by the standalone bench (op1_selective_decode.cu) and the trade-off driver
// (offsets_tradeoff.cu).
//
// Distinct from op1_row_decode.cu (which decodes ALL rows, thread-per-row): here the
// upstream filter has already selected a subset of rows, and the decode runs ONLY over
// a COMPACTED survivor row-id list -- `r = row_ids[tid]`, grid sized to the survivor
// count -- NOT predicate-off over the full row range. This is the "take-on-codes" /
// early-materialization path: it compacts, which is exactly what breaks the stored
// chunk_offsets (contrast with masked-OP4 late-mat, which preserves them). The survivor
// list is built by stream compaction (CUB DeviceSelect::Flagged over a counting
// iterator), whose cost the caller TIMES as part of OP1's price.
//
// Load imbalance (thread-per-row) is inherited from op1_row_decode.cu and is the point:
// OP1 is efficient for small/uniform rows and serial for long rows.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Per-SURVIVOR-row decoded-size reduction: thread s handles survivor row row_ids[s],
// summing lens[codes[t]] over that row's tokens. Reads the free stored row offsets;
// no dict gather, no output -- a scan of the codes for the survivors only.
__global__ void sel_row_sizes_kernel(const uint16_t *__restrict codes,
                                     const uint64_t *__restrict row_off,
                                     const uint8_t *__restrict lens,
                                     const uint32_t *__restrict row_ids,
                                     uint64_t n_sel,
                                     uint64_t *__restrict rowsize) {
  const uint64_t s = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_sel) return;
  const uint64_t r = (uint64_t)row_ids[s];
  const uint64_t t0 = row_off[r], t1 = row_off[r + 1];
  uint64_t sum = 0;
  for (uint64_t t = t0; t < t1; ++t) sum += (uint64_t)lens[codes[t]];
  rowsize[s] = sum;
}

// Write the terminal output offset so the survivor output-offset buffer is a complete
// Arrow-style array of n_sel+1 entries. Runs after the exclusive scan, in-stream.
__global__ void sel_write_terminal_kernel(uint64_t *__restrict out_off,
                                          const uint64_t *__restrict rowsize,
                                          uint64_t n_sel) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    out_off[n_sel] = out_off[n_sel - 1] + rowsize[n_sel - 1];
}

// Selective row-partition decode: thread s writes SURVIVOR row row_ids[s]'s tokens
// starting at out_off[s], copying EXACTLY len bytes per token (<= 16). Exact-length
// copy => no thread writes past its row's compacted end, so survivors never race.
// The output is GLOBALLY COMPACTED (survivor rows back-to-back, no gaps) -- this is
// the early-materialization layout that breaks the stored offsets.
__global__ void sel_row_decode_kernel(const uint16_t *__restrict codes,
                                      const uint64_t *__restrict row_off,
                                      const uint64_t *__restrict out_off,
                                      const uint8_t *__restrict dict_padded,
                                      const uint8_t *__restrict lens,
                                      const uint32_t *__restrict row_ids,
                                      uint64_t n_sel,
                                      uint32_t max_token,
                                      uint8_t *__restrict out) {
  const uint64_t s = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_sel) return;
  const uint64_t r = (uint64_t)row_ids[s];
  const uint64_t t0 = row_off[r], t1 = row_off[r + 1];
  uint64_t cur = out_off[s];
  for (uint64_t t = t0; t < t1; ++t) {
    const uint16_t c = codes[t];
    const uint32_t len = lens[c];
    const uint8_t *src = dict_padded + (size_t)c * (size_t)max_token;
    for (uint32_t j = 0; j < len; ++j) out[cur + j] = src[j];
    cur += len;
  }
}
