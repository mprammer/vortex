// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// op_gpu_regen.cu -- OP2 of the output-positioning trade-off: regenerate the per-batch
// output offsets (`chunk_offsets`) ON THE GPU with a scan over the compressed codes, then
// decode -- versus decoding with the offsets already resident (the "store" baseline, OP4).
//
// The offsets are a length prefix-sum sampled at 128-token (warp-batch) boundaries:
//   chunk_offsets[b] = sum over tokens t < b*128 of lens[codes[t]]  (decoded bytes before batch b).
// On the GPU that is (1) a per-batch decoded-size reduction over the codes + a length LUT, then
// (2) an exclusive prefix sum over the per-batch sizes. Neither touches the dictionary bytes or
// the output -- it is a scan of the *compressed* stream. This bench measures whether paying that
// scan at decode time (0 stored bytes) is competitive with storing the offsets (OP4).
//
// Input: the same "E2E1" dump onpair_bench.rs writes via ONPAIR_DUMP_E2E (codes, lens, dict_padded)
// -- reused so no new dump format is needed.
//
// Build (on the GPU box, from this directory):
//   nvcc -O3 -arch=native -std=c++17 op_gpu_regen.cu -o op_gpu_regen
// Run:
//   ./op_gpu_regen <dump.e2ebin> [iters]
// Emits one JSON object on stdout (diagnostics on stderr); exit 0 iff offsets + decode are exact.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

// The shipped single-dictionary decode kernel (onpair_shmem_4tpt_split8read).
#include "../../vortex-cuda/kernels/src/onpair_shmem_4tpt_split8read.cu"

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e_));                                         \
      exit(3);                                                                 \
    }                                                                          \
  } while (0)

// BATCH GRANULARITY. One offset per batch of TOK_PER_BATCH codes, and a warp batch is 32*K
// codes, so this constant IS the sidecar's write-time commitment to K: 128 is K=4, 192 is the
// paper's default K=6. Overridable at compile time (-DTOK_PER_BATCH_OVERRIDE=192) to test whether
// the regeneration cost depends on granularity: phase 1 reads every code regardless, and only the
// CUB scan over n_chunks shrinks, so the prediction is that it does not.
//
// The DECODE half cannot follow. The kernel included below hardcodes its own 128-token batch
// (chunk * 128u), so at any other granularity it would read the wrong offsets. At != 128 we
// therefore measure regeneration ALONE and report nulls for the decode and the overhead rather
// than a number that looks comparable and is not.
#ifndef TOK_PER_BATCH_OVERRIDE
#define TOK_PER_BATCH_OVERRIDE 128
#endif
static constexpr uint32_t TOK_PER_BATCH = TOK_PER_BATCH_OVERRIDE;
static constexpr bool DECODE_VALID = (TOK_PER_BATCH_OVERRIDE == 128u);
static constexpr int BLOCK_THREADS = 512;               // 16 warps/block
static constexpr uint32_t WARPS_PER_BLOCK = BLOCK_THREADS / 32;

// Per-batch decoded-size reduction: one warp per 128-token batch sums lens[codes[t]] over its
// (up to) 128 tokens and writes the total to batchsize[b]. Coalesced code reads (lane-consecutive);
// no dictionary-byte gather, no output -- a scan of the compressed codes + a cache-resident LUT.
__global__ void batch_sizes_kernel(const uint16_t *__restrict codes,
                                   const uint8_t *__restrict lens,
                                   uint64_t total_tokens,
                                   uint64_t *__restrict batchsize) {
  const int lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const uint64_t b = (uint64_t)blockIdx.x * (blockDim.x >> 5) + warp;
  const uint64_t base = b * (uint64_t)TOK_PER_BATCH;
  if (base >= total_tokens) return;
  uint32_t s = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint64_t i = base + (uint64_t)lane + (uint64_t)(k * 32);
    if (i < total_tokens) s += (uint32_t)lens[codes[i]];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if (lane == 0) batchsize[b] = (uint64_t)s;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <dump.e2ebin> [iters]\n", argv[0]);
    return 1;
  }
  const char *path = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 100;
  if (iters < 3) iters = 3;

  // ── load the E2E1 dump ─────────────────────────────────────────────────────
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
  char magic[4];
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "E2E1", 4) != 0) {
    fprintf(stderr, "bad magic in %s\n", path); return 2;
  }
  uint64_t total_tokens = 0;
  uint32_t dict_size = 0, max_token = 0;
  if (fread(&total_tokens, 8, 1, f) != 1 || fread(&dict_size, 4, 1, f) != 1 ||
      fread(&max_token, 4, 1, f) != 1) { fprintf(stderr, "short header\n"); return 2; }
  if (max_token != 16) {
    fprintf(stderr, "max_token=%u but the GPU kernel ABI requires 16\n", max_token); return 2;
  }
  if (dict_size == 0 || total_tokens == 0) { fprintf(stderr, "empty dump\n"); return 2; }
  // Bound total_tokens so the ceil (total_tokens+127)/128 cannot overflow and n_chunks stays within
  // CUB's int num_items. total_tokens <= INT_MAX*128 => n_chunks <= INT_MAX.
  if (total_tokens > (uint64_t)INT_MAX * TOK_PER_BATCH) {
    fprintf(stderr, "total_tokens=%llu too large; split the column\n", (unsigned long long)total_tokens);
    return 2;
  }
  std::vector<uint16_t> codes(total_tokens);
  std::vector<uint8_t> lens(dict_size);
  std::vector<uint8_t> dict_padded((size_t)dict_size * max_token);
  // NB: E2E1 writes the header + codes little-endian; this reads directly into native ints, so it
  // assumes a little-endian host (x86-64 / AArch64, both LE — the only build targets).
  if (fread(codes.data(), 2, total_tokens, f) != total_tokens ||
      fread(lens.data(), 1, dict_size, f) != dict_size ||
      fread(dict_padded.data(), 1, dict_padded.size(), f) != dict_padded.size()) {
    fprintf(stderr, "short body\n"); return 2;
  }
  fclose(f);
  // Bounds: every code must index a real entry, and every length must fit the fixed max_token-byte
  // entry (else the host memcpy over-reads dict_padded and the warp buffer over-writes).
  for (uint16_t c : codes) if (c >= dict_size) { fprintf(stderr, "code %u out of range\n", (unsigned)c); return 2; }
  for (uint8_t l : lens) if (l > max_token) { fprintf(stderr, "dict length %u exceeds max_token %u\n", (unsigned)l, max_token); return 2; }

  // ── host derivation: reference offsets, dict_s8, host reference decode ──────
  std::vector<uint64_t> tok_off(total_tokens + 1);
  tok_off[0] = 0;
  for (uint64_t i = 0; i < total_tokens; ++i) tok_off[i + 1] = tok_off[i] + lens[codes[i]];
  const uint64_t decoded_bytes = tok_off[total_tokens];
  if (decoded_bytes == 0) { fprintf(stderr, "dump decodes to 0 bytes; unsupported\n"); return 2; }

  std::vector<uint8_t> dict_s8((size_t)dict_size * 8, 0);
  for (uint32_t c = 0; c < dict_size; ++c) {
    uint32_t n8 = lens[c] < 8 ? lens[c] : 8;
    memcpy(&dict_s8[(size_t)c * 8], &dict_padded[(size_t)c * max_token], n8);
  }

  const uint64_t n_chunks = (total_tokens + TOK_PER_BATCH - 1) / TOK_PER_BATCH;
  // CUB DeviceScan takes an int num_items; reject rather than silently truncate a huge column.
  if (n_chunks > (uint64_t)INT_MAX) {
    fprintf(stderr, "n_chunks=%llu exceeds INT_MAX (CUB num_items); split the column\n",
            (unsigned long long)n_chunks);
    return 2;
  }
  // Host reference chunk_offsets (exclusive prefix at batch boundaries), for validation.
  std::vector<uint64_t> chunk_off_ref(n_chunks);
  for (uint64_t b = 0; b < n_chunks; ++b) chunk_off_ref[b] = tok_off[b * TOK_PER_BATCH];

  // Host reference decode (concatenate dict_padded[code][0:len]) for byte-exact validation.
  std::vector<uint8_t> cpu_out(decoded_bytes);
  {
    uint64_t cur = 0;
    for (uint64_t i = 0; i < total_tokens; ++i) {
      uint32_t c = codes[i], len = lens[c];
      memcpy(&cpu_out[cur], &dict_padded[(size_t)c * max_token], len);
      cur += len;
    }
  }

  // ── upload ──────────────────────────────────────────────────────────────────
  uint16_t *d_codes;
  uint8_t *d_lens, *d_s8, *d_padded, *d_out;
  uint64_t *d_batchsize, *d_choff, *d_choff_ref;
  CK(cudaMalloc(&d_codes, total_tokens * 2));
  CK(cudaMalloc(&d_lens, dict_size));
  CK(cudaMalloc(&d_s8, dict_s8.size()));
  CK(cudaMalloc(&d_padded, dict_padded.size()));
  CK(cudaMalloc(&d_out, decoded_bytes + 64));
  CK(cudaMalloc(&d_batchsize, n_chunks * 8));
  CK(cudaMalloc(&d_choff, n_chunks * 8));       // GPU-regenerated exclusive prefix
  CK(cudaMalloc(&d_choff_ref, n_chunks * 8));   // host reference, preloaded (OP4 baseline)
  CK(cudaMemcpy(d_codes, codes.data(), total_tokens * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lens, lens.data(), dict_size, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_s8, dict_s8.data(), dict_s8.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_padded, dict_padded.data(), dict_padded.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_choff_ref, chunk_off_ref.data(), n_chunks * 8, cudaMemcpyHostToDevice));

  const uint32_t sz_grid = (uint32_t)((n_chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  const uint32_t dec_grid = sz_grid;  // decode also runs one warp per 128-token chunk

  // CUB exclusive scan over the per-batch sizes (n_chunks fits int for our sizes; note it).
  void *d_tmp = nullptr;
  size_t tmp_bytes = 0;
  CK(cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, d_batchsize, d_choff, (int)n_chunks));
  CK(cudaMalloc(&d_tmp, tmp_bytes));

  auto regen = [&]() {
    batch_sizes_kernel<<<sz_grid, BLOCK_THREADS>>>(d_codes, d_lens, total_tokens, d_batchsize);
    CK(cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_batchsize, d_choff, (int)n_chunks));
  };
  auto decode = [&](uint64_t *choff) {
    onpair_shmem_4tpt_split8read<<<dec_grid, BLOCK_THREADS>>>(
        d_codes, choff, d_s8, d_padded, d_lens, d_out, total_tokens);
  };

  // ── validation: GPU-regen offsets == host ref; decode-with-regen == host decode ──
  regen();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint64_t> choff_gpu(n_chunks);
  CK(cudaMemcpy(choff_gpu.data(), d_choff, n_chunks * 8, cudaMemcpyDeviceToHost));
  bool offsets_ok = (memcmp(choff_gpu.data(), chunk_off_ref.data(), n_chunks * 8) == 0);

  bool decode_ok = false;
  if (DECODE_VALID) {
    CK(cudaMemset(d_out, 0xA5, decoded_bytes + 64));
    decode(d_choff);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::vector<uint8_t> gpu_out(decoded_bytes);
    CK(cudaMemcpy(gpu_out.data(), d_out, decoded_bytes, cudaMemcpyDeviceToHost));
    decode_ok = (memcmp(gpu_out.data(), cpu_out.data(), decoded_bytes) == 0);
  }

  // ── timing (CUDA events; min over iters; raw ns retained) ─────────────────
  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0));
  CK(cudaEventCreate(&ev1));
  auto time_min_raw = [&](auto &&fn, std::vector<uint64_t> &out) -> double {
    for (int w = 0; w < 3; ++w) fn();
    CK(cudaDeviceSynchronize());
    out.clear();
    out.reserve((size_t)iters);
    double best = 1e30;
    for (int k = 0; k < iters; ++k) {
      CK(cudaEventRecord(ev0));
      fn();
      CK(cudaEventRecord(ev1));
      CK(cudaEventSynchronize(ev1));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, ev0, ev1));
      out.push_back((uint64_t)((double)ms * 1e6 + 0.5));
      if (ms < best) best = ms;
    }
    return best;
  };

  std::vector<uint64_t> regen_ns, decode_ns, both_ns;
  double t_regen = time_min_raw(regen, regen_ns);
  double t_decode = 0.0, t_both = 0.0;
  if (DECODE_VALID) {
    t_decode = time_min_raw([&]() { decode(d_choff_ref); }, decode_ns);  // OP4: offsets preloaded
    t_both = time_min_raw([&]() { regen(); decode(d_choff); }, both_ns); // OP2: regen then decode
  }

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  auto gbps = [&](double ms) { return ms > 0.0 ? decoded_bytes / (ms / 1e3) / 1e9 : 0.0; };
  // OP2-vs-OP4 overhead is the END-TO-END comparison (regen+decode vs decode-with-offsets-preloaded),
  // not regen/decode: separately-minimized minima need not add. regen_ms is reported as the component.
  const double overhead_pct = t_decode > 0.0 ? 100.0 * (t_both / t_decode - 1.0) : 0.0;
  auto print_ns = [](const char *k, const std::vector<uint64_t> &v) {
    printf("  \"%s\": [", k);
    for (size_t i = 0; i < v.size(); ++i) printf("%s%llu", i ? "," : "", (unsigned long long)v[i]);
    printf("],\n");
  };

  printf("{\n");
  printf("  \"tok_per_batch\": %u,\n", TOK_PER_BATCH);
  printf("  \"decode_valid\": %s,\n", DECODE_VALID ? "true" : "false");
  printf("  \"gpu\": \"%s\",\n", prop.name);
  printf("  \"kernel\": \"onpair_shmem_4tpt_split8read + gpu-regen-offsets\",\n");
  printf("  \"total_tokens\": %llu,\n", (unsigned long long)total_tokens);
  printf("  \"n_chunks\": %llu,\n", (unsigned long long)n_chunks);
  printf("  \"decoded_bytes\": %llu,\n", (unsigned long long)decoded_bytes);
  printf("  \"iters\": %d,\n", iters);
  printf("  \"offsets_ok\": %s,\n", offsets_ok ? "true" : "false");
  printf("  \"decode_ok\": %s,\n", decode_ok ? "true" : "false");
  printf("  \"regen_ms\": %.5f,\n", t_regen);
  printf("  \"decode_ms\": %.5f,\n", t_decode);
  printf("  \"regen_plus_decode_ms\": %.5f,\n", t_both);
  print_ns("regen_ns_iters", regen_ns);
  print_ns("decode_ns_iters", decode_ns);
  print_ns("regen_plus_decode_ns_iters", both_ns);
  printf("  \"decode_gbps\": %.2f,\n", gbps(t_decode));
  printf("  \"regen_plus_decode_gbps\": %.2f,\n", gbps(t_both));
  printf("  \"regen_overhead_pct\": %.2f\n", overhead_pct);
  printf("}\n");

  fprintf(stderr,
          "\n=== op_gpu_regen (%s) ===\n"
          "%llu tokens, %llu chunks, %.1f MB decoded\n"
          "regen %.3f ms | decode %.3f ms | regen+decode %.3f ms | overhead %.1f%%\n"
          "offsets_ok=%s decode_ok=%s\n",
          prop.name, (unsigned long long)total_tokens, (unsigned long long)n_chunks,
          decoded_bytes / 1e6, t_regen, t_decode, t_both, overhead_pct,
          offsets_ok ? "YES" : "NO", decode_ok ? "YES" : "NO");

  // At a granularity the included decode kernel cannot serve, decode_ok is false BY CONSTRUCTION,
  // so gate the exit code on what was actually measured.
  return (offsets_ok && (decode_ok || !DECODE_VALID)) ? 0 : 4;
}
