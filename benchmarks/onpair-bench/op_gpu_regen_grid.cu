// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// op_gpu_regen_grid.cu -- the OP2 regeneration penalty at the SHIPPED launch configuration.
//
// WHY THIS EXISTS ALONGSIDE op_gpu_regen.cu. That bench answers the same question but can only
// answer it at K=4: it includes onpair_shmem_4tpt_split8read, which hardcodes a 128-token batch,
// so -DTOK_PER_BATCH_OVERRIDE=192 measures regeneration alone and reports null for the decode and
// the overhead (its own header says so). The published +15-19% is therefore a K=4 number, while
// the paper's recommended configuration is T=256, B=4, K=6. The penalty is a RATIO -- regen+decode
// over decode -- and both terms move with K, so it cannot be inferred from the K=4 ratio plus the
// separately measured -14.7% regen-at-192 (b300-regen-gran, 2026-08-25). This file closes that gap
// by decoding with the generated grid kernel whose batch actually matches the granularity.
//
// WHAT IS PARAMETERIZED. The decode kernel is chosen at compile time, and TOK_PER_BATCH is derived
// from it rather than set independently -- the two disagreeing is exactly the bug the reduction
// comment in op_gpu_regen.cu records:
//   -DREGEN_KERNEL_HEADER='"...onpair_dg_k6_t256_b4.cu"'  the translation unit to include
//   -DREGEN_KERNEL=onpair_dg_k6_t256_b4                   its extern "C" entry point
//   -DREGEN_K=6                                           tokens per thread; batch is 32*K
//   -DREGEN_BLOCK_THREADS=256                             must equal the kernel's launch bound
//   -DONPAIR_LOW_PLANE_BYTES=8|16                         W, read by the included header
// A mismatch between REGEN_K and the kernel's own TOKENS_PER_THREAD is a compile error, not a
// silent wrong answer (static_assert below).
//
// ABI. The generated kernels take a split dictionary and nibble-packed lengths, which the 4tpt
// kernel did not: (codes, chunk_offsets, dict_s8_lo, dict_s8_hi, packed_lens, out, total_tokens).
// dict_s8_lo has stride W; dict_s8_hi has stride 8 always (it is read as a uint2) and holds bytes
// [8,16) of each entry, so it is unused at W=16. packed_lens stores length-1 in a nibble, even
// codes in the low half. All three are derived here from the dump's dict_padded + lens.
//
// Input: the same "E2E1" dump onpair_bench.rs writes via ONPAIR_DUMP_E2E.
// Run:   ./op_gpu_regen_grid <dump.e2ebin> [iters]
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

// The decode kernel under test, chosen by the build. Its own translation unit defines
// TOKENS_PER_THREAD / ONPAIR_BLOCK_THREADS and includes onpair_decompress_tpt.cuh, which reads
// ONPAIR_LOW_PLANE_BYTES from the command line.
#ifndef REGEN_KERNEL_HEADER
#define REGEN_KERNEL_HEADER "../../vortex-cuda/kernels/src/onpair_dg_k6_t256_b4.cu"
#endif
#ifndef REGEN_KERNEL
#define REGEN_KERNEL onpair_dg_k6_t256_b4
#endif
#ifndef REGEN_K
#define REGEN_K 6
#endif
#ifndef REGEN_BLOCK_THREADS
#define REGEN_BLOCK_THREADS 256
#endif
#include REGEN_KERNEL_HEADER
#define REGEN_STR2(x) #x
#define REGEN_STR(x) REGEN_STR2(x)
#define REGEN_KERNEL_NAME_STR REGEN_STR(REGEN_KERNEL)

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e_));                                         \
      exit(3);                                                                 \
    }                                                                          \
  } while (0)

// BATCH GRANULARITY IS DERIVED, NOT DECLARED. The sidecar holds one offset per warp batch of
// 32*K codes, so the granularity IS the decode kernel's K. Deriving it from TOKENS_PER_THREAD --
// which the included kernel defines -- makes the two impossible to set inconsistently.
static_assert(TOKENS_PER_THREAD == (uint32_t)REGEN_K,
              "REGEN_K disagrees with the included kernel's TOKENS_PER_THREAD");
static_assert(ONPAIR_BLOCK_THREADS == (uint32_t)REGEN_BLOCK_THREADS,
              "REGEN_BLOCK_THREADS disagrees with the included kernel's launch bound");
static constexpr uint32_t TOK_PER_BATCH = 32u * (uint32_t)REGEN_K;

// The reduction that rebuilds the per-batch sizes is independent of the decode kernel's launch
// shape, so it keeps its own (wider) block. Only the decode grid must match the kernel.
static constexpr int BLOCK_THREADS = 512;
static constexpr uint32_t WARPS_PER_BLOCK = BLOCK_THREADS / 32;
static constexpr int DEC_BLOCK_THREADS = REGEN_BLOCK_THREADS;
static constexpr uint32_t DEC_WARPS_PER_BLOCK = DEC_BLOCK_THREADS / 32;

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
  // ROUNDS, not 4. A warp covers TOK_PER_BATCH tokens at 32 lanes per round, so hardcoding 4
  // summed 128 tokens of every batch whatever the granularity: at 192 that produced offsets the
  // host reference rejected (offsets_ok=NO) and a regen time ~30% low, because two thirds of the
  // codes were being read. The validator caught it; the timing alone would have looked like a win.
  constexpr int ROUNDS = (int)(TOK_PER_BATCH / 32u);
  static_assert(TOK_PER_BATCH % 32u == 0u, "TOK_PER_BATCH must be a whole number of warp rounds");
  uint32_t s = 0;
#pragma unroll
  for (int k = 0; k < ROUNDS; ++k) {
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

  // ── host derivation: reference offsets, split dictionary planes, reference decode ──
  std::vector<uint64_t> tok_off(total_tokens + 1);
  tok_off[0] = 0;
  for (uint64_t i = 0; i < total_tokens; ++i) tok_off[i + 1] = tok_off[i] + lens[codes[i]];
  const uint64_t decoded_bytes = tok_off[total_tokens];
  if (decoded_bytes == 0) { fprintf(stderr, "dump decodes to 0 bytes; unsupported\n"); return 2; }

  // Split planes. lo has stride W and holds the first min(len, W) bytes; hi has stride 8 always
  // (the kernel reads it as a uint2) and holds bytes [8,16), so it stays zero-filled at W=16.
  constexpr uint32_t W = ONPAIR_LOW_PLANE_BYTES;
  std::vector<uint8_t> dict_lo((size_t)dict_size * W, 0);
  std::vector<uint8_t> dict_hi((size_t)dict_size * 8, 0);
  for (uint32_t c = 0; c < dict_size; ++c) {
    const uint8_t *e = &dict_padded[(size_t)c * max_token];
    const uint32_t nlo = lens[c] < W ? lens[c] : W;
    memcpy(&dict_lo[(size_t)c * W], e, nlo);
    if (lens[c] > 8u && W == 8u) memcpy(&dict_hi[(size_t)c * 8], e + 8, lens[c] - 8u);
  }
  // Nibble-packed lengths, length-1 so 16 fits in four bits; even codes take the low nibble.
  // A zero-length entry cannot be represented and would decode as length 1, so reject it here
  // rather than silently emitting a wrong byte.
  for (uint32_t c = 0; c < dict_size; ++c) {
    if (lens[c] == 0) { fprintf(stderr, "dict entry %u has length 0; unsupported\n", c); return 2; }
  }
  std::vector<uint8_t> packed_lens(((size_t)dict_size + 1) / 2, 0);
  for (uint32_t c = 0; c < dict_size; ++c) {
    packed_lens[c >> 1] |= (uint8_t)((lens[c] - 1u) << ((c & 1u) * 4u));
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
  uint8_t *d_lens, *d_lo, *d_hi, *d_plens, *d_out;
  uint64_t *d_batchsize, *d_choff, *d_choff_ref;
  CK(cudaMalloc(&d_codes, total_tokens * 2));
  CK(cudaMalloc(&d_lens, dict_size));
  CK(cudaMalloc(&d_lo, dict_lo.size()));
  CK(cudaMalloc(&d_hi, dict_hi.size()));
  CK(cudaMalloc(&d_plens, packed_lens.size()));
  CK(cudaMalloc(&d_out, decoded_bytes + 64));
  CK(cudaMalloc(&d_batchsize, n_chunks * 8));
  CK(cudaMalloc(&d_choff, n_chunks * 8));       // GPU-regenerated exclusive prefix
  CK(cudaMalloc(&d_choff_ref, n_chunks * 8));   // host reference, preloaded (OP4 baseline)
  CK(cudaMemcpy(d_codes, codes.data(), total_tokens * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lens, lens.data(), dict_size, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lo, dict_lo.data(), dict_lo.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_hi, dict_hi.data(), dict_hi.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_plens, packed_lens.data(), packed_lens.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_choff_ref, chunk_off_ref.data(), n_chunks * 8, cudaMemcpyHostToDevice));

  const uint32_t sz_grid = (uint32_t)((n_chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  // One warp per batch on both sides, but the two use different block widths.
  const uint32_t dec_grid = (uint32_t)((n_chunks + DEC_WARPS_PER_BLOCK - 1) / DEC_WARPS_PER_BLOCK);

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
    REGEN_KERNEL<<<dec_grid, DEC_BLOCK_THREADS>>>(
        d_codes, choff, d_lo, d_hi, d_plens, d_out, total_tokens);
  };

  // ── validation: GPU-regen offsets == host ref; decode-with-regen == host decode ──
  regen();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint64_t> choff_gpu(n_chunks);
  CK(cudaMemcpy(choff_gpu.data(), d_choff, n_chunks * 8, cudaMemcpyDeviceToHost));
  bool offsets_ok = (memcmp(choff_gpu.data(), chunk_off_ref.data(), n_chunks * 8) == 0);

  // Unlike op_gpu_regen.cu there is no granularity at which the decode is out of scope: the
  // kernel's batch and the sidecar's batch are the same constant by construction.
  bool decode_ok = false;
  {
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
  double t_decode = time_min_raw([&]() { decode(d_choff_ref); }, decode_ns);  // OP4: preloaded
  double t_both = time_min_raw([&]() { regen(); decode(d_choff); }, both_ns);  // OP2: regen+decode

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
  printf("  \"k\": %u,\n", (unsigned)TOKENS_PER_THREAD);
  printf("  \"block_threads\": %u,\n", (unsigned)ONPAIR_BLOCK_THREADS);
  printf("  \"low_plane_bytes\": %u,\n", (unsigned)ONPAIR_LOW_PLANE_BYTES);
  printf("  \"dict_size\": %u,\n", dict_size);
  printf("  \"decode_valid\": true,\n");
  printf("  \"gpu\": \"%s\",\n", prop.name);
  printf("  \"kernel\": \"%s + gpu-regen-offsets\",\n", REGEN_KERNEL_NAME_STR);
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
  return (offsets_ok && decode_ok) ? 0 : 4;
}
