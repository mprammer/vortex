// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// fsst_scan.cu -- standalone FSST decode benchmark: naive vs. the OnPair recipe.
//
// Why this exists: the FastPair paper claims its decode recipe (fixed-stride
// table + warp-scan + staged-then-aligned drain) is a general GPU decode
// pattern, not something bespoke to OnPair. This bench substantiates that by
// applying the recipe to UNMODIFIED FSST-compressed data and comparing, on one
// GPU, all on the SAME staged inputs:
//   fsst_naive   -- this tree's thread-per-string FSST decoder (kernels/src/fsst.cu)
//   fsst_recipe  -- the recipe port (kernels/src/fsst_4tpt.cu)
// Both are validated byte-exact against a CPU reference decode, then timed with
// min-of-N CUDA events. The headline external comparison is GSST's published
// 191 GB/s (A100, TPC-H l_comment) — run this on an A100 on l_comment for an
// apples-to-apples decode-throughput number.
//
// Input is a blob dumped by the Rust side (FSST_DUMP in the vortex-fsst
// real-data bench), so the compression is produced by the tree's own FSST
// encoder and the decode here is byte-identical to what Vortex would decode.
// Layout (little-endian):
//   "FST1" | num_strings:u64 | num_input_bytes:u64 | num_symbols:u32
//          | symbols[num_symbols]:u64 | symbol_lengths[num_symbols]:u8
//          | codes_bytes[num_input_bytes]:u8
//          | codes_offsets[num_strings+1]:u64
//          | output_offsets[num_strings+1]:u64
//
// Build (on the GPU box, from this directory):
//   nvcc -O3 -arch=native -std=c++17 fsst_scan.cu -o fsst_scan
// Run:
//   ./fsst_scan <dump.fsstbin> [iters]
// Emits a JSON object on stdout (diagnostics on stderr).

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The naive thread-per-string decoder (defines Scratch, FSSTArgs, and the
// extern "C" fsst_u8/u16/u32/u64 kernels). It #includes config.cuh from its own
// directory, so ELEMENTS_PER_THREAD (=32) matches the Rust launch config.
#include "../../vortex-cuda/kernels/src/fsst.cu"
// The recipe port under test.
#include "../../vortex-cuda/kernels/src/fsst_4tpt.cu"

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e_));                                         \
      exit(3);                                                                 \
    }                                                                          \
  } while (0)

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

template <typename T>
static bool read_vec(FILE *f, std::vector<T> &v, size_t n) {
  v.resize(n);
  return n == 0 || fread(v.data(), sizeof(T), n, f) == n;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <dump.fsstbin> [iters]\n", argv[0]);
    return 1;
  }
  const char *path = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 100;
  if (iters < 3) iters = 3;

  // ── load the dump ──
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }
  char magic[4];
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FST1", 4) != 0) {
    fprintf(stderr, "bad magic in %s\n", path);
    return 2;
  }
  uint64_t num_strings = 0, num_input_bytes = 0;
  uint32_t num_symbols = 0;
  if (fread(&num_strings, 8, 1, f) != 1 ||
      fread(&num_input_bytes, 8, 1, f) != 1 ||
      fread(&num_symbols, 4, 1, f) != 1) {
    fprintf(stderr, "short header\n");
    return 2;
  }
  if (num_strings == 0 || num_input_bytes == 0 || num_symbols == 0 || num_symbols > 255) {
    fprintf(stderr, "invalid dimensions: strings=%llu input=%llu symbols=%u\n",
            (unsigned long long)num_strings,
            (unsigned long long)num_input_bytes, num_symbols);
    return 2;
  }
  // Symbol table: dumped as num_symbols entries; zero-pad to 256 so both
  // kernels can index [0,256) (a valid stream never emits code >= num_symbols).
  std::vector<uint64_t> symbols(256, 0);
  std::vector<uint8_t> symbol_lengths(256, 0);
  std::vector<uint64_t> sym_in;
  std::vector<uint8_t> symlen_in;
  std::vector<uint8_t> codes_bytes;
  std::vector<uint64_t> codes_offsets, output_offsets;
  if (!read_vec(f, sym_in, num_symbols) ||
      !read_vec(f, symlen_in, num_symbols) ||
      !read_vec(f, codes_bytes, num_input_bytes) ||
      !read_vec(f, codes_offsets, num_strings + 1) ||
      !read_vec(f, output_offsets, num_strings + 1)) {
    fprintf(stderr, "short body\n");
    return 2;
  }
  if (fgetc(f) != EOF) {
    fprintf(stderr, "trailing bytes after FST1 payload\n");
    return 2;
  }
  fclose(f);
  for (uint32_t i = 0; i < num_symbols; ++i) {
    if (symlen_in[i] == 0 || symlen_in[i] > 8) {
      fprintf(stderr, "symbol_lengths[%u]=%u outside 1..=8\n",
              i, (unsigned)symlen_in[i]);
      return 2;
    }
    symbols[i] = sym_in[i];
    symbol_lengths[i] = symlen_in[i];
  }
  if (codes_offsets.front() != 0 || codes_offsets.back() != num_input_bytes ||
      !std::is_sorted(codes_offsets.begin(), codes_offsets.end())) {
    fprintf(stderr, "codes_offsets must be monotone from 0 to num_input_bytes\n");
    return 2;
  }
  if (output_offsets.front() != 0 ||
      !std::is_sorted(output_offsets.begin(), output_offsets.end())) {
    fprintf(stderr, "output_offsets must be monotone and start at zero\n");
    return 2;
  }

  // Guard: recipe input offsets are u32 (halves the side-table size). Holds for
  // the paper's ~1 GB columns (compressed < 4 GB); assert so a huge column
  // fails loudly rather than silently truncating.
  if (num_input_bytes >= (1ull << 32)) {
    fprintf(stderr, "num_input_bytes %llu >= 2^32; widen group_in_off to u64\n",
            (unsigned long long)num_input_bytes);
    return 2;
  }

  // ── validate and CPU-decode each string ──
  // Decode within each stored code range so an escape byte cannot consume the
  // first byte of the following string. This is the independent CPU oracle.
  // The dump's per-string output_offsets total is the exact decoded size —
  // reserve to it (falls back to a loose cap if absent).
  const uint64_t decoded_hint =
      output_offsets.empty() ? num_input_bytes * 2 : output_offsets.back();
  std::vector<uint8_t> cpu_out;
  cpu_out.reserve(decoded_hint);
  uint64_t total_codes = 0;
  for (uint64_t row = 0; row < num_strings; ++row) {
    uint64_t in_pos = codes_offsets[row];
    const uint64_t in_end = codes_offsets[row + 1];
    while (in_pos < in_end) {
      const uint8_t code = codes_bytes[in_pos];
      if (code == 255u) {  // escape: next byte is a literal
        if (in_pos + 1 >= in_end) {
          fprintf(stderr, "truncated escape in row %llu at byte %llu\n",
                  (unsigned long long)row, (unsigned long long)in_pos);
          return 2;
        }
        cpu_out.push_back(codes_bytes[in_pos + 1]);
        in_pos += 2;
      } else {
        if (code >= num_symbols) {
          fprintf(stderr, "code %u at byte %llu is outside %u trained symbols\n",
                  (unsigned)code, (unsigned long long)in_pos, num_symbols);
          return 2;
        }
        const uint64_t sym = symbols[code];
        const uint32_t len = symbol_lengths[code];
        const uint8_t *sb = reinterpret_cast<const uint8_t *>(&sym);
        for (uint32_t j = 0; j < len; ++j) cpu_out.push_back(sb[j]);
        in_pos += 1;
      }
      total_codes++;
    }
    if (cpu_out.size() != output_offsets[row + 1]) {
      fprintf(stderr,
              "row %llu decode end %zu != output_offsets end %llu\n",
              (unsigned long long)row, cpu_out.size(),
              (unsigned long long)output_offsets[row + 1]);
      return 2;
    }
  }
  const uint64_t decoded_bytes = cpu_out.size();

  if (output_offsets.back() != decoded_bytes) {
    fprintf(stderr, "decoded bytes do not match output_offsets sentinel\n");
    return 2;
  }

  // Build the recipe-only metadata in a separate measured host pass. The CUDA
  // event timings below remain kernel-only; this cost is surfaced explicitly
  // rather than silently giving the recipe free preprocessing.
  const double metadata_t0 = now_s();
  std::vector<uint32_t> group_in_off;
  std::vector<uint64_t> batch_out_off;
  uint64_t code_idx = 0, decoded_pos = 0, in_pos = 0;
  while (in_pos < num_input_bytes) {
    if ((code_idx & 127u) == 0) batch_out_off.push_back(decoded_pos);
    if ((code_idx & 3u) == 0) group_in_off.push_back((uint32_t)in_pos);
    const uint8_t code = codes_bytes[in_pos];
    if (code == 255u) {
      decoded_pos += 1;
      in_pos += 2;
    } else {
      decoded_pos += symbol_lengths[code];
      in_pos += 1;
    }
    code_idx++;
  }
  batch_out_off.push_back(decoded_pos);
  if (code_idx != total_codes || decoded_pos != decoded_bytes) {
    fprintf(stderr, "recipe metadata pass disagrees with validated CPU decode\n");
    return 2;
  }
  const uint64_t num_batches = (total_codes + 127) / 128;
  // Pad group_in_off so every possible (batch,lane) group index is in-bounds;
  // trailing lanes past total_codes are inactive but still read their slot.
  const size_t groups_needed = (size_t)num_batches * 32 + 1;
  while (group_in_off.size() < groups_needed)
    group_in_off.push_back((uint32_t)num_input_bytes);
  const double recipe_metadata_build_ms = (now_s() - metadata_t0) * 1e3;

  // Validity: all strings valid.
  std::vector<uint8_t> validity_bits((num_strings + 7) / 8, 0xFF);

  const uint64_t compressed_bytes =
      num_input_bytes + (uint64_t)num_symbols * 8 + num_symbols;
  const uint64_t naive_metadata_bytes =
      (num_strings + 1) * sizeof(uint64_t) * 2 + validity_bits.size();
  const uint64_t recipe_metadata_bytes =
      group_in_off.size() * sizeof(uint32_t) + batch_out_off.size() * sizeof(uint64_t);
  const uint64_t naive_staged_input_bytes =
      num_input_bytes + 256 * sizeof(uint64_t) + 256 + naive_metadata_bytes;
  const uint64_t recipe_staged_input_bytes =
      num_input_bytes + 256 * sizeof(uint64_t) + 256 + recipe_metadata_bytes;

  // ── upload to device ──
  uint8_t *d_codes, *d_symlen, *d_out_naive, *d_out_recipe, *d_valid;
  uint64_t *d_symbols, *d_codes_off, *d_out_off, *d_batch_out_off;
  uint32_t *d_group_in_off;
  CK(cudaMalloc(&d_codes, num_input_bytes));
  CK(cudaMalloc(&d_symbols, 256 * 8));
  CK(cudaMalloc(&d_symlen, 256));
  CK(cudaMalloc(&d_codes_off, (num_strings + 1) * 8));
  CK(cudaMalloc(&d_out_off, (num_strings + 1) * 8));
  CK(cudaMalloc(&d_valid, validity_bits.size()));
  CK(cudaMalloc(&d_group_in_off, group_in_off.size() * 4));
  CK(cudaMalloc(&d_batch_out_off, batch_out_off.size() * 8));
  CK(cudaMalloc(&d_out_naive, decoded_bytes + 64));
  CK(cudaMalloc(&d_out_recipe, decoded_bytes + 64));
  CK(cudaMemcpy(d_codes, codes_bytes.data(), num_input_bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_symbols, symbols.data(), 256 * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_symlen, symbol_lengths.data(), 256, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_codes_off, codes_offsets.data(), (num_strings + 1) * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_out_off, output_offsets.data(), (num_strings + 1) * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_valid, validity_bits.data(), validity_bits.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_group_in_off, group_in_off.data(), group_in_off.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_batch_out_off, batch_out_off.data(), batch_out_off.size() * 8, cudaMemcpyHostToDevice));

  // ── launch geometry ──
  // Naive: matches the Rust config (THREADS_PER_BLOCK=64, ELEMENTS_PER_THREAD=32).
  const uint32_t naive_block = 64;
  const uint64_t naive_epb = (uint64_t)naive_block * ELEMENTS_PER_THREAD;  // 2048
  dim3 naive_grid((unsigned)((num_strings + naive_epb - 1) / naive_epb));
  dim3 naive_blk(naive_block);
  // Recipe: 512 threads = 16 warps, one warp per 128-code batch.
  dim3 recipe_blk(512);
  dim3 recipe_grid((unsigned)((num_batches + 15) / 16));

  auto launch_naive = [&]() {
    fsst_u64<<<naive_grid, naive_blk>>>(d_codes, d_codes_off, d_symbols, d_symlen,
                                        d_out_off, d_valid, d_out_naive, num_strings);
  };
  auto launch_recipe = [&]() {
    fsst_4tpt<<<recipe_grid, recipe_blk>>>(d_codes, d_group_in_off, d_batch_out_off,
                                           d_symbols, d_symlen, d_out_recipe, total_codes);
  };

  // ── correctness: decode once each, compare to the CPU reference ──
  CK(cudaMemset(d_out_naive, 0, decoded_bytes + 64));
  launch_naive();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint8_t> h_naive(decoded_bytes);
  CK(cudaMemcpy(h_naive.data(), d_out_naive, decoded_bytes, cudaMemcpyDeviceToHost));
  bool naive_ok = (memcmp(h_naive.data(), cpu_out.data(), decoded_bytes) == 0);

  CK(cudaMemset(d_out_recipe, 0, decoded_bytes + 64));
  launch_recipe();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint8_t> h_recipe(decoded_bytes);
  CK(cudaMemcpy(h_recipe.data(), d_out_recipe, decoded_bytes, cudaMemcpyDeviceToHost));
  bool recipe_ok = (memcmp(h_recipe.data(), cpu_out.data(), decoded_bytes) == 0);

  // Report the first mismatch to make debugging a failed port tractable.
  auto first_mismatch = [&](const std::vector<uint8_t> &got) -> long long {
    for (uint64_t i = 0; i < decoded_bytes; ++i)
      if (got[i] != cpu_out[i]) return (long long)i;
    return -1;
  };
  long long naive_mm = naive_ok ? -1 : first_mismatch(h_naive);
  long long recipe_mm = recipe_ok ? -1 : first_mismatch(h_recipe);
  if (!naive_ok || !recipe_ok) {
    fprintf(stderr,
            "validation failed before timing: naive_ok=%s mismatch=%lld recipe_ok=%s mismatch=%lld\n",
            naive_ok ? "true" : "false", naive_mm,
            recipe_ok ? "true" : "false", recipe_mm);
    return 4;
  }

  // ── timing (CUDA events, min over iters; raw per-iter ns retained) ──
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

  std::vector<uint64_t> naive_ns_iters, recipe_ns_iters;
  double t_naive = time_min_raw([&]() { launch_naive(); }, naive_ns_iters);
  double t_recipe = time_min_raw([&]() { launch_recipe(); }, recipe_ns_iters);

  auto gbps = [](uint64_t bytes, double ms) { return bytes / (ms / 1e3) / 1e9; };

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));

  auto print_ns_iters = [](const char *key, const std::vector<uint64_t> &v) {
    printf("  \"%s\": [", key);
    for (size_t i = 0; i < v.size(); ++i)
      printf("%s%llu", i ? "," : "", (unsigned long long)v[i]);
    printf("],\n");
  };

  printf("{\n");
  printf("  \"gpu\": \"%s\",\n", prop.name);
  printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
  printf("  \"codec\": \"fsst\",\n");
  printf("  \"num_strings\": %llu,\n", (unsigned long long)num_strings);
  printf("  \"num_codes\": %llu,\n", (unsigned long long)total_codes);
  printf("  \"num_input_bytes\": %llu,\n", (unsigned long long)num_input_bytes);
  printf("  \"num_symbols\": %u,\n", num_symbols);
  printf("  \"decoded_bytes\": %llu,\n", (unsigned long long)decoded_bytes);
  printf("  \"compressed_bytes\": %llu,\n", (unsigned long long)compressed_bytes);
  printf("  \"naive_metadata_bytes\": %llu,\n", (unsigned long long)naive_metadata_bytes);
  printf("  \"recipe_metadata_bytes\": %llu,\n", (unsigned long long)recipe_metadata_bytes);
  printf("  \"naive_staged_input_bytes\": %llu,\n", (unsigned long long)naive_staged_input_bytes);
  printf("  \"recipe_staged_input_bytes\": %llu,\n", (unsigned long long)recipe_staged_input_bytes);
  printf("  \"recipe_metadata_build_ms\": %.5f,\n", recipe_metadata_build_ms);
  printf("  \"timing_scope\": \"kernel_only_metadata_prebuilt\",\n");
  printf("  \"ratio\": %.4f,\n", (double)decoded_bytes / compressed_bytes);
  printf("  \"iters\": %d,\n", iters);
  printf("  \"naive_block_threads\": %u,\n", naive_block);
  printf("  \"recipe_block_threads\": %u,\n", recipe_blk.x);
  printf("  \"naive_ok\": %s,\n", naive_ok ? "true" : "false");
  printf("  \"recipe_ok\": %s,\n", recipe_ok ? "true" : "false");
  printf("  \"naive_first_mismatch\": %lld,\n", naive_mm);
  printf("  \"recipe_first_mismatch\": %lld,\n", recipe_mm);
  printf("  \"naive_ms\": %.5f,\n", t_naive);
  printf("  \"recipe_ms\": %.5f,\n", t_recipe);
  print_ns_iters("naive_ns_iters", naive_ns_iters);
  print_ns_iters("recipe_ns_iters", recipe_ns_iters);
  printf("  \"naive_gbps\": %.2f,\n", gbps(decoded_bytes, t_naive));
  printf("  \"recipe_gbps\": %.2f,\n", gbps(decoded_bytes, t_recipe));
  printf("  \"recipe_speedup_over_naive\": %.3f,\n", t_naive / t_recipe);
  printf("  \"gsst_published_gbps\": 191.0,\n");
  printf("  \"recipe_vs_gsst\": %.3f\n", gbps(decoded_bytes, t_recipe) / 191.0);
  printf("}\n");

  fprintf(stderr,
          "\n=== fsst_scan summary (%s) ===\n"
          "decoded=%.1f MB  ratio=%.2fx  codes=%.1fM  symbols=%u\n"
          "naive=%.1f GB/s  recipe=%.1f GB/s  (recipe %.2fx naive; GSST pub=191 GB/s)\n"
          "naive_ok=%s recipe_ok=%s%s\n",
          prop.name, decoded_bytes / 1e6, (double)decoded_bytes / compressed_bytes,
          total_codes / 1e6, num_symbols, gbps(decoded_bytes, t_naive),
          gbps(decoded_bytes, t_recipe), t_naive / t_recipe,
          naive_ok ? "YES" : "NO", recipe_ok ? "YES" : "NO",
          (naive_ok && recipe_ok)
              ? ""
              : (recipe_ok ? "  (naive mismatch!)" : "  (recipe MISMATCH!)"));
  return (naive_ok && recipe_ok) ? 0 : 4;
}
