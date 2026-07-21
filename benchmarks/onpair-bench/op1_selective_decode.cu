// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// op1_selective_decode.cu -- OP1 EARLY-materialization SELECTIVE decode bench.
//
// This is the selective sibling of op1_row_decode.cu. Where op1_row_decode.cu decodes
// EVERY row (thread-per-row, dense), this one models an upstream filter that has already
// selected a fraction `m` of the rows, and decodes ONLY the survivors via a COMPACTED
// survivor row-id list (r = row_ids[tid], grid sized to the survivor count) -- the
// "take-on-codes" / early-materialization path. The survivor list is built by stream
// compaction (CUB DeviceSelect::Flagged), whose cost is TIMED as part of OP1's price
// (per the experiment spec: "include the survivor-list construction as a counted cost").
//
// Because early-mat COMPACTS (survivor rows land back-to-back, no gaps), it produces a
// dense output but breaks the stored chunk_offsets -- the contrast the paper draws
// against masked-OP4 late-materialization (which preserves them).
//
// The per-row survivor selection is a DETERMINISTIC seeded PRNG (splitmix64) at rate m,
// so counts are reproducible run-to-run (deterministic Vikram rerun). m=100% selects
// every row and recovers op1_row_decode.cu's dense cost (plus the compaction tax).
//
// Input: the "OP11" dump onpair_bench.rs writes via ONPAIR_DUMP_OP1
//   "OP11" | total_tokens:u64 | n_rows:u64 | dict_size:u32 | max_token:u32
//     | codes(u16 LE) | row_offsets(u64 LE, n_rows+1) | lens(u8) | dict_padded(u8)
// Emits one JSON object on stdout; exit 0 iff offsets + decode are byte-exact.
//
// Build (on the GPU box, from this directory):
//   nvcc -O3 -arch=native -std=c++17 op1_selective_decode.cu -o op1_selective_decode
// Run:
//   ./op1_selective_decode <dump.op1bin> [selectivity_pct] [iters] [seed]
//     selectivity_pct default 100 (dense); e.g. 10, 1, 0.1, 0.01

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>   // CUDA 13 removed cub::CountingInputIterator
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include "op1_selective_kernels.cuh"

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e_));                                         \
      exit(3);                                                                 \
    }                                                                          \
  } while (0)

static constexpr uint32_t MAX_TOKEN = 16;
static constexpr int BLOCK_THREADS = 256;

// splitmix64 -- deterministic, seedable; one draw per row decides survival.
static inline uint64_t splitmix64(uint64_t &s) {
  uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <dump.op1bin> [selectivity_pct] [iters] [seed]\n", argv[0]);
    return 1;
  }
  const char *path = argv[1];
  double sel_pct = argc > 2 ? atof(argv[2]) : 100.0;
  int iters = argc > 3 ? atoi(argv[3]) : 100;
  uint64_t seed = argc > 4 ? strtoull(argv[4], nullptr, 10) : 0xC0FFEE1234567890ULL;
  if (iters < 3) iters = 3;
  if (sel_pct < 0.0) sel_pct = 0.0;
  if (sel_pct > 100.0) sel_pct = 100.0;
  const double m = sel_pct / 100.0;

  // ── load the OP11 dump ──────────────────────────────────────────────────────
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
  char magic[4];
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OP11", 4) != 0) {
    fprintf(stderr, "bad magic in %s (want OP11)\n", path); return 2;
  }
  uint64_t total_tokens = 0, n_rows = 0;
  uint32_t dict_size = 0, max_token = 0;
  if (fread(&total_tokens, 8, 1, f) != 1 || fread(&n_rows, 8, 1, f) != 1 ||
      fread(&dict_size, 4, 1, f) != 1 || fread(&max_token, 4, 1, f) != 1) {
    fprintf(stderr, "short header\n"); return 2;
  }
  if (max_token != MAX_TOKEN) {
    fprintf(stderr, "max_token=%u but this bench assumes %u\n", max_token, MAX_TOKEN); return 2;
  }
  if (dict_size == 0 || total_tokens == 0 || n_rows == 0) {
    fprintf(stderr, "empty/degenerate dump; skipping\n"); return 2;
  }
  if (n_rows > (uint64_t)INT_MAX) {
    fprintf(stderr, "n_rows=%llu exceeds INT_MAX (CUB num_items); split the column\n",
            (unsigned long long)n_rows); return 2;
  }
  std::vector<uint16_t> codes(total_tokens);
  std::vector<uint64_t> row_off(n_rows + 1);
  std::vector<uint8_t> lens(dict_size);
  std::vector<uint8_t> dict_padded((size_t)dict_size * max_token);
  if (fread(codes.data(), 2, total_tokens, f) != total_tokens ||
      fread(row_off.data(), 8, n_rows + 1, f) != n_rows + 1 ||
      fread(lens.data(), 1, dict_size, f) != dict_size ||
      fread(dict_padded.data(), 1, dict_padded.size(), f) != dict_padded.size()) {
    fprintf(stderr, "short body\n"); return 2;
  }
  fclose(f);

  for (uint16_t c : codes) if (c >= dict_size) { fprintf(stderr, "code %u out of range\n", (unsigned)c); return 2; }
  for (uint8_t l : lens) if (l > max_token) { fprintf(stderr, "dict length %u exceeds max_token %u\n", (unsigned)l, max_token); return 2; }
  if (row_off[0] != 0 || row_off[n_rows] != total_tokens) {
    fprintf(stderr, "row_off framing bad\n"); return 2;
  }
  for (uint64_t r = 0; r < n_rows; ++r)
    if (row_off[r + 1] < row_off[r] || row_off[r + 1] > total_tokens) {
      fprintf(stderr, "row_off not monotone at row %llu\n", (unsigned long long)r); return 2;
    }

  // ── deterministic per-row survivor flags at rate m (seeded splitmix64) ──
  // A row survives iff its uniform draw < m. Same seed => same survivor set every run
  // and across techniques (the driver uses the identical row-selection rule).
  std::vector<uint8_t> flags(n_rows);
  uint64_t s = seed;
  uint64_t n_sel_host = 0;
  for (uint64_t r = 0; r < n_rows; ++r) {
    double u = (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);  // [0,1)
    uint8_t keep = (u < m) ? 1u : 0u;
    flags[r] = keep;
    n_sel_host += keep;
  }
  if (n_sel_host == 0) {
    // At tiny m on a small column no row may survive: still a legal cell, but nothing
    // to decode. Report and exit cleanly (the driver enumerates larger inputs).
    fprintf(stderr, "no survivors at m=%.4g on n_rows=%llu; nothing to decode\n",
            m, (unsigned long long)n_rows);
    return 2;
  }

  // ── host reference: compacted survivor row-id list, out offsets, byte-exact decode ──
  std::vector<uint32_t> row_ids_ref;
  row_ids_ref.reserve(n_sel_host);
  for (uint64_t r = 0; r < n_rows; ++r) if (flags[r]) row_ids_ref.push_back((uint32_t)r);
  const uint64_t n_sel = row_ids_ref.size();

  std::vector<uint64_t> out_off_ref(n_sel + 1);
  out_off_ref[0] = 0;
  for (uint64_t i = 0; i < n_sel; ++i) {
    const uint64_t r = row_ids_ref[i];
    uint64_t rb = 0;
    for (uint64_t t = row_off[r]; t < row_off[r + 1]; ++t) rb += lens[codes[t]];
    out_off_ref[i + 1] = out_off_ref[i] + rb;
  }
  const uint64_t decoded_bytes = out_off_ref[n_sel];
  if (decoded_bytes == 0) { fprintf(stderr, "survivors decode to 0 bytes; unsupported\n"); return 2; }

  std::vector<uint8_t> cpu_out(decoded_bytes);
  {
    uint64_t cur = 0;
    for (uint64_t i = 0; i < n_sel; ++i) {
      const uint64_t r = row_ids_ref[i];
      for (uint64_t t = row_off[r]; t < row_off[r + 1]; ++t) {
        uint32_t c = codes[t], len = lens[c];
        if (len) memcpy(&cpu_out[cur], &dict_padded[(size_t)c * max_token], len);
        cur += len;
      }
    }
  }

  // ── upload ──────────────────────────────────────────────────────────────────
  uint16_t *d_codes;
  uint64_t *d_row_off, *d_rowsize, *d_out_off;
  uint8_t *d_lens, *d_padded, *d_out, *d_flags;
  uint32_t *d_row_ids;
  int *d_num_sel;
  CK(cudaMalloc(&d_codes, total_tokens * 2));
  CK(cudaMalloc(&d_row_off, (n_rows + 1) * 8));
  CK(cudaMalloc(&d_rowsize, n_sel * 8));
  CK(cudaMalloc(&d_out_off, (n_sel + 1) * 8));
  CK(cudaMalloc(&d_lens, dict_size));
  CK(cudaMalloc(&d_padded, dict_padded.size()));
  CK(cudaMalloc(&d_out, decoded_bytes + 64));
  CK(cudaMalloc(&d_flags, n_rows));
  CK(cudaMalloc(&d_row_ids, n_rows * sizeof(uint32_t)));   // upper bound = n_rows
  CK(cudaMalloc(&d_num_sel, sizeof(int)));
  CK(cudaMemcpy(d_codes, codes.data(), total_tokens * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_row_off, row_off.data(), (n_rows + 1) * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lens, lens.data(), dict_size, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_padded, dict_padded.data(), dict_padded.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_flags, flags.data(), n_rows, cudaMemcpyHostToDevice));

  // CUB stream-compaction: select the row indices whose flag is set, into d_row_ids.
  // Input indices come from a counting iterator (no materialized iota).
  thrust::counting_iterator<uint32_t> iota(0);   // CUB DeviceSelect input iterator (CUDA 13: thrust, not cub)
  void *d_tmp_sel = nullptr; size_t tmp_sel_bytes = 0;
  CK(cub::DeviceSelect::Flagged(nullptr, tmp_sel_bytes, iota, d_flags, d_row_ids,
                                d_num_sel, (int)n_rows));
  CK(cudaMalloc(&d_tmp_sel, tmp_sel_bytes));

  // CUB exclusive scan of per-survivor sizes -> survivor output offsets.
  void *d_tmp_scan = nullptr; size_t tmp_scan_bytes = 0;
  CK(cub::DeviceScan::ExclusiveSum(nullptr, tmp_scan_bytes, d_rowsize, d_out_off, (int)n_sel));
  CK(cudaMalloc(&d_tmp_scan, tmp_scan_bytes));

  const uint32_t grid = (uint32_t)((n_sel + BLOCK_THREADS - 1) / BLOCK_THREADS);

  // OP1 legs (all TIMED as OP1's price):
  auto compact = [&]() {  // stream-compaction: build the survivor row-id list
    cub::DeviceSelect::Flagged(d_tmp_sel, tmp_sel_bytes, iota, d_flags, d_row_ids,
                               d_num_sel, (int)n_rows);
  };
  auto rowgen = [&]() {   // survivor decoded-size reduction + exclusive scan + terminal
    sel_row_sizes_kernel<<<grid, BLOCK_THREADS>>>(d_codes, d_row_off, d_lens, d_row_ids, n_sel, d_rowsize);
    cub::DeviceScan::ExclusiveSum(d_tmp_scan, tmp_scan_bytes, d_rowsize, d_out_off, (int)n_sel);
    sel_write_terminal_kernel<<<1, 32>>>(d_out_off, d_rowsize, n_sel);
  };
  auto decode = [&]() {   // thread-per-survivor decode into the compacted buffer
    sel_row_decode_kernel<<<grid, BLOCK_THREADS>>>(d_codes, d_row_off, d_out_off, d_padded,
                                                   d_lens, d_row_ids, n_sel, max_token, d_out);
  };

  // ── validation: compaction count + offsets + decode all byte-exact vs host ──
  compact();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  int num_sel_gpu = 0;
  CK(cudaMemcpy(&num_sel_gpu, d_num_sel, sizeof(int), cudaMemcpyDeviceToHost));
  bool compact_ok = ((uint64_t)num_sel_gpu == n_sel);
  std::vector<uint32_t> row_ids_gpu(n_sel);
  CK(cudaMemcpy(row_ids_gpu.data(), d_row_ids, n_sel * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  compact_ok = compact_ok && (memcmp(row_ids_gpu.data(), row_ids_ref.data(), n_sel * sizeof(uint32_t)) == 0);

  rowgen();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint64_t> out_off_gpu(n_sel + 1);
  CK(cudaMemcpy(out_off_gpu.data(), d_out_off, (n_sel + 1) * 8, cudaMemcpyDeviceToHost));
  bool offsets_ok = (memcmp(out_off_gpu.data(), out_off_ref.data(), (n_sel + 1) * 8) == 0);

  CK(cudaMemset(d_out, 0xA5, decoded_bytes + 64));
  decode();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint8_t> gpu_out(decoded_bytes);
  CK(cudaMemcpy(gpu_out.data(), d_out, decoded_bytes, cudaMemcpyDeviceToHost));
  bool decode_ok = (memcmp(gpu_out.data(), cpu_out.data(), decoded_bytes) == 0);

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

  std::vector<uint64_t> compact_ns, rowgen_ns, decode_ns, all_ns;
  double t_compact = time_min_raw(compact, compact_ns);
  double t_rowgen = time_min_raw(rowgen, rowgen_ns);
  double t_decode = time_min_raw(decode, decode_ns);
  double t_all = time_min_raw([&]() { compact(); rowgen(); decode(); }, all_ns);

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  auto gbps = [&](double ms) { return ms > 0.0 ? decoded_bytes / (ms / 1e3) / 1e9 : 0.0; };
  auto print_ns = [](const char *k, const std::vector<uint64_t> &v) {
    printf("  \"%s\": [", k);
    for (size_t i = 0; i < v.size(); ++i) printf("%s%llu", i ? "," : "", (unsigned long long)v[i]);
    printf("],\n");
  };

  printf("{\n");
  printf("  \"gpu\": \"%s\",\n", prop.name);
  printf("  \"kernel\": \"op1_selective_decode (thread-per-survivor-row + compaction + gpu-offset-scan)\",\n");
  printf("  \"total_tokens\": %llu,\n", (unsigned long long)total_tokens);
  printf("  \"n_rows\": %llu,\n", (unsigned long long)n_rows);
  printf("  \"selectivity_pct\": %.6g,\n", sel_pct);
  printf("  \"seed\": %llu,\n", (unsigned long long)seed);
  printf("  \"n_sel\": %llu,\n", (unsigned long long)n_sel);
  printf("  \"decoded_bytes\": %llu,\n", (unsigned long long)decoded_bytes);
  printf("  \"iters\": %d,\n", iters);
  printf("  \"compact_ok\": %s,\n", compact_ok ? "true" : "false");
  printf("  \"offsets_ok\": %s,\n", offsets_ok ? "true" : "false");
  printf("  \"decode_ok\": %s,\n", decode_ok ? "true" : "false");
  printf("  \"compact_ms\": %.5f,\n", t_compact);
  printf("  \"rowgen_ms\": %.5f,\n", t_rowgen);
  printf("  \"decode_ms\": %.5f,\n", t_decode);
  printf("  \"compact_plus_rowgen_plus_decode_ms\": %.5f,\n", t_all);
  print_ns("compact_ns_iters", compact_ns);
  print_ns("rowgen_ns_iters", rowgen_ns);
  print_ns("decode_ns_iters", decode_ns);
  print_ns("all_ns_iters", all_ns);
  printf("  \"decode_gbps\": %.2f,\n", gbps(t_decode));
  printf("  \"all_gbps\": %.2f\n", gbps(t_all));
  printf("}\n");

  fprintf(stderr,
          "\n=== op1_selective_decode (%s) ===\n"
          "%llu tokens, %llu rows, m=%.4g%% -> %llu survivors, %.1f MB decoded\n"
          "compact %.3f ms | rowgen %.3f ms | decode %.3f ms | all %.3f ms\n"
          "compact_ok=%s offsets_ok=%s decode_ok=%s\n",
          prop.name, (unsigned long long)total_tokens, (unsigned long long)n_rows, sel_pct,
          (unsigned long long)n_sel, decoded_bytes / 1e6,
          t_compact, t_rowgen, t_decode, t_all,
          compact_ok ? "YES" : "NO", offsets_ok ? "YES" : "NO", decode_ok ? "YES" : "NO");

  return (compact_ok && offsets_ok && decode_ok) ? 0 : 4;
}
