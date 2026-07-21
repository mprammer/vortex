// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// op1_row_decode.cu -- OP1 of the output-positioning trade-off: partition decode by ROW using the
// per-row code offsets that ALREADY EXIST in the stored format (`op.codes_offsets()`), and generate
// the output row offsets on the device -- versus storing per-128-token chunk offsets (OP4) or
// regenerating them on the GPU (OP2, op_gpu_regen.cu).
//
// Why OP1 is distinct: the row offsets are free (every var-length column stores them), and the
// output row offsets this bench generates are exactly the offsets an Arrow string result needs, so
// in an integrated decoder they are required output, not throwaway. The cost is load imbalance:
// this first cut is THREAD-PER-ROW, so it is efficient for small/uniform rows (the regime OP1
// targets) and serial for long rows (a warp-/token-balanced kernel is future work). The bench
// therefore characterizes WHERE row-partition decode wins, not a universal claim.
//
// SCOPE (exploratory): this measures a specific point, not a production decoder. Zero-token /
// zero-byte columns (all-empty / all-null strings) are cleanly SKIPPED (exit 2), not decoded to
// empty output -- a non-throughput degenerate the campaign's real columns never produce. This is
// a deliberate limitation (terra+sol re-review MF2, accepted); the nonempty path is byte-exact and
// the fix audit is clean. Broaden to the full OP11 input domain only if OP1 graduates past the probe.
//
// Pipeline (matches op_gpu_regen.cu so the two are comparable per column):
//   rowgen  = per-row decoded-size reduction (row_sizes_kernel) + exclusive scan (CUB) -> row_out_off
//   decode  = row_decode_kernel: each thread writes its row's tokens at row_out_off[r], EXACT length
//             per token (no 16-B over-store, so no cross-row write race)
//
// Input: the "OP11" dump onpair_bench.rs writes via ONPAIR_DUMP_OP1 (codes, row_offsets, lens,
// dict_padded). Emits one JSON object on stdout; exit 0 iff offsets + decode are byte-exact.
//
// Build (on the GPU box, from this directory):
//   nvcc -O3 -arch=native -std=c++17 op1_row_decode.cu -o op1_row_decode
// Run:
//   ./op1_row_decode <dump.op1bin> [iters]

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

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e_));                                         \
      exit(3);                                                                 \
    }                                                                          \
  } while (0)

static constexpr uint32_t MAX_TOKEN = 16;      // dict_padded stride; asserted from the dump header
static constexpr int BLOCK_THREADS = 256;

// Per-row decoded-size reduction: thread r sums lens[codes[t]] over its row's tokens.
// Reads the FREE stored row offsets; no dictionary-byte gather, no output -- a scan of the codes.
__global__ void row_sizes_kernel(const uint16_t *__restrict codes,
                                 const uint64_t *__restrict row_off,
                                 const uint8_t *__restrict lens,
                                 uint64_t n_rows,
                                 uint64_t *__restrict rowsize) {
  const uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_rows) return;
  const uint64_t t0 = row_off[r], t1 = row_off[r + 1];
  uint64_t s = 0;
  for (uint64_t t = t0; t < t1; ++t) s += (uint64_t)lens[codes[t]];
  rowsize[r] = s;
}

// Write the terminal output offset so the generated buffer is a complete Arrow-style offset array
// of n_rows+1 entries (n_rows starts + the terminal == total decoded bytes), not just the starts.
// n_rows >= 1 is guaranteed by the header check. Runs after the scan in-stream, so the reads are
// ordered. Single-thread: one trailing element.
__global__ void write_terminal_kernel(uint64_t *__restrict row_out_off,
                                      const uint64_t *__restrict rowsize, uint64_t n_rows) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    row_out_off[n_rows] = row_out_off[n_rows - 1] + rowsize[n_rows - 1];
}

// Row-partition decode: thread r writes its row's tokens starting at row_out_off[r], copying EXACTLY
// len bytes per token (<= MAX_TOKEN). Exact-length copy means no thread writes past its row's end,
// so adjacent rows never race (a 16-B vector store WOULD spill into the next row -- avoided here).
__global__ void row_decode_kernel(const uint16_t *__restrict codes,
                                  const uint64_t *__restrict row_off,
                                  const uint64_t *__restrict row_out_off,
                                  const uint8_t *__restrict dict_padded,
                                  const uint8_t *__restrict lens,
                                  uint8_t *__restrict out,
                                  uint64_t n_rows) {
  const uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_rows) return;
  const uint64_t t0 = row_off[r], t1 = row_off[r + 1];
  uint64_t cur = row_out_off[r];
  for (uint64_t t = t0; t < t1; ++t) {
    const uint16_t c = codes[t];
    const uint32_t len = lens[c];
    const uint8_t *src = dict_padded + (size_t)c * MAX_TOKEN;
    for (uint32_t j = 0; j < len; ++j) out[cur + j] = src[j];
    cur += len;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <dump.op1bin> [iters]\n", argv[0]);
    return 1;
  }
  const char *path = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 100;
  if (iters < 3) iters = 3;

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
    // A zero-token / zero-byte column (e.g. all-empty/all-null strings) is a legal OP11 dump but
    // not a decode-throughput case; skip cleanly (exit 2) rather than measure nothing.
    fprintf(stderr, "empty/degenerate dump (dict=%u tokens=%llu rows=%llu) — not a throughput case; skipping\n",
            dict_size, (unsigned long long)total_tokens, (unsigned long long)n_rows);
    return 2;
  }
  // CUB DeviceScan takes an int num_items; reject rather than silently truncate.
  if (n_rows > (uint64_t)INT_MAX) {
    fprintf(stderr, "n_rows=%llu exceeds INT_MAX (CUB num_items); split the column\n",
            (unsigned long long)n_rows); return 2;
  }
  std::vector<uint16_t> codes(total_tokens);
  std::vector<uint64_t> row_off(n_rows + 1);
  std::vector<uint8_t> lens(dict_size);
  std::vector<uint8_t> dict_padded((size_t)dict_size * max_token);
  // OP11 is little-endian; this reads directly into native ints (LE host only: x86-64 / AArch64).
  if (fread(codes.data(), 2, total_tokens, f) != total_tokens ||
      fread(row_off.data(), 8, n_rows + 1, f) != n_rows + 1 ||
      fread(lens.data(), 1, dict_size, f) != dict_size ||
      fread(dict_padded.data(), 1, dict_padded.size(), f) != dict_padded.size()) {
    fprintf(stderr, "short body\n"); return 2;
  }
  fclose(f);

  // ── bounds: codes index real entries; lens fit the entry; row_off monotone, bounded, framing ──
  for (uint16_t c : codes) if (c >= dict_size) { fprintf(stderr, "code %u out of range\n", (unsigned)c); return 2; }
  for (uint8_t l : lens) if (l > max_token) { fprintf(stderr, "dict length %u exceeds max_token %u\n", (unsigned)l, max_token); return 2; }
  if (row_off[0] != 0 || row_off[n_rows] != total_tokens) {
    fprintf(stderr, "row_off framing bad: [0]=%llu [n]=%llu total_tokens=%llu\n",
            (unsigned long long)row_off[0], (unsigned long long)row_off[n_rows],
            (unsigned long long)total_tokens); return 2;
  }
  for (uint64_t r = 0; r < n_rows; ++r)
    if (row_off[r + 1] < row_off[r] || row_off[r + 1] > total_tokens) {
      fprintf(stderr, "row_off not monotone at row %llu\n", (unsigned long long)r); return 2;
    }

  // ── host reference: per-token offsets, decoded size, per-row output offsets, byte-exact decode ──
  std::vector<uint64_t> tok_off(total_tokens + 1);
  tok_off[0] = 0;
  for (uint64_t i = 0; i < total_tokens; ++i) tok_off[i + 1] = tok_off[i] + lens[codes[i]];
  const uint64_t decoded_bytes = tok_off[total_tokens];
  if (decoded_bytes == 0) { fprintf(stderr, "dump decodes to 0 bytes; unsupported\n"); return 2; }

  // Reference per-row output offsets, Arrow-style with n_rows+1 entries: row r starts at [r], and
  // the terminal [n_rows] == decoded_bytes (the offset buffer an Arrow string result needs).
  std::vector<uint64_t> row_out_off_ref(n_rows + 1);
  for (uint64_t r = 0; r <= n_rows; ++r) row_out_off_ref[r] = tok_off[row_off[r]];
  if (row_out_off_ref[n_rows] != decoded_bytes) {
    fprintf(stderr, "internal: terminal offset %llu != decoded_bytes %llu\n",
            (unsigned long long)row_out_off_ref[n_rows], (unsigned long long)decoded_bytes);
    return 2;
  }

  std::vector<uint8_t> cpu_out(decoded_bytes);
  {
    uint64_t cur = 0;
    for (uint64_t i = 0; i < total_tokens; ++i) {
      uint32_t c = codes[i], len = lens[c];
      if (len) memcpy(&cpu_out[cur], &dict_padded[(size_t)c * max_token], len);  // len==0: skip (no &cpu_out[size()])
      cur += len;
    }
  }

  // ── upload ──────────────────────────────────────────────────────────────────
  uint16_t *d_codes;
  uint64_t *d_row_off, *d_rowsize, *d_row_out_off, *d_row_out_off_ref;
  uint8_t *d_lens, *d_padded, *d_out;
  CK(cudaMalloc(&d_codes, total_tokens * 2));
  CK(cudaMalloc(&d_row_off, (n_rows + 1) * 8));
  CK(cudaMalloc(&d_rowsize, n_rows * 8));
  CK(cudaMalloc(&d_row_out_off, (n_rows + 1) * 8));       // GPU-generated: n_rows starts + terminal
  CK(cudaMalloc(&d_row_out_off_ref, (n_rows + 1) * 8));   // host reference, preloaded (decode baseline)
  CK(cudaMalloc(&d_lens, dict_size));
  CK(cudaMalloc(&d_padded, dict_padded.size()));
  CK(cudaMalloc(&d_out, decoded_bytes + 64));
  CK(cudaMemcpy(d_codes, codes.data(), total_tokens * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_row_off, row_off.data(), (n_rows + 1) * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lens, lens.data(), dict_size, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_padded, dict_padded.data(), dict_padded.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_row_out_off_ref, row_out_off_ref.data(), (n_rows + 1) * 8, cudaMemcpyHostToDevice));

  const uint32_t grid = (uint32_t)((n_rows + BLOCK_THREADS - 1) / BLOCK_THREADS);

  void *d_tmp = nullptr;
  size_t tmp_bytes = 0;
  CK(cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, d_rowsize, d_row_out_off, (int)n_rows));
  CK(cudaMalloc(&d_tmp, tmp_bytes));

  auto rowgen = [&]() {
    row_sizes_kernel<<<grid, BLOCK_THREADS>>>(d_codes, d_row_off, d_lens, n_rows, d_rowsize);
    CK(cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_rowsize, d_row_out_off, (int)n_rows));
    write_terminal_kernel<<<1, 32>>>(d_row_out_off, d_rowsize, n_rows);  // complete the n_rows+1 array
  };
  auto decode = [&](uint64_t *rowoff) {
    row_decode_kernel<<<grid, BLOCK_THREADS>>>(d_codes, d_row_off, rowoff, d_padded, d_lens, d_out, n_rows);
  };

  // ── validation: GPU row offsets == host ref; decode-with-generated-offsets == host decode ──
  rowgen();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint64_t> rowoff_gpu(n_rows + 1);
  CK(cudaMemcpy(rowoff_gpu.data(), d_row_out_off, (n_rows + 1) * 8, cudaMemcpyDeviceToHost));
  bool offsets_ok = (memcmp(rowoff_gpu.data(), row_out_off_ref.data(), (n_rows + 1) * 8) == 0);

  CK(cudaMemset(d_out, 0xA5, decoded_bytes + 64));
  decode(d_row_out_off);
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

  std::vector<uint64_t> rowgen_ns, decode_ns, both_ns;
  double t_rowgen = time_min_raw(rowgen, rowgen_ns);
  double t_decode = time_min_raw([&]() { decode(d_row_out_off_ref); }, decode_ns);   // offsets preloaded
  double t_both = time_min_raw([&]() { rowgen(); decode(d_row_out_off); }, both_ns);  // gen then decode

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  auto gbps = [&](double ms) { return ms > 0.0 ? decoded_bytes / (ms / 1e3) / 1e9 : 0.0; };
  const double gen_overhead_pct = t_decode > 0.0 ? 100.0 * (t_both / t_decode - 1.0) : 0.0;
  auto print_ns = [](const char *k, const std::vector<uint64_t> &v) {
    printf("  \"%s\": [", k);
    for (size_t i = 0; i < v.size(); ++i) printf("%s%llu", i ? "," : "", (unsigned long long)v[i]);
    printf("],\n");
  };

  printf("{\n");
  printf("  \"gpu\": \"%s\",\n", prop.name);
  printf("  \"kernel\": \"op1_row_decode (thread-per-row) + gpu-row-offset-scan\",\n");
  printf("  \"total_tokens\": %llu,\n", (unsigned long long)total_tokens);
  printf("  \"n_rows\": %llu,\n", (unsigned long long)n_rows);
  printf("  \"decoded_bytes\": %llu,\n", (unsigned long long)decoded_bytes);
  printf("  \"mean_row_bytes\": %.2f,\n", (double)decoded_bytes / (double)n_rows);
  printf("  \"iters\": %d,\n", iters);
  printf("  \"offsets_ok\": %s,\n", offsets_ok ? "true" : "false");
  printf("  \"decode_ok\": %s,\n", decode_ok ? "true" : "false");
  printf("  \"rowgen_ms\": %.5f,\n", t_rowgen);
  printf("  \"decode_ms\": %.5f,\n", t_decode);
  printf("  \"rowgen_plus_decode_ms\": %.5f,\n", t_both);
  print_ns("rowgen_ns_iters", rowgen_ns);
  print_ns("decode_ns_iters", decode_ns);
  print_ns("rowgen_plus_decode_ns_iters", both_ns);
  printf("  \"decode_gbps\": %.2f,\n", gbps(t_decode));
  printf("  \"rowgen_plus_decode_gbps\": %.2f,\n", gbps(t_both));
  printf("  \"gen_overhead_pct\": %.2f\n", gen_overhead_pct);
  printf("}\n");

  fprintf(stderr,
          "\n=== op1_row_decode (%s) ===\n"
          "%llu tokens, %llu rows (%.1f B/row mean), %.1f MB decoded\n"
          "rowgen %.3f ms | decode %.3f ms | rowgen+decode %.3f ms | overhead %.1f%%\n"
          "offsets_ok=%s decode_ok=%s\n",
          prop.name, (unsigned long long)total_tokens, (unsigned long long)n_rows,
          (double)decoded_bytes / (double)n_rows, decoded_bytes / 1e6,
          t_rowgen, t_decode, t_both, gen_overhead_pct,
          offsets_ok ? "YES" : "NO", decode_ok ? "YES" : "NO");

  return (offsets_ok && decode_ok) ? 0 : 4;
}
