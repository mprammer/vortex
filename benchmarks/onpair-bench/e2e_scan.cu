// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// e2e_scan.cu -- standalone end-to-end "decode -> on-device scan" benchmark.
//
// Why this exists: the paper times OnPair decode in isolation. A reviewer can
// fairly ask whether the decode win survives once a real operator consumes the
// decoded bytes -- i.e. whether query-part overhead eats the speedup. This bench
// answers that with a worst-case operator: a *rare-string* substring scan over
// the ENTIRE decoded column (a rare needle forces the operator to touch every
// decoded byte before it can rule the column out -- the heaviest selective scan).
//
// It measures, on one GPU, all on the same staged inputs (the EXACT encoder
// output dumped by onpair_bench.rs via ONPAIR_DUMP_E2E, so the decode here is
// byte-identical to the paper's and is validated against a CPU reference):
//   decode            -- OnPair decode alone (reproduces the paper's number)
//   scan              -- the substring scan alone, over the already-decoded bytes
//   e2e               -- decode THEN scan, back-to-back, bytes never leaving HBM
//   h2d_decompressed  -- PCIe H2D of the *decompressed* column (what a CPU-decode
//                        path must ship to scan on the GPU)
//   h2d_compressed    -- PCIe H2D of the *compressed* column (what the GPU-native
//                        path ships, then decodes on-device)
//   cpu_decode        -- single-thread CPU OnPair decode (reference leg of the
//                        CPU-decode + PCIe-transfer + scan baseline)
//
// The two claims it substantiates:
//   1. e2e ~= decode  (the scan is memory-bound and the decoded bytes are already
//      in HBM, so feeding the next operator is nearly free).
//   2. h2d_decompressed alone > decode+scan  (merely MOVING the decompressed column
//      across PCIe costs more than decoding it from scratch on-GPU and scanning it
//      -- so decompressing on the GPU avoids the PCIe round-trip that dominates the
//      CPU-decode baseline).
//
// Build (on the GPU box, from this directory):
//   nvcc -O3 -arch=native -std=c++17 e2e_scan.cu -o e2e_scan
// Run:
//   ./e2e_scan <dump.e2ebin> [needle] [iters]
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

// Reuse the EXACT decode kernel the paper ships (request-reducing split-read variant).
#include "../../vortex-cuda/kernels/src/onpair_shmem_4tpt_split8read.cu"

#define CK(x)                                                                   \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(e_));                                         \
      exit(3);                                                                 \
    }                                                                          \
  } while (0)

// SWAR: does the 4-byte word x contain a byte equal to n0 (broadcast bcast = n0*0x01010101)?
__device__ inline uint32_t word_has(uint32_t x, uint32_t bcast) {
  uint32_t y = x ^ bcast;
  return (y - 0x01010101u) & ~y & 0x80808080u;
}

// Vectorized substring-count scan. A byte-wise scan issues ~1 memory request per byte and
// is L1-request-bound (the very bottleneck this paper is about) -- not what we want to
// measure. So each thread loads a 16-byte chunk (one uint4 request) and SWAR-tests all 16
// bytes for the needle's FIRST byte. The common case is the uint4 load + 4 ALU tests: a
// coalesced HBM-bandwidth pass that touches every decoded byte. The needle's first byte is
// chosen rare (host side), so the candidate branch -- re-read the chunk's bytes from L1 and
// do the full compare -- almost never fires. Reads ~1 request / 16 bytes => bandwidth-bound.
__global__ void scan_needle(const uint8_t *__restrict__ data, uint64_t n,
                            const uint8_t *__restrict__ needle, uint32_t m,
                            uint8_t n0, uint32_t n0bcast,
                            unsigned long long *__restrict__ count) {
  const uint64_t nchunks = n >> 4;  // 16-byte chunks; the <16 B tail can't start a match
                                    // the CPU oracle reaches either (n%16 < m), so counts agree
  uint64_t c = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  const uint4 *d4 = reinterpret_cast<const uint4 *>(data);
  unsigned long long local = 0;
  for (; c < nchunks; c += stride) {
    uint4 v = d4[c];
    if (word_has(v.x, n0bcast) | word_has(v.y, n0bcast) |
        word_has(v.z, n0bcast) | word_has(v.w, n0bcast)) {
      const uint64_t base = c << 4;
      for (int k = 0; k < 16; ++k) {
        const uint64_t pos = base + (uint64_t)k;
        if (data[pos] == n0 && pos + m <= n) {  // re-read from L1 (just loaded); no &v spill
          bool hit = true;
          for (uint32_t j = 1; j < m; ++j) {
            if (data[pos + j] != needle[j]) { hit = false; break; }
          }
          if (hit) local++;
        }
      }
    }
  }
  // Only the (rare) threads that found a match touch the global counter. An
  // unconditional atomicAdd serializes ALL threads on one address -- with ~62 M
  // threads that dominated the kernel (scan time scaled with thread count, ~24 GB/s);
  // guarding it makes the scan a pure HBM-bandwidth pass for a selective needle.
  if (local) atomicAdd(count, local);
}

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <dump.e2ebin> [needle] [iters]\n", argv[0]);
    return 1;
  }
  const char *path = argv[1];
  const char *needle_arg = argc > 2 ? argv[2] : nullptr;
  int iters = argc > 3 ? atoi(argv[3]) : 100;
  if (iters < 3) iters = 3;

  // ── load the dump: "E2E1" | total_tokens u64 | dict_size u32 | max_token u32
  //    | codes(u16 LE) | lens(u8) | dict_padded(u8, dict_size*max_token) ──
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }
  char magic[4];
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "E2E1", 4) != 0) {
    fprintf(stderr, "bad magic in %s\n", path);
    return 2;
  }
  uint64_t total_tokens = 0;
  uint32_t dict_size = 0, max_token = 0;
  if (fread(&total_tokens, 8, 1, f) != 1 || fread(&dict_size, 4, 1, f) != 1 ||
      fread(&max_token, 4, 1, f) != 1) {
    fprintf(stderr, "short header\n");
    return 2;
  }
  std::vector<uint16_t> codes(total_tokens);
  std::vector<uint8_t> lens(dict_size);
  std::vector<uint8_t> dict_padded((size_t)dict_size * max_token);
  if (fread(codes.data(), 2, total_tokens, f) != total_tokens ||
      fread(lens.data(), 1, dict_size, f) != dict_size ||
      fread(dict_padded.data(), 1, dict_padded.size(), f) != dict_padded.size()) {
    fprintf(stderr, "short body\n");
    return 2;
  }
  fclose(f);

  // ── derive everything the kernel needs from {codes, lens, dict_padded} ──
  // per-token output offsets (cumulative decoded byte position) + decoded size
  std::vector<uint64_t> tok_off(total_tokens + 1);
  tok_off[0] = 0;
  uint64_t sum_dict_len = 0;
  for (uint32_t c = 0; c < dict_size; ++c) sum_dict_len += lens[c];
  for (uint64_t i = 0; i < total_tokens; ++i)
    tok_off[i + 1] = tok_off[i] + lens[codes[i]];
  const uint64_t decoded_bytes = tok_off[total_tokens];

  // dict_s8: first min(len,8) bytes of each padded entry (rest already zero)
  std::vector<uint8_t> dict_s8((size_t)dict_size * 8, 0);
  for (uint32_t c = 0; c < dict_size; ++c) {
    uint32_t n8 = lens[c] < 8 ? lens[c] : 8;
    memcpy(&dict_s8[(size_t)c * 8], &dict_padded[(size_t)c * max_token], n8);
  }

  // chunk_offsets at 128-token boundaries (kernel chunk = 128 tokens)
  const uint64_t n_chunks = (total_tokens + 127) / 128;
  std::vector<uint64_t> chunk_off(n_chunks + 1);
  for (uint64_t cidx = 0; cidx <= n_chunks; ++cidx) {
    uint64_t t = cidx * 128;
    if (t > total_tokens) t = total_tokens;
    chunk_off[cidx] = tok_off[t];
  }

  // CPU reference decode (also the correctness oracle and the baseline's decode leg)
  std::vector<uint8_t> cpu_out(decoded_bytes);
  double cpu_decode_ms = 1e30;
  for (int it = 0; it < 5; ++it) {
    double t0 = now_s();
    for (uint64_t i = 0; i < total_tokens; ++i) {
      uint32_t code = codes[i];
      memcpy(&cpu_out[tok_off[i]], &dict_padded[(size_t)code * max_token],
             lens[code]);
    }
    double ms = (now_s() - t0) * 1e3;
    if (ms < cpu_decode_ms) cpu_decode_ms = ms;
  }

  // ── choose the needle: explicit arg, else a deterministic 16-byte window from
  //    40% through the decoded column (present, and on real URL data rare) ──
  std::vector<uint8_t> needle;
  if (needle_arg && needle_arg[0]) {  // explicit, non-empty needle
    needle.assign((const uint8_t *)needle_arg,
                  (const uint8_t *)needle_arg + strlen(needle_arg));
  } else {  // AUTO: a 24-byte window whose FIRST byte is a rare byte value, so the vectorized
            // scan's first-byte gate fires rarely and stays bandwidth-bound. Taken at/after
            // 40% through; present by construction; a 24-byte string is near-unique on real data.
    uint64_t hist[256] = {0};
    for (uint64_t i = 0; i < decoded_bytes; ++i) hist[cpu_out[i]]++;
    int rb = -1;
    uint64_t bestc = 0;
    for (int v = 0; v < 256; ++v)
      if (hist[v] > 0 && (rb < 0 || hist[v] < bestc)) { rb = v; bestc = hist[v]; }
    uint64_t start = decoded_bytes * 2 / 5, pos = decoded_bytes;
    for (uint64_t i = start; i + 24 <= decoded_bytes; ++i)
      if (cpu_out[i] == (uint8_t)rb) { pos = i; break; }
    if (pos == decoded_bytes) pos = (start + 24 <= decoded_bytes) ? start : 0;
    uint32_t want = 24;
    if (pos + want > decoded_bytes) want = (uint32_t)(decoded_bytes - pos);
    needle.assign(&cpu_out[pos], &cpu_out[pos + want]);
  }
  const uint32_t m = (uint32_t)needle.size();

  // CPU match count (the oracle for the GPU scan's count)
  uint64_t cpu_matches = 0;
  for (uint64_t i = 0; i + m <= decoded_bytes; ++i)
    if (memcmp(&cpu_out[i], needle.data(), m) == 0) cpu_matches++;

  // ── upload to device ──
  uint16_t *d_codes;
  uint64_t *d_choff;
  uint8_t *d_dict_s8, *d_dict_padded, *d_lens, *d_out, *d_needle;
  unsigned long long *d_count;
  CK(cudaMalloc(&d_codes, total_tokens * 2));
  CK(cudaMalloc(&d_choff, (n_chunks + 1) * 8));
  CK(cudaMalloc(&d_dict_s8, dict_s8.size()));
  CK(cudaMalloc(&d_dict_padded, dict_padded.size()));
  CK(cudaMalloc(&d_lens, dict_size));
  CK(cudaMalloc(&d_out, decoded_bytes + 64));
  CK(cudaMalloc(&d_needle, m));
  CK(cudaMalloc(&d_count, sizeof(unsigned long long)));
  CK(cudaMemcpy(d_codes, codes.data(), total_tokens * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_choff, chunk_off.data(), (n_chunks + 1) * 8,
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_dict_s8, dict_s8.data(), dict_s8.size(),
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_dict_padded, dict_padded.data(), dict_padded.size(),
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lens, lens.data(), dict_size, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_needle, needle.data(), m, cudaMemcpyHostToDevice));

  // launch geometry
  dim3 dblock(512);
  dim3 dgrid((unsigned)((n_chunks + 15) / 16));  // 16 warps/block, 1 warp/chunk
  int sblk = 256;
  uint64_t s_nchunks = decoded_bytes >> 4;  // one thread per 16-byte uint4 chunk
  uint64_t sgrid64 = (s_nchunks + sblk - 1) / sblk;
  if (sgrid64 > 262144) sgrid64 = 262144;  // grid-stride caps the grid
  dim3 sgrid((unsigned)std::max<uint64_t>(sgrid64, 1));

  auto launch_decode = [&]() {
    onpair_shmem_4tpt_split8read<<<dgrid, dblock>>>(
        d_codes, d_choff, d_dict_s8, d_dict_padded, d_lens, d_out, total_tokens);
  };
  const uint8_t n0 = m ? needle[0] : 0;
  const uint32_t n0bcast = (uint32_t)n0 * 0x01010101u;
  auto launch_scan = [&]() {
    scan_needle<<<sgrid, sblk>>>(d_out, decoded_bytes, d_needle, m, n0, n0bcast, d_count);
  };

  // ── correctness: decode once on the GPU, compare to the CPU reference ──
  CK(cudaMemset(d_out, 0, decoded_bytes + 64));
  launch_decode();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<uint8_t> gpu_out(decoded_bytes);
  CK(cudaMemcpy(gpu_out.data(), d_out, decoded_bytes, cudaMemcpyDeviceToHost));
  bool decode_ok = (memcmp(gpu_out.data(), cpu_out.data(), decoded_bytes) == 0);

  // GPU scan count (over the freshly-decoded buffer) vs the CPU oracle
  CK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
  launch_scan();
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  unsigned long long gpu_matches = 0;
  CK(cudaMemcpy(&gpu_matches, d_count, sizeof(unsigned long long),
                cudaMemcpyDeviceToHost));
  bool scan_ok = (gpu_matches == cpu_matches);

  // ── timing helpers (CUDA events, min over iters -- matches the paper) ──
  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0));
  CK(cudaEventCreate(&ev1));
  auto time_min = [&](auto &&fn) -> double {
    for (int w = 0; w < 3; ++w) {
      fn();
    }
    CK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int k = 0; k < iters; ++k) {
      CK(cudaEventRecord(ev0));
      fn();
      CK(cudaEventRecord(ev1));
      CK(cudaEventSynchronize(ev1));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, ev0, ev1));
      if (ms < best) best = ms;
    }
    return best;
  };

  double t_decode = time_min([&]() { launch_decode(); });
  double t_scan = time_min([&]() { launch_scan(); });
  double t_e2e = time_min([&]() {
    launch_decode();
    launch_scan();
  });

  // ── PCIe transfers from pinned host memory (decompressed vs compressed) ──
  const uint64_t compressed_bytes =
      total_tokens * 2 + sum_dict_len + dict_size;  // codes + dict bytes + lens
  uint8_t *h_pin_dec = nullptr;
  uint8_t *h_pin_cmp = nullptr;
  CK(cudaHostAlloc(&h_pin_dec, decoded_bytes, cudaHostAllocDefault));
  CK(cudaHostAlloc(&h_pin_cmp, compressed_bytes, cudaHostAllocDefault));
  memcpy(h_pin_dec, cpu_out.data(), decoded_bytes);
  uint8_t *d_xfer;
  CK(cudaMalloc(&d_xfer, decoded_bytes));  // big enough for both
  double t_h2d_dec = time_min([&]() {
    cudaMemcpyAsync(d_xfer, h_pin_dec, decoded_bytes, cudaMemcpyHostToDevice);
  });
  double t_h2d_cmp = time_min([&]() {
    cudaMemcpyAsync(d_xfer, h_pin_cmp, compressed_bytes, cudaMemcpyHostToDevice);
  });

  // ── derived rates + headline ratios ──
  auto gbps = [](uint64_t bytes, double ms) {
    return bytes / (ms / 1e3) / 1e9;
  };
  double e2e_overhead_pct = (t_e2e - t_decode) / t_decode * 100.0;
  double gpu_native_ms = t_h2d_cmp + t_decode + t_scan;
  double cpu_baseline_ms = cpu_decode_ms + t_h2d_dec + t_scan;
  double pcie_vs_decodescan = t_h2d_dec / (t_decode + t_scan);

  // needle as hex for the JSON
  std::string nhex;
  char hb[3];
  for (uint32_t i = 0; i < m; ++i) {
    snprintf(hb, sizeof(hb), "%02x", needle[i]);
    nhex += hb;
  }

  // device name
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));

  printf("{\n");
  printf("  \"gpu\": \"%s\",\n", prop.name);
  printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
  printf("  \"kernel\": \"onpair_shmem_4tpt_split8read\",\n");
  printf("  \"total_tokens\": %llu,\n", (unsigned long long)total_tokens);
  printf("  \"dict_size\": %u,\n", dict_size);
  printf("  \"decoded_bytes\": %llu,\n", (unsigned long long)decoded_bytes);
  printf("  \"compressed_bytes\": %llu,\n", (unsigned long long)compressed_bytes);
  printf("  \"ratio\": %.4f,\n", (double)decoded_bytes / compressed_bytes);
  printf("  \"iters\": %d,\n", iters);
  printf("  \"needle_hex\": \"%s\",\n", nhex.c_str());
  printf("  \"needle_len\": %u,\n", m);
  printf("  \"decode_ok\": %s,\n", decode_ok ? "true" : "false");
  printf("  \"scan_ok\": %s,\n", scan_ok ? "true" : "false");
  printf("  \"cpu_matches\": %llu,\n", (unsigned long long)cpu_matches);
  printf("  \"gpu_matches\": %llu,\n", (unsigned long long)gpu_matches);
  printf("  \"decode_ms\": %.5f,\n", t_decode);
  printf("  \"scan_ms\": %.5f,\n", t_scan);
  printf("  \"e2e_ms\": %.5f,\n", t_e2e);
  printf("  \"h2d_decompressed_ms\": %.5f,\n", t_h2d_dec);
  printf("  \"h2d_compressed_ms\": %.5f,\n", t_h2d_cmp);
  printf("  \"cpu_decode_ms\": %.5f,\n", cpu_decode_ms);
  printf("  \"decode_gbps\": %.2f,\n", gbps(decoded_bytes, t_decode));
  printf("  \"scan_gbps\": %.2f,\n", gbps(decoded_bytes, t_scan));
  printf("  \"e2e_gbps\": %.2f,\n", gbps(decoded_bytes, t_e2e));
  printf("  \"h2d_decompressed_gbps\": %.2f,\n", gbps(decoded_bytes, t_h2d_dec));
  printf("  \"h2d_compressed_gbps\": %.2f,\n", gbps(compressed_bytes, t_h2d_cmp));
  printf("  \"cpu_decode_gbps\": %.2f,\n", gbps(decoded_bytes, cpu_decode_ms));
  printf("  \"e2e_overhead_pct\": %.2f,\n", e2e_overhead_pct);
  printf("  \"gpu_native_total_ms\": %.5f,\n", gpu_native_ms);
  printf("  \"cpu_decode_pcie_scan_total_ms\": %.5f,\n", cpu_baseline_ms);
  printf("  \"pcie_decompressed_vs_gpu_decode_scan\": %.2f\n", pcie_vs_decodescan);
  printf("}\n");

  fprintf(stderr,
          "\n=== e2e_scan summary (%s) ===\n"
          "decoded=%.1f MB  ratio=%.2fx  decode=%.1f GB/s  scan=%.1f GB/s  "
          "e2e=%.1f GB/s (+%.1f%% over decode)\n"
          "PCIe decompressed H2D=%.1f GB/s -> moving the decompressed column "
          "alone is %.1fx the GPU decode+scan time\n"
          "decode_ok=%s scan_ok=%s (cpu=%llu gpu=%llu matches)\n",
          prop.name, decoded_bytes / 1e6, (double)decoded_bytes / compressed_bytes,
          gbps(decoded_bytes, t_decode), gbps(decoded_bytes, t_scan),
          gbps(decoded_bytes, t_e2e), e2e_overhead_pct,
          gbps(decoded_bytes, t_h2d_dec), pcie_vs_decodescan,
          decode_ok ? "YES" : "NO", scan_ok ? "YES" : "NO",
          (unsigned long long)cpu_matches, (unsigned long long)gpu_matches);
  return decode_ok ? 0 : 4;
}
