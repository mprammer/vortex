// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// batch_decode.cu -- standalone "many small row-groups, one file" decode bench.
//
// Why this exists: the paper reports small columns (dbtext, TPC-H s_comment;
// 1--6 MB) only "for completeness" because a single such column launches under a
// fifth of one wave of thread blocks and reaches ~3% occupancy -- it is
// launch/latency-bound, not throughput-bound (Section 6.4). But a *file* is not
// one small column: it is hundreds of small row-groups, EACH independently
// OnPair-compressed with its OWN dictionary. This bench shows that decoding a
// realistic file's worth of small row-groups (hundreds of 1--6 MB groups,
// ~1 GB total) recovers throughput-bound rates -- and compares the three ways to
// launch that work:
//
//   sequential  -- N row-groups as N sequential kernel launches (the shipped
//                  per-chunk path; exposes the per-launch fixed cost).
//   streams     -- the same N launches round-robined over K CUDA streams (fills
//                  the device by overlapping many small grids).
//   graph       -- the same N launches captured once into a CUDA graph and
//                  replayed (kills the per-launch host-submit overhead; still N
//                  grids on the device).
//   multidict   -- ONE launch spanning EVERY row-group, via a per-warp-batch
//                  dictionary indirection (onpair_shmem_4tpt_multidict). One grid,
//                  sized by the whole file.
//
// All four decode the SAME staged inputs and are validated byte-exact against a
// CPU reference; the reported rate is decoded output bytes / min-over-iters time,
// aggregated across all row-groups (GB/s, decimal 1e9), matching the paper's
// estimator. Raw per-iteration nanoseconds are emitted for every mode so any
// reduction is recomputable at figure-generation.
//
// Input: a "row-group stream" dump produced by ONPAIR_DUMP_BATCH= on the decode
// path (see benchmarks/onpair-bench/README.md and BENCH-PLAN.md). It is a
// concatenation of self-describing records, one per row-group:
//   "RGB1" | total_tokens:u64 | dict_size:u32 | max_token:u32
//         | codes(u16 LE, total_tokens) | lens(u8, dict_size)
//         | dict_padded(u8, dict_size*max_token)
// read until EOF. This mirrors e2e_scan.cu's E2E1 body, once per row-group.
//
// Build (on the GPU box, from this directory):
//   nvcc -O3 -arch=native -std=c++17 batch_decode.cu -o batch_decode
// Run:
//   ./batch_decode <dump.batchbin> [iters] [streams]
// Emits a JSON object on stdout (diagnostics on stderr).

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The shipped single-dictionary decode kernel (used for sequential/streams/graph),
// then the multi-dictionary one-grid variant. Include split8read FIRST so its
// unguarded WARP_BUF_BYTES define lands before multidict's #ifndef guard.
#include "../../vortex-cuda/kernels/src/onpair_shmem_4tpt_split8read.cu"
#include "../../vortex-cuda/kernels/src/onpair_shmem_4tpt_multidict.cu"

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e_));                                         \
      exit(3);                                                                 \
    }                                                                          \
  } while (0)

static constexpr uint32_t TOK_PER_BATCH = 128;  // warp-batch = 128 tokens
static constexpr int BLOCK_THREADS = 512;       // 16 warps/block (matches launch bounds)
static constexpr uint32_t WARPS_PER_BLOCK = BLOCK_THREADS / 32;

struct RowGroup {
  uint64_t total_tokens;
  uint32_t dict_size;
  uint32_t max_token;
  std::vector<uint16_t> codes;       // total_tokens
  std::vector<uint8_t> lens;         // dict_size
  std::vector<uint8_t> dict_padded;  // dict_size * max_token
};

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <dump.batchbin> [iters] [streams]\n", argv[0]);
    return 1;
  }
  const char *path = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 100;
  if (iters < 3) iters = 3;
  int n_streams = argc > 3 ? atoi(argv[3]) : 8;
  if (n_streams < 1) n_streams = 1;

  // ── load the row-group stream ────────────────────────────────────────────
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }
  std::vector<RowGroup> rgs;
  uint32_t global_max_token = 0;
  for (;;) {
    char magic[4];
    size_t got = fread(magic, 1, 4, f);
    if (got == 0) break;  // clean EOF
    if (got != 4 || memcmp(magic, "RGB1", 4) != 0) {
      fprintf(stderr, "bad record magic at rg %zu\n", rgs.size());
      return 2;
    }
    RowGroup rg;
    if (fread(&rg.total_tokens, 8, 1, f) != 1 ||
        fread(&rg.dict_size, 4, 1, f) != 1 ||
        fread(&rg.max_token, 4, 1, f) != 1) {
      fprintf(stderr, "short header at rg %zu\n", rgs.size());
      return 2;
    }
    // ABI + sanity gates, mirroring e2e_scan.cu: both decode kernels hard-code a
    // 16-byte padded-dict stride (dict_padded + code*16), so max_token must be 16.
    // Empty records are outside this throughput bench's contract — they would
    // stage zero-sized grids/allocs — so reject them here rather than special-case
    // every launch/alloc below. (Codes/lengths are bounds-checked after the body.)
    if (rg.max_token != 16) {
      fprintf(stderr, "rg %zu: max_token=%u but the GPU kernel ABI requires 16\n",
              rgs.size(), rg.max_token);
      return 2;
    }
    if (rg.dict_size == 0 || rg.total_tokens == 0) {
      fprintf(stderr, "rg %zu: empty record (dict_size=%u total_tokens=%llu) unsupported\n",
              rgs.size(), rg.dict_size, (unsigned long long)rg.total_tokens);
      return 2;
    }
    rg.codes.resize(rg.total_tokens);
    rg.lens.resize(rg.dict_size);
    rg.dict_padded.resize((size_t)rg.dict_size * rg.max_token);
    if (fread(rg.codes.data(), 2, rg.total_tokens, f) != rg.total_tokens ||
        fread(rg.lens.data(), 1, rg.dict_size, f) != rg.dict_size ||
        fread(rg.dict_padded.data(), 1, rg.dict_padded.size(), f) !=
            rg.dict_padded.size()) {
      fprintf(stderr, "short body at rg %zu\n", rgs.size());
      return 2;
    }
    // Payload bounds: every length must fit the fixed 16-byte entry (and thus the
    // 2080-byte warp buffer the kernels stage through), and every code must index
    // a real dictionary entry — otherwise the host derivation and the device
    // gather both read out of bounds.
    for (uint8_t len : rg.lens) {
      if (len > rg.max_token) {
        fprintf(stderr, "rg %zu: dict length %u exceeds max_token %u\n", rgs.size(),
                (unsigned)len, rg.max_token);
        return 2;
      }
    }
    for (uint16_t code : rg.codes) {
      if (code >= rg.dict_size) {
        fprintf(stderr, "rg %zu: code %u out of range (dict_size=%u)\n", rgs.size(),
                (unsigned)code, rg.dict_size);
        return 2;
      }
    }
    global_max_token = std::max(global_max_token, rg.max_token);
    rgs.push_back(std::move(rg));
  }
  fclose(f);
  if (rgs.empty()) {
    fprintf(stderr, "no row-groups in %s\n", path);
    return 2;
  }
  const uint32_t MT = global_max_token;  // == vortex_onpair::MAX_TOKEN_SIZE (16)

  // ── derive the concatenated device layout + per-batch/per-rg metadata ─────
  //   codes_all / lens_all / dict_s8_all / dict_padded_all : concatenated blobs
  //   rg_dict_base[rg] : entry base into the dict blobs (cumulative dict_size)
  //   rg_out_base[rg]  : global byte offset of the row-group's decoded output
  //   rg_tok_base[rg]  : global token index of the row-group's first token
  //   per (global) warp-batch: tok_off, out_off, ntok, rg
  //   per-rg 0-based chunk_offsets (for the single-dict sequential/stream path)
  const size_t n_rgs = rgs.size();
  std::vector<uint64_t> rg_dict_base(n_rgs + 1, 0);
  std::vector<uint64_t> rg_out_base(n_rgs + 1, 0);
  std::vector<uint64_t> rg_tok_base(n_rgs + 1, 0);
  std::vector<uint64_t> rg_choff_base(n_rgs + 1, 0);  // into concatenated chunk-offsets
  std::vector<uint32_t> rg_nbatches(n_rgs, 0);

  // Align each row-group's output base to 16 bytes. split8read aligns its 16-byte
  // vector-store drain relative to a row-group-LOCAL offset, which is only a valid
  // GLOBAL alignment if the row-group's base is itself 16-aligned — as it is in
  // production, where every chunk owns a fresh (16+-aligned) output buffer. Packing
  // all row-groups into one shared buffer at unaligned cumulative offsets would make
  // every sequential/streams/graph `uint4 __stcs` misaligned (undefined: silent
  // rounding or a fault). Padding each base to 16 restores the invariant. multidict
  // is exempt (it addresses the true global offset directly) but the shared padding
  // is harmless to it. The <16-byte inter-row-group gaps stay at the poison byte in
  // BOTH the CPU reference and the device buffer, so the byte-exact memcmp is
  // unaffected; GB/s uses the true decoded byte count, not the padded span.
  auto align16 = [](uint64_t x) { return (x + 15u) & ~(uint64_t)15u; };
  std::vector<uint64_t> rg_decoded(n_rgs, 0);  // true decoded bytes per row-group
  uint64_t total_decoded = 0;                  // true decoded bytes, for GB/s
  for (size_t r = 0; r < n_rgs; ++r) {
    const RowGroup &rg = rgs[r];
    uint64_t decoded = 0;
    for (uint64_t t = 0; t < rg.total_tokens; ++t)
      decoded += rg.lens[rg.codes[t]];
    if (decoded == 0) {
      fprintf(stderr, "rg %zu decodes to 0 bytes; unsupported\n", r);
      return 2;
    }
    rg_decoded[r] = decoded;
    total_decoded += decoded;
    uint32_t nb = (uint32_t)((rg.total_tokens + TOK_PER_BATCH - 1) / TOK_PER_BATCH);
    rg_nbatches[r] = nb;
    rg_dict_base[r + 1] = rg_dict_base[r] + rg.dict_size;
    rg_out_base[r + 1] = align16(rg_out_base[r] + decoded);
    rg_tok_base[r + 1] = rg_tok_base[r] + rg.total_tokens;
    rg_choff_base[r + 1] = rg_choff_base[r] + (nb + 1);
  }
  const uint64_t total_tokens = rg_tok_base[n_rgs];
  const uint64_t total_out = rg_out_base[n_rgs];  // padded span: buffer + validation
  const uint64_t total_dict_entries = rg_dict_base[n_rgs];
  uint64_t total_batches = 0;
  for (size_t r = 0; r < n_rgs; ++r) total_batches += rg_nbatches[r];

  // Grid-packing confound (flagged in review): sequential/streams/graph round EACH
  // row-group's grid up to a whole block (WARPS_PER_BLOCK warp-slots), so short
  // final batches leave idle warp-slots per row-group; multidict packs adjacent
  // row-groups into shared blocks and rounds up only once. The multidict speedup
  // therefore blends launch amortization WITH this recovered occupancy. Emit the
  // per-row-group rounding waste so the paper claim can show it is negligible for
  // 1--6 MB groups (or attribute the gain to both effects, not launch cost alone).
  uint64_t per_rg_block_warps = 0;  // warp-slots the per-RG grids occupy
  for (size_t r = 0; r < n_rgs; ++r) {
    uint64_t blocks = (rg_nbatches[r] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    per_rg_block_warps += blocks * WARPS_PER_BLOCK;
  }
  const uint64_t grid_pack_waste_warps = per_rg_block_warps - total_batches;

  // Concatenated host blobs (dict blobs get a MAX_TOKEN tail pad for the last
  // entry's wide load, exactly as the single-dict staging does).
  std::vector<uint16_t> codes_all(total_tokens);
  std::vector<uint8_t> lens_all(total_dict_entries + MT, 0);
  std::vector<uint8_t> dict_s8_all((size_t)total_dict_entries * 8 + MT, 0);
  std::vector<uint8_t> dict_padded_all((size_t)total_dict_entries * MT + MT, 0);
  // per-rg 0-based chunk offsets, concatenated
  std::vector<uint64_t> choff_all(rg_choff_base[n_rgs], 0);
  // per-batch metadata
  std::vector<uint64_t> batch_tok_off(total_batches);
  std::vector<uint64_t> batch_out_off(total_batches);
  std::vector<uint32_t> batch_ntok(total_batches);
  std::vector<uint32_t> batch_rg(total_batches);
  // CPU reference. Pre-filled with a nonzero poison byte (also used to clear the
  // device buffer before each mode) so a byte the kernel FAILS to write shows up
  // as a mismatch instead of coincidentally matching a zero-initialised expected
  // byte; the inter-row-group padding gaps keep the poison in both buffers and so
  // compare equal.
  static constexpr uint8_t POISON = 0xA5;
  std::vector<uint8_t> cpu_out(total_out, POISON);

  uint64_t gb = 0;  // running global batch index
  for (size_t r = 0; r < n_rgs; ++r) {
    const RowGroup &rg = rgs[r];
    const uint64_t dbase = rg_dict_base[r];
    const uint64_t obase = rg_out_base[r];
    const uint64_t tbase = rg_tok_base[r];
    // codes
    memcpy(&codes_all[tbase], rg.codes.data(), rg.total_tokens * 2);
    // lens + dict blobs (dict_s8 = first min(len,8) bytes; dict_padded = full)
    for (uint32_t e = 0; e < rg.dict_size; ++e) {
      uint32_t len = rg.lens[e];
      lens_all[dbase + e] = (uint8_t)len;
      uint32_t n8 = len < 8 ? len : 8;
      memcpy(&dict_s8_all[(dbase + e) * 8], &rg.dict_padded[(size_t)e * MT], n8);
      memcpy(&dict_padded_all[(dbase + e) * MT], &rg.dict_padded[(size_t)e * MT], MT);
    }
    // per-token cumulative offsets (0-based within the row-group) + CPU decode
    uint64_t acc = 0;
    const uint64_t cob = rg_choff_base[r];
    choff_all[cob + 0] = 0;
    for (uint64_t t = 0; t < rg.total_tokens; ++t) {
      uint32_t code = rg.codes[t];
      uint32_t len = rg.lens[code];
      memcpy(&cpu_out[obase + acc], &rg.dict_padded[(size_t)code * MT], len);
      acc += len;
      if ((t + 1) % TOK_PER_BATCH == 0)
        choff_all[cob + (t + 1) / TOK_PER_BATCH] = acc;
    }
    choff_all[cob + rg_nbatches[r]] = acc;  // final (possibly short) batch end
    // per-batch metadata
    for (uint32_t b = 0; b < rg_nbatches[r]; ++b) {
      uint64_t t0 = (uint64_t)b * TOK_PER_BATCH;
      uint64_t t1 = std::min<uint64_t>(t0 + TOK_PER_BATCH, rg.total_tokens);
      batch_tok_off[gb] = tbase + t0;
      batch_out_off[gb] = obase + choff_all[cob + b];
      batch_ntok[gb] = (uint32_t)(t1 - t0);
      batch_rg[gb] = (uint32_t)r;
      ++gb;
    }
  }
  if (gb != total_batches) {  // metadata-derivation invariant
    fprintf(stderr, "internal error: staged %llu batches, expected %llu\n",
            (unsigned long long)gb, (unsigned long long)total_batches);
    return 3;
  }

  // ── row-group size stats (uncompressed decoded MB) ────────────────────────
  uint64_t min_rg = UINT64_MAX, max_rg = 0;
  for (size_t r = 0; r < n_rgs; ++r) {
    uint64_t sz = rg_decoded[r];
    min_rg = std::min(min_rg, sz);
    max_rg = std::max(max_rg, sz);
  }
  double mean_rg_mb = (double)total_decoded / n_rgs / 1e6;

  // ── upload ────────────────────────────────────────────────────────────────
  uint16_t *d_codes;
  uint8_t *d_lens, *d_s8, *d_padded, *d_out;
  uint64_t *d_choff, *d_btok, *d_bout, *d_rgbase;
  uint32_t *d_bntok, *d_brg;
  CK(cudaMalloc(&d_codes, codes_all.size() * 2));
  CK(cudaMalloc(&d_lens, lens_all.size()));
  CK(cudaMalloc(&d_s8, dict_s8_all.size()));
  CK(cudaMalloc(&d_padded, dict_padded_all.size()));
  CK(cudaMalloc(&d_out, total_out + 64));
  CK(cudaMalloc(&d_choff, choff_all.size() * 8));
  CK(cudaMalloc(&d_btok, batch_tok_off.size() * 8));
  CK(cudaMalloc(&d_bout, batch_out_off.size() * 8));
  CK(cudaMalloc(&d_bntok, batch_ntok.size() * 4));
  CK(cudaMalloc(&d_brg, batch_rg.size() * 4));
  CK(cudaMalloc(&d_rgbase, rg_dict_base.size() * 8));
  CK(cudaMemcpy(d_codes, codes_all.data(), codes_all.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_lens, lens_all.data(), lens_all.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_s8, dict_s8_all.data(), dict_s8_all.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_padded, dict_padded_all.data(), dict_padded_all.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_choff, choff_all.data(), choff_all.size() * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_btok, batch_tok_off.data(), batch_tok_off.size() * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bout, batch_out_off.data(), batch_out_off.size() * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bntok, batch_ntok.data(), batch_ntok.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_brg, batch_rg.data(), batch_rg.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rgbase, rg_dict_base.data(), rg_dict_base.size() * 8, cudaMemcpyHostToDevice));

  // ── launchers ─────────────────────────────────────────────────────────────
  // sequential/streams/graph: one shipped single-dict launch per row-group,
  // pointers rebased into the concatenated blobs -- byte-identical to the real
  // per-chunk harness path (0-based chunk offsets, its own output slice).
  auto launch_rg = [&](size_t r, cudaStream_t s) {
    const RowGroup &rg = rgs[r];
    uint32_t grid = (rg_nbatches[r] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    onpair_shmem_4tpt_split8read<<<grid, BLOCK_THREADS, 0, s>>>(
        d_codes + rg_tok_base[r], d_choff + rg_choff_base[r],
        d_s8 + rg_dict_base[r] * 8, d_padded + rg_dict_base[r] * (uint64_t)MT,
        d_lens + rg_dict_base[r], d_out + rg_out_base[r], rg.total_tokens);
  };
  auto launch_sequential = [&]() {
    for (size_t r = 0; r < n_rgs; ++r) launch_rg(r, 0);
  };
  // Timing events, created before the launchers so `launch_streams` can use the
  // start event (ev0) as an on-device barrier. ev0/ev1 bracket the timed region on
  // the default stream (0); every mode below runs so that ev1 waits for its GPU
  // work with NO host synchronization inside the interval.
  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0));
  CK(cudaEventCreate(&ev1));
  // Non-blocking worker streams: they must NOT implicitly serialize against the
  // legacy default stream (0) — the explicit events below are the only ordering,
  // so the K grids can actually overlap.
  std::vector<cudaStream_t> streams(n_streams);
  std::vector<cudaEvent_t> str_done(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    CK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    CK(cudaEventCreate(&str_done[i]));
  }
  auto launch_streams = [&]() {
    // Each worker waits for the start event (ev0, recorded on stream 0 by the
    // timer), round-robins its share of the launches, then records a completion
    // event that stream 0 waits on. The closing event ev1 (on stream 0) therefore
    // brackets the true GPU span, unlike a host cudaStreamSynchronize inside the
    // timed region, which would fold host wakeup/resubmit latency into the reported
    // time and bias this mode's GB/s.
    for (int i = 0; i < n_streams; ++i) CK(cudaStreamWaitEvent(streams[i], ev0, 0));
    for (size_t r = 0; r < n_rgs; ++r) launch_rg(r, streams[r % n_streams]);
    for (int i = 0; i < n_streams; ++i) {
      CK(cudaEventRecord(str_done[i], streams[i]));
      CK(cudaStreamWaitEvent(0, str_done[i], 0));
    }
  };
  // one-grid multi-dictionary launch spanning every row-group
  auto launch_multidict = [&]() {
    uint64_t grid = (total_batches + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    onpair_shmem_4tpt_multidict<<<(unsigned)grid, BLOCK_THREADS>>>(
        d_codes, d_s8, d_padded, d_lens, d_out, d_btok, d_bout, d_bntok, d_brg,
        d_rgbase, total_batches);
  };

  // ── CUDA graph capture of the sequential launches ─────────────────────────
  cudaStream_t gstream;
  CK(cudaStreamCreate(&gstream));
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CK(cudaStreamBeginCapture(gstream, cudaStreamCaptureModeThreadLocal));
  for (size_t r = 0; r < n_rgs; ++r) launch_rg(r, gstream);
  CK(cudaStreamEndCapture(gstream, &graph));
  CK(cudaGraphInstantiate(&graph_exec, graph, 0));  // CUDA 12.x 3-arg (flags) form
  // Replay on the DEFAULT stream (0), the same stream the timing events bracket,
  // so cudaEventElapsedTime measures the graph's GPU execution. (The graph was
  // captured on gstream; an instantiated graph is stream-agnostic to launch.)
  auto launch_graph = [&]() { CK(cudaGraphLaunch(graph_exec, 0)); };

  // ── validation: run each mode once into a poisoned buffer, compare to CPU ──
  auto validate = [&](auto &&launch) -> bool {
    CK(cudaMemset(d_out, POISON, total_out + 64));
    // Barrier BEFORE launching: the poison memset is async on stream 0, but the
    // streams mode's workers are non-blocking and (in this untimed path) wait on an
    // as-yet-unrecorded ev0 — i.e. nothing — so without this sync the memset could
    // race and overwrite their output. Timed iterations are safe (ev0 is recorded
    // then); validation is not on the hot path, so the extra sync is free.
    CK(cudaDeviceSynchronize());
    launch();
    CK(cudaDeviceSynchronize());  // covers every stream, incl. the K stream fan-out
    CK(cudaGetLastError());
    std::vector<uint8_t> gpu_out(total_out);
    CK(cudaMemcpy(gpu_out.data(), d_out, total_out, cudaMemcpyDeviceToHost));
    return memcmp(gpu_out.data(), cpu_out.data(), total_out) == 0;
  };
  bool seq_ok = validate(launch_sequential);
  bool str_ok = validate(launch_streams);
  bool grf_ok = validate(launch_graph);
  bool mdt_ok = validate(launch_multidict);

  // ── timing (CUDA events; min over iters; raw ns retained) ─────────────────
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

  std::vector<uint64_t> seq_ns, str_ns, grf_ns, mdt_ns;
  double t_seq = time_min_raw(launch_sequential, seq_ns);
  double t_str = time_min_raw(launch_streams, str_ns);
  double t_grf = time_min_raw(launch_graph, grf_ns);
  double t_mdt = time_min_raw(launch_multidict, mdt_ns);

  auto gbps = [&](double ms) { return total_decoded / (ms / 1e3) / 1e9; };

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));

  auto print_ns = [](const char *key, const std::vector<uint64_t> &v) {
    printf("  \"%s\": [", key);
    for (size_t i = 0; i < v.size(); ++i)
      printf("%s%llu", i ? "," : "", (unsigned long long)v[i]);
    printf("],\n");
  };

  printf("{\n");
  printf("  \"gpu\": \"%s\",\n", prop.name);
  printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
  printf("  \"kernel\": \"onpair_shmem_4tpt_split8read+multidict\",\n");
  printf("  \"n_row_groups\": %zu,\n", n_rgs);
  printf("  \"total_tokens\": %llu,\n", (unsigned long long)total_tokens);
  printf("  \"total_batches\": %llu,\n", (unsigned long long)total_batches);
  printf("  \"per_rg_block_warps\": %llu,\n", (unsigned long long)per_rg_block_warps);
  printf("  \"grid_pack_waste_warps\": %llu,\n", (unsigned long long)grid_pack_waste_warps);
  printf("  \"grid_pack_waste_frac\": %.4f,\n",
         total_batches ? (double)grid_pack_waste_warps / (double)total_batches : 0.0);
  printf("  \"total_decoded_bytes\": %llu,\n", (unsigned long long)total_decoded);
  printf("  \"total_dict_entries\": %llu,\n", (unsigned long long)total_dict_entries);
  printf("  \"mean_rg_mb\": %.3f,\n", mean_rg_mb);
  printf("  \"min_rg_bytes\": %llu,\n", (unsigned long long)min_rg);
  printf("  \"max_rg_bytes\": %llu,\n", (unsigned long long)max_rg);
  printf("  \"iters\": %d,\n", iters);
  printf("  \"n_streams\": %d,\n", n_streams);
  printf("  \"sequential_ok\": %s,\n", seq_ok ? "true" : "false");
  printf("  \"streams_ok\": %s,\n", str_ok ? "true" : "false");
  printf("  \"graph_ok\": %s,\n", grf_ok ? "true" : "false");
  printf("  \"multidict_ok\": %s,\n", mdt_ok ? "true" : "false");
  printf("  \"sequential_ms\": %.5f,\n", t_seq);
  printf("  \"streams_ms\": %.5f,\n", t_str);
  printf("  \"graph_ms\": %.5f,\n", t_grf);
  printf("  \"multidict_ms\": %.5f,\n", t_mdt);
  print_ns("sequential_ns_iters", seq_ns);
  print_ns("streams_ns_iters", str_ns);
  print_ns("graph_ns_iters", grf_ns);
  print_ns("multidict_ns_iters", mdt_ns);
  printf("  \"sequential_gbps\": %.2f,\n", gbps(t_seq));
  printf("  \"streams_gbps\": %.2f,\n", gbps(t_str));
  printf("  \"graph_gbps\": %.2f,\n", gbps(t_grf));
  printf("  \"multidict_gbps\": %.2f,\n", gbps(t_mdt));
  printf("  \"multidict_speedup_over_sequential\": %.3f\n", t_seq / t_mdt);
  printf("}\n");

  fprintf(stderr,
          "\n=== batch_decode summary (%s) ===\n"
          "%zu row-groups, mean %.2f MB, total %.1f MB decoded (%llu batches)\n"
          "sequential %.1f GB/s | streams(%d) %.1f GB/s | graph %.1f GB/s | "
          "multidict %.1f GB/s (%.2fx over sequential)\n"
          "byte-exact: seq=%s streams=%s graph=%s multidict=%s\n",
          prop.name, n_rgs, mean_rg_mb, total_decoded / 1e6,
          (unsigned long long)total_batches, gbps(t_seq), n_streams, gbps(t_str),
          gbps(t_grf), gbps(t_mdt), t_seq / t_mdt, seq_ok ? "YES" : "NO",
          str_ok ? "YES" : "NO", grf_ok ? "YES" : "NO", mdt_ok ? "YES" : "NO");

  bool all_ok = seq_ok && str_ok && grf_ok && mdt_ok;
  return all_ok ? 0 : 4;
}
