# E3: Byte-exact direct-store drain counterfactual

`onpair_shmem_4tpt_directstore` is the byte-exact counterfactual for the
staged aligned drain in `onpair_shmem_4tpt`. It preserves the production
kernel's 128-token warp batch, four tokens per thread, padded stride-16
dictionary `uint4` gather, and four warp prefix scans. The production kernel
then writes decoded token bytes to a per-warp shared-memory buffer and flushes
that buffer through an aligned, coalesced head/body/tail drain. This variant
does not allocate or write that staging buffer. Each lane instead uses the
same computed exclusive offset to copy its token directly from its gathered
`uint4` to global output.

## Byte-exactness

For token `k`, the direct-store address is
`chunk_offsets[chunk] + excl[k]`, where `excl[k]` is produced by the same
four-prefix-scan sequence as the shipped kernel. Its loop writes only indices
`0..len`, guarded per byte, so it never issues the 16-byte over-copy used by
the CPU decoder or any write beyond the token's logical extent. Prefix-sum
intervals for nonzero tokens are consecutive and disjoint inside a chunk; the
precomputed chunk offsets make adjacent chunk intervals consecutive and
disjoint too. Therefore no pair of lanes can race on an output byte, including
between neighbouring short tokens. This differs materially from
`onpair_shmem_4tpt_ablate_nodrain`, which is a timing-only proxy and does not
produce output bytes for validation.

## Expected result

`directstore` should be slower than `onpair_shmem_4tpt`: its per-byte global
stores have token-dependent addresses and cannot use the production drain's
aligned `uint4` coalescing. The slowdown is the measured benefit of staging
and draining, while retaining the gather and scan work in both variants.

## Build and run checklist (B300)

1. Use a CUDA Toolkit supported by the B300 and confirm `nvcc --version` and
   the driver can target the device. The crate's build script uses
   `-arch=native`, so build on the B300 itself:

   ```bash
   cargo build -p vortex-bench --features cuda --bin onpair-chunk-bench --release
   ```

2. Benchmark existing 1,000 MB OnPair cells with validation enabled. The CLI
   enumerates `onpair_shmem_4tpt_directstore` through `GPU_KERNELS`, times it,
   and compares its output byte-for-byte with CPU decode:

   ```bash
   ONPAIR_FAST=1 target/release/onpair-chunk-bench gpu-decode-vortex \
     --vortex <chunk1000mb-vortex-file-or-directory> \
     --column <column> --gpu-iters 100 --gpu-validate
   ```

   To regenerate the matching matrix from the registered source columns, use:

   ```bash
   python benchmarks/onpair-bench/run.py --gpu-decode --gpu-validate \
     --gpu-iters 100 --chunk-mb 1000 --bits 12,16 --threshold 0.2 \
     --datasets fineweb,wikipedia,book-reviews,tpch-sf10 \
     --columns text,ps_comment,l_comment
   ```

3. Run the bits12 `text` cells for FineWeb, Wikipedia, and Book Reviews, then
   the bits12 and bits16 `ps_comment` and `l_comment` TPC-H SF10 cells. These
   cover the small/large dictionary and short/long-token regimes used by the
   existing GPU evaluation. Keep the preset at threshold `0.2` and use the
   same cell for the shipped and counterfactual variants.

4. A valid result has an applicable
   `onpair_shmem_4tpt_directstore` row with `verified: true` (and no
   `validation_error`). For each cell report
   `X = 100 * (directstore_decode_ms / onpair_shmem_4tpt_decode_ms - 1)`.
   The expected result is `X > 0`: directstore passes validation and is `X%`
   slower, quantifying the staged drain's benefit. Preserve the raw
   `decode_ns_iters` in the JSON output for the final reduction.

## Unverified assumption

This machine has neither CUDA tooling nor a GPU, so this change has not been
compiled or executed here. The implementation assumes the existing build
script continues to compile every `kernels/src/*.cu` file into its own PTX
module, as it does for the neighbouring variants.
