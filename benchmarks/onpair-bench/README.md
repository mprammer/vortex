<!--
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->
# FastPair / OnPair GPU decode — experimental harness

> **Frozen experimental harness.** This commit is the pinned snapshot of the benchmark harness
> behind the paper *"FastPair: Towards Architecture-Optimal String Decompression."* The GPU decode
> kernels here are **byte-identical** to the revision lineage that produced the paper's committed
> results, so a fresh run reproduces the published throughput. The committed raw results, the figure
> generators, and a hardware-free `make verify` live in the **FastPair reproducibility-artifacts**
> repository; this branch is what regenerates that raw data on a GPU box.

The harness OnPair-compresses string columns into Vortex `ChunkedArray`s (one dictionary per chunk),
then times **CUDA kernel-only** decode across the tokens-per-thread kernel family with byte-exact
validation, and runs the two baselines the paper compares against — the Blackwell **hardware
Decompression Engine** and **software nvCOMP-Zstd** — plus the **end-to-end decode→scan** operator.

## Prerequisites (run on the GPU box)

Build *on* the target GPU — kernels compile with `-arch=native`, which targets the present device.

- **NVIDIA GPU + driver**, and the **CUDA Toolkit** with `nvcc` on `PATH`. Build targets CUDA ≥ 12.x;
  the nvCOMP 5.1 SDK and the standalone DE bench want **≥ 12.8**.
- **C++20 compiler** (GCC ≥ 11 / Clang ≥ 13) and **CMake ≥ 3.21** — for the OnPair C++ codec.
- **libclang** (for `bindgen`). If it isn't auto-found, set it: `export LIBCLANG_PATH=$(llvm-config --libdir)`.
- **Rust** — pinned by `rust-toolchain.toml` (1.91.0); `rustup` installs it automatically.
- **Python ≥ 3.11 + [uv](https://docs.astral.sh/uv/)** (the orchestrator is a uv project), or plain
  Python with `pip install 'pyarrow>=15'`.
- **Network on first build:** the build auto-fetches the **nvCOMP SDK** (from NVIDIA) and the **OnPair
  C++ reference** (`gargiulofrancesco/onpair_cpp`, pinned to a commit SHA) via CMake `FetchContent`.
  Both are cached after the first build. Plan ~5 min for the first `--features cuda` build.

## Reproduce the paper's numbers

### 1. GPU decode throughput (the headline)

```bash
# All datasets, bits 12+16, 1000 MB chunks, 100 timed iterations, byte-exact validated:
python benchmarks/onpair-bench/run.py --gpu-decode --gpu-validate --gpu-iters 100 --chunk-mb 1000

# Quick check on locally-generated data only (no downloads):
python benchmarks/onpair-bench/run.py --gpu-decode --gpu-validate --gpu-iters 100 \
    --chunk-mb 1000 --datasets tpch-sf10,synthetic
```

`--gpu-decode` builds with `--features cuda` and times every applicable kernel; `--gpu-validate`
copies each kernel's output back and byte-compares it against the CPU decode (failing kernels are
excluded). The best **shipped** kernel per cell (the `*ablate*` instrumentation builds are always
excluded from "best") is what the paper reports. Results land in
`vortex-bench/data/onpair-bench/summary.json` (full per-kernel breakdown, with raw per-iteration
timings reduced to the min — the microbenchmark convention).

> The paper uses **100** iterations (`--gpu-iters 100`); the default is 10 for quick smoke runs.

### 2. Hardware Decompression Engine baseline (Blackwell)

The DE comparison is the standalone `nvcomp_hw_bench.cu`; build + run per its header comment — in
brief, after a `--features cuda` build has fetched the SDK:

```bash
SDK=$(find target -path '*nvcomp-sdk' -type d | head -1)
nvcc -O3 -arch=native -std=c++17 benchmarks/onpair-bench/nvcomp_hw_bench.cu \
     -I"$SDK/include" -L"$SDK/lib" -lnvcomp -o nvcomp_hw_bench
LD_LIBRARY_PATH="$SDK/lib" ./nvcomp_hw_bench <column.bin>
```

It feeds the DE the **identical** uncompressed bytes OnPair decodes (round-trip checked), so the
comparison is apples-to-apples.

### 3. End-to-end decode→scan (the operator hand-off)

Dump the exact encoder output, then run the standalone scan (see the `e2e_scan.cu` header):

```bash
ONPAIR_DUMP_E2E=/tmp/clickbench_url.e2ebin \
  python benchmarks/onpair-bench/run.py --gpu-decode --datasets clickbench --columns URL --chunk-mb 1000
nvcc -O3 -arch=native -std=c++17 benchmarks/onpair-bench/e2e_scan.cu -o e2e_scan
./e2e_scan /tmp/clickbench_url.e2ebin
```

It decodes the column on-device and scans the decoded bytes for a rare string (a unique match, so the
scan must read the whole column), validated against a CPU reference; it exits non-zero unless both
decode and scan are byte-correct.

## Datasets & external provenance

`columns.py` is the registry; data lives **under the repo** (`vortex-bench/data`, resolved relative
to the source — no absolute paths). Each source is generated locally or fetched on first use:

| Source | Origin |
|---|---|
| TPC-H (`tpch-sf10`) | generated locally (`gen-tpch`) |
| `synthetic` URL corpus | generated locally (`gen-synth-urls`, seed 123; deterministic for the committed `Cargo.lock`) |
| ClickBench `hits` | `https://datasets.clickhouse.com/hits_compatible/hits.parquet` |
| FineWeb | HuggingFace `HuggingFaceFW/fineweb` (sample/10BT, v1.4.0) |
| Wikipedia | HuggingFace `wikimedia/wikipedia` (20231101.en) |
| `dbtext` | the FSST paper corpus, `cwida/fsst` |
| `book-reviews`, `amazon-movies`, `amazon-electronics` | Amazon-Reviews-2023 (McAuley Lab, UCSD) — *Books / Movies_and_TV / Electronics* review `text`, streamed on-box by `run.py` (set `HF_TOKEN`). Non-redistributable corpus; never committed. |
| OnPair codec | `encodings/onpair-sys` builds `gargiulofrancesco/onpair_cpp` (pinned SHA) — OnPair, [arXiv:2508.02280](https://arxiv.org/abs/2508.02280) |
| nvCOMP (DE + software-Zstd baselines) | NVIDIA nvCOMP SDK 5.1, auto-downloaded by the build |

For HuggingFace downloads (FineWeb, Wikipedia, the Amazon corpora), set `HF_TOKEN` to avoid rate
limits (sent only to `huggingface.co`). To reuse data already on the box instead of downloading,
point `ONPAIR_LOCAL_<DATASET>` (e.g. `ONPAIR_LOCAL_CLICKBENCH`, `ONPAIR_LOCAL_BOOK_REVIEWS`) at an
absolute parquet path.

The external download links were live as of **2026-06** (the paper's measurement window); each
source's landing page is recorded in `columns.py` if a direct link has since moved. This is a
reproducibility harness, not a data archive — we record best-effort provenance, not a frozen copy of
the corpora.

## Things you may need to fiddle with on a fresh box

- **libclang not found** → `export LIBCLANG_PATH=$(llvm-config --libdir)`.
- **`sccache: Operation not permitted`** (if sccache is globally configured) → `export RUSTC_WRAPPER=`.
- **A newer host compiler** can trip the kernels' `-Werror`; install a matching GCC/Clang or relax it.
- **A column was skipped** — `run.py` streams the Amazon corpora from HuggingFace automatically (set
  `HF_TOKEN` to avoid throttling); any dataset whose source can't be fetched is skipped with a stderr
  note and the run completes on the rest. Use `--datasets`/`--columns` to scope a run.

## Did it work?

`run.py` exits non-zero if any cell fails the round-trip or (with `--gpu-validate`) GPU byte-exact
check, and the table marks each GPU cell `(ok)`/`(bad)`. The e2e bench exits non-zero unless decode
*and* scan are byte-correct. A clean run with all cells `ok` reproduces the harness side of the paper.

## Layout

- `onpair-chunk-bench` (`vortex-bench/src/bin/onpair-chunk-bench.rs`) — the Rust bench: `gen-tpch`,
  `gen-synth-urls`, and `run` (compress + CUDA decode + validate + emit JSON).
- `run.py` — the orchestrator: column registry, ensure-data, drive the binary, aggregate the table.
- `columns.py` — the `(dataset, column)` registry (one-line append to add a column).
- `nvcomp_hw_bench.cu`, `e2e_scan.cu` — the standalone DE and decode→scan benches (above).
- decode kernels: `vortex-cuda/kernels/src/onpair_shmem_4tpt_split8read.cu` (the shipped hero) + family.
