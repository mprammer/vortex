// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OnPair chunked-array compression benchmark CLI.
//!
//! Two subcommands:
//! * `gen-tpch` — generate every TPC-H table parquet for a scale factor.
//! * `run` — sample a column, OnPair-compress it across a `bits × chunk ×
//!   threshold` matrix into Vortex files, verify the string round-trip, and
//!   print the per-cell results as JSON to stdout.
//!
//! The Python orchestrator (`benchmarks/onpair-bench/run.py`) drives this
//! binary across a registry of datasets/columns. The binary itself is
//! dataset-agnostic: point `--parquet` at any parquet file and `--column` at
//! any string column.

#![expect(clippy::print_stdout)]

use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use vortex_bench::onpair_bench::GpuBenchmarkConfig;
use vortex_bench::onpair_bench::ensure_tpch_all_parquet;
use vortex_bench::onpair_bench::run_column;
use vortex_bench::onpair_bench::run_vortex_gpu_decode;
use vortex_bench::setup_logging_and_tracing;

const MB: u64 = 1 << 20;

#[derive(Parser)]
#[command(name = "onpair-chunk-bench")]
#[command(about = "OnPair chunked-array compression benchmark")]
struct Args {
    #[command(subcommand)]
    command: Command,

    /// Enable verbose logging.
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Generate every TPC-H table parquet for a scale factor (idempotent).
    GenTpch {
        /// Scale factor (e.g. 10).
        #[arg(long, default_value_t = 10.0)]
        sf: f64,
        /// Output directory; tables land in `<out-dir>/parquet/<table>_0.parquet`.
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Generate every TPC-DS table parquet for a scale factor via DuckDB
    /// `dsdgen` (idempotent). Requires the `duckdb` CLI on PATH.
    GenTpcds {
        /// Scale factor (e.g. 10).
        #[arg(long, default_value_t = 10.0)]
        sf: f64,
        /// Output directory; tables land in `<out-dir>/parquet/<table>.parquet`.
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Generate a deterministic synthetic ClickBench-style URL corpus as a
    /// single-column (`url`) parquet (idempotent). Reproduces the seed-123
    /// generator the original `onpair_cuda` micro-benchmark used.
    GenSynthUrls {
        /// Number of URL rows to generate.
        #[arg(long, default_value_t = 10_000_000)]
        rows: usize,
        /// Distinct filler path segments to draw from, widening the OnPair dictionary.
        /// 0 (the default) reproduces the original seed-123 corpus byte-for-byte; any
        /// positive value yields a different corpus and must use its own output path.
        #[arg(long, default_value_t = 0)]
        vocab: usize,
        /// Output parquet path (single Utf8 column named `url`).
        #[arg(long)]
        out: PathBuf,
    },
    /// Compress one column across the matrix and emit JSON results.
    Run {
        /// Source parquet file.
        #[arg(long)]
        parquet: PathBuf,
        /// String column to compress.
        #[arg(long)]
        column: String,
        /// Stable dataset id used in the output path and results.
        #[arg(long)]
        dataset_id: String,
        /// OnPair dictionary bit widths.
        #[arg(long, value_delimiter = ',', default_value = "12,16")]
        bits: Vec<u32>,
        /// Per-chunk uncompressed byte budgets (default 1MB,10MB,100MB,1000MB).
        #[arg(long, value_delimiter = ',', default_values_t = [MB, 10 * MB, 100 * MB, 1000 * MB])]
        chunk_bytes: Vec<u64>,
        /// OnPair training thresholds.
        #[arg(long, value_delimiter = ',', default_value = "0.2")]
        threshold: Vec<f64>,
        /// OnPair training-shuffle seed. 0 preserves the historical random-device behavior;
        /// a nonzero seed makes the trained dictionary reproducible.
        #[arg(long, default_value_t = 0)]
        training_seed: u64,
        /// Raw-payload sample cap (default ~1GB).
        #[arg(long, default_value_t = 1_000_000_000)]
        sample_bytes: u64,
        /// Approximate per-file on-disk target (default ~200MB).
        #[arg(long, default_value_t = 200 * MB)]
        file_target_bytes: u64,
        /// Output root for the `.vortex` files.
        #[arg(long)]
        out_dir: PathBuf,
        /// Also benchmark CUDA kernel-only OnPair decompression.
        #[arg(long)]
        gpu_decode: bool,
        /// Timed CUDA iterations for each applicable kernel.
        #[arg(long, default_value_t = 10)]
        gpu_iters: u64,
        /// Copy GPU output bytes back and compare every applicable kernel against CPU decode.
        #[arg(long)]
        gpu_validate: bool,
        /// Exact CUDA kernel allowlist, comma-separated. Preserves order and rejects
        /// duplicates/unknown names. Use `tpt-matched` for the controlled ten-way comparison.
        #[arg(long, value_delimiter = ',')]
        gpu_kernels: Option<Vec<String>>,
        /// Stored codec: `onpair` (default) or `fsst12`. FSST-12 is a 12-bit codec, so
        /// --bits and --threshold do not apply to it and are ignored.
        #[arg(long, default_value = "onpair")]
        codec: String,
    },
    /// Run CUDA OnPair decode directly from existing `.vortex` files.
    GpuDecodeVortex {
        /// Existing `.vortex` file or directory containing `*.vortex` parts.
        #[arg(long, value_name = "FILE_OR_DIR", required = true)]
        vortex: Vec<PathBuf>,
        /// OnPair string column to extract from each file.
        #[arg(long)]
        column: String,
        /// Timed CUDA iterations for each applicable kernel.
        #[arg(long, default_value_t = 10)]
        gpu_iters: u64,
        /// Copy GPU output bytes back and compare every applicable kernel against CPU decode.
        #[arg(long)]
        gpu_validate: bool,
        /// Exact CUDA kernel allowlist, comma-separated; `tpt-matched` selects the
        /// controlled ten-way comparison.
        #[arg(long, value_delimiter = ',')]
        gpu_kernels: Option<Vec<String>>,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing(args.verbose, false)?;

    match args.command {
        Command::GenTpch { sf, out_dir } => {
            ensure_tpch_all_parquet(sf, &out_dir).await?;
            eprintln!("TPC-H (sf={sf}) ready under {}/parquet", out_dir.display());
        }
        Command::GenTpcds { sf, out_dir } => {
            vortex_bench::tpcds::duckdb::generate_tpcds(out_dir.clone(), format!("{sf}"))?;
            eprintln!("TPC-DS (sf={sf}) ready under {}/parquet", out_dir.display());
        }
        Command::GenSynthUrls { rows, vocab, out } => {
            gen_synth_urls(rows, vocab, &out)?;
            eprintln!(
                "synthetic URLs ({rows} rows, vocab {vocab}) ready at {}",
                out.display()
            );
        }
        Command::Run {
            parquet,
            column,
            dataset_id,
            bits,
            chunk_bytes,
            threshold,
            training_seed,
            sample_bytes,
            file_target_bytes,
            out_dir,
            gpu_decode,
            gpu_iters,
            gpu_validate,
            gpu_kernels,
            codec,
        } => {
            let results = run_column(
                &dataset_id,
                &parquet,
                &column,
                &bits,
                &chunk_bytes,
                &threshold,
                training_seed,
                sample_bytes,
                file_target_bytes,
                &out_dir,
                gpu_decode.then_some(GpuBenchmarkConfig {
                    iterations: gpu_iters,
                    validate: gpu_validate,
                    kernels: gpu_kernels.map(Into::into),
                }),
                &codec,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Command::GpuDecodeVortex {
            vortex,
            column,
            gpu_iters,
            gpu_validate,
            gpu_kernels,
        } => {
            let files = collect_vortex_files(&vortex)?;
            let result = run_vortex_gpu_decode(
                &files,
                &column,
                GpuBenchmarkConfig {
                    iterations: gpu_iters,
                    validate: gpu_validate,
                    kernels: gpu_kernels.map(Into::into),
                },
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Synthetic ClickBench-style URL corpus.
//
// Mirrors `vortex_fsst::test_utils::generate_clickbench_urls` (seed 123) — the
// deterministic generator the original (since-removed) `onpair_cuda` micro-benchmark
// used for its synthetic 10M-URL workload. Inlined so the corpus is regenerable from
// this binary alone. Determinism is exact for the committed `Cargo.lock`: StdRng's
// output stream is not guaranteed stable across `rand` major versions, so regenerating
// after a `cargo update` may shift the corpus (pin `rand` or checksum if that matters).
// ---------------------------------------------------------------------------

const CB_DOMAINS: &[&str] = &[
    "www.google.com",
    "yandex.ru",
    "mail.ru",
    "vk.com",
    "www.youtube.com",
    "www.facebook.com",
    "ok.ru",
    "go.mail.ru",
    "www.avito.ru",
    "pogoda.yandex.ru",
    "news.yandex.ru",
    "maps.yandex.ru",
    "market.yandex.ru",
    "afisha.yandex.ru",
    "auto.ru",
    "www.kinopoisk.ru",
    "www.ozon.ru",
    "www.wildberries.ru",
    "aliexpress.ru",
    "lenta.ru",
];

const CB_PATHS: &[&str] = &[
    "/search",
    "/catalog/electronics/smartphones",
    "/product/item/123456789",
    "/news/2024/03/15/article-about-technology",
    "/user/profile/settings/notifications",
    "/api/v2/catalog/search",
    "/checkout/cart/summary",
    "/blog/2024/how-to-optimize-database-queries-for-better-performance",
    "/category/home-and-garden/furniture/tables",
    "/",
];

const CB_PARAMS: &[&str] = &[
    "?utm_source=google&utm_medium=cpc&utm_campaign=spring_sale_2024&utm_content=banner_v2",
    "?q=buy+smartphone+online+cheap+free+shipping&category=electronics&sort=price_asc&page=3",
    "?ref=main_page_carousel_block_position_4&sessionid=abc123def456",
    "?from=tabbar&clid=2270455&text=weather+forecast+tomorrow",
    "?lr=213&msid=1234567890.12345&suggest_reqid=abcdef&csg=12345",
    "",
    "",
    "",
    "?page=1&per_page=20",
    "?source=serp&forceshow=1",
];

const CB_FRAGMENTS: &[&str] = &[
    "",
    "",
    "",
    "#section-reviews",
    "#comments",
    "#price-history",
    "",
    "",
    "",
    "",
];

const VOCAB_SEGMENT_WIDTH: usize = 3;
const VOCAB_ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
const VOCAB_CAPACITY: usize = 36 * 36 * 36;
const VOCAB_RNG_SEED: u64 = 456;
const VOCAB_DIVERSE_EVERY: u64 = 8;

/// One fixed-width, base-36 filler segment. Every positive rung adds exactly the same
/// number of bytes per row, so vocabulary size does not also become a row-length knob.
fn vocab_segment(mut i: usize) -> String {
    let mut s = String::with_capacity(VOCAB_SEGMENT_WIDTH);
    for _ in 0..VOCAB_SEGMENT_WIDTH {
        s.push(VOCAB_ALPHABET[i % VOCAB_ALPHABET.len()] as char);
        i /= VOCAB_ALPHABET.len();
    }
    s
}

/// Map one filler RNG draw to the pool. Keeping seven-eighths of the probability mass on
/// a common sentinel lets `vocab` vary tail diversity without also varying its frequency.
fn vocab_index(draw: u64, pool_len: u64) -> Result<usize> {
    anyhow::ensure!(pool_len > 0, "synthetic URL vocabulary must be nonzero");
    if pool_len == 1 || !draw.is_multiple_of(VOCAB_DIVERSE_EVERY) {
        return Ok(0);
    }
    Ok(1 + usize::try_from((draw / VOCAB_DIVERSE_EVERY) % (pool_len - 1))?)
}

/// Deterministic ClickBench-style URL generator (seed 123), deterministic for the
/// committed `Cargo.lock`. Mirrors `vortex_fsst::test_utils::generate_clickbench_urls`.
///
/// `vocab` widens the token vocabulary — and so the OnPair dictionary — by selecting one
/// extra path segment per row from a pool of `vocab` distinct strings. It exists to make
/// the kernel selector's dictionary-size thresholds *identifiable*. The measured corpus
/// clusters near 870, 4096, and 65536 entries with nothing in between, so a threshold
/// placed inside those gaps classifies no observation and cannot be fitted from data at
/// all; the Ada rule in `onpair_bench.rs` currently sits in exactly such a gap.
///
/// `vocab == 0` reproduces the original generator **exactly**, drawing the same five
/// random values per row in the same order. That is load-bearing rather than tidy: the
/// committed `synthetic/url` cell must stay byte-identical or it stops being comparable
/// with every measurement already in the artifact.
///
/// Positive rungs use a second RNG, so the scheme/domain/path/params/fragment stream is
/// also identical between rungs. Every row receives a fixed-width segment: seven-eighths
/// use the common pool entry and one-eighth draw from the diverse tail. Holding that
/// diversity mass fixed makes `vocab` a smoother dictionary-cardinality control than a
/// uniform pool, which abruptly fills the dictionary at campaign chunk sizes. A raw draw
/// is mapped without range-sampling rejection, so every positive rung consumes exactly
/// one filler draw per row.
fn generate_clickbench_urls(n: usize, vocab: usize) -> Result<Vec<String>> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;

    anyhow::ensure!(
        vocab <= VOCAB_CAPACITY,
        "synthetic URL vocabulary {vocab} exceeds the {VOCAB_CAPACITY}-segment fixed-width capacity"
    );
    let pool: Vec<String> = (0..vocab).map(vocab_segment).collect();
    let mut rng = StdRng::seed_from_u64(123);
    let mut vocab_rng = StdRng::seed_from_u64(VOCAB_RNG_SEED);
    let pool_len = u64::try_from(pool.len())?;
    (0..n)
        .map(|_| {
            let scheme = if rng.random_bool(0.7) {
                "https"
            } else {
                "http"
            };
            let domain = CB_DOMAINS[rng.random_range(0..CB_DOMAINS.len())];
            let path = CB_PATHS[rng.random_range(0..CB_PATHS.len())];
            let params = CB_PARAMS[rng.random_range(0..CB_PARAMS.len())];
            let fragment = CB_FRAGMENTS[rng.random_range(0..CB_FRAGMENTS.len())];
            if pool.is_empty() {
                // No sixth draw: the RNG stream stays identical to the original generator.
                Ok(format!("{scheme}://{domain}{path}{params}{fragment}"))
            } else {
                let idx = vocab_index(vocab_rng.random::<u64>(), pool_len)?;
                let seg = &pool[idx];
                Ok(format!("{scheme}://{domain}{path}/{seg}{params}{fragment}"))
            }
        })
        .collect()
}

/// Generate `rows` synthetic URLs and write them as a single-column (`url`)
/// parquet at `out` (idempotent; a no-op if `out` already exists).
fn gen_synth_urls(rows: usize, vocab: usize, out: &Path) -> Result<()> {
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use parquet::arrow::ArrowWriter;

    if out.exists() {
        return Ok(());
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let urls = generate_clickbench_urls(rows, vocab)?;
    let schema = Arc::new(Schema::new(vec![Field::new("url", DataType::Utf8, false)]));

    let tmp = out.with_extension("parquet.part");
    let file = std::fs::File::create(&tmp)?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None)?;
    // Bounded row groups so we never hold two full copies of the corpus.
    for chunk in urls.chunks(1_000_000) {
        let arr = StringArray::from_iter(chunk.iter().map(|s| Some(s.as_str())));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(arr)])?;
        writer.write(&batch)?;
    }
    writer.close()?;
    std::fs::rename(&tmp, out)?;
    Ok(())
}

fn collect_vortex_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut entries = std::fs::read_dir(path)?
                .map(|entry| entry.map(|e| e.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            entries.sort();
            files.extend(entries.into_iter().filter(|p| {
                p.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext == "vortex")
            }));
        } else {
            files.push(path.clone());
        }
    }

    if files.is_empty() {
        anyhow::bail!("no .vortex files found");
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare element-by-element with the still-maintained mirror of the original
    /// pre-2026-08-17 implementation. One million rows exercises the RNG stream far past
    /// a shape or spot-check without making the ordinary test suite materialise all 10M.
    #[test]
    fn vocab_zero_preserves_the_original_corpus() -> Result<()> {
        const ROWS: usize = 1_000_000;
        let actual = generate_clickbench_urls(ROWS, 0)?;
        let original = vortex_fsst::test_utils::generate_clickbench_urls(ROWS);
        assert_eq!(actual.len(), original.len());
        for (row, (actual, original)) in actual.iter().zip(&original).enumerate() {
            assert_eq!(actual, original, "synthetic URL changed at row {row}");
        }
        Ok(())
    }

    /// A positive rung may add only its fixed-width segment; drawing the segment must not
    /// perturb any of the five base URL choices on this or a later row.
    #[test]
    fn filler_rng_does_not_shift_the_base_corpus() -> Result<()> {
        let base = generate_clickbench_urls(20_000, 0)?;
        let widened = generate_clickbench_urls(20_000, 3584)?;
        for (row, (base, widened)) in base.iter().zip(&widened).enumerate() {
            let only_inserts_one_segment = (0..=base.len()).any(|at| {
                widened.as_bytes().get(at) == Some(&b'/')
                    && widened.get(..at) == base.get(..at)
                    && widened.get(at + 1 + VOCAB_SEGMENT_WIDTH..) == base.get(at..)
            });
            assert!(
                only_inserts_one_segment,
                "base URL stream changed at row {row}"
            );
        }
        Ok(())
    }

    /// Distinct pool indices must give distinct segments, otherwise the pool silently
    /// collapses and a rung of the ladder measures the same corpus as a lower one.
    #[test]
    fn vocab_segments_are_distinct() {
        use std::collections::HashSet;
        let segs: HashSet<String> = (0..VOCAB_CAPACITY).map(vocab_segment).collect();
        assert_eq!(segs.len(), VOCAB_CAPACITY);
        assert!(segs.iter().all(|seg| seg.len() == VOCAB_SEGMENT_WIDTH));
        assert!(generate_clickbench_urls(1, VOCAB_CAPACITY + 1).is_err());
    }

    #[test]
    fn sparse_vocab_keeps_diversity_mass_fixed() -> Result<()> {
        const POOL_LEN_U64: u64 = 512;
        const POOL_LEN_USIZE: usize = 512;
        let indices = (0..8_000)
            .map(|draw| vocab_index(draw, POOL_LEN_U64))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(indices.iter().filter(|&&idx| idx != 0).count(), 1_000);
        assert!(indices.iter().all(|&idx| idx < POOL_LEN_USIZE));
        assert!((0..8_000).all(|draw| matches!(vocab_index(draw, 1), Ok(0))));
        Ok(())
    }
}
