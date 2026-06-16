// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark for the Rust<->host-runtime Arrow boundary (proposal experiment E4).
//!
//! The native Rust path here is the floor; the JVM (`vortex-jni`) and Python (`vortex-python`)
//! runners read the same file with the same consume workload, and the difference is the Arrow
//! C Data Interface boundary overhead. See `README.md`.

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use bytes::Bytes;
use clap::Parser;
use clap::Subcommand;
use futures::StreamExt;
use futures::pin_mut;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteOptionsSessionExt;
use vortex_bench::CompactionStrategy;
use vortex_bench::SESSION;

/// Rust<->JVM/Python Arrow boundary microbenchmark.
#[derive(Parser)]
#[command(about = "Vortex Arrow boundary microbenchmark (experiment E4)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate a synthetic Vortex file (columns: id i64, value f64, category utf8).
    Gen {
        /// Total number of rows to generate.
        #[arg(long, default_value_t = 10_000_000)]
        rows: usize,
        /// Output `.vortex` path; read by every runtime's runner.
        #[arg(long)]
        out: PathBuf,
    },
    /// Read the file fully via the native Rust path and report throughput.
    Bench {
        /// Input `.vortex` file produced by `gen`.
        #[arg(long)]
        file: PathBuf,
        /// Measured iterations (after warmup).
        #[arg(long, default_value_t = 20)]
        iters: usize,
        /// Warmup iterations to discard.
        #[arg(long, default_value_t = 5)]
        warmup: usize,
    },
}

fn make_batch(rows: usize) -> Result<RecordBatch> {
    const CATS: [&str; 5] = ["alpha", "bravo", "charlie", "delta", "echo"];
    let id = Int64Array::from_iter_values(0..rows as i64);
    let value = Float64Array::from_iter_values((0..rows).map(|i| i as f64 * 1.5));
    let category = StringArray::from_iter_values((0..rows).map(|i| CATS[i % CATS.len()]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(id), Arc::new(value), Arc::new(category)],
    )?)
}

async fn write_vortex(array: &ArrayRef, options: VortexWriteOptions) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    options.write(&mut cursor, array.to_array_stream()).await?;
    Ok(buf)
}

async fn generate(rows: usize, out: PathBuf) -> Result<()> {
    let batch = make_batch(rows)?;
    let array = ArrayRef::from_arrow(batch.clone(), false)?;

    // Vortex, default write strategy.
    let buf = write_vortex(&array, SESSION.write_options()).await?;
    std::fs::write(&out, &buf)?;

    // Vortex, compact strategy (BtrBlocks cascading compression) — what we'd store in Iceberg.
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("boundary");
    let compact_path = out.with_file_name(format!("{stem}-compact.vortex"));
    let compact_opts = CompactionStrategy::Compact.apply_options(SESSION.write_options());
    let cbuf = write_vortex(&array, compact_opts).await?;
    std::fs::write(&compact_path, &cbuf)?;

    // Parquet, zstd — what Iceberg actually stores.
    let pq_path = out.with_extension("parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let pq_file = std::fs::File::create(&pq_path)?;
    let mut writer = ArrowWriter::try_new(pq_file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    let pq_len = std::fs::metadata(&pq_path)?.len();

    println!("wrote {rows} rows:");
    println!("  vortex          {:>10} bytes  {}", buf.len(), out.display());
    println!("  vortex-compact  {:>10} bytes  {}", cbuf.len(), compact_path.display());
    println!("  parquet-zstd    {:>10} bytes  {}", pq_len, pq_path.display());
    Ok(())
}

fn report(label: &str, times: &mut [f64], rows: usize, batches: usize) {
    times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    let p50 = times[times.len() / 2];
    let p95 = times[((times.len() as f64 * 0.95) as usize).min(times.len() - 1)];
    let ns_per_batch = p50 / batches.max(1) as f64 * 1e9;
    println!("{label}  rows={rows} batches={batches}");
    println!("  p50={:.3}ms  p95={:.3}ms", p50 * 1e3, p95 * 1e3);
    println!("  {:.1} Mrows/s  {ns_per_batch:.0} ns/batch", rows as f64 / p50 / 1e6);
}

/// Read a Parquet file fully via arrow-rs (in-process), same minimal consume as the Vortex path.
fn bench_parquet(file: PathBuf, iters: usize, warmup: usize) -> Result<()> {
    let data = Bytes::from(std::fs::read(&file)?);
    let mut times = Vec::with_capacity(iters);
    let mut rows = 0usize;
    let mut batches = 0usize;
    for it in 0..(iters + warmup) {
        let reader = ParquetRecordBatchReaderBuilder::try_new(data.clone())?
            .with_batch_size(131072)
            .build()?;
        let start = Instant::now();
        rows = 0;
        batches = 0;
        for batch in reader {
            let batch = batch?;
            rows += batch.num_rows();
            batches += 1;
            black_box(&batch);
        }
        if it >= warmup {
            times.push(start.elapsed().as_secs_f64());
        }
    }
    report("native-parquet", &mut times, rows, batches);
    Ok(())
}

/// Sum the `value` column and count rows/batches to force full materialization across the boundary.
async fn bench(file: PathBuf, iters: usize, warmup: usize) -> Result<()> {
    let data = Bytes::from(std::fs::read(&file)?);

    let schema = {
        let scan = SESSION.open_options().open_buffer(data.clone())?.scan()?;
        Arc::new(scan.dtype()?.to_arrow_schema()?)
    };

    let mut times = Vec::with_capacity(iters);
    let mut rows = 0usize;
    let mut batches = 0usize;

    for it in 0..(iters + warmup) {
        let scan = SESSION.open_options().open_buffer(data.clone())?.scan()?;
        let stream = scan.into_record_batch_stream(schema.clone())?;
        pin_mut!(stream);

        // Minimal consume: iterating the stream forces full decode + (in the JVM/Python runners)
        // the Arrow C Data Interface import, without a per-row compute that would differ by
        // language and swamp the boundary cost we are isolating.
        let start = Instant::now();
        rows = 0;
        batches = 0;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            rows += batch.num_rows();
            batches += 1;
            black_box(&batch);
        }
        if it >= warmup {
            times.push(start.elapsed().as_secs_f64());
        }
    }

    report("native", &mut times, rows, batches);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Gen { rows, out } => generate(rows, out).await,
        Cmd::Bench {
            file,
            iters,
            warmup,
        } => {
            if file.extension().and_then(|e| e.to_str()) == Some("parquet") {
                bench_parquet(file, iters, warmup)
            } else {
                bench(file, iters, warmup).await
            }
        }
    }
}
