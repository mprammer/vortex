// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST-12's stored footprint by component, from a raw column dump, at several granularities.
//!
//! Reads the campaign's per-column dump format -- `[u64 n][(n+1) u32 offsets][payload]` -- which is
//! byte-for-byte the sample the leg trained on. Using it instead of a parquet path removes
//! `build_sample` from the comparison, so a mismatch against the committed cell cannot be blamed on
//! sampling. See `fsst12_stored_components_from_rows`.

#![expect(clippy::print_stdout)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use vortex_bench::onpair_bench::fsst12_stored_components_from_rows;

#[derive(Parser)]
struct Args {
    /// Column dump: [u64 n][(n+1) u32 offsets][payload].
    #[arg(long)]
    bin: PathBuf,
    #[arg(long)]
    dataset_id: String,
    #[arg(long)]
    column: String,
    /// Comma-separated codes per sidecar entry. Include 192, the shipped K=6, to self-check.
    #[arg(long, default_value = "192,256,512")]
    tok_per_batch: String,
    #[arg(long, default_value_t = 1_048_576_000)]
    chunk_bytes: u64,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let raw = std::fs::read(&a.bin).with_context(|| format!("read {}", a.bin.display()))?;
    anyhow::ensure!(raw.len() >= 8, "dump too short to hold its row count");
    let n = u64::from_le_bytes(raw[..8].try_into()?) as usize;
    let off_end = 8 + 4 * (n + 1);
    anyhow::ensure!(raw.len() >= off_end, "dump too short for {n} offsets");
    let offs: Vec<u32> = raw[8..off_end]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    let payload = &raw[off_end..];
    anyhow::ensure!(
        offs[n] as usize == payload.len(),
        "dump self-check failed: last offset {} but {} payload bytes",
        offs[n], payload.len()
    );
    let rows: Vec<&[u8]> = (0..n)
        .map(|i| &payload[offs[i] as usize..offs[i + 1] as usize])
        .collect();

    let grans: Vec<usize> = a
        .tok_per_batch
        .split(',')
        .map(|t| t.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;
    for (_g, comp) in fsst12_stored_components_from_rows(&rows, &grans, a.chunk_bytes)? {
        let mut obj = serde_json::Map::new();
        obj.insert("dataset_id".into(), a.dataset_id.clone().into());
        obj.insert("column".into(), a.column.clone().into());
        for (k, v) in comp {
            obj.insert(k, serde_json::Value::from(v));
        }
        println!("{}", serde_json::to_string(&serde_json::Value::Object(obj))?);
    }
    Ok(())
}
