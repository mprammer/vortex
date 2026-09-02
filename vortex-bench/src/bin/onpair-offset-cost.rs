// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Print one materialized OnPair cell's stored sidecar cost, per batch granularity, host-side.
//!
//! The reported compression ratio charges the sidecar at the shipped K=6 while the plotted rate is
//! the best kernel for the column, and the kernel sweep varies K. The leg measured 32, 128 and 192
//! only, so the coarser granularities several reported kernels use were never sized. This recovers
//! them from the committed cells, without re-encoding and without a GPU.
//!
//! Always pass 192 among the granularities and diff it against the leg's own
//! `onpair_offset_cost.jsonl`: it must reproduce exactly. See `onpair_sidecar_by_granularity`.

#![expect(clippy::print_stdout)]

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use vortex_bench::onpair_bench::onpair_sidecar_by_granularity;

#[derive(Parser)]
struct Args {
    /// Cell directory holding `part_*.vortex`, e.g.
    /// `.../tpch-sf15/l_shipinstruct/bits12_chunk1000mb_thr0.20_seed20260819`.
    #[arg(long)]
    dir: PathBuf,
    #[arg(long)]
    column: String,
    #[arg(long)]
    dataset_id: String,
    #[arg(long)]
    bits: u32,
    /// Comma-separated codes per batch. Include 192 so the run validates itself against the leg.
    #[arg(long, default_value = "32,128,192,224,256,512")]
    tok_per_batch: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let a = Args::parse();
    let grans: Vec<usize> = a
        .tok_per_batch
        .split(',')
        .map(|t| t.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;
    for r in onpair_sidecar_by_granularity(&a.dir, &a.column, &grans).await? {
        let mut obj = serde_json::to_value(&r)?;
        let m = obj.as_object_mut().expect("record serializes to an object");
        m.insert("dataset".into(), a.dataset_id.clone().into());
        m.insert("column".into(), a.column.clone().into());
        m.insert("bits".into(), a.bits.into());
        println!("{}", serde_json::to_string(&obj)?);
    }
    Ok(())
}
