// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scaffold for the Apache Iceberg benchmark lane (vx-bench `Engine::ICEBERG`).
//!
//! The Iceberg integration is gated on `iceberg-rust` (`iceberg-datafusion`) supporting DataFusion
//! 53 — see `README.md`. Until then this binary is a placeholder: it parses the same CLI the
//! orchestrator's executor passes to the other bench binaries, then exits without running, so the
//! lane can be wired end-to-end and filled in once the dependency gate clears.

#![allow(dead_code)]

use anyhow::Result;
use clap::Parser;

/// Iceberg benchmark lane (scaffold). Mirrors the `datafusion-bench` / `duckdb-bench` CLI contract
/// so it slots into vx-bench's executor once the Iceberg table provider is wired up.
#[derive(Parser)]
#[command(about = "Apache Iceberg benchmark lane for Vortex (scaffold)")]
struct Cli {
    /// Benchmark suite (e.g. tpch, tpcds, clickbench).
    benchmark: String,
    /// Comma-separated data file formats (e.g. parquet,vortex).
    #[arg(long, value_delimiter = ',')]
    formats: Vec<String>,
    /// Specific query numbers to run.
    #[arg(long)]
    queries: Option<String>,
    /// Query numbers to skip.
    #[arg(long)]
    exclude_queries: Option<String>,
    /// Output display format.
    #[arg(long, default_value = "table")]
    display_format: String,
    /// Iterations per query.
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    #[arg(long)]
    track_memory: bool,
    #[arg(long)]
    tracing: bool,
    #[arg(long)]
    runner: Option<String>,
    #[arg(long)]
    gh_json_v3: Option<String>,
    /// Benchmark-specific options (key=value).
    #[arg(long)]
    opt: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    eprintln!(
        "iceberg-bench is a scaffold: the Apache Iceberg lane is pending iceberg-rust support for \
         DataFusion 53 (see benchmarks/iceberg-bench/README.md)."
    );
    eprintln!(
        "requested: benchmark={} formats={:?} queries={:?}",
        cli.benchmark, cli.formats, cli.queries
    );
    Ok(())
}
