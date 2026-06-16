# iceberg-bench (scaffold)

The Apache Iceberg benchmark lane for vx-bench: read the same data from an **Iceberg table** backed
by Parquet vs Vortex data files, through DataFusion, and report throughput — so we can measure the
format *inside* an Iceberg table rather than as a raw file.

This is currently a **scaffold**. It compiles, parses the bench-binary CLI contract, and is wired
into the orchestrator as `Engine.ICEBERG`, but it does not run an Iceberg query yet.

## Why it's gated

`iceberg-rust`'s `iceberg-datafusion` (latest 0.9.1) requires **DataFusion 52**, while this
workspace is on **DataFusion 53** (`datafusion = "53"`, `arrow = "58"`). DataFusion can't mix minor
versions in one binary, and the arrow versions differ, so adding `iceberg-datafusion` now would
either fail to build or create a two-arrow-versions split where iceberg-rust's `RecordBatch` and
Vortex's `RecordBatch` don't interoperate. This gap is transient — `iceberg-rust` is one DataFusion
minor behind and will catch up.

A working **Python** path already exists in the meantime: see `pyiceberg/` in the
`vortex-iceberg-demo` repo (PyIceberg + `vortex-data`), which does the full Iceberg+Vortex
round-trip today.

## Plan (when iceberg-rust reaches DataFusion 53)

1. Add deps: `iceberg = "0.x"`, `iceberg-datafusion = "0.x"` (DataFusion-53-compatible).
2. In `main()`:
   - Build/load an Iceberg table over a `MemoryCatalog` (or REST/SQL catalog), pointed at a
     warehouse of **Parquet** data files for the baseline.
   - Register it with DataFusion via `iceberg-datafusion`'s `IcebergTableProvider`.
   - Run the benchmark's query suite (reuse `vortex-bench`'s TPC-H/TPC-DS/ClickBench query sets and
     data generation), report p50/p95 + the JSONL records the orchestrator expects.
3. Add the **Vortex** format once Vortex is a registered Iceberg `FileFormat` (the File Format API
   `FormatModel` this proposal is about) — then the same lane reads Iceberg+Vortex.

## Bench-binary contract

The orchestrator's executor invokes each lane binary as:

```
iceberg-bench <benchmark> --display-format <fmt> --formats parquet,vortex \
  [--queries 1,3,5] [--exclude-queries ...] [--iterations N] \
  [--track-memory] [--tracing] [--runner <id>] [--gh-json-v3 <path>] [--opt k=v]...
```

`main.rs` already parses this; the execution body is the TODO above.

## Dataset

Default demo/bench dataset: **TPC-H** (start at SF1). Reuse `vortex-bench`'s TPC-H generation so the
Iceberg lane writes the same data as the other lanes for an apples-to-apples comparison.
