// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.arrow.ArrowAllocation;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.dataset.file.FileFormat;
import org.apache.arrow.dataset.file.FileSystemDatasetFactory;
import org.apache.arrow.dataset.jni.NativeMemoryPool;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.dataset.source.Dataset;
import org.apache.arrow.dataset.source.DatasetFactory;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * JVM-Parquet baseline for the boundary microbenchmark (experiment E4).
 *
 * <p>Reads the Parquet sibling produced by {@code ffi-boundary-bench gen} via Arrow-Dataset
 * (Arrow C++ decode &rarr; Arrow C Data Interface &rarr; JVM). This mirrors the {@code vortex-jni}
 * path (native decode &rarr; C Data Interface &rarr; JVM), so JVM-Vortex vs JVM-Parquet both pay
 * the same Arrow-to-JVM handoff and the comparison isolates the decoder.
 *
 * <p>Run (from {@code java/}):
 *
 * <pre>
 * BOUNDARY_PARQUET_FILE=/tmp/boundary.parquet BOUNDARY_OUT=/tmp/jvm_parquet.txt \
 *   ./gradlew --rerun-tasks :vortex-jni:test --tests 'dev.vortex.api.ParquetBoundaryBench'
 * </pre>
 */
public final class ParquetBoundaryBench {
    @Test
    public void benchParquetBoundary() throws Exception {
        String file = System.getenv("BOUNDARY_PARQUET_FILE");
        Assumptions.assumeTrue(file != null, "set BOUNDARY_PARQUET_FILE to run the parquet boundary benchmark");
        int iters = Integer.parseInt(System.getenv().getOrDefault("BOUNDARY_ITERS", "12"));
        int warmup = Integer.parseInt(System.getenv().getOrDefault("BOUNDARY_WARMUP", "4"));
        long batchSize = Long.parseLong(System.getenv().getOrDefault("BOUNDARY_BATCH", "131072"));
        String uri = file.contains("://") ? file : Paths.get(file).toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();

        List<Double> times = new ArrayList<>();
        long rows = 0;
        long batches = 0;
        for (int it = 0; it < iters + warmup; it++) {
            ScanOptions options = new ScanOptions(batchSize);
            long start = System.nanoTime();
            rows = 0;
            batches = 0;
            try (DatasetFactory factory = new FileSystemDatasetFactory(
                            allocator, NativeMemoryPool.getDefault(), FileFormat.PARQUET, uri);
                    Dataset dataset = factory.finish();
                    Scanner scanner = dataset.newScan(options);
                    ArrowReader reader = scanner.scanBatches()) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot root = reader.getVectorSchemaRoot();
                    rows += root.getRowCount();
                    batches++;
                }
            }
            double dt = (System.nanoTime() - start) / 1e9;
            if (it >= warmup) {
                times.add(dt);
            }
        }

        times.sort(Double::compareTo);
        double p50 = times.get(times.size() / 2);
        double p95 = times.get(Math.min((int) (times.size() * 0.95), times.size() - 1));
        String summary = String.format(
                "jvm-parquet  rows=%d batches=%d%n  p50=%.3fms  p95=%.3fms%n  %.1f Mrows/s  %.0f ns/batch%n",
                rows, batches, p50 * 1e3, p95 * 1e3, rows / p50 / 1e6, p50 / Math.max(batches, 1) * 1e9);
        System.out.print(summary);

        String out = System.getenv("BOUNDARY_OUT");
        if (out != null) {
            Files.writeString(Path.of(out), summary);
        }
    }
}
