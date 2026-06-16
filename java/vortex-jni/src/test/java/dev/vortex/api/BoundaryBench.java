// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.jni.NativeLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * JVM path for the Vortex Arrow boundary microbenchmark (experiment E4).
 *
 * <p>Reads the file produced by {@code ffi-boundary-bench gen} via the {@code vortex-jni} bindings
 * and runs the same consume workload as the native Rust runner (sum the {@code value} column), so
 * the delta vs native is the Rust&rarr;JVM Arrow C Data Interface boundary cost.
 *
 * <p>Run (from {@code java/}):
 *
 * <pre>
 * BOUNDARY_VORTEX_FILE=/tmp/boundary.vortex BOUNDARY_OUT=/tmp/jvm.txt \
 *   ./gradlew :vortex-jni:test --tests 'dev.vortex.api.BoundaryBench' -i
 * </pre>
 */
public final class BoundaryBench {
    @BeforeAll
    public static void load() {
        NativeLoader.loadJni();
    }

    @Test
    public void benchBoundary() throws Exception {
        String file = System.getenv("BOUNDARY_VORTEX_FILE");
        Assumptions.assumeTrue(file != null, "set BOUNDARY_VORTEX_FILE to run the boundary benchmark");
        int iters = Integer.parseInt(System.getenv().getOrDefault("BOUNDARY_ITERS", "12"));
        int warmup = Integer.parseInt(System.getenv().getOrDefault("BOUNDARY_WARMUP", "4"));
        String uri = file.contains("://") ? file : Paths.get(file).toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();

        List<Double> times = new ArrayList<>();
        long rows = 0;
        long batches = 0;
        for (int it = 0; it < iters + warmup; it++) {
            DataSource ds = DataSource.open(session, uri);
            long start = System.nanoTime();
            rows = 0;
            batches = 0;
            Scan scan = ds.scan(ScanOptions.of());
            while (scan.hasNext()) {
                Partition partition = scan.next();
                try (ArrowReader reader = partition.scanArrow(allocator)) {
                    // Minimal consume: loadNextBatch forces decode + the Arrow C Data Interface
                    // import; we deliberately do not touch values per-row, so the measurement
                    // isolates the boundary rather than the JVM's per-element access pattern.
                    while (reader.loadNextBatch()) {
                        VectorSchemaRoot root = reader.getVectorSchemaRoot();
                        rows += root.getRowCount();
                        batches++;
                    }
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
                "jvm  rows=%d batches=%d%n  p50=%.3fms  p95=%.3fms%n  %.1f Mrows/s  %.0f ns/batch%n",
                rows, batches, p50 * 1e3, p95 * 1e3, rows / p50 / 1e6, p50 / Math.max(batches, 1) * 1e9);
        System.out.print(summary);

        String out = System.getenv("BOUNDARY_OUT");
        if (out != null) {
            Files.writeString(Path.of(out), summary);
        }
    }
}
