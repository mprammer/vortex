#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Runs all three boundary-benchmark paths (native / python / jvm) over one shared Vortex file and
# prints a Mrows/s comparison. The native path is the floor; python/jvm minus native is the Arrow
# C Data Interface boundary cost. See README.md (experiment E4).
#
#   FILE=/tmp/boundary.vortex ROWS=5000000 ITERS=12 WARMUP=4 bash run_all.sh
set -uo pipefail

FILE=${FILE:-/tmp/boundary.vortex}
ROWS=${ROWS:-5000000}
ITERS=${ITERS:-12}
WARMUP=${WARMUP:-4}
JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

run() { echo "+ $*" >&2; "$@"; }

# 1. Generate the shared input (native runner owns data-gen).
[ -f "$FILE" ] || run cargo run -q -p ffi-boundary-bench --release -- gen --rows "$ROWS" --out "$FILE"

# 2. Native Rust baseline.
echo "## native"
cargo run -q -p ffi-boundary-bench --release -- bench --file "$FILE" --iters "$ITERS" --warmup "$WARMUP" \
  | tee /tmp/native.txt

# 3. Python (PyO3) path. Requires the vortex-data wheel.
echo "## python"
uv run --reinstall-package vortex-data python \
  benchmarks/ffi-boundary-bench/runners/python_runner.py --file "$FILE" --iters "$ITERS" --warmup "$WARMUP" \
  | tee /tmp/python.txt || echo "(python runner failed/skipped)"

# 4. JVM (vortex-jni) path. Downloads a JDK 17 toolchain on first run.
echo "## jvm"
( cd java && JAVA_HOME="$JAVA_HOME" \
    BOUNDARY_VORTEX_FILE="$FILE" BOUNDARY_OUT=/tmp/jvm.txt BOUNDARY_ITERS="$ITERS" BOUNDARY_WARMUP="$WARMUP" \
    ./gradlew -q :vortex-jni:test --tests 'dev.vortex.api.BoundaryBench' >/dev/null 2>&1 ) \
  && cat /tmp/jvm.txt || echo "(jvm runner failed/skipped)"

# 5. Comparison.
echo "## comparison"
for p in native python jvm; do
  f=/tmp/$p.txt
  [ -f "$f" ] && awk -v p="$p" '/Mrows\/s/{printf "  %-7s %8s Mrows/s   %8s ns/batch\n", p, $1, $3}' "$f"
done
