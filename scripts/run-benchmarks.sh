#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build-benchmarks"
LOG_DIR="$ROOT_DIR/logs/benchmarks"
mkdir -p "$LOG_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" --target benchmark_suite > /dev/null

out_file="$LOG_DIR/performance.txt"
"$BUILD_DIR/benchmark_suite" > "$out_file"

echo "Benchmark results are under $LOG_DIR"
