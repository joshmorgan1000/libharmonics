#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build-gpu-tests"
REPORT_DIR="$ROOT_DIR/logs/shader_reports"
LOG_DIR="$ROOT_DIR/logs"
COMPILE_LOG="$LOG_DIR/gpu_test_compile.log"
CLI_LOG="$LOG_DIR/gpu_test_run.log"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# Build shader_wrapper_cli if not already built
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" --target shader_wrapper_cli > /dev/null

# Compile shaders if tools are available
if command -v glslangValidator >/dev/null && command -v zstd >/dev/null; then
    if ! "$SCRIPT_DIR/compile_shaders.sh" > "$COMPILE_LOG" 2>&1; then
        echo "Shader compilation failed. See $COMPILE_LOG" >&2
        exit 1
    fi
fi

# Prepare input data
TMPDIR=$(mktemp -d)
cat <<EOT > "$TMPDIR/a.txt"
1 2 3
EOT
cat <<EOT > "$TMPDIR/b.txt"
3 2 1
EOT
cat <<EOT > "$TMPDIR/w.txt"
1 2 3 4
EOT
cat <<EOT > "$TMPDIR/x.txt"
1 1 1 1
EOT
cat <<EOT > "$TMPDIR/p.txt"
1 2 3 4
EOT
cat <<EOT > "$TMPDIR/g.txt"
0.1 0.1 0.1 0.1
EOT
cat <<EOT > "$TMPDIR/m.txt"
0 0 0 0
EOT
cat <<EOT > "$TMPDIR/v.txt"
0 0 0 0
EOT
cat <<EOT > "$TMPDIR/s.txt"
0 0 0 0
EOT

SHADERS="l2_distance relu sigmoid selu prelu barycentric_scores cross_entropy_loss mse_loss fully_connected sgd rmsprop adam"

declare -A EXPECTED
EXPECTED[l2_distance]="2.82843"
EXPECTED[relu]="1 2 3"
EXPECTED[sigmoid]="0.731059 0.880797 0.952574"
EXPECTED[selu]="1.050701 2.101402 3.152103"
EXPECTED[prelu]="1 2 3"
EXPECTED[barycentric_scores]="1 2 3"
EXPECTED[cross_entropy_loss]="3.57628e-07 2.38419e-07 1.19209e-07"
EXPECTED[mse_loss]="4 0 4"
EXPECTED[fully_connected]="10"
EXPECTED[sgd]="0.999 1.999 2.999 3.999"
EXPECTED[rmsprop]="0.968377 1.96838 2.96838 3.96838"
EXPECTED[adam]="0.99 1.99 2.99 3.99"

for shader in $SHADERS; do
    case $shader in
        l2_distance)
            args="--input $TMPDIR/a.txt --input $TMPDIR/b.txt"
            ;;
        relu|sigmoid|selu|prelu|barycentric_scores)
            args="--input $TMPDIR/a.txt"
            ;;
        cross_entropy_loss|mse_loss)
            args="--input $TMPDIR/a.txt --input $TMPDIR/b.txt"
            ;;
        fully_connected)
            args="--input $TMPDIR/w.txt --input $TMPDIR/x.txt"
            ;;
        sgd)
            args="--input $TMPDIR/p.txt --input $TMPDIR/g.txt"
            ;;
        rmsprop)
            args="--input $TMPDIR/p.txt --input $TMPDIR/g.txt --input $TMPDIR/s.txt"
            ;;
        adam)
            args="--input $TMPDIR/p.txt --input $TMPDIR/g.txt --input $TMPDIR/m.txt --input $TMPDIR/v.txt"
            ;;
    esac
    out_file="$REPORT_DIR/${shader}_report.txt"
    "$BUILD_DIR/shader_wrapper_cli" --shader "$shader" $args -o "$out_file" >> "$CLI_LOG" 2>&1
    if [ ! -f "$out_file" ]; then
        echo "Failed to generate output for $shader" >&2
        exit 1
    fi
    actual=$(grep '^Output:' "$out_file" | cut -d' ' -f2-)
    expected="${EXPECTED[$shader]}"
    if [ "$actual" != "$expected" ]; then
        echo "Unexpected result for $shader: $actual (expected $expected)" >&2
        exit 1
    fi
done

echo "Shader test reports are under $REPORT_DIR"
echo "Build and run logs are under $LOG_DIR"
