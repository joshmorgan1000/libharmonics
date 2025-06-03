#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPES=("Release" "Debug")

for bt in "${BUILD_TYPES[@]}"; do
    echo "=== Running tests for $bt ==="
    "$(dirname "$0")/run-tests.sh" "$bt"
    echo
done
