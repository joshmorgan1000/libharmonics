#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE=${1:-Release}
BUILD_DIR=build-${BUILD_TYPE}

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if command -v nproc >/dev/null 2>&1; then
    PARALLEL=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    PARALLEL=$(sysctl -n hw.ncpu)
else
    PARALLEL=${NUMBER_OF_PROCESSORS:-1}
fi
cmake --build "$BUILD_DIR" -j "$PARALLEL"
cmake --build "$BUILD_DIR" --target test
