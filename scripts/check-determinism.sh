#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE=${1:-Release}
COMPILERS=("gcc" "clang")

for cc in "${COMPILERS[@]}"; do
    cxx="${cc}++"
    build_dir="build-${cc}-${BUILD_TYPE}"
    cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    if command -v nproc >/dev/null 2>&1; then
        parallel=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        parallel=$(sysctl -n hw.ncpu)
    else
        parallel=${NUMBER_OF_PROCESSORS:-1}
    fi
    cmake --build "$build_dir" -j "$parallel"
    ctest --test-dir "$build_dir" -R int8_determinism_test --output-on-failure
    echo "Determinism check passed for compiler $cc"
done

