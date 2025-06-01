#!/usr/bin/env bash
set -uo pipefail
# Compile GLSL compute shaders and embed them as compressed binaries.
# Requires glslangValidator and zstd.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SHADER_DIR="$ROOT_DIR/shaders"
HEADER="$ROOT_DIR/include/gpu/Shaders.hpp"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/shader_compile.log"

mkdir -p "$LOG_DIR"
# Clear previous log
: > "$LOG_FILE"

if command -v nproc >/dev/null 2>&1; then
    PARALLEL=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    PARALLEL=$(sysctl -n hw.ncpu)
else
    PARALLEL=${NUMBER_OF_PROCESSORS:-1}
fi

if ! command -v glslangValidator >/dev/null; then
    echo "glslangValidator not found" >&2
    exit 1
fi

if ! command -v zstd >/dev/null; then
    echo "zstd not found" >&2
    exit 1
fi

# Write header prologue
{
    echo "#pragma once"
    echo "#include <cstdint>"
    echo "#include <vector>"
    echo
    echo "namespace harmonics {"
    echo
} > "$HEADER"

export SHADER_DIR LOG_FILE
find "$SHADER_DIR" -maxdepth 1 -name '*.comp' -print0 | \
    xargs -0 -P "$PARALLEL" -I{} bash -c '
file="$1"
base=$(basename "$file" .comp)
spv="$SHADER_DIR/$base.spv"
zst="$spv.zst"
if ! glslangValidator -V "$file" -o "$spv" >> "$LOG_FILE" 2>&1; then
    echo "glslangValidator failed for $file" >> "$LOG_FILE"
    exit 1
fi
if ! zstd -q -f "$spv" -o "$zst" >> "$LOG_FILE" 2>&1; then
    echo "zstd failed for $spv" >> "$LOG_FILE"
    exit 1
fi
' _ {}

for zst in "$SHADER_DIR"/*.spv.zst; do
    [ -e "$zst" ] || continue
    base=$(basename "$zst" .spv.zst)
    arr_name="$(echo "$base" | tr '[:lower:]' '[:upper:]')_COMP_ZST"
    {
        echo "static const std::vector<uint8_t> $arr_name = {"
        od -An -vtx1 "$zst" | tr -s ' ' '\n' | sed '/^$/d' |
            awk '{printf "0x%s, ", $1; if (NR%12==0) printf "\n"} END{if (NR%12!=0) printf "\n"}' |
            sed 's/^/    /'
        echo "};"
        echo
    } >> "$HEADER"
done

echo "} // namespace harmonics" >> "$HEADER"
