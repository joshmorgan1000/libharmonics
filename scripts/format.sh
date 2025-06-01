#!/usr/bin/env bash
set -euo pipefail

dirs=(include src examples tests)
files=()
for d in "${dirs[@]}"; do
    if [ -d "$d" ]; then
        while IFS= read -r -d '' f; do
            files+=("$f")
        done < <(find "$d" \( -name '*.hpp' -o -name '*.cpp' \) -print0)
    fi
done

if [ ${#files[@]} -gt 0 ]; then
    printf '%s\0' "${files[@]}" | xargs -0 clang-format -i
fi
