#pragma once

#include "harmonics/config.hpp"
#include "harmonics/core.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>

namespace harmonics {

#if defined(__EMSCRIPTEN__)
#define HARMONICS_USE_WASM_RT 1
#else
#define HARMONICS_USE_WASM_RT 0
#endif

struct WasmTensor {
    HTensor::DType dtype{HTensor::DType::Float32};
    HTensor::Shape shape{};
    std::vector<std::byte> data{};
};

enum class WasmBackend { None, Emscripten };

inline constexpr WasmBackend select_wasm_backend() {
#if HARMONICS_USE_WASM_RT
    return WasmBackend::Emscripten;
#else
    return WasmBackend::None;
#endif
}

inline constexpr bool wasm_available() { return select_wasm_backend() != WasmBackend::None; }

inline constexpr bool wasm_simd_available() {
#if HARMONICS_HAS_WASM_SIMD
    return true;
#else
    return false;
#endif
}

inline WasmTensor wasm_to_device(const HTensor& t) {
    return WasmTensor{t.dtype(), t.shape(), t.data()};
}

inline HTensor wasm_to_host(const WasmTensor& t) { return HTensor{t.dtype, t.shape, t.data}; }

inline bool wasm_runtime_available() {
#if HARMONICS_USE_WASM_RT
    const char* env = std::getenv("HARMONICS_ENABLE_WASM");
    if (!env || std::strcmp(env, "1") != 0)
        return false;
    return true;
#else
    return false;
#endif
}

} // namespace harmonics
