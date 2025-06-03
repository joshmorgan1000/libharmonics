#pragma once

#include "harmonics/fpga_backend.hpp"
#include "harmonics/gpu_backend.hpp"
#include <blake3.h>
#include <optional>
#include <string>
#include <vector>

#include "harmonics/fpga_backend.hpp"
#include "harmonics/gpu_backend.hpp"

namespace harmonics {

/** Available execution back ends. */
enum class Backend { CPU, GPU, FPGA, WASM, Auto };

/** Options for a single graph partition. */
struct PartitionOptions {
    Backend backend{Backend::CPU};               ///< execution back end
    std::string target{};                        ///< optional remote target string
    std::optional<uint32_t> device_index{};      ///< optional GPU device index
    std::optional<uint32_t> fpga_device_index{}; ///< optional OpenCL device index
    double weight{1.0};                          ///< relative load balancing weight
};

/** Descriptor for deploying a HarmonicGraph or its partitions. */
struct DeploymentDescriptor {
    Backend backend{Backend::CPU};               ///< default backend for the graph
    bool secure{false};                          ///< enable chain-of-custody proofs
    std::optional<uint32_t> gpu_device_index{};  ///< optional GPU device selection
    std::optional<uint32_t> fpga_device_index{}; ///< optional OpenCL device selection
    std::vector<PartitionOptions> partitions{};  ///< per-partition overrides
};

inline std::string to_hex(const unsigned char* data, std::size_t len) {
    static const char* hex = "0123456789abcdef";
    std::string out(2 * len, '0');
    for (std::size_t i = 0; i < len; ++i) {
        out[2 * i] = hex[data[i] >> 4];
        out[2 * i + 1] = hex[data[i] & 0xf];
    }
    return out;
}

/** Compute a BLAKE3 digest for arbitrary memory. */
inline std::string blake3(const void* data, std::size_t size) {
    uint8_t out[BLAKE3_OUT_LEN];
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, data, size);
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
    return to_hex(out, BLAKE3_OUT_LEN);
}

/** Select the best available accelerator backend at runtime. */
inline Backend select_accelerator_backend() {
#if HARMONICS_HAS_WASM
    return Backend::WASM;
#else
#if HARMONICS_HAS_VULKAN
    if (gpu_runtime_available())
        return Backend::GPU;
#endif
#if HARMONICS_HAS_OPENCL
    if (fpga_runtime_available())
        return Backend::FPGA;
#endif
    return Backend::CPU;
#endif
}

} // namespace harmonics
