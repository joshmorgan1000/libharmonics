#pragma once

#include "harmonics/config.hpp"
#include "harmonics/core.hpp"
#include "harmonics/opencl_adapter.hpp"
#include <future>
#include <vector>

namespace harmonics {

#if HARMONICS_HAS_OPENCL
using FpgaDeviceStorage = OpenCLBuffer;
#else
using FpgaDeviceStorage = std::vector<std::byte>;
#endif

struct FpgaTensor {
    HTensor::DType dtype{HTensor::DType::Float32};
    HTensor::Shape shape{};
    FpgaDeviceStorage device_data{};
};

enum class FpgaBackend { None, OpenCL };

inline constexpr FpgaBackend select_fpga_backend() {
#if HARMONICS_HAS_OPENCL
    return FpgaBackend::OpenCL;
#else
    return FpgaBackend::None;
#endif
}

inline constexpr bool fpga_available() { return select_fpga_backend() != FpgaBackend::None; }

inline bool fpga_runtime_available() {
#if HARMONICS_HAS_OPENCL
    return opencl_runtime_available();
#else
    return false;
#endif
}

inline FpgaTensor fpga_to_device(const HTensor& t) {
    FpgaTensor dev{t.dtype(), t.shape()};
#if HARMONICS_HAS_OPENCL
    dev.device_data = opencl_malloc(t.data().size());
    opencl_memcpy_to_device(dev.device_data, t.data().data(), t.data().size());
#else
    dev.device_data = t.data();
#endif
    return dev;
}

inline HTensor fpga_to_host(const FpgaTensor& t) {
#if HARMONICS_HAS_OPENCL
    HTensor host{t.dtype, t.shape};
    auto bytes = opencl_buffer_size(t.device_data);
    host.data().resize(bytes);
    opencl_memcpy_to_host(host.data().data(), t.device_data, bytes);
    return host;
#else
    return HTensor{t.dtype, t.shape, t.device_data};
#endif
}

inline FpgaTensor fpga_copy_tensor(const FpgaTensor& t) {
    FpgaTensor out{t.dtype, t.shape};
#if HARMONICS_HAS_OPENCL
    auto bytes = opencl_buffer_size(t.device_data);
    out.device_data = opencl_malloc(bytes);
    opencl_copy_buffer(out.device_data, t.device_data, bytes);
#else
    out.device_data = t.device_data;
#endif
    return out;
}

inline std::future<FpgaTensor> fpga_to_device_async(const HTensor& t) {
#if HARMONICS_HAS_OPENCL
    FpgaTensor dev{t.dtype(), t.shape()};
    dev.device_data = opencl_malloc(t.data().size());
    auto copy_future =
        opencl_memcpy_to_device_async(dev.device_data, t.data().data(), t.data().size());
    return std::async(std::launch::async, [f = std::move(copy_future), dev]() mutable {
        f.get();
        return dev;
    });
#else
    return std::async(std::launch::async, [t]() { return fpga_to_device(t); });
#endif
}

inline std::future<HTensor> fpga_to_host_async(const FpgaTensor& t) {
#if HARMONICS_HAS_OPENCL
    HTensor host{t.dtype, t.shape};
    auto bytes = opencl_buffer_size(t.device_data);
    host.data().resize(bytes);
    auto copy_future = opencl_memcpy_to_host_async(host.data().data(), t.device_data, bytes);
    return std::async(std::launch::async, [f = std::move(copy_future), host]() mutable {
        f.get();
        return host;
    });
#else
    return std::async(std::launch::async, [t]() { return fpga_to_host(t); });
#endif
}

} // namespace harmonics
