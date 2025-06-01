#pragma once

#include "gpu/Wrapper.h"
#include "harmonics/config.hpp"
#include "harmonics/core.hpp"
#include "harmonics/cuda_adapter.hpp"
#include "harmonics/vulkan_adapter.hpp"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <future>

namespace harmonics {

#if HARMONICS_HAS_VULKAN
using DeviceStorage = VulkanBuffer;
#elif HARMONICS_HAS_CUDA
using DeviceStorage = CudaBuffer;
#else
using DeviceStorage = std::vector<std::byte>;
#endif

/**
 * Simplified GPU tensor used by the reference implementation.
 *
 * The library does not depend on a real GPU runtime. Instead it
 * maintains a separate device buffer that mimics GPU memory.  The
 * conversion helpers copy data between the host tensor and this
 * buffer.  While not executing on an actual accelerator, this keeps
 * the interface compatible with a future GPU back end.
 */
struct GpuTensor {
    HTensor::DType dtype{HTensor::DType::Float32};
    HTensor::Shape shape{};
    DeviceStorage device_data{}; ///< Simulated device memory
};

/**
 * Available GPU back ends compiled into the library. The Vulkan path is the only
 * one currently implemented.
 */
enum class GpuBackend { None, Vulkan, Cuda };

/** Select the first available GPU back end at build time. */
inline constexpr GpuBackend select_gpu_backend() {
#if HARMONICS_HAS_CUDA
    return GpuBackend::Cuda;
#elif HARMONICS_HAS_VULKAN
    return GpuBackend::Vulkan;
#else
    return GpuBackend::None;
#endif
}

inline constexpr bool vulkan_available() {
#if HARMONICS_HAS_VULKAN
    return true;
#else
    return false;
#endif
}

inline constexpr bool cuda_available() {
#if HARMONICS_HAS_CUDA
    return true;
#else
    return false;
#endif
}

inline constexpr bool gpu_available() { return select_gpu_backend() != GpuBackend::None; }

inline std::size_t device_ring_size() {
    if (const char* env = std::getenv("HARMONICS_DEVICE_RING_SIZE")) {
        std::size_t val = std::strtoul(env, nullptr, 10);
        return val > 0 ? val : 3;
    }
    return 3;
}

inline Wrapper<DeviceStorage>& device_buffer_ring() {
    static Wrapper<DeviceStorage> ring{device_ring_size()};
    return ring;
}

inline std::vector<DeviceStorage>& device_buffer_pool() {
    static std::vector<DeviceStorage> pool;
    return pool;
}

inline std::size_t device_pool_limit() {
    if (const char* env = std::getenv("HARMONICS_DEVICE_POOL_LIMIT")) {
        std::size_t val = std::strtoul(env, nullptr, 10);
        return val > 0 ? val : device_ring_size() * 2;
    }
    return device_ring_size() * 2;
}

inline std::size_t device_buffer_capacity(const DeviceStorage& buf) {
#if HARMONICS_HAS_VULKAN
#if HARMONICS_USE_VULKAN_RT
    return buf.size;
#else
    return buf.data.size();
#endif
#elif HARMONICS_HAS_CUDA
    return cuda_buffer_size(buf);
#else
    return buf.size();
#endif
}

inline DeviceStorage pool_alloc(std::size_t bytes) {
    auto& pool = device_buffer_pool();
    for (auto it = pool.begin(); it != pool.end(); ++it) {
        if (device_buffer_capacity(*it) >= bytes) {
            DeviceStorage out = std::move(*it);
            pool.erase(it);
            return out;
        }
    }
#if HARMONICS_HAS_VULKAN
    return vulkan_malloc(bytes);
#elif HARMONICS_HAS_CUDA
    return cuda_malloc(bytes);
#else
    return DeviceStorage(bytes);
#endif
}

inline void pool_release(DeviceStorage buf) {
    auto& pool = device_buffer_pool();
    pool.push_back(std::move(buf));
    while (pool.size() > device_pool_limit()) {
        DeviceStorage extra = std::move(pool.front());
        pool.erase(pool.begin());
#if HARMONICS_HAS_VULKAN
#if HARMONICS_USE_VULKAN_RT
        vulkan_free(extra);
#endif
#elif HARMONICS_HAS_CUDA
#if HARMONICS_USE_CUDA_RT
        cuda_free(extra);
#endif
#endif
    }
}

inline void clear_device_buffer_pool() {
    auto& pool = device_buffer_pool();
    for (auto& buf : pool) {
#if HARMONICS_HAS_VULKAN
#if HARMONICS_USE_VULKAN_RT
        vulkan_free(buf);
#endif
#elif HARMONICS_HAS_CUDA
#if HARMONICS_USE_CUDA_RT
        cuda_free(buf);
#endif
#endif
    }
    pool.clear();
}

inline std::size_t device_pool_size() { return device_buffer_pool().size(); }

// Only the Vulkan path currently supports actual execution. The other
// runtime checks are placeholders for future back ends.
inline bool vulkan_runtime_available() {
#if HARMONICS_HAS_VULKAN
    const char* env = std::getenv("HARMONICS_ENABLE_VULKAN");
    if (!env || std::strcmp(env, "1") != 0)
        return false;
    void* handle = dlopen("libvulkan.so.1", RTLD_LAZY);
    if (handle) {
        dlclose(handle);
        return true;
    }
    return false;
#else
    return false;
#endif
}

inline bool gpu_runtime_available() {
    switch (select_gpu_backend()) {
    case GpuBackend::Vulkan:
        return vulkan_runtime_available();
    case GpuBackend::Cuda:
        return cuda_runtime_available();
    default:
        return false;
    }
}

inline GpuTensor to_device(const HTensor& t) {
    GpuTensor dev{t.dtype(), t.shape()};
#if HARMONICS_HAS_VULKAN
    auto& buf = device_buffer_ring().acquire();
#if HARMONICS_USE_VULKAN_RT
    if (buf.buffer == VK_NULL_HANDLE || buf.size < t.data().size()) {
        if (buf.buffer != VK_NULL_HANDLE)
            pool_release(std::move(buf));
        buf = pool_alloc(t.data().size());
    }
#else
    if (buf.data.size() < t.data().size()) {
        if (!buf.data.empty())
            pool_release(std::move(buf));
        buf = pool_alloc(t.data().size());
    }
#endif
    vulkan_memcpy_to_device(buf, t.data().data(), t.data().size());
    dev.device_data = buf;
#elif HARMONICS_HAS_CUDA
    auto& buf = device_buffer_ring().acquire();
#if HARMONICS_USE_CUDA_RT
    if (buf.ptr == nullptr || buf.size < t.data().size()) {
        if (buf.ptr)
            pool_release(std::move(buf));
        buf = pool_alloc(t.data().size());
    }
#else
    if (buf.data.size() < t.data().size()) {
        if (!buf.data.empty())
            pool_release(std::move(buf));
        buf = pool_alloc(t.data().size());
    }
#endif
    cuda_memcpy_to_device(buf, t.data().data(), t.data().size());
    dev.device_data = buf;
#else
    dev.device_data = t.data();
#endif
    return dev;
}

inline HTensor to_host(const GpuTensor& t) {
#if HARMONICS_HAS_VULKAN
    HTensor host{t.dtype, t.shape};
#if HARMONICS_USE_VULKAN_RT
    host.data().resize(t.device_data.size);
    vulkan_memcpy_to_host(host.data().data(), t.device_data, t.device_data.size);
#else
    host.data().resize(t.device_data.data.size());
    vulkan_memcpy_to_host(host.data().data(), t.device_data, t.device_data.data.size());
#endif
    return host;
#elif HARMONICS_HAS_CUDA
    HTensor host{t.dtype, t.shape};
    host.data().resize(cuda_buffer_size(t.device_data));
    cuda_memcpy_to_host(host.data().data(), t.device_data, host.data().size());
    return host;
#else
    return HTensor{t.dtype, t.shape, t.device_data};
#endif
}

inline std::future<GpuTensor> to_device_async(const HTensor& t) {
#if HARMONICS_HAS_VULKAN
    GpuTensor dev{t.dtype(), t.shape()};
    auto& buf = device_buffer_ring().acquire();
#if HARMONICS_USE_VULKAN_RT
    if (buf.buffer == VK_NULL_HANDLE || buf.size < t.data().size()) {
        if (buf.buffer != VK_NULL_HANDLE)
            pool_release(std::move(buf));
        buf = pool_alloc(t.data().size());
    }
#else
    if (buf.data.size() < t.data().size()) {
        if (!buf.data.empty())
            pool_release(std::move(buf));
        buf = pool_alloc(t.data().size());
    }
#endif
    auto copy_future = vulkan_memcpy_to_device_async(buf, t.data().data(), t.data().size());
    dev.device_data = buf;
    return std::async(std::launch::async, [f = std::move(copy_future), dev]() mutable {
        f.get();
        return dev;
    });
#elif HARMONICS_HAS_CUDA
    GpuTensor dev{t.dtype(), t.shape()};
    auto& buf = device_buffer_ring().acquire();
#if HARMONICS_USE_CUDA_RT
    if (buf.ptr == nullptr || buf.size < t.data().size()) {
        if (buf.ptr)
            pool_release(std::move(buf));
        buf = pool_alloc(t.data().size());
    }
#else
    if (buf.data.size() < t.data().size()) {
        if (!buf.data.empty())
            pool_release(std::move(buf));
        buf = pool_alloc(t.data().size());
    }
#endif
    auto copy_future = cuda_memcpy_to_device_async(buf, t.data().data(), t.data().size());
    dev.device_data = buf;
    return std::async(std::launch::async, [f = std::move(copy_future), dev]() mutable {
        f.get();
        return dev;
    });
#else
    return std::async(std::launch::async, [t]() { return to_device(t); });
#endif
}

inline std::future<HTensor> to_host_async(const GpuTensor& t) {
#if HARMONICS_HAS_VULKAN
    HTensor host{t.dtype, t.shape};
#if HARMONICS_USE_VULKAN_RT
    host.data().resize(t.device_data.size);
    auto copy_future =
        vulkan_memcpy_to_host_async(host.data().data(), t.device_data, t.device_data.size);
#else
    host.data().resize(t.device_data.data.size());
    auto copy_future =
        vulkan_memcpy_to_host_async(host.data().data(), t.device_data, t.device_data.data.size());
#endif
    return std::async(std::launch::async, [f = std::move(copy_future), host]() mutable {
        f.get();
        return host;
    });
#elif HARMONICS_HAS_CUDA
    HTensor host{t.dtype, t.shape};
    host.data().resize(cuda_buffer_size(t.device_data));
    auto copy_future =
        cuda_memcpy_to_host_async(host.data().data(), t.device_data, host.data().size());
    return std::async(std::launch::async, [f = std::move(copy_future), host]() mutable {
        f.get();
        return host;
    });
#else
    return std::async(std::launch::async, [t]() { return to_host(t); });
#endif
}

} // namespace harmonics
