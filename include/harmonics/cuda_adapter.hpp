#pragma once

#include "harmonics/config.hpp"
#include "harmonics/memory_profiler.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <optional>
#include <vector>

#if HARMONICS_HAS_CUDA && __has_include(<cuda_runtime.h>)
#define HARMONICS_USE_CUDA_RT 1
#include <cuda_runtime.h>
#else
#define HARMONICS_USE_CUDA_RT 0
#endif

namespace harmonics {

inline std::optional<uint32_t>& cuda_device_override() {
    static std::optional<uint32_t> index;
    return index;
}

inline void set_cuda_device_index(uint32_t index) { cuda_device_override() = index; }

#if HARMONICS_USE_CUDA_RT
struct CudaBuffer {
    void* ptr{nullptr};
    std::size_t size{0};
};

inline CudaBuffer cuda_malloc(std::size_t bytes) {
    CudaBuffer buf{};
    if (cudaMalloc(&buf.ptr, bytes) == cudaSuccess) {
        buf.size = bytes;
    }
    return buf;
}

inline void cuda_memcpy_to_device(CudaBuffer& dst, const void* src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dst.ptr, src, bytes, cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_device(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    dst.size = bytes;
}

inline void cuda_memcpy_to_host(void* dst, const CudaBuffer& src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dst, src.ptr, bytes, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_host(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline void cuda_free(CudaBuffer& buf) {
    if (buf.ptr)
        cudaFree(buf.ptr);
    buf.ptr = nullptr;
    buf.size = 0;
}

inline std::size_t cuda_buffer_size(const CudaBuffer& buf) { return buf.size; }

inline std::future<void> cuda_memcpy_to_device_async(CudaBuffer& dst, const void* src,
                                                     std::size_t bytes) {
    return std::async(std::launch::async,
                      [&dst, src, bytes]() { cuda_memcpy_to_device(dst, src, bytes); });
}

inline std::future<void> cuda_memcpy_to_host_async(void* dst, const CudaBuffer& src,
                                                   std::size_t bytes) {
    return std::async(std::launch::async,
                      [dst, &src, bytes]() { cuda_memcpy_to_host(dst, src, bytes); });
}

// Minimal compute kernel used when CUDA is available. This is a first step
// towards a full compute path and mirrors the OpenCL copy helper.
#if HARMONICS_USE_CUDA_RT && defined(__CUDACC__)
__global__ void copy_buf_kernel(const unsigned char* in, unsigned char* out) {
    std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    out[id] = in[id];
}
#endif

inline void cuda_copy_buffer(CudaBuffer& dst, const CudaBuffer& src, std::size_t bytes) {
#if HARMONICS_USE_CUDA_RT
#if defined(__CUDACC__)
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    std::size_t threads = 256;
    std::size_t blocks = (bytes + threads - 1) / threads;
    copy_buf_kernel<<<blocks, threads>>>(static_cast<const unsigned char*>(src.ptr),
                                         static_cast<unsigned char*>(dst.ptr));
    cudaDeviceSynchronize();
    dst.size = bytes;
#else
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    cudaMemcpy(dst.ptr, src.ptr, bytes, cudaMemcpyDeviceToDevice);
    dst.size = bytes;
#endif
#else
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    std::memcpy(dst.data.data(), src.data.data(), bytes);
#endif
}

#else

struct CudaBuffer {
    std::vector<std::byte> data{};
};

inline CudaBuffer cuda_malloc(std::size_t bytes) {
    return CudaBuffer{std::vector<std::byte>(bytes)};
}

inline void cuda_memcpy_to_device(CudaBuffer& dst, const void* src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    dst.data.resize(bytes);
    std::memcpy(dst.data.data(), src, bytes);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_device(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline void cuda_memcpy_to_host(void* dst, const CudaBuffer& src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst, src.data.data(), bytes);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_host(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline void cuda_free(CudaBuffer& buf) { buf.data.clear(); }

inline std::size_t cuda_buffer_size(const CudaBuffer& buf) { return buf.data.size(); }

inline std::future<void> cuda_memcpy_to_device_async(CudaBuffer& dst, const void* src,
                                                     std::size_t bytes) {
    return std::async(std::launch::async,
                      [&dst, src, bytes]() { cuda_memcpy_to_device(dst, src, bytes); });
}

inline std::future<void> cuda_memcpy_to_host_async(void* dst, const CudaBuffer& src,
                                                   std::size_t bytes) {
    return std::async(std::launch::async,
                      [dst, &src, bytes]() { cuda_memcpy_to_host(dst, src, bytes); });
}

#endif // HARMONICS_USE_CUDA_RT

inline bool cuda_runtime_available() {
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
    const char* env = std::getenv("HARMONICS_ENABLE_CUDA");
    if (!env || std::strcmp(env, "1") != 0)
        return false;
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

inline uint32_t cuda_device_index() {
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
    int index = 0;
    if (const char* env = std::getenv("HARMONICS_CUDA_DEVICE")) {
        index = std::atoi(env);
    } else if (cuda_device_override()) {
        index = static_cast<int>(*cuda_device_override());
    }
    return static_cast<uint32_t>(index);
#else
    if (cuda_device_override())
        return *cuda_device_override();
    if (const char* env = std::getenv("HARMONICS_CUDA_DEVICE"))
        return static_cast<uint32_t>(std::atoi(env));
    return 0;
#endif
}

} // namespace harmonics
