#pragma once

#include "harmonics/config.hpp"
#include "harmonics/memory_profiler.hpp"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <optional>
#include <random>
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

inline cudaStream_t& cuda_default_stream() {
    static cudaStream_t stream = [] {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        return s;
    }();
    return stream;
}

inline std::future<void> cuda_memcpy_to_device_async(CudaBuffer& dst, const void* src,
                                                     std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(dst.ptr, src, bytes, cudaMemcpyHostToDevice, cuda_default_stream());
    return std::async(std::launch::async, [=, &dst]() {
        cudaStreamSynchronize(cuda_default_stream());
        auto end = std::chrono::high_resolution_clock::now();
        record_memcpy_to_device(
            bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        dst.size = bytes;
    });
}

inline std::future<void> cuda_memcpy_to_host_async(void* dst, const CudaBuffer& src,
                                                   std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(dst, src.ptr, bytes, cudaMemcpyDeviceToHost, cuda_default_stream());
    return std::async(std::launch::async, [=]() {
        cudaStreamSynchronize(cuda_default_stream());
        auto end = std::chrono::high_resolution_clock::now();
        record_memcpy_to_host(
            bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    });
}

// Minimal compute kernels used when CUDA is available. These are a first step
// towards a full compute path and mirror the OpenCL helpers.
#if HARMONICS_USE_CUDA_RT && defined(__CUDACC__)
__global__ void copy_buf_kernel(const unsigned char* in, unsigned char* out) {
    std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    out[id] = in[id];
}

__global__ void relu_f32_kernel(const float* in, float* out, std::size_t n) {
    std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float v = in[id];
        out[id] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void sigmoid_f32_kernel(const float* in, float* out, std::size_t n) {
    std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float v = in[id];
        out[id] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void softmax_f32_kernel(const float* in, float* out, std::size_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float maxv = in[0];
        for (std::size_t i = 1; i < n; ++i) {
            float v = in[i];
            if (v > maxv)
                maxv = v;
        }
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float e = expf(in[i] - maxv);
            out[i] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (std::size_t i = 0; i < n; ++i)
            out[i] *= inv;
    }
}

__global__ void gelu_f32_kernel(const float* in, float* out, std::size_t n) {
    std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float v = in[id];
        out[id] = 0.5f * v * (1.0f + erff(v / sqrtf(2.0f)));
    }
}

__global__ void dropout_f32_kernel(const float* in, float* out, std::size_t n, float rate,
                                   uint32_t seed) {
    std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        uint32_t state = seed ^ static_cast<uint32_t>(id);
        state ^= 61u ^ (state >> 16);
        state *= 9u;
        state ^= state >> 4;
        state *= 0x27d4eb2d;
        state ^= state >> 15;
        float rnd = static_cast<float>(state & 0xffffff) / 16777215.0f;
        out[id] = rnd > rate ? in[id] : 0.0f;
    }
}

__global__ void attention_f32_kernel(const float* in, float* out, std::size_t n, float temp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float maxv = in[0];
        for (std::size_t i = 1; i < n; ++i) {
            float v = in[i];
            if (v > maxv)
                maxv = v;
        }
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
            sum += expf((in[i] - maxv) / temp);
        float inv = 1.0f / sum;
        float attn = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
            attn += expf((in[i] - maxv) / temp) * inv * in[i];
        for (std::size_t i = 0; i < n; ++i)
            out[i] = attn;
    }
}

__global__ void cross_attention_f32_kernel(const float* in, float* out, std::size_t rows,
                                           std::size_t cols, float temp) {
    int r = blockIdx.x;
    if (r >= static_cast<int>(rows))
        return;
    const float* row = in + r * cols;
    float* row_out = out + r * cols;
    float maxv = row[0];
    for (std::size_t i = 1; i < cols; ++i)
        if (row[i] > maxv)
            maxv = row[i];
    float sum = 0.0f;
    for (std::size_t i = 0; i < cols; ++i)
        sum += expf((row[i] - maxv) / temp);
    float inv = 1.0f / sum;
    float attn = 0.0f;
    for (std::size_t i = 0; i < cols; ++i)
        attn += expf((row[i] - maxv) / temp) * inv * row[i];
    for (std::size_t i = 0; i < cols; ++i)
        row_out[i] = attn;
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

inline void cuda_relu_buffer(CudaBuffer& dst, const CudaBuffer& src, std::size_t elems) {
#if HARMONICS_USE_CUDA_RT
#if defined(__CUDACC__)
    std::size_t bytes = elems * sizeof(float);
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    std::size_t threads = 256;
    std::size_t blocks = (elems + threads - 1) / threads;
    relu_f32_kernel<<<blocks, threads>>>(static_cast<const float*>(src.ptr),
                                         static_cast<float*>(dst.ptr), elems);
    cudaDeviceSynchronize();
    dst.size = bytes;
#else
    cuda_copy_buffer(dst, src, elems * sizeof(float));
#endif
#else
    std::size_t bytes = elems * sizeof(float);
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    const float* in = reinterpret_cast<const float*>(src.data.data());
    float* out = reinterpret_cast<float*>(dst.data.data());
    for (std::size_t i = 0; i < elems; ++i)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
#endif
}

inline void cuda_sigmoid_buffer(CudaBuffer& dst, const CudaBuffer& src, std::size_t elems) {
#if HARMONICS_USE_CUDA_RT
#if defined(__CUDACC__)
    std::size_t bytes = elems * sizeof(float);
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    std::size_t threads = 256;
    std::size_t blocks = (elems + threads - 1) / threads;
    sigmoid_f32_kernel<<<blocks, threads>>>(static_cast<const float*>(src.ptr),
                                            static_cast<float*>(dst.ptr), elems);
    cudaDeviceSynchronize();
    dst.size = bytes;
#else
    cuda_copy_buffer(dst, src, elems * sizeof(float));
#endif
#else
    std::size_t bytes = elems * sizeof(float);
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    const float* in = reinterpret_cast<const float*>(src.data.data());
    float* out = reinterpret_cast<float*>(dst.data.data());
    for (std::size_t i = 0; i < elems; ++i)
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
#endif
}

inline void cuda_softmax_buffer(CudaBuffer& dst, const CudaBuffer& src, std::size_t elems) {
#if HARMONICS_USE_CUDA_RT
#if defined(__CUDACC__)
    std::size_t bytes = elems * sizeof(float);
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    softmax_f32_kernel<<<1, 1>>>(static_cast<const float*>(src.ptr), static_cast<float*>(dst.ptr),
                                 elems);
    cudaDeviceSynchronize();
    dst.size = bytes;
#else
    cuda_copy_buffer(dst, src, elems * sizeof(float));
#endif
#else
    std::size_t bytes = elems * sizeof(float);
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    const float* in = reinterpret_cast<const float*>(src.data.data());
    float* out = reinterpret_cast<float*>(dst.data.data());
    float maxv = in[0];
    for (std::size_t i = 1; i < elems; ++i)
        if (in[i] > maxv)
            maxv = in[i];
    float sum = 0.0f;
    for (std::size_t i = 0; i < elems; ++i) {
        float e = std::exp(in[i] - maxv);
        out[i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (std::size_t i = 0; i < elems; ++i)
        out[i] *= inv;
#endif
}

inline void cuda_gelu_buffer(CudaBuffer& dst, const CudaBuffer& src, std::size_t elems) {
#if HARMONICS_USE_CUDA_RT
#if defined(__CUDACC__)
    std::size_t bytes = elems * sizeof(float);
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    std::size_t threads = 256;
    std::size_t blocks = (elems + threads - 1) / threads;
    gelu_f32_kernel<<<blocks, threads>>>(static_cast<const float*>(src.ptr),
                                         static_cast<float*>(dst.ptr), elems);
    cudaDeviceSynchronize();
    dst.size = bytes;
#else
    cuda_copy_buffer(dst, src, elems * sizeof(float));
#endif
#else
    std::size_t bytes = elems * sizeof(float);
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    const float* in = reinterpret_cast<const float*>(src.data.data());
    float* out = reinterpret_cast<float*>(dst.data.data());
    for (std::size_t i = 0; i < elems; ++i) {
        float v = in[i];
        out[i] = 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
    }
#endif
}

inline void cuda_dropout_buffer(CudaBuffer& dst, const CudaBuffer& src, std::size_t elems,
                                float rate) {
#if HARMONICS_USE_CUDA_RT
#if defined(__CUDACC__)
    std::size_t bytes = elems * sizeof(float);
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    std::size_t threads = 256;
    std::size_t blocks = (elems + threads - 1) / threads;
    dropout_f32_kernel<<<blocks, threads>>>(static_cast<const float*>(src.ptr),
                                            static_cast<float*>(dst.ptr), elems, rate, 123456789u);
    cudaDeviceSynchronize();
    dst.size = bytes;
#else
    cuda_copy_buffer(dst, src, elems * sizeof(float));
#endif
#else
    std::size_t bytes = elems * sizeof(float);
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    const float* in = reinterpret_cast<const float*>(src.data.data());
    float* out = reinterpret_cast<float*>(dst.data.data());
    std::mt19937 rng{123456789u};
    std::bernoulli_distribution keep(1.0f - rate);
    for (std::size_t i = 0; i < elems; ++i)
        out[i] = keep(rng) ? in[i] : 0.0f;
#endif
}

inline void cuda_attention_buffer(CudaBuffer& dst, const CudaBuffer& src, std::size_t elems,
                                  float temp) {
#if HARMONICS_USE_CUDA_RT
#if defined(__CUDACC__)
    std::size_t bytes = elems * sizeof(float);
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    attention_f32_kernel<<<1, 1>>>(static_cast<const float*>(src.ptr), static_cast<float*>(dst.ptr),
                                   elems, temp);
    cudaDeviceSynchronize();
    dst.size = bytes;
#else
    cuda_copy_buffer(dst, src, elems * sizeof(float));
#endif
#else
    std::size_t bytes = elems * sizeof(float);
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    const float* in = reinterpret_cast<const float*>(src.data.data());
    float* out = reinterpret_cast<float*>(dst.data.data());
    float maxv = in[0];
    for (std::size_t i = 1; i < elems; ++i)
        if (in[i] > maxv)
            maxv = in[i];
    float sum = 0.0f;
    for (std::size_t i = 0; i < elems; ++i)
        sum += std::exp((in[i] - maxv) / temp);
    float inv = 1.0f / sum;
    float attn = 0.0f;
    for (std::size_t i = 0; i < elems; ++i)
        attn += std::exp((in[i] - maxv) / temp) * inv * in[i];
    for (std::size_t i = 0; i < elems; ++i)
        out[i] = attn;
#endif
}

inline void cuda_cross_attention_buffer(CudaBuffer& dst, const CudaBuffer& src, std::size_t rows,
                                        std::size_t cols, float temp) {
#if HARMONICS_USE_CUDA_RT
#if defined(__CUDACC__)
    std::size_t bytes = rows * cols * sizeof(float);
    if (dst.ptr == nullptr || dst.size < bytes) {
        if (dst.ptr)
            cuda_free(dst);
        dst = cuda_malloc(bytes);
    }
    dim3 grid(static_cast<unsigned int>(rows), 1, 1);
    cross_attention_f32_kernel<<<grid, 1>>>(static_cast<const float*>(src.ptr),
                                            static_cast<float*>(dst.ptr), rows, cols, temp);
    cudaDeviceSynchronize();
    dst.size = bytes;
#else
    cuda_copy_buffer(dst, src, rows * cols * sizeof(float));
#endif
#else
    std::size_t bytes = rows * cols * sizeof(float);
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    const float* in = reinterpret_cast<const float*>(src.data.data());
    float* out = reinterpret_cast<float*>(dst.data.data());
    for (std::size_t r = 0; r < rows; ++r) {
        const float* row = in + r * cols;
        float* row_out = out + r * cols;
        float maxv = row[0];
        for (std::size_t i = 1; i < cols; ++i)
            if (row[i] > maxv)
                maxv = row[i];
        float sum = 0.0f;
        for (std::size_t i = 0; i < cols; ++i)
            sum += std::exp((row[i] - maxv) / temp);
        float inv = 1.0f / sum;
        float attn = 0.0f;
        for (std::size_t i = 0; i < cols; ++i)
            attn += std::exp((row[i] - maxv) / temp) * inv * row[i];
        for (std::size_t i = 0; i < cols; ++i)
            row_out[i] = attn;
    }
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
