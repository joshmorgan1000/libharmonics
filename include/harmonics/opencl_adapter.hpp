#pragma once

#include "harmonics/config.hpp"
#include "harmonics/memory_profiler.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <future>
#include <optional>
#include <unordered_map>
#include <vector>

#if HARMONICS_HAS_OPENCL && __has_include(<CL/cl.h>)
#define HARMONICS_USE_OPENCL_RT 1
#include <CL/cl.h>
#else
#define HARMONICS_USE_OPENCL_RT 0
#endif

namespace harmonics {

inline std::optional<uint32_t>& opencl_device_override() {
    static std::optional<uint32_t> index;
    return index;
}

inline void set_opencl_device_index(uint32_t index) { opencl_device_override() = index; }

#if HARMONICS_USE_OPENCL_RT
struct OpenCLBuffer {
    cl_mem buffer{nullptr};
    cl_context context{nullptr};
    std::size_t size{0};
};

struct OpenCLContext {
    cl_context context{nullptr};
    cl_command_queue queue{nullptr};
    cl_device_id device{nullptr};
    cl_platform_id platform{nullptr};
    uint32_t device_index{0};
};

inline OpenCLContext& get_opencl_context() {
    static OpenCLContext ctx;
    static bool init = false;
    if (!init) {
        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);
        cl_uint index = 0;
        if (const char* env = std::getenv("HARMONICS_OPENCL_PLATFORM")) {
            int idx = std::atoi(env);
            if (idx >= 0 && static_cast<cl_uint>(idx) < platform_count)
                index = static_cast<cl_uint>(idx);
        }
        ctx.platform = platforms[index];
        cl_uint device_count = 0;
        clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
        std::vector<cl_device_id> devices(device_count);
        clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr);
        cl_uint dev_idx = 0;
        if (const char* env = std::getenv("HARMONICS_OPENCL_DEVICE")) {
            int idx = std::atoi(env);
            if (idx >= 0 && static_cast<cl_uint>(idx) < device_count)
                dev_idx = static_cast<cl_uint>(idx);
        } else if (opencl_device_override()) {
            uint32_t idx = *opencl_device_override();
            if (idx < device_count)
                dev_idx = static_cast<cl_uint>(idx);
        }
        ctx.device = devices[dev_idx];
        ctx.device_index = dev_idx;
        ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, nullptr);
        ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, nullptr);
        init = true;
    }
    return ctx;
}

inline uint32_t opencl_device_index() { return get_opencl_context().device_index; }

inline uint32_t opencl_device_count() {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS)
        return 0;
    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    cl_uint total = 0;
    for (cl_uint i = 0; i < platform_count; ++i) {
        cl_uint count = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &count);
        total += count;
    }
    return static_cast<uint32_t>(total);
}

inline OpenCLBuffer opencl_malloc(std::size_t bytes) {
    auto& ctx = get_opencl_context();
    OpenCLBuffer buf;
    buf.context = ctx.context;
    buf.buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, bytes, nullptr, nullptr);
    buf.size = bytes;
    return buf;
}

inline void opencl_memcpy_to_device(OpenCLBuffer& dst, const void* src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    auto& ctx = get_opencl_context();
    clEnqueueWriteBuffer(ctx.queue, dst.buffer, CL_TRUE, 0, bytes, src, 0, nullptr, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_device(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    dst.size = bytes;
}

inline void opencl_memcpy_to_host(void* dst, const OpenCLBuffer& src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    auto& ctx = get_opencl_context();
    clEnqueueReadBuffer(ctx.queue, src.buffer, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_host(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline void opencl_free(OpenCLBuffer& buf) {
    if (buf.buffer)
        clReleaseMemObject(buf.buffer);
    buf.buffer = nullptr;
    buf.context = nullptr;
    buf.size = 0;
}

inline std::size_t opencl_buffer_size(const OpenCLBuffer& buf) { return buf.size; }

inline std::future<void> opencl_memcpy_to_device_async(OpenCLBuffer& dst, const void* src,
                                                       std::size_t bytes) {
    return std::async(std::launch::async,
                      [&dst, src, bytes]() { opencl_memcpy_to_device(dst, src, bytes); });
}

inline std::future<void> opencl_memcpy_to_host_async(void* dst, const OpenCLBuffer& src,
                                                     std::size_t bytes) {
    return std::async(std::launch::async,
                      [dst, &src, bytes]() { opencl_memcpy_to_host(dst, src, bytes); });
}

inline constexpr const char* OPENCL_COPY_KERNEL_SRC =
    "__kernel void copy_buf(__global const uchar* in, __global uchar* out) "
    "{ size_t id = get_global_id(0); out[id] = in[id]; }";

inline void opencl_copy_buffer(OpenCLBuffer& dst, const OpenCLBuffer& src, std::size_t bytes) {
    static OpenCLKernel kernel = opencl_build_kernel(OPENCL_COPY_KERNEL_SRC, "copy_buf");
    if (!kernel.kernel)
        return;
    auto& ctx = get_opencl_context();
    clSetKernelArg(kernel.kernel, 0, sizeof(cl_mem), &src.buffer);
    clSetKernelArg(kernel.kernel, 1, sizeof(cl_mem), &dst.buffer);
    size_t global = bytes;
    clEnqueueNDRangeKernel(ctx.queue, kernel.kernel, 1, nullptr, &global, nullptr, 0, nullptr,
                           nullptr);
    clFinish(ctx.queue);
    dst.size = bytes;
}

struct OpenCLKernel {
    cl_kernel kernel{nullptr};
};

inline OpenCLKernel opencl_build_kernel(const char* source, const char* name) {
    auto& ctx = get_opencl_context();
    cl_int err = 0;
    cl_program prog = clCreateProgramWithSource(ctx.context, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS)
        return {};
    err = clBuildProgram(prog, 1, &ctx.device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return {};
    }
    cl_kernel k = clCreateKernel(prog, name, &err);
    clReleaseProgram(prog);
    if (err != CL_SUCCESS)
        return {};
    return OpenCLKernel{k};
}

inline void opencl_destroy_kernel(OpenCLKernel& k) {
    if (k.kernel)
        clReleaseKernel(k.kernel);
    k.kernel = nullptr;
}

inline OpenCLKernel opencl_build_activation_kernel(const std::string& name) {
    static std::unordered_map<std::string, OpenCLKernel> cache;
    auto it = cache.find(name);
    if (it != cache.end())
        return it->second;

    OpenCLKernel k{};
    if (name == "relu") {
        static const char* SRC =
            "__kernel void relu_f32(__global const float* in, __global float* out) "
            "{ size_t id = get_global_id(0); float v = in[id]; out[id] = v > 0.0f ? v : 0.0f; }";
        k = opencl_build_kernel(SRC, "relu_f32");
    } else if (name == "sigmoid") {
        static const char* SRC =
            "__kernel void sigmoid_f32(__global const float* in, __global float* out) "
            "{ size_t id = get_global_id(0); float v = in[id]; out[id] = 1.0f / (1.0f + exp(-v)); "
            "}";
        k = opencl_build_kernel(SRC, "sigmoid_f32");
    } else if (name == "identity" || name == "id") {
        k = opencl_build_kernel(OPENCL_COPY_KERNEL_SRC, "copy_buf");
    }

    cache.emplace(name, k);
    return k;
}

#else

struct OpenCLBuffer {
    std::vector<std::byte> data{};
};

struct OpenCLKernel {
    void* kernel{nullptr};
};

inline OpenCLKernel opencl_build_kernel(const char* source, const char* name) {
    (void)source;
    (void)name;
    return {};
}

inline void opencl_destroy_kernel(OpenCLKernel&) {}

inline OpenCLKernel opencl_build_activation_kernel(const std::string& name) {
    static std::unordered_map<std::string, OpenCLKernel> cache;
    auto it = cache.find(name);
    if (it != cache.end())
        return it->second;
    OpenCLKernel k{};
    cache.emplace(name, k);
    return k;
}

inline OpenCLBuffer opencl_malloc(std::size_t bytes) {
    return OpenCLBuffer{std::vector<std::byte>(bytes)};
}

inline void opencl_copy_buffer(OpenCLBuffer& dst, const OpenCLBuffer& src, std::size_t bytes) {
    if (dst.data.size() < bytes)
        dst.data.resize(bytes);
    std::memcpy(dst.data.data(), src.data.data(), bytes);
}

inline void opencl_memcpy_to_device(OpenCLBuffer& dst, const void* src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    dst.data.resize(bytes);
    std::memcpy(dst.data.data(), src, bytes);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_device(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline void opencl_memcpy_to_host(void* dst, const OpenCLBuffer& src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst, src.data.data(), bytes);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_host(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline void opencl_free(OpenCLBuffer& buf) { buf.data.clear(); }

inline std::size_t opencl_buffer_size(const OpenCLBuffer& buf) { return buf.data.size(); }

inline std::future<void> opencl_memcpy_to_device_async(OpenCLBuffer& dst, const void* src,
                                                       std::size_t bytes) {
    return std::async(std::launch::async,
                      [&dst, src, bytes]() { opencl_memcpy_to_device(dst, src, bytes); });
}

inline std::future<void> opencl_memcpy_to_host_async(void* dst, const OpenCLBuffer& src,
                                                     std::size_t bytes) {
    return std::async(std::launch::async,
                      [dst, &src, bytes]() { opencl_memcpy_to_host(dst, src, bytes); });
}

inline uint32_t opencl_device_index() {
    if (opencl_device_override())
        return *opencl_device_override();
    if (const char* env = std::getenv("HARMONICS_OPENCL_DEVICE"))
        return static_cast<uint32_t>(std::atoi(env));
    return 0;
}

inline uint32_t opencl_device_count() { return 0; }

#endif // HARMONICS_USE_OPENCL_RT

inline bool opencl_runtime_available() {
#if HARMONICS_USE_OPENCL_RT
    const char* env = std::getenv("HARMONICS_ENABLE_OPENCL");
    if (!env || std::strcmp(env, "1") != 0)
        return false;
    void* handle = dlopen("libOpenCL.so", RTLD_LAZY);
    if (handle) {
        dlclose(handle);
        return opencl_device_count() > 0;
    }
    return false;
#else
    return false;
#endif
}

} // namespace harmonics
