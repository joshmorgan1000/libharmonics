#pragma once

#include "harmonics/config.hpp"
#include "harmonics/cuda_adapter.hpp"
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
#include <cuda.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

namespace harmonics {

inline std::optional<uint32_t>& cuda_device_override() {
    static std::optional<uint32_t> index;
    return index;
}

inline void set_cuda_device_index(uint32_t index) { cuda_device_override() = index; }

#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
inline int select_best_cuda_device(int count) {
    int best = 0;
    std::size_t best_score = 0;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            std::size_t score = prop.totalGlobalMem;
            if (!prop.integrated)
                score += prop.totalGlobalMem; // favour discrete GPUs
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
    }
    return best;
}
#endif

#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
struct CudaContext {
    int device_index{0};
};

inline CudaContext& get_cuda_context() {
    static CudaContext ctx{};
    static bool init = false;
    if (!init) {
        int count = 0;
        if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
            uint32_t index = 0;
            if (const char* env = std::getenv("HARMONICS_CUDA_DEVICE")) {
                int idx = std::atoi(env);
                if (idx >= 0 && idx < count)
                    index = static_cast<uint32_t>(idx);
            } else if (cuda_device_override()) {
                uint32_t idx = *cuda_device_override();
                if (idx < static_cast<uint32_t>(count))
                    index = idx;
            } else {
                index = static_cast<uint32_t>(select_best_cuda_device(count));
            }
            ctx.device_index = static_cast<int>(index);
            cudaSetDevice(ctx.device_index);
        }
        init = true;
    }
    return ctx;
}

inline void dispatch_cuda_kernel(CUfunction func, dim3 grid, dim3 block, void** args) {
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
    cuLaunchKernel(func, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, 0, args, nullptr);
#else
    (void)func;
    (void)grid;
    (void)block;
    (void)args;
#endif
}

struct CudaPipeline {
    CUmodule module{0};
    CUfunction func{0};
};

inline CudaPipeline create_compute_pipeline(const std::vector<uint32_t>& ptx) {
    CudaPipeline pipe{};
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
    cuModuleLoadData(&pipe.module, ptx.data());
    cuModuleGetFunction(&pipe.func, pipe.module, "main");
#else
    (void)ptx;
#endif
    return pipe;
}

inline void destroy_pipeline(CudaPipeline& pipe) {
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
    if (pipe.module)
        cuModuleUnload(pipe.module);
#endif
    pipe.module = 0;
    pipe.func = 0;
}

inline void dispatch_compute_pipeline(const CudaPipeline& pipe, uint32_t x, uint32_t y = 1,
                                      uint32_t z = 1) {
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
    void* params[] = {};
    cuLaunchKernel(pipe.func, x, y, z, 1, 1, 1, 0, 0, params, nullptr);
#else
    (void)pipe;
    (void)x;
    (void)y;
    (void)z;
#endif
}

/**
 * @brief Dispatch a CUDA pipeline asynchronously.
 */
inline std::future<void> dispatch_compute_pipeline_async(const CudaPipeline& pipe, uint32_t x,
                                                         uint32_t y = 1, uint32_t z = 1) {
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
    return std::async(std::launch::async, [=]() { dispatch_compute_pipeline(pipe, x, y, z); });
#else
    (void)pipe;
    (void)x;
    (void)y;
    (void)z;
    return std::async(std::launch::async, [] {});
#endif
}

#else
struct CudaContext {
    int device_index{0};
};

inline CudaContext& get_cuda_context() {
    static CudaContext ctx{};
    return ctx;
}

inline void dispatch_cuda_kernel(CUfunction /*func*/, dim3 /*grid*/, dim3 /*block*/,
                                 void** /*args*/) {}

struct CudaPipeline {
    CUmodule module{0};
    CUfunction func{0};
};

inline CudaPipeline create_compute_pipeline(const std::vector<uint32_t>&) { return {}; }

inline void destroy_pipeline(CudaPipeline&) {}

inline void dispatch_compute_pipeline(const CudaPipeline&, uint32_t, uint32_t = 1, uint32_t = 1) {}

inline std::future<void> dispatch_compute_pipeline_async(const CudaPipeline&, uint32_t,
                                                         uint32_t = 1, uint32_t = 1) {
    return std::async(std::launch::async, [] {});
}
#endif

inline uint32_t cuda_device_index() {
#if HARMONICS_HAS_CUDA && HARMONICS_USE_CUDA_RT
    return static_cast<uint32_t>(get_cuda_context().device_index);
#else
    if (cuda_device_override())
        return *cuda_device_override();
    return 0;
#endif
}

} // namespace harmonics
