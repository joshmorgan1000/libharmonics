#pragma once

/**
 * @file cycle.hpp
 * @brief Core execution engine used by the unit tests.
 *
 * This header implements a miniature runtime capable of executing the
 * simple graphs used throughout the examples and tests. Only the parts
 * relevant for exercising control flow and backend selection are kept.
 *
 * The runtime executes a graph in "cycles" where each line contains one
 * or more arrows describing tensor movement between nodes. Backends may
 * compile these operations into device specific kernels, however the
 * CPU implementation simply interprets the structure step by step.
 */

#include "harmonics/constant_slab.hpp"
#include "harmonics/thread_pool.hpp"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>
#if HARMONICS_HAS_WASM_SIMD
#include <wasm_simd128.h>
#endif

#include "harmonics/config.hpp"
#include "harmonics/core.hpp"
#include "harmonics/deployment.hpp"
#include "harmonics/fpga_backend.hpp"
#include "harmonics/function_registry.hpp"
#include "harmonics/gpu_backend.hpp"
#if HARMONICS_HAS_VULKAN
#include "harmonics/gpu/vulkan/context.hpp"
#endif
#if HARMONICS_HAS_CUDA
#include "harmonics/gpu/cuda/context.hpp"
#endif
#include "harmonics/cuda_scheduler.hpp"
#include "harmonics/graph.hpp"
#include "harmonics/precision_policy.hpp"
#include "harmonics/wasm_backend.hpp"
#if HARMONICS_HAS_VULKAN
#include <gpu/GlobalFunctionRegistry.hpp>
#endif

// The following code is intentionally header-only to keep compilation of the
// unit tests fast. In a real project the runtime would likely be split into
// separate translation units, but here simplicity and ease of inspection take
// precedence over build times.

namespace harmonics {

struct CudaScheduler;

// Forward declarations for serialization helpers.
void write_tensor(std::ostream& out, const HTensor& t);
HTensor read_tensor(std::istream& in);
void write_string(std::ostream& out, const std::string& s);
std::string read_string(std::istream& in);
int version();

/**
 * @brief Transient state used during execution of a graph cycle.
 *
 * The runtime stores tensors produced at each stage so that both forward
 * and backward edges can access them. Precision decisions are also kept
 * here which allows policies to influence later passes.
 */
struct CycleState {
    std::vector<HTensor> producer_tensors{}; ///< Most recent tensors from producers.
    std::vector<HTensor> layer_tensors{};    ///< Activations for each layer.
    std::vector<HTensor> consumer_tensors{}; ///< Tensors forwarded to consumers.
    std::vector<HTensor> weights{};          ///< Trainable weights per layer.
    std::vector<int> precision_bits{};       ///< Selected bit width per layer.
    ConstantSlab<float> variables{};         ///< Runtime sensor/appendage memory.

    // The vectors above are resized by CycleRuntime based on the graph
    // description. No dynamic allocations occur while stepping through a
    // cycle which keeps the runtime deterministic and easy to reason about
    // in the tests.
};

/**
 * @brief Lightweight runtime capable of executing a single cycle.
 *
 * This class hides the details of backend selection and proof generation
 * used in secure mode. For the purposes of the unit tests it only
 * implements the small subset of functionality required to run the
 * hand written graphs.
 */
class CycleRuntime {
  public:
    explicit CycleRuntime(const HarmonicGraph& g, std::shared_ptr<PrecisionPolicy> policy = nullptr,
                          const DeploymentDescriptor& deploy = {});

    const HarmonicGraph& graph() const { return graph_; }
    /// Access the mutable execution state.
    CycleState& state() { return state_; }
    /// Access the immutable execution state.
    const CycleState& state() const { return state_; }
    Backend backend() const { return backend_; }
    GpuBackend gpu_backend() const { return gpu_backend_; }
    FpgaBackend fpga_backend() const { return fpga_backend_; }
    WasmBackend wasm_backend() const { return wasm_backend_; }

    /** Latest proof digest after a forward pass in secure mode. */
    const std::string& proof() const;
    /** Verify the current proof given the previous digest. */
    bool verify_chain(const std::string& previous) const;
    /** Set the previous digest when verifying across partitions. */
    void set_chain(const std::string& previous) { chain_ = previous; }
    /** Get the digest used as the previous link in the chain. */
    const std::string& chain() const { return chain_; }

    /** Serialize the runtime state to a stream. */
    void save_checkpoint(std::ostream& out) const;
    /** Restore the runtime state from a stream. */
    void load_checkpoint(std::istream& in);

    /** Callback invoked after each arrow during debugging. */
    using DebugCallback = std::function<void(NodeId, NodeId, const HTensor&, bool,
                                             const std::optional<std::string>&)>;
    /// Set an optional debug callback.
    void set_debug_callback(DebugCallback cb) { debug_callback_ = std::move(cb); }

    /** Execute a single forward pass over the graph's cycle. */
    void forward();

    /// Enable or disable multi-threaded CPU execution.
    void enable_multi_threading(bool enable = true) {
        multi_threaded_ = enable;
        if (multi_threaded_ && !pool_)
            pool_ = std::make_unique<ThreadPool>();
    }
    /// Query whether multi-threaded execution is active.
    bool multi_threading_enabled() const { return multi_threaded_; }

  private:
    const HarmonicGraph& graph_;
    std::shared_ptr<PrecisionPolicy> policy_{make_hardware_policy()};
    bool secure_{false};
    Backend backend_{Backend::CPU};
    GpuBackend gpu_backend_{GpuBackend::None};
    FpgaBackend fpga_backend_{FpgaBackend::None};
    WasmBackend wasm_backend_{WasmBackend::None};
    std::string proof_{};
    std::string chain_{};
    CycleState state_{};
    DebugCallback debug_callback_{};
    bool multi_threaded_{false};
    std::unique_ptr<ThreadPool> pool_{};

    void compute_proof();
    void forward_cpu();
    void forward_cpu_threaded();
    void forward_gpu();
    void forward_fpga();
    void forward_wasm();
};

inline void CycleRuntime::forward() {
#if HARMONICS_HAS_VULKAN
    if (backend_ == Backend::GPU) {
        forward_gpu();
        return;
    }
#endif
#if HARMONICS_HAS_OPENCL
    if (backend_ == Backend::FPGA) {
        forward_fpga();
        return;
    }
#endif
#if HARMONICS_HAS_WASM
    if (backend_ == Backend::WASM) {
        forward_wasm();
        return;
    }
#endif
    if (multi_threaded_)
        forward_cpu_threaded();
    else
        forward_cpu();
}

inline CycleRuntime::CycleRuntime(const HarmonicGraph& g, std::shared_ptr<PrecisionPolicy> policy,
                                  const DeploymentDescriptor& deploy)
    : graph_{g}, policy_{policy ? std::move(policy) : make_hardware_policy()},
      secure_{deploy.secure} {
#if HARMONICS_HAS_VULKAN
    if (deploy.gpu_device_index)
        set_vulkan_device_index(*deploy.gpu_device_index);
#endif
    bool handled = false;
    auto try_gpu = [&]() {
#if HARMONICS_HAS_VULKAN
        gpu_backend_ = select_gpu_backend();
        if (gpu_backend_ != GpuBackend::None && gpu_runtime_available()) {
            backend_ = Backend::GPU;
            return true;
        }
        gpu_backend_ = GpuBackend::None;
#endif
        return false;
    };

    auto try_fpga = [&]() {
#if HARMONICS_HAS_OPENCL
        fpga_backend_ = select_fpga_backend();
        if (fpga_backend_ != FpgaBackend::None && fpga_runtime_available()) {
            backend_ = Backend::FPGA;
            return true;
        }
        fpga_backend_ = FpgaBackend::None;
#endif
        return false;
    };

    auto try_wasm = [&]() {
#if HARMONICS_HAS_WASM
        wasm_backend_ = select_wasm_backend();
        if (wasm_backend_ != WasmBackend::None && wasm_runtime_available()) {
            backend_ = Backend::WASM;
            return true;
        }
        wasm_backend_ = WasmBackend::None;
#endif
        return false;
    };

    switch (deploy.backend) {
    case Backend::GPU:
        handled = try_gpu() || try_fpga();
        break;
    case Backend::FPGA:
        handled = try_fpga() || try_gpu();
        break;
    case Backend::WASM:
        handled = try_wasm();
        break;
    case Backend::CPU:
    default: {
        Backend auto_bk = select_accelerator_backend();
        if (auto_bk == Backend::GPU)
            handled = try_gpu();
        else if (auto_bk == Backend::FPGA)
            handled = try_fpga();
        else if (auto_bk == Backend::WASM)
            handled = try_wasm();
        break;
    }
    }
    if (!handled)
        backend_ = Backend::CPU;

    state_.producer_tensors.resize(g.producers.size());
    state_.layer_tensors.resize(g.layers.size());
    state_.consumer_tensors.resize(g.consumers.size());
    state_.weights.resize(g.layers.size());
    state_.precision_bits.resize(g.layers.size());
}

inline void CycleRuntime::forward_cpu() {
    // --------------------------------------------------------------
    // CPU reference implementation of cycle execution.
    // --------------------------------------------------------------

    // Track which producer values have already been fetched for this step.
    std::vector<bool> prod_fetched(graph_.producers.size(), false);
    for (const auto& line : graph_.cycle) {
        HTensor value;

        // Fetch the tensor referenced by the source node.
        switch (line.source.kind) {
        case NodeKind::Producer: {
            if (!prod_fetched[line.source.index]) {
                auto prod = graph_.producer_bindings[line.source.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                value = prod->next();
                // Cache the fetched value for other arrows that read it.
                state_.producer_tensors[line.source.index] = value;
                prod_fetched[line.source.index] = true;
            } else {
                value = state_.producer_tensors[line.source.index];
            }
            break;
        }
        case NodeKind::Layer:
            value = state_.layer_tensors[line.source.index];
            break;
        case NodeKind::Consumer:
            value = state_.consumer_tensors[line.source.index];
            break;
        }

        for (const auto& arrow : line.arrows) {
            if (arrow.backward) {
                HTensor target;
                switch (arrow.target.kind) {
                case NodeKind::Producer: {
                    auto prod = graph_.producer_bindings[arrow.target.index];
                    if (!prod)
                        throw std::runtime_error("producer not bound");
                    target = prod->next();
                    state_.producer_tensors[arrow.target.index] = target;
                    break;
                }
                case NodeKind::Layer:
                    target = state_.layer_tensors[arrow.target.index];
                    break;
                case NodeKind::Consumer:
                    target = state_.consumer_tensors[arrow.target.index];
                    break;
                }
                if (arrow.func && line.source.kind == NodeKind::Layer) {
                    const auto& fn = getLoss(*arrow.func);
                    state_.weights[line.source.index] = fn(value, target);
                }
                if (debug_callback_)
                    debug_callback_(line.source, arrow.target, state_.weights[line.source.index],
                                    true, arrow.func);
                continue; // skip forward update
            }

            if (arrow.func) {
                const auto& fn = getActivation(*arrow.func);
                value = fn(value);
            }

            switch (arrow.target.kind) {
            case NodeKind::Producer:
                state_.producer_tensors[arrow.target.index] = value;
                break;
            case NodeKind::Layer:
                state_.layer_tensors[arrow.target.index] = value;
                if (state_.precision_bits[arrow.target.index] == 0)
                    state_.precision_bits[arrow.target.index] =
                        policy_->select_bits(arrow.target.index);
                break;
            case NodeKind::Consumer:
                state_.consumer_tensors[arrow.target.index] = value;
                break;
            }
            if (debug_callback_)
                debug_callback_(line.source, arrow.target, value, false, arrow.func);
        }
    }

    if (secure_)
        compute_proof();
}

inline void CycleRuntime::forward_cpu_threaded() {
    if (!pool_)
        pool_ = std::make_unique<ThreadPool>();
    std::vector<bool> prod_fetched(graph_.producers.size(), false);
    for (const auto& line : graph_.cycle) {
        HTensor value;
        switch (line.source.kind) {
        case NodeKind::Producer: {
            if (!prod_fetched[line.source.index]) {
                auto prod = graph_.producer_bindings[line.source.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                value = prod->next();
                state_.producer_tensors[line.source.index] = value;
                prod_fetched[line.source.index] = true;
            } else {
                value = state_.producer_tensors[line.source.index];
            }
            break;
        }
        case NodeKind::Layer:
            value = state_.layer_tensors[line.source.index];
            break;
        case NodeKind::Consumer:
            value = state_.consumer_tensors[line.source.index];
            break;
        }

        for (const auto& arrow : line.arrows) {
            pool_->schedule([&, arrow, value]() {
                HTensor local = value;
                if (arrow.backward) {
                    HTensor target;
                    switch (arrow.target.kind) {
                    case NodeKind::Producer: {
                        auto prod = graph_.producer_bindings[arrow.target.index];
                        if (!prod)
                            throw std::runtime_error("producer not bound");
                        target = prod->next();
                        state_.producer_tensors[arrow.target.index] = target;
                        break;
                    }
                    case NodeKind::Layer:
                        target = state_.layer_tensors[arrow.target.index];
                        break;
                    case NodeKind::Consumer:
                        target = state_.consumer_tensors[arrow.target.index];
                        break;
                    }
                    if (arrow.func && line.source.kind == NodeKind::Layer) {
                        const auto& fn = getLoss(*arrow.func);
                        state_.weights[line.source.index] = fn(local, target);
                    }
                    if (debug_callback_)
                        debug_callback_(line.source, arrow.target,
                                        state_.weights[line.source.index], true, arrow.func);
                    return;
                }

                if (arrow.func) {
                    const auto& fn = getActivation(*arrow.func);
                    local = fn(local);
                }

                switch (arrow.target.kind) {
                case NodeKind::Producer:
                    state_.producer_tensors[arrow.target.index] = local;
                    break;
                case NodeKind::Layer:
                    state_.layer_tensors[arrow.target.index] = local;
                    if (state_.precision_bits[arrow.target.index] == 0)
                        state_.precision_bits[arrow.target.index] =
                            policy_->select_bits(arrow.target.index);
                    break;
                case NodeKind::Consumer:
                    state_.consumer_tensors[arrow.target.index] = local;
                    break;
                }
                if (debug_callback_)
                    debug_callback_(line.source, arrow.target, local, false, arrow.func);
            });
        }
        pool_->wait();
    }

    if (secure_)
        compute_proof();
}

/** Information about compiled GPU kernels implementing the cycle. */
struct GpuCycleKernels {
    GpuBackend backend{GpuBackend::None};
    bool compiled{false};
    uint32_t device_index{0};
    struct Op {
        NodeId source{};
        NodeId target{};
        bool backward{false};
        std::optional<std::string> func{};
        std::string shader{};
        std::vector<uint32_t> spirv{};
        int bits{32};
#if HARMONICS_HAS_VULKAN
        VulkanPipeline pipeline{};
#elif HARMONICS_HAS_CUDA
        CudaPipeline pipeline{};
#endif
    };
    std::vector<Op> ops{};
};

struct FpgaCycleKernels {
    FpgaBackend backend{FpgaBackend::None};
    bool compiled{false};
    uint32_t device_index{0};
    struct Op {
        NodeId source{};
        NodeId target{};
        bool backward{false};
        std::optional<std::string> func{};
#if HARMONICS_HAS_OPENCL
        OpenCLKernel kernel{};
#endif
    };
    std::vector<Op> ops{};
};

/// Counter for compile_fpga_cycle_kernels invocations that resulted in a compile.
inline int& compile_fpga_cycle_kernel_compiles() {
    static int count = 0;
    return count;
}

/// Cache of compiled shader bytecode indexed by shader name.
inline std::unordered_map<std::string, std::vector<uint32_t>>& shader_compile_cache() {
    static std::unordered_map<std::string, std::vector<uint32_t>> cache;
    return cache;
}

inline std::size_t shader_cache_limit() {
    if (const char* env = std::getenv("HARMONICS_SHADER_CACHE_LIMIT")) {
        std::size_t val = std::strtoul(env, nullptr, 10);
        return val > 0 ? val : 64;
    }
    return 64;
}

inline void trim_shader_cache() {
    auto& cache = shader_compile_cache();
    while (cache.size() > shader_cache_limit())
        cache.erase(cache.begin());
}

/// Directory used for the persistent shader cache.
inline std::string shader_cache_directory() {
    const char* env = std::getenv("HARMONICS_SHADER_CACHE");
    std::string dir = env ? env : "shader_cache";
    std::filesystem::create_directories(dir);
    return dir;
}

/// Try load a compiled shader from disk.
inline std::optional<std::vector<uint32_t>> load_cached_shader(const std::string& name) {
    std::string key = blake3(name.data(), name.size());
    std::filesystem::path path = shader_cache_directory();
    path /= key + ".spv";
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::nullopt;
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint32_t> data(static_cast<std::size_t>(size) / sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

/// Store a compiled shader to the persistent cache on disk.
inline void save_cached_shader(const std::string& name, const std::vector<uint32_t>& spv) {
    std::string key = blake3(name.data(), name.size());
    std::filesystem::path path = shader_cache_directory();
    path /= key + ".spv";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(spv.data()),
              static_cast<std::streamsize>(spv.size() * sizeof(uint32_t)));
}

#if HARMONICS_HAS_VULKAN || HARMONICS_HAS_CUDA
/// Directory where GLSL shader sources reside.
inline std::string shader_source_directory() {
    const char* env = std::getenv("HARMONICS_SHADER_DIR");
    return env ? env : "shaders";
}

#endif

#if HARMONICS_HAS_VULKAN || HARMONICS_HAS_CUDA
inline std::vector<uint32_t> real_compile_shader(const std::string& name) {
    std::filesystem::path dir = shader_source_directory();
    std::filesystem::path src = dir / (name + ".comp");
    std::filesystem::path out = std::filesystem::temp_directory_path() / (name + ".spv");
    std::string cmd =
        "glslangValidator -V \"" + src.string() + "\" -o \"" + out.string() + "\" > /dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("shader compilation failed: " + src.string());
    std::ifstream in(out, std::ios::binary);
    if (!in)
        throw std::runtime_error("unable to read compiled shader: " + out.string());
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint32_t> data(static_cast<std::size_t>(size) / sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(data.data()), size);
    std::filesystem::remove(out);
    return data;
}

inline std::vector<uint32_t> compile_shader(const std::string& name) {
#if HARMONICS_HAS_VULKAN
    auto& reg = GPUFunctionRegistry::getInstance();
    auto spv = reg.getKernel(name);
    if (!spv.empty())
        return spv;
#endif
    return real_compile_shader(name);
}
#endif

/// Counter for compile_cycle_kernels invocations that resulted in a compile.
inline int& compile_cycle_kernel_compiles() {
    static int count = 0;
    return count;
}

/**
 * @brief Create GPU kernel descriptors from the graph.
 */
inline GpuCycleKernels compile_cycle_kernels(const HarmonicGraph& g, PrecisionPolicy& policy) {
    static std::unordered_map<std::string, GpuCycleKernels> cache;
    std::string key = graph_digest(g);
    for (std::size_t i = 0; i < g.layers.size(); ++i)
        key += "_" + std::to_string(policy.select_bits(i));
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    ++compile_cycle_kernel_compiles();

    GpuCycleKernels kernels{};
#if HARMONICS_HAS_VULKAN
    kernels.backend = GpuBackend::Vulkan;
    kernels.compiled = true;
    kernels.device_index = vulkan_device_index();
#elif HARMONICS_HAS_CUDA
    kernels.backend = GpuBackend::Cuda;
    kernels.compiled = true;
    kernels.device_index = cuda_device_index();
#else
    kernels.backend = GpuBackend::None;
    kernels.compiled = false;
#endif
    for (const auto& line : g.cycle) {
        for (const auto& arrow : line.arrows) {
            int bits = 32;
            if (arrow.target.kind == NodeKind::Layer)
                bits = policy.select_bits(arrow.target.index);
            std::string shader = arrow.func ? *arrow.func : "identity";
            std::string shader_key = shader + "_" + std::to_string(bits);
            std::vector<uint32_t> spv{};
#if HARMONICS_HAS_VULKAN || HARMONICS_HAS_CUDA
            auto& scache = shader_compile_cache();
            auto sit = scache.find(shader_key);
            if (sit == scache.end()) {
#if HARMONICS_HAS_VULKAN
                if (auto disk = load_cached_shader(shader_key)) {
                    spv = *disk;
                } else {
                    spv = compile_shader(shader);
                    save_cached_shader(shader_key, spv);
                }
#else
                spv = compile_shader(shader_key);
#endif
                scache.emplace(shader_key, spv);
                trim_shader_cache();
            } else {
                spv = sit->second;
            }
#endif
            GpuCycleKernels::Op op{line.source,
                                   arrow.target,
                                   arrow.backward,
                                   arrow.func,
                                   shader_key,
                                   spv,
                                   bits
#if HARMONICS_HAS_VULKAN
                                   ,
                                   create_compute_pipeline(spv)
#elif HARMONICS_HAS_CUDA
                                   ,
                                   create_compute_pipeline(spv)
#endif
            };
            kernels.ops.push_back(op);
        }
    }
    (void)g;
    cache.emplace(key, kernels);
    return kernels;
}

/**
 * @brief Create FPGA kernel descriptors from the graph.
 */
inline FpgaCycleKernels compile_fpga_cycle_kernels(const HarmonicGraph& g) {
    static std::unordered_map<std::string, FpgaCycleKernels> cache;
    std::string key = graph_digest(g);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    ++compile_fpga_cycle_kernel_compiles();

    FpgaCycleKernels kernels{};
#if HARMONICS_HAS_OPENCL
    kernels.backend = FpgaBackend::OpenCL;
    kernels.compiled = true;
    kernels.device_index = opencl_device_index();
#else
    kernels.backend = FpgaBackend::None;
    kernels.compiled = false;
#endif
    for (const auto& line : g.cycle) {
        for (const auto& arrow : line.arrows) {
            FpgaCycleKernels::Op op{line.source, arrow.target, arrow.backward, arrow.func};
#if HARMONICS_HAS_OPENCL
            if (kernels.backend == FpgaBackend::OpenCL) {
                if (arrow.func)
                    op.kernel = opencl_build_activation_kernel(*arrow.func);
                else
                    op.kernel = opencl_build_activation_kernel("identity");
            }
#endif
            kernels.ops.push_back(op);
        }
    }
    (void)g;
    cache.emplace(key, kernels);
    return kernels;
}

/**
 * @brief Execute the compiled GPU kernels for a cycle.
 */
inline void launch_cycle_kernel(const GpuCycleKernels& kernels, const HarmonicGraph& g,
                                CycleState& state, PrecisionPolicy& policy) {
    if (!kernels.compiled)
        throw std::runtime_error("GPU kernels not available");
    std::vector<GpuTensor> prod_dev(state.producer_tensors.size());
    std::vector<GpuTensor> layer_dev(state.layer_tensors.size());
    std::vector<GpuTensor> cons_dev(state.consumer_tensors.size());
    std::vector<bool> prod_fetched(g.producers.size(), false);

    for (const auto& op : kernels.ops) {
        HTensor value_host;
        GpuTensor value_dev;

        // Load the value of the operation's source node.
        switch (op.source.kind) {
        case NodeKind::Producer: {
            if (!prod_fetched[op.source.index]) {
                auto prod = g.producer_bindings[op.source.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                value_host = prod->next();
                value_dev = to_device(value_host);
                prod_dev[op.source.index] = value_dev;
                prod_fetched[op.source.index] = true;
            } else {
                value_dev = prod_dev[op.source.index];
                value_host = to_host(value_dev);
            }
            break;
        }
        case NodeKind::Layer:
            value_dev = layer_dev[op.source.index];
            value_host = to_host(value_dev);
            break;
        case NodeKind::Consumer:
            value_dev = cons_dev[op.source.index];
            value_host = to_host(value_dev);
            break;
        }

        if (op.backward) {
            HTensor target_host;
            switch (op.target.kind) {
            case NodeKind::Producer: {
                auto prod = g.producer_bindings[op.target.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                target_host = prod->next();
                prod_dev[op.target.index] = to_device(target_host);
                break;
            }
            case NodeKind::Layer:
                target_host = to_host(layer_dev[op.target.index]);
                break;
            case NodeKind::Consumer:
                target_host = to_host(cons_dev[op.target.index]);
                break;
            }
            if (op.func && op.source.kind == NodeKind::Layer) {
                const auto& fn = getLoss(*op.func);
                state.weights[op.source.index] = fn(value_host, target_host);
            }
            continue;
        }

        if (op.func) {
            const auto& fn = getActivation(*op.func);
            value_host = fn(value_host);
            value_dev = to_device(value_host);
        }

        switch (op.target.kind) {
        case NodeKind::Producer:
            prod_dev[op.target.index] = value_dev;
            break;
        case NodeKind::Layer:
            layer_dev[op.target.index] = value_dev;
            if (state.precision_bits[op.target.index] == 0)
                state.precision_bits[op.target.index] = policy.select_bits(op.target.index);
            break;
        case NodeKind::Consumer:
            cons_dev[op.target.index] = value_dev;
            break;
        }
    }

    // Copy device tensors back to host memory for inspection.
    std::vector<std::future<HTensor>> prod_futs(prod_dev.size());
    for (std::size_t i = 0; i < prod_dev.size(); ++i)
        prod_futs[i] = to_host_async(prod_dev[i]);
    std::vector<std::future<HTensor>> layer_futs(layer_dev.size());
    for (std::size_t i = 0; i < layer_dev.size(); ++i)
        layer_futs[i] = to_host_async(layer_dev[i]);
    std::vector<std::future<HTensor>> cons_futs(cons_dev.size());
    for (std::size_t i = 0; i < cons_dev.size(); ++i)
        cons_futs[i] = to_host_async(cons_dev[i]);
    for (std::size_t i = 0; i < prod_dev.size(); ++i)
        state.producer_tensors[i] = prod_futs[i].get();
    for (std::size_t i = 0; i < layer_dev.size(); ++i)
        state.layer_tensors[i] = layer_futs[i].get();
    for (std::size_t i = 0; i < cons_dev.size(); ++i)
        state.consumer_tensors[i] = cons_futs[i].get();
}

/**
 * @brief Execute the compiled FPGA kernels for a cycle.
 */
inline void launch_fpga_cycle_kernel(const FpgaCycleKernels& kernels, const HarmonicGraph& g,
                                     CycleState& state, PrecisionPolicy& policy) {
    if (!kernels.compiled)
        throw std::runtime_error("FPGA kernels not available");
    std::vector<FpgaTensor> prod_dev(state.producer_tensors.size());
    std::vector<FpgaTensor> layer_dev(state.layer_tensors.size());
    std::vector<FpgaTensor> cons_dev(state.consumer_tensors.size());
    std::vector<bool> prod_fetched(g.producers.size(), false);

    for (const auto& op : kernels.ops) {
        HTensor value_host;
        FpgaTensor value_dev;

        // Load the value of the operation's source node.
        switch (op.source.kind) {
        case NodeKind::Producer: {
            if (!prod_fetched[op.source.index]) {
                auto prod = g.producer_bindings[op.source.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                value_host = prod->next();
                value_dev = fpga_to_device(value_host);
                prod_dev[op.source.index] = value_dev;
                prod_fetched[op.source.index] = true;
            } else {
                value_dev = prod_dev[op.source.index];
                value_host = fpga_to_host(value_dev);
            }
            break;
        }
        case NodeKind::Layer:
            value_dev = layer_dev[op.source.index];
            value_host = fpga_to_host(value_dev);
            break;
        case NodeKind::Consumer:
            value_dev = cons_dev[op.source.index];
            value_host = fpga_to_host(value_dev);
            break;
        }

        if (op.backward) {
            HTensor target_host;
            switch (op.target.kind) {
            case NodeKind::Producer: {
                auto prod = g.producer_bindings[op.target.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                target_host = prod->next();
                prod_dev[op.target.index] = fpga_to_device(target_host);
                break;
            }
            case NodeKind::Layer:
                target_host = fpga_to_host(layer_dev[op.target.index]);
                break;
            case NodeKind::Consumer:
                target_host = fpga_to_host(cons_dev[op.target.index]);
                break;
            }
            if (op.func && op.source.kind == NodeKind::Layer) {
                const auto& fn = getLoss(*op.func);
                state.weights[op.source.index] = fn(value_host, target_host);
            }
            continue;
        }

        FpgaTensor result_dev;
#if HARMONICS_USE_OPENCL_RT
        if (kernels.backend == FpgaBackend::OpenCL && op.kernel.kernel) {
            std::size_t bytes = opencl_buffer_size(value_dev.device_data);
            result_dev.dtype = value_dev.dtype;
            result_dev.shape = value_dev.shape;
            result_dev.device_data = opencl_malloc(bytes);
            auto& ctx = get_opencl_context();
            clSetKernelArg(op.kernel.kernel, 0, sizeof(cl_mem), &value_dev.device_data.buffer);
            clSetKernelArg(op.kernel.kernel, 1, sizeof(cl_mem), &result_dev.device_data.buffer);
            size_t global = bytes / sizeof(float);
            clEnqueueNDRangeKernel(ctx.queue, op.kernel.kernel, 1, nullptr, &global, nullptr, 0,
                                   nullptr, nullptr);
            clFinish(ctx.queue);
        } else
#endif
            if (op.func) {
            const auto& fn = getActivation(*op.func);
            value_host = fn(value_host);
            result_dev = fpga_to_device(value_host);
        } else {
            result_dev = fpga_copy_tensor(value_dev);
        }

        switch (op.target.kind) {
        case NodeKind::Producer:
            prod_dev[op.target.index] = result_dev;
            break;
        case NodeKind::Layer:
            layer_dev[op.target.index] = result_dev;
            if (state.precision_bits[op.target.index] == 0)
                state.precision_bits[op.target.index] = policy.select_bits(op.target.index);
            break;
        case NodeKind::Consumer:
            cons_dev[op.target.index] = result_dev;
            break;
        }
    }

    // Copy device tensors back to host memory for inspection.
    std::vector<std::future<HTensor>> prod_futs(prod_dev.size());
    for (std::size_t i = 0; i < prod_dev.size(); ++i)
        prod_futs[i] = fpga_to_host_async(prod_dev[i]);
    std::vector<std::future<HTensor>> layer_futs(layer_dev.size());
    for (std::size_t i = 0; i < layer_dev.size(); ++i)
        layer_futs[i] = fpga_to_host_async(layer_dev[i]);
    std::vector<std::future<HTensor>> cons_futs(cons_dev.size());
    for (std::size_t i = 0; i < cons_dev.size(); ++i)
        cons_futs[i] = fpga_to_host_async(cons_dev[i]);
    for (std::size_t i = 0; i < prod_dev.size(); ++i)
        state.producer_tensors[i] = prod_futs[i].get();
    for (std::size_t i = 0; i < layer_dev.size(); ++i)
        state.layer_tensors[i] = layer_futs[i].get();
    for (std::size_t i = 0; i < cons_dev.size(); ++i)
        state.consumer_tensors[i] = cons_futs[i].get();
}

inline void CycleRuntime::forward_gpu() {
    // Execute the cycle using the GPU backend when available.
    auto kernels = compile_cycle_kernels(graph_, *policy_);
    if (kernels.backend == GpuBackend::Cuda)
        CudaScheduler::launch(kernels, graph_, state_, *policy_);
    else
        launch_cycle_kernel(kernels, graph_, state_, *policy_);
    if (secure_)
        compute_proof();
}

inline void CycleRuntime::forward_fpga() {
    // Execute the cycle using the FPGA backend when available.
    auto kernels = compile_fpga_cycle_kernels(graph_);
    launch_fpga_cycle_kernel(kernels, graph_, state_, *policy_);
    if (secure_)
        compute_proof();
}

inline void wasm_copy_tensor(const HTensor& src, HTensor& dst) {
    dst = HTensor{src.dtype(), src.shape()};
    dst.data().resize(src.data().size());
#if HARMONICS_HAS_WASM_SIMD
    if (src.dtype() == HTensor::DType::Float32) {
        const float* s = reinterpret_cast<const float*>(src.data().data());
        float* d = reinterpret_cast<float*>(dst.data().data());
        std::size_t n = src.data().size() / sizeof(float);
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            v128_t v = wasm_v128_load(&s[i]);
            wasm_v128_store(&d[i], v);
        }
        for (; i < n; ++i)
            d[i] = s[i];
    } else {
        std::memcpy(dst.data().data(), src.data().data(), src.data().size());
    }
#else
    dst.data() = src.data();
#endif
}

inline void CycleRuntime::forward_wasm() {
    std::vector<bool> prod_fetched(graph_.producers.size(), false);
    for (const auto& line : graph_.cycle) {
        HTensor value;

        switch (line.source.kind) {
        case NodeKind::Producer: {
            if (!prod_fetched[line.source.index]) {
                auto prod = graph_.producer_bindings[line.source.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                value = prod->next();
                wasm_copy_tensor(value, state_.producer_tensors[line.source.index]);
                prod_fetched[line.source.index] = true;
            } else {
                value = state_.producer_tensors[line.source.index];
            }
            break;
        }
        case NodeKind::Layer:
            value = state_.layer_tensors[line.source.index];
            break;
        case NodeKind::Consumer:
            value = state_.consumer_tensors[line.source.index];
            break;
        }

        for (const auto& arrow : line.arrows) {
            if (arrow.backward) {
                HTensor target;
                switch (arrow.target.kind) {
                case NodeKind::Producer: {
                    auto prod = graph_.producer_bindings[arrow.target.index];
                    if (!prod)
                        throw std::runtime_error("producer not bound");
                    target = prod->next();
                    wasm_copy_tensor(target, state_.producer_tensors[arrow.target.index]);
                    break;
                }
                case NodeKind::Layer:
                    target = state_.layer_tensors[arrow.target.index];
                    break;
                case NodeKind::Consumer:
                    target = state_.consumer_tensors[arrow.target.index];
                    break;
                }
                if (arrow.func && line.source.kind == NodeKind::Layer) {
                    const auto& fn = getLoss(*arrow.func);
                    state_.weights[line.source.index] = fn(value, target);
                }
                continue;
            }

            HTensor out_val = value;
            if (arrow.func) {
                const auto& fn = getActivation(*arrow.func);
                out_val = fn(value);
            }

            switch (arrow.target.kind) {
            case NodeKind::Producer:
                wasm_copy_tensor(out_val, state_.producer_tensors[arrow.target.index]);
                break;
            case NodeKind::Layer:
                wasm_copy_tensor(out_val, state_.layer_tensors[arrow.target.index]);
                if (state_.precision_bits[arrow.target.index] == 0)
                    state_.precision_bits[arrow.target.index] =
                        policy_->select_bits(arrow.target.index);
                break;
            case NodeKind::Consumer:
                wasm_copy_tensor(out_val, state_.consumer_tensors[arrow.target.index]);
                break;
            }
        }
    }

    if (secure_)
        compute_proof();
}

// ---------------------------------------------------------------------------
// Proof generation and verification
//
// When running in secure mode the runtime computes a short digest after
// each forward pass. This digest is used to verify that all parties
// involved in a distributed computation observed the same intermediate
// values without having to share those values directly. The mechanism is
// intentionally simplistic but sufficient for testing the proof chaining
// logic.
// ---------------------------------------------------------------------------

inline void CycleRuntime::compute_proof() {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    // Chain the previous digest into the hash to form a proof chain.
    if (!chain_.empty())
        blake3_hasher_update(&hasher, chain_.data(), chain_.size());
    for (const auto& t : state_.layer_tensors) {
        if (!t.data().empty())
            blake3_hasher_update(&hasher, t.data().data(), t.data().size());
    }
    uint8_t out[BLAKE3_OUT_LEN];
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
    proof_ = to_hex(out, BLAKE3_OUT_LEN);
    chain_ = proof_;
}

inline const std::string& CycleRuntime::proof() const { return proof_; }

inline bool CycleRuntime::verify_chain(const std::string& previous) const {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    if (!previous.empty())
        // Include the provided digest when verifying across partitions.
        blake3_hasher_update(&hasher, previous.data(), previous.size());
    for (const auto& t : state_.layer_tensors) {
        if (!t.data().empty())
            blake3_hasher_update(&hasher, t.data().data(), t.data().size());
    }
    uint8_t out[BLAKE3_OUT_LEN];
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
    auto expected = to_hex(out, BLAKE3_OUT_LEN);
    return expected == proof_;
}

// End of execution engine implementation.

/*
 * The implementations above are intentionally concise and primarily exist
 * to support the Harmonics unit tests. They should not be considered a
 * production quality runtime. The goal is to demonstrate how different
 * backends might be integrated and to provide a vehicle for exercising
 * the API surface without introducing unnecessary complexity.
 */

} // namespace harmonics

#include "harmonics/serialization.hpp"

namespace harmonics {

inline void save_cycle_state(const CycleState& s, std::ostream& out) {
    auto write_vec = [&](const std::vector<HTensor>& v) {
        std::uint32_t n = static_cast<std::uint32_t>(v.size());
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& t : v)
            write_tensor(out, t);
    };
    write_vec(s.producer_tensors);
    write_vec(s.layer_tensors);
    write_vec(s.consumer_tensors);
    write_vec(s.weights);
    std::uint32_t n = static_cast<std::uint32_t>(s.precision_bits.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (int b : s.precision_bits)
        out.write(reinterpret_cast<const char*>(&b), sizeof(b));
}

inline CycleState load_cycle_state(std::istream& in) {
    auto read_vec = [&](std::vector<HTensor>& v) {
        std::uint32_t n;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        v.resize(n);
        for (std::uint32_t i = 0; i < n; ++i)
            v[i] = read_tensor(in);
    };
    CycleState s;
    read_vec(s.producer_tensors);
    read_vec(s.layer_tensors);
    read_vec(s.consumer_tensors);
    read_vec(s.weights);
    std::uint32_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    s.precision_bits.resize(n);
    for (std::uint32_t i = 0; i < n; ++i)
        in.read(reinterpret_cast<char*>(&s.precision_bits[i]), sizeof(int));
    return s;
}

inline void CycleRuntime::save_checkpoint(std::ostream& out) const {
    out.write("HRTC", 4);
    std::uint32_t ver = static_cast<std::uint32_t>(version());
    out.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    save_cycle_state(state_, out);
    write_string(out, chain_);
}

inline void CycleRuntime::load_checkpoint(std::istream& in) {
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "HRTC")
        throw std::runtime_error("invalid checkpoint");
    std::uint32_t ver;
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    (void)ver;
    state_ = load_cycle_state(in);
    chain_ = read_string(in);
    proof_.clear();
}

inline void CudaScheduler::launch(const GpuCycleKernels& kernels, const HarmonicGraph& g,
                                  CycleState& state, PrecisionPolicy& policy) {
    if (!kernels.compiled)
        throw std::runtime_error("CUDA kernels not available");
    std::vector<GpuTensor> prod_dev(state.producer_tensors.size());
    std::vector<GpuTensor> layer_dev(state.layer_tensors.size());
    std::vector<GpuTensor> cons_dev(state.consumer_tensors.size());
    std::vector<bool> prod_fetched(g.producers.size(), false);

    for (const auto& op : kernels.ops) {
        HTensor value_host;
        GpuTensor value_dev;

        switch (op.source.kind) {
        case NodeKind::Producer: {
            if (!prod_fetched[op.source.index]) {
                auto prod = g.producer_bindings[op.source.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                value_host = prod->next();
                value_dev = to_device(value_host);
                prod_dev[op.source.index] = value_dev;
                prod_fetched[op.source.index] = true;
            } else {
                value_dev = prod_dev[op.source.index];
                value_host = to_host(value_dev);
            }
            break;
        }
        case NodeKind::Layer:
            value_dev = layer_dev[op.source.index];
            value_host = to_host(value_dev);
            break;
        case NodeKind::Consumer:
            value_dev = cons_dev[op.source.index];
            value_host = to_host(value_dev);
            break;
        }

        if (op.backward) {
            HTensor target_host;
            switch (op.target.kind) {
            case NodeKind::Producer: {
                auto prod = g.producer_bindings[op.target.index];
                if (!prod)
                    throw std::runtime_error("producer not bound");
                target_host = prod->next();
                prod_dev[op.target.index] = to_device(target_host);
                break;
            }
            case NodeKind::Layer:
                target_host = to_host(layer_dev[op.target.index]);
                break;
            case NodeKind::Consumer:
                target_host = to_host(cons_dev[op.target.index]);
                break;
            }
            if (op.func && op.source.kind == NodeKind::Layer) {
                const auto& fn = getLoss(*op.func);
                state.weights[op.source.index] = fn(value_host, target_host);
            }
            continue;
        }

        if (op.func) {
            const auto& fn = getActivation(*op.func);
            value_host = fn(value_host);
            value_dev = to_device(value_host);
        }

        switch (op.target.kind) {
        case NodeKind::Producer:
            prod_dev[op.target.index] = value_dev;
            break;
        case NodeKind::Layer:
            layer_dev[op.target.index] = value_dev;
            if (state.precision_bits[op.target.index] == 0)
                state.precision_bits[op.target.index] = policy.select_bits(op.target.index);
            break;
        case NodeKind::Consumer:
            cons_dev[op.target.index] = value_dev;
            break;
        }
    }

    std::vector<std::future<HTensor>> prod_futs(prod_dev.size());
    for (std::size_t i = 0; i < prod_dev.size(); ++i)
        prod_futs[i] = to_host_async(prod_dev[i]);
    std::vector<std::future<HTensor>> layer_futs(layer_dev.size());
    for (std::size_t i = 0; i < layer_dev.size(); ++i)
        layer_futs[i] = to_host_async(layer_dev[i]);
    std::vector<std::future<HTensor>> cons_futs(cons_dev.size());
    for (std::size_t i = 0; i < cons_dev.size(); ++i)
        cons_futs[i] = to_host_async(cons_dev[i]);
    for (std::size_t i = 0; i < prod_dev.size(); ++i)
        state.producer_tensors[i] = prod_futs[i].get();
    for (std::size_t i = 0; i < layer_dev.size(); ++i)
        state.layer_tensors[i] = layer_futs[i].get();
    for (std::size_t i = 0; i < cons_dev.size(); ++i)
        state.consumer_tensors[i] = cons_futs[i].get();
}

} // namespace harmonics
