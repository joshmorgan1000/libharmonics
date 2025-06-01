# Accelerator Backends
# CUDA Backend

The CUDA backend provides full compute and memory management helpers for
executing kernels on NVIDIA GPUs. All wrapper functions live in
`include/harmonics/cuda_adapter.hpp` and mirror the interface of the Vulkan
adapter so tensors can be moved between host and device memory with a few
calls. The context utilities are available in
`include/harmonics/gpu/cuda/context.hpp`.

## Building

Enable CUDA support when configuring CMake:

```bash
cmake -S . -B build -DHARMONICS_HAS_CUDA=1
```

A CUDA toolkit (11 or newer) must be available so that `cuda_runtime.h` can be located by the compiler.

## Runtime selection

Set `HARMONICS_ENABLE_CUDA=1` to allow the loader to initialise CUDA at runtime. A specific device can be selected with the environment variable `HARMONICS_CUDA_DEVICE`. If no device is specified the runtime picks the GPU with the largest amount of memory. If no compatible device is found the runtime automatically falls back to the CPU path.

## Using the wrapper

Tensors can be copied to GPU memory with `to_device()` and transferred back with
`to_host()`. The helper functions allocate `CudaBuffer` objects which hold the
device pointers returned by `cudaMalloc`. When the project is built without a
CUDA toolkit these helpers fall back to host allocations so unit tests can run
without a GPU.

# Vulkan GPU Wrapper

The Vulkan backend provides a minimal wrapper around compute shaders and device buffers.
All helper functions live in `include/harmonics/vulkan_adapter.hpp` and abstract
away the low level Vulkan API so that tensors can be moved to and from GPU
memory with a few function calls.

## Building

Enable the Vulkan path by configuring CMake with `-DHARMONICS_HAS_VULKAN=1`.
A Vulkan SDK (version 1.3 or newer) must be installed on the host system.

## Runtime selection

Choose the GPU backend at runtime by setting `backend = Backend::GPU` in the
`DeploymentDescriptor` or by calling `select_accelerator_backend()` to pick the
first available accelerator. A specific Vulkan device can be selected with the
environment variable `HARMONICS_VULKAN_DEVICE` or by calling
`set_vulkan_device_index()` before constructing a `CycleRuntime`.

The device buffer ring used to stream tensors defaults to three buffers. This
can be adjusted by setting `HARMONICS_DEVICE_RING_SIZE` before launching the
runtime.

Buffers returned to the pool are kept for reuse to avoid repeated allocations.
The maximum number of cached buffers can be set with `HARMONICS_DEVICE_POOL_LIMIT`.
When the limit is exceeded the oldest buffers are freed.

Copy operations between host and device are timed and accumulated in the global
`MemoryTransferStats` counters. Applications can access these metrics via
`memory_transfer_stats()` to analyse throughput.

If no device is specified, the runtime scans all available GPUs and selects the
one with the largest amount of device-local memory, preferring discrete GPUs
when present.

## Compilation workflow

Use `compile_cycle_kernels(graph, policy)` to translate the arrows of a
`HarmonicGraph` into SPIR-V shaders. Compiled bytecode is cached in memory and
can optionally be persisted on disk. The cache directory defaults to
`shader_cache/` and may be overridden with the `HARMONICS_SHADER_CACHE`
environment variable. The number of entries kept in memory is limited by
`HARMONICS_SHADER_CACHE_LIMIT` which defaults to `64`.

After compilation the kernels are launched with
`launch_cycle_kernel(kernels, graph, state, policy)`. See
[IncrementalCompilationCache.md](IncrementalCompilationCache.md) for a complete
overview of the caching strategy.

# FPGA Backend

The FPGA backend uses the OpenCL adapter to offload tensors to devices that expose an OpenCL implementation. When no accelerator is available the runtime transparently falls back to the CPU implementation.

## Building

Enable the OpenCL path when configuring CMake:

```bash
cmake -S . -B build -DHARMONICS_HAS_OPENCL=1
```

A working OpenCL SDK is required so that `<CL/cl.h>` can be found by the compiler.

## Runtime selection

Set `HARMONICS_ENABLE_OPENCL=1` at runtime to initialise the OpenCL subsystem. When multiple platforms are installed, the environment variable `HARMONICS_OPENCL_PLATFORM` selects the platform index while `HARMONICS_OPENCL_DEVICE` chooses the device on that platform. The backend is chosen by setting `DeploymentDescriptor::backend` to `Backend::FPGA` or by calling `select_accelerator_backend()` and letting the runtime pick the first available device.

Call `opencl_device_count()` to query the number of devices detected by the runtime.

If the runtime checks fail the graph transparently falls back to the CPU implementation.

## Copying tensors

`FpgaTensor` objects hold device storage. Use `fpga_to_device()` to copy an `HTensor` to the accelerator and `fpga_to_host()` to retrieve the result back into host memory.

```cpp
harmonics::HTensor t{harmonics::HTensor::DType::Float32, {2}};
auto dev = harmonics::fpga_to_device(t);
auto host = harmonics::fpga_to_host(dev);
```

These helpers provide the functionality required by the production runtime.


# Incremental Compilation Cache Workflow

When graphs are executed on the GPU the runtime compiles each operation into a SPIR-V shader. To avoid recompiling identical shaders every run, Harmonics stores the results in an in-memory cache. Subsequent compilations reuse the cached bytecode which greatly speeds up startup times for large graphs.

## 1. Graph level cache

The helper `compile_cycle_kernels()` takes a `HarmonicGraph` and returns the GPU kernel descriptors for its cycle. A static map keyed by `graph_digest()` keeps the compiled descriptors. Calling the function again with a graph that yields the same digest simply returns the cached descriptors without touching the shader compiler.

## 2. Shader level cache

While compiling a new graph, each arrow in the cycle resolves to a shader name. Before invoking the compiler, `compile_cycle_kernels()` consults `shader_compile_cache()`. If the name already exists in the map the previously compiled SPIR-V blob is reused. Otherwise the shader is compiled and inserted into the cache.

```cpp
// Force a clean compile pass
harmonics::shader_compile_cache().clear();

auto kernels = harmonics::compile_cycle_kernels(graph, policy);
```

When Vulkan support is enabled, compiled shaders are also written to disk. The
files reside under `shader_cache/` by default and the directory can be changed
via the `HARMONICS_SHADER_CACHE` environment variable. Clearing the in-memory
map or deleting these files forces shaders to be recompiled.

## 3. Workflow summary

1. `compile_cycle_kernels()` computes the digest of the graph.
2. If a descriptor for that digest exists, it is returned immediately.
3. Otherwise each shader is looked up in `shader_compile_cache()` and compiled only once.
4. The resulting descriptors are stored in the graph level cache for future use.

See [Architecture.md](Architecture.md#93-incremental-compilation-cache) for an overview of how this fits into the GPU back end.

