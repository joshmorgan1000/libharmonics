# CUDA Compute Pipeline

This document explains how Harmonics executes compute kernels on NVIDIA GPUs.
The CUDA path mirrors the Vulkan backend so shaders can be written once and
used across accelerators.

## Compilation

`compile_cycle_kernels(graph, policy)` walks over the arrows of a graph and
calls `compile_shader()` for each unique operation. The GLSL source is
translated to PTX by `glslangValidator`. Compiled kernels are cached in memory
and stored under `shader_cache/` when `HARMONICS_SHADER_CACHE` is set. The
cache avoids recompiling identical shaders on subsequent runs.

Each entry contains the raw PTX blob and a `CudaPipeline` object returned by
`create_compute_pipeline()`. The pipeline wraps a `CUmodule` and `CUfunction`
ready for dispatch.

## Execution

`CudaScheduler::launch()` streams tensors to device memory with `to_device()` and
applies the compiled kernels in sequence. Activations such as `relu` and
`sigmoid` are dispatched on the GPU while other operations currently fall back to
CPU implementations. Results are copied back to host memory via `to_host()` so
the cycle state can be inspected.

## Device selection

Unless explicitly overridden with `HARMONICS_CUDA_DEVICE` or
`set_cuda_device_index()`, the loader picks the GPU with the largest amount of
memory. If no compatible device is present the runtime transparently falls back
to the CPU path.
