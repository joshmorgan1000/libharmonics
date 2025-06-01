# Shader Build Process

This guide explains how Harmonics compiles the GLSL compute shaders shipped with the project.

## Requirements

- `glslangValidator` to translate `.comp` files into SPIR-V bytecode.
- `zstd` to compress the compiled shaders.

Both tools must be available on the host system.

## Building the shaders

The script `scripts/compile_shaders.sh` scans the `shaders/` directory for `.comp` files. Each shader is compiled with `glslangValidator` and the resulting SPIR-V is compressed with `zstd`. The compressed binaries are embedded into `include/gpu/Shaders.hpp` as byte arrays.

The CMake file defines the `generate_shaders` target which runs this script automatically when building the project:

```bash
cmake --build build --target generate_shaders
```

Compilation logs are written to `logs/shader_compile.log`. If a shader fails to compile, check this file for diagnostic output.

## Rebuilding

Run the script again or remove the `.spv` and `.spv.zst` files under `shaders/` to force a clean rebuild. The header must be regenerated whenever the shader sources change.

## Testing

`scripts/gpu_test_shaders.sh` compiles and executes a small suite of shaders to verify that the toolchain works. The helper creates logs under `logs/` and compares the output of each shader against known values.

## Shader compilation pipeline

At runtime `compile_cycle_kernels()` translates the arrows of a `HarmonicGraph` into GLSL source and invokes `glslangValidator` to produce SPIR-V bytecode. Compilation happens lazily the first time a graph is executed. The pipeline looks up each shader name in `shader_compile_cache()` before invoking the external compiler so that identical kernels are only built once.

1. A unique name is generated for each arrow based on its operation and precision.
2. If the name exists in the in-memory cache the stored SPIR-V blob is reused.
3. Otherwise the shader is compiled, compressed with `zstd` when supported and inserted into the cache. When Vulkan is available the bytecode can also be persisted to the directory specified by `HARMONICS_SHADER_CACHE`.
4. The SPIR-V is then turned into a GPU compute pipeline and stored in the graph-level cache so subsequent executions skip the entire compilation step.

See `Accelerators.md` and `IncrementalCompilationCache.md` for a complete overview of the caching strategy.
