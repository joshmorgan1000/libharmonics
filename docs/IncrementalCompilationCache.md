# Incremental Compilation Cache

The incremental compilation cache stores compiled GPU shaders so that identical graphs do not trigger a full compilation step on every run. Shaders are cached at two levels and can optionally be persisted on disk.

## Graph level cache

`compile_cycle_kernels()` hashes the structure of a cycle and keeps the resulting descriptors in a static map. Reusing the same graph with compatible precision policies simply returns the cached descriptors.

## Shader level cache

Within a compile pass each arrow resolves to a shader name. Before invoking the compiler the runtime looks up the name in `shader_compile_cache()`. If present the cached SPIR-V blob is reused; otherwise the shader is compiled and inserted into the map.

### Persistent cache

When Vulkan support is available the compiled SPIR-V can be written to disk. The directory defaults to `shader_cache/` and can be overridden with the `HARMONICS_SHADER_CACHE` environment variable. Cached files are keyed by a digest of the shader name and reloaded on the next run.

Clearing the in-memory cache or removing the files forces a fresh compile.

## Usage example

```cpp
// Force a clean pass
harmonics::shader_compile_cache().clear();
auto kernels = harmonics::compile_cycle_kernels(graph, policy);
```

The cache greatly reduces startup time for large graphs and allows recompilation to be skipped when kernels are unchanged.
