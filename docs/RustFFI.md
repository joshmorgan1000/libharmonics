# Rust FFI Bindings

The header `include/harmonics/rust_ffi.hpp` exposes a small C API so that
Harmonics can be controlled from Rust or other languages. The library target
`harmonics_ffi` compiles these bindings.

## Building the library

Enable the bindings by building the `harmonics_ffi` target with CMake:

```bash
cmake -S . -B build && cmake --build build --target harmonics_ffi
```

This produces a static library that can be linked from your Rust crate.

## Available functions

The API allows a graph to be created, executed and trained. It also exposes a
production-ready dataset loader so inputs can be streamed from Rust:

```cpp
HarmonicGraph* harmonics_parse_graph(const char* src);
void harmonics_destroy_graph(HarmonicGraph* g);
CycleRuntime* harmonics_create_runtime(const HarmonicGraph* g);
void harmonics_destroy_runtime(CycleRuntime* rt);
void harmonics_forward(CycleRuntime* rt);
Producer* harmonics_create_csv_producer(const char* path);
void harmonics_destroy_producer(Producer* p);
void harmonics_bind_producer(HarmonicGraph* g, const char* name, Producer* p);
void harmonics_fit(HarmonicGraph* g, std::size_t epochs);
```

## Example usage in Rust

Declare the functions in Rust using `extern "C"` and call them through a
`build.rs` script or `bindgen` generated bindings. The snippet below lists the
complete API, including dataset helpers and the training function:

```rust
extern "C" {
    fn harmonics_parse_graph(src: *const c_char) -> *mut HarmonicGraph;
    fn harmonics_destroy_graph(g: *mut HarmonicGraph);
    fn harmonics_create_runtime(g: *const HarmonicGraph) -> *mut CycleRuntime;
    fn harmonics_destroy_runtime(rt: *mut CycleRuntime);
    fn harmonics_forward(rt: *mut CycleRuntime);
    fn harmonics_create_csv_producer(path: *const c_char) -> *mut Producer;
    fn harmonics_destroy_producer(p: *mut Producer);
    fn harmonics_bind_producer(g: *mut HarmonicGraph, name: *const c_char, p: *mut Producer);
    fn harmonics_fit(g: *mut HarmonicGraph, epochs: usize);
}
let GRAPH_SRC = CString::new("producer d{1}; consumer c; cycle{ d -> c; }").unwrap();
let DATA_PATH = CString::new("train.csv").unwrap();
let INPUT_NAME = CString::new("d").unwrap();

unsafe {
    let g = harmonics_parse_graph(GRAPH_SRC.as_ptr());
    let data = harmonics_create_csv_producer(DATA_PATH.as_ptr());
    harmonics_bind_producer(g, INPUT_NAME.as_ptr(), data);
    harmonics_fit(g, 5);
    harmonics_destroy_producer(data);
    harmonics_destroy_graph(g);
}
```

The bindings provide only a thin wrapper so resource management remains manual.

## Distributed scheduler bindings

The FFI can build a `DistributedScheduler` for graphs that have been partitioned.
Use `harmonics_auto_partition()` to split a graph across multiple backends and
then construct the scheduler with `harmonics_create_distributed_scheduler()`.

```cpp
HarmonicGraph** harmonics_auto_partition(const HarmonicGraph* g,
                                         const harmonics_backend_t* backends,
                                         std::size_t count);
void harmonics_destroy_partitions(HarmonicGraph** parts, std::size_t count);
DistributedScheduler* harmonics_create_distributed_scheduler(HarmonicGraph** parts,
                                                             std::size_t count,
                                                             const harmonics_backend_t* backends,
                                                             bool secure);
void harmonics_destroy_distributed_scheduler(DistributedScheduler* sched);
void harmonics_scheduler_bind_producer(DistributedScheduler* sched, std::size_t part,
                                       const char* name, Producer* p);
CycleRuntime* harmonics_scheduler_runtime(DistributedScheduler* sched, std::size_t part);
void harmonics_scheduler_step(DistributedScheduler* sched);
void harmonics_scheduler_fit(DistributedScheduler* sched, std::size_t epochs);
```

Each call to `harmonics_scheduler_step` processes one iteration across all
partitions while `harmonics_scheduler_fit` trains the entire graph for the given
number of epochs.

## WebAssembly helpers

When built with WASM support the following functions report runtime
capabilities:

```cpp
bool harmonics_wasm_available();
bool harmonics_wasm_runtime_available();
bool harmonics_wasm_simd_available();
```

`harmonics_wasm_available` signals that the library includes the WebAssembly
backend. `harmonics_wasm_runtime_available` checks that the runtime can be
initialised on the current system. `harmonics_wasm_simd_available` indicates
SIMD support for accelerated kernels.
