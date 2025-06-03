# Python Bindings

Harmonics can be used from Python through a small extension module. The bindings
mirror the JavaScript helper functions and expose a minimal API for compiling
DSL graphs and running cycles.

## Building

Enable the module by configuring CMake with `-DHARMONICS_HAS_PY=1` and make sure
Python headers are available:

```bash
cmake -S . -B build -DHARMONICS_HAS_PY=1
cmake --build build -j $(nproc)
```

The resulting module is named `harmonics_py` (for example `harmonics_py.so` on
Unix-like systems). Import it like a regular Python extension.

## Available functions

- `compile_graph(source)` – parse a DSL string and return a graph handle.
- `destroy_graph(graph)` – release the graph handle.
- `create_runtime(graph)` – create a `CycleRuntime` for the graph.
- `destroy_runtime(runtime)` – free the runtime instance.
- `run_cycle(runtime)` – execute one cycle of the graph.

## Example usage

```python
import harmonics_py as hmx

src = "harmonic demo { producer x; consumer y; cycle { x -> y; } }"
g = hmx.compile_graph(src)
rt = hmx.create_runtime(g)
hmx.run_cycle(rt)
hmx.destroy_runtime(rt)
hmx.destroy_graph(g)
```

These bindings provide just enough functionality for quick experimentation.
For larger applications consider using the C API defined in
`include/harmonics/rust_ffi.hpp` together with `ctypes` or `cffi`.
