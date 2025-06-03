<p align="center"><img src="docs/images/harmonics.png" alt="Harmonics logo" width="800"></p>

Harmonics is a header-only C++17 library implementing a concise domain specific
language (DSL) for describing neural network flows.  It embraces the
principle **"Write once, resonate anywhere"** so that the same graph can run
unchanged on CPUs, GPUs or future accelerators.  See
[`docs/Philosophy.md`](docs/Philosophy.md) and
[`docs/Architecture.md`](docs/Architecture.md) for the guiding principles and
the design in depth.

All documentation has been consolidated in the [docs](docs/README.md) directory.

One of the long-term aims of the project is to let a model be **partitioned by layer** (or even by sub‑sections of layers) so that training or inference can run across a distributed system.  The runtime will expose the primitives needed to split a `HarmonicGraph`; the actual message transport between distributed pieces is left to whichever module embeds the engine.

The formal grammar of the DSL is provided in `grammar/harp.peg`.

## Vision

Harmonics views a model as a streaming flow that adapts to whichever substrate
is available. Producers and consumers decouple data logistics from layer logic
so the same graph can span multiple machines. Precision and width are negotiated
at runtime, enabling execution on CPUs, GPUs or future accelerators.
Training and inference share the same DSL while transports remain pluggable.
As summarised in the [philosophy](docs/Philosophy.md): **Write once, resonate anywhere.**

## Philosophy & Architecture

The language treats every value as a negotiable stream.  Precision,
width and even the execution substrate are resolved just before a graph
is materialised.  This flexibility stems from a few core goals:

- **Declarative topology** – models read as data flows rather than imperative code.
- **Producer/Consumer decoupling** – the same graph runs on files, sockets or custom stores.
- **Substrate agnosticism** – CPU, GPU or FPGA targets require no DSL changes.
- **Distributed graph partitioning** – layers can be split across processes.
- **Self‑negotiating precision & width** – size and bit‑width resolve at runtime.
- **Pluggable ops** – activations and losses are registered objects, not keywords.
- **Clear namespace** – runtime types are prefixed with `H` to avoid clashes.
- **Zero Python dependency** – the reference interpreter is modern C++.
 - **Quantum circuit support** – includes a simulator and a pluggable hardware backend.
   Build with `-DHARMONICS_HAS_QUANTUM_HW=1` and set `HARMONICS_ENABLE_QUANTUM_HW=1`
   so `execute_on_hardware()` loads `libquantum_hw.so` (or the path given in
   `HARMONICS_QUANTUM_HW_LIB`). Use `quantum_hardware_runtime_available()` to
   check whether a device library was found. See [docs/Quantum.md](docs/Quantum.md)
   for more details.

### Build prerequisites

- CMake **3.16** or newer
- A C++17 compatible compiler
- [Optional] **Protobuf 3.21+** if you need ONNX weight import. Older versions
  lack `google/protobuf/port_def.inc` which results in a build failure.

## Building

The project uses CMake.  A convenience script `scripts/run-tests.sh` configures the build directory, builds all targets and runs the test suite.  It accepts an optional build type (Debug or Release).  For example:

```bash
# Build in Release mode and run the tests
./scripts/run-tests.sh Release
```
To verify that the suite behaves identically across build types, run `scripts/run-tests-all.sh` which sequentially builds and tests both Release and Debug configurations.


To build without running the tests, invoke CMake manually:

```bash
cmake -S . -B build
cmake --build build -j $(nproc)
```

Unity builds are enabled by default to reduce compilation time. If you
encounter issues or prefer traditional builds, disable this behaviour by
passing `-DENABLE_UNITY_BUILD=OFF` when invoking CMake.

Link time optimisation (LTO) and profile guided optimisation (PGO) are also
available. Enable LTO with `-DENABLE_LTO=ON`. PGO is controlled through the
`PGO_PHASE` option which can be `GENERATE` or `USE`:

```bash
# Instrument the build to generate profiles
cmake -S . -B build -DPGO_PHASE=GENERATE
# After running your workload, rebuild using the collected data
cmake -S . -B build -DPGO_PHASE=USE
```

The repository also provides a small benchmark suite. Run `scripts/run-benchmarks.sh`
to build and execute the benchmarks. Results are written under
`logs/benchmarks`. If you already have a build directory you can instead run

```bash
cmake --build build --target benchmarks
```

which invokes the same script from CMake.
A separate script `scripts/check-determinism.sh` builds the project with both GCC and Clang and runs the `int8_determinism_test` to ensure consistent results across compilers.

### Windows

On Windows the project can be built from a Developer Command Prompt or Git Bash.
Invoke the same helper script:

```bash
bash ./scripts/run-tests.sh Release
```

Alternatively configure Visual Studio projects directly:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Running the examples

After building, the example binaries are placed in the build directory.  For instance, to run the basic example:

```bash
./build/basic_example
```

The MNIST example can be run similarly as `./build/mnist_example`.

Additional programs demonstrate advanced features:

- `plugin_example` loads a custom layer from `examples/layer_plugin`.
- `training_dashboard` visualises progress in the browser when used with `WebSocketTrainingVisualizer`. Run the Node.js server in
  `examples/training_dashboard` with `npm install ws && node server.js` and open
  `http://localhost:8080` to view an interactive chart of the training metrics.
  The page provides pause/resume controls and shows the gradient norm together
  with the loss value and learning rate used for each step.
- `training_visualizer_example` runs a minimal training loop and streams progress
  metrics to the dashboard using `WebSocketTrainingVisualizer`.
  - `quantum_backend_demo` shows the quantum circuit simulator in action.
  - `quantum_hardware_demo` demonstrates running the same circuit on a
    device library when hardware support is enabled. `quantum_hardware_large`
    executes a bigger test circuit using the same path.

## Dataset utilities

See [docs/Datasets.md](docs/Datasets.md) for helpers that load common formats such as CSV and IDX.
The document also covers streaming variants, memory-mapped loaders and asynchronous HTTP sources for
large or remote datasets.

### Augmentation helpers

The header `augmentation.hpp` implements a range of convenience operations that can
be chained together using `make_augmentation_pipeline`:

* `flip_horizontal` / `flip_vertical`
* `rotate90` and generic `rotate`
* `random_crop` and `random_rotate`
* `scale`
* `colour_jitter`
* `add_noise`

These functions return transformed copies of the input `HTensor` and can be wrapped
around dataset producers to build on-the-fly augmentation pipelines.

## Precision policies

Use a precision policy to constrain activations to a fixed bit width. This can
be useful when targeting hardware with limited integer precision:

```cpp
auto bits8 = harmonics::make_max_bits_policy(8);
auto final = g.fit(5, bits8, opt);
auto out = g.inference({}, bits8);

// ultra low precision examples
auto bits4 = harmonics::make_max_bits_policy(4);
auto bits3 = harmonics::make_max_bits_policy(3);
```

The runtime dispatches specialised kernels when the precision drops below eight
bits. Using `make_max_bits_policy(3)` enables the `int3_matmul` shader so matrix
multiplications operate on packed 3‑bit integers transparently.

## Activation functions

The runtime registers a library of common activation functions when
`register_builtin_shaders()` is called. The following IDs are available:

- `relu`
- `sigmoid`
- `softmax`
- `gelu`
- `selu`
- `prelu`

They can be used directly in the DSL or accessed programmatically through
`getActivation(name)`.

## Plugins

Custom activation, loss and layer implementations can be loaded at runtime from
shared libraries. A plugin must expose an entry point named
`harmonics_register`:

```cpp
extern "C" void harmonics_register(harmonics::FunctionRegistry& reg) {
    reg.register_activation("my_act", std::make_shared<MyActivation>());
    reg.register_loss("my_loss", std::make_shared<MyLoss>());
    reg.register_layer("my_layer", std::make_shared<MyLayer>());
}
```

Plugins can be loaded explicitly via `load_plugin(path)` or automatically from a
search path using `load_plugins_from_path()`. Asynchronous variants of these
functions return futures so that multiple plugins can be processed in parallel.
The function scans directories in the colon separated list passed in or the
`HARMONICS_PLUGIN_PATH` environment variable. An example plugin and loader
program can be found under `examples/plugin_example`. Build the `sample_plugin`
target and run `plugin_example`, specifying a plugin path or setting
`HARMONICS_PLUGIN_PATH`.

Plugins remain loaded until `unload_plugin(handle)` (or `unload_plugin_async`)
is called on the returned handle.

Additional details on building and loading plugins can be found in
[docs/Plugins.md](docs/Plugins.md). Packaging plugins is documented in
[docs/PluginPackaging.md](docs/PluginPackaging.md) and the shader
build process is described in [docs/ShaderBuild.md](docs/ShaderBuild.md).

## Layer behaviour configuration

The built-in convolution, normalisation and attention layers expose runtime
parameters that can be tweaked through helper functions:

```cpp
harmonics::set_convolution_kernel(5);
harmonics::set_norm_epsilon(1e-3f);
harmonics::set_attention_temperature(0.5f);
harmonics::register_builtin_layers();
```

Calling `register_builtin_layers()` after adjusting the values re-registers the
layers with the new behaviour.

## Streaming I/O

Producers and consumers for reading and writing serialized tensors are
documented in `docs/DataTransports.md`. They allow graphs to interact with
files, standard streams, sockets or in-memory message buses.

## Command line tools

Harmonics provides a few command line utilities for working with graphs and weights.

### `harmonics_cli`

Compile a DSL file into a graph:
```bash
harmonics_cli --compile model.harp -o graph.hgr
```
Run a compiled graph:
```bash
harmonics_cli --run graph.hgr --secure --bits 16
```

### `graph_cli`

Inspect or modify an existing graph:
```bash
graph_cli info graph.hgr --bits 16
```
Add a layer to a graph:
```bash
graph_cli add-layer new_layer graph.hgr -o out.hgr
```
Remove a layer from a graph:
```bash
graph_cli remove-layer old_layer graph.hgr -o out.hgr
```
Add a data flow between layers:
```bash
graph_cli add-flow src_layer dst_layer graph.hgr -o out.hgr
```
Remove a data flow:
```bash
graph_cli remove-flow src_layer dst_layer graph.hgr -o out.hgr
```

Generate a Graphviz representation of a graph and render it as an image:

```bash
graph_cli dot graph.hgr -o graph.dot
dot -Tpng graph.dot -o graph.png
```

Apply multiple edits from a script with undo/redo support:

```bash
graph_cli batch graph.hgr edits.txt -o out.hgr
```
Each line of `edits.txt` contains a single command such as `add-layer name`,
`remove-layer name`, `add-flow a b`, `remove-flow a b`, `undo` or `redo`.

Start an interactive editing session with undo/redo support:

```bash
graph_cli interactive graph.hgr -o out.hgr
graph> run
graph> info
graph> add-layer extra
graph> add-flow input extra
graph> remove-layer extra
graph> undo
graph> save
graph> quit
```

### `graph_diff`

Compute or apply differences between two graphs:

```bash
graph_diff diff base.hgr update.hgr
graph_diff merge base.hgr update.hgr -o merged.hgr
```

### `graph_debugger`

Step through a graph one arrow at a time and print the source and destination of
each flow:

```bash
graph_debugger graph.hgr --bits 16
```

### `graph_info`

Print a summary of a graph:

```bash
graph_info graph.hgr
```

### `dataset_convert`

Convert CSV or IDX datasets into the minimal HDF5 format:

```bash
dataset_convert --csv train.csv -o train.hdf5
dataset_convert --idx images.idx -o images.hdf5
```

### `model_convert`

Convert framework weights to the `HNWT` format:
```bash
model_convert --onnx model.onnx -o weights.hnwt
```

### `dataset_schema_cli`

Validate the expected shape and data type of tensors in an HDF5 dataset:

```bash
dataset_schema_cli images.hdf5 f32 784
```

### `plugin_packager`

Bundle and install plugins as compressed archives:

```bash
plugin_packager package my_plugin my_plugin.tar.zst
plugin_packager install my_plugin.tar.zst ~/.local/harmonics/plugins
```

## Introspection utilities

Utilities for inspecting a graph at runtime are documented in
`docs/Introspection.md`. They expose functions such as `get_layer_info`
and `layer_weights` which are used by the interactive shell example.

## Accelerator support

GPU acceleration is available through the Vulkan or CUDA backends. Enable a backend by configuring CMake with
`-DHARMONICS_HAS_VULKAN=1` or `-DHARMONICS_HAS_CUDA=1`.
When GPU support is enabled but no compatible device is detected, the runtime
falls back to the CPU path. A convenience wrapper under `include/gpu/Wrapper.h`
manages buffers and pipeline setup so a compute shader can be executed by
supplying input producers and retrieving the output. The wrapper allocates a
ring of two buffer pairs by default so uploads and dispatches overlap on modern
GPUs. The ring size can be increased via the constructor or `setRingSize()`
to keep more dispatches in flight. Select a specific Vulkan device by setting
`HARMONICS_VULKAN_DEVICE` or by calling `harmonics::set_vulkan_device_index()`
before constructing a `CycleRuntime`. For CUDA the environment variable
`HARMONICS_CUDA_DEVICE` chooses the device. The index can also be specified
through the `DeploymentDescriptor` using the `gpu_device_index` field.

### Multi-threaded CPU runtime

The CPU implementation can execute arrows in parallel. Call
`CycleRuntime::enable_multi_threading()` before invoking `forward()` to run
flows concurrently across multiple threads.

An experimental FPGA path is provided as a design stub. Build with
`-DHARMONICS_HAS_OPENCL=1` to include the OpenCL adapter. At runtime set
`HARMONICS_ENABLE_OPENCL=1` so the loader initialises the OpenCL subsystem.
When multiple platforms are installed the environment variable
`HARMONICS_OPENCL_PLATFORM` selects the platform index and `HARMONICS_OPENCL_DEVICE`
chooses the device. The selection can also be overridden through
`DeploymentDescriptor::fpga_device_index`. Select the backend by
setting `DeploymentDescriptor::backend` to `Backend::FPGA`. If no compatible
device is available the runtime transparently falls back to the CPU
implementation. See [docs/Accelerators.md#fpga-backend](docs/Accelerators.md#fpga-backend)
for a complete example.

WebAssembly builds are also supported through Emscripten. Compile with
`-DHARMONICS_HAS_WASM=1` to enable the backend and target browsers.
When built this way, automatic backend selection returns `Backend::WASM`
and the runtime falls back to the CPU implementation.

JavaScript bindings can also be built by enabling `-DHARMONICS_HAS_JS=1`.
The resulting `libharmonics_js` module links against V8 and provides the
`compileGraph`, `createRuntime` and `runCycle` helpers for use from Node.js or
other V8 environments.

Python bindings are available with `-DHARMONICS_HAS_PY=1`. The module is named
`harmonics_py` and exposes the same helper functions for use from Python.

## Documentation

See [docs/README.md](docs/README.md) for an overview of the available guides.
The directory contains the official documentation for model import helpers,
streaming I/O, graph introspection and more.

## Contributing

Before committing code, run `scripts/format.sh` to apply `clang-format` using the repository style.  All new functionality should come with unit tests in the `tests/` directory.

## License

Harmonics is released under the [Apache License 2.0](LICENSE).

Over 90% of this code was generated with the assistance of OpenAI Codex.
