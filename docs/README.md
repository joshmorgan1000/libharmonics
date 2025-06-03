# Harmonics Documentation

This directory contains all of the project's reference material.
Below is a quick index of the available guides.
## Quick Start

A minimal training loop:

```cpp
const char* src = R"(
harmonic tiny {
  producer data;
  producer label 1/1 data;
  layer hidden;
  cycle {
    data -(relu)-> hidden;
    hidden <-(cross_entropy)- label;
  }
}
)";
harmonics::Parser p{src};
auto g = harmonics::build_graph(p.parse_declarations());

harmonics::FitOptions opt;
opt.learning_rate = 0.001f;

auto state = g.fit(5, harmonics::make_auto_policy(), opt);

// Run inference with the trained weights
auto out = g.inference({});
```
## Core Guides

- [Architecture.md](Architecture.md) – overview of the system design.
- [Philosophy.md](Philosophy.md) – guiding principles of the DSL.
- [Datasets.md](Datasets.md) – dataset helpers and conversion tools.
- [Training.md](Training.md) – training flows and optimisers.
- [Introspection.md](Introspection.md) – APIs for inspecting graphs at runtime.

## Data & I/O

- [DataTransports.md](DataTransports.md) – streaming tensors between processes.
- [Datasets.md](Datasets.md) – dataset loaders and caching utilities.

## Model Import & Export

- [ModelImportExport.md](ModelImportExport.md) – import weights and export graphs as ONNX models.

## Runtime & Scheduling

- [Runtime.md](Runtime.md) – distributed scheduler, parameter server and checkpointing.
- [ConstantSlab.md](ConstantSlab.md) – fixed layout for sensor and appendage data.
- [ClusterDeployment.md](ClusterDeployment.md) – running partitions across multiple machines.
- [Accelerators.md](Accelerators.md) – CUDA, Vulkan and FPGA back ends.
- [CudaPipeline.md](CudaPipeline.md) – compute pipeline for the CUDA backend.
- [Runtime.md#parameter-server-example](Runtime.md#parameter-server-example) – minimal gRPC parameter server example.

## Plugins & Tools

- [Plugins.md](Plugins.md) – plugin system, packaging and graph utilities.
- [PluginPackaging.md](PluginPackaging.md) – packaging plugins with `plugin_packager`.
- [ShaderBuild.md](ShaderBuild.md) – compiling GLSL shaders into the runtime.
- [RustFFI.md](RustFFI.md) – using Harmonics from Rust.
- [JavaScriptRuntime.md](JavaScriptRuntime.md) – using Harmonics from Node.js.
- [PythonBindings.md](PythonBindings.md) – using Harmonics from Python.

## Quantum

- [Quantum.md](Quantum.md) – simulator and hardware interface.

The guides are grouped by topic so related functionality is documented together. Begin with the core guides to understand the overall design and how graphs are created and trained. The data transport and dataset sections cover the various input pipelines and caching helpers. Runtime topics explore distributed execution and accelerator support. Plugin development and graph utilities are documented together so custom functionality can be added with minimal boilerplate. The quantum documentation collects the simulator and hardware notes in one place.

Each document lives in this directory so cross-references work offline. The markdown files are plain text and can be read directly from the repository or rendered by any viewer. Most documents include small code snippets that illustrate the API calls in context. Larger examples reside under the `examples` directory and can be built with the helper script mentioned in the root `README.md`.

If you are new to Harmonics start with `Philosophy.md` and `Architecture.md` to learn why the DSL was designed around streams. Then explore `Training.md` to see how graphs are executed and optimised. The remaining documents can be consulted as needed. The `README.md` at the repository root contains additional build instructions and explains how to run the example programs.


### Document overview

- **Architecture.md** – details the components that make up the runtime and how graphs are lowered to executable kernels.
- **Philosophy.md** – explains the design principles that shaped the DSL.
- **Datasets.md** – describes dataset loaders, caching and conversion helpers in depth.
- **DataTransports.md** – lists the available producers and consumers for files and sockets.
- **ModelImportExport.md** – shows how to load pretrained weights and write ONNX files.
- **Runtime.md** – covers partitioning graphs, running them across processes and restoring checkpoints.
- **ConstantSlab.md** – documents the fixed memory layout for variable slots.
- **Accelerators.md** – documents the CUDA, Vulkan and FPGA backends together with the shader cache.
- **Plugins.md** – explains the plugin system, hot reloading and graph inspection tools.
- **Training.md** – walks through the training API and options, including the WebSocket visualizer.
- **Quantum.md** – demonstrates the experimental quantum mapping helpers and simulator.
- **RustFFI.md** – contains the C bindings used by the Rust interface.
- **JavaScriptRuntime.md** – shows how to build and use the JavaScript bindings.
- **Introspection.md** – outlines runtime graph editing and inspection utilities.

Each topic can be read independently, but they reference each other where features interact. Scanning the list above should reveal where to look when a specific question arises. As the project grows these documents will be updated and expanded with new examples.


### Contributing documentation

The documentation files are written in Markdown so they can easily be edited in any text editor. When contributing new sections, keep related material in the same file wherever possible. This helps reduce the number of pages and keeps topics together. New documents should only be created when the content would exceed a few hundred lines and cannot logically fit elsewhere. Run `scripts/format.sh` and `scripts/run-tests-all.sh` before submitting changes.

### Keeping the docs concise

Short documents have been merged into larger pages to keep the structure easy to browse. Each section contains anchors so that specific topics can be linked directly. Feel free to extend an existing page if you notice an area that could use more explanation. The index above should remain compact and point to the main entry points for each aspect of the library.

When reading the documentation in a browser, the built in table of contents can help navigate long pages. All headers use Markdown `##` and `###` markers so viewers can generate a sidebar automatically. Printed or offline copies should retain these headings for reference.


The focus on a small number of comprehensive pages keeps navigation straightforward while still covering advanced features. Refer back to this index whenever you need an overview of the available material.

