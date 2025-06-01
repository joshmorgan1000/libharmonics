# Plugins and Tools
# Plugin System

Harmonics allows custom layers, activations and loss functions to be loaded at runtime.
This is accomplished through the plugin registry found in `include/harmonics/plugin.hpp`.
A plugin is a shared library that exposes a single C-style entry point:

```cpp
extern "C" void harmonics_register(harmonics::FunctionRegistry& reg);
```

Inside this function the plugin receives a `FunctionRegistry` object and registers
any new implementations it wishes to make available. Each type of object is identified
by a string name:

```cpp
reg.register_activation("my_act", std::make_shared<MyActivation>());
reg.register_loss("my_loss", std::make_shared<MyLoss>());
reg.register_layer("my_layer", std::make_shared<MyLayer>());
```

The runtime looks up these names when parsing a graph or when loading weights.
Plugins can therefore override existing functions or supply completely new ones.

## Loading plugins

Use `load_plugin(path)` to load a specific shared library. To load multiple plugins
from a directory tree call `load_plugins_from_path(dir)`. This helper scans all
files named `*.so` (or `*.dll` on Windows) under `dir` and invokes the register
function in each of them.

For convenience the `HARMONICS_PLUGIN_PATH` environment variable can hold a colon
separated list of directories to search. If the variable is set, `load_plugins_from_path()`
will automatically scan the given paths.

```cpp
harmonics::load_plugins_from_path("./plugins");
```

## Discovery mechanism

`load_plugin` simply opens the shared library at the given path and invokes the
`harmonics_register` entry point. For larger setups
`load_plugins_from_path` accepts a colon separated list of directories. Each
directory is scanned recursively for files ending in `.so` (or `.dll` on
Windows). Any libraries found are loaded and missing registration symbols are
ignored so that partially populated trees do not abort start‑up. If the function
is called with an empty string it checks the `HARMONICS_PLUGIN_PATH` environment
variable instead.

The helper returns handles for every successfully loaded plugin. These handles
can later be passed to `unload_plugin` once the plugin is no longer needed.

Asynchronous variants (`load_plugin_async`, `load_plugins_from_path_async` and
`unload_plugin_async`) return `std::future` objects so multiple plugins can be
loaded or unloaded in parallel.

## Unloading plugins

Plugins remain loaded for the lifetime of the process unless explicitly
unloaded. Call `unload_plugin(handle)` when the functionality provided by a
library is no longer required. The registered functions stay in the global
`FunctionRegistry`, so any graphs created after unloading will still resolve the
identifiers. Ensure that no code from the plugin is executing when the library
is closed.

## Runtime registry

Plugins register their implementations with the global `FunctionRegistry`. This
singleton maps string identifiers to activation, loss and layer objects. The
registry is passed to the plugin's `harmonics_register` function and offers the
methods `register_activation`, `register_loss` and `register_layer`. Runtime
lookups use `getActivation`, `getLoss` and `getLayer`.

Registrations override existing entries by default. Starting with version 1.1
all registry operations are protected by a mutex so that multiple plugins can be
loaded concurrently.

## Example project

A simple demonstration plugin lives under `examples/plugin_example`. The project
builds a `sample_plugin` library implementing a custom activation and then loads it
via the `plugin_example` executable. The example shows how to compile and link a
plugin using CMake and how to call `load_plugin()` at runtime.

## Packaging and discovery

Plugins are usually distributed as shared libraries inside a small directory
containing an optional `plugin.json` file. The layout and helper commands are
described in [PluginPackaging.md](PluginPackaging.md). A packed plugin can be
archived with `plugin_packager package` and installed with
`plugin_packager install`. At runtime set `HARMONICS_PLUGIN_PATH` to the root
directory or pass the path to `load_plugins_from_path()` to load all plugins
recursively.

## Hot reloading

Plugins can be updated without restarting the process. Call `reload_plugin(path)`
to unload the existing library at `path` and immediately load the new file. Any
functions previously registered remain available while the updated plugin can
register additional implementations or replace the old ones.

Helpers `reload_plugins_in_directory` and `reload_plugins_from_path` mirror the
load variants and make it easy to refresh entire plugin trees. Asynchronous
versions of all reload APIs are also provided.

## Version metadata

Each plugin may optionally export a `harmonics_plugin_version()` function that
returns an integer identifying the plugin version. When present the loader
records this value in the returned `PluginHandle`. Applications can inspect the
`version` field to ensure compatibility before invoking the plugin's functions.

All load and unload operations are protected by a mutex so plugins can be hot
reloaded safely even when multiple threads are active.

# Plugin System Tutorial

This tutorial walks through creating and loading a simple plugin in Harmonics. Plugins allow you to add custom layers, activations and losses without modifying the core library.

## 1. Create the plugin source

A plugin is a shared library that defines a `harmonics_register` function. Inside this function you register your extensions with the provided `FunctionRegistry`:

```cpp
#include <harmonics/function_registry.hpp>
#include <memory>

struct ExampleActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override {
        return x; // identity for demonstration
    }
};

extern "C" void harmonics_register(harmonics::FunctionRegistry& reg) {
    reg.register_activation("example_act", std::make_shared<ExampleActivation>());
}
```

Save the file as `sample_plugin.cpp` and place it in a directory of your choice (the examples directory contains the same file under `examples/plugin_example`).

## 2. Build the plugin

Use CMake to compile the file as a shared library. A minimal `CMakeLists.txt` looks like:

```cmake
cmake_minimum_required(VERSION 3.16)
project(sample_plugin)

add_library(sample_plugin SHARED sample_plugin.cpp)
```

Configure and build the project:

```bash
cmake -S . -B build
cmake --build build
```

This produces `libsample_plugin.so` (or `sample_plugin.dll` on Windows) in the build directory.

## 3. Load the plugin

At runtime call `load_plugin()` or `load_plugins_from_path()` to register the new functionality:

```cpp
#include <harmonics/plugin.hpp>
#include <iostream>

int main() {
    auto handles = harmonics::load_plugins_from_path("./build");
    std::cout << "Loaded " << handles.size() << " plugin(s)" << std::endl;
    const auto& act = harmonics::getActivation("example_act");
    harmonics::HTensor t{};
    act(t);
    harmonics::unload_plugins(handles);
}
```

Running the above program loads the plugin, invokes the custom activation and then unloads the shared library.

## 4. Automatic discovery

`load_plugins_from_path()` accepts a colon separated list of directories and scans them recursively for shared libraries. If called with an empty string it falls back to the `HARMONICS_PLUGIN_PATH` environment variable:

```bash
export HARMONICS_PLUGIN_PATH=/path/to/plugins:/other/path
my_program
```

Any libraries containing a `harmonics_register` function are loaded automatically.

## 5. Next steps

See [PluginSystem.md](PluginSystem.md) for a detailed reference of all helper functions including asynchronous loading and unloading.

# Plugin Packaging Guidelines

Plugins are distributed as shared libraries together with a small descriptor file. The loader scans directories recursively, so packaging all files in a single folder keeps installation simple.

## Directory layout

```
my_plugin/
  sample_plugin.so
  plugin.json
```

The optional `plugin.json` contains metadata such as the plugin name, version and any library dependencies. Applications may read this information before loading the library.
## Plugin Manifest

A plugin directory can include a `plugin.json` file with metadata that describes the library. Only the `name` and `version` fields are expected, but additional keys may be added by projects.

```json
{
  "name": "my_plugin",
  "version": "1.0"
}
```


Package the folder as a zip or tar archive for distribution (using zstd compression for tar):

```bash
tar --zstd -cf my_plugin.tar.zst my_plugin
# or
zip -r my_plugin.zip my_plugin
```

At runtime set the `HARMONICS_PLUGIN_PATH` environment variable to the location of the unpacked directory. `load_plugins_from_path()` will discover the library automatically.

## Packaging CLI

The `plugin_packager` tool automates creating and installing plugin archives. It uses the system `tar` command internally so no additional dependencies are required.

```bash
plugin_packager package my_plugin my_plugin.tar.zst
plugin_packager install my_plugin.tar.zst ~/.local/harmonics/plugins
```

The first command bundles the contents of `my_plugin` into the archive `my_plugin.tar.zst`. The second extracts the archive into the target directory. Set `HARMONICS_PLUGIN_PATH` to the folder containing the unpacked plugins so the runtime can load them at startup.

# Graph Diff and Merge

Changes between two graphs can be analysed and applied at runtime. The header `include/harmonics/graph_diff.hpp` provides helpers to compute a diff and merge graphs.

## API Overview

```cpp
auto diff = harmonics::diff_graphs(before, after);
harmonics::apply_diff(graph, diff);
auto merged = harmonics::merge_graphs(base, update);
```

`GraphDiff` lists added and removed layers as well as new or deleted flows. `apply_diff` modifies a graph in place while `merge_graphs` returns a new graph containing the updates.

## Command line tool

The `graph_diff` utility wraps these functions for serialized graphs:

```bash
graph_diff diff base.hgr update.hgr
graph_diff merge base.hgr update.hgr -o merged.hgr
```

`diff` prints the required edits to standard output. `merge` applies the changes from the second graph and writes the result to the file given by `-o` (or overwrites the first file when omitted).

# Graph Debugger

`graph_debugger` provides an interactive way to step through a compiled graph. It executes one arrow at a time and prints the source and destination nodes so you can observe the data flow.

## Usage

```bash
graph_debugger <graph> [--bits n]
```

The argument `<graph>` is a `.hgr` file produced by `harmonics_cli --compile`. The optional `--bits` flag selects the execution precision with `32` as the default.

Running the tool loads the graph, binds zero producers for each input and performs a single forward pass. For every arrow the program prints a line like:

```
source -> destination [shape]
```

Backward arrows are printed with `~>` instead of `->`. If the arrow was generated by a specific layer function, the name appears in parentheses. After printing, the debugger waits for you to press Enter before continuing to the next arrow.

## Example

```bash
graph_debugger model.hgr --bits 16
```

This prints each data flow and pauses, allowing you to verify that the graph executes in the expected order and with the correct tensor shapes.

# Graph Info CLI

The `graph_info` tool prints a summary of a serialized graph. It lists all
producers, layers and consumers followed by each data flow.

## Usage

```bash
graph_info <graph>
```

The output can be used to quickly inspect the structure of a compiled model.
Backward flows are printed with `~>` and layer function names appear in
parentheses.

