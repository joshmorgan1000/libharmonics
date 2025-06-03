# Graph Introspection

Harmonics exposes a small set of utilities for inspecting a graph at
runtime. They are defined in `include/harmonics/introspection.hpp` and
work with any `HarmonicGraph` and `CycleRuntime`.

## Layer metadata

`get_layer_info(graph)` returns a vector of `LayerInfo` records. Each
record contains the layer name, its resolved width and whether the layer
has trainable parameters. The width is `0` when the shape is inferred
from the first batch of data.

```cpp
#include <harmonics/introspection.hpp>

harmonics::HarmonicGraph g = ...;
auto layers = harmonics::get_layer_info(g);
for (const auto& l : layers) {
    std::cout << l.name << " width=" << l.width << '\n';
}
```

## Editing graphs

`graph_edit.hpp` offers helpers like `add_layer`, `remove_layer`, `add_flow` and `remove_flow` for modifying a `HarmonicGraph` after it has been parsed. Ratios are automatically propagated. When a graph is edited after a `CycleRuntime` has been created, call `sync_runtime(runtime)` to resize the runtime state vectors.

```cpp
harmonics::add_layer(g, "extra");
harmonics::add_flow(g, "input", "extra");
```

## Accessing weights

After a graph has been executed, individual layer weights can be read or
modified via `layer_weights(runtime, name)` which returns an `HTensor`
reference. Passing a const `CycleRuntime` yields a const reference.

```cpp
harmonics::CycleRuntime rt{g};
rt.forward();

// Print the size of a layer's weight tensor
const auto& w = harmonics::layer_weights(rt, "hidden");
std::cout << w.data().size() << " bytes" << '\n';
```

## Profiling runtime memory

`profile_runtime_memory(runtime)` returns a `RuntimeMemoryStats` record summarising
the bytes reserved for tensors and weights.

```cpp
harmonics::CycleRuntime rt{g};
rt.forward();
auto stats = harmonics::profile_runtime_memory(rt);
std::cout << stats.total() << " bytes used" << '\n';
```

## Interactive shell example

The `shell_example` binary demonstrates these helpers. It reads DSL lines
from standard input and supports the commands `run`, `info`, `layers`,
`weights <name>`, `precision <bits>`, `clear` and `exit`. After running a
graph, `info` prints each layer's width and the size of its weight tensor
using the introspection API. `layers` lists all layer names, `weights`
prints the values of a layer's weight tensor and `precision` changes the
bit width used when the graph is executed.

To build and run the example:

```bash
./scripts/run-tests.sh        # build the project
./build/shell_example
```

Type DSL statements followed by `run` to execute a forward pass, then
use `info` to inspect the graph.

## Graph visualisation

`export_dot()` converts a `HarmonicGraph` to Graphviz DOT format. The helper is
used by the `graph_cli` tool:

```bash
graph_cli dot graph.hgr -o graph.dot
dot -Tpng graph.dot -o graph.png
```

The generated image shows producers as boxes, layers as ellipses and consumers
as ovals with arrows representing the data flows.

## Interactive graph_cli

`graph_cli interactive` opens a simple REPL for editing a graph. The same
commands as batch mode are recognised along with `run`, `info`, `save`,
`quit` and `exit` to execute the graph, inspect it and write the graph
before leaving the shell. Batch mode commands include `add-layer <name>`,
`remove-layer <name>`, `add-flow <src> <dst> [--backward]` and
`remove-flow <src> <dst> [--backward]`.

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
