# JavaScript Runtime Bindings

The library exposes a thin V8 interface so Harmonics graphs can be used from Node.js or any V8 based environment. The bindings are optional and disabled by default.

## Building

Configure CMake with `-DHARMONICS_HAS_JS=1` and provide a V8 installation. This builds the shared library `harmonics_js` which exports the JavaScript functions.

```bash
cmake -S . -B build -DHARMONICS_HAS_JS=1
cmake --build build -j $(nproc)
```

The resulting module can be loaded as a Node.js addon, typically named `libharmonics_js.so` on Unix-like systems or `harmonics_js.dll` on Windows.

## Available functions

The bindings register the following helpers on the exported object:

- `compileGraph(source)` – parse a DSL string and return a graph handle.
- `destroyGraph(graph)` – release the graph handle.
- `createRuntime(graph)` – create a `CycleRuntime` for the graph.
- `destroyRuntime(runtime)` – free the runtime instance.
- `runCycle(runtime)` – execute one cycle of the graph.

## Example usage

```javascript
const harmonics = require('./build/Release/libharmonics_js');

const src = `harmonic demo { producer x; consumer y; cycle { x -> y; } }`;
const g = harmonics.compileGraph(src);
const rt = harmonics.createRuntime(g);
harmonics.runCycle(rt);
harmonics.destroyRuntime(rt);
harmonics.destroyGraph(g);
```

Browser builds follow the same API when compiled with Emscripten. The module exposes the exported functions directly on the WebAssembly instance.

