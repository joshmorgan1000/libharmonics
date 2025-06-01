# Quantum Stub

The quantum stub demonstrates how a `HarmonicGraph` can be translated into a quantum circuit. It serves as a proof of concept for mapping the DSL onto non-classical targets and highlights that the framework is ready for future quantum accelerators.

## map_to_quantum()

The helper `map_to_quantum()` is defined in `include/harmonics/quantum_stub.hpp`. It assigns a qubit to every producer, layer and consumer in the graph. Each flow arrow becomes a controlled-X gate (`CX`), and consumers are measured at the end of the circuit.

```cpp
harmonics::Parser parser{"producer p {1}; consumer c {1}; layer l; cycle { p -> l; l -> c; }"};
auto ast = parser.parse_declarations();
auto g = harmonics::build_graph(ast);
auto qc = harmonics::map_to_quantum(g);
```

`QuantumCircuit` holds the number of qubits and a list of `QuantumOp` instructions describing the gates.

## Demonstration

The unit test `quantum_stub_test.cpp` builds a small graph and verifies the generated circuit. Each node maps to a qubit and the final operations include measurement.

## Larger graph example

For a more substantial demonstration the program `examples/quantum_stub_demo.cpp`
constructs a graph with multiple producers and layers. The graph contains
branching flows so the resulting circuit includes several controlled-X gates.

```cpp
const char* src = R"(
    producer a {2};
    producer b {2};
    layer l1;
    layer l2;
    consumer out {2};
    cycle {
        a -> l1;
        b -> l1;
        l1 -> l2;
        l1 -> out;
        l2 -> out;
    }
)";
harmonics::Parser parser{src};
auto ast = parser.parse_declarations();
auto g = harmonics::build_graph(ast);
auto qc = harmonics::map_to_quantum(g);
```

Running the example prints the number of qubits followed by a list of quantum
operations. Each flow in the graph appears as a `CX` instruction while the
consumer qubits are measured at the end.

### Running the demo

Build the project with the helper script and run the `quantum_stub_example`
binary from the build directory:

```bash
./scripts/run-tests.sh         # configures and builds
./build-Release/quantum_stub_example
```

The demo outputs the qubit count and each gate on a separate line. A typical
run prints:

```
Qubits: 5
CX 0 3
CX 1 3
CX 3 4
CX 3 2
CX 4 2
Measure 2
```

## Expanded graph example

The program `examples/quantum_stub_large.cpp` demonstrates mapping a bigger
graph with three layers and multiple producers. The additional branching
generates more `CX` operations and showcases the stub on a slightly larger
model.

```cpp
const char* src = R"(
    producer a {2};
    producer b {2};
    producer c {2};
    layer l1;
    layer l2;
    layer l3;
    consumer out {2};
    cycle {
        a -> l1;
        b -> l1;
        c -> l2;
        l1 -> l2;
        l2 -> l3;
        l1 -> out;
        l2 -> out;
        l3 -> out;
    }
)";
harmonics::Parser parser{src};
auto ast = parser.parse_declarations();
auto g = harmonics::build_graph(ast);
auto qc = harmonics::map_to_quantum(g);
```

Run it the same way:

```bash
./scripts/run-tests.sh
./build-Release/quantum_stub_large_example
```

Example output:

```
Qubits: 7
CX 0 3
CX 1 3
CX 2 4
CX 3 4
CX 4 5
CX 3 6
CX 4 6
CX 5 6
Measure 6
```

## Complex model example

To illustrate the stub on a deeper network the program
`examples/quantum_stub_complex.cpp` builds a graph with six layers,
four producers and multiple skip connections. The resulting circuit
contains many more controlled-X operations.

```cpp
const char* src = R"(
    producer a {2};
    producer b {2};
    producer c {2};
    producer d {2};
    layer l1;
    layer l2;
    layer l3;
    layer l4;
    layer l5;
    layer l6;
    consumer out {2};
    cycle {
        a -> l1;
        b -> l1;
        c -> l2;
        d -> l3;
        l1 -> l2;
        l2 -> l3;
        l3 -> l4;
        l2 -> l4;
        l4 -> l5;
        l1 -> l5;
        l5 -> l6;
        l3 -> l6;
        l6 -> out;
    }
)";
harmonics::Parser parser{src};
auto ast = parser.parse_declarations();
auto g = harmonics::build_graph(ast);
auto qc = harmonics::map_to_quantum(g);
```

Build and run it using the helper script:

```bash
./scripts/run-tests.sh
./build-Release/quantum_stub_complex_example
```

## Quantum hardware backend

Hardware execution is enabled when Harmonics is built with
`-DHARMONICS_HAS_QUANTUM_HW=1`.  The header
`<harmonics/quantum_hardware.hpp>` exposes `execute_on_hardware()` which mirrors
the simulator API.  At runtime the function attempts to load a hardware
library when the environment variable `HARMONICS_ENABLE_QUANTUM_HW=1` is set.
The library path defaults to `libquantum_hw.so` but can be overridden with the
variable `HARMONICS_QUANTUM_HW_LIB`.  If loading fails or the variables are
unset the call transparently falls back to the state vector simulator.

```cpp
#include <harmonics/quantum_hardware.hpp>

auto qc = harmonics::map_to_quantum(g);
auto result = harmonics::execute_on_hardware(qc);
```

### Example programs

The executables `quantum_hardware_demo` and `quantum_hardware_large` illustrate
the hardware path.  Build the project with `scripts/run-tests.sh` and run the
programs from the build directory:

```bash
./scripts/run-tests.sh
./build-Release/quantum_hardware_demo
./build-Release/quantum_hardware_large
```

Both programs print the measurement results returned by the device.  The unit
test `quantum_hardware_test.cpp` exercises the same code path and verifies that
the library falls back to the simulator when the hardware library is missing.

## Future work

The stub is intentionally minimal. Future releases may extend it into a full backend. The [project roadmap](../ROADMAP.md) tracks the task to showcase the stub on a larger model and expand this documentation.
