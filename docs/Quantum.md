# Quantum Mapping

This document explains how a `HarmonicGraph` can be translated into a quantum circuit and executed either on the simulator or via the hardware backend.

> **Note**
> A lightweight hardware backend is now available through
> `harmonics::execute_on_hardware()`. When the project is built with
> `-DHARMONICS_HAS_QUANTUM_HW=1` the runtime loads a device library at
> execution time. If the library cannot be found the call transparently falls
> back to the built‑in simulator.

## map_to_quantum()

The helper `map_to_quantum()` assigns a qubit to every producer, layer and consumer in the graph. Each flow arrow becomes a controlled-X gate (`CX`), and consumers are measured at the end of the circuit.

```cpp
harmonics::Parser parser{"producer p {1}; consumer c {1}; layer l; cycle { p -> l; l -> c; }"};
auto ast = parser.parse_declarations();
auto g = harmonics::build_graph(ast);
auto qc = harmonics::map_to_quantum(g);
```

`QuantumCircuit` holds the number of qubits and a list of `QuantumOp` instructions describing the gates.

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

### Hardware backend

A minimal plugin is provided in `examples/quantum_hw_backend.cpp`. When Harmonics
is configured with `-DHARMONICS_HAS_QUANTUM_HW=1` this file builds into
`libquantum_hw.so` alongside the examples. The backend uses the runtime device
selection helpers and forwards circuits to the simulator. Load the library at
runtime by setting the environment variables and run an example program:

```bash
export HARMONICS_ENABLE_QUANTUM_HW=1
export HARMONICS_QUANTUM_HW_LIB=libquantum_hw.so
./build-Release/quantum_hardware_demo
```

### Running on a real device

To target an actual quantum accelerator, implement a shared library that
provides `harmonics_quantum_execute()` and translates `QuantumCircuit`
objects to your device SDK.  A minimal sketch might look like:

```cpp
#include <harmonics/quantum_backend.hpp>
#include <my_qpu_sdk.hpp>

extern "C" harmonics::QuantumResult
harmonics_quantum_execute(const harmonics::QuantumCircuit& qc) {
    auto dev_circuit = translate(qc); // convert to device representation
    auto measurements = run_on_qpu(dev_circuit);
    return {measurements};
}
```

Compile the library (for example `libqpu.so`) and set the environment
variables so that Harmonics loads it at runtime:

```bash
export HARMONICS_ENABLE_QUANTUM_HW=1
export HARMONICS_QUANTUM_HW_LIB=libqpu.so
./build-Release/quantum_hardware_demo
```


The demo will now execute the circuit on the device and print the
measurement results returned by the hardware.

### Hardware library interface

The device library loaded by `execute_on_hardware()` must export the
function

```cpp
extern "C" harmonics::QuantumResult
harmonics_quantum_execute(const harmonics::QuantumCircuit& qc);
```

The runtime calls this entry point for every circuit and expects the
returned measurements in the same format as the simulator. Device
selection is handled through `set_quantum_device_index()` or the
`HARMONICS_QUANTUM_HW_DEVICE` environment variable. Refer to
`examples/quantum_hw_backend.cpp` for a minimal implementation.

### Checking runtime availability

Call `quantum_hardware_runtime_available()` to verify that the
environment variables are configured correctly and that a device
library can be loaded.  The helper returns `true` when
`HARMONICS_ENABLE_QUANTUM_HW=1` is set and the library specified by
`HARMONICS_QUANTUM_HW_LIB` (or `libquantum_hw.so` by default) can be
found on the system.

```cpp
if (!harmonics::quantum_hardware_runtime_available()) {
    std::cerr << "Quantum hardware not found, falling back to simulator\n";
}
```

### Device selection

When multiple quantum accelerators are available, set `HARMONICS_QUANTUM_HW_DEVICE`
to pick a specific device. The current backend exposes helper functions
`set_quantum_device_index()` and `quantum_device_index()` which mirror the
behaviour of the GPU and OpenCL selectors.

## Future work

Further hardware integrations and optimised compilation passes are planned.
The [project roadmap](../ROADMAP.md) tracks upcoming tasks and features.
