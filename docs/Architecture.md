# Harmonics DSL — Architecture v0.3

> A declarative, stream‑native language for building, training and serving neural flows that automatically negotiate precision, width and substrate.
This engine has been written from scratch so that it can break free from the traditional philosophy that model architectures must be set in stone.  See
[`Philosophy.md`](Philosophy.md) for the guiding principles that underpin the design.

---

## 1  Design Goals

| Goal | Why it matters |
|------|----------------|
| **Declarative topology** | Models read as data‑flow, not imperative code. |
| **Producer/Consumer decoupling** | Same graph runs on files, sockets, message buses, custom stores. |
| **Substrate agnosticism** | Source targets CPU, GPU, FPGA or encrypted datastore without edits. |
| **Distributed graph partitioning** | Layers or even sub‑sections can be split across processes; the host application implements the message layer. |
| **Self‑negotiating precision & width** | Layer sizes and bit‑widths resolved at materialisation. |
| **Pluggable ops** | Activations & loss functions are objects registered at runtime. |
| **Clear namespace** | Runtime types prefixed with `H` (e.g. `HTensor`) to avoid clashes with third‑party libs. |
| **Zero Python dependency** | Reference interpreter is modern C++; GPU support on the roadmap. |

### Overview

```
Producer -> [AsyncProducer] -> Layers -> Consumer
```

An optional `AsyncProducer` prefetches samples on a background thread before
feeding them into the layer stack executed by `CycleRuntime`. Consumers receive
output tensors once each cycle completes.

Wrapper producers such as `ShuffleProducer`, `BatchProducer` and
`AugmentProducer` transform the output of another producer before it reaches the
layers. They can be combined freely and wrapped in `AsyncProducer` when
prefetching is desired.

---

## 2  Reserved Keywords

```
harmonic   producer   consumer   layer   cycle
```
Only five words are reserved. Everything else—layer names, activation IDs, loss IDs—is user‑defined.  Common functions (`relu`, `sigmoid`, `cross_entropy` …) ship as pre‑registered objects in the runtime’s **function registry** and can be overridden.

---

## 3  Declarations

### 3.1  Streams
```
producer <name> [ {shape} ] [ ratio ] ;
consumer <name> [ {shape} ] ;
```
* **Shape** optional—if omitted, inferred from first sample.
* **Ratio** declares temporal alignment (e.g. `1/1` means lock‑step with another stream).

### 3.2  Layers
```
layer <name> [ ratio ] ;
```
`ratio` sets width relative to another layer, e.g. `layer hidden 1/2 input;`.

---

## 4  Flow Syntax

### 4.1  Function‑tagged arrows
```
# Forward pass with activation
a -(relu)-> b

# Linear map
a -> b

# Training tap with explicit loss
b <-(cross_entropy)- labelStream
```
* Tag contents are IDs looked up in the **function registry**.  
* Presence of *any* backward arrow inside a `cycle{}` block triggers training; otherwise the graph runs inference on demand as long as the cycle is running and the producer continues to produce records

### 4.2  Cycle block
A `cycle { … }` section is epoch‑aware; meaning it can execute as a live operator while the defined conditions are met.  
These conditions may be that you only want to run training cycles for an hour, perhaps you want to run them until loss hits a specific threshold... or perhaps you just want it to run indefinitely. How you set this parameter will determine the optimization behavior of the engine. 
When the cycle is not running, the HarmonicGraph can act as a static inference operator, or called on demand.

---

## 5  End‑to‑End Example (MNIST)

```harmonics
harmonic mnist_train_cycle {

  producer mnist_training_data;             // 784‑wide vectors inferred
  producer mnist_label_data 1/1 mnist_training_data;

  layer input;
  layer hidden 1/2 input;
  layer output 1/2 hidden;
  // Note: the input and output layer dimensions may be bound by the detected dimensions
  // of the data (we know 784 / 2 / 2 != 10 for instance). The engine will take these as a "suggestion" in those cases.

  cycle {
    mnist_training_data -(relu)-> input
                       -(relu)-> hidden
                       -(sigmoid)-> output;
    output <-(cross_entropy)- mnist_label_data;   // training tap
  }
}
```
*Remove the last line to convert the same graph to pure inference.*

---

## 6  Runtime API Sketch (C++)

### 6.1  Core runtime types
```cpp
class HTensor { /* ND‑array; dtype & shape resolved at runtime */ };

// Function registry entries
class ActivationFunction {
  // minimal callable interface; extended metadata (e.g. GPU shader name) lives here
  virtual HTensor operator()(const HTensor& x) const = 0;
};
class LossFunction {
  virtual HTensor operator()(const HTensor& pred, const HTensor& target) const = 0;
};

// Stream interfaces
class Producer {
public:
  virtual HTensor next()            = 0;  // pull next sample
  virtual std::size_t size() const  = 0;  // dataset size for scheduler hints
  virtual ~Producer() = default;
};
class Consumer {
public:
  virtual void push(const HTensor&) = 0;
  virtual ~Consumer() = default;
};
```

### 6.2  Loading & training a graph
```cpp
// Parse DSL file
HarmonicGraph g = parse_hmx("mnist_train_cycle.hmx");

// Data sources ----------------------------------------------------------------
struct FileProducer : Producer {
  std::ifstream file;
  explicit FileProducer(std::ifstream&& f) : file(std::move(f)) {}
  HTensor next() override        { return loadNextRecord(file); }
  std::size_t size() const override { return fileRecordCount(file); }
};

auto img_file = std::ifstream("train-images.idx");
auto lbl_file = std::ifstream("train-labels.idx");

g.bindProducer("mnist_training_data", std::make_shared<FileProducer>(std::move(img_file)));
g.bindProducer("mnist_label_data",   std::make_shared<FileProducer>(std::move(lbl_file)));

// Override activation if desired
g.registerActivation("relu", std::make_shared<MyFusedReluShader>());

// Train for one hour with auto‑tuned precision
g.fit(std::chrono::hours(1), make_auto_policy());

// Persist model ----------------------------------------------------------------
std::span<const std::byte> blob{static_cast<const std::byte*>(g.data()), g.size()};
writeBinary("mnist_weights.bin", blob);
```

### 6.3  Precision negotiation

The runtime resolves layer widths and numeric bit‑widths when a graph is
materialised.  Producer shapes act as anchors: any layer referencing another
layer or producer via a `ratio` inherits its width by applying the ratio's
numerator and denominator.  Widths are propagated until all ratios are
resolved.  During the first training cycle the engine samples activation ranges
and chooses the smallest integer bit‑width that satisfies the configured
precision policy (for example `bits ≤ 8` or a target entropy threshold).

### 6.4  Dynamic plugin loading

Custom activation, loss and layer implementations can be loaded at runtime from
shared libraries. Call `load_plugin()` with the path to a `.so` file containing
an entry point named `harmonics_register`:

```cpp
// inside my_plugin.cpp
extern "C" void harmonics_register(harmonics::FunctionRegistry& reg) {
    reg.register_activation("my_act", std::make_shared<MyActivation>());
}
```

The plugin is loaded like so:

```cpp
harmonics::load_plugin("./libmy_plugin.so");
```

This registers functions with the global `FunctionRegistry` for use in DSL
graphs.


### 6.5  Importing third-party models

Existing networks can be ingested by translating their weight blobs into the Harmonics tensor format. The first wave of adapters will cover common open-source standards:

* **ONNX** (`.onnx`)
* **TensorFlow SavedModel** (`.pb`) and **Keras** (`.h5`)
* **PyTorch** state dictionaries and **TorchScript** (`.pt`, `.pth`)
* **Flax** / **JAX** checkpoints
* **Hugging Face Safetensors** (`.safetensors`)
* **TensorFlow Lite** (`.tflite`)
* **Caffe** (`.caffemodel`)
* **MXNet** (`.params`)
* **Scikit-learn** pickled models

Additional importers can register themselves via the plugin interface.

---

## 7  Grammar (abridged PEG)

```peg
harmonic      <- 'harmonic' IDENT block
block         <- '{' stmt* '}'
stmt          <- producer / consumer / layer / cycle
producer      <- 'producer' IDENT shape? ratio? ';'
consumer      <- 'consumer' IDENT shape? ';'
layer         <- 'layer' IDENT ratio? ';'
cycle         <- 'cycle' flowBlock
flowBlock     <- '{' flowLine* '}'
flowLine      <- IDENT (fArrow IDENT)+ ';'
fArrow        <- '->' / '-(' IDENT ')->' / '<-' / '<-(' IDENT ')-'
ratio         <- NUMBER '/' NUMBER IDENT
shape         <- '{' INT '}'
```
(The full grammar—including parallel branches—lives in `grammar/harp.peg`.)

---

## 8  Extension Features

Harmonics already ships several optional components that extend the core engine:

* **Layer behaviors** &ndash; convolution, normalisation and attention layers can
  be reconfigured at runtime via helper functions. Each behaviour expands into
  primitive operations when a graph is materialised so specialised hardware
  pipelines can be described without editing the DSL.
* **Precision policies** &ndash; select the activation bit width using the
  `PrecisionPolicy` interface. Policies such as `make_max_bits_policy()` pin a
  fixed format while `make_auto_policy()` chooses a hardware&ndash;dependent
  mixed‑precision configuration automatically.
* **Deployment descriptors** &ndash; control back‑end selection and per‑partition
  options when running distributed graphs.
* **Introspection tools** &ndash; query and edit graphs while they are running.
* **Quantum mapping helpers** &ndash; convert graphs into quantum circuits and execute them on supported hardware.

---

## 9  GPU Back Ends

The reference interpreter currently executes on the CPU. A Vulkan based GPU
back end keeps the same runtime interface so graphs run without changes. Cycle
blocks compile to GPU kernels that stream tensors through device memory.
Production back ends for CUDA and OpenCL mirror the Vulkan path.

### 9.1 Vulkan build requirements

Building with Vulkan requires the Vulkan SDK from LunarG (version 1.3 or
newer). Configure the project with `-DHARMONICS_HAS_VULKAN=1` so that the Vulkan
path is compiled. The reference implementation does not depend on a specific
GPU vendor.

### 9.2 Runtime configuration

Select the GPU backend at runtime by setting `backend = Backend::GPU` in the
`DeploymentDescriptor`. Individual partitions can override this by pushing a
`PartitionOptions` record into `deploy.partitions`. When GPU support is enabled
but no compatible device is present, execution transparently falls back to the
CPU path.

Setting `backend = Backend::Auto` lets the runtime choose the best accelerator.
The helper `select_accelerator_backend()` checks for GPU and FPGA support and
returns the optimal backend, defaulting to the CPU when no accelerator is
available.

### 9.3 Incremental compilation cache

When GPU kernels are compiled the resulting SPIR-V is stored in an in-memory
cache indexed by shader name. Compiling another graph that uses the same shader
reuses the cached bytecode to avoid redundant compilation steps. The workflow is
described in detail in [IncrementalCompilationCache.md](IncrementalCompilationCache.md).

---

## 10  Closing Thought

Write flows, plug in functions, and let the Harmonics engine handle *how* and *where* they resonate.
