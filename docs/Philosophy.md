# Harmonics DSL – Design Philosophy

> *"Let computation flow like music – coherent, adaptive, and substrate‑agnostic."*

---

## 1  Continuous Flows, Not Discrete Steps

Traditional languages view programs as **ordered statements** executed by a deterministic clock. Harmonics inverts that premise:

* Every value is a **stream** – potentially infinite, time‑indexed, and lazily materialised.
* Control flow is expressed by **relationships** (arrows), not instruction order.
* Time is implicit; precision and cadence are negotiable properties of a stream, not baked into syntax.

This mirrors physical reality (signals, fields, feedback) and aligns with future substrates such as analog arrays or quantum amplitudes.

---

## 2  Symbolic Fidelity Over Premature Quantisation

Binary silicon forces irrational numbers and infinite sets into coarse containers. Harmonics keeps them **symbolic** until the last responsible moment.

| Instead of…                  | We…                                                    |
| ---------------------------- | ------------------------------------------------------ |
| `π ≈ 3.14159f`               | Keep `π` symbolic; quantise only at execution site.    |
| Hard‑coded `float32` tensors | Allow width/precision to be negotiated by the runtime. |

Result: exact algebra, reduced rounding noise, and a codebase already compatible with post‑binary hardware.

---

## 3  Declarative Topology, Minimal Vocabulary

Programming should read like a schematic:

```harmonics
producer images;
layer   embed 1/4 input;
cycle { images -(relu)-> embed; }
```

Only **five** reserved words (`harmonic producer consumer layer cycle`). Everything else – layers, activations, losses – is user space. This prevents DSL stagnation and empowers teams to extend functionality without language forks.

---

## 4  Function Registry, Not Keywords

Activations and loss functions are **objects**, not syntax. They live in a runtime registry:

* Built‑ins (`relu`, `sigmoid`, `cross_entropy`) are pre‑registered.
* Users can swap or extend them without touching the compiler.
* Future metadata (GPU shader IDs, FPGA bitstreams) attaches naturally to those objects.

This isolates math from infrastructure – a core tenet of Harmonics.

---

## 5  Self‑Negotiation of Precision & Scale

The runtime is a **marketplace** that balances:

* Bit‑width vs. energy.
* Layer width vs. memory.
* Latency vs. throughput.

Programmers express *constraints* (ratios, tolerances). The engine finds any instantiation that satisfies them. This yields:

* Seamless scaling from microcontrollers to datacenter pods.
* Automatic exploitation of mixed‑precision units.
* Future‑proofing for analog or quantum processing elements.

---

## 6  Decoupling Data Logistics From Model Logic

By declaring producers and consumers outside the flow, Harmonics lets engineers:

* Rewire a graph to new datasets without edits.
* Use the same pipeline for real‑time streams and offline batches.
* Unit‑test models with synthetic producers in <10 LOC.
* Split a model across machines by binding remote producers and consumers; the transport layer lives outside the core engine.

---

## 7  Namespace Hygiene & Interop

Runtime types are prefixed with **`H`** (`HTensor`, `HProducer`) to:

* Prevent clashes with existing ML libraries.
* Signal their substrate‑agnostic semantics.

Interop shims can wrap PyTorch, TensorFlow, or custom kernels behind the same interfaces.

---

## 8  Roadmap Alignment

| Phase                         | Philosophy Tie‑in                                                               |
| ----------------------------- | ------------------------------------------------------------------------------- |
| **Parser + CPU interpreter**  | Prove declarative flow and symbolic fidelity work end‑to‑end.                   |
| **Auto‑tuned precision**      | Demonstrate self‑negotiation on heterogeneous CPUs.                             |
| **Integration with others**   | Add the ability to import and train or fine-tune existing LLMs or other models. |
| **Model Hacking/Disassembly** | Enable introspection and modification of Harmonics models.                      |
| **GPU / FPGA back‑ends**      | Validate substrate‑agnostic promise; attach metadata to registry objects.       |
| **Quantum stub**              | Keep identical DSL while mapping flows to qubit amplitudes.                     |

---

## 9  Closing Mantra

> **Write once, resonate anywhere.**

Harmonics refuses to fossilise today’s hardware assumptions into tomorrow’s software. The same DSL should run on
anything from edge devices to quantum processors (in the future of course), without having to change any code.
By treating computation as flowing resonance rather than discrete ticking, it charts a path that will remain
coherent – whether executed on CMOS, photonics, or qubits.
