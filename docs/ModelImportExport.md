# Model Import Design

This document sketches the intended approach for loading weights from various frameworks into Harmonics.

## Goals

* Allow models trained in common frameworks to be brought into a Harmonics runtime.
* Keep Harmonics itself header only; heavy framework dependencies remain optional.
* Support both plain ONNX files and checkpoints from TensorFlow and PyTorch.

## API overview

Weights are returned as a vector of `NamedTensor` pairs defined in
`include/harmonics/model_import.hpp`. All importer helpers share the same simple
signature so tooling can call them interchangeably:

```cpp
using NamedTensor = std::pair<std::string, HTensor>;
```

The following helpers load weights from their respective formats:

```cpp
std::vector<NamedTensor> import_onnx_weights(const std::string& path);
std::vector<NamedTensor> import_tensorflow_weights(const std::string& path);
std::vector<NamedTensor> import_pytorch_weights(const std::string& path);
void attach_named_weights(CycleRuntime& rt,
                          const std::vector<NamedTensor>& weights);
```

All functions throw `std::runtime_error` on I/O failures or when support for a
framework is not compiled in. `attach_named_weights` matches tensors to layers
by name and silently ignores any unmatched entries.

## ONNX

When the `onnx` protobuf headers are available at build time the importer parses
the model's `initializer` tensors directly. It returns the weights as a vector
of `NamedTensor`.
If the headers are missing the build falls back to a stub that throws at
runtime.
The generated headers rely on features added in **Protobuf 3.21**. If your
system ships an older version you may see a build error complaining that
`google/protobuf/port_def.inc` is missing. Updating your Protobuf installation
to 3.21 or newer resolves the issue.

## TensorFlow

Loading TensorFlow checkpoints requires linking against TensorFlow's protobuf definitions.
When the headers are available Harmonics uses the lightweight `SavedModel` reader
to locate variable tensors and converts them into `HTensor` objects. The helper
`import_tensorflow_weights(path)` mirrors the ONNX variant and returns a vector of
`NamedTensor` pairs.

If TensorFlow is not detected at compile time the function falls back to a stub
that throws at runtime. This keeps the core library header only while still
allowing optional integration.

## PyTorch

PyTorch models are typically saved as TorchScript archives or state dictionaries.
The importer uses LibTorch to load the archive or `.pt` file and iterates over
the contained tensors, converting them into `HTensor` objects. The helper
`import_pytorch_weights(path)` mirrors the TensorFlow interface and returns a
`NamedTensor` list. When LibTorch is not available the function throws at runtime
and the importer is omitted from the build.

## Transformers

HuggingFace models can usually be exported to ONNX or TorchScript. Once the above importers exist the recommended path is to convert a trained transformer using the appropriate tool
(e.g. `transformers.onnx`) and then load the resulting file via the ONNX importer. Direct integration may be added later but is out of scope for the first implementation.

## Command line conversion

Weights can be extracted from existing models using the `model_convert` utility.
The tool wraps the import helpers above and writes a `HNWT` file that preserves
tensor names:

```bash
model_convert --onnx model.onnx -o weights.hnwt
```

The resulting file can be loaded with `load_named_weights()` and attached to a
runtime via `attach_named_weights()`.

