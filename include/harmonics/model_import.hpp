#pragma once

#include "harmonics/core.hpp"
#include "harmonics/cycle.hpp"
#include "harmonics/graph.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#if __has_include(<onnx/onnx_pb.h>)
#include <onnx/onnx_pb.h>
#endif
#if __has_include(<tensorflow/cc/saved_model/loader.h>)
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/util/tensor_bundle/tensor_bundle.h>
#endif

namespace harmonics {

/**
 * \brief Common helpers for loading model weights from external frameworks.
 *
 * These utilities read framework-specific checkpoints and convert them into a
 * vector of `NamedTensor` pairs. Each function shares the same signature and
 * throws `std::runtime_error` if an I/O error occurs or the required headers are
 * not available at compile time.
 */

/** Pair of tensor name and tensor value loaded from a model. */
using NamedTensor = std::pair<std::string, HTensor>;

/**
 * Load initializer weights from an ONNX model file.
 *
 * The function parses the ONNX `initializer` tensors from the model graph and
 * converts them into `HTensor` objects. If the ONNX headers are not available
 * at compile time, the function throws `std::runtime_error`.
 */
inline std::vector<NamedTensor> import_onnx_weights(const std::string& path) {
#if __has_include(<onnx/onnx_pb.h>)
    onnx::ModelProto model;
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open ONNX file");
    if (!model.ParseFromIstream(&in))
        throw std::runtime_error("failed to parse ONNX file");
    std::vector<NamedTensor> out;
    const auto& graph = model.graph();
    out.reserve(graph.initializer_size());
    for (const auto& t : graph.initializer()) {
        HTensor::Shape shape(t.dims().begin(), t.dims().end());
        std::vector<std::byte> data;
        if (!t.raw_data().empty()) {
            data.resize(t.raw_data().size());
            std::memcpy(data.data(), t.raw_data().data(), data.size());
        } else if (t.float_data_size() > 0) {
            data.resize(t.float_data_size() * sizeof(float));
            std::memcpy(data.data(), t.float_data().data(), data.size());
        } else if (t.int32_data_size() > 0) {
            data.resize(t.int32_data_size() * sizeof(std::int32_t));
            std::memcpy(data.data(), t.int32_data().data(), data.size());
        } else if (t.int64_data_size() > 0) {
            data.resize(t.int64_data_size() * sizeof(std::int64_t));
            std::memcpy(data.data(), t.int64_data().data(), data.size());
        } else if (t.data_type() == onnx::BOOL && t.int32_data_size() > 0) {
            data.resize(t.int32_data_size());
            for (int i = 0; i < t.int32_data_size(); ++i)
                data[i] = static_cast<std::byte>(t.int32_data(i) != 0);
        }

        HTensor::DType dt = HTensor::DType::Float32;
        switch (t.data_type()) {
        case onnx::FLOAT:
            dt = HTensor::DType::Float32;
            break;
        case onnx::DOUBLE:
            dt = HTensor::DType::Float64;
            break;
        case onnx::INT32:
            dt = HTensor::DType::Int32;
            break;
        case onnx::INT64:
            dt = HTensor::DType::Int64;
            break;
        case onnx::UINT8:
            dt = HTensor::DType::UInt8;
            break;
        case onnx::BOOL:
            dt = HTensor::DType::UInt8;
            break;
        default:
            throw std::runtime_error("unsupported ONNX tensor type");
        }

        out.emplace_back(t.name(), HTensor{dt, std::move(shape), std::move(data)});
    }
    return out;
#else
    (void)path;
    throw std::runtime_error("ONNX import not supported (onnx headers missing)");
#endif
}

/**
 * Load TensorFlow SavedModel weights.
 *
 * When TensorFlow headers are available the function loads the variables from
 * the SavedModel located at `path`. Each variable tensor is converted into an
 * `HTensor` and returned alongside its name. If TensorFlow is not available the
 * function throws at runtime.
 */
inline std::vector<NamedTensor> import_tensorflow_weights(const std::string& path) {
#if __has_include(<tensorflow/cc/saved_model/loader.h>)
    namespace tf = tensorflow;

    // SavedModel variables are stored inside "variables" with the file prefix
    // "variables". Join the pieces here so the reader locates
    // "variables/{variables.index,variables.data-*}" correctly.
    tf::BundleReader reader(tf::Env::Default(), tf::io::JoinPath(path, "variables", "variables"));
    if (!reader.status().ok())
        throw std::runtime_error(reader.status().ToString());

    std::vector<NamedTensor> out;
    std::vector<std::string> names;
    auto status = reader.ListVariables(&names);
    if (!status.ok())
        throw std::runtime_error(status.ToString());
    for (const auto& name : names) {
        tf::Tensor t;
        status = reader.Lookup(name, &t);
        if (!status.ok())
            throw std::runtime_error(status.ToString());
        HTensor::Shape shape(t.shape().dim_sizes().begin(), t.shape().dim_sizes().end());
        HTensor::DType dt = HTensor::DType::Float32;
        switch (t.dtype()) {
        case tf::DT_FLOAT:
            dt = HTensor::DType::Float32;
            break;
        case tf::DT_DOUBLE:
            dt = HTensor::DType::Float64;
            break;
        case tf::DT_INT32:
            dt = HTensor::DType::Int32;
            break;
        case tf::DT_INT64:
            dt = HTensor::DType::Int64;
            break;
        case tf::DT_UINT8:
            dt = HTensor::DType::UInt8;
            break;
        default:
            throw std::runtime_error("unsupported TensorFlow tensor type");
        }

        std::vector<std::byte> data(t.tensor_data().size());
        std::memcpy(data.data(), t.tensor_data().data(), data.size());
        out.emplace_back(name, HTensor{dt, std::move(shape), std::move(data)});
    }
    return out;
#else
    (void)path;
    throw std::runtime_error("TensorFlow import not supported in this build");
#endif
}

/**
 * Load PyTorch model weights from a TorchScript or state dictionary file.
 *
 * When LibTorch headers are available the function attempts to load the file as
 * a TorchScript module first. If that fails it falls back to loading a plain
 * state dictionary using `torch::load`. The tensors are converted into
 * `HTensor` objects and returned alongside their parameter names.
 *
 * If LibTorch is not available at compile time the function throws at runtime.
 */
inline std::vector<NamedTensor> import_pytorch_weights(const std::string& path) {
#if __has_include(<torch/script.h>)
    namespace th = torch;
    std::vector<NamedTensor> out;

    auto convert = [](const th::Tensor& t) -> std::pair<HTensor::DType, std::vector<std::byte>> {
        HTensor::DType dt{};
        switch (t.scalar_type()) {
        case th::kFloat:
            dt = HTensor::DType::Float32;
            break;
        case th::kDouble:
            dt = HTensor::DType::Float64;
            break;
        case th::kInt:
            dt = HTensor::DType::Int32;
            break;
        case th::kLong:
            dt = HTensor::DType::Int64;
            break;
        case th::kByte:
            dt = HTensor::DType::UInt8;
            break;
        default:
            throw std::runtime_error("unsupported PyTorch tensor type");
        }

        th::Tensor ct = t.contiguous();
        std::size_t bytes = ct.numel() * ct.element_size();
        std::vector<std::byte> data(bytes);
        std::memcpy(data.data(), ct.data_ptr(), bytes);
        return {dt, std::move(data)};
    };

    try {
        th::jit::script::Module mod = th::jit::load(path, th::kCPU);
        for (const auto& p : mod.named_parameters(/*recurse=*/true)) {
            HTensor::Shape shape(p.value.sizes().begin(), p.value.sizes().end());
            auto conv = convert(p.value);
            out.emplace_back(p.name, HTensor{conv.first, std::move(shape), std::move(conv.second)});
        }
        return out;
    } catch (const c10::Error&) {
        // Try loading as a plain state dictionary
    }

    try {
        std::map<std::string, th::Tensor> dict;
        th::load(dict, path);
        for (const auto& kv : dict) {
            HTensor::Shape shape(kv.second.sizes().begin(), kv.second.sizes().end());
            auto conv = convert(kv.second);
            out.emplace_back(kv.first,
                             HTensor{conv.first, std::move(shape), std::move(conv.second)});
        }
        return out;
    } catch (const c10::Error& e) {
        throw std::runtime_error(e.what());
    }
#else
    (void)path;
    throw std::runtime_error("PyTorch import not supported in this build");
#endif
}

/** Attach imported weights to the corresponding layers of a runtime. */
inline void attach_named_weights(CycleRuntime& rt, const std::vector<NamedTensor>& weights) {
    for (const auto& named : weights) {
        const auto& name = named.first;
        const auto& tensor = named.second;
        try {
            NodeId id = rt.graph().find(name);
            if (id.kind == NodeKind::Layer)
                rt.state().weights[id.index] = tensor;
        } catch (const std::exception&) {
            // Ignore weights that don't match a layer
        }
    }
}

} // namespace harmonics
