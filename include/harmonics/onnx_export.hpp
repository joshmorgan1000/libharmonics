#pragma once

#include "harmonics/graph.hpp"
#if __has_include(<onnx/onnx_pb.h>)
#include <onnx/onnx_pb.h>
#endif
#include <fstream>
#include <string>

namespace harmonics {

#if __has_include(<onnx/onnx_pb.h>)

inline std::string node_name(const HarmonicGraph& g, NodeId id) {
    switch (id.kind) {
    case NodeKind::Producer:
        return g.producers[id.index].name;
    case NodeKind::Consumer:
        return g.consumers[id.index].name;
    case NodeKind::Layer:
        return g.layers[id.index].name;
    }
    return {};
}

/** Export the graph structure to an ONNX ModelProto.
 *  The tiny ONNX schema is used where nodes and edges are stored as
 *  initializer tensors.
 */
inline onnx::ModelProto export_onnx_model(const HarmonicGraph& g) {
    onnx::ModelProto model;
    onnx::GraphProto* graph = model.mutable_graph();

    auto add_node = [&](const std::string& name, int kind, std::optional<int> width) {
        onnx::TensorProto* t = graph->add_initializer();
        t->set_name("node:" + name);
        t->set_data_type(onnx::INT32);
        t->add_int32_data(kind);
        t->add_int32_data(width ? *width : 0);
    };

    for (const auto& p : g.producers)
        add_node(p.name, 0, p.shape);
    for (const auto& c : g.consumers)
        add_node(c.name, 1, c.shape);
    for (const auto& l : g.layers)
        add_node(l.name, 2, l.shape);

    for (const auto& line : g.cycle) {
        std::string src = node_name(g, line.source);
        for (const auto& ar : line.arrows) {
            std::string dst = node_name(g, ar.target);
            onnx::TensorProto* t = graph->add_initializer();
            t->set_name("edge:" + src + ":" + dst);
            t->set_data_type(onnx::INT32);
            t->add_int32_data(ar.backward ? 1 : 0);
        }
    }
    return model;
}

inline void export_onnx_file(const HarmonicGraph& g, const std::string& path) {
    onnx::ModelProto m = export_onnx_model(g);
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("failed to open output file");
    if (!m.SerializeToOstream(&out))
        throw std::runtime_error("failed to write ONNX file");
}

#else

inline onnx::ModelProto export_onnx_model(const HarmonicGraph&) {
    throw std::runtime_error("ONNX export not supported (onnx headers missing)");
}
inline void export_onnx_file(const HarmonicGraph&, const std::string&) {
    throw std::runtime_error("ONNX export not supported (onnx headers missing)");
}

#endif

} // namespace harmonics
