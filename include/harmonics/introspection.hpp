#pragma once

#include <string>
#include <vector>

#include "harmonics/cycle.hpp"
#include "harmonics/graph.hpp"

namespace harmonics {

/** Information about a layer in a graph. */
struct LayerInfo {
    std::string name;  ///< Layer name
    std::size_t width; ///< Resolved width or 0 if unknown
    bool trainable;    ///< Whether the layer has trainable parameters
};

/** Return metadata for all layers in the graph. */
inline std::vector<LayerInfo> get_layer_info(const HarmonicGraph& g) {
    std::vector<LayerInfo> info;
    info.reserve(g.layers.size());
    for (const auto& l : g.layers) {
        std::size_t w = l.shape ? static_cast<std::size_t>(*l.shape) : 0;
        info.push_back({l.name, w, true});
    }
    return info;
}

/** Resize a CycleRuntime's state vectors after the graph has changed. */
inline void sync_runtime(CycleRuntime& rt) {
    const auto& g = rt.graph();
    auto& st = rt.state();
    st.producer_tensors.resize(g.producers.size());
    st.layer_tensors.resize(g.layers.size());
    st.consumer_tensors.resize(g.consumers.size());
    st.weights.resize(g.layers.size());
    st.precision_bits.resize(g.layers.size());
}

/** Access mutable weights for a layer by name. */
inline HTensor& layer_weights(CycleRuntime& rt, const std::string& name) {
    NodeId id = rt.graph().find(name);
    if (id.kind != NodeKind::Layer)
        throw std::runtime_error(name + " is not a layer");
    return rt.state().weights[id.index];
}

/** Access const weights for a layer by name. */
inline const HTensor& layer_weights(const CycleRuntime& rt, const std::string& name) {
    NodeId id = rt.graph().find(name);
    if (id.kind != NodeKind::Layer)
        throw std::runtime_error(name + " is not a layer");
    return rt.state().weights[id.index];
}

} // namespace harmonics
