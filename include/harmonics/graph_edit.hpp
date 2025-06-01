#pragma once

#include <algorithm>
#include <optional>
#include <string>

#include "harmonics/graph.hpp"

namespace harmonics {

/** Add a new layer to the graph and propagate ratio widths. */
inline NodeId add_layer(HarmonicGraph& g, const std::string& name,
                        std::optional<Ratio> ratio = std::nullopt) {
    for (const auto& l : g.layers)
        if (l.name == name)
            throw std::runtime_error("duplicate layer: " + name);
    g.layers.push_back({name, ratio});
    propagate_ratios(g);
    return {NodeKind::Layer, g.layers.size() - 1};
}

/** Remove the layer with the given name from the graph. */
inline void remove_layer(HarmonicGraph& g, const std::string& name) {
    for (std::size_t i = 0; i < g.layers.size(); ++i) {
        if (g.layers[i].name == name) {
            g.layers.erase(g.layers.begin() + static_cast<long>(i));
            g.cycle.erase(std::remove_if(g.cycle.begin(), g.cycle.end(),
                                         [i](const HarmonicGraph::FlowLine& line) {
                                             return line.source.kind == NodeKind::Layer &&
                                                    line.source.index == i;
                                         }),
                          g.cycle.end());
            for (auto& line : g.cycle) {
                line.arrows.erase(std::remove_if(line.arrows.begin(), line.arrows.end(),
                                                 [i](const HarmonicGraph::Arrow& a) {
                                                     return a.target.kind == NodeKind::Layer &&
                                                            a.target.index == i;
                                                 }),
                                  line.arrows.end());
            }
            for (auto& line : g.cycle) {
                if (line.source.kind == NodeKind::Layer && line.source.index > i)
                    --line.source.index;
                for (auto& a : line.arrows)
                    if (a.target.kind == NodeKind::Layer && a.target.index > i)
                        --a.target.index;
            }
            propagate_ratios(g);
            return;
        }
    }
    throw std::runtime_error("unknown layer: " + name);
}

/**
 * Add a forward or backward connection between two nodes.
 * Both nodes must already exist in the graph.
 */
inline void add_flow(HarmonicGraph& g, const std::string& src, const std::string& dst,
                     std::optional<std::string> func = std::nullopt, bool backward = false) {
    NodeId s = g.find(src);
    NodeId d = g.find(dst);
    HarmonicGraph::Arrow arrow{backward, func, d};
    for (auto& line : g.cycle) {
        if (line.source.kind == s.kind && line.source.index == s.index) {
            line.arrows.push_back(arrow);
            return;
        }
    }
    g.cycle.push_back({s, {arrow}});
}

/** Remove the first flow from src to dst matching the direction. */
inline void remove_flow(HarmonicGraph& g, const std::string& src, const std::string& dst,
                        bool backward = false) {
    NodeId s = g.find(src);
    NodeId d = g.find(dst);
    for (auto& line : g.cycle) {
        if (line.source.kind == s.kind && line.source.index == s.index) {
            auto it = std::find_if(line.arrows.begin(), line.arrows.end(),
                                   [d, backward](const HarmonicGraph::Arrow& a) {
                                       return a.target.kind == d.kind &&
                                              a.target.index == d.index && a.backward == backward;
                                   });
            if (it != line.arrows.end()) {
                line.arrows.erase(it);
                return;
            }
        }
    }
}

} // namespace harmonics
