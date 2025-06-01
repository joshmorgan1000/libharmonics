#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "harmonics/graph.hpp"
#include "harmonics/graph_edit.hpp"

namespace harmonics {

/** Information about edits between two graphs. */
struct GraphDiff {
    struct Flow {
        std::string src;
        std::string dst;
        bool backward{false};
        std::optional<std::string> func{};
        bool operator==(const Flow& other) const {
            return src == other.src && dst == other.dst && backward == other.backward &&
                   func == other.func;
        }
    };

    std::vector<HarmonicGraph::Layer> added_layers{}; ///< Layers present only in the new graph
    std::vector<std::string> removed_layers{};        ///< Layers removed from the old graph
    std::vector<Flow> added_flows{};                  ///< Flows present only in the new graph
    std::vector<Flow> removed_flows{};                ///< Flows removed from the old graph
};

/** Compute a diff transforming @p before into @p after. */
inline GraphDiff diff_graphs(const HarmonicGraph& before, const HarmonicGraph& after) {
    GraphDiff diff;

    std::unordered_set<std::string> before_layers;
    for (const auto& l : before.layers)
        before_layers.insert(l.name);
    std::unordered_set<std::string> after_layers;
    for (const auto& l : after.layers)
        after_layers.insert(l.name);

    for (const auto& l : after.layers)
        if (!before_layers.count(l.name))
            diff.added_layers.push_back(l);
    for (const auto& l : before.layers)
        if (!after_layers.count(l.name))
            diff.removed_layers.push_back(l.name);

    auto collect_flows = [](const HarmonicGraph& g) {
        std::vector<GraphDiff::Flow> flows;
        for (const auto& fl : g.cycle) {
            std::string src;
            switch (fl.source.kind) {
            case NodeKind::Producer:
                src = g.producers[fl.source.index].name;
                break;
            case NodeKind::Consumer:
                src = g.consumers[fl.source.index].name;
                break;
            case NodeKind::Layer:
                src = g.layers[fl.source.index].name;
                break;
            }
            for (const auto& ar : fl.arrows) {
                std::string dst;
                switch (ar.target.kind) {
                case NodeKind::Producer:
                    dst = g.producers[ar.target.index].name;
                    break;
                case NodeKind::Consumer:
                    dst = g.consumers[ar.target.index].name;
                    break;
                case NodeKind::Layer:
                    dst = g.layers[ar.target.index].name;
                    break;
                }
                flows.push_back({src, dst, ar.backward, ar.func});
            }
        }
        return flows;
    };

    auto before_flows = collect_flows(before);
    auto after_flows = collect_flows(after);
    std::unordered_set<std::string> before_set;
    auto flow_key = [](const GraphDiff::Flow& f) {
        return f.src + "->" + f.dst + (f.backward ? "b" : "f") + (f.func ? *f.func : "");
    };
    for (const auto& f : before_flows)
        before_set.insert(flow_key(f));
    std::unordered_set<std::string> after_set;
    for (const auto& f : after_flows)
        after_set.insert(flow_key(f));

    for (const auto& f : after_flows)
        if (!before_set.count(flow_key(f)))
            diff.added_flows.push_back(f);
    for (const auto& f : before_flows)
        if (!after_set.count(flow_key(f)))
            diff.removed_flows.push_back(f);

    return diff;
}

/** Apply a diff to modify @p g in place. */
inline void apply_diff(HarmonicGraph& g, const GraphDiff& diff) {
    for (const auto& f : diff.removed_flows)
        remove_flow(g, f.src, f.dst, f.backward);
    for (const auto& name : diff.removed_layers)
        remove_layer(g, name);
    for (const auto& l : diff.added_layers)
        add_layer(g, l.name, l.ratio);
    for (const auto& f : diff.added_flows)
        add_flow(g, f.src, f.dst, f.func, f.backward);
}

/** Merge @p base with @p update and return the resulting graph. */
inline HarmonicGraph merge_graphs(const HarmonicGraph& base, const HarmonicGraph& update) {
    HarmonicGraph result = base;
    auto diff = diff_graphs(base, update);
    apply_diff(result, diff);
    return result;
}

} // namespace harmonics
