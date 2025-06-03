#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "harmonics/graph.hpp"
#include "harmonics/graph_edit.hpp"

namespace harmonics {

/**
 * @brief Information about edits between two graphs.
 *
 * GraphDiff captures structural changes required to transform one
 * HarmonicGraph into another. The diff can be computed with
 * @ref diff_graphs and later applied using @ref apply_diff. Each edit
 * is represented either as a layer addition/removal or as a flow change
 * between nodes.
 */
struct GraphDiff {
    /**
     * @brief Representation of a single edge in the graph.
     *
     * A flow describes an arrow from @p src to @p dst. When @p backward is
     * true the arrow carries gradients during backpropagation. Optional
     * @p func stores the activation or loss function associated with the edge
     * if one exists.
     */
    struct Flow {
        std::string src;                   ///< Name of the source node
        std::string dst;                   ///< Name of the destination node
        bool backward{false};              ///< Indicates a backward edge
        std::optional<std::string> func{}; ///< Activation or loss function

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

/**
 * @brief Compute edits required to transform one graph into another.
 *
 * The function walks both graphs collecting added and removed layers as well
 * as flow changes. The resulting GraphDiff can later be applied using
 * @ref apply_diff. Only simple structural differences are detected; changes to
 * tensor shapes or other metadata are ignored.
 */
inline GraphDiff diff_graphs(const HarmonicGraph& before, const HarmonicGraph& after) {
    GraphDiff diff;

    // Record the layer names present in each graph so we can detect additions
    // and removals by simple set difference operations.
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

    // Helper converting cycle entries into a stable list of flows. Using names
    // rather than indices keeps the diff readable and independent of ordering.
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

    // Build flow lists for both graphs and then compute simple set differences
    // based on a unique textual key. This avoids O(n^2) comparisons.
    auto before_flows = collect_flows(before);
    auto after_flows = collect_flows(after);
    // Store textual representations of all flows from the first graph.
    std::unordered_set<std::string> before_set;
    auto flow_key = [](const GraphDiff::Flow& f) {
        return f.src + "->" + f.dst + (f.backward ? "b" : "f") + (f.func ? *f.func : "");
    };
    for (const auto& f : before_flows)
        before_set.insert(flow_key(f));
    // Same for the second graph so we can quickly check membership.
    std::unordered_set<std::string> after_set;
    for (const auto& f : after_flows)
        after_set.insert(flow_key(f));

    // Compare the two sets to determine added and removed flows.
    for (const auto& f : after_flows)
        if (!before_set.count(flow_key(f)))
            diff.added_flows.push_back(f);
    for (const auto& f : before_flows)
        if (!after_set.count(flow_key(f)))
            diff.removed_flows.push_back(f);

    return diff;
}

/**
 * @brief Apply the edits contained in a GraphDiff to a graph.
 *
 * Layers and flows referenced by the diff are added or removed from @p g. The
 * operation mutates the graph in place and performs no validation beyond what
 * the helper functions in @ref graph_edit provide. Callers should ensure that
 * the diff was generated from a compatible graph.
 */
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

/**
 * @brief Create a new graph by merging @p update into @p base.
 *
 * This is a convenience wrapper combining @ref diff_graphs and @ref apply_diff.
 * It first computes the diff between the two graphs and then applies it to a
 * copy of @p base. The original graphs remain unmodified.
 */
inline HarmonicGraph merge_graphs(const HarmonicGraph& base, const HarmonicGraph& update) {
    HarmonicGraph result = base;
    auto diff = diff_graphs(base, update);
    apply_diff(result, diff);
    return result;
}

} // namespace harmonics
