#pragma once

#include "harmonics/graph.hpp"
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>

namespace harmonics {

/**
 * Partition a HarmonicGraph into two subgraphs at the given layer index.
 *
 * Layers with index lower than \p split remain in the first graph while the
 * rest are placed in the second graph. Cross-partition flows are replaced by
 * automatically created boundary producers/consumers so that tensors can be
 * forwarded by an embedding transport layer.
 */
inline std::pair<HarmonicGraph, HarmonicGraph> partition_by_layer(const HarmonicGraph& g,
                                                                  std::size_t split) {
    if (split > g.layers.size())
        throw std::runtime_error("split index out of range");

    HarmonicGraph first, second;
    first.producers = g.producers;
    first.consumers = g.consumers;
    first.layers.assign(g.layers.begin(), g.layers.begin() + split);
    first.producer_bindings.resize(first.producers.size());

    second.producers = g.producers;
    second.consumers = g.consumers;
    second.layers.assign(g.layers.begin() + split, g.layers.end());
    second.producer_bindings.resize(second.producers.size());

    auto is_first = [&](NodeId id) { return id.kind != NodeKind::Layer || id.index < split; };
    auto is_second = [&](NodeId id) { return id.kind != NodeKind::Layer || id.index >= split; };
    auto map_first = [&](NodeId id) {
        if (id.kind == NodeKind::Layer)
            return NodeId{NodeKind::Layer, id.index};
        return id;
    };
    auto map_second = [&](NodeId id) {
        if (id.kind == NodeKind::Layer)
            return NodeId{NodeKind::Layer, id.index - split};
        return id;
    };

    std::size_t boundary_count = 0;

    for (const auto& line : g.cycle) {
        bool src1 = is_first(line.source);
        bool src2 = is_second(line.source);

        HarmonicGraph::FlowLine fl1, fl2;
        if (src1)
            fl1.source = map_first(line.source);
        if (src2)
            fl2.source = map_second(line.source);

        for (const auto& ar : line.arrows) {
            bool dst1 = is_first(ar.target);
            bool dst2 = is_second(ar.target);

            if (src1 && dst1)
                fl1.arrows.push_back({ar.backward, ar.func, map_first(ar.target)});
            else if (src2 && dst2)
                fl2.arrows.push_back({ar.backward, ar.func, map_second(ar.target)});
            else if (src1 && dst2) {
                std::string name = "boundary" + std::to_string(boundary_count++);
                std::size_t ci = first.consumers.size();
                first.consumers.push_back({name, std::nullopt});
                std::size_t pi = second.producers.size();
                second.producers.push_back({name, std::nullopt, std::nullopt});
                fl1.arrows.push_back({ar.backward, ar.func, {NodeKind::Consumer, ci}});
                HarmonicGraph::FlowLine bl;
                bl.source = {NodeKind::Producer, pi};
                bl.arrows.push_back({ar.backward, ar.func, map_second(ar.target)});
                second.cycle.push_back(bl);
            } else if (src2 && dst1) {
                std::string name = "boundary" + std::to_string(boundary_count++);
                std::size_t ci = second.consumers.size();
                second.consumers.push_back({name, std::nullopt});
                std::size_t pi = first.producers.size();
                first.producers.push_back({name, std::nullopt, std::nullopt});
                fl2.arrows.push_back({ar.backward, ar.func, {NodeKind::Consumer, ci}});
                HarmonicGraph::FlowLine bl;
                bl.source = {NodeKind::Producer, pi};
                bl.arrows.push_back({ar.backward, ar.func, map_first(ar.target)});
                first.cycle.push_back(bl);
            }
        }

        if (src1 && !fl1.arrows.empty())
            first.cycle.push_back(fl1);
        if (src2 && !fl2.arrows.empty())
            second.cycle.push_back(fl2);
    }

    propagate_ratios(first);
    propagate_ratios(second);
    return {first, second};
}

/**
 * Automatically partition a graph into multiple subgraphs.
 *
 * The graph is split evenly by layer count across the number of
 * partitions in the deployment descriptor. If no partitions are
 * specified, the original graph is returned.
 */
inline std::vector<HarmonicGraph> auto_partition(const HarmonicGraph& g,
                                                 const DeploymentDescriptor& deploy) {
    std::size_t parts = deploy.partitions.empty() ? 1 : deploy.partitions.size();
    if (parts <= 1)
        return {g};

    auto backend_weight = [](Backend b) {
        switch (b) {
        case Backend::GPU:
            return 4.0;
        case Backend::FPGA:
            return 2.0;
        default:
            return 1.0;
        }
    };

    std::vector<double> weights(parts, 1.0);
    for (std::size_t i = 0; i < parts; ++i)
        weights[i] = backend_weight(deploy.partitions[i].backend);

    double total = 0.0;
    for (double w : weights)
        total += w;

    std::vector<HarmonicGraph> out;
    out.reserve(parts);

    HarmonicGraph remaining = g;
    std::size_t prev = 0;
    double accum = 0.0;
    for (std::size_t i = 0; i + 1 < parts; ++i) {
        accum += weights[i];
        std::size_t split = static_cast<std::size_t>(
            std::llround((accum / total) * static_cast<double>(g.layers.size())));
        auto res = partition_by_layer(remaining, split - prev);
        out.push_back(std::move(res.first));
        remaining = std::move(res.second);
        prev = split;
    }
    out.push_back(std::move(remaining));
    return out;
}

} // namespace harmonics
