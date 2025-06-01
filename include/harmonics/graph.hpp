#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "harmonics/core.hpp"
#include "harmonics/deployment.hpp"
#include "harmonics/parser.hpp"
#include "harmonics/precision_policy.hpp"
#include "harmonics/training.hpp"

namespace harmonics {

// ---------------------------------------------------------------------------
// Graph representation overview
// ---------------------------------------------------------------------------
// The HarmonicGraph structure captures the nodes and edges of a small neural
// network in memory. It is intentionally lightweight so that unit tests can
// easily construct graphs without touching any heavy runtime components. The
// graph is built from parser output and then used to drive the CycleRuntime
// which performs forward and backward passes.
//
// Nodes are divided into three categories:
//  - Producers generate tensors that feed into the graph.
//  - Consumers receive tensors produced by the graph.
//  - Layers transform tensors and may hold trainable parameters.
//
// A simple cycle description defines the flow of data between these nodes. Each
// line in the cycle specifies a source node and a list of arrows leading to
// target nodes. Arrows may optionally be marked as backward which is used to
// implement training taps for gradient propagation.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Reading the graph
// ---------------------------------------------------------------------------
// Graphs are typically built from the domain specific language parsed by the
// Harmonics parser. The parser produces a DeclarationsAST which is then fed into
// build_graph to construct the in-memory representation. Once instantiated the
// graph can be inspected or modified programmatically before being executed by
// a CycleRuntime instance.
//
// Modifying graphs directly is intentionally straightforward. Producers,
// consumers and layers are stored in plain vectors allowing callers to add or
// remove entries as needed. The cycle vector describes the data flow in a very
// compact form and can likewise be edited by hand when constructing custom test
// scenarios.
//
// During execution the runtime relies on node indices for efficiency. The
// convenience function `find` is provided to map human readable names to these
// indices. Helper functions in `graph_edit.hpp` build upon this low level API to
// offer higher level editing commands.
//
// A typical workflow is:
//   1. Parse source using `Parser` and obtain a `DeclarationsAST`.
//   2. Call `build_graph` to get a `HarmonicGraph`.
//   3. Optionally edit the graph using helper functions.
//   4. Bind producers via `bindProducer`.
//   5. Create a `CycleRuntime` for execution.
//
// Keeping the representation simple keeps compile times short and ensures the
// library remains easy to reason about during testing and experimentation.
// ---------------------------------------------------------------------------

struct CycleState;
class CycleRuntime;

/**
 * Type of a named node in a HarmonicGraph.
 *
 * The enumeration is used throughout the runtime to quickly identify what kind
 * of operation a node represents without performing expensive dynamic casts.
 * Each category corresponds to one of the vectors stored in the graph
 * structure.
 */
enum class NodeKind { Producer, Consumer, Layer };

/** Identifier referencing a graph node. */
struct NodeId {
    NodeKind kind;     ///< Which node vector the index refers to
    std::size_t index; ///< Position within that vector
    /// Convenience comparison operator used throughout the runtime.
    bool operator==(const NodeId& other) const {
        return kind == other.kind && index == other.index;
    }
};

/**
 * In-memory representation of a harmonic network.
 *
 * The graph stores producers, consumers and layers along with a cycle
 * describing how data flows between them.
 */
struct HarmonicGraph {
    /** Description of a producer node in the graph. */
    struct Producer {
        std::string name;           ///< Unique identifier used in the DSL
        std::optional<int> shape;   ///< Optional width specified by the user
        std::optional<Ratio> ratio; ///< Optional ratio reference to resolve shape
    };
    /** Description of a consumer node in the graph. */
    struct Consumer {
        std::string name;         ///< Identifier matching a graph output
        std::optional<int> shape; ///< Expected width or empty if unspecified
    };
    /** Definition of a layer node. */
    struct Layer {
        std::string name;           ///< Layer name used for lookups
        std::optional<Ratio> ratio; ///< Optional ratio controlling width
        std::optional<int> shape;   ///< Width resolved from ratio propagation
    };

    /** Connection between two nodes in the cycle. */
    struct Arrow {
        bool backward{false};              ///< True if this arrow carries gradients
        std::optional<std::string> func{}; ///< Optional activation or loss function name
        NodeId target{};                   ///< Destination node in the graph
    };
    /** Source node and its outgoing arrows. */
    struct FlowLine {
        NodeId source{};             ///< Originating node
        std::vector<Arrow> arrows{}; ///< Outgoing connections from the source
    };

    std::vector<Producer> producers{}; ///< Ordered list of producer nodes
    std::vector<Consumer> consumers{}; ///< Ordered list of consumer nodes
    std::vector<Layer> layers{};       ///< All layer nodes in the network
    std::vector<FlowLine> cycle{};     ///< Cycle describing data flow between nodes
    std::vector<std::shared_ptr<harmonics::Producer>>
        producer_bindings{}; ///< Runtime bindings for producers

    NodeId find(const std::string& name) const;
    void bindProducer(const std::string& name, std::shared_ptr<harmonics::Producer> prod);
    /** Return true if any flow line contains a backward arrow. */
    bool hasTrainingTaps() const;

    CycleState inference(const DeploymentDescriptor& deploy = {},
                         std::shared_ptr<PrecisionPolicy> policy = nullptr) const;

    template <class Rep, class Period>
    CycleState fit(std::chrono::duration<Rep, Period> duration,
                   std::shared_ptr<PrecisionPolicy> policy, FitOptions options = {},
                   const DeploymentDescriptor& deploy = {}) const;

    /** Train for a fixed number of epochs. */
    CycleState fit(std::size_t epochs, std::shared_ptr<PrecisionPolicy> policy,
                   FitOptions options = {}, const DeploymentDescriptor& deploy = {}) const;

    /**
     * Train until a stopping predicate returns true.
     *
     * The predicate is called with the current CycleState after each forward
     * pass. Training always runs at least once even if the predicate returns
     * true initially.
     */
    template <class StopPredicate>
    CycleState fit_until(StopPredicate stop, std::shared_ptr<PrecisionPolicy> policy,
                         FitOptions options = {}, const DeploymentDescriptor& deploy = {}) const;
};

/** Build a HarmonicGraph from the given declarations AST. */
HarmonicGraph build_graph(const DeclarationsAST& ast);
/** Resolve widths for producers and layers based on ratio references. */
void propagate_ratios(HarmonicGraph& g);

// Lookup a node identifier by name. Throws if the name does not correspond to a
// producer, consumer or layer in the graph.
inline NodeId HarmonicGraph::find(const std::string& name) const {
    for (std::size_t i = 0; i < producers.size(); ++i)
        if (producers[i].name == name)
            return {NodeKind::Producer, i};
    for (std::size_t i = 0; i < consumers.size(); ++i)
        if (consumers[i].name == name)
            return {NodeKind::Consumer, i};
    for (std::size_t i = 0; i < layers.size(); ++i)
        if (layers[i].name == name)
            return {NodeKind::Layer, i};
    throw std::runtime_error("unknown node: " + name);
}

// Associate a runtime Producer with a producer node by name. The optional shape
// check ensures the bound producer emits tensors with the expected width if the
// graph specified one.
inline void HarmonicGraph::bindProducer(const std::string& name,
                                        std::shared_ptr<harmonics::Producer> prod) {
    NodeId id = find(name);
    if (id.kind != NodeKind::Producer)
        throw std::runtime_error(name + " is not a producer");

    if (producers[id.index].shape) {
        auto sample = prod->next();
        const auto& shape = sample.shape();
        if (shape.empty() || shape[0] != static_cast<std::size_t>(*producers[id.index].shape)) {
            throw std::runtime_error("producer shape mismatch for " + name);
        }
    }

    if (id.index >= producer_bindings.size())
        producer_bindings.resize(id.index + 1);
    // Store the producer so it can supply data during execution.
    producer_bindings[id.index] = std::move(prod);
}

// Check whether any arrow in the cycle is marked as backward which indicates a
// training tap. This information is used to decide whether gradient computation
// is required during execution.
inline bool HarmonicGraph::hasTrainingTaps() const {
    for (const auto& line : cycle) {
        for (const auto& arrow : line.arrows) {
            if (arrow.backward)
                return true;
        }
    }
    // No backward arrows found; the graph contains only inference flows.
    return false;
}

// ---------------------------------------------------------------------------
// build_graph
// ---------------------------------------------------------------------------
// Convert a parsed declaration AST into a fully formed HarmonicGraph. This
// routine performs minimal validation of the AST and resolves node references in
// the cycle description. Ratio propagation is executed afterwards to compute
// missing tensor widths.
inline HarmonicGraph build_graph(const DeclarationsAST& ast) {
    HarmonicGraph g;
    std::unordered_map<std::string, NodeKind> names;
    // Mapping from node names to their kind. Used to detect duplicate
    // identifiers when constructing the graph.

    auto add_name = [&names](const std::string& n, NodeKind k) {
        auto it = names.find(n);
        if (it != names.end())
            throw std::runtime_error("duplicate name: " + n);
        names.emplace(n, k);
    };

    // Populate producers from the AST. Duplicate names are rejected so that all
    // nodes can be uniquely addressed later on.
    for (const auto& p : ast.producers) {
        add_name(p.name, NodeKind::Producer);
        g.producers.push_back({p.name, p.shape, p.ratio});
    }
    // Consumers merely record their optional shape information.
    for (const auto& c : ast.consumers) {
        add_name(c.name, NodeKind::Consumer);
        g.consumers.push_back({c.name, c.shape});
    }
    // Layers may reference ratios which are resolved later once all node widths
    // are known.
    for (const auto& l : ast.layers) {
        add_name(l.name, NodeKind::Layer);
        g.layers.push_back({l.name, l.ratio});
    }

    // Allocate space for producer runtime bindings. Each graph producer may be
    // bound to a concrete Producer instance prior to execution.
    g.producer_bindings.resize(g.producers.size());

    if (ast.cycle) {
        // Translate the AST cycle representation into runtime node indices.
        for (const auto& line : ast.cycle->lines) {
            HarmonicGraph::FlowLine fl;
            fl.source = g.find(line.source);
            for (const auto& a : line.arrows) {
                HarmonicGraph::Arrow ar;
                ar.backward = a.backward;
                ar.func = a.func;
                ar.target = g.find(a.target);
                fl.arrows.push_back(ar);
            }
            g.cycle.push_back(fl);
        }
    }

    propagate_ratios(g);
    return g;
}

// ---------------------------------------------------------------------------
// propagate_ratios
// ---------------------------------------------------------------------------
// Resolve tensor widths for producers and layers that reference ratios. The
// algorithm iteratively propagates known widths through the ratio references
// until no further progress can be made.
inline void propagate_ratios(HarmonicGraph& g) {
    std::unordered_map<std::string, int> widths;
    for (auto& p : g.producers)
        if (p.shape)
            widths[p.name] = *p.shape;
    for (auto& l : g.layers)
        if (l.shape)
            widths[l.name] = *l.shape;

    bool progress = true;
    // Iterate until no further shapes can be resolved. Each pass may
    // compute new widths based on ratio references discovered in the
    // previous iteration.
    while (progress) {
        progress = false;
        // Propagate known widths to producers referencing ratios.
        for (auto& p : g.producers) {
            if (!p.shape && p.ratio) {
                auto it = widths.find(p.ratio->ref);
                if (it != widths.end()) {
                    p.shape = (it->second * p.ratio->lhs) / p.ratio->rhs;
                    widths[p.name] = *p.shape;
                    progress = true;
                }
            }
        }
        // Do the same for layers that reference other node widths.
        for (auto& l : g.layers) {
            if (!l.shape && l.ratio) {
                auto it = widths.find(l.ratio->ref);
                if (it != widths.end()) {
                    l.shape = (it->second * l.ratio->lhs) / l.ratio->rhs;
                    widths[l.name] = *l.shape;
                    progress = true;
                }
            }
        }
    }
}

inline std::string graph_digest(const HarmonicGraph& g) {
    std::ostringstream ss;
    for (const auto& p : g.producers) {
        ss << p.name << ':';
        if (p.shape)
            ss << *p.shape;
        ss << '|';
        if (p.ratio)
            ss << p.ratio->lhs << '/' << p.ratio->rhs << ':' << p.ratio->ref;
        ss << ';';
    }
    ss << '#';
    for (const auto& c : g.consumers) {
        ss << c.name << ':';
        if (c.shape)
            ss << *c.shape;
        ss << ';';
    }
    ss << '#';
    for (const auto& l : g.layers) {
        ss << l.name << ':';
        if (l.ratio)
            ss << l.ratio->lhs << '/' << l.ratio->rhs << ':' << l.ratio->ref;
        ss << '|';
        if (l.shape)
            ss << *l.shape;
        ss << ';';
    }
    ss << '#';
    for (const auto& fl : g.cycle) {
        ss << static_cast<int>(fl.source.kind) << ':' << fl.source.index << '>';
        for (const auto& ar : fl.arrows) {
            if (ar.backward)
                ss << 'b';
            if (ar.func)
                ss << *ar.func;
            ss << ':' << static_cast<int>(ar.target.kind) << ':' << ar.target.index << ',';
        }
        ss << ';';
    }
    auto data = ss.str();
    return blake3(data.data(), data.size());
}

} // namespace harmonics
