#pragma once

#include "harmonics/graph.hpp"
#include <vector>

namespace harmonics {

/** Simple enumeration of quantum gates used by the stub. */
enum class QubitGate { H, X, Y, Z, CX, Measure };

/** One instruction in a quantum circuit. */
struct QuantumOp {
    QubitGate gate{};
    std::vector<int> qubits{};
};

/** Minimal quantum circuit representation produced by the stub. */
struct QuantumCircuit {
    int qubit_count{0};
    std::vector<QuantumOp> ops{};
};

/**
 * Build a quantum circuit from a HarmonicGraph.
 *
 * Each node in the graph corresponds to a qubit. Flow arrows are mapped
 * to controlled-X gates and consumer nodes are measured at the end.
 */
inline QuantumCircuit map_to_quantum(const HarmonicGraph& g) {
    QuantumCircuit qc;
    int next_qubit = 0;
    std::vector<int> prod_q(g.producers.size(), -1);
    std::vector<int> cons_q(g.consumers.size(), -1);
    std::vector<int> layer_q(g.layers.size(), -1);

    auto assign = [&](NodeId id) -> int {
        int* arr = nullptr;
        switch (id.kind) {
        case NodeKind::Producer:
            arr = prod_q.data();
            break;
        case NodeKind::Consumer:
            arr = cons_q.data();
            break;
        case NodeKind::Layer:
            arr = layer_q.data();
            break;
        }
        int& q = arr[id.index];
        if (q == -1)
            q = next_qubit++;
        return q;
    };

    for (std::size_t i = 0; i < g.producers.size(); ++i)
        assign({NodeKind::Producer, i});
    for (std::size_t i = 0; i < g.consumers.size(); ++i)
        assign({NodeKind::Consumer, i});
    for (std::size_t i = 0; i < g.layers.size(); ++i)
        assign({NodeKind::Layer, i});

    qc.qubit_count = next_qubit;

    for (const auto& line : g.cycle) {
        int src = assign(line.source);
        for (const auto& ar : line.arrows) {
            int dst = assign(ar.target);
            qc.ops.push_back({QubitGate::CX, {src, dst}});
        }
    }

    for (std::size_t i = 0; i < g.consumers.size(); ++i) {
        int q = cons_q[i];
        qc.ops.push_back({QubitGate::Measure, {q}});
    }

    return qc;
}

} // namespace harmonics
