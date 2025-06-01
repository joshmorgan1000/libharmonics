#pragma once

#include "harmonics/quantum_stub.hpp"
#include <cmath>
#include <complex>
#include <vector>

namespace harmonics {

/** Result of simulating a quantum circuit. */
struct QuantumResult {
    std::vector<int> measurements; ///< measurement results in operation order
};

/** Simple state vector simulator for QuantumCircuit. */
inline QuantumResult simulate(const QuantumCircuit& qc) {
    using complex = std::complex<double>;
    QuantumResult result{};
    std::vector<complex> state(1ull << qc.qubit_count, complex{0.0, 0.0});
    state[0] = complex{1.0, 0.0};
    auto apply_x = [&](int q) {
        std::size_t step = 1ull << q;
        for (std::size_t i = 0; i < state.size(); i += step * 2) {
            for (std::size_t j = 0; j < step; ++j)
                std::swap(state[i + j], state[i + j + step]);
        }
    };
    auto apply_h = [&](int q) {
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        std::size_t step = 1ull << q;
        for (std::size_t i = 0; i < state.size(); i += step * 2) {
            for (std::size_t j = 0; j < step; ++j) {
                complex a = state[i + j];
                complex b = state[i + j + step];
                state[i + j] = (a + b) * inv_sqrt2;
                state[i + j + step] = (a - b) * inv_sqrt2;
            }
        }
    };
    auto apply_y = [&](int q) {
        std::size_t step = 1ull << q;
        const complex I{0.0, 1.0};
        for (std::size_t i = 0; i < state.size(); i += step * 2) {
            for (std::size_t j = 0; j < step; ++j) {
                complex a = state[i + j];
                complex b = state[i + j + step];
                state[i + j] = -I * b;
                state[i + j + step] = I * a;
            }
        }
    };
    auto apply_z = [&](int q) {
        std::size_t step = 1ull << q;
        for (std::size_t i = 0; i < state.size(); i += step * 2) {
            for (std::size_t j = 0; j < step; ++j) {
                state[i + j + step] *= -1.0;
            }
        }
    };
    auto apply_cx = [&](int c, int t) {
        std::size_t step_t = 1ull << t;
        std::size_t step_c = 1ull << c;
        for (std::size_t i = 0; i < state.size(); ++i) {
            if ((i & step_c) != 0 && (i & step_t) == 0) {
                std::size_t j = i | step_t;
                std::swap(state[i], state[j]);
            }
        }
    };
    auto measure = [&](int q) {
        std::size_t mask = 1ull << q;
        double p1 = 0.0;
        for (std::size_t i = 0; i < state.size(); ++i) {
            if (i & mask)
                p1 += std::norm(state[i]);
        }
        int bit = p1 > 0.5 ? 1 : 0;
        std::vector<complex> new_state(state.size(), complex{0.0, 0.0});
        for (std::size_t i = 0; i < state.size(); ++i) {
            if (((i & mask) ? 1 : 0) == bit)
                new_state[i] = state[i];
        }
        double norm = 0.0;
        for (auto& c : new_state)
            norm += std::norm(c);
        norm = std::sqrt(norm);
        if (norm != 0.0) {
            for (auto& c : new_state)
                c /= norm;
        }
        state.swap(new_state);
        result.measurements.push_back(bit);
    };

    for (const auto& op : qc.ops) {
        switch (op.gate) {
        case QubitGate::H:
            apply_h(op.qubits[0]);
            break;
        case QubitGate::X:
            apply_x(op.qubits[0]);
            break;
        case QubitGate::Y:
            apply_y(op.qubits[0]);
            break;
        case QubitGate::Z:
            apply_z(op.qubits[0]);
            break;
        case QubitGate::CX:
            apply_cx(op.qubits[0], op.qubits[1]);
            break;
        case QubitGate::Measure:
            measure(op.qubits[0]);
            break;
        }
    }
    return result;
}

} // namespace harmonics
