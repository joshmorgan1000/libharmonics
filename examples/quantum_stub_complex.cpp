#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/quantum_stub.hpp>
#include <iostream>

int main() {
    const char* src = R"(
        producer a {2};
        producer b {2};
        producer c {2};
        producer d {2};
        layer l1;
        layer l2;
        layer l3;
        layer l4;
        layer l5;
        layer l6;
        consumer out {2};
        cycle {
            a -> l1;
            b -> l1;
            c -> l2;
            d -> l3;
            l1 -> l2;
            l2 -> l3;
            l3 -> l4;
            l2 -> l4;
            l4 -> l5;
            l1 -> l5;
            l5 -> l6;
            l3 -> l6;
            l6 -> out;
        }
    )";

    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto qc = harmonics::map_to_quantum(g);

    std::cout << "Qubits: " << qc.qubit_count << '\n';
    for (const auto& op : qc.ops) {
        switch (op.gate) {
        case harmonics::QubitGate::H:
            std::cout << "H";
            break;
        case harmonics::QubitGate::X:
            std::cout << "X";
            break;
        case harmonics::QubitGate::Y:
            std::cout << "Y";
            break;
        case harmonics::QubitGate::Z:
            std::cout << "Z";
            break;
        case harmonics::QubitGate::CX:
            std::cout << "CX";
            break;
        case harmonics::QubitGate::Measure:
            std::cout << "Measure";
            break;
        }
        for (int q : op.qubits)
            std::cout << ' ' << q;
        std::cout << '\n';
    }
}
