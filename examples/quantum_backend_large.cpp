#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/quantum_backend.hpp>
#include <harmonics/quantum_stub.hpp>
#include <iostream>

int main() {
    const char* src = R"(
        producer a {2};
        producer b {2};
        producer c {2};
        layer l1;
        layer l2;
        layer l3;
        consumer out {2};
        cycle {
            a -> l1;
            b -> l1;
            c -> l2;
            l1 -> l2;
            l2 -> l3;
            l1 -> out;
            l2 -> out;
            l3 -> out;
        }
    )";

    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto qc = harmonics::map_to_quantum(g);
    auto result = harmonics::simulate(qc);

    std::cout << "Measurements:";
    for (int m : result.measurements)
        std::cout << ' ' << m;
    std::cout << '\n';
}
