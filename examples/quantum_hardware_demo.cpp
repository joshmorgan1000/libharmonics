#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/quantum_hardware.hpp>
#include <harmonics/quantum_stub.hpp>
#include <iostream>

int main() {
    const char* src = R"(
        producer p {1};
        consumer c {1};
        layer l;
        cycle { p -> l; l -> c; }
    )";

    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto qc = harmonics::map_to_quantum(g);
    auto result = harmonics::execute_on_hardware(qc);

    std::cout << "Measurements:";
    for (int m : result.measurements)
        std::cout << ' ' << m;
    std::cout << '\n';
}
