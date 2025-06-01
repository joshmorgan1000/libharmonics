#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <iostream>

int main() {
    const char* src = "producer a; consumer b;";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto graph = harmonics::build_graph(ast);
    std::cout << "Producers: " << graph.producers.size() << '\n';
    std::cout << "Consumers: " << graph.consumers.size() << '\n';
}
