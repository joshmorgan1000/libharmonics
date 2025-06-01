#include <fstream>
#include <iostream>
#include <string>

#include <harmonics/dot_export.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/serialization.hpp>

using namespace harmonics;

namespace {
void usage() { std::cerr << "Usage: graph_info <graph>\n"; }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string path = argv[1];
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "failed to open graph file\n";
        return 1;
    }
    auto g = load_graph(in);

    for (const auto& p : g.producers)
        std::cout << "producer " << p.name << '\n';
    for (const auto& l : g.layers)
        std::cout << "layer " << l.name << '\n';
    for (const auto& c : g.consumers)
        std::cout << "consumer " << c.name << '\n';

    for (const auto& line : g.cycle) {
        std::string src = node_name(g, line.source);
        for (const auto& ar : line.arrows) {
            std::string dst = node_name(g, ar.target);
            std::cout << src << (ar.backward ? " ~> " : " -> ") << dst;
            if (ar.func)
                std::cout << " (" << *ar.func << ")";
            std::cout << '\n';
        }
    }
    return 0;
}
