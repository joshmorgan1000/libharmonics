#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <harmonics/dot_export.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/graph_edit.hpp>
#include <harmonics/introspection.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/precision_policy.hpp>
#include <harmonics/serialization.hpp>
#include <harmonics/simple_io.hpp>

using namespace harmonics;

namespace {
using harmonics::make_zero_producer;

void usage() { std::cerr << "Usage: graph_debugger <graph> [--bits n]\n"; }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string path = argv[1];
    int bits = 32;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bits" && i + 1 < argc)
            bits = std::stoi(argv[++i]);
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "failed to open graph file\n";
        return 1;
    }
    auto g = load_graph(in);

    // Bind zero producers so the graph can execute.
    for (const auto& p : g.producers) {
        std::size_t w = p.shape ? static_cast<std::size_t>(*p.shape) : 1;
        g.bindProducer(p.name, make_zero_producer(w));
    }

    CycleRuntime rt{g, make_max_bits_policy(bits)};
    rt.set_debug_callback([&](NodeId src, NodeId dst, const HTensor& t, bool back,
                              const std::optional<std::string>& func) {
        std::cout << node_name(g, src) << (back ? " ~> " : " -> ") << node_name(g, dst);
        if (func)
            std::cout << " (" << *func << ")";
        std::cout << " [";
        for (std::size_t i = 0; i < t.shape().size(); ++i) {
            if (i)
                std::cout << 'x';
            std::cout << t.shape()[i];
        }
        std::cout << "]\n";
        std::cout << "Press Enter to continue..." << std::flush;
        std::cin.get();
    });

    rt.forward();
    return 0;
}
