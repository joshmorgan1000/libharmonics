#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/precision_policy.hpp>
#include <harmonics/serialization.hpp>
#include <harmonics/simple_io.hpp>
#define HARMONICS_PLUGIN_IMPL
#include <harmonics/plugin.hpp>

// ---------------------------------------------------------------------------
// Harmonics command line tool
// ---------------------------------------------------------------------------
// Provides a minimal entry point for compiling and running graphs. The design
// favours simplicity so that the overall architecture of the library can be
// understood without wading through boilerplate. When run in compile mode the
// input source is parsed and saved to a binary file. In run mode that binary is
// executed with dummy producers and the resulting proof string printed if
// secure mode is enabled.
// ---------------------------------------------------------------------------

using namespace harmonics;

namespace {
/**
 * @brief Aliases for simple producer/consumer helpers.
 *
 * These are provided by @ref simple_io.hpp and emit constant tensors or
 * discard all outputs.
 */
using harmonics::make_zero_producer;
using DiscardConsumer = harmonics::DiscardConsumer;

void compile_file(const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path);
    if (!in)
        throw std::runtime_error("failed to open input file");
    std::string src((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // Parse the source into an AST and then build a graph object which
    // can be serialized for later execution.
    Parser parser{src.c_str()};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    std::ofstream out(out_path, std::ios::binary);
    if (!out)
        throw std::runtime_error("failed to open output file");
    save_graph(g, out); // binary format keeps files small
}

void run_graph(const std::string& path, bool secure, int bits) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open graph file");
    auto g = load_graph(in);
    for (std::size_t i = 0; i < g.producers.size(); ++i) {
        std::size_t w = g.producers[i].shape ? static_cast<std::size_t>(*g.producers[i].shape) : 1;
        g.bindProducer(g.producers[i].name, make_zero_producer(w));
    }
    DeploymentDescriptor desc;
    desc.secure = secure; // toggle zero-knowledge proof generation
    CycleRuntime rt{g, make_max_bits_policy(bits), desc};
    rt.forward(); // execute once using the dummy producers
    if (secure)
        std::cout << "proof: " << rt.proof() << '\n';
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: harmonics --compile <src> [-o out] | --run <graph> [--secure] [--bits "
                     "n] [--plugin-path path]\n";
        return 1;
    }
    std::string plugin_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--plugin-path" && i + 1 < argc) {
            // Additional plugins can be loaded from a custom directory
            // before executing any commands.
            plugin_path = argv[++i];
        }
    }

    // Load optional plugins so custom layers can be used when compiling
    // or executing graphs. The return value is ignored but kept for
    // completeness should diagnostics be added in the future.
    auto loaded = harmonics::load_plugins_from_path(plugin_path);

    std::string cmd = argv[1];
    try {
        if (cmd == "--compile") {
            std::string out = "graph.hgr";
            std::string in = argv[2];
            for (int i = 3; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "-o" && i + 1 < argc) {
                    out = argv[++i];
                } else if (arg == "--plugin-path") {
                    ++i;
                }
            }
            // Translate the source file into a binary graph representation.
            compile_file(in, out);
            return 0;
        } else if (cmd == "--run") {
            bool secure = false;
            int bits = 32;
            std::string path = argv[2];
            for (int i = 3; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--secure") {
                    secure = true;
                } else if (arg == "--bits" && i + 1 < argc) {
                    bits = std::stoi(argv[++i]);
                } else if (arg == "--plugin-path") {
                    ++i;
                }
            }
            // Execute the previously compiled graph with dummy inputs.
            run_graph(path, secure, bits);
            return 0;
        } else {
            std::cerr << "Unknown command\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
