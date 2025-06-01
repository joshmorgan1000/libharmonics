#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <harmonics/dot_export.hpp>
#include <harmonics/graph_edit.hpp>
#include <harmonics/introspection.hpp>
#include <harmonics/precision_policy.hpp>
#include <harmonics/serialization.hpp>
#include <harmonics/simple_io.hpp>

using namespace harmonics;

// ---------------------------------------------------------------------------
// Graph CLI overview
// ---------------------------------------------------------------------------
// This utility provides a collection of basic commands for inspecting and
// modifying graph files produced by the parser. It is deliberately minimal
// and avoids depending on any third party libraries so that it can be used as
// a reference implementation. The code is written in a very imperative style
// to make the control flow easy to follow when stepping through in a
// debugger. Each command manipulates the graph in place and then writes the
// result back out when requested.
// ---------------------------------------------------------------------------

namespace {
/**
 * @brief Alias for the zero producer helper.
 */
using harmonics::make_zero_producer;

void usage() {
    std::cerr << "Usage: graph_cli <command> <graph> [args] [-o out] [--bits n]\n";
    std::cerr << "Commands:\n";
    std::cerr << "  info\n";
    std::cerr << "  add-layer <name>\n";
    std::cerr << "  remove-layer <name>\n";
    std::cerr << "  add-flow <src> <dst> [--backward]\n";
    std::cerr << "  remove-flow <src> <dst> [--backward]\n";
    std::cerr << "  dot [-o out.dot]\n";
    std::cerr << "  batch <script> [-o out]\n";
    std::cerr << "  interactive [-o out]\n";
}

void bind_zero_producers(HarmonicGraph& g) {
    for (const auto& p : g.producers) {
        std::size_t w = p.shape ? static_cast<std::size_t>(*p.shape) : 1;
        g.bindProducer(p.name, make_zero_producer(w));
    }
}

void save_graph_file(const HarmonicGraph& g, const std::string& out_path) {
    std::ofstream out(out_path, std::ios::binary);
    if (!out)
        throw std::runtime_error("failed to open output file");
    save_graph(g, out);
}

int run_batch(HarmonicGraph& g, const std::string& script_path, const std::string& out_path) {
    std::ifstream script(script_path);
    if (!script) {
        std::cerr << "failed to open script file\n";
        return 1;
    }
    std::vector<HarmonicGraph> undo_stack;
    std::vector<HarmonicGraph> redo_stack;
    std::string line;
    // Helper capturing the current graph state so that editing commands
    // can be undone. Any time a mutation is performed we snapshot the
    // graph and clear the redo history.
    auto push_undo = [&](const HarmonicGraph& cur) {
        undo_stack.push_back(cur);
        redo_stack.clear();
    };
    while (std::getline(script, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tok;
        std::string t;
        while (iss >> t)
            tok.push_back(t);
        // Skip blank lines and comments to make scripts easier to read.
        if (tok.empty() || tok[0].front() == '#')
            continue;
        const std::string& c = tok[0];
        if (c == "undo") {
            if (!undo_stack.empty()) {
                redo_stack.push_back(g);
                g = undo_stack.back();
                undo_stack.pop_back();
            }
        } else if (c == "redo") {
            if (!redo_stack.empty()) {
                undo_stack.push_back(g);
                g = redo_stack.back();
                redo_stack.pop_back();
            }
        } else if (c == "add-layer" && tok.size() >= 2) {
            push_undo(g);
            add_layer(g, tok[1]);
        } else if (c == "remove-layer" && tok.size() >= 2) {
            push_undo(g);
            remove_layer(g, tok[1]);
        } else if (c == "add-flow" && tok.size() >= 3) {
            bool backward = false;
            for (std::size_t i = 3; i < tok.size(); ++i)
                if (tok[i] == "--backward")
                    backward = true;
            push_undo(g);
            add_flow(g, tok[1], tok[2], std::nullopt, backward);
        } else if (c == "remove-flow" && tok.size() >= 3) {
            bool backward = false;
            for (std::size_t i = 3; i < tok.size(); ++i)
                if (tok[i] == "--backward")
                    backward = true;
            push_undo(g);
            remove_flow(g, tok[1], tok[2], backward);
        }
    }
    save_graph_file(g, out_path);
    return 0;
}

int run_interactive(HarmonicGraph& g, const std::string& out_path, int bits) {
    std::vector<HarmonicGraph> undo_stack;
    std::vector<HarmonicGraph> redo_stack;
    std::unique_ptr<CycleRuntime> runtime;
    std::string line;
    auto push_undo = [&](const HarmonicGraph& cur) {
        undo_stack.push_back(cur);
        redo_stack.clear();
    };
    while (true) {
        std::cout << "graph> " << std::flush;
        if (!std::getline(std::cin, line))
            break;
        std::istringstream iss(line);
        std::vector<std::string> tok;
        std::string t;
        while (iss >> t)
            tok.push_back(t);
        // As with the batch mode, ignore comments and blank lines so
        // users can document their interactive sessions.
        if (tok.empty() || tok[0].front() == '#')
            continue;
        const std::string& c = tok[0];
        if (c == "exit" || c == "quit") {
            break;
        } else if (c == "save") {
            save_graph_file(g, out_path);
        } else if (c == "run") {
            bind_zero_producers(g);
            runtime = std::make_unique<CycleRuntime>(g, make_max_bits_policy(bits));
            // Execute a single inference cycle so that subsequent commands
            // can inspect layer information or export updated graphs.
            runtime->forward();
            std::cout << "Executed forward pass\n";
        } else if (c == "info") {
            if (!runtime) {
                std::cout << "Run the graph first\n";
            } else {
                auto info = get_layer_info(g);
                for (const auto& l : info) {
                    const auto& w = layer_weights(*runtime, l.name);
                    std::cout << l.name << " width=" << l.width << " weights=" << w.data().size()
                              << " bytes" << '\n';
                }
            }
        } else if (c == "undo") {
            if (!undo_stack.empty()) {
                redo_stack.push_back(g);
                g = undo_stack.back();
                undo_stack.pop_back();
                if (runtime)
                    sync_runtime(*runtime);
            }
        } else if (c == "redo") {
            if (!redo_stack.empty()) {
                undo_stack.push_back(g);
                g = redo_stack.back();
                redo_stack.pop_back();
                if (runtime)
                    sync_runtime(*runtime);
            }
        } else if (c == "add-layer" && tok.size() >= 2) {
            push_undo(g);
            add_layer(g, tok[1]);
        } else if (c == "remove-layer" && tok.size() >= 2) {
            push_undo(g);
            remove_layer(g, tok[1]);
        } else if (c == "add-flow" && tok.size() >= 3) {
            bool backward = false;
            for (std::size_t i = 3; i < tok.size(); ++i)
                if (tok[i] == "--backward")
                    backward = true;
            push_undo(g);
            add_flow(g, tok[1], tok[2], std::nullopt, backward);
        } else if (c == "remove-flow" && tok.size() >= 3) {
            bool backward = false;
            for (std::size_t i = 3; i < tok.size(); ++i)
                if (tok[i] == "--backward")
                    backward = true;
            push_undo(g);
            remove_flow(g, tok[1], tok[2], backward);
        }
    }
    save_graph_file(g, out_path);
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage();
        return 1;
    }

    std::string cmd = argv[1];
    std::string path = argv[2];

    int bits = 32;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bits" && i + 1 < argc) {
            bits = std::stoi(argv[++i]);
        }
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "failed to open graph file\n";
        return 1;
    }
    auto g = load_graph(in);

    std::string out_path = path;
    bool backward = false;
    if (cmd == "info") {
        bind_zero_producers(g);
        CycleRuntime rt{g, make_max_bits_policy(bits)};
        rt.forward();
        auto info = get_layer_info(g);
        for (const auto& l : info) {
            const auto& w = layer_weights(rt, l.name);
            std::cout << l.name << " width=" << l.width << " weights=" << w.data().size()
                      << " bytes" << '\n';
        }
        return 0;
    } else if (cmd == "add-layer") {
        if (argc < 4) {
            usage();
            return 1;
        }
        add_layer(g, argv[3]);
        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-o" && i + 1 < argc) {
                out_path = argv[++i];
            }
        }
        save_graph_file(g, out_path);
        return 0;
    } else if (cmd == "remove-layer") {
        if (argc < 4) {
            usage();
            return 1;
        }
        remove_layer(g, argv[3]);
        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-o" && i + 1 < argc) {
                out_path = argv[++i];
            }
        }
        save_graph_file(g, out_path);
        return 0;
    } else if (cmd == "add-flow") {
        if (argc < 5) {
            usage();
            return 1;
        }
        std::string src = argv[3];
        std::string dst = argv[4];
        for (int i = 5; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-o" && i + 1 < argc) {
                out_path = argv[++i];
            } else if (arg == "--backward") {
                backward = true;
            }
        }
        add_flow(g, src, dst, std::nullopt, backward);
        save_graph_file(g, out_path);
        return 0;
    } else if (cmd == "remove-flow") {
        if (argc < 5) {
            usage();
            return 1;
        }
        std::string src = argv[3];
        std::string dst = argv[4];
        for (int i = 5; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-o" && i + 1 < argc) {
                out_path = argv[++i];
            } else if (arg == "--backward") {
                backward = true;
            }
        }
        remove_flow(g, src, dst, backward);
        save_graph_file(g, out_path);
        return 0;
    } else if (cmd == "dot") {
        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-o" && i + 1 < argc) {
                out_path = argv[++i];
            }
        }
        std::string dot = export_dot(g);
        if (out_path != path) {
            std::ofstream out(out_path);
            if (!out) {
                std::cerr << "failed to open output file\n";
                return 1;
            }
            out << dot;
        } else {
            std::cout << dot;
        }
        return 0;
    } else if (cmd == "batch") {
        if (argc < 4) {
            usage();
            return 1;
        }
        std::string script = argv[3];
        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-o" && i + 1 < argc) {
                out_path = argv[++i];
            }
        }
        return run_batch(g, script, out_path);
    } else if (cmd == "interactive") {
        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-o" && i + 1 < argc) {
                out_path = argv[++i];
            }
        }
        return run_interactive(g, out_path, bits);
    }

    usage();
    return 1;
}
