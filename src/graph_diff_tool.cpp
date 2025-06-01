#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <harmonics/graph_diff.hpp>
#include <harmonics/serialization.hpp>

using namespace harmonics;

namespace {
void usage() {
    std::cerr << "Usage: graph_diff <command> <graph1> <graph2> [-o out]\n";
    std::cerr << "Commands:\n";
    std::cerr << "  diff  \tPrint changes required to transform graph1 into graph2\n";
    std::cerr << "  merge \tApply changes from graph2 to graph1 and write result\n";
}

void print_diff(const GraphDiff& diff, std::ostream& os) {
    for (const auto& l : diff.added_layers)
        os << "+ layer " << l.name << '\n';
    for (const auto& l : diff.removed_layers)
        os << "- layer " << l << '\n';
    auto flow_line = [](char sign, const GraphDiff::Flow& f) {
        std::ostringstream s;
        s << sign << " flow " << f.src << " -> " << f.dst;
        if (f.backward)
            s << " [backward]";
        if (f.func)
            s << " (" << *f.func << ')';
        return s.str();
    };
    for (const auto& f : diff.added_flows)
        os << flow_line('+', f) << '\n';
    for (const auto& f : diff.removed_flows)
        os << flow_line('-', f) << '\n';
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        usage();
        return 1;
    }
    std::string cmd = argv[1];
    std::string g1_path = argv[2];
    std::string g2_path = argv[3];
    std::string out_path;
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--out") && i + 1 < argc)
            out_path = argv[++i];
    }
    std::ifstream in1(g1_path, std::ios::binary);
    std::ifstream in2(g2_path, std::ios::binary);
    if (!in1 || !in2) {
        std::cerr << "failed to open graph file\n";
        return 1;
    }
    auto g1 = load_graph(in1);
    auto g2 = load_graph(in2);

    if (cmd == "diff") {
        auto diff = diff_graphs(g1, g2);
        if (out_path.empty()) {
            print_diff(diff, std::cout);
        } else {
            std::ofstream out(out_path);
            if (!out) {
                std::cerr << "failed to open output file\n";
                return 1;
            }
            print_diff(diff, out);
        }
        return 0;
    }
    if (cmd == "merge") {
        auto merged = merge_graphs(g1, g2);
        if (out_path.empty())
            out_path = g1_path;
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            std::cerr << "failed to open output file\n";
            return 1;
        }
        save_graph(merged, out);
        return 0;
    }

    usage();
    return 1;
}
