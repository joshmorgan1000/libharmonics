#pragma once

#include <fstream>
#include <sstream>
#include <string>

#include "harmonics/graph.hpp"

namespace harmonics {

inline std::string node_name(const HarmonicGraph& g, NodeId id) {
    switch (id.kind) {
    case NodeKind::Producer:
        return g.producers[id.index].name;
    case NodeKind::Consumer:
        return g.consumers[id.index].name;
    case NodeKind::Layer:
        return g.layers[id.index].name;
    }
    return {};
}

inline std::string export_dot(const HarmonicGraph& g) {
    std::ostringstream out;
    out << "digraph HarmonicGraph {\n";
    for (const auto& p : g.producers)
        out << "  \"" << p.name << "\" [shape=box];\n";
    for (const auto& c : g.consumers)
        out << "  \"" << c.name << "\" [shape=oval];\n";
    for (const auto& l : g.layers)
        out << "  \"" << l.name << "\" [shape=ellipse];\n";

    for (const auto& line : g.cycle) {
        std::string src = node_name(g, line.source);
        for (const auto& ar : line.arrows) {
            std::string dst = node_name(g, ar.target);
            out << "  \"" << src << "\" -> \"" << dst << "\"";
            bool first = true;
            if (ar.backward || ar.func) {
                out << " [";
                if (ar.backward) {
                    out << "style=dashed";
                    first = false;
                }
                if (ar.func) {
                    if (!first)
                        out << ",";
                    out << "label=\"" << *ar.func << "\"";
                }
                out << "]";
            }
            out << ";\n";
        }
    }
    out << "}\n";
    return out.str();
}

inline void export_dot_file(const HarmonicGraph& g, const std::string& path) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("failed to open output file");
    f << export_dot(g);
}

} // namespace harmonics
