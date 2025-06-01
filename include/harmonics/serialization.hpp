#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "harmonics/deployment.hpp"
#include "harmonics/graph.hpp"
#include "harmonics/version.hpp"

namespace harmonics {

using NamedTensor = std::pair<std::string, HTensor>;

inline void write_string(std::ostream& out, const std::string& s) {
    std::uint32_t len = static_cast<std::uint32_t>(s.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(s.data(), len);
}

inline std::string read_string(std::istream& in) {
    std::uint32_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!in)
        throw std::runtime_error("failed to read string length");
    std::string s(len, '\0');
    in.read(s.data(), len);
    if (!in)
        throw std::runtime_error("failed to read string data");
    return s;
}

inline void write_tensor(std::ostream& out, const HTensor& t) {
    std::uint8_t dt = static_cast<std::uint8_t>(t.dtype());
    out.write(reinterpret_cast<const char*>(&dt), sizeof(dt));
    std::uint32_t dims = static_cast<std::uint32_t>(t.shape().size());
    out.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
    for (auto d : t.shape()) {
        std::uint32_t dim = static_cast<std::uint32_t>(d);
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    std::uint32_t size = static_cast<std::uint32_t>(t.data().size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(t.data().data()), size);
}

inline HTensor read_tensor(std::istream& in) {
    std::uint8_t dt;
    in.read(reinterpret_cast<char*>(&dt), sizeof(dt));
    std::uint32_t dims;
    in.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    if (!in)
        throw std::runtime_error("failed to read tensor header");
    HTensor::Shape shape(dims);
    for (std::uint32_t i = 0; i < dims; ++i) {
        std::uint32_t dim;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!in)
            throw std::runtime_error("failed to read tensor shape");
        shape[i] = dim;
    }
    std::uint32_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::vector<std::byte> data(size);
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in)
        throw std::runtime_error("failed to read tensor data");
    return HTensor{static_cast<HTensor::DType>(dt), std::move(shape), std::move(data)};
}

inline void save_graph(const HarmonicGraph& g, std::ostream& out) {
    out.write("HGRF", 4);
    std::uint32_t ver = static_cast<std::uint32_t>(version());
    out.write(reinterpret_cast<const char*>(&ver), sizeof(ver));

    std::uint32_t count = static_cast<std::uint32_t>(g.producers.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& p : g.producers) {
        write_string(out, p.name);
        std::uint8_t has_shape = p.shape ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&has_shape), sizeof(has_shape));
        if (p.shape) {
            std::int32_t s = *p.shape;
            out.write(reinterpret_cast<const char*>(&s), sizeof(s));
        }
        std::uint8_t has_ratio = p.ratio ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&has_ratio), sizeof(has_ratio));
        if (p.ratio) {
            out.write(reinterpret_cast<const char*>(&p.ratio->lhs), sizeof(int));
            out.write(reinterpret_cast<const char*>(&p.ratio->rhs), sizeof(int));
            write_string(out, p.ratio->ref);
        }
    }

    count = static_cast<std::uint32_t>(g.consumers.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& c : g.consumers) {
        write_string(out, c.name);
        std::uint8_t has_shape = c.shape ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&has_shape), sizeof(has_shape));
        if (c.shape) {
            std::int32_t s = *c.shape;
            out.write(reinterpret_cast<const char*>(&s), sizeof(s));
        }
    }

    count = static_cast<std::uint32_t>(g.layers.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& l : g.layers) {
        write_string(out, l.name);
        std::uint8_t has_ratio = l.ratio ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&has_ratio), sizeof(has_ratio));
        if (l.ratio) {
            out.write(reinterpret_cast<const char*>(&l.ratio->lhs), sizeof(int));
            out.write(reinterpret_cast<const char*>(&l.ratio->rhs), sizeof(int));
            write_string(out, l.ratio->ref);
        }
        std::uint8_t has_shape = l.shape ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&has_shape), sizeof(has_shape));
        if (l.shape) {
            std::int32_t s = *l.shape;
            out.write(reinterpret_cast<const char*>(&s), sizeof(s));
        }
    }

    count = static_cast<std::uint32_t>(g.cycle.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& fl : g.cycle) {
        std::uint8_t kind = static_cast<std::uint8_t>(fl.source.kind);
        std::uint32_t index = static_cast<std::uint32_t>(fl.source.index);
        out.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
        out.write(reinterpret_cast<const char*>(&index), sizeof(index));

        std::uint32_t arrows = static_cast<std::uint32_t>(fl.arrows.size());
        out.write(reinterpret_cast<const char*>(&arrows), sizeof(arrows));
        for (const auto& ar : fl.arrows) {
            std::uint8_t backward = ar.backward ? 1 : 0;
            out.write(reinterpret_cast<const char*>(&backward), sizeof(backward));
            std::uint8_t has_func = ar.func ? 1 : 0;
            out.write(reinterpret_cast<const char*>(&has_func), sizeof(has_func));
            if (ar.func)
                write_string(out, *ar.func);
            kind = static_cast<std::uint8_t>(ar.target.kind);
            index = static_cast<std::uint32_t>(ar.target.index);
            out.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
            out.write(reinterpret_cast<const char*>(&index), sizeof(index));
        }
    }
}

inline HarmonicGraph load_graph(std::istream& in) {
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "HGRF")
        throw std::runtime_error("invalid graph file");
    std::uint32_t ver;
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    (void)ver; // future compatibility

    HarmonicGraph g;

    std::uint32_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    g.producers.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        auto& p = g.producers[i];
        p.name = read_string(in);
        std::uint8_t has_shape;
        in.read(reinterpret_cast<char*>(&has_shape), sizeof(has_shape));
        if (has_shape) {
            std::int32_t s;
            in.read(reinterpret_cast<char*>(&s), sizeof(s));
            p.shape = s;
        }
        std::uint8_t has_ratio;
        in.read(reinterpret_cast<char*>(&has_ratio), sizeof(has_ratio));
        if (has_ratio) {
            Ratio r;
            in.read(reinterpret_cast<char*>(&r.lhs), sizeof(int));
            in.read(reinterpret_cast<char*>(&r.rhs), sizeof(int));
            r.ref = read_string(in);
            p.ratio = r;
        }
    }

    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    g.consumers.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        auto& c = g.consumers[i];
        c.name = read_string(in);
        std::uint8_t has_shape;
        in.read(reinterpret_cast<char*>(&has_shape), sizeof(has_shape));
        if (has_shape) {
            std::int32_t s;
            in.read(reinterpret_cast<char*>(&s), sizeof(s));
            c.shape = s;
        }
    }

    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    g.layers.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        auto& l = g.layers[i];
        l.name = read_string(in);
        std::uint8_t has_ratio;
        in.read(reinterpret_cast<char*>(&has_ratio), sizeof(has_ratio));
        if (has_ratio) {
            Ratio r;
            in.read(reinterpret_cast<char*>(&r.lhs), sizeof(int));
            in.read(reinterpret_cast<char*>(&r.rhs), sizeof(int));
            r.ref = read_string(in);
            l.ratio = r;
        }
        std::uint8_t has_shape;
        in.read(reinterpret_cast<char*>(&has_shape), sizeof(has_shape));
        if (has_shape) {
            std::int32_t s;
            in.read(reinterpret_cast<char*>(&s), sizeof(s));
            l.shape = s;
        }
    }

    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    g.cycle.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        auto& fl = g.cycle[i];
        std::uint8_t kind;
        std::uint32_t index;
        in.read(reinterpret_cast<char*>(&kind), sizeof(kind));
        in.read(reinterpret_cast<char*>(&index), sizeof(index));
        fl.source = {static_cast<NodeKind>(kind), index};
        std::uint32_t arrows;
        in.read(reinterpret_cast<char*>(&arrows), sizeof(arrows));
        fl.arrows.resize(arrows);
        for (std::uint32_t j = 0; j < arrows; ++j) {
            auto& ar = fl.arrows[j];
            std::uint8_t backward;
            in.read(reinterpret_cast<char*>(&backward), sizeof(backward));
            ar.backward = backward != 0;
            std::uint8_t has_func;
            in.read(reinterpret_cast<char*>(&has_func), sizeof(has_func));
            if (has_func)
                ar.func = read_string(in);
            in.read(reinterpret_cast<char*>(&kind), sizeof(kind));
            in.read(reinterpret_cast<char*>(&index), sizeof(index));
            ar.target = {static_cast<NodeKind>(kind), index};
        }
    }

    g.producer_bindings.resize(g.producers.size());
    return g;
}

inline void save_weights(const std::vector<HTensor>& w, std::ostream& out) {
    out.write("HWTS", 4);
    std::uint32_t ver = static_cast<std::uint32_t>(version());
    out.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    std::uint32_t count = static_cast<std::uint32_t>(w.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& t : w)
        write_tensor(out, t);
}

inline std::vector<HTensor> load_weights(std::istream& in) {
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "HWTS")
        throw std::runtime_error("invalid weights file");
    std::uint32_t ver;
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    (void)ver;
    std::uint32_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<HTensor> w;
    w.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i)
        w.push_back(read_tensor(in));
    return w;
}

inline void save_named_weights(const std::vector<NamedTensor>& w, std::ostream& out) {
    out.write("HNWT", 4);
    std::uint32_t ver = static_cast<std::uint32_t>(version());
    out.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    std::uint32_t count = static_cast<std::uint32_t>(w.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& n : w) {
        write_string(out, n.first);
        write_tensor(out, n.second);
    }
}

inline std::vector<NamedTensor> load_named_weights(std::istream& in) {
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "HNWT")
        throw std::runtime_error("invalid named weights file");
    std::uint32_t ver;
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    (void)ver;
    std::uint32_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<NamedTensor> w;
    w.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::string name = read_string(in);
        HTensor t = read_tensor(in);
        w.emplace_back(std::move(name), std::move(t));
    }
    return w;
}

} // namespace harmonics
