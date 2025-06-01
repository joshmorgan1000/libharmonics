#include <harmonics/graph.hpp>
#include <harmonics/introspection.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace harmonics;

static std::size_t dtype_size(HTensor::DType dt) {
    switch (dt) {
    case HTensor::DType::Float32:
    case HTensor::DType::Int32:
        return 4;
    case HTensor::DType::Float64:
    case HTensor::DType::Int64:
        return 8;
    case HTensor::DType::UInt8:
    default:
        return 1;
    }
}

int main() {
    std::cout << "Harmonics Shell\n";
    std::cout << "Enter DSL lines. Commands: run, info, layers, weights <name>, precision <bits>, "
                 "clear, exit\n";

    std::string code;
    std::unique_ptr<HarmonicGraph> graph;
    std::unique_ptr<CycleRuntime> runtime;
    int bits = 32;
    std::string line;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, line))
            break;
        if (line == "exit") {
            break;
        } else if (line == "clear") {
            code.clear();
            graph.reset();
            runtime.reset();
        } else if (line.rfind("precision", 0) == 0) {
            std::istringstream iss(line);
            std::string cmd;
            int b = 32;
            iss >> cmd >> b;
            bits = b;
            std::cout << "Precision set to " << bits << " bits\n";
        } else if (line == "run") {
            try {
                Parser parser{code.c_str()};
                auto ast = parser.parse_declarations();
                graph = std::make_unique<HarmonicGraph>(build_graph(ast));
                runtime = std::make_unique<CycleRuntime>(*graph, make_max_bits_policy(bits));
                runtime->forward();
                std::cout << "Executed forward pass\n";
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << '\n';
            }
        } else if (line == "info") {
            if (!graph || !runtime) {
                std::cout << "No graph executed yet\n";
            } else {
                auto layers = get_layer_info(*graph);
                for (const auto& l : layers) {
                    const auto& w = layer_weights(*runtime, l.name);
                    std::cout << l.name << " width=" << l.width << " weights=" << w.data().size()
                              << " bytes" << '\n';
                }
            }
        } else if (line == "layers") {
            if (!graph) {
                std::cout << "No graph parsed yet\n";
            } else {
                for (const auto& l : get_layer_info(*graph))
                    std::cout << l.name << '\n';
            }
        } else if (line.rfind("weights", 0) == 0) {
            std::istringstream iss(line);
            std::string cmd, name;
            iss >> cmd >> name;
            if (!runtime) {
                std::cout << "Run the graph first\n";
            } else {
                try {
                    const auto& w = layer_weights(*runtime, name);
                    std::size_t elems = w.data().size() / dtype_size(w.dtype());
                    switch (w.dtype()) {
                    case HTensor::DType::Float32: {
                        const float* d = reinterpret_cast<const float*>(w.data().data());
                        for (std::size_t i = 0; i < elems; ++i)
                            std::cout << d[i] << ' ';
                        break;
                    }
                    case HTensor::DType::Float64: {
                        const double* d = reinterpret_cast<const double*>(w.data().data());
                        for (std::size_t i = 0; i < elems; ++i)
                            std::cout << d[i] << ' ';
                        break;
                    }
                    case HTensor::DType::Int32: {
                        const std::int32_t* d =
                            reinterpret_cast<const std::int32_t*>(w.data().data());
                        for (std::size_t i = 0; i < elems; ++i)
                            std::cout << d[i] << ' ';
                        break;
                    }
                    case HTensor::DType::Int64: {
                        const std::int64_t* d =
                            reinterpret_cast<const std::int64_t*>(w.data().data());
                        for (std::size_t i = 0; i < elems; ++i)
                            std::cout << d[i] << ' ';
                        break;
                    }
                    case HTensor::DType::UInt8: {
                        const std::uint8_t* d =
                            reinterpret_cast<const std::uint8_t*>(w.data().data());
                        for (std::size_t i = 0; i < elems; ++i)
                            std::cout << static_cast<int>(d[i]) << ' ';
                        break;
                    }
                    }
                    std::cout << '\n';
                } catch (const std::exception& e) {
                    std::cout << e.what() << '\n';
                }
            }
        } else {
            code += line;
            code += '\n';
        }
    }
    return 0;
}
