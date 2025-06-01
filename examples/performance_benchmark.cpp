#include <chrono>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace harmonics;

struct IdActivation : ActivationFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

struct BenchmarkProducer : Producer {
    explicit BenchmarkProducer(std::size_t elements) : shape{elements} {}
    HTensor next() override {
        HTensor t{HTensor::DType::Float32, shape};
        t.data().resize(sizeof(float) * shape[0]);
        return t;
    }
    std::size_t size() const override { return 1; }
    HTensor::Shape shape{};
};

static double benchmark(Backend backend, std::size_t size, std::size_t runs) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    auto prod = std::make_shared<BenchmarkProducer>(size);
    g.bindProducer("p", prod);
    registerActivation("id", std::make_shared<IdActivation>());

    DeploymentDescriptor desc{};
    desc.backend = backend;
    double total = 0.0;
    for (std::size_t i = 0; i < runs; ++i) {
        CycleRuntime rt{g, make_auto_policy(), desc};
        auto start = std::chrono::high_resolution_clock::now();
        rt.forward();
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total / static_cast<double>(runs);
}

int main(int argc, char** argv) {
    std::size_t size = 1024 * 1024; // elements per tensor
    std::size_t runs = 10;
    if (argc > 1)
        size = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc > 2)
        runs = static_cast<std::size_t>(std::stoul(argv[2]));

    std::cout << "Benchmarking forward pass with " << size << " elements\n";

    double cpu_ms = benchmark(Backend::CPU, size, runs);
    double gpu_ms = benchmark(Backend::GPU, size, runs);
    double fpga_ms = benchmark(Backend::FPGA, size, runs);

    std::cout << "CPU  : " << cpu_ms << " ms\n";
    std::cout << "GPU  : " << gpu_ms << " ms\n";
    std::cout << "FPGA : " << fpga_ms << " ms\n";
    return 0;
}
