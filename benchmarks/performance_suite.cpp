#include <chrono>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>
#include <harmonics/multi_accelerator_scheduler.hpp>
#include <harmonics/partition.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

static double benchmark_multi(std::size_t size, std::size_t runs) {
    const char* src = "producer p {1}; consumer c {1}; layer l1; layer l2; cycle { p -(id)-> l1 -(id)-> l2 -> c; }";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    auto prod = std::make_shared<BenchmarkProducer>(size);
    g.bindProducer("p", prod);
    registerActivation("id", std::make_shared<IdActivation>());

    DeploymentDescriptor desc;
    desc.partitions = {{Backend::GPU}, {Backend::CPU}};
    auto parts = auto_partition(g, desc);
    double total = 0.0;
    for (std::size_t i = 0; i < runs; ++i) {
        MultiAcceleratorScheduler sched{parts, desc};
        auto start = std::chrono::high_resolution_clock::now();
        sched.step();
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total / static_cast<double>(runs);
}

int main(int argc, char** argv) {
    std::size_t runs = 10;
    std::vector<std::size_t> sizes{1024, 1024 * 1024, 16 * 1024 * 1024};
    if (argc > 1)
        runs = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc > 2) {
        sizes.clear();
        for (int i = 2; i < argc; ++i)
            sizes.push_back(static_cast<std::size_t>(std::stoul(argv[i])));
    }

    for (auto size : sizes) {
        std::cout << "Elements: " << size << "\n";
        double cpu_ms = benchmark(Backend::CPU, size, runs);
        double gpu_ms = benchmark(Backend::GPU, size, runs);
        double fpga_ms = benchmark(Backend::FPGA, size, runs);
        double mixed_ms = benchmark_multi(size, runs);
        std::cout << "  CPU   : " << cpu_ms << " ms\n";
        std::cout << "  GPU   : " << gpu_ms << " ms\n";
        std::cout << "  FPGA  : " << fpga_ms << " ms\n";
        std::cout << "  Mixed : " << mixed_ms << " ms\n";
    }
    return 0;
}
