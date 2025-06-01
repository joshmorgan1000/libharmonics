#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/partition.hpp>
#include <harmonics/remote_scheduler.hpp>
#include <iostream>

using namespace harmonics;

struct IdActivation : ActivationFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

struct FixedProducer : Producer {
    explicit FixedProducer(std::size_t s) : shape{s} {}
    HTensor next() override { return HTensor{HTensor::DType::Float32, {shape}}; }
    std::size_t size() const override { return 1; }
    std::size_t shape;
};

static std::pair<HarmonicGraph, HarmonicGraph> build_parts() {
    const char* src = R"(
producer p {1};
consumer c {1};
layer l1;
layer l2;
cycle { p -(id)-> l1 -(id)-> l2 -> c; }
)";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    return partition_by_layer(g, 1);
}

int main(int argc, char** argv) {
#ifdef __unix__
    if (argc < 2) {
        std::cerr << "Usage: cluster_client <server_host>\n";
        return 1;
    }
    std::string host = argv[1];
    auto parts = build_parts();
    registerActivation("id", std::make_shared<IdActivation>());
    auto prod = std::make_shared<FixedProducer>(1);
    parts.first.bindProducer("p", prod);

    std::vector<RemoteBinding> cons_bindings{{"boundary0", host, 6000, RemoteTransport::TCP}};
    RemoteScheduler sched{parts.first, {}, cons_bindings};

    sched.step();
#else
    (void)argc;
    (void)argv;
    std::cout << "Example requires POSIX networking support\n";
#endif
    return 0;
}
