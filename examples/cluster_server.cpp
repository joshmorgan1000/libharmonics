#include <harmonics/distributed_io.hpp>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/partition.hpp>
#include <harmonics/remote_scheduler.hpp>
#include <iostream>
#include <thread>

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

int main() {
#ifdef __unix__
    auto parts = build_parts();
    registerActivation("id", std::make_shared<IdActivation>());

    SocketServer in_srv(6000);  // receive from client
    SocketServer out_srv(6001); // send to scheduler
    std::thread bridge([&]() {
        auto in_prod = in_srv.accept_producer();
        auto out_cons = out_srv.accept_consumer();
        auto t = in_prod->next();
        out_cons->push(t);
    });

    std::vector<RemoteBinding> prod_bindings{
        {"boundary0", "127.0.0.1", 6001, RemoteTransport::TCP}};
    RemoteScheduler sched{parts.second, prod_bindings, {}};

    sched.step();
    bridge.join();

    auto idx = parts.second.find("c").index;
    std::cout << "Output dim: " << sched.runtime().state().consumer_tensors[idx].shape()[0] << '\n';
#else
    std::cout << "Example requires POSIX networking support\n";
#endif
    return 0;
}
