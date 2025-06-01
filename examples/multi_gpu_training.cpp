#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/partition.hpp>
#include <harmonics/runtime.hpp>
#include <harmonics/shaders.hpp>
#include <harmonics/stream_io.hpp>
#include <iostream>

using namespace harmonics;

namespace {
struct FixedProducer : Producer {
    explicit FixedProducer(std::size_t width) : shape{width} {}
    HTensor next() override { return HTensor{HTensor::DType::Float32, {shape}}; }
    std::size_t size() const override { return 1; }
    std::size_t shape;
};
} // namespace

int main() {
    register_builtin_shaders();

    const char* src = R"(
producer data {1};
producer label {1};
layer l1;
layer l2;
cycle {
  data -(relu)-> l1;
  l1 -(relu)-> l2;
  l2 <-(mse)- label;
}
)";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);

    auto parts = partition_by_layer(g, 1);
    auto& first = parts.first;   // runs on GPU 0
    auto& second = parts.second; // runs on GPU 1

    auto bus = std::make_shared<MessageBus>();
    auto bus_prod = std::make_shared<BusProducer>(bus);
    BusConsumer bus_cons(bus);

    first.bindProducer("data", std::make_shared<FixedProducer>(1));
    second.bindProducer("boundary0", bus_prod);
    second.bindProducer("label", std::make_shared<FixedProducer>(1));

    DeploymentDescriptor d0;
    d0.backend = Backend::GPU;
    d0.gpu_device_index = 0;
    DeploymentDescriptor d1;
    d1.backend = Backend::GPU;
    d1.gpu_device_index = 1;

    FitOptions opt;
    opt.learning_rate = 0.01f;

    for (int epoch = 0; epoch < 5; ++epoch) {
        CycleRuntime rt1{first, make_auto_policy(), d0};
        rt1.forward();
        auto t = rt1.state().consumer_tensors[first.find("boundary0").index];
        bus_cons.push(t);
        second.fit(1, make_auto_policy(), opt, d1);
    }

    std::cout << "Training complete on GPUs 0 and 1" << std::endl;
    return 0;
}