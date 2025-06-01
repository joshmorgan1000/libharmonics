#include <cstring>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>
#include <harmonics/training_visualizer.hpp>
#include <iostream>

struct IdActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

struct DummyLoss : harmonics::LossFunction {
    harmonics::HTensor operator()(const harmonics::HTensor&,
                                  const harmonics::HTensor&) const override {
        float v = 1.0f;
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &v, sizeof(float));
        return {harmonics::HTensor::DType::Float32, {1}, std::move(data)};
    }
};

struct DummyProducer : harmonics::Producer {
    explicit DummyProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override { return {harmonics::HTensor::DType::Float32, shape}; }
    std::size_t size() const override { return 1; }
    harmonics::HTensor::Shape shape{};
};

int main() {
    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; l <-(loss)-> p; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<DummyProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());
    harmonics::registerLoss("loss", std::make_shared<DummyLoss>());

    harmonics::FitOptions opt;
    opt.progress = harmonics::WebSocketTrainingVisualizer("127.0.0.1", 8080);
    g.fit(3, harmonics::make_auto_policy(), opt);

    std::cout << "Training finished\n";
}
