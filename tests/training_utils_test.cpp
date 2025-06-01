#include <cstring>
#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>

struct IdActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

struct LargeGradLoss : harmonics::LossFunction {
    harmonics::HTensor operator()(const harmonics::HTensor&,
                                  const harmonics::HTensor&) const override {
        float v = 10.0f;
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &v, sizeof(float));
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}, std::move(data)};
    }
};

struct CountingProducer : harmonics::Producer {
    explicit CountingProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override {
        ++calls;
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, shape};
    }
    std::size_t size() const override { return 1; }
    harmonics::HTensor::Shape shape{};
    int calls{0};
};

TEST(TrainingUtils, GradientClippingAppliesLimit) {
    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; l <-(large)-> p; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<CountingProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());
    harmonics::registerLoss("large", std::make_shared<LargeGradLoss>());

    harmonics::FitOptions opt;
    opt.learning_rate = 0.1f;
    opt.grad_clip = 1.0f;
    auto state = g.fit(1, harmonics::make_auto_policy(), opt);
    auto val = reinterpret_cast<const float*>(state.weights[0].data().data())[0];
    EXPECT_NEAR(val, -0.1f, 1e-5f);
}

TEST(TrainingUtils, EarlyStoppingStopsLoop) {
    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; l <-(large)-> p; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<CountingProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());
    harmonics::registerLoss("large", std::make_shared<LargeGradLoss>());

    harmonics::FitOptions opt;
    opt.early_stop_patience = 1;
    opt.early_stop_delta = 0.1f;
    auto state = g.fit(10, harmonics::make_auto_policy(), opt);
    EXPECT_LT(prod->calls, 10);
    EXPECT_FALSE(state.weights.empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
