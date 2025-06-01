#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>
#include <harmonics/shaders.hpp>

struct IdActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

struct FixedProducer : harmonics::Producer {
    explicit FixedProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override {
        ++calls;
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, shape};
    }
    std::size_t size() const override { return 1; }
    harmonics::HTensor::Shape shape{};
    int calls{0};
};

TEST(RuntimeApiTest, InferenceRunsForward) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    auto state = g.inference();
    EXPECT_EQ(state.layer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(state.layer_tensors[0].shape()[0], 1u);
}

TEST(RuntimeApiTest, FitLoopsAtLeastOnce) {
    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    auto state = g.fit(std::chrono::milliseconds(0), harmonics::make_auto_policy());
    EXPECT_EQ(state.layer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(state.layer_tensors[0].shape()[0], 1u);
}

TEST(RuntimeApiTest, FitEpochCount) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    (void)g.fit(3, harmonics::make_auto_policy());
    EXPECT_EQ(prod->calls, 3);
}

TEST(RuntimeApiTest, PrecisionNegotiationBitsSet) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    auto state = g.fit(1, harmonics::make_max_bits_policy(8));
    EXPECT_EQ(state.precision_bits.size(), 1u);
    EXPECT_EQ(state.precision_bits[0], 8);
}

TEST(RuntimeApiTest, PrecisionNegotiationEntropy) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    auto state = g.fit(1, harmonics::make_entropy_limit_policy(0.1f));
    EXPECT_EQ(state.precision_bits.size(), 1u);
    EXPECT_EQ(state.precision_bits[0], 4);
}

TEST(RuntimeApiTest, PerLayerPrecisionPolicy) {
    const char* src = "producer p; layer l1; layer l2; cycle { p -(id)-> l1; l1 -(id)-> l2; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    auto policy = harmonics::make_layer_bits_policy({8, 4});
    auto state = g.fit(1, policy);
    EXPECT_EQ(state.precision_bits.size(), 2u);
    EXPECT_EQ(state.precision_bits[0], 8);
    EXPECT_EQ(state.precision_bits[1], 4);
}

TEST(RuntimeApiTest, HardwareGuidedPolicy) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    auto state = g.fit(1, harmonics::make_hardware_policy());
#if HARMONICS_HAS_VULKAN
    EXPECT_EQ(state.precision_bits[0], 16);
#else
    EXPECT_EQ(state.precision_bits[0], 32);
#endif
}

TEST(RuntimeApiTest, FitUntilPredicate) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    auto state = g.fit_until([&](const harmonics::CycleState&) { return prod->calls >= 5; },
                             harmonics::make_auto_policy());
    EXPECT_EQ(prod->calls, 5);
    EXPECT_EQ(state.layer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(state.layer_tensors[0].shape()[0], 1u);
}

TEST(RuntimeApiTest, AdamOptimizer) {
    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    harmonics::FitOptions opt;
    opt.optimizer = harmonics::Optimizer::Adam;
    opt.learning_rate = 0.001f;
    auto state = g.fit(1, harmonics::make_auto_policy(), opt);
    EXPECT_EQ(state.layer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(state.layer_tensors[0].shape()[0], 1u);
}

TEST(RuntimeApiTest, GradientAccumulation) {
    const char* src = "producer p {1}; layer l; l <-(mse)- p; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());
    harmonics::registerLoss("mse", std::make_shared<harmonics::MSELoss>());

    harmonics::FitOptions opt;
    opt.accumulate_steps = 2;
    std::vector<std::size_t> steps;
    opt.progress = [&](std::size_t step, float, float, float) { steps.push_back(step); };
    (void)g.fit(4, harmonics::make_auto_policy(), opt);
    EXPECT_EQ(steps.size(), 2u);
    EXPECT_EQ(steps[0], 1u);
    EXPECT_EQ(steps[1], 2u);
}

TEST(RuntimeApiTest, GpuBackendFallsBack) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    harmonics::DeploymentDescriptor desc;
    desc.backend = harmonics::Backend::GPU;
    auto state = g.inference(desc);
    EXPECT_EQ(state.layer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(state.layer_tensors[0].shape()[0], 1u);
}

TEST(RuntimeApiTest, FpgaBackendFallsBack) {
    const char* src = "producer p; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    harmonics::DeploymentDescriptor desc;
    desc.backend = harmonics::Backend::FPGA;
    auto state = g.inference(desc);
    EXPECT_EQ(state.layer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(state.layer_tensors[0].shape()[0], 1u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
