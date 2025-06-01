#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

TEST(CycleRuntimeTest, InitializesStateSizes) {
    const char* src = "producer p; consumer c; layer l;";
    harmonics::Parser p{src};
    auto ast = p.parse_declarations();
    auto g = harmonics::build_graph(ast);

    harmonics::CycleRuntime runtime{g};
    const auto& state = runtime.state();
    EXPECT_EQ(state.producer_tensors.size(), g.producers.size());
    EXPECT_EQ(state.consumer_tensors.size(), g.consumers.size());
    EXPECT_EQ(state.layer_tensors.size(), g.layers.size());
    EXPECT_EQ(state.weights.size(), g.layers.size());
}

struct IdActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

struct FixedProducer : harmonics::Producer {
    explicit FixedProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override {
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, shape};
    }
    std::size_t size() const override { return 1; }
    harmonics::HTensor::Shape shape{};
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

TEST(CycleRuntimeTest, ForwardPassCopiesProducerTensor) {
    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    harmonics::CycleRuntime runtime{g};
    runtime.forward();

    EXPECT_EQ(runtime.state().layer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(runtime.state().layer_tensors[0].shape()[0], 1u);
}

TEST(CycleRuntimeTest, BranchingUsesSingleProducerSample) {
    const char* src = "producer p; layer a; layer b; cycle { p -> a; -> b; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<CountingProducer>(1);
    g.bindProducer("p", prod);
    harmonics::CycleRuntime rt{g};
    rt.forward();
    EXPECT_EQ(prod->calls, 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
