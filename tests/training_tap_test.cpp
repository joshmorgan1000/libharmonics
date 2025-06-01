#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

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

struct DummyLoss : harmonics::LossFunction {
    harmonics::HTensor operator()(const harmonics::HTensor&,
                                  const harmonics::HTensor&) const override {
        ++calls;
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}};
    }
    mutable int calls{0};
};

TEST(TrainingTapTest, BackwardArrowInvokesLoss) {
    const char* src =
        "producer p {1}; producer label {1}; layer l; cycle { p -> l; l <-(dummy)- label; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);

    auto prod = std::make_shared<FixedProducer>(1);
    auto lbl = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    g.bindProducer("label", lbl);

    auto loss = std::make_shared<DummyLoss>();
    harmonics::registerLoss("dummy", loss);

    harmonics::CycleRuntime rt{g};
    rt.forward();

    EXPECT_EQ(loss->calls, 1);
    EXPECT_EQ(rt.state().weights[0].shape().size(), 1u);
    EXPECT_EQ(rt.state().weights[0].shape()[0], 1u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
