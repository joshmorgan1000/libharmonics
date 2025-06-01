#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>

struct ReluActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override {
        ++calls;
        return x;
    }
    mutable int calls{0};
};

struct SigmoidActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override {
        ++calls;
        return x;
    }
    mutable int calls{0};
};

struct CrossEntropyLoss : harmonics::LossFunction {
    harmonics::HTensor operator()(const harmonics::HTensor&,
                                  const harmonics::HTensor&) const override {
        ++calls;
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}};
    }
    mutable int calls{0};
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

TEST(MnistTrainingCycle, FitProducesWeightsAndPrecisionBits) {
    const char* src = R"(
    harmonic mnist_train_cycle {
      producer img {1};
      producer lbl {1} 1/1 img;
      layer input;
      layer hidden 1/1 input;
      layer output 1/1 hidden;
      cycle {
        img -(relu)-> input -(relu)-> hidden -(sigmoid)-> output;
        output <-(cross_entropy)- lbl;
      }
    }
    )";
    harmonics::Parser parser{src};
    auto ast = parser.parse_harmonic();
    auto g = harmonics::build_graph(ast.decls);

    auto prod = std::make_shared<FixedProducer>(1);
    auto lbl = std::make_shared<FixedProducer>(1);
    g.bindProducer("img", prod);
    g.bindProducer("lbl", lbl);

    auto relu = std::make_shared<ReluActivation>();
    auto sigmoid = std::make_shared<SigmoidActivation>();
    auto ce = std::make_shared<CrossEntropyLoss>();
    harmonics::registerActivation("relu", relu);
    harmonics::registerActivation("sigmoid", sigmoid);
    harmonics::registerLoss("cross_entropy", ce);

    auto state = g.fit(1, harmonics::make_auto_policy());

    EXPECT_EQ(relu->calls, 2);
    EXPECT_EQ(sigmoid->calls, 1);
    EXPECT_EQ(ce->calls, 1);
    EXPECT_EQ(state.weights.size(), g.layers.size());
    EXPECT_EQ(state.weights[2].shape().size(), 0u); // params not updated yet
    EXPECT_EQ(state.precision_bits.size(), g.layers.size());
#if HARMONICS_HAS_VULKAN
    int expected_bits = 16;
#else
    int expected_bits = 32;
#endif
    EXPECT_EQ(state.precision_bits[0], expected_bits);
    EXPECT_EQ(state.precision_bits[1], expected_bits);
    EXPECT_EQ(state.precision_bits[2], expected_bits);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
