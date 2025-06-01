#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/introspection.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>

struct IdActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

TEST(IntrospectionTest, LayerInfoAndWeightsAccess) {
    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());

    auto info = harmonics::get_layer_info(g);
    EXPECT_EQ(info.size(), 1u);
    EXPECT_EQ(info[0].name, "l");
    EXPECT_EQ(info[0].width, 0u);

    harmonics::CycleRuntime rt{g};
    auto& w = harmonics::layer_weights(rt, "l");
    w = harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}};
    w.data().resize(sizeof(float));

    const auto& cw = harmonics::layer_weights(rt, "l");
    EXPECT_EQ(cw.shape().size(), 1u);
    EXPECT_EQ(cw.shape()[0], 1u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
