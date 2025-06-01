#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>

using harmonics::ActivationFunction;
using harmonics::HTensor;
using harmonics::LayerFunction;
using harmonics::LossFunction;

struct IdentityActivation : ActivationFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

struct ZeroLoss : LossFunction {
    HTensor operator()(const HTensor&, const HTensor&) const override { return HTensor{}; }
};

struct IdentityLayer : LayerFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

TEST(FunctionRegistryTest, RegistersAndRetrieves) {
    auto act = std::make_shared<IdentityActivation>();
    auto loss = std::make_shared<ZeroLoss>();
    auto layer = std::make_shared<IdentityLayer>();
    harmonics::registerActivation("id", act);
    harmonics::registerLoss("zero", loss);
    harmonics::registerLayer("identity", layer);

    const auto& act_ref = harmonics::getActivation("id");
    const auto& loss_ref = harmonics::getLoss("zero");
    const auto& layer_ref = harmonics::getLayer("identity");

    HTensor t{};
    auto r = act_ref(t);
    auto l = loss_ref(t, t);
    auto o = layer_ref(t);
    (void)r;
    (void)l;
    (void)o;
    EXPECT_EQ(&act_ref, act.get());
    EXPECT_EQ(&loss_ref, loss.get());
    EXPECT_EQ(&layer_ref, layer.get());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
