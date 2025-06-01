#include <gtest/gtest.h>
#include <harmonics/int8_activations.hpp>

using harmonics::HTensor;

TEST(Int8ActivationTest, ReluHardSigmoidSoftmax) {
    harmonics::register_int8_lut_activations();

    HTensor t{HTensor::DType::UInt8, {3}};
    t.data().resize(3);
    auto* p = reinterpret_cast<int8_t*>(t.data().data());
    p[0] = -10;
    p[1] = 0;
    p[2] = 20;

    const auto& relu = harmonics::getActivation("int8_relu");
    auto r = relu(t);
    const int8_t* rv = reinterpret_cast<const int8_t*>(r.data().data());
    EXPECT_EQ(rv[0], 0);
    EXPECT_EQ(rv[1], 0);
    EXPECT_EQ(rv[2], 20);

    const auto& hs = harmonics::getActivation("int8_hardsigmoid");
    auto h = hs(t);
    const int8_t* hv = reinterpret_cast<const int8_t*>(h.data().data());
    EXPECT_GE(hv[0], 0);
    EXPECT_GE(hv[1], 0);
    EXPECT_GE(hv[2], hv[1]);

    const auto& sm = harmonics::getActivation("int8_softmax");
    auto smt = sm(t);
    const int8_t* sv = reinterpret_cast<const int8_t*>(smt.data().data());
    int sum = sv[0] + sv[1] + sv[2];
    EXPECT_NE(sum, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
