#include <cmath>
#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/shaders.hpp>

using harmonics::HTensor;

TEST(ActivationLibraryTest, GeluSeluPrelu) {
    harmonics::register_builtin_shaders();

    HTensor t{HTensor::DType::Float32, {3}};
    t.data().resize(3 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = -1.0f;
    p[1] = 0.0f;
    p[2] = 2.0f;

    const auto& gelu = harmonics::getActivation("gelu");
    auto g = gelu(t);
    const float* gv = reinterpret_cast<const float*>(g.data().data());
    for (int i = 0; i < 3; ++i) {
        float v = p[i];
        float expected = 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
        EXPECT_EQ(gv[i], expected);
    }

    const auto& selu = harmonics::getActivation("selu");
    auto s = selu(t);
    const float* sv = reinterpret_cast<const float*>(s.data().data());
    constexpr float lambda = 1.050701f;
    constexpr float alpha = 1.67326f;
    for (int i = 0; i < 3; ++i) {
        float v = p[i];
        float expected = v > 0.0f ? lambda * v : lambda * (alpha * (std::exp(v) - 1.0f));
        EXPECT_EQ(sv[i], expected);
    }

    const auto& prelu = harmonics::getActivation("prelu");
    auto pr = prelu(t);
    const float* pv = reinterpret_cast<const float*>(pr.data().data());
    for (int i = 0; i < 3; ++i) {
        float v = p[i];
        float expected = v > 0.0f ? v : 0.25f * v;
        EXPECT_EQ(pv[i], expected);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
