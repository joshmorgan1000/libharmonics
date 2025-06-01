#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/layers.hpp>
#include <vector>

using harmonics::HTensor;

TEST(LayerBehaviorConfig, OverridesDefaults) {
    using namespace harmonics;

    set_convolution_kernel(2);
    set_norm_epsilon(1.0f);
    set_attention_temperature(0.5f);
    register_builtin_layers();

    HTensor t{HTensor::DType::Float32, {3}};
    t.data().resize(3 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;

    const auto& conv = getLayer("conv");
    auto c = conv(t);
    ASSERT_EQ(c.shape()[0], 2u);
    const float* cv = reinterpret_cast<const float*>(c.data().data());
    EXPECT_EQ(cv[0], 3.0f);
    EXPECT_EQ(cv[1], 5.0f);

    const auto& norm = getLayer("norm");
    auto n = norm(t);
    const float* nv = reinterpret_cast<const float*>(n.data().data());
    float norm_sum = std::sqrt(1.f * 1.f + 2.f * 2.f + 3.f * 3.f) + 1.0f;
    EXPECT_EQ(nv[0], 1.0f / norm_sum);
    EXPECT_EQ(nv[1], 2.0f / norm_sum);
    EXPECT_EQ(nv[2], 3.0f / norm_sum);

    const auto& attn = getLayer("attention");
    auto a = attn(t);
    const float* av = reinterpret_cast<const float*>(a.data().data());
    float maxv = 3.0f;
    std::vector<float> weights = {std::exp((1.0f - maxv) / 0.5f), std::exp((2.0f - maxv) / 0.5f),
                                  std::exp((3.0f - maxv) / 0.5f)};
    float sum = weights[0] + weights[1] + weights[2];
    for (auto& w : weights)
        w /= sum;
    float attn_scalar = weights[0] * 1.0f + weights[1] * 2.0f + weights[2] * 3.0f;
    EXPECT_EQ(av[0], attn_scalar);
    EXPECT_EQ(av[1], attn_scalar);
    EXPECT_EQ(av[2], attn_scalar);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
