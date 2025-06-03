#include <algorithm>
#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/layers.hpp>

using harmonics::HTensor;

TEST(LayerFunctionsTest, ConvolutionNormalizationAttention) {
    harmonics::set_convolution_kernel(3);
    harmonics::set_norm_epsilon(1e-6f);
    harmonics::set_attention_temperature(1.0f);
    harmonics::register_builtin_layers();

    // prepare input tensor
    HTensor t{HTensor::DType::Float32, {5}};
    t.data().resize(5 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    p[3] = 4.0f;
    p[4] = 5.0f;

    const auto& conv = harmonics::getLayer("conv");
    const auto& norm = harmonics::getLayer("norm");
    const auto& attn = harmonics::getLayer("attention");

    auto c = conv(t);
    EXPECT_EQ(c.shape().size(), 1u);
    EXPECT_EQ(c.shape()[0], 3u);
    const float* cv = reinterpret_cast<const float*>(c.data().data());
    EXPECT_EQ(cv[0], 6.0f);
    EXPECT_EQ(cv[1], 9.0f);
    EXPECT_EQ(cv[2], 12.0f);

    auto n = norm(t);
    EXPECT_EQ(n.shape(), t.shape());
    const float* nv = reinterpret_cast<const float*>(n.data().data());
    float norm_sum = 0.0f;
    for (int i = 0; i < 5; ++i)
        norm_sum += p[i] * p[i];
    norm_sum = std::sqrt(norm_sum) + 1e-6f;
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(nv[i], p[i] / norm_sum);

    auto a = attn(t);
    EXPECT_EQ(a.shape(), t.shape());
    const float* av = reinterpret_cast<const float*>(a.data().data());
    // compute expected attention scalar
    float maxv = p[0];
    for (int i = 1; i < 5; ++i)
        if (p[i] > maxv)
            maxv = p[i];
    float sum = 0.0f;
    std::vector<float> weights(5);
    for (int i = 0; i < 5; ++i) {
        weights[i] = std::exp(p[i] - maxv);
        sum += weights[i];
    }
    float attn_scalar = 0.0f;
    for (int i = 0; i < 5; ++i) {
        weights[i] /= sum;
        attn_scalar += weights[i] * p[i];
    }
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(av[i], attn_scalar);
}

TEST(LayerFunctionsTest, MultiHeadAttention) {
    harmonics::set_attention_heads(2);
    harmonics::register_builtin_layers();

    HTensor t{HTensor::DType::Float32, {4}};
    t.data().resize(4 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    p[3] = 4.0f;

    const auto& mh = harmonics::getLayer("multihead_attention");
    auto out = mh(t);
    EXPECT_EQ(out.shape(), t.shape());
    const float* ov = reinterpret_cast<const float*>(out.data().data());

    float max1 = std::max(p[0], p[1]);
    float w0 = std::exp(p[0] - max1);
    float w1 = std::exp(p[1] - max1);
    float sum1 = w0 + w1;
    w0 /= sum1;
    w1 /= sum1;
    float attn1 = w0 * p[0] + w1 * p[1];

    float max2 = std::max(p[2], p[3]);
    float w2 = std::exp(p[2] - max2);
    float w3 = std::exp(p[3] - max2);
    float sum2 = w2 + w3;
    w2 /= sum2;
    w3 /= sum2;
    float attn2 = w2 * p[2] + w3 * p[3];

    EXPECT_EQ(std::fabs(ov[0] - attn1) < 1e-5f, true);
    EXPECT_EQ(std::fabs(ov[1] - attn1) < 1e-5f, true);
    EXPECT_EQ(std::fabs(ov[2] - attn2) < 1e-5f, true);
    EXPECT_EQ(std::fabs(ov[3] - attn2) < 1e-5f, true);
}

TEST(LayerFunctionsTest, CrossAttention) {
    harmonics::set_attention_temperature(1.0f);
    harmonics::register_builtin_layers();

    HTensor t{HTensor::DType::Float32, {2, 3}};
    t.data().resize(6 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    p[3] = 4.0f;
    p[4] = 5.0f;
    p[5] = 6.0f;

    const auto& ca = harmonics::getLayer("cross_attention");
    auto out = ca(t);
    EXPECT_EQ(out.shape(), t.shape());
    const float* ov = reinterpret_cast<const float*>(out.data().data());
    for (int r = 0; r < 2; ++r) {
        const float* row = p + r * 3;
        float maxv = std::max({row[0], row[1], row[2]});
        float w0 = std::exp(row[0] - maxv);
        float w1 = std::exp(row[1] - maxv);
        float w2 = std::exp(row[2] - maxv);
        float sum = w0 + w1 + w2;
        w0 /= sum;
        w1 /= sum;
        w2 /= sum;
        float attn = w0 * row[0] + w1 * row[1] + w2 * row[2];
        for (int c = 0; c < 3; ++c)
            EXPECT_FLOAT_EQ(ov[r * 3 + c], attn);
    }
}

TEST(LayerFunctionsTest, PoolingLayers) {
    harmonics::set_pool_window(2);
    harmonics::register_builtin_layers();

    HTensor t{HTensor::DType::Float32, {4}};
    t.data().resize(4 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 3.0f;
    p[2] = 2.0f;
    p[3] = 4.0f;

    const auto& maxp = harmonics::getLayer("max_pool");
    const auto& avgp = harmonics::getLayer("avg_pool");

    auto m = maxp(t);
    EXPECT_EQ(m.shape().size(), 1u);
    EXPECT_EQ(m.shape()[0], 2u);
    const float* mv = reinterpret_cast<const float*>(m.data().data());
    EXPECT_EQ(mv[0], 3.0f);
    EXPECT_EQ(mv[1], 4.0f);

    auto a = avgp(t);
    EXPECT_EQ(a.shape().size(), 1u);
    EXPECT_EQ(a.shape()[0], 2u);
    const float* av = reinterpret_cast<const float*>(a.data().data());
    EXPECT_FLOAT_EQ(av[0], 2.0f);
    EXPECT_FLOAT_EQ(av[1], 3.0f);
}

TEST(LayerFunctionsTest, DropoutLayer) {
    harmonics::register_builtin_layers();

    HTensor t{HTensor::DType::Float32, {4}};
    t.data().resize(4 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    p[3] = 4.0f;

    harmonics::registerLayer("dropout", std::make_shared<harmonics::DropoutLayer>(0.f));
    auto out = harmonics::getLayer("dropout")(t);
    EXPECT_EQ(out.shape(), t.shape());
    const float* ov = reinterpret_cast<const float*>(out.data().data());
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(ov[i], p[i]);

    harmonics::registerLayer("dropout", std::make_shared<harmonics::DropoutLayer>(1.f));
    auto out2 = harmonics::getLayer("dropout")(t);
    EXPECT_EQ(out2.shape(), t.shape());
    const float* ov2 = reinterpret_cast<const float*>(out2.data().data());
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(ov2[i], 0.0f);
}

TEST(LayerFunctionsTest, PoolingLayerEdgeCases) {
    harmonics::set_pool_window(1);
    harmonics::register_builtin_layers();

    HTensor t{HTensor::DType::Float32, {3}};
    t.data().resize(3 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 5.0f;
    p[1] = -1.0f;
    p[2] = 2.0f;

    const auto& maxp = harmonics::getLayer("max_pool");
    const auto& avgp = harmonics::getLayer("avg_pool");

    auto m = maxp(t);
    EXPECT_EQ(m.shape(), t.shape());
    const float* mv = reinterpret_cast<const float*>(m.data().data());
    for (int i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(mv[i], p[i]);

    auto a = avgp(t);
    EXPECT_EQ(a.shape(), t.shape());
    const float* av = reinterpret_cast<const float*>(a.data().data());
    for (int i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(av[i], p[i]);

    harmonics::set_pool_window(0);
    harmonics::register_builtin_layers();
    auto disabled = maxp(t);
    EXPECT_EQ(disabled.shape(), t.shape());
    const float* dv = reinterpret_cast<const float*>(disabled.data().data());
    for (int i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(dv[i], p[i]);
}

TEST(LayerFunctionsTest, PoolingWindowThree) {
    harmonics::set_pool_window(3);
    harmonics::register_builtin_layers();

    HTensor t{HTensor::DType::Float32, {4}};
    t.data().resize(4 * sizeof(float));
    float* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 3.0f;
    p[2] = 2.0f;
    p[3] = 4.0f;

    const auto& maxp = harmonics::getLayer("max_pool");
    const auto& avgp = harmonics::getLayer("avg_pool");

    auto m = maxp(t);
    EXPECT_EQ(m.shape().size(), 1u);
    EXPECT_EQ(m.shape()[0], 1u);
    const float* mv = reinterpret_cast<const float*>(m.data().data());
    EXPECT_FLOAT_EQ(mv[0], std::max({p[0], p[1], p[2]}));

    auto a = avgp(t);
    EXPECT_EQ(a.shape().size(), 1u);
    EXPECT_EQ(a.shape()[0], 1u);
    const float* av = reinterpret_cast<const float*>(a.data().data());
    EXPECT_FLOAT_EQ(av[0], (p[0] + p[1] + p[2]) / 3.0f);
}

TEST(LayerFunctionsTest, Int8Layers) {
    harmonics::set_convolution_kernel(3);
    harmonics::set_pool_window(2);
    harmonics::register_builtin_layers();

    HTensor t{HTensor::DType::UInt8, {4}};
    t.data().resize(4);
    int8_t* p = reinterpret_cast<int8_t*>(t.data().data());
    p[0] = 1;
    p[1] = 3;
    p[2] = 2;
    p[3] = 4;

    const auto& conv = harmonics::getLayer("conv");
    auto c = conv(t);
    EXPECT_EQ(c.dtype(), HTensor::DType::UInt8);
    EXPECT_EQ(c.shape().size(), 1u);
    EXPECT_EQ(c.shape()[0], 2u);
    const int8_t* cv = reinterpret_cast<const int8_t*>(c.data().data());
    EXPECT_EQ(cv[0], 6);
    EXPECT_EQ(cv[1], 9);

    const auto& maxp = harmonics::getLayer("max_pool");
    auto m = maxp(t);
    EXPECT_EQ(m.dtype(), HTensor::DType::UInt8);
    const int8_t* mv = reinterpret_cast<const int8_t*>(m.data().data());
    EXPECT_EQ(mv[0], 3);
    EXPECT_EQ(mv[1], 4);

    const auto& avgp = harmonics::getLayer("avg_pool");
    auto a = avgp(t);
    EXPECT_EQ(a.dtype(), HTensor::DType::UInt8);
    const int8_t* av = reinterpret_cast<const int8_t*>(a.data().data());
    EXPECT_EQ(av[0], 2);
    EXPECT_EQ(av[1], 3);
}

TEST(LayerFunctionsTest, DropoutLayerTypes) {
    harmonics::register_builtin_layers();

    HTensor t64{HTensor::DType::Float64, {2}};
    t64.data().resize(2 * sizeof(double));
    double* dp = reinterpret_cast<double*>(t64.data().data());
    dp[0] = 1.0;
    dp[1] = 2.0;

    harmonics::registerLayer("dropout", std::make_shared<harmonics::DropoutLayer>(1.f));
    auto out64 = harmonics::getLayer("dropout")(t64);
    EXPECT_EQ(out64.dtype(), HTensor::DType::Float64);
    const double* dv = reinterpret_cast<const double*>(out64.data().data());
    EXPECT_EQ(dv[0], 0.0);
    EXPECT_EQ(dv[1], 0.0);

    HTensor ti32{HTensor::DType::Int32, {3}};
    ti32.data().resize(3 * sizeof(std::int32_t));
    std::int32_t* ip = reinterpret_cast<std::int32_t*>(ti32.data().data());
    ip[0] = 7;
    ip[1] = 8;
    ip[2] = 9;

    auto outi32 = harmonics::getLayer("dropout")(ti32);
    EXPECT_EQ(outi32.dtype(), HTensor::DType::Int32);
    const std::int32_t* iout = reinterpret_cast<const std::int32_t*>(outi32.data().data());
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(iout[i], 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
