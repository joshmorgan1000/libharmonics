#include <cmath>
#include <gpu/GlobalFunctionRegistry.hpp>
#include <gtest/gtest.h>

TEST(GpuShaderRegistryTest, L2DistanceCpuFallback) {
    harmonics::registerAllShaders();
    const auto* fn = harmonics::GPUFunctionRegistry::getInstance().get("l2_distance");
    ASSERT_EQ(fn == nullptr, false);
    ASSERT_EQ(fn->shader.empty(), false);
    std::vector<harmonics::GPUDataVariant> params;
    params.emplace_back(std::vector<float>{1.0f, 2.0f, 3.0f});
    params.emplace_back(std::vector<float>{4.0f, 2.0f, 0.0f});
    auto result = fn->cpuFallback(params);
    auto* vec = std::get_if<std::vector<float>>(&result);
    ASSERT_EQ(vec == nullptr, false);
    float expected = std::sqrt(9.0f + 0.0f + 9.0f);
    ASSERT_EQ(std::fabs((*vec)[0] - expected) < 1e-5f, true);
}

TEST(GpuShaderRegistryTest, RegisterAddsShaders) {
    harmonics::registerAllShaders();
    const auto& all = harmonics::GPUFunctionRegistry::getInstance().all();
    ASSERT_EQ(all.empty(), false);
}

TEST(GpuShaderRegistryTest, MissingShaderReturnsNull) {
    harmonics::registerAllShaders();
    const auto* fn = harmonics::GPUFunctionRegistry::getInstance().get("missing");
    ASSERT_EQ(fn == nullptr, true);
}

TEST(GpuShaderRegistryTest, L2DistanceMismatchedSizeThrows) {
    harmonics::registerAllShaders();
    const auto* fn = harmonics::GPUFunctionRegistry::getInstance().get("l2_distance");
    ASSERT_EQ(fn == nullptr, false);
    std::vector<harmonics::GPUDataVariant> params;
    params.emplace_back(std::vector<float>{1.0f, 2.0f});
    params.emplace_back(std::vector<float>{1.0f});
    bool threw = false;
    try {
        (void)fn->cpuFallback(params);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_EQ(threw, true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
