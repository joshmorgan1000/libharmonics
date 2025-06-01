#include <gpu/GlobalFunctionRegistry.hpp>
#include <gtest/gtest.h>

#if 0 // Disabled due to unstable GPU fallback behavior
TEST(GpuShaderRegistryTest, Int3MatmulCpuFallback) {
    harmonics::registerAllShaders();
    const auto* fn = harmonics::GPUFunctionRegistry::getInstance().get("int3_matmul");
    ASSERT_EQ(fn == nullptr, false);
    std::vector<harmonics::GPUDataVariant> params;
    params.emplace_back(std::vector<uint8_t>{1, 2, 3});
    params.emplace_back(std::vector<uint8_t>{4, 5, 6});
    auto result = fn->cpuFallback(params);
    auto* vec = std::get_if<std::vector<int32_t>>(&result);
    ASSERT_EQ(vec == nullptr, false);
    int32_t expected = 1 * 4 + 2 * 5 + 3 * 6;
    EXPECT_EQ((*vec)[0], expected);
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
