#include <gpu/GlobalFunctionRegistry.hpp>
#include <gtest/gtest.h>

TEST(GpuKernelApiTest, GetKernelReturnsData) {
    harmonics::registerAllShaders();
    auto& reg = harmonics::GPUFunctionRegistry::getInstance();
    auto spv = reg.getKernel("l2_distance");
    ASSERT_EQ(spv.empty(), false);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
