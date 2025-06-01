#define HARMONICS_HAS_VULKAN 0
#define HARMONICS_HAS_OPENCL 0

#include <gtest/gtest.h>
#include <harmonics/deployment.hpp>

TEST(AcceleratorSelectionTest, NoneAvailableReturnsCpu) {
    EXPECT_EQ(harmonics::select_accelerator_backend(), harmonics::Backend::CPU);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
