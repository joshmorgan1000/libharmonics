#define HARMONICS_HAS_OPENCL 0
#include <gtest/gtest.h>
#include <harmonics/opencl_adapter.hpp>

TEST(OpenCLDeviceTest, OverrideDeviceIndex) {
    harmonics::set_opencl_device_index(4);
    EXPECT_EQ(harmonics::opencl_device_index(), 4u);
}

TEST(OpenCLDeviceTest, DeviceCountZeroWithoutRuntime) {
    EXPECT_EQ(harmonics::opencl_device_count(), 0u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
