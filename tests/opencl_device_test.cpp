#define HARMONICS_HAS_OPENCL 0
#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/opencl_adapter.hpp>
#include <harmonics/parser.hpp>

TEST(OpenCLDeviceTest, OverrideDeviceIndex) {
    harmonics::set_opencl_device_index(4);
    EXPECT_EQ(harmonics::opencl_device_index(), 4u);
}

TEST(OpenCLDeviceTest, DeviceCountZeroWithoutRuntime) {
    EXPECT_EQ(harmonics::opencl_device_count(), 0u);
}

TEST(OpenCLDeviceTest, DeploymentDescriptorSelectsDevice) {
    harmonics::set_opencl_device_index(0);
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    harmonics::DeploymentDescriptor desc;
    desc.backend = harmonics::Backend::FPGA;
    desc.fpga_device_index = 3;
    harmonics::CycleRuntime rt{g, harmonics::make_auto_policy(), desc};
    EXPECT_EQ(harmonics::opencl_device_index(), 3u);
    EXPECT_EQ(rt.backend(), harmonics::Backend::CPU);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
