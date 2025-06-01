#define HARMONICS_HAS_VULKAN 0
#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/vulkan_adapter.hpp>

TEST(VulkanDeviceTest, OverrideDeviceIndex) {
    harmonics::set_vulkan_device_index(3);
    EXPECT_EQ(harmonics::vulkan_device_index(), 3u);
}

TEST(VulkanDeviceTest, DeploymentDescriptorSelectsDevice) {
    harmonics::set_vulkan_device_index(0);
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    harmonics::DeploymentDescriptor desc;
    desc.backend = harmonics::Backend::GPU;
    desc.gpu_device_index = 7;
    harmonics::CycleRuntime rt{g, harmonics::make_auto_policy(), desc};
    EXPECT_EQ(harmonics::vulkan_device_index(), 7u);
    EXPECT_EQ(rt.backend(), harmonics::Backend::CPU);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
