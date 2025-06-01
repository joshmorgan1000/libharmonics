#define HARMONICS_HAS_CUDA 0
#include <gtest/gtest.h>
#include <harmonics/cuda_adapter.hpp>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

TEST(CudaDeviceTest, OverrideDeviceIndex) {
    harmonics::set_cuda_device_index(2);
    EXPECT_EQ(harmonics::cuda_device_index(), 2u);
}

TEST(CudaDeviceTest, DeploymentDescriptorSelectsDevice) {
    harmonics::set_cuda_device_index(0);
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    harmonics::DeploymentDescriptor desc;
    desc.backend = harmonics::Backend::GPU;
    desc.gpu_device_index = 5;
    harmonics::CycleRuntime rt{g, harmonics::make_auto_policy(), desc};
    EXPECT_EQ(harmonics::cuda_device_index(), 5u);
    EXPECT_EQ(rt.backend(), harmonics::Backend::CPU);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
