#define HARMONICS_HAS_CUDA 1
#define HARMONICS_HAS_VULKAN 0

#include <cstring>
#include <gtest/gtest.h>
#include <harmonics/gpu_backend.hpp>

TEST(CudaBackendEnabledTest, DeviceRoundTripPreservesTensor) {
    harmonics::HTensor t{harmonics::HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    auto* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 2.0f;

    auto dev = harmonics::to_device(t);
    auto host = harmonics::to_host(dev);

    EXPECT_EQ(host.dtype(), t.dtype());
    EXPECT_EQ(host.shape(), t.shape());
    EXPECT_EQ(host.data().size(), t.data().size());
    EXPECT_EQ(std::memcmp(host.data().data(), t.data().data(), t.data().size()), 0);
}

TEST(CudaBackendEnabledTest, SelectGpuBackendCuda) {
    EXPECT_EQ(harmonics::select_gpu_backend(), harmonics::GpuBackend::Cuda);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
