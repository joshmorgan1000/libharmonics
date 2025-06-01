#include <cstring>
#include <gtest/gtest.h>
#include <harmonics/fpga_backend.hpp>

TEST(FpgaBackendTest, DeviceRoundTripPreservesTensor) {
    harmonics::HTensor t{harmonics::HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    auto* p = reinterpret_cast<float*>(t.data().data());
    p[0] = 1.0f;
    p[1] = 2.0f;

    auto dev = harmonics::fpga_to_device(t);
    auto host = harmonics::fpga_to_host(dev);

    EXPECT_EQ(host.dtype(), t.dtype());
    EXPECT_EQ(host.shape(), t.shape());
    EXPECT_EQ(host.data().size(), t.data().size());
    EXPECT_EQ(std::memcmp(host.data().data(), t.data().data(), t.data().size()), 0);
}

TEST(FpgaBackendTest, SelectFpgaBackendMatchesBuild) {
#if HARMONICS_HAS_OPENCL
    EXPECT_NE(harmonics::select_fpga_backend(), harmonics::FpgaBackend::None);
#else
    EXPECT_EQ(harmonics::select_fpga_backend(), harmonics::FpgaBackend::None);
#endif
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
