#define HARMONICS_HAS_QUANTUM_HW 0
#include <gtest/gtest.h>
#include <harmonics/quantum_hardware.hpp>

TEST(QuantumDeviceTest, OverrideDeviceIndex) {
    harmonics::set_quantum_device_index(2);
    EXPECT_EQ(harmonics::quantum_device_index(), 2u);
}

TEST(QuantumDeviceTest, DeviceCountZeroWithoutRuntime) {
    EXPECT_EQ(harmonics::quantum_device_count(), 0u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
