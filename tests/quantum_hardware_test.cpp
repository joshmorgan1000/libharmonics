#include <gtest/gtest.h>
#include <harmonics/quantum_hardware.hpp>

TEST(QuantumHardwareTest, UsesExternalLibraryWhenAvailable) {
    setenv("HARMONICS_ENABLE_QUANTUM_HW", "1", 1);
    setenv("HARMONICS_QUANTUM_HW_LIB", "./libquantum_hw.so", 1);
    EXPECT_TRUE(harmonics::quantum_hardware_runtime_available());
    harmonics::QuantumCircuit qc;
    qc.qubit_count = 1;
    qc.ops.push_back({harmonics::QubitGate::X, {0}});
    qc.ops.push_back({harmonics::QubitGate::Measure, {0}});
    auto result = harmonics::execute_on_hardware(qc);
    unsetenv("HARMONICS_ENABLE_QUANTUM_HW");
    unsetenv("HARMONICS_QUANTUM_HW_LIB");
    ASSERT_EQ(result.measurements.size(), 1u);
    EXPECT_EQ(result.measurements[0], 1);
}

TEST(QuantumHardwareTest, ReportsUnavailableWhenDisabled) {
    unsetenv("HARMONICS_ENABLE_QUANTUM_HW");
    unsetenv("HARMONICS_QUANTUM_HW_LIB");
    EXPECT_FALSE(harmonics::quantum_hardware_runtime_available());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
