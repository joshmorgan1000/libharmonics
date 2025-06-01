#include <gtest/gtest.h>
#include <harmonics/quantum_backend.hpp>

TEST(QuantumBackendTest, XGateProducesOne) {
    harmonics::QuantumCircuit qc;
    qc.qubit_count = 1;
    qc.ops.push_back({harmonics::QubitGate::X, {0}});
    qc.ops.push_back({harmonics::QubitGate::Measure, {0}});
    auto result = harmonics::simulate(qc);
    ASSERT_EQ(result.measurements.size(), 1u);
    EXPECT_EQ(result.measurements[0], 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
