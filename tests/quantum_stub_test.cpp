#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/quantum_stub.hpp>

TEST(QuantumStubTest, CircuitFromSimpleGraph) {
    const char* src = R"(
        producer p {1};
        consumer c {1};
        layer l;
        cycle { p -> l; l -> c; }
    )";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto qc = harmonics::map_to_quantum(g);
    EXPECT_EQ(qc.qubit_count, 3);
    EXPECT_EQ(qc.ops.size(), 3u);
    EXPECT_EQ(qc.ops[0].gate, harmonics::QubitGate::CX);
    EXPECT_EQ(qc.ops[1].gate, harmonics::QubitGate::CX);
    EXPECT_EQ(qc.ops[2].gate, harmonics::QubitGate::Measure);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
