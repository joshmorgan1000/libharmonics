#define HARMONICS_HAS_OPENCL 1
#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

TEST(FpgaKernelCompiler, BuildsOps) {
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto kernels = harmonics::compile_fpga_cycle_kernels(g);
    EXPECT_EQ(kernels.ops.size(), 1u);
    EXPECT_EQ(kernels.ops[0].source.kind, harmonics::NodeKind::Producer);
    EXPECT_EQ(kernels.ops[0].target.kind, harmonics::NodeKind::Layer);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
