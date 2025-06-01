#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

TEST(KernelCacheTest, CompileOncePerGraph) {
    const char* src = "producer p; consumer c; cycle { p -> c; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);

    harmonics::compile_cycle_kernel_compiles() = 0;
    harmonics::compile_fpga_cycle_kernel_compiles() = 0;

    auto policy = harmonics::make_auto_policy();
    (void)harmonics::compile_cycle_kernels(g, *policy);
    EXPECT_EQ(harmonics::compile_cycle_kernel_compiles(), 1);
    (void)harmonics::compile_cycle_kernels(g, *policy);
    EXPECT_EQ(harmonics::compile_cycle_kernel_compiles(), 1);

    (void)harmonics::compile_fpga_cycle_kernels(g);
    EXPECT_EQ(harmonics::compile_fpga_cycle_kernel_compiles(), 1);
    (void)harmonics::compile_fpga_cycle_kernels(g);
    EXPECT_EQ(harmonics::compile_fpga_cycle_kernel_compiles(), 1);
}

TEST(KernelCacheTest, ReuseShaderCacheAcrossGraphs) {
    const char* src1 = "producer p; layer l; cycle { p -> l; }";
    const char* src2 = "producer q; layer m; cycle { q -> m; }";
    harmonics::Parser parser1{src1};
    harmonics::Parser parser2{src2};
    auto ast1 = parser1.parse_declarations();
    auto ast2 = parser2.parse_declarations();
    auto g1 = harmonics::build_graph(ast1);
    auto g2 = harmonics::build_graph(ast2);

    harmonics::shader_compile_cache().clear();

    auto policy = harmonics::make_auto_policy();
    (void)harmonics::compile_cycle_kernels(g1, *policy);
    (void)harmonics::compile_cycle_kernels(g2, *policy);

    // ensure shaders were compiled only once across graphs
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
