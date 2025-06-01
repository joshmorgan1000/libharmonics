#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/graph_edit.hpp>
#include <harmonics/introspection.hpp>
#include <harmonics/parser.hpp>

TEST(GraphEditTest, AddAndRemoveLayerAndFlow) {
    const char* src = "producer p {1}; consumer c; cycle { p -> c; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    EXPECT_EQ(g.layers.size(), 0u);

    harmonics::add_layer(g, "l");
    EXPECT_EQ(g.layers.size(), 1u);

    harmonics::add_flow(g, "p", "l");
    harmonics::add_flow(g, "l", "c");
    EXPECT_EQ(g.cycle.size(), 2u);

    harmonics::remove_flow(g, "p", "c");
    EXPECT_EQ(g.cycle[0].arrows.size(), 1u);

    harmonics::remove_layer(g, "l");
    EXPECT_EQ(g.layers.size(), 0u);

    auto info = harmonics::get_layer_info(g);
    EXPECT_EQ(info.size(), 0u);
}

TEST(GraphEditTest, SyncRuntimeAfterEdit) {
    const char* src = "producer p {1}; consumer c; cycle { p -> c; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);

    harmonics::CycleRuntime rt{g};
    EXPECT_EQ(rt.state().layer_tensors.size(), 0u);

    harmonics::add_layer(g, "l");
    harmonics::add_flow(g, "p", "l");
    harmonics::add_flow(g, "l", "c");

    harmonics::sync_runtime(rt);

    EXPECT_EQ(rt.state().layer_tensors.size(), 1u);
    EXPECT_EQ(rt.state().weights.size(), 1u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
