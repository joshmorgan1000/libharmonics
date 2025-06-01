#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/graph_diff.hpp>
#include <harmonics/graph_edit.hpp>
#include <harmonics/parser.hpp>

TEST(GraphDiffTest, DiffAndMerge) {
    const char* base_src = "producer p; consumer c; cycle { p -> c; }";
    const char* upd_src = "producer p; consumer c; layer l; cycle { p -> l; l -> c; }";
    harmonics::Parser p1{base_src};
    auto ast1 = p1.parse_declarations();
    auto g1 = harmonics::build_graph(ast1);

    harmonics::Parser p2{upd_src};
    auto ast2 = p2.parse_declarations();
    auto g2 = harmonics::build_graph(ast2);

    auto diff = harmonics::diff_graphs(g1, g2);
    harmonics::apply_diff(g1, diff);

    EXPECT_EQ(harmonics::graph_digest(g1), harmonics::graph_digest(g2));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
