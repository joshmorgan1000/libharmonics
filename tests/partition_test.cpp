#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/partition.hpp>

using harmonics::build_graph;
using harmonics::Parser;
using harmonics::partition_by_layer;

TEST(PartitionTest, SplitGraphByLayer) {
    const char* src = R"(
producer p {1};
consumer c {1};
layer l1;
layer l2;
cycle {
  p -> l1;
  l1 -> l2;
  l2 -> c;
  l2 <-(loss)- c;
}
)";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);

    auto parts = partition_by_layer(g, 1);
    const auto& g1 = parts.first;
    const auto& g2 = parts.second;

    EXPECT_EQ(g1.layers.size(), 1u);
    EXPECT_EQ(g2.layers.size(), 1u);
    EXPECT_EQ(g1.consumers.size(), 2u); // original + boundary
    EXPECT_EQ(g2.producers.size(), 2u); // original + boundary

    // Check that boundary flow was created
    EXPECT_EQ(g1.cycle.size(), 2u);
    EXPECT_EQ(g1.cycle[1].source.kind, harmonics::NodeKind::Layer);
    EXPECT_EQ(g1.cycle[1].arrows[0].target.kind, harmonics::NodeKind::Consumer);

    bool found_boundary = false;
    for (const auto& line : g2.cycle) {
        if (line.source.kind == harmonics::NodeKind::Producer && line.source.index == 1) {
            found_boundary = true;
            EXPECT_EQ(line.arrows[0].target.kind, harmonics::NodeKind::Layer);
        }
    }
    EXPECT_EQ(found_boundary, true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
