#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/partition.hpp>

using harmonics::auto_partition;
using harmonics::Backend;
using harmonics::build_graph;
using harmonics::DeploymentDescriptor;
using harmonics::Parser;

TEST(AutoPartitionTest, EvenSplit) {
    const char* src = "producer p{1}; consumer c{1}; layer l1; layer l2; layer l3; layer l4; cycle "
                      "{ p -> l1; l1 -> l2; l2 -> l3; l3 -> l4; l4 -> c; }";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    DeploymentDescriptor d;
    d.partitions.resize(2);
    auto parts = auto_partition(g, d);
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0].layers.size(), 2u);
    EXPECT_EQ(parts[1].layers.size(), 2u);
}

TEST(AutoPartitionTest, HeterogeneousSplit) {
    const char* src = "producer p{1}; consumer c{1}; layer l1; layer l2; layer l3; layer l4; cycle "
                      "{ p -> l1; l1 -> l2; l2 -> l3; l3 -> l4; l4 -> c; }";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    DeploymentDescriptor d;
    d.partitions.resize(2);
    d.partitions[0].backend = Backend::GPU;
    d.partitions[1].backend = Backend::CPU;
    auto parts = auto_partition(g, d);
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0].layers.size(), 3u);
    EXPECT_EQ(parts[1].layers.size(), 1u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
