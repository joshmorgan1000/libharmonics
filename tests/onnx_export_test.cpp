#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/onnx_export.hpp>
#include <harmonics/parser.hpp>
#include <onnx/onnx_pb.h>

TEST(OnnxExportTest, SimpleGraphExport) {
    const char* src = R"(
        producer p {1};
        consumer c {1};
        layer l;
        cycle { p -> l; l -> c; }
    )";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);

    auto model = harmonics::export_onnx_model(g);
    const auto& inits = model.graph().initializer();
    // expect 3 nodes + 2 edges
    EXPECT_EQ(inits.size(), 5);
    bool found_node_p = false;
    bool found_edge = false;
    for (const auto& t : inits) {
        if (t.name() == "node:p") {
            found_node_p = true;
            EXPECT_EQ(t.int32_data_size(), 2);
            EXPECT_EQ(t.int32_data(0), 0);
        }
        if (t.name() == "edge:p:l") {
            found_edge = true;
            EXPECT_EQ(t.int32_data_size(), 1);
            EXPECT_EQ(t.int32_data(0), 0);
        }
    }
    EXPECT_EQ(found_node_p, true);
    EXPECT_EQ(found_edge, true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
