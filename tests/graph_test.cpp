#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <memory>

using harmonics::build_graph;
using harmonics::Parser;

TEST(GraphTest, BuildGraph) {
    const char* src = R"(
producer a;
consumer b;
layer l;
cycle {
  a -> l;
  l <-(loss)- b;
}
)";
    Parser p{src};
    auto ast = p.parse_declarations();
    auto g = build_graph(ast);
    EXPECT_EQ(g.producers.size(), 1u);
    EXPECT_EQ(g.consumers.size(), 1u);
    EXPECT_EQ(g.layers.size(), 1u);
    EXPECT_EQ(g.cycle.size(), 2u);
    EXPECT_EQ(g.producers[0].name, "a");
    EXPECT_EQ(g.consumers[0].name, "b");
    EXPECT_EQ(g.layers[0].name, "l");
    EXPECT_EQ(g.cycle[0].arrows.size(), 1u);
    EXPECT_EQ(g.cycle[0].arrows[0].backward, false);
    EXPECT_EQ(g.cycle[0].arrows[0].target.kind, harmonics::NodeKind::Layer);
    EXPECT_EQ(g.cycle[1].arrows.size(), 1u);
    EXPECT_EQ(g.cycle[1].arrows[0].backward, true);
    EXPECT_EQ(g.cycle[1].arrows[0].target.kind, harmonics::NodeKind::Consumer);
}

TEST(GraphTest, DuplicateNameFails) {
    const char* src = "producer a; consumer a;";
    Parser p{src};
    auto ast = p.parse_declarations();
    bool threw = false;
    try {
        (void)build_graph(ast);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_EQ(threw, true);
}

struct FixedProducer : harmonics::Producer {
    explicit FixedProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override {
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, shape};
    }
    std::size_t size() const override { return 0; }
    harmonics::HTensor::Shape shape{};
};

TEST(GraphTest, BindProducerStoresBinding) {
    const char* src = "producer p {1};";
    Parser p{src};
    auto ast = p.parse_declarations();
    auto g = build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    EXPECT_EQ(g.producer_bindings[0], prod);
}

TEST(GraphTest, BindProducerShapeMismatch) {
    const char* src = "producer p {2};";
    Parser p{src};
    auto ast = p.parse_declarations();
    auto g = build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    bool threw = false;
    try {
        g.bindProducer("p", prod);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_EQ(threw, true);
}

TEST(GraphTest, RatioWidthPropagation) {
    const char* src = "producer input {4}; layer hidden 1/2 input;";
    Parser p{src};
    auto ast = p.parse_declarations();
    auto g = build_graph(ast);
    EXPECT_EQ(g.layers[0].shape.has_value(), true);
    EXPECT_EQ(*g.layers[0].shape, 2);
}

TEST(GraphTest, RatioChainPropagation) {
    const char* src = "producer a {8}; layer b 1/2 a; layer c 1/2 b;";
    Parser p{src};
    auto ast = p.parse_declarations();
    auto g = build_graph(ast);
    EXPECT_EQ(g.layers.size(), 2u);
    EXPECT_EQ(g.layers[0].shape.has_value(), true);
    EXPECT_EQ(*g.layers[0].shape, 4);
    EXPECT_EQ(g.layers[1].shape.has_value(), true);
    EXPECT_EQ(*g.layers[1].shape, 2);
}

TEST(GraphTest, RatioUnresolvedWhenReferenceMissing) {
    const char* src = "producer a; layer b 1/2 a;";
    Parser p{src};
    auto ast = p.parse_declarations();
    auto g = build_graph(ast);
    EXPECT_EQ(g.layers.size(), 1u);
    EXPECT_EQ(g.layers[0].shape.has_value(), false);
}

TEST(GraphTest, HasTrainingTaps) {
    const char* src = R"(
producer a;
producer b;
layer l;
cycle {
  a -> l;
  l <-(loss)- b;
}
)";
    Parser p{src};
    auto ast = p.parse_declarations();
    auto g = build_graph(ast);
    EXPECT_EQ(g.hasTrainingTaps(), true);
}

TEST(GraphTest, NoTrainingTaps) {
    const char* src = R"(
producer a;
layer l;
cycle {
  a -> l;
}
)";
    Parser p{src};
    auto ast = p.parse_declarations();
    auto g = build_graph(ast);
    EXPECT_EQ(g.hasTrainingTaps(), false);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
