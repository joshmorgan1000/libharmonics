#include <gtest/gtest.h>
#include <harmonics/parser.hpp>

using harmonics::DeclarationsAST;
using harmonics::Parser;

TEST(ParserTest, ParseDeclarations) {
    const char* src = R"(
producer input {28} 1/1 other;
consumer target {10};
layer hidden 1/2 input;
)";
    Parser p{src};
    DeclarationsAST ast = p.parse_declarations();
    EXPECT_EQ(ast.producers.size(), 1u);
    EXPECT_EQ(ast.producers[0].name, "input");
    EXPECT_EQ(static_cast<bool>(ast.producers[0].shape), true);
    EXPECT_EQ(*ast.producers[0].shape, 28);
    EXPECT_EQ(static_cast<bool>(ast.producers[0].ratio), true);
    EXPECT_EQ(ast.producers[0].ratio->lhs, 1);
    EXPECT_EQ(ast.producers[0].ratio->rhs, 1);
    EXPECT_EQ(ast.producers[0].ratio->ref, "other");

    EXPECT_EQ(ast.consumers.size(), 1u);
    EXPECT_EQ(ast.consumers[0].name, "target");
    EXPECT_EQ(static_cast<bool>(ast.consumers[0].shape), true);
    EXPECT_EQ(*ast.consumers[0].shape, 10);

    EXPECT_EQ(ast.layers.size(), 1u);
    EXPECT_EQ(ast.layers[0].name, "hidden");
    EXPECT_EQ(static_cast<bool>(ast.layers[0].ratio), true);
    EXPECT_EQ(ast.layers[0].ratio->lhs, 1);
    EXPECT_EQ(ast.layers[0].ratio->rhs, 2);
    EXPECT_EQ(ast.layers[0].ratio->ref, "input");
}

TEST(ParserTest, OptionalParts) {
    const char* src = "producer p; consumer c; layer l;";
    Parser p{src};
    auto ast = p.parse_declarations();
    EXPECT_EQ(ast.producers.size(), 1u);
    EXPECT_EQ(ast.producers[0].name, "p");
    EXPECT_EQ(static_cast<bool>(ast.producers[0].shape), false);
    EXPECT_EQ(static_cast<bool>(ast.producers[0].ratio), false);
    EXPECT_EQ(ast.consumers.size(), 1u);
    EXPECT_EQ(ast.consumers[0].name, "c");
    EXPECT_EQ(static_cast<bool>(ast.consumers[0].shape), false);
    EXPECT_EQ(ast.layers.size(), 1u);
    EXPECT_EQ(ast.layers[0].name, "l");
    EXPECT_EQ(static_cast<bool>(ast.layers[0].ratio), false);
}

TEST(ParserTest, ParseCycle) {
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
    EXPECT_EQ(static_cast<bool>(ast.cycle.has_value()), true);
    EXPECT_EQ(ast.cycle->lines.size(), 2u);

    const auto& line1 = ast.cycle->lines[0];
    EXPECT_EQ(line1.source, "a");
    EXPECT_EQ(line1.arrows.size(), 1u);
    EXPECT_EQ(line1.arrows[0].backward, false);
    EXPECT_EQ(line1.arrows[0].target, "l");

    const auto& line2 = ast.cycle->lines[1];
    EXPECT_EQ(line2.source, "l");
    EXPECT_EQ(line2.arrows.size(), 1u);
    EXPECT_EQ(line2.arrows[0].backward, true);
    EXPECT_EQ(static_cast<bool>(line2.arrows[0].func.has_value()), true);
    EXPECT_EQ(*line2.arrows[0].func, "loss");
    EXPECT_EQ(line2.arrows[0].target, "b");
}

TEST(ParserTest, ParseBranchingCycle) {
    const char* src = R"(
producer a;
layer b;
layer c;
cycle {
  a -> b;
    -> c;
}
)";
    Parser p{src};
    auto ast = p.parse_declarations();
    EXPECT_EQ(static_cast<bool>(ast.cycle.has_value()), true);
    EXPECT_EQ(ast.cycle->lines.size(), 2u);
    EXPECT_EQ(ast.cycle->lines[0].source, "a");
    EXPECT_EQ(ast.cycle->lines[1].source, "a");
    EXPECT_EQ(ast.cycle->lines[0].arrows[0].target, "b");
    EXPECT_EQ(ast.cycle->lines[1].arrows[0].target, "c");
}

TEST(ParserTest, ParseParallelArrows) {
    const char* src = R"(
producer a;
layer b;
layer c;
cycle {
  a -> b | -> c;
}
)";
    Parser p{src};
    auto ast = p.parse_declarations();
    EXPECT_EQ(ast.cycle.has_value(), true);
    EXPECT_EQ(ast.cycle->lines.size(), 1u);
    const auto& line = ast.cycle->lines[0];
    EXPECT_EQ(line.arrows.size(), 2u);
    EXPECT_EQ(line.arrows[0].target, "b");
    EXPECT_EQ(line.arrows[1].target, "c");
}

TEST(ParserTest, ParseConditionalBlock) {
    const char* src = R"(
producer a;
layer b;
layer c;
cycle {
  if cond {
    a -> b;
  } else {
    a -> c;
  }
}
)";
    Parser p{src};
    auto ast = p.parse_declarations();
    EXPECT_EQ(ast.cycle.has_value(), true);
    EXPECT_EQ(ast.cycle->conditionals.size(), 1u);
    const auto& cond = ast.cycle->conditionals[0];
    EXPECT_EQ(cond.condition, "cond");
    EXPECT_EQ(cond.if_branch != nullptr, true);
    EXPECT_EQ(cond.if_branch->lines.size(), 1u);
    EXPECT_EQ(cond.if_branch->lines[0].arrows[0].target, "b");
    EXPECT_EQ(cond.else_branch != nullptr, true);
    EXPECT_EQ(cond.else_branch->lines[0].arrows[0].target, "c");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
