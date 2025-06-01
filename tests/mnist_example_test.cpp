#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

TEST(MnistExampleTest, ParseAndBuild) {
    const char* src = R"(
harmonic mnist_train_cycle {

  producer mnist_training_data;
  producer mnist_label_data 1/1 mnist_training_data;

  layer input;
  layer hidden 1/2 input;
  layer output 1/2 hidden;

  cycle {
    mnist_training_data -(relu)-> input
                       -(relu)-> hidden
                       -(sigmoid)-> output;
    output <-(cross_entropy)- mnist_label_data;
  }
}
)";
    harmonics::Parser p{src};
    auto harm = p.parse_harmonic();
    EXPECT_EQ(harm.name, "mnist_train_cycle");
    auto g = harmonics::build_graph(harm.decls);
    EXPECT_EQ(g.producers.size(), 2u);
    EXPECT_EQ(g.consumers.size(), 0u);
    EXPECT_EQ(g.layers.size(), 3u);
    EXPECT_EQ(g.cycle.size(), 2u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
