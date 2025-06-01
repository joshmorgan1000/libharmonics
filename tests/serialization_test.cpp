#include <gtest/gtest.h>
#include <sstream>

#include <harmonics/graph.hpp>
#include <harmonics/model_import.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/serialization.hpp>

using harmonics::build_graph;
using harmonics::load_graph;
using harmonics::load_weights;
using harmonics::Parser;
using harmonics::save_graph;
using harmonics::save_weights;

TEST(SerializationTest, SaveLoadGraph) {
    const char* src = R"(
producer p {4};
consumer c {2};
layer l;
cycle {
  p -> l;
  l -> c;
}
)";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);

    std::stringstream buf;
    save_graph(g, buf);
    auto g2 = load_graph(buf);

    EXPECT_EQ(g2.producers.size(), 1u);
    EXPECT_EQ(g2.consumers.size(), 1u);
    EXPECT_EQ(g2.layers.size(), 1u);
    EXPECT_EQ(g2.cycle.size(), 2u);
    EXPECT_EQ(g2.producers[0].name, "p");
    EXPECT_EQ(g2.consumers[0].name, "c");
    EXPECT_EQ(g2.layers[0].name, "l");
}

TEST(SerializationTest, SaveLoadWeights) {
    harmonics::HTensor t{harmonics::HTensor::DType::Float32, {2}};
    t.data().resize(2 * sizeof(float));
    float* d = reinterpret_cast<float*>(t.data().data());
    d[0] = 1.0f;
    d[1] = 2.0f;

    std::vector<harmonics::HTensor> w{t};
    std::stringstream buf;
    save_weights(w, buf);
    auto w2 = load_weights(buf);

    EXPECT_EQ(w2.size(), 1u);
    EXPECT_EQ(w2[0].dtype(), harmonics::HTensor::DType::Float32);
    EXPECT_EQ(w2[0].shape().size(), 1u);
    EXPECT_EQ(w2[0].shape()[0], 2u);
    const float* d2 = reinterpret_cast<const float*>(w2[0].data().data());
    EXPECT_EQ(d2[0], 1.0f);
    EXPECT_EQ(d2[1], 2.0f);
}

TEST(SerializationTest, SaveLoadNamedWeights) {
    harmonics::HTensor t{harmonics::HTensor::DType::Float32, {1}};
    t.data().resize(sizeof(float));
    reinterpret_cast<float*>(t.data().data())[0] = 3.0f;
    std::vector<harmonics::NamedTensor> nw{{"foo", t}};
    std::stringstream buf;
    harmonics::save_named_weights(nw, buf);
    auto nw2 = harmonics::load_named_weights(buf);
    EXPECT_EQ(nw2.size(), 1u);
    EXPECT_EQ(nw2[0].first, "foo");
    const float* d = reinterpret_cast<const float*>(nw2[0].second.data().data());
    EXPECT_EQ(d[0], 3.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
