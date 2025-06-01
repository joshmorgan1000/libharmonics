#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

using namespace harmonics;

struct IdActivation : ActivationFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

struct FixedProducer : Producer {
    explicit FixedProducer(std::size_t s) : shape{s} {}
    HTensor next() override { return HTensor{HTensor::DType::Float32, {shape}}; }
    std::size_t size() const override { return 1; }
    std::size_t shape;
};

TEST(RuntimeCheckpoint, RoundtripSavesState) {
    const char* src = R"(
producer p {1};
consumer c {1};
layer l;
cycle { p -(id)-> l -> c; }
)";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);

    registerActivation("id", std::make_shared<IdActivation>());
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);

    CycleRuntime rt1{g};
    rt1.forward();
    rt1.set_chain("prev");
    std::ostringstream out;
    rt1.save_checkpoint(out);

    CycleRuntime rt2{g};
    std::istringstream in(out.str());
    rt2.load_checkpoint(in);

    EXPECT_EQ(rt2.chain(), rt1.chain());
    EXPECT_EQ(rt2.state().consumer_tensors[0].shape(), rt1.state().consumer_tensors[0].shape());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
