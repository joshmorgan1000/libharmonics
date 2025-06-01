#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

struct CountingProducer : harmonics::Producer {
    explicit CountingProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override {
        ++calls;
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, shape};
    }
    std::size_t size() const override { return 1; }
    harmonics::HTensor::Shape shape{};
    int calls{0};
};

TEST(ThreadedCycleRuntimeTest, BranchingUsesSingleProducerSample) {
    const char* src = "producer p; layer a; layer b; cycle { p -> a; -> b; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<CountingProducer>(1);
    g.bindProducer("p", prod);

    harmonics::CycleRuntime rt{g};
    rt.enable_multi_threading();
    rt.forward();

    EXPECT_EQ(prod->calls, 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
