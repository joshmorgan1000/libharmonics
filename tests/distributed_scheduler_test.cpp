#include <gtest/gtest.h>

#include <harmonics/distributed_scheduler.hpp>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/partition.hpp>

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

TEST(DistributedScheduler, ForwardsAcrossPartitions) {
    const char* src = R"(
producer p {1};
consumer c {1};
layer l1;
layer l2;
cycle { p -(id)-> l1 -(id)-> l2 -> c; }
)";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    auto parts = partition_by_layer(g, 1);

    registerActivation("id", std::make_shared<IdActivation>());
    auto prod = std::make_shared<FixedProducer>(1);
    parts.first.bindProducer("p", prod);

    std::vector<HarmonicGraph> graphs{parts.first, parts.second};
    DeploymentDescriptor deploy;
    deploy.partitions = {{Backend::CPU}, {Backend::CPU}};
    DistributedScheduler sched{std::move(graphs), deploy};
    sched.fit(1);

    const auto& state = sched.runtime(1).state();
    EXPECT_EQ(state.consumer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(state.consumer_tensors[0].shape()[0], 1u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
