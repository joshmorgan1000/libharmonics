#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/rust_ffi.hpp>

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

TEST(RustFFI, DistributedScheduler) {
    const char* src = R"(
producer p {1};
consumer c {1};
layer l1;
layer l2;
cycle { p -(id)-> l1 -(id)-> l2 -> c; }
)";
    HarmonicGraph* g = harmonics_parse_graph(src);
    harmonics_backend_t bk[2] = {HARMONICS_BACKEND_CPU, HARMONICS_BACKEND_CPU};
    HarmonicGraph** parts = harmonics_auto_partition(g, bk, 2);
    DistributedScheduler* sched = harmonics_create_distributed_scheduler(parts, 2, bk, false);
    harmonics_destroy_partitions(parts, 2);

    registerActivation("id", std::make_shared<IdActivation>());
    auto* prod = new FixedProducer(1);
    harmonics_scheduler_bind_producer(sched, 0, "p", prod);

    harmonics_scheduler_fit(sched, 1);

    const auto& state = harmonics_scheduler_runtime(sched, 1)->state();
    EXPECT_EQ(state.consumer_tensors[0].shape().size(), 1u);
    EXPECT_EQ(state.consumer_tensors[0].shape()[0], 1u);

    harmonics_destroy_distributed_scheduler(sched);
    harmonics_destroy_graph(g);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
