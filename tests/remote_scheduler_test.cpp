#include <gtest/gtest.h>
#include <harmonics/distributed_io.hpp>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/partition.hpp>
#include <harmonics/remote_scheduler.hpp>
#include <thread>

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

#ifdef __unix__
TEST(RemoteScheduler, ForwardsAcrossTcp) {
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

    SocketServer in_srv;
    SocketServer out_srv;
    std::thread bridge([&]() {
        auto in_prod = in_srv.accept_producer();
        auto out_cons = out_srv.accept_consumer();
        auto t = in_prod->next();
        out_cons->push(t);
    });

    std::vector<RemoteBinding> cons_bindings{
        {"boundary0", "127.0.0.1", in_srv.port(), RemoteTransport::TCP}};
    RemoteScheduler sched1{parts.first, {}, cons_bindings};

    std::vector<RemoteBinding> prod_bindings{
        {"boundary0", "127.0.0.1", out_srv.port(), RemoteTransport::TCP}};
    RemoteScheduler sched2{parts.second, prod_bindings, {}};

    sched1.step();
    sched2.step();
    bridge.join();

    const auto& state = sched2.runtime().state();
    auto idx = parts.second.find("c").index;
    EXPECT_EQ(state.consumer_tensors[idx].shape().size(), 1u);
    EXPECT_EQ(state.consumer_tensors[idx].shape()[0], 1u);
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#ifdef __unix__
TEST(RemoteScheduler, ForwardsAcrossTcpCompressed) {
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

    SocketServer in_srv;
    SocketServer out_srv;
    std::thread bridge([&]() {
        auto in_prod = in_srv.accept_producer();
        auto out_cons = out_srv.accept_consumer();
        auto t = in_prod->next();
        out_cons->push(t);
    });

    std::vector<RemoteBinding> cons_bindings{
        {"boundary0", "127.0.0.1", in_srv.port(), RemoteTransport::TCP, true}};
    RemoteScheduler sched1{parts.first, {}, cons_bindings};

    std::vector<RemoteBinding> prod_bindings{
        {"boundary0", "127.0.0.1", out_srv.port(), RemoteTransport::TCP, true}};
    RemoteScheduler sched2{parts.second, prod_bindings, {}};

    sched1.step();
    sched2.step();
    bridge.join();

    const auto& state = sched2.runtime().state();
    auto idx = parts.second.find("c").index;
    EXPECT_EQ(state.consumer_tensors[idx].shape().size(), 1u);
    EXPECT_EQ(state.consumer_tensors[idx].shape()[0], 1u);
}

TEST(RemoteScheduler, FitAcrossTcp) {
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

    SocketServer in_srv;
    SocketServer out_srv;
    std::thread bridge([&]() {
        auto in_prod = in_srv.accept_producer();
        auto out_cons = out_srv.accept_consumer();
        auto t = in_prod->next();
        out_cons->push(t);
    });

    std::vector<RemoteBinding> cons_bindings{
        {"boundary0", "127.0.0.1", in_srv.port(), RemoteTransport::TCP}};
    RemoteScheduler sched1{parts.first, {}, cons_bindings};

    std::vector<RemoteBinding> prod_bindings{
        {"boundary0", "127.0.0.1", out_srv.port(), RemoteTransport::TCP}};
    RemoteScheduler sched2{parts.second, prod_bindings, {}};

    sched1.fit(1);
    sched2.fit(1);
    bridge.join();

    const auto& state = sched2.runtime().state();
    auto idx = parts.second.find("c").index;
    EXPECT_EQ(state.consumer_tensors[idx].shape().size(), 1u);
    EXPECT_EQ(state.consumer_tensors[idx].shape()[0], 1u);
}
#endif