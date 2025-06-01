#include <gtest/gtest.h>
#include <harmonics/distributed_io.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/partition.hpp>
#include <harmonics/runtime.hpp>
#include <harmonics/secure_io.hpp>
#include <harmonics/stream_io.hpp>
#include <thread>

struct FixedProducer : harmonics::Producer {
    explicit FixedProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override {
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, shape};
    }
    std::size_t size() const override { return 1; }
    harmonics::HTensor::Shape shape{};
};

TEST(SecureModeTest, ProducesDigest) {
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);

    harmonics::DeploymentDescriptor desc;
    desc.secure = true;

    auto state = g.inference(desc);
    (void)state;

    harmonics::CycleRuntime rt{g, harmonics::make_auto_policy(), desc};
    rt.forward();
    EXPECT_EQ(rt.proof().size(), 64u); // 32 bytes hex encoded
}

TEST(SecureModeTest, ChainOfCustodyAcrossCycles) {
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);

    harmonics::DeploymentDescriptor desc;
    desc.secure = true;

    harmonics::CycleRuntime rt{g, harmonics::make_auto_policy(), desc};
    rt.forward();
    std::string first = rt.proof();
    rt.forward();
    std::string second = rt.proof();
    EXPECT_EQ(first == second, false);
    EXPECT_EQ(rt.verify_chain(first), true);
}

TEST(SecureModeTest, ChainOfCustodyAcrossPartitions) {
    const char* src = "producer p {1}; layer l1; layer l2; cycle { p -> l1; l1 -> l2; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);

    auto parts = harmonics::partition_by_layer(g, 1);
    auto& g1 = parts.first;
    auto& g2 = parts.second;

    auto bus = std::make_shared<harmonics::ProofMessageBus>();
    auto prod = std::make_shared<FixedProducer>(1);
    g1.bindProducer("p", prod);
    auto pbprod = std::make_shared<harmonics::ProofBusProducer>(bus);
    g2.bindProducer("boundary0", pbprod);
    auto bcid = g1.find("boundary0");

    harmonics::DeploymentDescriptor desc;
    desc.secure = true;

    harmonics::CycleRuntime rt1{g1, harmonics::make_auto_policy(), desc};
    harmonics::CycleRuntime rt2{g2, harmonics::make_auto_policy(), desc};

    rt1.forward();
    auto tensor = rt1.state().consumer_tensors[bcid.index];
    harmonics::ProofBusConsumer cons(bus);
    cons.push(tensor, rt1.proof());
    pbprod->fetch();
    rt2.set_chain(pbprod->proof());
    rt2.forward();
    EXPECT_EQ(rt2.verify_chain(pbprod->proof()), true);
}

#ifdef __unix__
TEST(SecureModeTest, ChainOfCustodyOverSocket) {
    const char* src = "producer p {1}; layer l1; layer l2; cycle { p -> l1; l1 -> l2; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);

    auto parts = harmonics::partition_by_layer(g, 1);
    auto& g1 = parts.first;
    auto& g2 = parts.second;

    harmonics::SocketServer server;
    std::thread srv([&]() {
        auto prod = server.accept_proof_producer();
        harmonics::DeploymentDescriptor desc;
        desc.secure = true;
        g2.bindProducer("boundary0", prod);
        prod->fetch();
        harmonics::CycleRuntime rt2{g2, harmonics::make_auto_policy(), desc};
        rt2.set_chain(prod->proof());
        rt2.forward();
        EXPECT_EQ(rt2.verify_chain(prod->proof()), true);
    });
    auto prod = std::make_shared<FixedProducer>(1);
    g1.bindProducer("p", prod);
    harmonics::ProofTcpConsumer cons("127.0.0.1", server.port());

    harmonics::DeploymentDescriptor desc;
    desc.secure = true;
    harmonics::CycleRuntime rt1{g1, harmonics::make_auto_policy(), desc};

    rt1.forward();
    auto tensor = rt1.state().consumer_tensors[g1.find("boundary0").index];
    cons.push(tensor, rt1.proof());
    srv.join();
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
