#include <cstring>
#include <gtest/gtest.h>
#include <thread>

#include <harmonics/distributed_io.hpp>

using harmonics::HTensor;

static HTensor make_tensor(float a, float b) {
    HTensor t{HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

TEST(DistributedIO, StreamingFileRoundtrip) {
    const char* path = "dist_io.bin";
    {
        harmonics::StreamingFileConsumer cons(path);
        auto t = make_tensor(1.f, 2.f);
        cons.push(t);
    }
    harmonics::StreamingFileProducer prod(path);
    auto r = prod.next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    std::remove(path);
}

#ifdef __unix__
TEST(DistributedIO, SocketServerRoundtrip) {
    harmonics::SocketServer server;
    std::thread srv([&]() {
        auto p = server.accept_producer();
        auto r = p->next();
        const float* d = reinterpret_cast<const float*>(r.data().data());
        EXPECT_EQ(d[0], 3.f);
        EXPECT_EQ(d[1], 4.f);
    });
    harmonics::TcpConsumer cons("127.0.0.1", server.port());
    auto t = make_tensor(3.f, 4.f);
    cons.push(t);
    srv.join();
}

TEST(DistributedIO, SocketServerSendsData) {
    harmonics::SocketServer server;
    std::thread srv([&]() {
        auto c = server.accept_consumer();
        auto t = make_tensor(7.f, 8.f);
        c->push(t);
    });
    harmonics::TcpProducer prod("127.0.0.1", server.port());
    auto r = prod.next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 7.f);
    EXPECT_EQ(d[1], 8.f);
    srv.join();
}

TEST(DistributedIO, GrpcServerRoundtrip) {
    harmonics::GrpcServer server;
    std::thread srv([&]() {
        auto r = server.pop();
        const float* d = reinterpret_cast<const float*>(r.data().data());
        EXPECT_EQ(d[0], 5.f);
        EXPECT_EQ(d[1], 6.f);
    });
    harmonics::GrpcConsumer cons("127.0.0.1", server.port());
    auto t = make_tensor(5.f, 6.f);
    cons.push(t);
    srv.join();
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
