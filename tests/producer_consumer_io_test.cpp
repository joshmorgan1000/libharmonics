#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#ifdef __unix__
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <harmonics/stream_io.hpp>

using harmonics::HTensor;

static HTensor make_tensor(float a, float b) {
    HTensor t{HTensor::DType::Float32, {2}};
    t.data().resize(2 * sizeof(float));
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

TEST(ProducerConsumerIO, FileRoundtrip) {
    const char* path = "io_test.bin";
    {
        harmonics::FileConsumer cons(path);
        auto t = make_tensor(1.f, 2.f);
        cons.push(t);
        cons.push(t);
    }

    harmonics::FileProducer prod(path);
    EXPECT_EQ(prod.size(), 2u);
    auto a = prod.next();
    auto b = prod.next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    EXPECT_EQ(da[0], 1.f);
    EXPECT_EQ(da[1], 2.f);
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(db[0], 1.f);
    EXPECT_EQ(db[1], 2.f);
    std::remove(path);
}

#ifdef __unix__
TEST(ProducerConsumerIO, SocketRoundtrip) {
    int sv[2];
    EXPECT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    harmonics::SocketProducer prod(sv[0]);
    harmonics::SocketConsumer cons(sv[1]);
    auto t = make_tensor(3.f, 4.f);
    cons.push(t);
    auto r = prod.next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 3.f);
    EXPECT_EQ(d[1], 4.f);
    close(sv[0]);
    close(sv[1]);
}
#endif

TEST(ProducerConsumerIO, BusRoundtrip) {
    auto bus = std::make_shared<harmonics::MessageBus>();
    harmonics::BusProducer prod(bus);
    harmonics::BusConsumer cons(bus);
    auto t = make_tensor(5.f, 6.f);
    std::thread th([&] { cons.push(t); });
    auto r = prod.next();
    th.join();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 5.f);
    EXPECT_EQ(d[1], 6.f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
