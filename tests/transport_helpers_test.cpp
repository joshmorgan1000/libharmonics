#include <cstring>
#include <gtest/gtest.h>
// #include <harmonics/flight_io.hpp> // Flight disabled
#include <harmonics/serialization.hpp>
#include <harmonics/transport_helpers.hpp>
#include <thread>
#ifdef __unix__
#include <sys/socket.h>
#include <unistd.h>
#endif

using harmonics::HTensor;

static HTensor make_tensor(float a, float b) {
    HTensor t{HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

TEST(TransportHelpers, FileRoundtrip) {
    const char* path = "transport_test.bin";
    {
        auto cons = harmonics::make_consumer(std::string("file:") + path);
        auto t = make_tensor(1.f, 2.f);
        cons->push(t);
    }
    auto prod = harmonics::make_producer(std::string("file:") + path);
    auto r = prod->next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    std::remove(path);
}

#ifdef __unix__
TEST(TransportHelpers, SocketRoundtrip) {
    int sv[2];
    EXPECT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    auto prod = harmonics::make_producer("socket:" + std::to_string(sv[0]));
    auto cons = harmonics::make_consumer("socket:" + std::to_string(sv[1]));
    auto t = make_tensor(3.f, 4.f);
    cons->push(t);
    auto r = prod->next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 3.f);
    EXPECT_EQ(d[1], 4.f);
    close(sv[0]);
    close(sv[1]);
}

TEST(TransportHelpers, ProofSocketRoundtrip) {
    int sv[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    auto prod = harmonics::make_proof_producer("socket:" + std::to_string(sv[0]));
    auto cons = harmonics::make_proof_consumer("socket:" + std::to_string(sv[1]));
    auto pcons = std::dynamic_pointer_cast<harmonics::ProofSocketConsumer>(cons);
    auto pprod = std::dynamic_pointer_cast<harmonics::ProofSocketProducer>(prod);
    auto t = make_tensor(5.f, 6.f);
    std::string proof = "abc";
    pcons->push(t, proof);
    pprod->fetch();
    EXPECT_EQ(pprod->proof(), proof);
    auto r = pprod->next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 5.f);
    EXPECT_EQ(d[1], 6.f);
    close(sv[0]);
    close(sv[1]);
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
