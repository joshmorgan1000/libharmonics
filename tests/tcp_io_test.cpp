#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#ifdef __unix__
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <harmonics/stream_io.hpp>
#include <harmonics/tcp_io.hpp>

using harmonics::HTensor;

static HTensor make_tensor(float a, float b) {
    HTensor t{HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

#ifdef __unix__
TEST(TcpIOTest, ProducerReceivesData) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_EQ(true, server_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    EXPECT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    EXPECT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    EXPECT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::thread server([&]() {
        int client = ::accept(server_fd, nullptr, nullptr);
        harmonics::SocketConsumer cons(client);
        auto t = make_tensor(1.f, 2.f);
        cons.push(t);
        ::close(client);
        ::close(server_fd);
    });

    harmonics::TcpProducer prod("127.0.0.1", port);
    auto r = prod.next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    server.join();
}

TEST(TcpIOTest, ConsumerSendsData) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_EQ(true, server_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    EXPECT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    EXPECT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    EXPECT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::thread server([&]() {
        int client = ::accept(server_fd, nullptr, nullptr);
        harmonics::SocketProducer prod(client);
        auto r = prod.next();
        const float* d = reinterpret_cast<const float*>(r.data().data());
        EXPECT_EQ(d[0], 3.f);
        EXPECT_EQ(d[1], 4.f);
        ::close(client);
        ::close(server_fd);
    });

    harmonics::TcpConsumer cons("127.0.0.1", port);
    auto t = make_tensor(3.f, 4.f);
    cons.push(t);
    server.join();
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
