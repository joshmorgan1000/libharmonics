#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#ifdef __unix__
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <harmonics/spark_dataset.hpp>

using harmonics::HTensor;

static HTensor make_tensor(float a, float b) {
    HTensor t{HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

#ifdef __unix__
TEST(SparkDataset, ProducerReceivesData) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_TRUE(server_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ASSERT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    ASSERT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::thread server([&]() {
        int client = ::accept(server_fd, nullptr, nullptr);
        char magic[4];
        EXPECT_EQ(::recv(client, magic, 4, MSG_WAITALL), 4);
        EXPECT_EQ(std::memcmp(magic, "HSPK", 4), 0);
        EXPECT_EQ(::send(client, "HSPK", 4, 0), 4);
        auto t = make_tensor(1.f, 2.f);
        std::ostringstream out;
        harmonics::write_tensor(out, t);
        auto str = out.str();
        std::uint32_t size = htonl(static_cast<std::uint32_t>(str.size()));
        ::send(client, &size, 4, 0);
        ::send(client, str.data(), str.size(), 0);
        ::close(client);
        ::close(server_fd);
    });

    harmonics::SparkProducer prod("127.0.0.1", port);
    auto r = prod.next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    server.join();
}

TEST(SparkDataset, ConsumerSendsData) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_TRUE(server_fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ASSERT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    ASSERT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::thread server([&]() {
        int client = ::accept(server_fd, nullptr, nullptr);
        char magic[4];
        EXPECT_EQ(::recv(client, magic, 4, MSG_WAITALL), 4);
        EXPECT_EQ(std::memcmp(magic, "HSPK", 4), 0);
        EXPECT_EQ(::send(client, "HSPK", 4, 0), 4);
        std::uint32_t size_be;
        EXPECT_EQ(::recv(client, &size_be, 4, MSG_WAITALL), 4);
        std::uint32_t size = ntohl(size_be);
        std::string buf(size, '\0');
        EXPECT_EQ(::recv(client, buf.data(), size, MSG_WAITALL), static_cast<ssize_t>(size));
        std::istringstream in(buf);
        auto r = harmonics::read_tensor(in);
        const float* d = reinterpret_cast<const float*>(r.data().data());
        EXPECT_EQ(d[0], 3.f);
        EXPECT_EQ(d[1], 4.f);
        ::close(client);
        ::close(server_fd);
    });

    harmonics::SparkConsumer cons("127.0.0.1", port);
    auto t = make_tensor(3.f, 4.f);
    cons.push(t);
    server.join();
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
