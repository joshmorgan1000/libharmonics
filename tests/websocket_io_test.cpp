#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#ifdef __unix__
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <harmonics/websocket_io.hpp>

using harmonics::HTensor;

static HTensor make_tensor(float a, float b) {
    HTensor t{HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

#ifdef __unix__
static void send_frame(int fd, const std::string& data) {
    unsigned char hdr[4];
    hdr[0] = 0x82;
    hdr[1] = static_cast<unsigned char>(data.size());
    ::send(fd, hdr, 2, 0);
    ::send(fd, data.data(), data.size(), 0);
}

static std::string recv_frame(int fd) {
    unsigned char hdr[2];
    ssize_t n = ::recv(fd, hdr, 2, MSG_WAITALL);
    if (n != 2)
        return {};
    std::size_t len = hdr[1] & 0x7F;
    std::string out(len, '\0');
    n = ::recv(fd, out.data(), len, MSG_WAITALL);
    if (n != static_cast<ssize_t>(len))
        return {};
    return out;
}

TEST(WebSocketIOTest, ProducerReceivesData) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_EQ(server_fd >= 0, true);
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
        char buf[1024];
        ssize_t n;
        std::size_t pos = 0;
        while ((n = ::recv(client, buf + pos, 1, 0)) > 0) {
            pos += static_cast<std::size_t>(n);
            if (pos >= 4 && std::memcmp(buf + pos - 4, "\r\n\r\n", 4) == 0)
                break;
        }
        const char* resp =
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n";
        ::send(client, resp, std::strlen(resp), 0);
        auto t = make_tensor(1.f, 2.f);
        std::ostringstream out;
        harmonics::write_tensor(out, t);
        send_frame(client, out.str());
        ::close(client);
        ::close(server_fd);
    });

    harmonics::WebSocketProducer prod("127.0.0.1", port);
    auto r = prod.next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    server.join();
}

TEST(WebSocketIOTest, ConsumerSendsData) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_EQ(server_fd >= 0, true);
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
        char buf[1024];
        ssize_t n;
        std::size_t pos = 0;
        while ((n = ::recv(client, buf + pos, 1, 0)) > 0) {
            pos += static_cast<std::size_t>(n);
            if (pos >= 4 && std::memcmp(buf + pos - 4, "\r\n\r\n", 4) == 0)
                break;
        }
        const char* resp =
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n";
        ::send(client, resp, std::strlen(resp), 0);
        auto payload = recv_frame(client);
        std::istringstream in(payload);
        auto r = harmonics::read_tensor(in);
        const float* d = reinterpret_cast<const float*>(r.data().data());
        EXPECT_EQ(d[0], 3.f);
        EXPECT_EQ(d[1], 4.f);
        ::close(client);
        ::close(server_fd);
    });

    harmonics::WebSocketConsumer cons("127.0.0.1", port);
    auto t = make_tensor(3.f, 4.f);
    cons.push(t);
    server.join();
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
