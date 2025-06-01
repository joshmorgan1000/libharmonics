#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>
#include <harmonics/training_visualizer.hpp>
#include <thread>
#ifdef __unix__
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <cstring>

using harmonics::HTensor;

#ifdef __unix__
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

struct IdActivation : harmonics::ActivationFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

struct DummyLoss : harmonics::LossFunction {
    HTensor operator()(const HTensor&, const HTensor&) const override {
        float v = 1.0f;
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &v, sizeof(float));
        return {HTensor::DType::Float32, {1}, std::move(data)};
    }
};

struct DummyProducer : harmonics::Producer {
    explicit DummyProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    HTensor next() override { return {HTensor::DType::Float32, shape}; }
    std::size_t size() const override { return 1; }
    HTensor::Shape shape{};
};

TEST(TrainingVisualizer, SendsProgressFrames) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_EQ(server_fd >= 0, true);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ASSERT_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(server_fd, 1), 0);
    socklen_t len = sizeof(addr);
    ASSERT_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    unsigned short port = ntohs(addr.sin_port);

    std::vector<float> steps;
    std::vector<float> norms;
    std::vector<float> losses;
    std::vector<float> lrs;
    std::thread server([&]() {
        try {
            int client = ::accept(server_fd, nullptr, nullptr);
            char buf[1024];
            ssize_t n;
            std::size_t pos = 0;
            while ((n = ::recv(client, buf + pos, 1, 0)) > 0) {
                pos += static_cast<std::size_t>(n);
                if (pos >= 4 && std::memcmp(buf + pos - 4, "\r\n\r\n", 4) == 0)
                    break;
            }
            const char* resp = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: "
                               "websocket\r\nConnection: Upgrade\r\n\r\n";
            ::send(client, resp, std::strlen(resp), 0);

            for (;;) {
                auto payload = recv_frame(client);
                if (payload.empty())
                    break;
                std::istringstream in(payload);
                auto t = harmonics::read_tensor(in);
                const float* d = reinterpret_cast<const float*>(t.data().data());
                steps.push_back(d[0]);
                norms.push_back(d[1]);
                losses.push_back(d[2]);
                lrs.push_back(d[3]);
            }
            ::close(client);
            ::close(server_fd);
        } catch (...) {
        }
    });

    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; l <-(loss)-> p; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<DummyProducer>(1);
    g.bindProducer("p", prod);
    harmonics::registerActivation("id", std::make_shared<IdActivation>());
    harmonics::registerLoss("loss", std::make_shared<DummyLoss>());

    harmonics::FitOptions opt;
    opt.progress = harmonics::WebSocketTrainingVisualizer("127.0.0.1", port);
    try {
        g.fit(2, harmonics::make_auto_policy(), opt);
    } catch (...) {
    }
    server.join();
    ASSERT_EQ(steps.size() >= 2u, true);
    EXPECT_EQ(static_cast<int>(steps[0]), 1);
    EXPECT_EQ(static_cast<int>(steps[1]), 2);
    ASSERT_EQ(norms.size(), steps.size());
    ASSERT_EQ(losses.size(), steps.size());
    ASSERT_EQ(lrs.size(), steps.size());
    EXPECT_FLOAT_EQ(norms[0], 1.0f);
    EXPECT_FLOAT_EQ(losses[0], 1.0f);
    EXPECT_FLOAT_EQ(lrs[0], 0.01f);
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
