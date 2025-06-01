#include <harmonics/dataset_cache.hpp>
#include <harmonics/distributed_io.hpp>
#include <harmonics/tcp_io.hpp>

#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using namespace harmonics;

namespace {

struct CountingProducer : Producer {
    explicit CountingProducer(int n) : limit(n) {}
    HTensor next() override {
        if (index >= limit)
            return {};
        float v = static_cast<float>(index++);
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &v, sizeof(float));
        return HTensor{HTensor::DType::Float32, {1}, std::move(data)};
    }
    std::size_t size() const override { return limit; }
    int limit;
    int index{0};
};

} // namespace

int main() {
#ifdef __unix__
    SocketServer server;
    std::thread node_a([&]() {
        auto prod = server.accept_producer();
        DistributedCachedProducer cache(nullptr, "node_a_cache.h5",
                                        std::shared_ptr<Producer>(std::move(prod)));
        for (std::size_t i = 0; i < cache.size(); ++i) {
            auto t = cache.next();
            const float* d = reinterpret_cast<const float*>(t.data().data());
            std::cout << "Node A tensor " << i << ": " << d[0] << '\n';
        }
    });

    {
        auto remote = std::make_shared<TcpConsumer>("127.0.0.1", server.port());
        DistributedCachedProducer cache(std::make_shared<CountingProducer>(3), "node_b_cache.h5",
                                        nullptr, remote);
        for (std::size_t i = 0; i < cache.size(); ++i)
            cache.next();
    }

    node_a.join();
#else
    std::cout << "Example requires POSIX networking support\n";
#endif
    std::remove("node_a_cache.h5");
    std::remove("node_b_cache.h5");
    return 0;
}
