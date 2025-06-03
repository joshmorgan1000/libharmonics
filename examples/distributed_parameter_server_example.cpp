#include <cstring>
#include <harmonics/distributed_io.hpp>
#include <iostream>
#include <memory>
#include <thread>

using namespace harmonics;

namespace {

struct ParameterServer {
    explicit ParameterServer(float lr) : lr(lr) {}
    void apply(float grad) { param -= lr * grad; }
    float param{1.f};
    float lr{0.1f};
};

static HTensor make_tensor(float v) {
    HTensor t{HTensor::DType::Float32, {1}};
    t.data().resize(sizeof(float));
    std::memcpy(t.data().data(), &v, sizeof(float));
    return t;
}

} // namespace

int main() {
#ifdef __unix__
    ParameterServer ps{0.1f};
    GrpcServer param_srv;
    GrpcServer grad_srv;

    std::thread worker([&]() {
        GrpcProducer p1("127.0.0.1", param_srv.port());
        auto t = p1.next();
        const float* d = reinterpret_cast<const float*>(t.data().data());
        std::cout << "Initial parameter: " << d[0] << '\n';

        float grad_val = 0.5f;
        GrpcConsumer grad_cons("127.0.0.1", grad_srv.port());
        grad_cons.push(make_tensor(grad_val));

        GrpcProducer p2("127.0.0.1", param_srv.port());
        auto t2 = p2.next();
        const float* d2 = reinterpret_cast<const float*>(t2.data().data());
        std::cout << "Updated parameter: " << d2[0] << '\n';
    });

    param_srv.push(make_tensor(ps.param));

    auto g = grad_srv.pop();
    float grad = *reinterpret_cast<const float*>(g.data().data());
    ps.apply(grad);

    param_srv.push(make_tensor(ps.param));

    worker.join();
#else
    std::cout << "Example requires POSIX networking support\n";
#endif
    return 0;
}
