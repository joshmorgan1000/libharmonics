#if 0 // Flight tests disabled
#include <gtest/gtest.h>
#include <thread>

#include <harmonics/flight_io.hpp>

using namespace harmonics;

static HTensor make_tensor_hp(float v) {
    HTensor t{HTensor::DType::Float32, {1}};
    t.data().resize(sizeof(float));
    std::memcpy(t.data().data(), &v, sizeof(float));
    return t;
}

TEST(HighPerfFlightServerTest, InMemoryRoundTrip) {
    auto server = std::make_shared<TensorFlightServer>();
    FlightProducer prod(server);
    FlightConsumer cons(server);

    auto t = make_tensor_hp(3.f);
    std::thread th([&] { cons.push(t); });
    auto out = prod.next();
    th.join();
    EXPECT_EQ(*reinterpret_cast<float*>(out.data().data()), 3.f);
}

#ifdef __unix__
TEST(HighPerfFlightServerTest, NetworkRoundTrip) {
    auto server = std::make_shared<TensorFlightServer>(0, 0);
    FlightConsumer client_in("127.0.0.1", server->in_port());
    FlightProducer client_out("127.0.0.1", server->out_port());

    auto t = make_tensor_hp(7.f);
    client_in.push(t);
    auto recv = server->GetTensor();
    EXPECT_EQ(*reinterpret_cast<float*>(recv.data().data()), 7.f);

    auto t2 = make_tensor_hp(9.f);
    server->PutTensor(t2);
    auto out = client_out.next();
    EXPECT_EQ(*reinterpret_cast<float*>(out.data().data()), 9.f);
}
#endif

#else
int main() { return 0; }
#endif
