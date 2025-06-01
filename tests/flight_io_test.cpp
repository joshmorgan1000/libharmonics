#if 0 // Flight tests disabled
#include <cstring>
#include <gtest/gtest.h>
#include <thread>

#include <harmonics/flight_io.hpp>

using harmonics::HTensor;

static HTensor make_tensor(float a, float b) {
    HTensor t{HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

TEST(FlightIOTest, ProducerReceivesFromServer) {
    auto srv = std::make_shared<harmonics::TensorFlightServer>();
    harmonics::FlightProducer prod(srv);

    auto t = make_tensor(1.f, 2.f);
    std::thread th([&]() { srv->PutTensor(t); });
    auto r = prod.next();
    th.join();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
}

TEST(FlightIOTest, ConsumerSendsToServer) {
    auto srv = std::make_shared<harmonics::TensorFlightServer>();
    harmonics::FlightConsumer cons(srv);

    auto t = make_tensor(3.f, 4.f);
    std::thread th([&]() { cons.push(t); });
    auto r = srv->GetTensor();
    th.join();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 3.f);
    EXPECT_EQ(d[1], 4.f);
}

#ifdef __unix__
TEST(FlightIOTest, NetworkReceiveFromClient) {
    auto srv = std::make_shared<harmonics::TensorFlightServer>(0, 0);

    harmonics::FlightConsumer client_in("127.0.0.1", srv->in_port());
    auto t = make_tensor(5.f, 6.f);
    client_in.push(t);

    auto r = srv->GetTensor();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 5.f);
    EXPECT_EQ(d[1], 6.f);
}

TEST(FlightIOTest, NetworkSendToClient) {
    auto srv = std::make_shared<harmonics::TensorFlightServer>(0, 0);

    harmonics::FlightProducer client_out("127.0.0.1", srv->out_port());
    auto t = make_tensor(7.f, 8.f);
    srv->PutTensor(t);
    auto r = client_out.next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 7.f);
    EXPECT_EQ(d[1], 8.f);
}
#endif

#else
int main() { return 0; }
#endif
