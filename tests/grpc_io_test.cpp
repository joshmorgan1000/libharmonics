#include <cstring>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <harmonics/grpc_io.hpp>
#include <tensor_stream.grpc.pb.h>
#include <thread>

using harmonics::HTensor;

static HTensor make_tensor(float a, float b) {
    HTensor t{HTensor::DType::Float32, {2}};
    t.data().resize(sizeof(float) * 2);
    float vals[2] = {a, b};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

#ifdef __unix__
TEST(GrpcIOTest, ProducerReceivesData) {
    harmonics::GrpcServer server;
    server.push(make_tensor(1.f, 2.f));
    harmonics::GrpcProducer prod("127.0.0.1", server.port());
    auto r = prod.next();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
}

TEST(GrpcIOTest, ConsumerSendsData) {
    harmonics::GrpcServer server;
    harmonics::GrpcConsumer cons("127.0.0.1", server.port());
    auto t = make_tensor(3.f, 4.f);
    cons.push(t);
    auto r = server.pop();
    const float* d = reinterpret_cast<const float*>(r.data().data());
    EXPECT_EQ(d[0], 3.f);
    EXPECT_EQ(d[1], 4.f);
}

TEST(GrpcIOTest, HandshakeFailure) {
    harmonics::GrpcServer server;
    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(server.port()),
                                       grpc::InsecureChannelCredentials());
    auto stub = harmonics::TensorService::NewStub(channel);
    grpc::ClientContext ctx;
    google::protobuf::Empty req;
    auto reader = stub->PopTensor(&ctx, req);
    harmonics::TensorData resp;
    EXPECT_FALSE(reader->Read(&resp));
    auto status = reader->Finish();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::PERMISSION_DENIED);
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
