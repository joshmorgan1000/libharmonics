#pragma once

#include "harmonics/stream_io.hpp"
#include "tensor_stream.grpc.pb.h"

#include <chrono>
#include <condition_variable>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace harmonics {

constexpr char kGrpcMagicKey[] = "magic";
constexpr char kGrpcMagicValue[] = "HGRP";

class GrpcProducer : public Producer {
  public:
    GrpcProducer() = default;
    GrpcProducer(const std::string& host, unsigned short port, int max_message_size = 0,
                 int timeout_ms = 0)
        : addr_{host + ":" + std::to_string(port)}, max_message_size_{max_message_size},
          timeout_ms_{timeout_ms} {
        connect();
    }

    HTensor next() override {
        ensure_reader();
        TensorData resp;
        if (!reader_->Read(&resp)) {
            reconnect();
            if (!reader_ || !reader_->Read(&resp))
                throw std::runtime_error("gRPC PopTensor failed");
        }
        std::istringstream in(resp.serialized());
        return read_tensor(in);
    }

    std::size_t size() const override { return 0; }

  private:
    void connect() {
        grpc::ChannelArguments args;
        if (max_message_size_ > 0) {
            args.SetMaxReceiveMessageSize(max_message_size_);
            args.SetMaxSendMessageSize(max_message_size_);
        }
        channel_ = grpc::CreateCustomChannel(addr_, grpc::InsecureChannelCredentials(), args);
        stub_ = TensorService::NewStub(channel_);
    }

    void start_stream() {
        ctx_ = std::make_unique<grpc::ClientContext>();
        ctx_->AddMetadata(kGrpcMagicKey, kGrpcMagicValue);
        if (timeout_ms_ > 0)
            ctx_->set_deadline(std::chrono::system_clock::now() +
                               std::chrono::milliseconds(timeout_ms_));
        google::protobuf::Empty req;
        reader_ = stub_->PopTensor(ctx_.get(), req);
    }

    void ensure_reader() {
        if (!reader_)
            start_stream();
    }

    void reconnect() {
        reader_.reset();
        ctx_.reset();
        connect();
        start_stream();
    }

    std::string addr_{};
    int max_message_size_{0};
    int timeout_ms_{0};
    std::shared_ptr<grpc::Channel> channel_{};
    std::unique_ptr<TensorService::Stub> stub_{};
    std::unique_ptr<grpc::ClientContext> ctx_{};
    std::unique_ptr<grpc::ClientReader<TensorData>> reader_{};
};

class GrpcConsumer : public Consumer {
  public:
    GrpcConsumer() = default;
    GrpcConsumer(const std::string& host, unsigned short port, int max_message_size = 0,
                 int timeout_ms = 0)
        : addr_{host + ":" + std::to_string(port)}, max_message_size_{max_message_size},
          timeout_ms_{timeout_ms} {
        connect();
        start_stream();
    }

    void push(const HTensor& t) override {
        ensure_writer();
        TensorData data;
        std::ostringstream out;
        write_tensor(out, t);
        data.set_serialized(out.str());
        if (!writer_->Write(data)) {
            reconnect();
            if (!writer_->Write(data))
                throw std::runtime_error("gRPC PushTensor failed");
        }
    }

  private:
    void connect() {
        grpc::ChannelArguments args;
        if (max_message_size_ > 0) {
            args.SetMaxReceiveMessageSize(max_message_size_);
            args.SetMaxSendMessageSize(max_message_size_);
        }
        channel_ = grpc::CreateCustomChannel(addr_, grpc::InsecureChannelCredentials(), args);
        stub_ = TensorService::NewStub(channel_);
    }

    void start_stream() {
        ctx_ = std::make_unique<grpc::ClientContext>();
        ctx_->AddMetadata(kGrpcMagicKey, kGrpcMagicValue);
        if (timeout_ms_ > 0)
            ctx_->set_deadline(std::chrono::system_clock::now() +
                               std::chrono::milliseconds(timeout_ms_));
        writer_response_ = std::make_unique<google::protobuf::Empty>();
        writer_ = stub_->PushTensor(ctx_.get(), writer_response_.get());
    }

    void ensure_writer() {
        if (!writer_)
            start_stream();
    }

    void reconnect() {
        writer_.reset();
        ctx_.reset();
        connect();
        start_stream();
    }

    std::string addr_{};
    int max_message_size_{0};
    int timeout_ms_{0};
    std::shared_ptr<grpc::Channel> channel_{};
    std::unique_ptr<TensorService::Stub> stub_{};
    std::unique_ptr<grpc::ClientContext> ctx_{};
    std::unique_ptr<google::protobuf::Empty> writer_response_{};
    std::unique_ptr<grpc::ClientWriter<TensorData>> writer_{};
};

class GrpcServer {
  public:
    explicit GrpcServer(unsigned short port = 0, int max_message_size = 0) : service_{*this} {
        std::string addr = "0.0.0.0:" + std::to_string(port);
        grpc::ServerBuilder builder;
        builder.AddListeningPort(addr, grpc::InsecureServerCredentials(), &port_);
        if (max_message_size > 0) {
            builder.SetMaxReceiveMessageSize(max_message_size);
            builder.SetMaxSendMessageSize(max_message_size);
        }
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
    }

    ~GrpcServer() {
        if (server_)
            server_->Shutdown();
    }

    unsigned short port() const { return static_cast<unsigned short>(port_); }

    void push(const HTensor& t) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(t);
        cv_.notify_all();
    }

    HTensor pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty(); });
        HTensor t = queue_.front();
        queue_.pop();
        return t;
    }

  private:
    class ServiceImpl : public TensorService::Service {
      public:
        explicit ServiceImpl(GrpcServer& parent) : parent_(parent) {}

        bool valid_magic(grpc::ServerContext* ctx) {
            auto it = ctx->client_metadata().find(kGrpcMagicKey);
            return it != ctx->client_metadata().end() &&
                   std::string(it->second.data(), it->second.size()) == kGrpcMagicValue;
        }

        grpc::Status PushTensor(grpc::ServerContext* ctx, grpc::ServerReader<TensorData>* reader,
                                google::protobuf::Empty*) override {
            if (!valid_magic(ctx))
                return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "bad handshake");
            TensorData req;
            while (reader->Read(&req)) {
                std::istringstream in(req.serialized());
                auto t = read_tensor(in);
                {
                    std::lock_guard<std::mutex> lock(parent_.mutex_);
                    parent_.queue_.push(std::move(t));
                }
                parent_.cv_.notify_all();
            }
            return grpc::Status::OK;
        }

        grpc::Status PopTensor(grpc::ServerContext* ctx, const google::protobuf::Empty*,
                               grpc::ServerWriter<TensorData>* writer) override {
            if (!valid_magic(ctx))
                return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "bad handshake");
            while (!ctx->IsCancelled()) {
                std::unique_lock<std::mutex> lock(parent_.mutex_);
                parent_.cv_.wait(lock,
                                 [&] { return !parent_.queue_.empty() || ctx->IsCancelled(); });
                if (ctx->IsCancelled())
                    break;
                HTensor t = parent_.queue_.front();
                parent_.queue_.pop();
                lock.unlock();
                TensorData resp;
                std::ostringstream out;
                write_tensor(out, t);
                resp.set_serialized(out.str());
                if (!writer->Write(resp))
                    break;
            }
            return grpc::Status::OK;
        }

      private:
        GrpcServer& parent_;
    };

    std::unique_ptr<grpc::Server> server_{};
    int port_{0};
    ServiceImpl service_;
    std::mutex mutex_{};
    std::condition_variable cv_{};
    std::queue<HTensor> queue_{};
};

} // namespace harmonics
