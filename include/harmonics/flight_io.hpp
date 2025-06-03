#pragma once

#include "harmonics/distributed_io.hpp"
#include "harmonics/grpc_io.hpp"
#include "harmonics/net_utils.hpp"
#include "harmonics/stream_io.hpp"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>

namespace harmonics {

// ---------------------------------------------------------------------------
// Simplified Flight transport helpers
// ---------------------------------------------------------------------------
// These classes emulate a minimal subset of Apache Arrow Flight using the same
// framing as the gRPC helpers. They are intended for testing and do not
// implement the full Flight protocol.
// ---------------------------------------------------------------------------

class FlightProducer : public Producer {
  public:
    FlightProducer(const std::string& host, unsigned short port, int max_message_size = 0,
                   int timeout_ms = 0)
        : prod_{host, port, max_message_size, timeout_ms} {}
    explicit FlightProducer(const std::shared_ptr<class TensorFlightServer>& srv,
                            int max_message_size = 0, int timeout_ms = 0);
    HTensor next() override { return prod_.next(); }
    std::size_t size() const override { return 0; }

  private:
    GrpcProducer prod_;
};

class FlightConsumer : public Consumer {
  public:
    FlightConsumer(const std::string& host, unsigned short port, int max_message_size = 0,
                   int timeout_ms = 0)
        : cons_{host, port, max_message_size, timeout_ms} {}
    explicit FlightConsumer(const std::shared_ptr<class TensorFlightServer>& srv,
                            int max_message_size = 0, int timeout_ms = 0);
    void push(const HTensor& t) override { cons_.push(t); }

  private:
    GrpcConsumer cons_;
};

/// Simple in-memory server used by unit tests.
class TensorFlightServer : public std::enable_shared_from_this<TensorFlightServer> {
  public:
    TensorFlightServer(int max_message_size = 0);
#ifdef __unix__
    TensorFlightServer(unsigned short in_port, unsigned short out_port, int max_message_size = 0);
#endif
    ~TensorFlightServer();

    void PutTensor(const HTensor& t);
    HTensor GetTensor();

#ifdef __unix__
    unsigned short in_port() const { return in_server_ ? in_server_->port() : 0; }
    unsigned short out_port() const { return out_server_ ? out_server_->port() : 0; }
#endif

  private:
    std::mutex mutex_{};
    std::condition_variable cv_{};
    std::queue<HTensor> queue_{};

#ifdef __unix__
    std::unique_ptr<GrpcServer> in_server_{};
    std::unique_ptr<GrpcServer> out_server_{};
#endif
};

inline FlightProducer::FlightProducer(const std::shared_ptr<TensorFlightServer>& srv,
                                      int max_message_size, int timeout_ms) {
#ifdef __unix__
    if (!srv)
        throw std::runtime_error("null server");
    auto p = srv->out_port();
    prod_ = GrpcProducer("127.0.0.1", p, max_message_size, timeout_ms);
#else
    (void)srv;
#endif
}

inline FlightConsumer::FlightConsumer(const std::shared_ptr<TensorFlightServer>& srv,
                                      int max_message_size, int timeout_ms) {
#ifdef __unix__
    if (!srv)
        throw std::runtime_error("null server");
    auto p = srv->in_port();
    cons_ = GrpcConsumer("127.0.0.1", p, max_message_size, timeout_ms);
#else
    (void)srv;
#endif
}

inline TensorFlightServer::TensorFlightServer(int max_message_size) {
#ifdef __unix__
    (void)max_message_size;
#else
    (void)max_message_size;
#endif
}

#ifdef __unix__
inline TensorFlightServer::TensorFlightServer(unsigned short in_port, unsigned short out_port,
                                              int max_message_size) {
    in_server_ = std::make_unique<GrpcServer>(in_port, max_message_size);
    out_server_ = std::make_unique<GrpcServer>(out_port, max_message_size);
}
#endif

inline TensorFlightServer::~TensorFlightServer() {}

inline void TensorFlightServer::PutTensor(const HTensor& t) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(t);
    cv_.notify_all();
}

inline HTensor TensorFlightServer::GetTensor() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !queue_.empty(); });
    HTensor t = queue_.front();
    queue_.pop();
    return t;
}

} // namespace harmonics
