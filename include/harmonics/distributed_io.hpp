
#pragma once

/// \file distributed_io.hpp
/// \brief Helper classes for streaming tensors over files and sockets.
///
/// This header groups together tiny wrappers that adapt the low level
/// transports in the library into simple producers and consumers. They
/// are intentionally minimal so that examples can perform basic network
/// communication without pulling in a heavyweight dependency.

#include "harmonics/grpc_io.hpp"
#include "harmonics/net_utils.hpp"
#include "harmonics/secure_io.hpp"
#include "harmonics/stream_io.hpp"
#include "harmonics/tcp_io.hpp"

#include <fstream>
#include <memory>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace harmonics {

/// \brief Producer that streams tensors from a binary file.
///
/// Each call to \ref next deserializes the next tensor from the
/// file until EOF is reached. The helper performs no buffering so it
/// can be used to stream large files incrementally.
class StreamingFileProducer : public Producer {
  public:
    /// Construct the producer reading from \p path.
    explicit StreamingFileProducer(const std::string& path) : in_(path, std::ios::binary) {
        if (!in_)
            throw std::runtime_error("failed to open file");
    }

    /// Deserialize the next tensor or return an empty one on error/EOF.
    HTensor next() override {
        if (!in_ || in_.peek() == EOF)
            return {};
        try {
            return read_tensor(in_);
        } catch (...) {
            return {};
        }
    }
    /// Streaming producers do not know their size in advance.
    std::size_t size() const override { return 0; }

  private:
    std::ifstream in_{};
};

/// \brief Consumer that appends tensors to a binary file.
class StreamingFileConsumer : public Consumer {
  public:
    /// Open \p path and append each pushed tensor to it.
    explicit StreamingFileConsumer(const std::string& path)
        : out_(path, std::ios::binary | std::ios::app) {
        if (!out_)
            throw std::runtime_error("failed to open file");
    }

    /// Serialize \p t and flush it to disk immediately.
    void push(const HTensor& t) override {
        write_tensor(out_, t);
        out_.flush();
    }

  private:
    std::ofstream out_{};
};

/// \brief Minimal TCP server used by tests to stream tensors.
class SocketServer {
  public:
    /// Create a server listening on an ephemeral or specified port.
    explicit SocketServer(unsigned short port = 0) {
        net_init();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ == invalid_socket)
            throw std::runtime_error("failed to create socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            net_close(fd_);
            throw std::runtime_error("failed to bind");
        }
        if (::listen(fd_, 1) < 0) {
            net_close(fd_);
            throw std::runtime_error("failed to listen");
        }
        socklen_t len = sizeof(addr);
        ::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
        port_ = ntohs(addr.sin_port);
    }
    ~SocketServer() {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }
    unsigned short port() const { return port_; }
    /// Accept the next client connection and return the raw socket.
    socket_t accept_fd() { return ::accept(fd_, nullptr, nullptr); }
    /// Accept a connection and wrap it in a SocketProducer.
    std::unique_ptr<SocketProducer> accept_producer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_unique<SocketProducer>(c);
    }
    /// Accept a connection and wrap it in a SocketConsumer.
    std::unique_ptr<SocketConsumer> accept_consumer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_unique<SocketConsumer>(c);
    }
    /// Accept a connection that will produce proof-enabled tensors.
    std::shared_ptr<ProofSocketProducer> accept_proof_producer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_shared<ProofSocketProducer>(c);
    }
    /// Accept a connection that will consume proof-enabled tensors.
    std::shared_ptr<ProofSocketConsumer> accept_proof_consumer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_shared<ProofSocketConsumer>(c);
    }

  private:
    socket_t fd_{invalid_socket};
    unsigned short port_{0};
};

} // namespace harmonics
