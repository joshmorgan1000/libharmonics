#pragma once

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

class StreamingFileProducer : public Producer {
  public:
    explicit StreamingFileProducer(const std::string& path) : in_(path, std::ios::binary) {
        if (!in_)
            throw std::runtime_error("failed to open file");
    }
    HTensor next() override {
        if (!in_ || in_.peek() == EOF)
            return {};
        try {
            return read_tensor(in_);
        } catch (...) {
            return {};
        }
    }
    std::size_t size() const override { return 0; }

  private:
    std::ifstream in_{};
};

class StreamingFileConsumer : public Consumer {
  public:
    explicit StreamingFileConsumer(const std::string& path)
        : out_(path, std::ios::binary | std::ios::app) {
        if (!out_)
            throw std::runtime_error("failed to open file");
    }
    void push(const HTensor& t) override {
        write_tensor(out_, t);
        out_.flush();
    }

  private:
    std::ofstream out_{};
};

class SocketServer {
  public:
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
    socket_t accept_fd() { return ::accept(fd_, nullptr, nullptr); }
    std::unique_ptr<SocketProducer> accept_producer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_unique<SocketProducer>(c);
    }
    std::unique_ptr<SocketConsumer> accept_consumer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_unique<SocketConsumer>(c);
    }
    std::shared_ptr<ProofSocketProducer> accept_proof_producer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_shared<ProofSocketProducer>(c);
    }
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

class GrpcServer {
  public:
    explicit GrpcServer(unsigned short port = 0) {
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
    ~GrpcServer() {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }
    unsigned short port() const { return port_; }
    socket_t accept_fd() { return ::accept(fd_, nullptr, nullptr); }
    std::unique_ptr<GrpcProducer> accept_producer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_unique<GrpcProducer>(c);
    }
    std::unique_ptr<GrpcConsumer> accept_consumer() {
        socket_t c = accept_fd();
        if (c == invalid_socket)
            throw std::runtime_error("accept failed");
        return std::make_unique<GrpcConsumer>(c);
    }

  private:
    socket_t fd_{invalid_socket};
    unsigned short port_{0};
};

} // namespace harmonics
