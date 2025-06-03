#pragma once

#include "harmonics/net_utils.hpp"
#include "harmonics/secure_io.hpp"
#include "harmonics/stream_io.hpp"

#include <memory>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// TCP based producer and consumer helpers
// ---------------------------------------------------------------------------
// These classes wrap the socket based stream helpers to provide a very thin
// layer of convenience when communicating over the network. They do not
// implement any reconnect logic or sophisticated protocols, their purpose is
// simply to transport serialized tensors during tests.
// ---------------------------------------------------------------------------

namespace harmonics {

/**
 * @brief Producer that communicates with a remote socket via TCP.
 *
 * The connection is established during construction and @ref next reads
 * tensors from the socket using the same serialization format as
 * @ref SocketProducer.
 */
class TcpProducer : public Producer {
  public:
    TcpProducer(const std::string& host, unsigned short port) {
        net_init();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ == invalid_socket)
            throw std::runtime_error("failed to create socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            net_close(fd_);
            throw std::runtime_error("invalid address");
        }
        addr.sin_port = htons(port);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            net_close(fd_);
            throw std::runtime_error("failed to connect");
        }
        prod_ = std::make_unique<SocketProducer>(fd_);
    }

    ~TcpProducer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    HTensor next() override { return prod_->next(); }
    std::size_t size() const override { return 0; }

  private:
    socket_t fd_{(socket_t)(-1)};
    std::unique_ptr<SocketProducer> prod_{};
};

/**
 * @brief Consumer counterpart for @ref TcpProducer.
 *
 * Data written via @ref push is serialized and sent over the same TCP
 * connection using the @ref SocketConsumer helper.
 */
class TcpConsumer : public Consumer {
  public:
    TcpConsumer(const std::string& host, unsigned short port) {
        net_init();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ == invalid_socket)
            throw std::runtime_error("failed to create socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            net_close(fd_);
            throw std::runtime_error("invalid address");
        }
        addr.sin_port = htons(port);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            net_close(fd_);
            throw std::runtime_error("failed to connect");
        }
        cons_ = std::make_unique<SocketConsumer>(fd_);
    }

    ~TcpConsumer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    void push(const HTensor& t) override { cons_->push(t); }

  private:
    socket_t fd_{(socket_t)(-1)};
    std::unique_ptr<SocketConsumer> cons_{};
};

/**
 * @brief Proof-enabled producer that connects over TCP.
 *
 * Internally wraps a @ref ProofSocketProducer to reuse the simple proof
 * transmission protocol while handling the TCP connection setup.
 */
class ProofTcpProducer : public Producer {
  public:
    ProofTcpProducer(const std::string& host, unsigned short port) {
        net_init();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ == invalid_socket)
            throw std::runtime_error("failed to create socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            net_close(fd_);
            throw std::runtime_error("invalid address");
        }
        addr.sin_port = htons(port);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            net_close(fd_);
            throw std::runtime_error("failed to connect");
        }
        prod_ = std::make_shared<ProofSocketProducer>(fd_);
    }

    ~ProofTcpProducer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    void fetch() { prod_->fetch(); }

    HTensor next() override { return prod_->next(); }
    const std::string& proof() const { return prod_->proof(); }
    std::size_t size() const override { return 0; }

  private:
    socket_t fd_{(socket_t)(-1)};
    std::shared_ptr<ProofSocketProducer> prod_{};
};

/**
 * @brief Proof-enabled consumer counterpart for @ref ProofTcpProducer.
 *
 * Wraps a @ref ProofSocketConsumer to send tensors with proof strings
 * over a TCP connection.
 */
class ProofTcpConsumer : public Consumer {
  public:
    ProofTcpConsumer(const std::string& host, unsigned short port) {
        net_init();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ == invalid_socket)
            throw std::runtime_error("failed to create socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            net_close(fd_);
            throw std::runtime_error("invalid address");
        }
        addr.sin_port = htons(port);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            net_close(fd_);
            throw std::runtime_error("failed to connect");
        }
        cons_ = std::make_shared<ProofSocketConsumer>(fd_);
    }

    ~ProofTcpConsumer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    void push(const HTensor& t) override { cons_->push(t); }
    void push(const HTensor& t, const std::string& proof) { cons_->push(t, proof); }

  private:
    socket_t fd_{(socket_t)(-1)};
    std::shared_ptr<ProofSocketConsumer> cons_{};
};
} // namespace harmonics
