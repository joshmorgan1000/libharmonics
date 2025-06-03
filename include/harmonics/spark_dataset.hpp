#pragma once

#include "harmonics/net_utils.hpp"
#include "harmonics/serialization.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace harmonics {

class SparkProducer : public Producer {
  public:
    explicit SparkProducer(int fd) : fd_{fd} {}
    SparkProducer(const std::string& host, unsigned short port) {
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
        send_handshake();
        if (!recv_handshake()) {
            net_close(fd_);
            throw std::runtime_error("handshake failed");
        }
    }
    ~SparkProducer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    HTensor next() override {
        std::uint32_t len_be = 0;
        if (!read_exact(&len_be, 4))
            return {};
        std::uint32_t len = ntohl(len_be);
        std::string buf(len, '\0');
        if (!read_exact(buf.data(), len))
            return {};
        std::istringstream in(buf);
        return read_tensor(in);
    }

    std::size_t size() const override { return 0; }

  private:
    socket_t fd_{invalid_socket};
    static constexpr char magic_[5] = "HSPK";
    void send_handshake() { net_write(fd_, magic_, 4); }
    bool recv_handshake() {
        char buf[4];
        if (read_exact(buf, 4))
            return std::memcmp(buf, magic_, 4) == 0;
        return false;
    }
    bool read_exact(void* dst, std::size_t bytes) {
        std::size_t off = 0;
        while (off < bytes) {
            auto n =
                net_read(fd_, reinterpret_cast<char*>(dst) + off, static_cast<int>(bytes - off));
            if (n <= 0)
                return false;
            off += static_cast<std::size_t>(n);
        }
        return true;
    }
};

class SparkConsumer : public Consumer {
  public:
    explicit SparkConsumer(int fd) : fd_{fd} {}
    SparkConsumer(const std::string& host, unsigned short port) {
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
        send_handshake();
        if (!recv_handshake()) {
            net_close(fd_);
            throw std::runtime_error("handshake failed");
        }
    }
    ~SparkConsumer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    void push(const HTensor& t) override {
        std::ostringstream out;
        write_tensor(out, t);
        auto str = out.str();
        std::uint32_t len = htonl(static_cast<std::uint32_t>(str.size()));
        net_write(fd_, &len, 4);
        net_write(fd_, str.data(), static_cast<int>(str.size()));
    }

  private:
    socket_t fd_{invalid_socket};
    static constexpr char magic_[5] = "HSPK";
    void send_handshake() { net_write(fd_, magic_, 4); }
    bool recv_handshake() {
        char buf[4];
        if (read_exact(buf, 4))
            return std::memcmp(buf, magic_, 4) == 0;
        return false;
    }
    bool read_exact(void* dst, std::size_t bytes) {
        std::size_t off = 0;
        while (off < bytes) {
            auto n =
                net_read(fd_, reinterpret_cast<char*>(dst) + off, static_cast<int>(bytes - off));
            if (n <= 0)
                return false;
            off += static_cast<std::size_t>(n);
        }
        return true;
    }
};

} // namespace harmonics
