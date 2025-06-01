#pragma once

#include "harmonics/net_utils.hpp"
#include "harmonics/stream_io.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace harmonics {

class GrpcProducer : public Producer {
  public:
    explicit GrpcProducer(int fd) : fd_{fd} {}
    GrpcProducer(const std::string& host, unsigned short port) {
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
    }
    ~GrpcProducer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    HTensor next() override {
        std::uint8_t flag = 0;
        if (!read_exact(&flag, 1))
            return {};
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

class GrpcConsumer : public Consumer {
  public:
    explicit GrpcConsumer(int fd) : fd_{fd} {}
    GrpcConsumer(const std::string& host, unsigned short port) {
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
    }
    ~GrpcConsumer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    void push(const HTensor& t) override {
        std::ostringstream out;
        write_tensor(out, t);
        auto str = out.str();
        std::uint8_t flag = 0;
        std::uint32_t len = static_cast<std::uint32_t>(str.size());
        std::uint32_t len_be = htonl(len);
        net_write(fd_, &flag, 1);
        net_write(fd_, &len_be, 4);
        net_write(fd_, str.data(), static_cast<int>(str.size()));
    }

  private:
    socket_t fd_{invalid_socket};
};

} // namespace harmonics
