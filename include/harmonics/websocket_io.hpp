#pragma once

#include "harmonics/net_utils.hpp"
#include "harmonics/stream_io.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// WebSocket IO helpers
// ---------------------------------------------------------------------------
// Extremely small implementations of producers and consumers that speak a
// subset of the WebSocket protocol. They are intended solely for the unit
// tests and example programs where using a full WebSocket library would add
// unnecessary complexity. As such only the features required by those tests
// are implemented.
// ---------------------------------------------------------------------------

namespace harmonics {

/**
 * @brief Minimal WebSocket client that produces tensors.
 *
 * The class implements the bare minimum of the WebSocket protocol
 * to stream serialized tensors from a remote endpoint. It avoids
 * external dependencies so the handshake and frame parsing logic
 * is implemented manually using blocking socket operations.
 */
class WebSocketProducer : public Producer {
  public:
    WebSocketProducer(const std::string& host, unsigned short port, const std::string& path = "/") {
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
        send_handshake(host, port, path);
        recv_handshake();
    }

    ~WebSocketProducer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    HTensor next() override {
        std::vector<std::byte> payload;
        if (!read_frame(payload))
            return {};
        // Deserialize the tensor payload using the normal stream based
        // helpers. The frame contents are copied into a temporary string
        // so that an istringstream can be used without worrying about
        // lifetime issues of the underlying buffer.
        std::istringstream in(
            std::string(reinterpret_cast<const char*>(payload.data()), payload.size()));
        return read_tensor(in);
    }
    std::size_t size() const override { return 0; }

  private:
    socket_t fd_{invalid_socket};

    void send_handshake(const std::string& host, unsigned short port, const std::string& path) {
        // Craft a minimal HTTP upgrade request.  No optional headers are
        // included as the server implementation used in the tests only
        // checks for these mandatory fields.
        std::string req = "GET " + path + " HTTP/1.1\r\n";
        req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
        req += "Upgrade: websocket\r\n";
        req += "Connection: Upgrade\r\n";
        req += "Sec-WebSocket-Version: 13\r\n";
        req += "Sec-WebSocket-Key: dummy==\r\n\r\n";
        net_write(fd_, req.c_str(), static_cast<int>(req.size()));
    }

    void recv_handshake() {
        // Read bytes until the end of the HTTP headers. A real parser
        // would handle status codes and additional fields but for our
        // lightweight client it is sufficient to just search for the
        // blank line separating headers from the response body.
        char buf[1024];
        std::size_t pos = 0;
        while (pos < sizeof(buf)) {
            auto n = net_read(fd_, buf + pos, 1);
            if (n <= 0)
                throw std::runtime_error("handshake failed");
            pos += static_cast<std::size_t>(n);
            if (pos >= 4 && std::memcmp(buf + pos - 4, "\r\n\r\n", 4) == 0)
                break;
        }
    }

    bool read_exact(void* dst, std::size_t bytes) {
        // Utility used by frame reading logic to ensure exactly the
        // requested number of bytes are read before returning.
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

    bool read_frame(std::vector<std::byte>& out) {
        // Parse a single WebSocket frame. The implementation only
        // handles the subset required by the tests: binary frames
        // with an optional masking key.
        unsigned char hdr[2];
        if (!read_exact(hdr, 2))
            return false;
        bool mask = (hdr[1] & 0x80) != 0;
        std::uint64_t len = hdr[1] & 0x7F;
        if (len == 126) {
            unsigned char b[2];
            if (!read_exact(b, 2))
                return false;
            len = (static_cast<std::uint16_t>(b[0]) << 8) | b[1];
        } else if (len == 127) {
            unsigned char b[8];
            if (!read_exact(b, 8))
                return false;
            len = 0;
            for (int i = 0; i < 8; ++i)
                len = (len << 8) | b[i];
        }
        unsigned char mask_key[4];
        if (mask) {
            if (!read_exact(mask_key, 4))
                return false;
        }
        std::vector<unsigned char> buf(len);
        if (!read_exact(buf.data(), static_cast<std::size_t>(len)))
            return false;
        if (mask) {
            // Unmask the payload by XOR'ing with the repeating mask key.
            for (std::size_t i = 0; i < len; ++i)
                buf[i] ^= mask_key[i % 4];
        }
        out.resize(len);
        std::memcpy(out.data(), buf.data(), len);
        return true;
    }
};

/**
 * @brief Minimal WebSocket client that consumes tensors.
 *
 * This counterpart to @ref WebSocketProducer sends serialized tensors
 * over a WebSocket connection. The implementation mirrors the producer
 * by hand crafting the handshake and frame encoding so that the
 * example tools remain dependency free.
 */
class WebSocketConsumer : public Consumer {
  public:
    WebSocketConsumer(const std::string& host, unsigned short port, const std::string& path = "/") {
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
        send_handshake(host, port, path);
        recv_handshake();
    }

    ~WebSocketConsumer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    void push(const HTensor& t) override {
        std::ostringstream out;
        write_tensor(out, t);
        auto str = out.str();
        send_frame(reinterpret_cast<const unsigned char*>(str.data()), str.size());
    }

  private:
    socket_t fd_{invalid_socket};

    void send_handshake(const std::string& host, unsigned short port, const std::string& path) {
        std::string req = "GET " + path + " HTTP/1.1\r\n";
        req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
        req += "Upgrade: websocket\r\n";
        req += "Connection: Upgrade\r\n";
        req += "Sec-WebSocket-Version: 13\r\n";
        req += "Sec-WebSocket-Key: dummy==\r\n\r\n";
        net_write(fd_, req.c_str(), static_cast<int>(req.size()));
    }

    void recv_handshake() {
        char buf[1024];
        std::size_t pos = 0;
        while (pos < sizeof(buf)) {
            auto n = net_read(fd_, buf + pos, 1);
            if (n <= 0)
                throw std::runtime_error("handshake failed");
            pos += static_cast<std::size_t>(n);
            if (pos >= 4 && std::memcmp(buf + pos - 4, "\r\n\r\n", 4) == 0)
                break;
        }
    }

    void send_frame(const unsigned char* data, std::size_t len) {
        // Construct a very small WebSocket frame header. Only binary
        // frames are generated and they are always marked as final.
        // The length encoding mirrors the rules from RFC 6455.
        unsigned char hdr[10];
        std::size_t hdr_len = 0;
        hdr[hdr_len++] = 0x82; // FIN + binary
        if (len < 126) {
            hdr[hdr_len++] = static_cast<unsigned char>(len);
        } else if (len < 65536) {
            hdr[hdr_len++] = 126;
            hdr[hdr_len++] = static_cast<unsigned char>((len >> 8) & 0xFF);
            hdr[hdr_len++] = static_cast<unsigned char>(len & 0xFF);
        } else {
            hdr[hdr_len++] = 127;
            for (int i = 7; i >= 0; --i)
                hdr[hdr_len++] = static_cast<unsigned char>((len >> (8 * i)) & 0xFF);
        }
        // First send the header followed by the raw payload bytes.
        net_write(fd_, hdr, static_cast<int>(hdr_len));
        net_write(fd_, data, static_cast<int>(len));
    }
};

} // namespace harmonics
