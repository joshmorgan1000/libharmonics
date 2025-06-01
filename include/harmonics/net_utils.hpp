#pragma once

#ifdef _WIN32
#include <cstdint>
#include <winsock2.h>
#include <ws2tcpip.h>

namespace harmonics {

inline void net_init() {
    static bool initialized = false;
    if (!initialized) {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        initialized = true;
    }
}

using socket_t = SOCKET;
inline void net_close(socket_t s) { closesocket(s); }
inline int net_read(socket_t s, void* buf, int len) {
    return recv(s, static_cast<char*>(buf), len, 0);
}
inline int net_write(socket_t s, const void* buf, int len) {
    return send(s, static_cast<const char*>(buf), len, 0);
}
constexpr socket_t invalid_socket = INVALID_SOCKET;

} // namespace harmonics

#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace harmonics {

inline void net_init() {}

using socket_t = int;
inline void net_close(socket_t s) { ::close(s); }
inline ssize_t net_read(socket_t s, void* buf, size_t len) { return ::read(s, buf, len); }
inline ssize_t net_write(socket_t s, const void* buf, size_t len) { return ::write(s, buf, len); }
constexpr socket_t invalid_socket = -1;

} // namespace harmonics
#endif
