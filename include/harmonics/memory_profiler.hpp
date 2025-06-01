#pragma once

#include <cstddef>
#include <cstdint>

namespace harmonics {

struct MemoryTransferStats {
    std::size_t bytes_to_device{0};
    std::size_t bytes_to_host{0};
    std::uint64_t ns_to_device{0};
    std::uint64_t ns_to_host{0};
};

inline MemoryTransferStats& memory_transfer_stats() {
    static MemoryTransferStats stats{};
    return stats;
}

inline void record_memcpy_to_device(std::size_t bytes, std::uint64_t ns) {
    auto& s = memory_transfer_stats();
    s.bytes_to_device += bytes;
    s.ns_to_device += ns;
}

inline void record_memcpy_to_host(std::size_t bytes, std::uint64_t ns) {
    auto& s = memory_transfer_stats();
    s.bytes_to_host += bytes;
    s.ns_to_host += ns;
}

inline MemoryTransferStats reset_memory_transfer_stats() {
    MemoryTransferStats old = memory_transfer_stats();
    memory_transfer_stats() = MemoryTransferStats{};
    return old;
}

} // namespace harmonics
