#pragma once

#include "harmonics/cycle.hpp"
#include <cstddef>
#include <vector>

namespace harmonics {

struct RuntimeMemoryStats {
    std::size_t producer_bytes{0};
    std::size_t layer_bytes{0};
    std::size_t consumer_bytes{0};
    std::size_t weight_bytes{0};
    std::size_t precision_bytes{0};
    std::size_t variable_bytes{0};

    std::size_t total() const {
        return producer_bytes + layer_bytes + consumer_bytes + weight_bytes + precision_bytes +
               variable_bytes;
    }
};

inline std::size_t tensor_memory_usage(const HTensor& t) {
    return t.shape().capacity() * sizeof(std::size_t) + t.data().capacity();
}

inline std::size_t vector_memory_usage(const std::vector<HTensor>& v) {
    std::size_t bytes = v.capacity() * sizeof(HTensor);
    for (const auto& t : v)
        bytes += tensor_memory_usage(t);
    return bytes;
}

inline RuntimeMemoryStats profile_runtime_memory(const CycleRuntime& rt) {
    RuntimeMemoryStats stats{};
    const auto& st = rt.state();
    stats.producer_bytes = vector_memory_usage(st.producer_tensors);
    stats.layer_bytes = vector_memory_usage(st.layer_tensors);
    stats.consumer_bytes = vector_memory_usage(st.consumer_tensors);
    stats.weight_bytes = vector_memory_usage(st.weights);
    stats.precision_bytes = st.precision_bits.capacity() * sizeof(int);
    stats.variable_bytes = sizeof(st.variables);
    return stats;
}

} // namespace harmonics
