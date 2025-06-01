#pragma once

#include <cstddef>
#include <vector>

namespace harmonics {

/**
 * @brief Simple ring buffer wrapper for device buffers.
 */
template <typename Buffer>
class Wrapper {
  public:
    explicit Wrapper(std::size_t ring_size = 3)
        : buffers_(ring_size), index_(0) {}

    Buffer& acquire() {
        index_ = (index_ + 1) % buffers_.size();
        return buffers_[index_];
    }

    const Buffer& current() const { return buffers_[index_]; }

    std::size_t size() const { return buffers_.size(); }

  private:
    std::vector<Buffer> buffers_;
    std::size_t index_;
};

} // namespace harmonics
