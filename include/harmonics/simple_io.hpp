#pragma once

#include "harmonics/core.hpp"
#include <cstddef>
#include <cstring>
#include <vector>

namespace harmonics {

/**
 * @brief Producer that emits the same tensor every time.
 */
class ConstantProducer : public Producer {
  public:
    explicit ConstantProducer(HTensor t) : tensor_{std::move(t)} {}

    ConstantProducer(float value, std::size_t width) {
        std::vector<std::byte> data(width * sizeof(float));
        for (std::size_t i = 0; i < width; ++i)
            std::memcpy(data.data() + i * sizeof(float), &value, sizeof(float));
        tensor_ = HTensor{HTensor::DType::Float32, {width}, std::move(data)};
    }

    HTensor next() override { return tensor_; }
    std::size_t size() const override { return 1; }

  private:
    HTensor tensor_{};
};

/**
 * @brief Producer that emits a zero-filled tensor of the given width.
 */
inline std::shared_ptr<Producer> make_zero_producer(std::size_t width) {
    return std::make_shared<ConstantProducer>(0.0f, width);
}

/**
 * @brief Consumer that discards all incoming tensors.
 */
class DiscardConsumer : public Consumer {
  public:
    void push(const HTensor&) override {}
};

} // namespace harmonics
