#pragma once

#include <cmath>
#include <memory>
#include <vector>

namespace harmonics {

/**
 * Utility converting an entropy limit to a bit precision.
 */
inline int bits_from_entropy(float limit) {
    if (limit <= 0.0f)
        return 32;
    float b = -std::log2(limit);
    int bits = static_cast<int>(std::ceil(b));
    if (bits < 2)
        bits = 2;
    if (bits > 32)
        bits = 32;
    return bits;
}

/**
 * Return a bit precision based on available hardware features.
 */
inline int bits_from_hardware() {
#if HARMONICS_HAS_VULKAN
    return 16;
#else
    return 32;
#endif
}

/** Interface selecting the bit width used for intermediate tensors. */
/** Interface selecting the bit width used for intermediate tensors. */
class PrecisionPolicy {
  public:
    virtual ~PrecisionPolicy() = default;

    /**
     * Select the bit width for the given layer.
     *
     * @param layer index of the layer within the graph
     * @return chosen bit precision for that layer
     */
    virtual int select_bits(std::size_t layer) const = 0;
};

/** Policy that always selects full 32 bits of precision. */
class AutoPrecisionPolicy : public PrecisionPolicy {
  public:
    int select_bits(std::size_t /*layer*/) const override { return bits_from_hardware(); }
};

/** Policy selecting a user configured bit width. */
class MaxBitsPolicy : public PrecisionPolicy {
  public:
    explicit MaxBitsPolicy(int bits) : bits_{bits} {}
    int select_bits(std::size_t /*layer*/) const override { return bits_; }

  private:
    int bits_{32};
};

/**
 * Policy deriving precision from an entropy limit expressed as a
 * probability bound.
 */
class EntropyLimitPolicy : public PrecisionPolicy {
  public:
    explicit EntropyLimitPolicy(float limit) : limit_{limit} {}
    int select_bits(std::size_t /*layer*/) const override { return bits_from_entropy(limit_); }

  private:
    float limit_{0.0f};
};

/** Policy selecting a bit width guided by the available hardware. */
class HardwareGuidedPolicy : public PrecisionPolicy {
  public:
    int select_bits(std::size_t /*layer*/) const override { return bits_from_hardware(); }
};

/** Policy returning predefined bit widths for each layer. */
class LayerBitsPolicy : public PrecisionPolicy {
  public:
    explicit LayerBitsPolicy(std::vector<int> bits) : bits_{std::move(bits)} {}

    int select_bits(std::size_t layer) const override {
        if (layer < bits_.size())
            return bits_[layer];
        return bits_.empty() ? 32 : bits_.back();
    }

  private:
    std::vector<int> bits_{};
};

inline std::shared_ptr<PrecisionPolicy> make_auto_policy() {
    return std::make_shared<AutoPrecisionPolicy>();
}

inline std::shared_ptr<PrecisionPolicy> make_max_bits_policy(int bits) {
    return std::make_shared<MaxBitsPolicy>(bits);
}

inline std::shared_ptr<PrecisionPolicy> make_entropy_limit_policy(float limit) {
    return std::make_shared<EntropyLimitPolicy>(limit);
}

inline std::shared_ptr<PrecisionPolicy> make_hardware_policy() {
    return std::make_shared<HardwareGuidedPolicy>();
}

inline std::shared_ptr<PrecisionPolicy> make_layer_bits_policy(std::vector<int> bits) {
    return std::make_shared<LayerBitsPolicy>(std::move(bits));
}

} // namespace harmonics
