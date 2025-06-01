#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace harmonics {

/** Runtime tensor holding arbitrary shaped data. */
class HTensor {
  public:
    /// Supported element types.
    enum class DType { Float32, Float64, Int32, Int64, UInt8 };

    using Shape = std::vector<std::size_t>;

    HTensor() = default;
    HTensor(DType t, Shape s, std::vector<std::byte> d = {})
        : type_{t}, shape_{std::move(s)}, data_{std::move(d)} {}
    HTensor(const HTensor& other) : type_{other.type_}, shape_{other.shape_}, data_{other.data_} {}
    HTensor(HTensor&& other) noexcept = default;
    HTensor& operator=(const HTensor& other) {
        type_ = other.type_;
        shape_ = other.shape_;
        data_ = other.data_;
        return *this;
    }
    HTensor& operator=(HTensor&& other) noexcept = default;

    const Shape& shape() const { return shape_; }
    DType dtype() const { return type_; }

    const std::vector<std::byte>& data() const { return data_; }
    std::vector<std::byte>& data() { return data_; }

  private:
    DType type_{DType::Float32};
    Shape shape_{};
    std::vector<std::byte> data_{};
};

/** Interface for activation functions. */
class ActivationFunction {
  public:
    virtual ~ActivationFunction() = default;
    virtual HTensor operator()(const HTensor& x) const = 0;
};

/** Interface for loss functions. */
class LossFunction {
  public:
    virtual ~LossFunction() = default;
    virtual HTensor operator()(const HTensor& pred, const HTensor& target) const = 0;
};

/** Interface for parameterized layer behaviors. */
class LayerFunction {
  public:
    virtual ~LayerFunction() = default;
    virtual HTensor operator()(const HTensor& x) const = 0;
};

/** Data source interface. */
class Producer {
  public:
    virtual ~Producer() = default;
    virtual HTensor next() = 0;
    virtual std::size_t size() const = 0;
};

/** Data sink interface. */
class Consumer {
  public:
    virtual ~Consumer() = default;
    virtual void push(const HTensor& value) = 0;
};

} // namespace harmonics
