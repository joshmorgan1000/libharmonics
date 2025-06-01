#pragma once

#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "harmonics/dataset.hpp"

#include "harmonics/core.hpp"

namespace harmonics {

/**
 * @brief Wrapper that validates tensor schema when loading data.
 *
 * The constructor checks the first sample of the wrapped producer to
 * verify the expected shape and data type. Optionally every sample can
 * be validated by passing @p validate_all.
 */
class SchemaValidatingProducer : public Producer {
  public:
    SchemaValidatingProducer(std::shared_ptr<Producer> inner, HTensor::Shape expected_shape,
                             HTensor::DType expected_dtype, bool validate_all = false)
        : inner_{std::move(inner)}, shape_{std::move(expected_shape)}, dtype_{expected_dtype},
          validate_all_{validate_all} {
        if (!inner_)
            throw std::runtime_error("null producer");
        auto sample = inner_->next();
        check(sample, 0);
        if (!validate_all_) {
            cached_ = sample;
        } else {
            std::size_t count = inner_->size();
            for (std::size_t i = 1; i < count; ++i)
                check(inner_->next(), i);
        }
    }

    HTensor next() override {
        if (cached_) {
            HTensor t = *cached_;
            cached_.reset();
            return t;
        }
        HTensor t = inner_->next();
        if (validate_all_)
            check(t, index_++);
        return t;
    }

    std::size_t size() const override { return inner_->size(); }

  private:
    void check(const HTensor& t, std::size_t idx) const {
        if (t.dtype() != dtype_ || t.shape() != shape_) {
            std::ostringstream msg;
            msg << "dataset schema mismatch at record " << idx << ": expected "
                << dtype_name(dtype_) << shape_to_string(shape_) << ", got "
                << dtype_name(t.dtype()) << shape_to_string(t.shape());
            throw std::runtime_error(msg.str());
        }
    }

    std::shared_ptr<Producer> inner_{};
    HTensor::Shape shape_{};
    HTensor::DType dtype_{};
    bool validate_all_{};
    std::optional<HTensor> cached_{};
    std::size_t index_{1};
};

} // namespace harmonics
