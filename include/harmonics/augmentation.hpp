#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "harmonics/dataset.hpp"

namespace harmonics {

/** Flip a tensor horizontally along the last dimension. */
inline HTensor flip_horizontal(const HTensor& t) {
    if (t.shape().empty())
        return t;
    // Determine the size of a single element so we can copy bytes correctly.
    std::size_t elem_size = dtype_size(t.dtype());
    std::size_t width = t.shape().back();
    std::size_t elems = t.data().size() / elem_size;
    std::size_t rows = width == 0 ? 0 : elems / width;
    std::vector<std::byte> data(t.data().size());
    for (std::size_t r = 0; r < rows; ++r) {
        const std::byte* src = t.data().data() + r * width * elem_size;
        std::byte* dst = data.data() + r * width * elem_size;
        for (std::size_t c = 0; c < width; ++c)
            std::memcpy(dst + c * elem_size, src + (width - 1 - c) * elem_size, elem_size);
    }
    // The noise is applied in-place in the copy so the output tensor retains
    // the same shape and dtype as the input.
    return HTensor{t.dtype(), t.shape(), std::move(data)};
}

/** Flip a tensor vertically along the second last dimension. */
inline HTensor flip_vertical(const HTensor& t) {
    if (t.shape().size() < 2)
        return t;
    std::size_t elem_size = dtype_size(t.dtype());
    std::size_t h = t.shape()[t.shape().size() - 2];
    std::size_t w = t.shape()[t.shape().size() - 1];
    std::vector<std::byte> data(t.data().size());
    std::size_t outer = 1;
    for (std::size_t i = 0; i < t.shape().size() - 2; ++i)
        outer *= t.shape()[i];
    for (std::size_t o = 0; o < outer; ++o) {
        const std::byte* src_base = t.data().data() + o * h * w * elem_size;
        std::byte* dst_base = data.data() + o * h * w * elem_size;
        for (std::size_t y = 0; y < h; ++y) {
            const std::byte* src = src_base + (h - 1 - y) * w * elem_size;
            std::byte* dst = dst_base + y * w * elem_size;
            std::memcpy(dst, src, w * elem_size);
        }
    }
    return HTensor{t.dtype(), t.shape(), std::move(data)};
}

/** Rotate a tensor 90 degrees clockwise along the last two dimensions. */
inline HTensor rotate90(const HTensor& t) {
    if (t.shape().size() < 2)
        return t;
    // Height and width of the tensor assuming the last two dimensions
    // represent a 2D plane that should be rotated.
    std::size_t elem_size = dtype_size(t.dtype());
    std::size_t h = t.shape()[t.shape().size() - 2];
    std::size_t w = t.shape()[t.shape().size() - 1];
    std::vector<std::size_t> shape = t.shape();
    shape[shape.size() - 2] = w;
    shape[shape.size() - 1] = h;
    std::vector<std::byte> data(t.data().size());
    for (std::size_t i = 0; i < h; ++i)
        for (std::size_t j = 0; j < w; ++j) {
            const std::byte* src = t.data().data() + (i * w + j) * elem_size;
            std::byte* dst = data.data() + (j * h + (h - 1 - i)) * elem_size;
            std::memcpy(dst, src, elem_size);
        }
    // The rotated tensor has its height and width swapped as expected from a
    // 90 degree rotation.
    return HTensor{t.dtype(), std::move(shape), std::move(data)};
}

/** Add Gaussian noise to a float tensor. */
inline HTensor add_noise(const HTensor& t, float stddev) {
    if (t.dtype() != HTensor::DType::Float32 && t.dtype() != HTensor::DType::Float64)
        return t;
    std::vector<std::byte> data(t.data());
    std::mt19937 rng{std::random_device{}()};
    // Choose the correct numeric distribution based on tensor type.
    if (t.dtype() == HTensor::DType::Float32) {
        float* dst = reinterpret_cast<float*>(data.data());
        const float* src = reinterpret_cast<const float*>(t.data().data());
        std::size_t count = t.data().size() / sizeof(float);
        std::normal_distribution<float> dist(0.0f, stddev);
        for (std::size_t i = 0; i < count; ++i)
            dst[i] = src[i] + dist(rng);
    } else {
        double* dst = reinterpret_cast<double*>(data.data());
        const double* src = reinterpret_cast<const double*>(t.data().data());
        std::size_t count = t.data().size() / sizeof(double);
        std::normal_distribution<double> dist(0.0, stddev);
        for (std::size_t i = 0; i < count; ++i)
            dst[i] = src[i] + dist(rng);
    }
    return HTensor{t.dtype(), t.shape(), std::move(data)};
}

/** Randomly crop a tensor along the last two dimensions. */
inline HTensor random_crop(const HTensor& t, std::size_t crop_h, std::size_t crop_w) {
    if (t.shape().size() < 2)
        return t;
    std::size_t h = t.shape()[t.shape().size() - 2];
    std::size_t w = t.shape()[t.shape().size() - 1];
    if (crop_h > h || crop_w > w)
        return t;
    std::vector<std::size_t> shape = t.shape();
    shape[shape.size() - 2] = crop_h;
    shape[shape.size() - 1] = crop_w;

    std::size_t elem_size = dtype_size(t.dtype());
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> dist_h(0, h - crop_h);
    std::uniform_int_distribution<std::size_t> dist_w(0, w - crop_w);
    std::size_t off_h = dist_h(rng);
    std::size_t off_w = dist_w(rng);

    std::size_t outer = 1;
    for (std::size_t i = 0; i < t.shape().size() - 2; ++i)
        outer *= t.shape()[i];
    std::vector<std::byte> data(shape[shape.size() - 2] * shape[shape.size() - 1] * outer *
                                elem_size);
    std::size_t dst_stride = crop_w;
    std::size_t src_stride = w;
    for (std::size_t o = 0; o < outer; ++o) {
        const std::byte* src_base = t.data().data() + o * h * w * elem_size;
        std::byte* dst_base = data.data() + o * crop_h * crop_w * elem_size;
        for (std::size_t y = 0; y < crop_h; ++y) {
            const std::byte* src = src_base + ((off_h + y) * src_stride + off_w) * elem_size;
            std::byte* dst = dst_base + y * dst_stride * elem_size;
            std::memcpy(dst, src, crop_w * elem_size);
        }
    }
    // Return the cropped region as a new tensor.
    return HTensor{t.dtype(), std::move(shape), std::move(data)};
}

/** Apply random brightness and contrast jitter to a tensor. */
inline HTensor colour_jitter(const HTensor& t, float brightness, float contrast) {
    if (t.dtype() != HTensor::DType::Float32 && t.dtype() != HTensor::DType::Float64 &&
        t.dtype() != HTensor::DType::UInt8)
        return t;
    std::vector<std::byte> data(t.data());
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> br(1.0f - brightness, 1.0f + brightness);
    std::uniform_real_distribution<float> co(1.0f - contrast, 1.0f + contrast);
    float b = br(rng);
    float c = co(rng);
    // Helper lambda to apply the jitter in a type generic manner.

    auto apply = [&](auto* dst, const auto* src, std::size_t count) {
        using T = std::remove_reference_t<decltype(*dst)>;
        T mean = 0;
        for (std::size_t i = 0; i < count; ++i)
            mean += src[i];
        mean /= static_cast<T>(count);
        for (std::size_t i = 0; i < count; ++i) {
            T val = static_cast<T>(src[i]) * static_cast<T>(b);
            val = (val - mean) * static_cast<T>(c) + mean;
            if constexpr (std::is_integral_v<T>) {
                val = std::clamp<T>(val, 0, 255);
            }
            dst[i] = val;
        }
    };

    if (t.dtype() == HTensor::DType::Float32) {
        auto* dst = reinterpret_cast<float*>(data.data());
        const float* src = reinterpret_cast<const float*>(t.data().data());
        std::size_t count = t.data().size() / sizeof(float);
        apply(dst, src, count);
    } else if (t.dtype() == HTensor::DType::Float64) {
        auto* dst = reinterpret_cast<double*>(data.data());
        const double* src = reinterpret_cast<const double*>(t.data().data());
        std::size_t count = t.data().size() / sizeof(double);
        apply(dst, src, count);
    } else {
        auto* dst = reinterpret_cast<unsigned char*>(data.data());
        const unsigned char* src = reinterpret_cast<const unsigned char*>(t.data().data());
        std::size_t count = t.data().size();
        apply(dst, src, count);
    }
    return HTensor{t.dtype(), t.shape(), std::move(data)};
}

/** Randomly rotate a tensor by up to +/-max_degrees around its centre. */
inline HTensor random_rotate(const HTensor& t, float max_degrees) {
    if (t.shape().size() < 2)
        return t;
    std::size_t h = t.shape()[t.shape().size() - 2];
    std::size_t w = t.shape()[t.shape().size() - 1];
    std::size_t elem_size = dtype_size(t.dtype());

    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-max_degrees, max_degrees);
    float angle = dist(rng) * static_cast<float>(M_PI) / 180.0f;
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);

    std::vector<std::byte> data(t.data().size());
    std::size_t outer = 1;
    for (std::size_t i = 0; i < t.shape().size() - 2; ++i)
        outer *= t.shape()[i];
    float cx = static_cast<float>(w - 1) / 2.0f;
    float cy = static_cast<float>(h - 1) / 2.0f;
    for (std::size_t o = 0; o < outer; ++o) {
        const std::byte* src_base = t.data().data() + o * h * w * elem_size;
        std::byte* dst_base = data.data() + o * h * w * elem_size;
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                float fx = static_cast<float>(x) - cx;
                float fy = static_cast<float>(y) - cy;
                float sx = cos_a * fx + sin_a * fy + cx;
                float sy = -sin_a * fx + cos_a * fy + cy;
                std::size_t dst_idx = (y * w + x) * elem_size;
                if (sx >= 0 && sx < static_cast<float>(w) && sy >= 0 &&
                    sy < static_cast<float>(h)) {
                    std::size_t isx = static_cast<std::size_t>(std::lround(sx));
                    std::size_t isy = static_cast<std::size_t>(std::lround(sy));
                    const std::byte* src = src_base + (isy * w + isx) * elem_size;
                    std::memcpy(dst_base + dst_idx, src, elem_size);
                } else {
                    std::memset(dst_base + dst_idx, 0, elem_size);
                }
            }
        }
    }
    return HTensor{t.dtype(), t.shape(), std::move(data)};
}

/** Rotate a tensor by the given angle in degrees around its centre. */
inline HTensor rotate(const HTensor& t, float degrees) {
    if (t.shape().size() < 2)
        return t;
    std::size_t h = t.shape()[t.shape().size() - 2];
    std::size_t w = t.shape()[t.shape().size() - 1];
    std::size_t elem_size = dtype_size(t.dtype());

    float angle = degrees * static_cast<float>(M_PI) / 180.0f;
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);

    std::vector<std::byte> data(t.data().size());
    std::size_t outer = 1;
    for (std::size_t i = 0; i < t.shape().size() - 2; ++i)
        outer *= t.shape()[i];
    float cx = static_cast<float>(w - 1) / 2.0f;
    float cy = static_cast<float>(h - 1) / 2.0f;
    for (std::size_t o = 0; o < outer; ++o) {
        const std::byte* src_base = t.data().data() + o * h * w * elem_size;
        std::byte* dst_base = data.data() + o * h * w * elem_size;
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                float fx = static_cast<float>(x) - cx;
                float fy = static_cast<float>(y) - cy;
                float sx = cos_a * fx + sin_a * fy + cx;
                float sy = -sin_a * fx + cos_a * fy + cy;
                std::size_t dst_idx = (y * w + x) * elem_size;
                if (sx >= 0 && sx < static_cast<float>(w) && sy >= 0 &&
                    sy < static_cast<float>(h)) {
                    std::size_t isx = static_cast<std::size_t>(std::lround(sx));
                    std::size_t isy = static_cast<std::size_t>(std::lround(sy));
                    const std::byte* src = src_base + (isy * w + isx) * elem_size;
                    std::memcpy(dst_base + dst_idx, src, elem_size);
                } else {
                    std::memset(dst_base + dst_idx, 0, elem_size);
                }
            }
        }
    }
    return HTensor{t.dtype(), t.shape(), std::move(data)};
}

/** Scale a tensor along the last two dimensions by the given factor. */
inline HTensor scale(const HTensor& t, float factor) {
    if (t.shape().size() < 2 || factor <= 0.0f)
        return t;
    std::size_t h = t.shape()[t.shape().size() - 2];
    std::size_t w = t.shape()[t.shape().size() - 1];
    std::size_t new_h = static_cast<std::size_t>(std::lround(h * factor));
    std::size_t new_w = static_cast<std::size_t>(std::lround(w * factor));
    std::vector<std::size_t> shape = t.shape();
    shape[shape.size() - 2] = new_h;
    shape[shape.size() - 1] = new_w;
    std::size_t elem_size = dtype_size(t.dtype());

    std::vector<std::byte> data(new_h * new_w * elem_size);
    std::size_t outer = 1;
    for (std::size_t i = 0; i < t.shape().size() - 2; ++i)
        outer *= t.shape()[i];
    for (std::size_t o = 0; o < outer; ++o) {
        const std::byte* src_base = t.data().data() + o * h * w * elem_size;
        std::byte* dst_base = data.data() + o * new_h * new_w * elem_size;
        for (std::size_t y = 0; y < new_h; ++y) {
            std::size_t sy = std::min<std::size_t>(static_cast<std::size_t>(y / factor), h - 1);
            for (std::size_t x = 0; x < new_w; ++x) {
                std::size_t sx = std::min<std::size_t>(static_cast<std::size_t>(x / factor), w - 1);
                const std::byte* src = src_base + (sy * w + sx) * elem_size;
                std::byte* dst = dst_base + (y * new_w + x) * elem_size;
                std::memcpy(dst, src, elem_size);
            }
        }
    }
    return HTensor{t.dtype(), std::move(shape), std::move(data)};
}

/** Wrap a producer and flip each sample horizontally. */
inline std::shared_ptr<Producer> make_flip_horizontal(std::shared_ptr<Producer> inner) {
    return std::make_shared<AugmentProducer>(std::move(inner),
                                             [](const HTensor& t) { return flip_horizontal(t); });
}

/** Wrap a producer and flip each sample vertically. */
inline std::shared_ptr<Producer> make_flip_vertical(std::shared_ptr<Producer> inner) {
    return std::make_shared<AugmentProducer>(std::move(inner),
                                             [](const HTensor& t) { return flip_vertical(t); });
}

/** Wrap a producer and rotate each sample by 90 degrees clockwise. */
inline std::shared_ptr<Producer> make_rotate90(std::shared_ptr<Producer> inner) {
    return std::make_shared<AugmentProducer>(std::move(inner),
                                             [](const HTensor& t) { return rotate90(t); });
}

/** Wrap a producer and rotate each sample by the given number of degrees. */
inline std::shared_ptr<Producer> make_rotate(std::shared_ptr<Producer> inner, float degrees) {
    return std::make_shared<AugmentProducer>(
        std::move(inner), [degrees](const HTensor& t) { return rotate(t, degrees); });
}

/** Wrap a producer and add Gaussian noise with the given standard deviation. */
inline std::shared_ptr<Producer> make_add_noise(std::shared_ptr<Producer> inner, float stddev) {
    return std::make_shared<AugmentProducer>(
        std::move(inner), [stddev](const HTensor& t) { return add_noise(t, stddev); });
}

/** Wrap a producer and randomly crop each sample. */
inline std::shared_ptr<Producer> make_random_crop(std::shared_ptr<Producer> inner,
                                                  std::size_t crop_h, std::size_t crop_w) {
    return std::make_shared<AugmentProducer>(std::move(inner), [crop_h, crop_w](const HTensor& t) {
        return random_crop(t, crop_h, crop_w);
    });
}

/** Wrap a producer and randomly rotate each sample. */
inline std::shared_ptr<Producer> make_random_rotation(std::shared_ptr<Producer> inner,
                                                      float max_degrees) {
    return std::make_shared<AugmentProducer>(std::move(inner), [max_degrees](const HTensor& t) {
        return random_rotate(t, max_degrees);
    });
}

/** Wrap a producer and scale each sample. */
inline std::shared_ptr<Producer> make_scale(std::shared_ptr<Producer> inner, float factor) {
    return std::make_shared<AugmentProducer>(
        std::move(inner), [factor](const HTensor& t) { return scale(t, factor); });
}

/** Wrap a producer and apply colour jitter. */
inline std::shared_ptr<Producer> make_colour_jitter(std::shared_ptr<Producer> inner,
                                                    float brightness, float contrast) {
    return std::make_shared<AugmentProducer>(std::move(inner),
                                             [brightness, contrast](const HTensor& t) {
                                                 return colour_jitter(t, brightness, contrast);
                                             });
}

/**
 * @brief Chain multiple augmentation steps together.
 *
 * Each function in @p steps is applied to the incoming sample in order.
 * When @p cache is true the resulting tensors are stored in memory on the
 * first pass so subsequent epochs can reuse the processed values without
 * recomputing the pipeline.
 */
class AugmentationPipeline : public Producer {
  public:
    using Fn = AugmentProducer::Fn;

    AugmentationPipeline(std::shared_ptr<Producer> inner, std::vector<Fn> steps, bool cache = false)
        : inner_{std::move(inner)}, steps_{std::move(steps)}, cache_enabled_{cache} {}

    HTensor next() override {
        if (!cache_enabled_)
            return apply(inner_->next());

        if (cache_filled_) {
            if (cache_index_ >= cache_.size())
                cache_index_ = 0;
            return cache_[cache_index_++];
        }

        if (cache_.size() < inner_->size()) {
            auto t = apply(inner_->next());
            cache_.push_back(t);
            ++cache_index_;
            if (cache_.size() == inner_->size()) {
                cache_filled_ = true;
                cache_index_ = 0;
            }
            return t;
        }

        cache_filled_ = true;
        cache_index_ = 0;
        return cache_.empty() ? HTensor{} : cache_[cache_index_++];
    }

    std::size_t size() const override { return inner_->size(); }

  private:
    HTensor apply(HTensor t) const {
        for (const auto& fn : steps_)
            t = fn(t);
        return t;
    }

    std::shared_ptr<Producer> inner_{};
    std::vector<Fn> steps_{};
    bool cache_enabled_{false};
    std::vector<HTensor> cache_{};
    std::size_t cache_index_{0};
    bool cache_filled_{false};
};

inline std::shared_ptr<Producer>
make_augmentation_pipeline(std::shared_ptr<Producer> inner,
                           std::vector<AugmentationPipeline::Fn> steps, bool cache = false) {
    return std::make_shared<AugmentationPipeline>(std::move(inner), std::move(steps), cache);
}

} // namespace harmonics
