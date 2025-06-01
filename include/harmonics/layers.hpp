#pragma once

#include "harmonics/config.hpp"
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>
#if HARMONICS_HAS_WASM_SIMD
#include <wasm_simd128.h>
#endif
#if HARMONICS_HAS_SSE2
#include <immintrin.h>
#endif

#include "harmonics/core.hpp"
#include "harmonics/function_registry.hpp"

namespace harmonics {

/** Configuration controlling built-in layer behaviour. */
struct LayerBehavior {
    std::size_t conv_kernel{3};
    float norm_epsilon{1e-6f};
    float attention_temperature{1.0f};
    std::size_t attention_heads{2};
    std::size_t pool_window{2};
};

inline LayerBehavior& layer_behavior() {
    static LayerBehavior b{};
    return b;
}

inline void set_convolution_kernel(std::size_t k) { layer_behavior().conv_kernel = k; }
inline void set_norm_epsilon(float eps) { layer_behavior().norm_epsilon = eps; }
inline void set_attention_temperature(float t) { layer_behavior().attention_temperature = t; }
inline void set_attention_heads(std::size_t h) { layer_behavior().attention_heads = h; }
inline void set_pool_window(std::size_t w) { layer_behavior().pool_window = w; }

/**
 * Simple convolution layer operating on 1D float tensors.
 * The kernel weights are fixed to 1 and stride is 1.
 */
class ConvolutionLayer : public LayerFunction {
  public:
    explicit ConvolutionLayer(std::size_t kernel = layer_behavior().conv_kernel)
        : kernel_{kernel} {}

    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32 || x.shape().size() != 1)
            return x;
        const auto n = x.shape()[0];
        if (n < kernel_)
            return x;
        std::vector<float> vals(n - kernel_ + 1, 0.0f);
        const float* in = reinterpret_cast<const float*>(x.data().data());
        for (std::size_t i = 0; i + kernel_ <= n; ++i) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < kernel_; ++k)
                sum += in[i + k];
            vals[i] = sum;
        }
        std::vector<std::byte> bytes(vals.size() * sizeof(float));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return {HTensor::DType::Float32, {vals.size()}, std::move(bytes)};
    }

  private:
    std::size_t kernel_{3};
};

/** L2 normalisation layer for 1D float tensors. */
class NormalizationLayer : public LayerFunction {
  public:
    explicit NormalizationLayer(float eps = layer_behavior().norm_epsilon) : eps_{eps} {}

    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32 || x.shape().size() != 1)
            return x;
        const auto n = x.shape()[0];
        std::vector<float> vals(n);
        const float* in = reinterpret_cast<const float*>(x.data().data());
        float norm = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
            norm += in[i] * in[i];
        norm = std::sqrt(norm) + eps_;
        for (std::size_t i = 0; i < n; ++i)
            vals[i] = in[i] / norm;
        std::vector<std::byte> bytes(vals.size() * sizeof(float));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return {HTensor::DType::Float32, {n}, std::move(bytes)};
    }

  private:
    float eps_{1e-6f};
};

/** Simplified self-attention layer for 1D float tensors. */
class AttentionLayer : public LayerFunction {
  public:
    explicit AttentionLayer(float temp = layer_behavior().attention_temperature) : temp_{temp} {}

    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32 || x.shape().size() != 1)
            return x;
        const auto n = x.shape()[0];
        std::vector<float> vals(n);
        const float* in = reinterpret_cast<const float*>(x.data().data());
        std::vector<float> weights(n);
        float maxv = in[0];
        for (std::size_t i = 1; i < n; ++i)
            if (in[i] > maxv)
                maxv = in[i];
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            weights[i] = std::exp((in[i] - maxv) / temp_);
            sum += weights[i];
        }
        float attn = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            weights[i] /= sum;
            attn += weights[i] * in[i];
        }
        for (std::size_t i = 0; i < n; ++i)
            vals[i] = attn;
        std::vector<std::byte> bytes(vals.size() * sizeof(float));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return {HTensor::DType::Float32, {n}, std::move(bytes)};
    }

  private:
    float temp_{1.0f};
};

/** Cross-attention layer operating on 2D float tensors. */
class CrossAttentionLayer : public LayerFunction {
  public:
    explicit CrossAttentionLayer(float temp = layer_behavior().attention_temperature)
        : temp_{temp} {}

    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32 || x.shape().size() != 2)
            return x;
        const auto rows = x.shape()[0];
        const auto cols = x.shape()[1];
        std::vector<float> vals(rows * cols);
        const float* in = reinterpret_cast<const float*>(x.data().data());
        std::vector<float> weights(cols);
        for (std::size_t r = 0; r < rows; ++r) {
            const float* row = in + r * cols;
            float maxv = row[0];
#if HARMONICS_HAS_WASM_SIMD
            {
                std::size_t i = 0;
                v128_t vmax = wasm_v128_load(&row[0]);
                for (; i + 4 <= cols; i += 4) {
                    v128_t v = wasm_v128_load(&row[i]);
                    vmax = wasm_f32x4_max(vmax, v);
                }
                float tmp[4];
                wasm_v128_store(tmp, vmax);
                for (int j = 0; j < 4; ++j)
                    if (tmp[j] > maxv)
                        maxv = tmp[j];
                for (; i < cols; ++i)
                    if (row[i] > maxv)
                        maxv = row[i];
            }
#elif HARMONICS_HAS_SSE2
            {
                std::size_t i = 0;
                __m128 vmax = _mm_loadu_ps(&row[0]);
                for (; i + 4 <= cols; i += 4) {
                    __m128 v = _mm_loadu_ps(&row[i]);
                    vmax = _mm_max_ps(vmax, v);
                }
                alignas(16) float tmp[4];
                _mm_store_ps(tmp, vmax);
                for (int j = 0; j < 4; ++j)
                    if (tmp[j] > maxv)
                        maxv = tmp[j];
                for (; i < cols; ++i)
                    if (row[i] > maxv)
                        maxv = row[i];
            }
#else
            for (std::size_t i = 1; i < cols; ++i)
                if (row[i] > maxv)
                    maxv = row[i];
#endif
            for (std::size_t i = 0; i < cols; ++i)
                weights[i] = std::exp((row[i] - maxv) / temp_);

            float sum = 0.0f;
#if HARMONICS_HAS_WASM_SIMD
            {
                std::size_t i = 0;
                v128_t acc = wasm_f32x4_splat(0.0f);
                for (; i + 4 <= cols; i += 4) {
                    v128_t v = wasm_v128_load(&weights[i]);
                    acc = wasm_f32x4_add(acc, v);
                }
                float tmp[4];
                wasm_v128_store(tmp, acc);
                sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                for (; i < cols; ++i)
                    sum += weights[i];
            }
#elif HARMONICS_HAS_SSE2
            {
                std::size_t i = 0;
                __m128 acc = _mm_setzero_ps();
                for (; i + 4 <= cols; i += 4) {
                    __m128 v = _mm_loadu_ps(&weights[i]);
                    acc = _mm_add_ps(acc, v);
                }
                alignas(16) float tmp[4];
                _mm_store_ps(tmp, acc);
                sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                for (; i < cols; ++i)
                    sum += weights[i];
            }
#else
            for (std::size_t i = 0; i < cols; ++i)
                sum += weights[i];
#endif
            float inv_sum = 1.0f / sum;
            float attn = 0.0f;
#if HARMONICS_HAS_WASM_SIMD
            {
                std::size_t i = 0;
                v128_t vacc = wasm_f32x4_splat(0.0f);
                v128_t vinv = wasm_f32x4_splat(inv_sum);
                for (; i + 4 <= cols; i += 4) {
                    v128_t w = wasm_v128_load(&weights[i]);
                    v128_t xw = wasm_v128_load(&row[i]);
                    w = wasm_f32x4_mul(w, vinv);
                    xw = wasm_f32x4_mul(w, xw);
                    vacc = wasm_f32x4_add(vacc, xw);
                }
                float tmp[4];
                wasm_v128_store(tmp, vacc);
                attn = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                for (; i < cols; ++i)
                    attn += (weights[i] * inv_sum) * row[i];
            }
#elif HARMONICS_HAS_SSE2
            {
                std::size_t i = 0;
                __m128 vacc = _mm_setzero_ps();
                __m128 vinv = _mm_set1_ps(inv_sum);
                for (; i + 4 <= cols; i += 4) {
                    __m128 w = _mm_loadu_ps(&weights[i]);
                    __m128 xw = _mm_loadu_ps(&row[i]);
                    w = _mm_mul_ps(w, vinv);
                    xw = _mm_mul_ps(w, xw);
                    vacc = _mm_add_ps(vacc, xw);
                }
                alignas(16) float tmp[4];
                _mm_store_ps(tmp, vacc);
                attn = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                for (; i < cols; ++i)
                    attn += (weights[i] * inv_sum) * row[i];
            }
#else
            for (std::size_t i = 0; i < cols; ++i)
                attn += (weights[i] * inv_sum) * row[i];
#endif
            for (std::size_t i = 0; i < cols; ++i)
                vals[r * cols + i] = attn;
        }
        std::vector<std::byte> bytes(vals.size() * sizeof(float));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return {HTensor::DType::Float32, {rows, cols}, std::move(bytes)};
    }

  private:
    float temp_{1.0f};
};

/** Multi-head self-attention layer for 1D float tensors. */
class MultiHeadAttentionLayer : public LayerFunction {
  public:
    explicit MultiHeadAttentionLayer(std::size_t heads = layer_behavior().attention_heads,
                                     float temp = layer_behavior().attention_temperature)
        : heads_{heads}, temp_{temp} {}

    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32 || x.shape().size() != 1)
            return x;
        const auto n = x.shape()[0];
        if (heads_ == 0 || n % heads_ != 0)
            return x;
        std::size_t head_size = n / heads_;
        const float* in = reinterpret_cast<const float*>(x.data().data());
        std::vector<float> vals(n);
        for (std::size_t h = 0; h < heads_; ++h) {
            const float* head_in = in + h * head_size;
            float maxv = head_in[0];
            for (std::size_t i = 1; i < head_size; ++i)
                if (head_in[i] > maxv)
                    maxv = head_in[i];
            std::vector<float> weights(head_size);
            float sum = 0.0f;
            for (std::size_t i = 0; i < head_size; ++i) {
                weights[i] = std::exp((head_in[i] - maxv) / temp_);
                sum += weights[i];
            }
            float attn = 0.0f;
            for (std::size_t i = 0; i < head_size; ++i) {
                weights[i] /= sum;
                attn += weights[i] * head_in[i];
            }
            for (std::size_t i = 0; i < head_size; ++i)
                vals[h * head_size + i] = attn;
        }
        std::vector<std::byte> bytes(vals.size() * sizeof(float));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return {HTensor::DType::Float32, {n}, std::move(bytes)};
    }

  private:
    std::size_t heads_{1};
    float temp_{1.0f};
};

/** Max pooling layer for 1D float tensors. */
class MaxPoolingLayer : public LayerFunction {
  public:
    explicit MaxPoolingLayer(std::size_t window = layer_behavior().pool_window) : window_{window} {}

    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32 || x.shape().size() != 1)
            return x;
        const auto n = x.shape()[0];
        if (window_ == 0 || n < window_)
            return x;
        std::size_t out_n = n / window_;
        std::vector<float> vals(out_n);
        const float* in = reinterpret_cast<const float*>(x.data().data());
        for (std::size_t i = 0; i < out_n; ++i) {
            float maxv = in[i * window_];
            for (std::size_t j = 1; j < window_; ++j) {
                float v = in[i * window_ + j];
                if (v > maxv)
                    maxv = v;
            }
            vals[i] = maxv;
        }
        std::vector<std::byte> bytes(vals.size() * sizeof(float));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return {HTensor::DType::Float32, {out_n}, std::move(bytes)};
    }

  private:
    std::size_t window_{2};
};

/** Average pooling layer for 1D float tensors. */
class AveragePoolingLayer : public LayerFunction {
  public:
    explicit AveragePoolingLayer(std::size_t window = layer_behavior().pool_window)
        : window_{window} {}

    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32 || x.shape().size() != 1)
            return x;
        const auto n = x.shape()[0];
        if (window_ == 0 || n < window_)
            return x;
        std::size_t out_n = n / window_;
        std::vector<float> vals(out_n);
        const float* in = reinterpret_cast<const float*>(x.data().data());
        for (std::size_t i = 0; i < out_n; ++i) {
            float sum = 0.0f;
            for (std::size_t j = 0; j < window_; ++j)
                sum += in[i * window_ + j];
            vals[i] = sum / static_cast<float>(window_);
        }
        std::vector<std::byte> bytes(vals.size() * sizeof(float));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return {HTensor::DType::Float32, {out_n}, std::move(bytes)};
    }

  private:
    std::size_t window_{2};
};

/** Dropout layer for float tensors. */
class DropoutLayer : public LayerFunction {
  public:
    explicit DropoutLayer(float rate = 0.5f) : rate_{rate} {}

    HTensor operator()(const HTensor& x) const override {
        if (rate_ <= 0.0f)
            return x;
        if (x.dtype() != HTensor::DType::Float32 && x.dtype() != HTensor::DType::Float64)
            return x;
        std::vector<std::byte> bytes(x.data());
        std::mt19937 rng{std::random_device{}()};
        std::bernoulli_distribution keep(1.0f - rate_);
        if (x.dtype() == HTensor::DType::Float32) {
            const float* src = reinterpret_cast<const float*>(x.data().data());
            float* dst = reinterpret_cast<float*>(bytes.data());
            std::size_t n = x.data().size() / sizeof(float);
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = keep(rng) ? src[i] : 0.0f;
        } else {
            const double* src = reinterpret_cast<const double*>(x.data().data());
            double* dst = reinterpret_cast<double*>(bytes.data());
            std::size_t n = x.data().size() / sizeof(double);
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = keep(rng) ? src[i] : 0.0;
        }
        return {x.dtype(), x.shape(), std::move(bytes)};
    }

  private:
    float rate_{0.5f};
};

/** Register simple built-in layer implementations. */
inline void register_builtin_layers() {
    registerLayer("conv", std::make_shared<ConvolutionLayer>());
    registerLayer("norm", std::make_shared<NormalizationLayer>());
    registerLayer("attention", std::make_shared<AttentionLayer>());
    registerLayer("cross_attention", std::make_shared<CrossAttentionLayer>());
    registerLayer("multihead_attention",
                  std::make_shared<MultiHeadAttentionLayer>(layer_behavior().attention_heads));
    registerLayer("max_pool", std::make_shared<MaxPoolingLayer>());
    registerLayer("avg_pool", std::make_shared<AveragePoolingLayer>());
    registerLayer("dropout", std::make_shared<DropoutLayer>());
}

} // namespace harmonics
