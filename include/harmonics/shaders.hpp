#pragma once

#include "harmonics/config.hpp"
#include "harmonics/core.hpp"
#include "harmonics/function_registry.hpp"
#include "harmonics/int8_activations.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif
#if defined(__SSE2__)
#include <immintrin.h>
#endif

namespace harmonics {

class ReluActivation : public ActivationFunction {
  public:
    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() == HTensor::DType::UInt8) {
            HTensor out{HTensor::DType::UInt8, x.shape()};
            out.data().resize(x.data().size());
            const int8_t* in = reinterpret_cast<const int8_t*>(x.data().data());
            int8_t* p = reinterpret_cast<int8_t*>(out.data().data());
            const auto& tbl = relu_table();
            std::size_t n = x.data().size();
            for (std::size_t i = 0; i < n; ++i)
                p[i] = tbl[static_cast<uint8_t>(in[i])];
            return out;
        }
        if (x.dtype() != HTensor::DType::Float32)
            return x;
        HTensor out{HTensor::DType::Float32, x.shape()};
        out.data().resize(x.data().size());
        const float* in = reinterpret_cast<const float*>(x.data().data());
        float* p = reinterpret_cast<float*>(out.data().data());
        std::size_t n = x.data().size() / sizeof(float);
#if defined(__wasm_simd128__)
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            v128_t v = wasm_v128_load(&in[i]);
            v128_t zero = wasm_f32x4_splat(0.0f);
            v128_t r = wasm_f32x4_max(v, zero);
            wasm_v128_store(&p[i], r);
        }
        for (; i < n; ++i)
            p[i] = std::max(in[i], 0.0f);
#else
        for (std::size_t i = 0; i < n; ++i)
            p[i] = std::max(in[i], 0.0f);
#endif
        return out;
    }
};

class SigmoidActivation : public ActivationFunction {
  public:
    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() == HTensor::DType::UInt8) {
            HTensor out{HTensor::DType::UInt8, x.shape()};
            out.data().resize(x.data().size());
            const int8_t* in = reinterpret_cast<const int8_t*>(x.data().data());
            int8_t* p = reinterpret_cast<int8_t*>(out.data().data());
            const auto& tbl = hard_sigmoid_table();
            std::size_t n = x.data().size();
            for (std::size_t i = 0; i < n; ++i)
                p[i] = tbl[static_cast<uint8_t>(in[i])];
            return out;
        }
        if (x.dtype() != HTensor::DType::Float32)
            return x;
        HTensor out{HTensor::DType::Float32, x.shape()};
        out.data().resize(x.data().size());
        const float* in = reinterpret_cast<const float*>(x.data().data());
        float* p = reinterpret_cast<float*>(out.data().data());
        std::size_t n = x.data().size() / sizeof(float);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = 1.0f / (1.0f + std::exp(-in[i]));
        return out;
    }
};

class SoftmaxActivation : public ActivationFunction {
  public:
    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() == HTensor::DType::UInt8 && x.shape().size() == 1) {
            std::size_t n = x.shape()[0];
            HTensor out{HTensor::DType::UInt8, {n}};
            out.data().resize(n);
            const int8_t* in = reinterpret_cast<const int8_t*>(x.data().data());
            int8_t* p = reinterpret_cast<int8_t*>(out.data().data());

            int8_t maxv = in[0];
            for (std::size_t i = 1; i < n; ++i)
                if (in[i] > maxv)
                    maxv = in[i];

            const auto& tbl = exp_table();
            std::vector<uint16_t> tmp(n);
            uint32_t sum = 0;
            for (std::size_t i = 0; i < n; ++i) {
                int diff = static_cast<int>(in[i]) - static_cast<int>(maxv);
                if (diff < -128)
                    diff = -128;
                uint16_t e = tbl[static_cast<uint8_t>(diff + 128)];
                tmp[i] = e;
                sum += e;
            }
            for (std::size_t i = 0; i < n; ++i) {
                int32_t val = (static_cast<uint32_t>(tmp[i]) * 127 + sum / 2) / sum;
                p[i] = static_cast<int8_t>(val);
            }
            return out;
        }
        if (x.dtype() != HTensor::DType::Float32 || x.shape().size() != 1)
            return x;
        std::size_t n = x.shape()[0];
        HTensor out{HTensor::DType::Float32, {n}};
        out.data().resize(n * sizeof(float));
        const float* in = reinterpret_cast<const float*>(x.data().data());
        float* p = reinterpret_cast<float*>(out.data().data());
        float maxv = in[0];
#if HARMONICS_HAS_WASM_SIMD
        {
            std::size_t i = 0;
            v128_t vmax = wasm_v128_load(&in[0]);
            for (; i + 4 <= n; i += 4) {
                v128_t v = wasm_v128_load(&in[i]);
                vmax = wasm_f32x4_max(vmax, v);
            }
            float tmp[4];
            wasm_v128_store(tmp, vmax);
            for (int j = 0; j < 4; ++j)
                if (tmp[j] > maxv)
                    maxv = tmp[j];
            for (; i < n; ++i)
                if (in[i] > maxv)
                    maxv = in[i];
        }
#elif HARMONICS_HAS_SSE2
        {
            std::size_t i = 0;
            __m128 vmax = _mm_loadu_ps(&in[0]);
            for (; i + 4 <= n; i += 4) {
                __m128 v = _mm_loadu_ps(&in[i]);
                vmax = _mm_max_ps(vmax, v);
            }
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, vmax);
            for (int j = 0; j < 4; ++j)
                if (tmp[j] > maxv)
                    maxv = tmp[j];
            for (; i < n; ++i)
                if (in[i] > maxv)
                    maxv = in[i];
        }
#else
        for (std::size_t i = 1; i < n; ++i)
            if (in[i] > maxv)
                maxv = in[i];
#endif
        for (std::size_t i = 0; i < n; ++i)
            p[i] = std::exp(in[i] - maxv);

        float sum = 0.0f;
#if HARMONICS_HAS_WASM_SIMD
        {
            std::size_t i = 0;
            v128_t acc = wasm_f32x4_splat(0.0f);
            for (; i + 4 <= n; i += 4) {
                v128_t v = wasm_v128_load(&p[i]);
                acc = wasm_f32x4_add(acc, v);
            }
            float tmp[4];
            wasm_v128_store(tmp, acc);
            sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            for (; i < n; ++i)
                sum += p[i];
        }
#elif HARMONICS_HAS_SSE2
        {
            std::size_t i = 0;
            __m128 acc = _mm_setzero_ps();
            for (; i + 4 <= n; i += 4) {
                __m128 v = _mm_loadu_ps(&p[i]);
                acc = _mm_add_ps(acc, v);
            }
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, acc);
            sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            for (; i < n; ++i)
                sum += p[i];
        }
#else
        for (std::size_t i = 0; i < n; ++i)
            sum += p[i];
#endif
        float inv_sum = 1.0f / sum;
        for (std::size_t i = 0; i < n; ++i)
            p[i] *= inv_sum;
        return out;
    }
};

class GeluActivation : public ActivationFunction {
  public:
    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32)
            return x;
        HTensor out{HTensor::DType::Float32, x.shape()};
        out.data().resize(x.data().size());
        const float* in = reinterpret_cast<const float*>(x.data().data());
        float* p = reinterpret_cast<float*>(out.data().data());
        std::size_t n = x.data().size() / sizeof(float);
        for (std::size_t i = 0; i < n; ++i) {
            float v = in[i];
            p[i] = 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
        }
        return out;
    }
};

class SeluActivation : public ActivationFunction {
  public:
    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32)
            return x;
        HTensor out{HTensor::DType::Float32, x.shape()};
        out.data().resize(x.data().size());
        const float* in = reinterpret_cast<const float*>(x.data().data());
        float* p = reinterpret_cast<float*>(out.data().data());
        constexpr float lambda = 1.050701f;
        constexpr float alpha = 1.67326f;
        std::size_t n = x.data().size() / sizeof(float);
        for (std::size_t i = 0; i < n; ++i) {
            float v = in[i];
            if (v > 0.0f)
                p[i] = lambda * v;
            else
                p[i] = lambda * (alpha * (std::exp(v) - 1.0f));
        }
        return out;
    }
};

class PReluActivation : public ActivationFunction {
  public:
    explicit PReluActivation(float a = 0.25f) : a_{a} {}

    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::Float32)
            return x;
        HTensor out{HTensor::DType::Float32, x.shape()};
        out.data().resize(x.data().size());
        const float* in = reinterpret_cast<const float*>(x.data().data());
        float* p = reinterpret_cast<float*>(out.data().data());
        std::size_t n = x.data().size() / sizeof(float);
        for (std::size_t i = 0; i < n; ++i) {
            float v = in[i];
            p[i] = v > 0.0f ? v : a_ * v;
        }
        return out;
    }

  private:
    float a_{0.25f};
};

class CrossEntropyLoss : public LossFunction {
  public:
    HTensor operator()(const HTensor& pred, const HTensor& target) const override {
        if (pred.dtype() != HTensor::DType::Float32 || target.dtype() != HTensor::DType::Float32 ||
            pred.shape() != target.shape() || pred.shape().size() != 1)
            return HTensor{};
        std::size_t n = pred.shape()[0];
        const float* p = reinterpret_cast<const float*>(pred.data().data());
        const float* t = reinterpret_cast<const float*>(target.data().data());
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float clamped = std::clamp(p[i], 1e-7f, 1.0f - 1e-7f);
            sum += -t[i] * std::log(clamped);
        }
        HTensor out{HTensor::DType::Float32, {1}};
        out.data().resize(sizeof(float));
        *reinterpret_cast<float*>(out.data().data()) = sum;
        return out;
    }
};

class MSELoss : public LossFunction {
  public:
    HTensor operator()(const HTensor& pred, const HTensor& target) const override {
        if (pred.dtype() != HTensor::DType::Float32 || target.dtype() != HTensor::DType::Float32 ||
            pred.shape() != target.shape() || pred.shape().size() != 1)
            return HTensor{};
        std::size_t n = pred.shape()[0];
        const float* p = reinterpret_cast<const float*>(pred.data().data());
        const float* t = reinterpret_cast<const float*>(target.data().data());
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float diff = p[i] - t[i];
            sum += diff * diff;
        }
        float mse = sum / static_cast<float>(n);
        HTensor out{HTensor::DType::Float32, {1}};
        out.data().resize(sizeof(float));
        *reinterpret_cast<float*>(out.data().data()) = mse;
        return out;
    }
};

inline void register_builtin_shaders() {
    registerActivation("relu", std::make_shared<ReluActivation>());
    registerActivation("sigmoid", std::make_shared<SigmoidActivation>());
    registerActivation("softmax", std::make_shared<SoftmaxActivation>());
    registerActivation("gelu", std::make_shared<GeluActivation>());
    registerActivation("selu", std::make_shared<SeluActivation>());
    registerActivation("prelu", std::make_shared<PReluActivation>());
    registerLoss("cross_entropy", std::make_shared<CrossEntropyLoss>());
    registerLoss("mse", std::make_shared<MSELoss>());
}

} // namespace harmonics
