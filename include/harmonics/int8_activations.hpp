#pragma once

#include "harmonics/core.hpp"
#include "harmonics/function_registry.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace harmonics {

inline const std::array<int8_t, 256>& relu_table() {
    static std::array<int8_t, 256> tbl{};
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) {
            int v = i - 128;
            tbl[i] = static_cast<int8_t>(v < 0 ? 0 : v);
        }
        init = true;
    }
    return tbl;
}

class Int8ReluActivation : public ActivationFunction {
  public:
    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::UInt8)
            return x;
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
};

inline const std::array<int8_t, 256>& hard_sigmoid_table() {
    static std::array<int8_t, 256> tbl{};
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) {
            float x = static_cast<float>(i - 128) / 127.0f;
            float y = 0.2f * x + 0.5f;
            if (y < 0.0f)
                y = 0.0f;
            if (y > 1.0f)
                y = 1.0f;
            tbl[i] = static_cast<int8_t>(std::round(y * 127.0f));
        }
        init = true;
    }
    return tbl;
}

class Int8HardSigmoidActivation : public ActivationFunction {
  public:
    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::UInt8)
            return x;
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
};

inline const std::array<uint16_t, 256>& exp_table() {
    static std::array<uint16_t, 256> tbl{};
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) {
            float x = static_cast<float>(i - 128) / 16.0f; // [-8,7.9375]
            float e = std::exp(x);
            tbl[i] = static_cast<uint16_t>(std::round(e * 256.0f));
        }
        init = true;
    }
    return tbl;
}

class Int8SoftmaxActivation : public ActivationFunction {
  public:
    HTensor operator()(const HTensor& x) const override {
        if (x.dtype() != HTensor::DType::UInt8 || x.shape().size() != 1)
            return x;
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
};

inline void register_int8_lut_activations() {
    registerActivation("int8_relu", std::make_shared<Int8ReluActivation>());
    registerActivation("int8_hardsigmoid", std::make_shared<Int8HardSigmoidActivation>());
    registerActivation("int8_softmax", std::make_shared<Int8SoftmaxActivation>());
}

} // namespace harmonics
