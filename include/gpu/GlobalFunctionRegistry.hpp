#pragma once

#include "gpu/GPUFunction.hpp"
#include "gpu/Shaders.hpp"
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <zstd.h>

namespace harmonics {

inline std::vector<uint32_t> decompressShader(const std::vector<uint8_t>& comp) {
    size_t size = ZSTD_getFrameContentSize(comp.data(), comp.size());
    std::vector<uint8_t> tmp(size);
    size_t got = ZSTD_decompress(tmp.data(), size, comp.data(), comp.size());
    if (ZSTD_isError(got))
        throw std::runtime_error("shader decompression failed");
    std::vector<uint32_t> out(size / sizeof(uint32_t));
    std::memcpy(out.data(), tmp.data(), size);
    return out;
}

inline const std::vector<uint8_t>* shaderData(const std::string& name) {
    static const std::unordered_map<std::string, const std::vector<uint8_t>*> table = {
        {"adam", &ADAM_COMP_ZST},
        {"barycentric_scores", &BARYCENTRIC_SCORES_COMP_ZST},
        {"cross_entropy_loss", &CROSS_ENTROPY_LOSS_COMP_ZST},
        {"fully_connected", &FULLY_CONNECTED_COMP_ZST},
        {"l2_distance", &L2_DISTANCE_COMP_ZST},
        {"int8_matmul", &INT8_MATMUL_COMP_ZST},
        {"int4_matmul", &INT4_MATMUL_COMP_ZST},
        {"int3_matmul", &INT3_MATMUL_COMP_ZST},
        {"relu", &RELU_COMP_ZST},
        {"selu", &SELU_COMP_ZST},
        {"prelu", &PRELU_COMP_ZST},
        {"rmsprop", &RMSPROP_COMP_ZST},
        {"sgd", &SGD_COMP_ZST},
        {"sigmoid", &SIGMOID_COMP_ZST},
        {"mse_loss", &MSE_LOSS_COMP_ZST},
        {"max_pool", nullptr},
        {"avg_pool", nullptr},
    };
    auto it = table.find(name);
    if (it == table.end())
        return nullptr;
    return it->second;
}

class GPUFunctionRegistry {
  public:
    static GPUFunctionRegistry& getInstance() {
        static GPUFunctionRegistry inst;
        return inst;
    }

    void registerFunction(GPUFunction fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        funcs_.emplace(fn.name, std::move(fn));
    }

    const GPUFunction* get(const std::string& name) const {
        auto it = funcs_.find(name);
        if (it == funcs_.end())
            return nullptr;
        return &it->second;
    }

    const std::unordered_map<std::string, GPUFunction>& all() const { return funcs_; }

    std::vector<uint32_t> getKernel(const std::string& name) const {
        const auto* comp = shaderData(name);
        if (!comp)
            return {};
        return decompressShader(*comp);
    }

  private:
    std::unordered_map<std::string, GPUFunction> funcs_;
    mutable std::mutex mutex_;
};

inline void registerAllShaders() {
    auto& reg = GPUFunctionRegistry::getInstance();
    if (!reg.all().empty())
        return;

    GPUFunction l2;
    l2.name = "l2_distance";
    l2.shader = "l2_distance";
    l2.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 2)
            throw std::runtime_error("l2_distance expects 2 params");
        const auto& a = std::get<std::vector<float>>(params[0]);
        const auto& b = std::get<std::vector<float>>(params[1]);
        if (a.size() != b.size())
            throw std::runtime_error("size mismatch");
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::vector<float>{std::sqrt(sum)};
    };
    reg.registerFunction(std::move(l2));

    GPUFunction im;
    im.name = "int8_matmul";
    im.shader = "int8_matmul";
    im.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 2)
            throw std::runtime_error("int8_matmul expects 2 params");
        const auto& a = std::get<std::vector<uint8_t>>(params[0]);
        const auto& b = std::get<std::vector<uint8_t>>(params[1]);
        if (a.size() != b.size())
            throw std::runtime_error("size mismatch");
        int32_t sum = 0;
        for (size_t i = 0; i < a.size(); ++i)
            sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
        return std::vector<int32_t>{sum};
    };
    reg.registerFunction(std::move(im));

    GPUFunction im4;
    im4.name = "int4_matmul";
    im4.shader = "int4_matmul";
    im4.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 2)
            throw std::runtime_error("int4_matmul expects 2 params");
        const auto& a = std::get<std::vector<uint8_t>>(params[0]);
        const auto& b = std::get<std::vector<uint8_t>>(params[1]);
        if (a.size() != b.size())
            throw std::runtime_error("size mismatch");
        int32_t sum = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            int8_t va = static_cast<int8_t>(static_cast<int8_t>(a[i] << 4) >> 4);
            int8_t vb = static_cast<int8_t>(static_cast<int8_t>(b[i] << 4) >> 4);
            sum += static_cast<int32_t>(va) * static_cast<int32_t>(vb);
        }
        return std::vector<int32_t>{sum};
    };
    reg.registerFunction(std::move(im4));

    GPUFunction im3;
    im3.name = "int3_matmul";
    im3.shader = "int3_matmul";
    im3.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 2)
            throw std::runtime_error("int3_matmul expects 2 params");
        const auto& a = std::get<std::vector<uint8_t>>(params[0]);
        const auto& b = std::get<std::vector<uint8_t>>(params[1]);
        if (a.size() != b.size())
            throw std::runtime_error("size mismatch");
        int32_t sum = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            int8_t va = static_cast<int8_t>(static_cast<int8_t>(a[i] << 5) >> 5);
            int8_t vb = static_cast<int8_t>(static_cast<int8_t>(b[i] << 5) >> 5);
            sum += static_cast<int32_t>(va) * static_cast<int32_t>(vb);
        }
        return std::vector<int32_t>{sum};
    };
    reg.registerFunction(std::move(im3));

    GPUFunction relu;
    relu.name = "relu";
    relu.shader = "relu";
    relu.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 1)
            throw std::runtime_error("relu expects 1 param");
        const auto& in = std::get<std::vector<float>>(params[0]);
        std::vector<float> out(in.size());
        for (size_t i = 0; i < in.size(); ++i)
            out[i] = in[i] > 0.0f ? in[i] : 0.0f;
        return out;
    };
    reg.registerFunction(std::move(relu));

    GPUFunction sigmoid;
    sigmoid.name = "sigmoid";
    sigmoid.shader = "sigmoid";
    sigmoid.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 1)
            throw std::runtime_error("sigmoid expects 1 param");
        const auto& in = std::get<std::vector<float>>(params[0]);
        std::vector<float> out(in.size());
        for (size_t i = 0; i < in.size(); ++i)
            out[i] = 1.0f / (1.0f + std::exp(-in[i]));
        return out;
    };
    reg.registerFunction(std::move(sigmoid));

    GPUFunction selu;
    selu.name = "selu";
    selu.shader = "selu";
    selu.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 1)
            throw std::runtime_error("selu expects 1 param");
        const auto& in = std::get<std::vector<float>>(params[0]);
        std::vector<float> out(in.size());
        constexpr float lambda = 1.050701f;
        constexpr float alpha = 1.67326f;
        for (size_t i = 0; i < in.size(); ++i) {
            float v = in[i];
            out[i] = v > 0.0f ? lambda * v : lambda * (alpha * (std::exp(v) - 1.0f));
        }
        return out;
    };
    reg.registerFunction(std::move(selu));

    GPUFunction prelu;
    prelu.name = "prelu";
    prelu.shader = "prelu";
    prelu.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 1)
            throw std::runtime_error("prelu expects 1 param");
        const auto& in = std::get<std::vector<float>>(params[0]);
        std::vector<float> out(in.size());
        constexpr float a = 0.25f;
        for (size_t i = 0; i < in.size(); ++i) {
            float v = in[i];
            out[i] = v > 0.0f ? v : a * v;
        }
        return out;
    };
    reg.registerFunction(std::move(prelu));

    GPUFunction bary;
    bary.name = "barycentric_scores";
    bary.shader = "barycentric_scores";
    bary.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 1)
            throw std::runtime_error("barycentric_scores expects 1 param");
        return params[0];
    };
    reg.registerFunction(std::move(bary));

    GPUFunction cel;
    cel.name = "cross_entropy_loss";
    cel.shader = "cross_entropy_loss";
    cel.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 2)
            throw std::runtime_error("cross_entropy_loss expects 2 params");
        const auto& pred = std::get<std::vector<float>>(params[0]);
        const auto& label = std::get<std::vector<float>>(params[1]);
        if (pred.size() != label.size())
            throw std::runtime_error("size mismatch");
        std::vector<float> out(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            float p = std::clamp(pred[i], 1e-7f, 1.0f - 1e-7f);
            out[i] = -label[i] * std::log(p);
        }
        return out;
    };
    reg.registerFunction(std::move(cel));

    GPUFunction mse;
    mse.name = "mse_loss";
    mse.shader = "mse_loss";
    mse.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 2)
            throw std::runtime_error("mse_loss expects 2 params");
        const auto& pred = std::get<std::vector<float>>(params[0]);
        const auto& target = std::get<std::vector<float>>(params[1]);
        if (pred.size() != target.size())
            throw std::runtime_error("size mismatch");
        std::vector<float> out(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            float diff = pred[i] - target[i];
            out[i] = diff * diff;
        }
        return out;
    };
    reg.registerFunction(std::move(mse));

    GPUFunction fc;
    fc.name = "fully_connected";
    fc.shader = "fully_connected";
    fc.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 2)
            throw std::runtime_error("fully_connected expects 2 params");
        const auto& w = std::get<std::vector<float>>(params[0]);
        const auto& x = std::get<std::vector<float>>(params[1]);
        if (w.size() != x.size())
            throw std::runtime_error("size mismatch");
        float sum = 0.0f;
        for (size_t i = 0; i < w.size(); ++i)
            sum += w[i] * x[i];
        return std::vector<float>{sum};
    };
    reg.registerFunction(std::move(fc));

    GPUFunction maxp;
    maxp.name = "max_pool";
    maxp.shader = "max_pool";
    maxp.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 1)
            throw std::runtime_error("max_pool expects 1 param");
        const auto& in = std::get<std::vector<float>>(params[0]);
        std::vector<float> out(in.size() / 2);
        for (size_t i = 0; i < out.size(); ++i) {
            float a = in[i * 2];
            float b = in[i * 2 + 1];
            out[i] = a > b ? a : b;
        }
        return out;
    };
    reg.registerFunction(std::move(maxp));

    GPUFunction avgp;
    avgp.name = "avg_pool";
    avgp.shader = "avg_pool";
    avgp.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 1)
            throw std::runtime_error("avg_pool expects 1 param");
        const auto& in = std::get<std::vector<float>>(params[0]);
        std::vector<float> out(in.size() / 2);
        for (size_t i = 0; i < out.size(); ++i) {
            float a = in[i * 2];
            float b = in[i * 2 + 1];
            out[i] = (a + b) * 0.5f;
        }
        return out;
    };
    reg.registerFunction(std::move(avgp));

    GPUFunction sgd;
    sgd.name = "sgd";
    sgd.shader = "sgd";
    sgd.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 2)
            throw std::runtime_error("sgd expects 2 params");
        const auto& param = std::get<std::vector<float>>(params[0]);
        const auto& grad = std::get<std::vector<float>>(params[1]);
        if (param.size() != grad.size())
            throw std::runtime_error("size mismatch");
        std::vector<float> out(param.size());
        constexpr float lr = 0.01f;
        for (size_t i = 0; i < param.size(); ++i)
            out[i] = param[i] - lr * grad[i];
        return out;
    };
    reg.registerFunction(std::move(sgd));

    GPUFunction rms;
    rms.name = "rmsprop";
    rms.shader = "rmsprop";
    rms.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 3)
            throw std::runtime_error("rmsprop expects 3 params");
        const auto& param = std::get<std::vector<float>>(params[0]);
        const auto& grad = std::get<std::vector<float>>(params[1]);
        const auto& s = std::get<std::vector<float>>(params[2]);
        if (param.size() != grad.size() || param.size() != s.size())
            throw std::runtime_error("size mismatch");
        std::vector<float> out(param.size());
        constexpr float lr = 0.01f;
        constexpr float decay = 0.9f;
        constexpr float eps = 1e-8f;
        for (size_t i = 0; i < param.size(); ++i) {
            float s_new = decay * s[i] + (1.0f - decay) * grad[i] * grad[i];
            out[i] = param[i] - lr * grad[i] / (std::sqrt(s_new) + eps);
        }
        return out;
    };
    reg.registerFunction(std::move(rms));

    GPUFunction adam;
    adam.name = "adam";
    adam.shader = "adam";
    adam.cpuFallback = [](const std::vector<GPUDataVariant>& params) -> GPUDataVariant {
        if (params.size() != 4)
            throw std::runtime_error("adam expects 4 params");
        const auto& param = std::get<std::vector<float>>(params[0]);
        const auto& grad = std::get<std::vector<float>>(params[1]);
        const auto& m = std::get<std::vector<float>>(params[2]);
        const auto& v = std::get<std::vector<float>>(params[3]);
        if (param.size() != grad.size() || param.size() != m.size() || param.size() != v.size())
            throw std::runtime_error("size mismatch");
        std::vector<float> out(param.size());
        constexpr float lr = 0.01f;
        constexpr float beta1 = 0.9f;
        constexpr float beta2 = 0.999f;
        constexpr float eps = 1e-8f;
        constexpr float invBias1 = 1.0f / (1.0f - beta1);
        constexpr float invBias2 = 1.0f / (1.0f - beta2);
        for (size_t i = 0; i < param.size(); ++i) {
            float m_new = beta1 * m[i] + (1.0f - beta1) * grad[i];
            float v_new = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
            float m_hat = m_new * invBias1;
            float v_hat = v_new * invBias2;
            out[i] = param[i] - lr * m_hat / (std::sqrt(v_hat) + eps);
        }
        return out;
    };
    reg.registerFunction(std::move(adam));
}

} // namespace harmonics
