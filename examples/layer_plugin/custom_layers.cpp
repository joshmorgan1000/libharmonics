#include <harmonics/function_registry.hpp>
#include <harmonics/layers.hpp>
#include <memory>

// Simple plugin overriding the builtin conv, norm and attention layers.
// The implementations here demonstrate custom behaviour.

struct WideConvolution : harmonics::LayerFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override {
        harmonics::ConvolutionLayer conv{5};
        return conv(x);
    }
};

struct IdentityNorm : harmonics::LayerFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

struct NegatingAttention : harmonics::LayerFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override {
        auto base = harmonics::AttentionLayer{}(x);
        auto out = base;
        if (out.dtype() == harmonics::HTensor::DType::Float32) {
            auto* p = reinterpret_cast<float*>(out.data().data());
            for (std::size_t i = 0; i < out.data().size() / sizeof(float); ++i)
                p[i] = -p[i];
        }
        return out;
    }
};

extern "C" void harmonics_register(harmonics::FunctionRegistry& reg) {
    reg.register_layer("conv", std::make_shared<WideConvolution>());
    reg.register_layer("norm", std::make_shared<IdentityNorm>());
    reg.register_layer("attention", std::make_shared<NegatingAttention>());
}
