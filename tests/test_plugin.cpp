#include <harmonics/function_registry.hpp>
#include <memory>

struct PlugActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

struct PlugLoss : harmonics::LossFunction {
    harmonics::HTensor operator()(const harmonics::HTensor&,
                                  const harmonics::HTensor&) const override {
        return harmonics::HTensor{};
    }
};

struct PlugLayer : harmonics::LayerFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

extern "C" void harmonics_register(harmonics::FunctionRegistry& reg) {
    reg.register_activation("plug_act", std::make_shared<PlugActivation>());
    reg.register_loss("plug_loss", std::make_shared<PlugLoss>());
    reg.register_layer("plug_layer", std::make_shared<PlugLayer>());
}

extern "C" int harmonics_plugin_version() { return 1; }
