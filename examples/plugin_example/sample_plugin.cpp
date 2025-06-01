#include <harmonics/function_registry.hpp>
#include <memory>

struct ExampleActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

extern "C" void harmonics_register(harmonics::FunctionRegistry& reg) {
    reg.register_activation("example_act", std::make_shared<ExampleActivation>());
}

extern "C" int harmonics_plugin_version() { return 1; }
