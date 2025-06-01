#define HARMONICS_PLUGIN_IMPL
#include <harmonics/function_registry.hpp>
#include <harmonics/plugin.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::string path;
    if (argc > 1)
        path = argv[1];
    auto handles = harmonics::load_plugins_from_path(path);
    std::cout << "Loaded " << handles.size() << " plugin(s)" << std::endl;
    const auto& act = harmonics::getActivation("example_act");
    harmonics::HTensor t{};
    act(t);
    harmonics::unload_plugins(handles);
}
