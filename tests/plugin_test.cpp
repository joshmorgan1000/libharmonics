#include <gtest/gtest.h>
#define HARMONICS_PLUGIN_IMPL
#include <harmonics/function_registry.hpp>
#include <harmonics/plugin.hpp>

TEST(PluginTest, LoadAndRegister) {
    auto handles = harmonics::load_plugins_from_path(".");

    const auto& act = harmonics::getActivation("plug_act");
    const auto& loss = harmonics::getLoss("plug_loss");
    const auto& layer = harmonics::getLayer("plug_layer");

    harmonics::HTensor t{};
    auto a = act(t);
    auto l = loss(t, t);
    auto o = layer(t);
    (void)a;
    (void)l;
    (void)o;

    EXPECT_EQ(false, handles.empty());
    harmonics::unload_plugins(handles);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
