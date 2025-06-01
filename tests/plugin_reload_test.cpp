#include <gtest/gtest.h>
#define HARMONICS_PLUGIN_IMPL
#include <harmonics/function_registry.hpp>
#include <harmonics/plugin.hpp>

TEST(PluginReloadTest, ReloadPlugin) {
    auto h1 = harmonics::load_plugin("./libtest_plugin.so");
    EXPECT_EQ(1u, harmonics::plugin_cache.size());
    harmonics::HTensor t{};
    (void)harmonics::getActivation("plug_act")(t);

    auto h2 = harmonics::reload_plugin("./libtest_plugin.so");
    EXPECT_EQ(1u, harmonics::plugin_cache.size());
    (void)harmonics::getLayer("plug_layer")(t);

    harmonics::unload_plugin(h2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
