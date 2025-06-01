#include <gtest/gtest.h>
#define HARMONICS_PLUGIN_IMPL
#include <future>
#include <harmonics/function_registry.hpp>
#include <harmonics/plugin.hpp>

TEST(PluginAsyncTest, LoadAndUnloadAsync) {
    auto load_future = harmonics::load_plugins_from_path_async(".");
    auto handles = load_future.get();

    const auto& act = harmonics::getActivation("plug_act");
    harmonics::HTensor t{};
    (void)act(t);

    EXPECT_EQ(false, handles.empty());

    auto unload_future = harmonics::unload_plugins_async(handles);
    unload_future.get();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
