#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#define HARMONICS_PLUGIN_IMPL
#include <harmonics/function_registry.hpp>
#include <harmonics/plugin.hpp>

TEST(PluginPackagerTest, PackageAndInstall) {
    namespace fs = std::filesystem;
    fs::path plugin_dir = "pack_plugin";
    fs::create_directory(plugin_dir);
    fs::copy_file("libtest_plugin.so", plugin_dir / "libtest_plugin.so",
                  fs::copy_options::overwrite_existing);
    std::ofstream(plugin_dir / "plugin.json") << "{}";

    ASSERT_EQ(std::system("./plugin_packager package pack_plugin plugin.tar.zst > /dev/null"), 0);
    fs::create_directory("install_dir");
    ASSERT_EQ(std::system("./plugin_packager install plugin.tar.zst install_dir > /dev/null"), 0);

    auto handles = harmonics::load_plugins_from_path("install_dir");
    EXPECT_EQ(handles.empty(), false);
    harmonics::unload_plugins(handles);

    fs::remove_all(plugin_dir);
    fs::remove_all("install_dir");
    std::remove("plugin.tar.zst");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
