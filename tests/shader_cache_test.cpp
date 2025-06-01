#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

TEST(ShaderCacheTest, PersistentCacheLoadsFromDisk) {
    const char* src = "producer p; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);

    std::filesystem::path cache_dir =
        std::filesystem::temp_directory_path() / "harmonics_shader_cache_test";
    std::filesystem::remove_all(cache_dir);
    std::filesystem::create_directories(cache_dir);
    setenv("HARMONICS_SHADER_CACHE", cache_dir.c_str(), 1);

    harmonics::shader_compile_cache().clear();

    auto policy = harmonics::make_auto_policy();
    (void)harmonics::compile_cycle_kernels(g, *policy);

    std::string key = harmonics::blake3("identity_32", 11);
    std::filesystem::path shader = cache_dir / (key + ".spv");
    EXPECT_EQ(std::filesystem::exists(shader), true);

    auto cached = harmonics::load_cached_shader("identity_32");
    EXPECT_EQ(cached.has_value(), true);

    std::filesystem::remove_all(cache_dir);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
