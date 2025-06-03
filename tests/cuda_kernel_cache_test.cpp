#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <harmonics/deployment.hpp>
#include <vector>

static std::string shader_cache_directory() {
    const char* env = std::getenv("HARMONICS_SHADER_CACHE");
    std::string dir = env ? env : "shader_cache";
    std::filesystem::create_directories(dir);
    return dir;
}

static std::optional<std::vector<uint32_t>> load_cached_cuda_kernel(const std::string& name) {
    std::string key = harmonics::blake3(name.data(), name.size());
    std::filesystem::path path = shader_cache_directory();
    path /= key + ".ptx";
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::nullopt;
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint32_t> data(static_cast<std::size_t>(size) / sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

static void save_cached_cuda_kernel(const std::string& name, const std::vector<uint32_t>& ptx) {
    std::string key = harmonics::blake3(name.data(), name.size());
    std::filesystem::path path = shader_cache_directory();
    path /= key + ".ptx";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(ptx.data()),
              static_cast<std::streamsize>(ptx.size() * sizeof(uint32_t)));
}

TEST(CudaKernelCacheTest, PersistAndLoad) {
    std::filesystem::path cache_dir =
        std::filesystem::temp_directory_path() / "harmonics_cuda_cache_test";
    std::filesystem::remove_all(cache_dir);
    std::filesystem::create_directories(cache_dir);
    setenv("HARMONICS_SHADER_CACHE", cache_dir.c_str(), 1);

    std::vector<uint32_t> data = {1, 2, 3, 4};
    save_cached_cuda_kernel("unit_test_kernel", data);

    auto loaded = load_cached_cuda_kernel("unit_test_kernel");
    ASSERT_EQ(loaded.has_value(), true);
    ASSERT_EQ(*loaded, data);

    std::filesystem::remove_all(cache_dir);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
