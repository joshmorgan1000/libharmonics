#define HARMONICS_HAS_VULKAN 0
#define HARMONICS_HAS_CUDA 0

#include <gtest/gtest.h>
#include <harmonics/gpu_backend.hpp>

TEST(DeviceBufferPoolTest, RecyclesBuffers) {
    auto buf1 = harmonics::pool_alloc(16);
    EXPECT_EQ(harmonics::device_pool_size(), 0u);
    harmonics::pool_release(std::move(buf1));
    EXPECT_EQ(harmonics::device_pool_size(), 1u);

    auto buf2 = harmonics::pool_alloc(8);
    // Should reuse buf1 from the pool
    EXPECT_EQ(harmonics::device_pool_size(), 0u);
    harmonics::pool_release(std::move(buf2));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
