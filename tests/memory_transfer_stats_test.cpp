#define HARMONICS_HAS_CUDA 1
#define HARMONICS_HAS_VULKAN 0

#include <gtest/gtest.h>
#include <harmonics/gpu_backend.hpp>
#include <harmonics/memory_profiler.hpp>

TEST(MemoryTransferStatsTest, RecordsBytesAndTime) {
    harmonics::HTensor t{harmonics::HTensor::DType::Float32, {4}};
    t.data().resize(sizeof(float) * 4);

    harmonics::reset_memory_transfer_stats();

    auto dev = harmonics::to_device(t);
    auto host = harmonics::to_host(dev);

    auto stats = harmonics::memory_transfer_stats();
    EXPECT_GT(stats.bytes_to_device, 0u);
    EXPECT_GT(stats.bytes_to_host, 0u);
    EXPECT_GT(stats.ns_to_device, 0u);
    EXPECT_GT(stats.ns_to_host, 0u);
    (void)host; // suppress unused variable warning
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
