#include <gtest/gtest.h>
#include <harmonics/precision_policy.hpp>

TEST(PrecisionUtilsTest, BitsFromEntropyClampsAndRounds) {
    EXPECT_EQ(harmonics::bits_from_entropy(1.0f), 2);            // clamp minimum
    EXPECT_EQ(harmonics::bits_from_entropy(0.0f), 32);           // <=0 returns 32
    EXPECT_EQ(harmonics::bits_from_entropy(std::exp2(-40)), 32); // clamp maximum
}

TEST(PrecisionUtilsTest, BitsFromHardwareMatchesBackend) {
#if HARMONICS_HAS_VULKAN
    EXPECT_EQ(harmonics::bits_from_hardware(), 16);
#else
    EXPECT_EQ(harmonics::bits_from_hardware(), 32);
#endif
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
