#include <gtest/gtest.h>
#include <harmonics/training.hpp>

TEST(FpLRScheduleTest, ExponentialDecay) {
    harmonics::ExponentialDecaySchedule sched{0.1f, 0.5f, 2, 0.01f};
    EXPECT_FLOAT_EQ(sched(0), 0.1f);
    EXPECT_FLOAT_EQ(sched(1), 0.1f);
    EXPECT_FLOAT_EQ(sched(2), 0.05f);
    EXPECT_FLOAT_EQ(sched(4), 0.025f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
