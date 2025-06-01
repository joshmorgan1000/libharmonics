#include <gtest/gtest.h>
#include <harmonics/int8_math.hpp>

TEST(IntLRScheduleTest, StepDecayIncrements) {
    harmonics::StepDecayShiftSchedule sched{2, 3, 5};
    EXPECT_EQ(sched(0), 2);
    EXPECT_EQ(sched(2), 2);
    EXPECT_EQ(sched(3), 3);
    EXPECT_EQ(sched(8), 4);
    EXPECT_EQ(sched(15), 5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
