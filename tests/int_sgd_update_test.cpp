#include <gtest/gtest.h>
#include <harmonics/int8_math.hpp>
#include <harmonics/runtime.hpp>

TEST(IntSGDUpdateTest, BasicUpdate) {
    std::vector<int8_t> w = {0, 0, 0};
    std::vector<int32_t> g = {16, -16, 32};
    harmonics::apply_integer_sgd_update(w, g, 4);
    EXPECT_EQ(w[0], static_cast<int8_t>(-1));
    EXPECT_EQ(w[1], static_cast<int8_t>(1));
    EXPECT_EQ(w[2], static_cast<int8_t>(-2));
}

TEST(IntSGDUpdateTest, Saturation) {
    std::vector<int8_t> w = {120};
    std::vector<int32_t> g = {-1024};
    harmonics::apply_integer_sgd_update(w, g, 2);
    EXPECT_EQ(w[0], static_cast<int8_t>(127));
}

TEST(IntSGDUpdateTest, ScheduleHTensor) {
    harmonics::HTensor w{harmonics::HTensor::DType::UInt8, {3}};
    w.data().resize(3);
    harmonics::HTensor g{harmonics::HTensor::DType::Int32, {3}};
    g.data().resize(3 * sizeof(int32_t));
    auto* wp = reinterpret_cast<int8_t*>(w.data().data());
    auto* gp = reinterpret_cast<int32_t*>(g.data().data());
    wp[0] = 0;
    wp[1] = 0;
    wp[2] = 0;
    gp[0] = 16;
    gp[1] = -16;
    gp[2] = 32;
    harmonics::StepDecayShiftSchedule sched{2, 1, 4};
    harmonics::apply_integer_sgd_update(w, g, sched, 0);
    EXPECT_EQ(wp[0], static_cast<int8_t>(-4));
    EXPECT_EQ(wp[1], static_cast<int8_t>(4));
    EXPECT_EQ(wp[2], static_cast<int8_t>(-8));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
