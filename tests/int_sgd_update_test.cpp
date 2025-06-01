#include <gtest/gtest.h>
#include <harmonics/int8_math.hpp>

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
