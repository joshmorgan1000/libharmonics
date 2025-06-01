#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>

TEST(ConstantSlabIntegrationTest, StateInitialisation) {
    harmonics::CycleState state{};
    for (std::size_t i = 0; i < harmonics::MAX_VARIABLE_SLOTS; ++i) {
        EXPECT_EQ(state.variables.sensor_active[i], 0);
        EXPECT_EQ(state.variables.appendage_active[i], 0);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
