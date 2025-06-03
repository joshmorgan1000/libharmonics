#include <cstdint>
#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>

TEST(ConstantSlabIntegrationTest, StateInitialisation) {
    harmonics::CycleState state{};
    for (std::size_t i = 0; i < harmonics::MAX_VARIABLE_SLOTS; ++i) {
        EXPECT_EQ(state.variables.sensor_active[i], 0);
        EXPECT_EQ(state.variables.appendage_active[i], 0);
    }
}

TEST(ConstantSlabIntegrationTest, MemoryLayout) {
    harmonics::ConstantSlab<float> slab{};
    for (std::size_t i = 0; i < harmonics::MAX_VARIABLE_SLOTS; ++i) {
        EXPECT_EQ(slab.sensor_slot(i),
                  slab.sensor_data + harmonics::ConstantSlab<float>::slot_offset(i));
        EXPECT_EQ(slab.appendage_slot(i),
                  slab.appendage_data + harmonics::ConstantSlab<float>::slot_offset(i));
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(slab.sensor_slot(i)) % 32, 0u);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(slab.appendage_slot(i)) % 32, 0u);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
