#include <gpu/Wrapper.h>
#include <gtest/gtest.h>

TEST(WrapperRingBufferTest, AcquireCyclesThroughBuffers) {
    harmonics::Wrapper<int> ring(2);
    int* first = &ring.acquire();
    *first = 1;
    int* second = &ring.acquire();
    *second = 2;
    int* third = &ring.acquire();
    EXPECT_EQ(third, first);
    EXPECT_EQ(&ring.current(), third);
    EXPECT_EQ(ring.size(), 2u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
