#include <gtest/gtest.h>
#include <harmonics/int8_math.hpp>

TEST(Int8DeterminismTest, DigestStable) {
    std::vector<int8_t> A = {1, 2, 3, 4};
    std::vector<int8_t> B = {5, 6, 7, 8};
    auto result = harmonics::int8_matmul(A, B, 2, 2, 2);
    auto digest = harmonics::digest(result);
    EXPECT_EQ(digest, "f5d551b0eb8c185f4ad794fcdeb022cebe15a23ec1a6b8e6499c20a4e8a8d021");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
