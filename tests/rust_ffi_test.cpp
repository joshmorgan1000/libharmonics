#include <fstream>
#include <gtest/gtest.h>
#include <harmonics/rust_ffi.hpp>

TEST(RustFFITest, ParseAndDestroyGraph) {
    const char* src = "producer p{1}; consumer c; cycle{ p -> c; }";
    harmonics::HarmonicGraph* g = harmonics::harmonics_parse_graph(src);
    ASSERT_EQ(g != nullptr, true);
    harmonics::harmonics_destroy_graph(g);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
