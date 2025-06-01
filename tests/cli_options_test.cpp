#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>

TEST(CliOptionsTest, HarmonicsRunWithPluginPathAndBits) {
    const char* src = "producer p {1}; layer l; cycle { p -(plug_act)-> l; }";
    {
        std::ofstream out("cli_test.harp");
        out << src;
    }
    ASSERT_EQ(std::system("./harmonics_cli --compile cli_test.harp -o cli_test.hgr > /dev/null"),
              0);
    ASSERT_EQ(
        std::system("./harmonics_cli --run cli_test.hgr --plugin-path . --bits 16 > /dev/null"), 0);
    std::remove("cli_test.harp");
    std::remove("cli_test.hgr");
}

TEST(CliOptionsTest, GraphCliInfoWithBits) {
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    {
        std::ofstream out("cli_graph.harp");
        out << src;
    }
    ASSERT_EQ(std::system("./harmonics_cli --compile cli_graph.harp -o cli_graph.hgr > /dev/null"),
              0);
    ASSERT_EQ(std::system("./graph_cli info cli_graph.hgr --bits 8 > /dev/null"), 0);
    std::remove("cli_graph.harp");
    std::remove("cli_graph.hgr");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
