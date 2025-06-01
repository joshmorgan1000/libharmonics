#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/serialization.hpp>

TEST(GraphInfoCli, PrintsStructure) {
    const char* src = "producer p {1}; consumer c {1}; layer l; cycle { p -> l; l -> c; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    {
        std::ofstream out("info_cli.hgr", std::ios::binary);
        harmonics::save_graph(g, out);
    }

    FILE* pipe = popen("./graph_info info_cli.hgr", "r");
    ASSERT_EQ(pipe != nullptr, true);
    char buf[256];
    std::string out;
    while (fgets(buf, sizeof(buf), pipe))
        out += buf;
    int status = pclose(pipe);
    ASSERT_EQ(WIFEXITED(status), true);
    ASSERT_EQ(WEXITSTATUS(status), 0);
    EXPECT_EQ(out.find("producer p") != std::string::npos, true);
    EXPECT_EQ(out.find("layer l") != std::string::npos, true);
    EXPECT_EQ(out.find("p -> l") != std::string::npos, true);
    std::remove("info_cli.hgr");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
