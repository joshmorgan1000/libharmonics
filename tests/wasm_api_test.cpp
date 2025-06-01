#include <gtest/gtest.h>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/wasm_api.hpp>

struct DummyProducer : harmonics::Producer {
    harmonics::HTensor next() override {
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}};
    }
    std::size_t size() const override { return 1; }
};

TEST(WasmApiTest, CreateRunDestroyRuntime) {
    const char* src = "producer p{1}; consumer c; cycle{ p -> c; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto* g = new harmonics::HarmonicGraph(harmonics::build_graph(ast));
    auto prod = std::make_shared<DummyProducer>();
    g->bindProducer("p", prod);

    auto* rt = harmonics::wasm_create_runtime(g);
    ASSERT_EQ(rt != nullptr, true);
    rt->forward();
    harmonics::wasm_destroy_runtime(rt);
    harmonics::wasm_destroy_graph(g);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
