#define HARMONICS_HAS_VULKAN 1
#define HARMONICS_HAS_OPENCL 1

#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

namespace {
struct FixedProducer : harmonics::Producer {
    explicit FixedProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override {
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, shape};
    }
    std::size_t size() const override { return 1; }
    harmonics::HTensor::Shape shape{};
};
} // namespace

TEST(IntegrationFallbackTest, GpuForwardFallback) {
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);

    harmonics::DeploymentDescriptor desc;
    desc.backend = harmonics::Backend::GPU;

    harmonics::CycleRuntime rt{g, harmonics::make_auto_policy(), desc};
    EXPECT_EQ(rt.backend(), harmonics::Backend::CPU);
    rt.forward();
}

TEST(IntegrationFallbackTest, FpgaForwardFallback) {
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    auto prod = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);

    harmonics::DeploymentDescriptor desc;
    desc.backend = harmonics::Backend::FPGA;

    harmonics::CycleRuntime rt{g, harmonics::make_auto_policy(), desc};
    EXPECT_EQ(rt.backend(), harmonics::Backend::CPU);
    rt.forward();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
