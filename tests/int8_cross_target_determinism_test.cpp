#include <cstring>
#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/int8_activations.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>

using namespace harmonics;

namespace {
struct ConstantInt8Producer : Producer {
    HTensor next() override {
        int8_t v = -5;
        std::vector<std::byte> data(sizeof(int8_t));
        std::memcpy(data.data(), &v, sizeof(int8_t));
        return HTensor{HTensor::DType::UInt8, {1}, std::move(data)};
    }
    std::size_t size() const override { return 1; }
};

int8_t run_inference(Backend backend) {
    const char* src = "producer p {1}; layer l; cycle { p -(int8_relu)-> l; }";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    auto prod = std::make_shared<ConstantInt8Producer>();
    g.bindProducer("p", prod);
    register_int8_lut_activations();

    DeploymentDescriptor desc;
    desc.backend = backend;
    auto state = g.inference(desc);
    const int8_t* out = reinterpret_cast<const int8_t*>(state.layer_tensors[0].data().data());
    return out ? out[0] : 0;
}
} // namespace

TEST(Int8CrossTargetDeterminism, CpuGpuFpgaMatch) {
    int8_t cpu = run_inference(Backend::CPU);
    int8_t gpu = run_inference(Backend::GPU);
    int8_t fpga = run_inference(Backend::FPGA);
    EXPECT_EQ(cpu, gpu);
    EXPECT_EQ(cpu, fpga);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
