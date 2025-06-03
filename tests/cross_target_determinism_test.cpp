#include <cstring>
#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>

using namespace harmonics;

namespace {
struct ConstantProducer : Producer {
    HTensor next() override {
        float v = 42.0f;
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &v, sizeof(float));
        return HTensor{HTensor::DType::Float32, {1}, std::move(data)};
    }
    std::size_t size() const override { return 1; }
};

struct IdActivation : ActivationFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

float run_inference(Backend backend) {
    const char* src = "producer p {1}; layer l; cycle { p -(id)-> l; }";
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);
    auto prod = std::make_shared<ConstantProducer>();
    g.bindProducer("p", prod);
    registerActivation("id", std::make_shared<IdActivation>());

    DeploymentDescriptor desc;
    desc.backend = backend;
    auto state = g.inference(desc);
    const float* out = reinterpret_cast<const float*>(state.layer_tensors[0].data().data());
    return out ? out[0] : 0.0f;
}
} // namespace

TEST(CrossTargetDeterminism, CpuCudaFpgaMatch) {
    float cpu = run_inference(Backend::CPU);
    float gpu = run_inference(Backend::GPU);
    float fpga = run_inference(Backend::FPGA);
    EXPECT_FLOAT_EQ(cpu, gpu);
    EXPECT_FLOAT_EQ(cpu, fpga);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
