#include <cmath>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/introspection.hpp>
#include <harmonics/layers.hpp>
#include <harmonics/model_import.hpp>
#include <harmonics/parser.hpp>
#include <onnx/onnx_pb.h>

using harmonics::HTensor;

struct IdActivation : harmonics::ActivationFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

struct ConstantProducer : harmonics::Producer {
    explicit ConstantProducer(std::vector<float> v) : vals(std::move(v)) {}
    HTensor next() override {
        std::vector<std::byte> bytes(vals.size() * sizeof(float));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return HTensor{HTensor::DType::Float32, {vals.size()}, std::move(bytes)};
    }
    std::size_t size() const override { return 1; }
    std::vector<float> vals;
};

TEST(SmallLLMTest, RunInferenceWithImportedWeights) {
    // Build minimal ONNX file with one weight tensor named "attn".
    onnx::TensorProto t;
    t.set_name("attn");
    t.set_data_type(onnx::FLOAT);
    t.add_dims(1);
    float val = 0.5f;
    t.set_raw_data(&val, sizeof(val));

    onnx::GraphProto g;
    *g.add_initializer() = t;

    onnx::ModelProto m;
    *m.mutable_graph() = g;

    const std::string path = "tiny.onnx";
    {
        std::ofstream out(path, std::ios::binary);
        if (!m.SerializeToOstream(&out))
            throw std::runtime_error("failed to write onnx");
    }

    auto weights = harmonics::import_onnx_weights(path);
    EXPECT_EQ(weights.size(), 1u);
    std::remove(path.c_str());

    const char* src = "producer in {2}; layer attn; cycle { in -(id)-> attn; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto hg = harmonics::build_graph(ast);
    auto id = std::make_shared<IdActivation>();
    harmonics::registerActivation("id", id);

    auto prod = std::make_shared<ConstantProducer>(std::vector<float>{1.f, 2.f});
    hg.bindProducer("in", prod);

    harmonics::CycleRuntime rt{hg};
    harmonics::attach_named_weights(rt, weights);

    rt.forward();

    const auto& out = rt.state().layer_tensors[0];
    EXPECT_EQ(out.shape().size(), 1u);
    EXPECT_EQ(out.shape()[0], 2u);
    const float* d = reinterpret_cast<const float*>(out.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);

    const auto& lw = harmonics::layer_weights(rt, "attn");
    EXPECT_EQ(lw.shape().size(), 1u);
    EXPECT_EQ(lw.shape()[0], 1u);
    const float* lw_d = reinterpret_cast<const float*>(lw.data().data());
    EXPECT_EQ(lw_d[0], val);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
