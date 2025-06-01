#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/introspection.hpp>
#include <harmonics/model_import.hpp>
#include <harmonics/parser.hpp>
#include <onnx/onnx_pb.h>

TEST(ModelImportTest, ImportOnnxWeights) {
    onnx::TensorProto t;
    t.set_name("w");
    t.set_data_type(onnx::FLOAT);
    t.add_dims(2);
    t.add_dims(2);
    float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    t.set_raw_data(values, sizeof(values));

    onnx::GraphProto g;
    *g.add_initializer() = t;

    onnx::ModelProto m;
    *m.mutable_graph() = g;

    const std::string path = "test.onnx";
    {
        std::ofstream out(path, std::ios::binary);
        bool ok = m.SerializeToOstream(&out);
        EXPECT_EQ(ok, true);
    }

    auto weights = harmonics::import_onnx_weights(path);
    EXPECT_EQ(weights.size(), 1u);
    EXPECT_EQ(weights[0].first, "w");
    const auto& tensor = weights[0].second;
    EXPECT_EQ(tensor.shape(), (harmonics::HTensor::Shape{2, 2}));
    const float* d = reinterpret_cast<const float*>(tensor.data().data());
    EXPECT_EQ(d[0], 1.0f);
    EXPECT_EQ(d[3], 4.0f);

    std::remove(path.c_str());
}

TEST(ModelImportTest, ImportOnnxTypedData) {
    onnx::TensorProto t;
    t.set_name("v");
    t.set_data_type(onnx::INT32);
    t.add_dims(3);
    t.add_int32_data(10);
    t.add_int32_data(20);
    t.add_int32_data(30);

    onnx::GraphProto g;
    *g.add_initializer() = t;

    onnx::ModelProto m;
    *m.mutable_graph() = g;

    const std::string path = "typed.onnx";
    {
        std::ofstream out(path, std::ios::binary);
        bool ok = m.SerializeToOstream(&out);
        EXPECT_EQ(ok, true);
    }

    auto weights = harmonics::import_onnx_weights(path);
    EXPECT_EQ(weights.size(), 1u);
    EXPECT_EQ(weights[0].first, "v");
    const auto& tensor = weights[0].second;
    EXPECT_EQ(tensor.dtype(), harmonics::HTensor::DType::Int32);
    EXPECT_EQ(tensor.shape(), (harmonics::HTensor::Shape{3}));
    const std::int32_t* d = reinterpret_cast<const std::int32_t*>(tensor.data().data());
    EXPECT_EQ(d[0], 10);
    EXPECT_EQ(d[2], 30);

    std::remove(path.c_str());
}

TEST(ModelImportTest, ImportOnnxMultipleWeights) {
    onnx::TensorProto t1;
    t1.set_name("a");
    t1.set_data_type(onnx::FLOAT);
    t1.add_dims(1);
    float fv = 1.0f;
    t1.set_raw_data(&fv, sizeof(fv));

    onnx::TensorProto t2;
    t2.set_name("b");
    t2.set_data_type(onnx::INT64);
    t2.add_dims(1);
    t2.add_int64_data(42);

    onnx::GraphProto g;
    *g.add_initializer() = t1;
    *g.add_initializer() = t2;

    onnx::ModelProto m;
    *m.mutable_graph() = g;

    const std::string path = "multi.onnx";
    {
        std::ofstream out(path, std::ios::binary);
        m.SerializeToOstream(&out);
    }

    auto weights = harmonics::import_onnx_weights(path);
    EXPECT_EQ(weights.size(), 2u);
    EXPECT_EQ(weights[0].first, "a");
    EXPECT_EQ(weights[1].first, "b");
    std::remove(path.c_str());
}

TEST(ModelImportTest, ImportOnnxBool) {
    onnx::TensorProto t;
    t.set_name("flag");
    t.set_data_type(onnx::BOOL);
    t.add_dims(2);
    t.add_int32_data(1);
    t.add_int32_data(0);

    onnx::GraphProto g;
    *g.add_initializer() = t;

    onnx::ModelProto m;
    *m.mutable_graph() = g;

    const std::string path = "bool.onnx";
    {
        std::ofstream out(path, std::ios::binary);
        m.SerializeToOstream(&out);
    }

    auto weights = harmonics::import_onnx_weights(path);
    EXPECT_EQ(weights.size(), 1u);
    const auto& tensor = weights[0].second;
    EXPECT_EQ(tensor.dtype(), harmonics::HTensor::DType::UInt8);
    EXPECT_EQ(tensor.shape(), (harmonics::HTensor::Shape{2}));
    EXPECT_EQ(static_cast<int>(tensor.data()[0]), 1);
    EXPECT_EQ(static_cast<int>(tensor.data()[1]), 0);

    std::remove(path.c_str());
}

TEST(ModelImportTest, ImportOnnxMissingFile) {
    bool threw = false;
    try {
        harmonics::import_onnx_weights("missing.onnx");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_EQ(threw, true);
}

TEST(ModelImportTest, TensorFlowImportUnsupportedThrows) {
    bool threw = false;
    try {
        harmonics::import_tensorflow_weights("dummy");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_EQ(threw, true);
}

TEST(ModelImportTest, PyTorchImportUnsupportedThrows) {
    bool threw = false;
    try {
        harmonics::import_pytorch_weights("dummy");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_EQ(threw, true);
}

TEST(ModelImportTest, AttachNamedWeights) {
    const char* src = "producer p {1}; layer l; cycle { p -> l; }";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);
    harmonics::CycleRuntime rt{g};
    harmonics::HTensor w{harmonics::HTensor::DType::Float32, {1}};
    w.data().resize(sizeof(float));
    reinterpret_cast<float*>(w.data().data())[0] = 42.0f;
    harmonics::NamedTensor n{"l", w};
    harmonics::attach_named_weights(rt, {n});
    const auto& lw = harmonics::layer_weights(rt, "l");
    EXPECT_EQ(lw.shape().size(), 1u);
    const float* d = reinterpret_cast<const float*>(lw.data().data());
    EXPECT_EQ(d[0], 42.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
