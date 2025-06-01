#include <cstring>
#include <gtest/gtest.h>
#include <harmonics/augmentation.hpp>
#include <memory>

using harmonics::HTensor;

struct FixedTensorProducer : harmonics::Producer {
    explicit FixedTensorProducer(const HTensor& t) : tensor{t} {}
    HTensor next() override { return tensor; }
    std::size_t size() const override { return 1; }
    HTensor tensor;
};

struct CountingProducer : harmonics::Producer {
    explicit CountingProducer(int n) : limit{n} {}
    HTensor next() override {
        if (index >= limit)
            return {};
        float v = static_cast<float>(index++);
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &v, sizeof(float));
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}, std::move(data)};
    }
    std::size_t size() const override { return limit; }
    int limit;
    int index{0};
};

static HTensor make_tensor() {
    HTensor t{HTensor::DType::Float32, {2, 2}};
    t.data().resize(sizeof(float) * 4);
    float vals[4] = {1.f, 2.f, 3.f, 4.f};
    std::memcpy(t.data().data(), vals, sizeof(vals));
    return t;
}

TEST(AugmentationTest, FlipHorizontal) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_flip_horizontal(base);
    auto t = aug->next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(d[0], 2.f);
    EXPECT_EQ(d[1], 1.f);
    EXPECT_EQ(d[2], 4.f);
    EXPECT_EQ(d[3], 3.f);
}

TEST(AugmentationTest, FlipVertical) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_flip_vertical(base);
    auto t = aug->next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(d[0], 3.f);
    EXPECT_EQ(d[1], 4.f);
    EXPECT_EQ(d[2], 1.f);
    EXPECT_EQ(d[3], 2.f);
}

TEST(AugmentationTest, Rotate90) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_rotate90(base);
    auto t = aug->next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 2u);
    EXPECT_EQ(d[0], 3.f);
    EXPECT_EQ(d[1], 1.f);
    EXPECT_EQ(d[2], 4.f);
    EXPECT_EQ(d[3], 2.f);
}

TEST(AugmentationTest, Rotate180) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_rotate(base, 180.f);
    auto t = aug->next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 2u);
    EXPECT_EQ(d[0], 4.f);
    EXPECT_EQ(d[1], 3.f);
    EXPECT_EQ(d[2], 2.f);
    EXPECT_EQ(d[3], 1.f);
}

TEST(AugmentationTest, AddNoiseZeroStddev) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_add_noise(base, 0.f);
    auto t = aug->next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    EXPECT_EQ(d[2], 3.f);
    EXPECT_EQ(d[3], 4.f);
}

TEST(AugmentationTest, RandomCropNoChange) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_random_crop(base, 2, 2);
    auto t = aug->next();
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 2u);
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    EXPECT_EQ(d[2], 3.f);
    EXPECT_EQ(d[3], 4.f);
}

TEST(AugmentationTest, ColourJitterZero) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_colour_jitter(base, 0.f, 0.f);
    auto t = aug->next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    EXPECT_EQ(d[2], 3.f);
    EXPECT_EQ(d[3], 4.f);
}

TEST(AugmentationTest, RandomRotationZeroDegrees) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_random_rotation(base, 0.f);
    auto t = aug->next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 2u);
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    EXPECT_EQ(d[2], 3.f);
    EXPECT_EQ(d[3], 4.f);
}

TEST(AugmentationTest, ScaleFactorOne) {
    auto base = std::make_shared<FixedTensorProducer>(make_tensor());
    auto aug = harmonics::make_scale(base, 1.f);
    auto t = aug->next();
    const float* d = reinterpret_cast<const float*>(t.data().data());
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 2u);
    EXPECT_EQ(d[0], 1.f);
    EXPECT_EQ(d[1], 2.f);
    EXPECT_EQ(d[2], 3.f);
    EXPECT_EQ(d[3], 4.f);
}

TEST(AugmentationTest, PipelineCachesResults) {
    auto base = std::make_shared<CountingProducer>(2);
    std::vector<harmonics::AugmentationPipeline::Fn> steps;
    steps.push_back([](const HTensor& t) {
        float v = *reinterpret_cast<const float*>(t.data().data());
        float out = v + 1.f;
        std::vector<std::byte> data(sizeof(float));
        std::memcpy(data.data(), &out, sizeof(float));
        return HTensor{HTensor::DType::Float32, {1}, std::move(data)};
    });
    auto pipe = harmonics::make_augmentation_pipeline(base, steps, true);

    auto a = pipe->next();
    const float* da = reinterpret_cast<const float*>(a.data().data());
    EXPECT_EQ(da[0], 1.f);
    EXPECT_EQ(base->index, 1);

    auto b = pipe->next();
    const float* db = reinterpret_cast<const float*>(b.data().data());
    EXPECT_EQ(db[0], 2.f);
    EXPECT_EQ(base->index, 2);

    auto c = pipe->next();
    const float* dc = reinterpret_cast<const float*>(c.data().data());
    EXPECT_EQ(dc[0], 1.f);
    EXPECT_EQ(base->index, 2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
