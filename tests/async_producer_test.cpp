#include <cstring>
#include <gtest/gtest.h>
#include <harmonics/async_producer.hpp>
#include <vector>

using harmonics::HTensor;

struct CountingProducer : harmonics::Producer {
    HTensor next() override {
        std::vector<std::byte> data(sizeof(float));
        float value = static_cast<float>(count++);
        std::memcpy(data.data(), &value, sizeof(float));
        HTensor t{HTensor::DType::Float32, {1}, std::move(data)};
        return t;
    }
    std::size_t size() const override { return 0; }
    int count{0};
};

TEST(AsyncProducerTest, PrefetchesValues) {
    auto base = std::make_shared<CountingProducer>();
    harmonics::AsyncProducer async{base};

    auto t1 = async.next();
    auto d1 = reinterpret_cast<const float*>(t1.data().data());
    EXPECT_EQ(d1[0], 0.0f);

    auto t2 = async.next();
    auto d2 = reinterpret_cast<const float*>(t2.data().data());
    EXPECT_EQ(d2[0], 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
