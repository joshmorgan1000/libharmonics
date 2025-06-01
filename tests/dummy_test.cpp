#include <gtest/gtest.h>
#include <harmonics/core.hpp>
#include <harmonics/version.hpp>

TEST(VersionTest, ReturnsOne) { EXPECT_EQ(harmonics::version(), 1); }

struct IdentityActivation : harmonics::ActivationFunction {
    harmonics::HTensor operator()(const harmonics::HTensor& x) const override { return x; }
};

struct ZeroLoss : harmonics::LossFunction {
    harmonics::HTensor operator()(const harmonics::HTensor&,
                                  const harmonics::HTensor&) const override {
        return harmonics::HTensor{};
    }
};

struct DummyProducer : harmonics::Producer {
    harmonics::HTensor next() override { return harmonics::HTensor{}; }
    std::size_t size() const override { return 0; }
};

struct DummyConsumer : harmonics::Consumer {
    void push(const harmonics::HTensor&) override {}
};

TEST(CoreTypesTest, InterfacesCompile) {
    IdentityActivation act;
    ZeroLoss loss;
    DummyProducer prod;
    DummyConsumer cons;

    auto t = prod.next();
    cons.push(act(t));
    cons.push(loss(t, t));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
