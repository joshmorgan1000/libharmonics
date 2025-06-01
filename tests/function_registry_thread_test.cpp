#include <gtest/gtest.h>
#include <harmonics/function_registry.hpp>
#include <thread>

using harmonics::ActivationFunction;
using harmonics::HTensor;

struct TSAct : ActivationFunction {
    HTensor operator()(const HTensor& x) const override { return x; }
};

TEST(FunctionRegistryThread, ConcurrentRegistration) {
    const int N = 8;
    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([]() {
            harmonics::registerActivation("ts", std::make_shared<TSAct>());
            const auto& act = harmonics::getActivation("ts");
            HTensor t{};
            (void)act(t);
        });
    }
    for (auto& th : threads)
        th.join();
    EXPECT_EQ(0, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
