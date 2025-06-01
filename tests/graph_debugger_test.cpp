#include <gtest/gtest.h>
#include <harmonics/cycle.hpp>
#include <harmonics/dot_export.hpp>
#include <harmonics/function_registry.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>

struct FixedProducer : harmonics::Producer {
    explicit FixedProducer(int s) : shape{static_cast<std::size_t>(s)} {}
    harmonics::HTensor next() override {
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, shape};
    }
    std::size_t size() const override { return 1; }
    harmonics::HTensor::Shape shape{};
};

struct DummyLoss : harmonics::LossFunction {
    harmonics::HTensor operator()(const harmonics::HTensor&,
                                  const harmonics::HTensor&) const override {
        return harmonics::HTensor{harmonics::HTensor::DType::Float32, {1}};
    }
};

TEST(GraphDebuggerTest, CallbackInvokedForFlows) {
    const char* src = R"(
producer p {1};
producer lbl {1};
consumer c {1};
layer l;
cycle {
  p -> l;
  l -> c;
  l <-(dummy)- lbl;
})";
    harmonics::Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = harmonics::build_graph(ast);

    auto prod = std::make_shared<FixedProducer>(1);
    auto lbl = std::make_shared<FixedProducer>(1);
    g.bindProducer("p", prod);
    g.bindProducer("lbl", lbl);

    harmonics::registerLoss("dummy", std::make_shared<DummyLoss>());

    harmonics::CycleRuntime rt{g};
    std::vector<std::string> events;
    rt.set_debug_callback([&](harmonics::NodeId src, harmonics::NodeId dst,
                              const harmonics::HTensor&, bool backward,
                              const std::optional<std::string>&) {
        events.push_back(harmonics::node_name(g, src) + (backward ? " ~> " : " -> ") +
                         harmonics::node_name(g, dst));
    });

    rt.forward();

    ASSERT_EQ(events.size(), 3u);
    EXPECT_EQ(events[0], "p -> l");
    EXPECT_EQ(events[1], "l -> c");
    EXPECT_EQ(events[2], "l ~> lbl");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
