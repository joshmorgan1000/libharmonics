#include <harmonics/dataset.hpp>
#include <harmonics/graph.hpp>
#include <harmonics/parser.hpp>
#include <harmonics/runtime.hpp>
#include <harmonics/shaders.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace harmonics;

namespace {

// Convert UInt8 tensors to Float32 in the range [0,1].
class FloatCastProducer : public Producer {
  public:
    explicit FloatCastProducer(std::shared_ptr<Producer> inner) : inner_{std::move(inner)} {}

    HTensor next() override {
        HTensor t = inner_->next();
        if (t.data().empty())
            return {};
        std::size_t elems = t.data().size();
        std::vector<std::byte> data(elems * sizeof(float));
        const auto* in = reinterpret_cast<const unsigned char*>(t.data().data());
        auto* out = reinterpret_cast<float*>(data.data());
        for (std::size_t i = 0; i < elems; ++i)
            out[i] = static_cast<float>(in[i]) / 255.0f;
        return {HTensor::DType::Float32, t.shape(), std::move(data)};
    }

    std::size_t size() const override { return inner_->size(); }

  private:
    std::shared_ptr<Producer> inner_{};
};

// Flatten higher rank tensors into 1-D vectors.
class FlattenProducer : public Producer {
  public:
    explicit FlattenProducer(std::shared_ptr<Producer> inner) : inner_{std::move(inner)} {}

    HTensor next() override {
        HTensor t = inner_->next();
        if (t.data().empty())
            return {};
        std::size_t elems = 1;
        for (std::size_t d : t.shape())
            elems *= d;
        HTensor out{t.dtype(), {elems}};
        out.data() = t.data();
        return out;
    }

    std::size_t size() const override { return inner_->size(); }

  private:
    std::shared_ptr<Producer> inner_{};
};

// Convert integer labels to one-hot Float32 vectors.
class OneHotProducer : public Producer {
  public:
    OneHotProducer(std::shared_ptr<Producer> inner, std::size_t classes)
        : inner_{std::move(inner)}, classes_{classes} {}

    HTensor next() override {
        HTensor t = inner_->next();
        if (t.data().empty())
            return {};
        int label = 0;
        if (t.dtype() == HTensor::DType::UInt8)
            label = static_cast<int>(reinterpret_cast<const unsigned char*>(t.data().data())[0]);
        else if (t.dtype() == HTensor::DType::Int32)
            label = reinterpret_cast<const int32_t*>(t.data().data())[0];
        std::vector<std::byte> data(classes_ * sizeof(float));
        auto* out = reinterpret_cast<float*>(data.data());
        for (std::size_t i = 0; i < classes_; ++i)
            out[i] = static_cast<float>(i == static_cast<std::size_t>(label));
        return {HTensor::DType::Float32, {classes_}, std::move(data)};
    }

    std::size_t size() const override { return inner_->size(); }

  private:
    std::shared_ptr<Producer> inner_{};
    std::size_t classes_{0};
};

} // namespace

int main() {
    register_builtin_shaders();

    const char* src = R"(
producer img {784};
producer lbl {10};
layer input 1/1 img;
layer hidden 1/2 input;
layer output 1/1 lbl;
cycle {
  img -(relu)-> input -(relu)-> hidden -(sigmoid)-> output;
  output <-(cross_entropy)- lbl;
}
)";

    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto g = build_graph(ast);

    auto images = std::make_shared<IdxProducer>("examples/mnist/train/train-images.idx3-ubyte");
    auto labels = std::make_shared<IdxProducer>("examples/mnist/train/train-labels.idx1-ubyte");

    auto img_pipeline = std::make_shared<BatchProducer>(
        std::make_shared<FlattenProducer>(std::make_shared<FloatCastProducer>(images)), 32);
    auto lbl_pipeline =
        std::make_shared<BatchProducer>(std::make_shared<OneHotProducer>(labels, 10), 32);

    g.bindProducer("img", img_pipeline);
    g.bindProducer("lbl", lbl_pipeline);

    FitOptions opt;
    opt.learning_rate = 0.01f;
    opt.progress = [](std::size_t step, float grad, float loss, float lr) {
        std::cout << "step " << step << " loss " << loss << " grad " << grad << '\n';
    };

    g.fit(5, make_auto_policy(), opt);

    std::cout << "Training finished" << std::endl;
    return 0;
}
