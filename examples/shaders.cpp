#include <harmonics/function_registry.hpp>
#include <harmonics/shaders.hpp>
#include <iostream>

int main() {
    harmonics::register_builtin_shaders();

    harmonics::HTensor t{harmonics::HTensor::DType::Float32, {3}};
    t.data().resize(3 * sizeof(float));
    auto* p = reinterpret_cast<float*>(t.data().data());
    p[0] = -1.0f;
    p[1] = 0.5f;
    p[2] = 2.0f;

    const auto& relu = harmonics::getActivation("relu");
    auto r = relu(t);
    const auto* rv = reinterpret_cast<const float*>(r.data().data());
    std::cout << "ReLU: " << rv[0] << ' ' << rv[1] << ' ' << rv[2] << '\n';

    const auto& sig = harmonics::getActivation("sigmoid");
    auto s = sig(t);
    const auto* sv = reinterpret_cast<const float*>(s.data().data());
    std::cout << "Sigmoid: " << sv[0] << ' ' << sv[1] << ' ' << sv[2] << '\n';

    const auto& sm = harmonics::getActivation("softmax");
    auto smt = sm(t);
    const auto* smv = reinterpret_cast<const float*>(smt.data().data());
    std::cout << "Softmax sum: " << smv[0] + smv[1] + smv[2] << '\n';

    const auto& mse = harmonics::getLoss("mse");
    auto mse_out = mse(t, t);
    std::cout << "MSE: " << reinterpret_cast<const float*>(mse_out.data().data())[0] << '\n';

    const auto& ce = harmonics::getLoss("cross_entropy");
    auto ce_out = ce(smt, smt);
    std::cout << "CrossEntropy: " << reinterpret_cast<const float*>(ce_out.data().data())[0]
              << '\n';
}
