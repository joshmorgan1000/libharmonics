#include <harmonics/runtime.hpp>
#include <iostream>

int main() {
    using namespace harmonics;
    HTensor param{HTensor::DType::Float32, {3}};
    param.data().resize(3 * sizeof(float));
    HTensor grad{HTensor::DType::Float32, {3}};
    grad.data().resize(3 * sizeof(float));
    float* p = reinterpret_cast<float*>(param.data().data());
    float* g = reinterpret_cast<float*>(grad.data().data());
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    g[0] = 0.1f;
    g[1] = 0.1f;
    g[2] = 0.1f;

    HTensor m{HTensor::DType::Float32, {3}};
    HTensor v{HTensor::DType::Float32, {3}};
    HTensor s{HTensor::DType::Float32, {3}};

    HTensor ps = param;
    apply_sgd_update(ps, grad, 0.01f);
    std::cout << "SGD first value: " << reinterpret_cast<float*>(ps.data().data())[0] << '\n';

    HTensor pa = param;
    HTensor ma = m;
    HTensor va = v;
    apply_adam_update(pa, grad, ma, va, 1, 0.01f);
    std::cout << "Adam first value: " << reinterpret_cast<float*>(pa.data().data())[0] << '\n';

    HTensor pr = param;
    HTensor sr = s;
    apply_rmsprop_update(pr, grad, sr, 0.01f);
    std::cout << "RMSProp first value: " << reinterpret_cast<float*>(pr.data().data())[0] << '\n';

    return 0;
}
