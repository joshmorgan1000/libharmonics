#pragma once

#include "harmonics/config.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#if HARMONICS_HAS_WASM_SIMD
#include <wasm_simd128.h>
#endif
#if HARMONICS_HAS_SSE2
#include <immintrin.h>
#endif

#include "harmonics/cycle.hpp"
#include "harmonics/deployment.hpp"
#include "harmonics/graph.hpp"
#include "harmonics/int8_math.hpp"
#include "harmonics/training.hpp"

namespace harmonics {

/**
 * @brief Collection of low level training helpers.
 *
 * This header contains a variety of small functions that implement the
 * numeric heavy lifting used by the runtime. They are kept separate from
 * the higher level graph logic so that the algorithms can be reviewed in
 * isolation. Many of the helpers include specialised SIMD code paths for
 * x86 and WebAssembly which keeps execution fast even though the
 * implementations are straightforward.
 *
 * The file is intentionally verbose in its documentation because the
 * algorithms are heavily optimised. Each function provides a short
 * description followed by comments inside the more complex loops to make
 * it clear how the maths maps onto the code. New contributors are
 * encouraged to skim through these helpers to gain a deeper
 * understanding of how the runtime manipulates tensors during training.
 */

/** Clip tensor values to the range `[-limit, limit]`. */
inline void clip_tensor(HTensor& t, float limit) {
    if (limit <= 0.0f)
        return;
    switch (t.dtype()) {
    case HTensor::DType::Float32: {
        auto* d = reinterpret_cast<float*>(t.data().data());
        std::size_t n = t.data().size() / sizeof(float);
#if HARMONICS_HAS_WASM_SIMD
        v128_t vlimit = wasm_f32x4_splat(limit);
        v128_t nlimit = wasm_f32x4_neg(vlimit);
        std::size_t i = 0;
        // Clamp four floats at a time using SIMD operations. This
        // provides a significant speed up for large tensors compared
        // to clamping each element individually.
        for (; i + 4 <= n; i += 4) {
            v128_t v = wasm_v128_load(&d[i]);
            v = wasm_f32x4_max(v, nlimit);
            v = wasm_f32x4_min(v, vlimit);
            wasm_v128_store(&d[i], v);
        }
        for (; i < n; ++i) {
            if (d[i] > limit)
                d[i] = limit;
            else if (d[i] < -limit)
                d[i] = -limit;
        }
#elif HARMONICS_HAS_SSE2
        __m128 vlimit = _mm_set1_ps(limit);
        __m128 nlimit = _mm_set1_ps(-limit);
        std::size_t i = 0;
        // SSE2 path performs the same clamping but using x86 vector
        // instructions when available on the host CPU.
        for (; i + 4 <= n; i += 4) {
            __m128 v = _mm_loadu_ps(&d[i]);
            v = _mm_max_ps(v, nlimit);
            v = _mm_min_ps(v, vlimit);
            _mm_storeu_ps(&d[i], v);
        }
        for (; i < n; ++i) {
            if (d[i] > limit)
                d[i] = limit;
            else if (d[i] < -limit)
                d[i] = -limit;
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            if (d[i] > limit)
                d[i] = limit;
            else if (d[i] < -limit)
                d[i] = -limit;
        }
#endif
        break;
    }
    case HTensor::DType::Float64: {
        auto* d = reinterpret_cast<double*>(t.data().data());
        std::size_t n = t.data().size() / sizeof(double);
#if HARMONICS_HAS_SSE2
        __m128d vlimit = _mm_set1_pd(static_cast<double>(limit));
        __m128d nlimit = _mm_set1_pd(-static_cast<double>(limit));
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            __m128d v = _mm_loadu_pd(&d[i]);
            v = _mm_max_pd(v, nlimit);
            v = _mm_min_pd(v, vlimit);
            _mm_storeu_pd(&d[i], v);
        }
        for (; i < n; ++i) {
            if (d[i] > static_cast<double>(limit))
                d[i] = static_cast<double>(limit);
            else if (d[i] < -static_cast<double>(limit))
                d[i] = -static_cast<double>(limit);
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            if (d[i] > static_cast<double>(limit))
                d[i] = static_cast<double>(limit);
            else if (d[i] < -static_cast<double>(limit))
                d[i] = -static_cast<double>(limit);
        }
#endif
        break;
    }
    default:
        break;
    }
}

/** Compute the L2 norm of a tensor. */
inline float tensor_l2_norm(const HTensor& t) {
    switch (t.dtype()) {
    case HTensor::DType::Float32: {
        const auto* d = reinterpret_cast<const float*>(t.data().data());
        std::size_t n = t.data().size() / sizeof(float);
#if HARMONICS_HAS_WASM_SIMD
        float sum = 0.0f;
        v128_t acc = wasm_f32x4_splat(0.0f);
        std::size_t i = 0;
        // Accumulate partial sums of four squared elements using SIMD.
        for (; i + 4 <= n; i += 4) {
            v128_t v = wasm_v128_load(&d[i]);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(v, v));
        }
        float tmp[4];
        wasm_v128_store(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < n; ++i)
            sum += d[i] * d[i];
#elif HARMONICS_HAS_SSE2
        float sum = 0.0f;
        __m128 acc = _mm_setzero_ps();
        std::size_t i = 0;
        // SSE2 version computing four products per iteration.
        for (; i + 4 <= n; i += 4) {
            __m128 v = _mm_loadu_ps(&d[i]);
            acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
        }
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < n; ++i)
            sum += d[i] * d[i];
#else
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
            sum += d[i] * d[i];
#endif
        return std::sqrt(sum);
    }
    case HTensor::DType::Float64: {
        const auto* d = reinterpret_cast<const double*>(t.data().data());
        std::size_t n = t.data().size() / sizeof(double);
#if HARMONICS_HAS_SSE2
        __m128d acc = _mm_setzero_pd();
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            __m128d v = _mm_loadu_pd(&d[i]);
            acc = _mm_add_pd(acc, _mm_mul_pd(v, v));
        }
        alignas(16) double tmp[2];
        _mm_store_pd(tmp, acc);
        double sum = tmp[0] + tmp[1];
        for (; i < n; ++i)
            sum += d[i] * d[i];
#else
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            sum += d[i] * d[i];
#endif
        return static_cast<float>(std::sqrt(sum));
    }
    default:
        return 0.0f;
    }
}

/** Compute the combined L2 norm of a collection of gradients. */
inline float gradients_l2_norm(const std::vector<HTensor>& grads) {
    float sum = 0.0f;
    for (const auto& g : grads) {
        float n = tensor_l2_norm(g);
        sum += n * n;
    }
    return std::sqrt(sum);
}

inline void add_tensor_inplace(HTensor& dst, const HTensor& src) {
    if (dst.dtype() != src.dtype() || dst.shape() != src.shape())
        return;
    if (dst.data().empty())
        dst.data().assign(src.data().size(), std::byte{});
    switch (dst.dtype()) {
    case HTensor::DType::Float32: {
        auto* d = reinterpret_cast<float*>(dst.data().data());
        const auto* s = reinterpret_cast<const float*>(src.data().data());
        std::size_t n = dst.data().size() / sizeof(float);
        for (std::size_t i = 0; i < n; ++i)
            d[i] += s[i];
        break;
    }
    case HTensor::DType::Float64: {
        auto* d = reinterpret_cast<double*>(dst.data().data());
        const auto* s = reinterpret_cast<const double*>(src.data().data());
        std::size_t n = dst.data().size() / sizeof(double);
        for (std::size_t i = 0; i < n; ++i)
            d[i] += s[i];
        break;
    }
    default:
        break;
    }
}

inline void scale_tensor_inplace(HTensor& t, float s) {
    if (s == 1.0f)
        return;
    switch (t.dtype()) {
    case HTensor::DType::Float32: {
        auto* d = reinterpret_cast<float*>(t.data().data());
        std::size_t n = t.data().size() / sizeof(float);
        for (std::size_t i = 0; i < n; ++i)
            d[i] *= s;
        break;
    }
    case HTensor::DType::Float64: {
        auto* d = reinterpret_cast<double*>(t.data().data());
        std::size_t n = t.data().size() / sizeof(double);
        for (std::size_t i = 0; i < n; ++i)
            d[i] *= static_cast<double>(s);
        break;
    }
    default:
        break;
    }
}

inline void zero_tensor(HTensor& t) { std::fill(t.data().begin(), t.data().end(), std::byte{}); }

/** Apply a stochastic gradient descent update to a parameter tensor. */
inline void apply_sgd_update(HTensor& param, const HTensor& grad, float lr) {
    if (param.dtype() != grad.dtype() || param.shape() != grad.shape())
        return;

    if (param.data().empty())
        param.data().assign(grad.data().size(), std::byte{});

    switch (param.dtype()) {
    case HTensor::DType::Float32: {
        auto* p = reinterpret_cast<float*>(param.data().data());
        const auto* g = reinterpret_cast<const float*>(grad.data().data());
        std::size_t n = param.data().size() / sizeof(float);
#if HARMONICS_HAS_WASM_SIMD
        v128_t lrv = wasm_f32x4_splat(lr);
        std::size_t i = 0;
        // Vectorised SGD update working on four parameters per loop.
        for (; i + 4 <= n; i += 4) {
            v128_t pv = wasm_v128_load(&p[i]);
            v128_t gv = wasm_v128_load(&g[i]);
            pv = wasm_f32x4_sub(pv, wasm_f32x4_mul(lrv, gv));
            wasm_v128_store(&p[i], pv);
        }
        for (; i < n; ++i)
            p[i] -= lr * g[i];
#elif HARMONICS_HAS_SSE2
        __m128 lrv = _mm_set1_ps(lr);
        std::size_t i = 0;
        // SSE2 variant performing the same update using x86 SIMD.
        for (; i + 4 <= n; i += 4) {
            __m128 pv = _mm_loadu_ps(&p[i]);
            __m128 gv = _mm_loadu_ps(&g[i]);
            pv = _mm_sub_ps(pv, _mm_mul_ps(lrv, gv));
            _mm_storeu_ps(&p[i], pv);
        }
        for (; i < n; ++i)
            p[i] -= lr * g[i];
#else
        for (std::size_t i = 0; i < n; ++i)
            p[i] -= lr * g[i];
#endif
        break;
    }
    case HTensor::DType::Float64: {
        auto* p = reinterpret_cast<double*>(param.data().data());
        const auto* g = reinterpret_cast<const double*>(grad.data().data());
        std::size_t n = param.data().size() / sizeof(double);
#if HARMONICS_HAS_SSE2
        __m128d lrv = _mm_set1_pd(static_cast<double>(lr));
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            __m128d pv = _mm_loadu_pd(&p[i]);
            __m128d gv = _mm_loadu_pd(&g[i]);
            pv = _mm_sub_pd(pv, _mm_mul_pd(lrv, gv));
            _mm_storeu_pd(&p[i], pv);
        }
        for (; i < n; ++i)
            p[i] -= static_cast<double>(lr) * g[i];
#else
        for (std::size_t i = 0; i < n; ++i)
            p[i] -= lr * g[i];
#endif
        break;
    }
    default:
        break;
    }
}

/** Apply an integer SGD update with a right-shift learning rate. */
inline void apply_integer_sgd_update(HTensor& param, const HTensor& grad, uint8_t lr_shift) {
    if (param.dtype() != HTensor::DType::UInt8 || grad.dtype() != HTensor::DType::Int32 ||
        param.shape() != grad.shape())
        return;

    if (param.data().empty())
        param.data().assign(grad.data().size() / sizeof(int32_t), std::byte{});

    auto* p = reinterpret_cast<int8_t*>(param.data().data());
    const auto* g = reinterpret_cast<const int32_t*>(grad.data().data());
    std::size_t n = param.data().size();
    for (std::size_t i = 0; i < n; ++i) {
        int32_t delta = g[i] >> lr_shift;
        int32_t updated = static_cast<int32_t>(p[i]) - delta;
        if (updated > 127)
            updated = 127;
        else if (updated < -128)
            updated = -128;
        p[i] = static_cast<int8_t>(updated);
    }
}

/** Apply an integer SGD update using a shift schedule. */
inline void apply_integer_sgd_update(HTensor& param, const HTensor& grad,
                                     const StepDecayShiftSchedule& sched, std::size_t step) {
    apply_integer_sgd_update(param, grad, sched(step));
}

/**
 * Update a parameter using the Adam optimization algorithm.
 *
 * @param param parameter tensor to update
 * @param grad  gradient tensor for the parameter
 * @param m     first moment estimate
 * @param v     second moment estimate
 * @param t     current timestep starting at 1
 * @param lr    learning rate
 * @param beta1 decay for the first moment estimate
 * @param beta2 decay for the second moment estimate
 * @param eps   small constant to avoid division by zero
 */
inline void apply_adam_update(HTensor& param, const HTensor& grad, HTensor& m, HTensor& v,
                              std::size_t t, float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                              float eps = 1e-8f) {
    if (param.dtype() != grad.dtype() || param.shape() != grad.shape())
        return;

    if (param.data().empty())
        param.data().assign(grad.data().size(), std::byte{});
    if (m.data().empty())
        m.data().assign(grad.data().size(), std::byte{});
    if (v.data().empty())
        v.data().assign(grad.data().size(), std::byte{});

    switch (param.dtype()) {
    case HTensor::DType::Float32: {
        auto* p = reinterpret_cast<float*>(param.data().data());
        const auto* g = reinterpret_cast<const float*>(grad.data().data());
        auto* mdat = reinterpret_cast<float*>(m.data().data());
        auto* vdat = reinterpret_cast<float*>(v.data().data());
        std::size_t n = param.data().size() / sizeof(float);
        float bias1 = 1.f - std::pow(beta1, static_cast<float>(t));
        float bias2 = 1.f - std::pow(beta2, static_cast<float>(t));
#if HARMONICS_HAS_WASM_SIMD
        v128_t vbeta1 = wasm_f32x4_splat(beta1);
        v128_t vone_b1 = wasm_f32x4_splat(1.f - beta1);
        v128_t vbeta2 = wasm_f32x4_splat(beta2);
        v128_t vone_b2 = wasm_f32x4_splat(1.f - beta2);
        v128_t vlr = wasm_f32x4_splat(lr);
        v128_t vbias1 = wasm_f32x4_splat(bias1);
        v128_t vbias2 = wasm_f32x4_splat(bias2);
        v128_t veps = wasm_f32x4_splat(eps);
        std::size_t i = 0;
        // Process four parameters per iteration with WebAssembly SIMD.
        for (; i + 4 <= n; i += 4) {
            v128_t mvec = wasm_v128_load(&mdat[i]);
            v128_t gvec = wasm_v128_load(&g[i]);
            mvec = wasm_f32x4_add(wasm_f32x4_mul(vbeta1, mvec), wasm_f32x4_mul(vone_b1, gvec));
            wasm_v128_store(&mdat[i], mvec);
            v128_t vvec = wasm_v128_load(&vdat[i]);
            v128_t g2 = wasm_f32x4_mul(gvec, gvec);
            vvec = wasm_f32x4_add(wasm_f32x4_mul(vbeta2, vvec), wasm_f32x4_mul(vone_b2, g2));
            wasm_v128_store(&vdat[i], vvec);
            v128_t m_hat = wasm_f32x4_div(mvec, vbias1);
            v128_t v_hat = wasm_f32x4_div(vvec, vbias2);
            v128_t denom = wasm_f32x4_add(wasm_f32x4_sqrt(v_hat), veps);
            v128_t upd = wasm_f32x4_mul(vlr, wasm_f32x4_div(m_hat, denom));
            v128_t pvec = wasm_v128_load(&p[i]);
            pvec = wasm_f32x4_sub(pvec, upd);
            wasm_v128_store(&p[i], pvec);
        }
        // Handle any remaining elements with the scalar fallback.
        for (; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.f - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.f - beta2) * g[i] * g[i];
            float m_hat = mdat[i] / bias1;
            float v_hat = vdat[i] / bias2;
            p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
#elif HARMONICS_HAS_SSE2
        __m128 vbeta1 = _mm_set1_ps(beta1);
        __m128 vone_b1 = _mm_set1_ps(1.f - beta1);
        __m128 vbeta2 = _mm_set1_ps(beta2);
        __m128 vone_b2 = _mm_set1_ps(1.f - beta2);
        __m128 vlr = _mm_set1_ps(lr);
        __m128 vbias1 = _mm_set1_ps(bias1);
        __m128 vbias2 = _mm_set1_ps(bias2);
        __m128 veps = _mm_set1_ps(eps);
        std::size_t i = 0;
        // SSE2 implementation mirrors the WebAssembly path using x86 intrinsics.
        for (; i + 4 <= n; i += 4) {
            __m128 mvec = _mm_loadu_ps(&mdat[i]);
            __m128 gvec = _mm_loadu_ps(&g[i]);
            mvec = _mm_add_ps(_mm_mul_ps(vbeta1, mvec), _mm_mul_ps(vone_b1, gvec));
            _mm_storeu_ps(&mdat[i], mvec);
            __m128 vvec = _mm_loadu_ps(&vdat[i]);
            __m128 g2 = _mm_mul_ps(gvec, gvec);
            vvec = _mm_add_ps(_mm_mul_ps(vbeta2, vvec), _mm_mul_ps(vone_b2, g2));
            _mm_storeu_ps(&vdat[i], vvec);
            __m128 m_hat = _mm_div_ps(mvec, vbias1);
            __m128 v_hat = _mm_div_ps(vvec, vbias2);
            __m128 denom = _mm_add_ps(_mm_sqrt_ps(v_hat), veps);
            __m128 upd = _mm_mul_ps(vlr, _mm_div_ps(m_hat, denom));
            __m128 pvec = _mm_loadu_ps(&p[i]);
            pvec = _mm_sub_ps(pvec, upd);
            _mm_storeu_ps(&p[i], pvec);
        }
        for (; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.f - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.f - beta2) * g[i] * g[i];
            float m_hat = mdat[i] / bias1;
            float v_hat = vdat[i] / bias2;
            p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.f - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.f - beta2) * g[i] * g[i];
            float m_hat = mdat[i] / bias1;
            float v_hat = vdat[i] / bias2;
            p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
#endif
        break;
    }
    case HTensor::DType::Float64: {
        auto* p = reinterpret_cast<double*>(param.data().data());
        const auto* g = reinterpret_cast<const double*>(grad.data().data());
        auto* mdat = reinterpret_cast<double*>(m.data().data());
        auto* vdat = reinterpret_cast<double*>(v.data().data());
        std::size_t n = param.data().size() / sizeof(double);
        double bias1 = 1.0 - std::pow(beta1, static_cast<double>(t));
        double bias2 = 1.0 - std::pow(beta2, static_cast<double>(t));
#if HARMONICS_HAS_SSE2
        __m128d vbeta1 = _mm_set1_pd(beta1);
        __m128d vone_b1 = _mm_set1_pd(1.0 - beta1);
        __m128d vbeta2 = _mm_set1_pd(beta2);
        __m128d vone_b2 = _mm_set1_pd(1.0 - beta2);
        __m128d vlr = _mm_set1_pd(lr);
        __m128d vbias1 = _mm_set1_pd(bias1);
        __m128d vbias2 = _mm_set1_pd(bias2);
        __m128d veps = _mm_set1_pd(eps);
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            __m128d mvec = _mm_loadu_pd(&mdat[i]);
            __m128d gvec = _mm_loadu_pd(&g[i]);
            mvec = _mm_add_pd(_mm_mul_pd(vbeta1, mvec), _mm_mul_pd(vone_b1, gvec));
            _mm_storeu_pd(&mdat[i], mvec);
            __m128d vvec = _mm_loadu_pd(&vdat[i]);
            __m128d g2 = _mm_mul_pd(gvec, gvec);
            vvec = _mm_add_pd(_mm_mul_pd(vbeta2, vvec), _mm_mul_pd(vone_b2, g2));
            _mm_storeu_pd(&vdat[i], vvec);
            __m128d m_hat = _mm_div_pd(mvec, vbias1);
            __m128d v_hat = _mm_div_pd(vvec, vbias2);
            __m128d denom = _mm_add_pd(_mm_sqrt_pd(v_hat), veps);
            __m128d upd = _mm_mul_pd(vlr, _mm_div_pd(m_hat, denom));
            __m128d pvec = _mm_loadu_pd(&p[i]);
            pvec = _mm_sub_pd(pvec, upd);
            _mm_storeu_pd(&p[i], pvec);
        }
        for (; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.0 - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.0 - beta2) * g[i] * g[i];
            double m_hat = mdat[i] / bias1;
            double v_hat = vdat[i] / bias2;
            p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.0 - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.0 - beta2) * g[i] * g[i];
            double m_hat = mdat[i] / bias1;
            double v_hat = vdat[i] / bias2;
            p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
#endif
        break;
    }
    default:
        break;
    }
}

/**
 * Update a parameter using the RMSProp algorithm.
 *
 * @param param parameter tensor to update
 * @param grad  gradient tensor for the parameter
 * @param ms    moving average of squared gradients
 * @param lr    learning rate
 * @param decay decay rate for the moving average
 * @param eps   small constant to avoid division by zero
 */
inline void apply_rmsprop_update(HTensor& param, const HTensor& grad, HTensor& ms, float lr,
                                 float decay = 0.9f, float eps = 1e-8f) {
    if (param.dtype() != grad.dtype() || param.shape() != grad.shape())
        return;

    if (param.data().empty())
        param.data().assign(grad.data().size(), std::byte{});
    if (ms.data().empty())
        ms.data().assign(grad.data().size(), std::byte{});

    switch (param.dtype()) {
    case HTensor::DType::Float32: {
        auto* p = reinterpret_cast<float*>(param.data().data());
        const auto* g = reinterpret_cast<const float*>(grad.data().data());
        auto* s = reinterpret_cast<float*>(ms.data().data());
        std::size_t n = param.data().size() / sizeof(float);
#if HARMONICS_HAS_WASM_SIMD
        v128_t vdecay = wasm_f32x4_splat(decay);
        v128_t v1mdecay = wasm_f32x4_splat(1.f - decay);
        v128_t vlr = wasm_f32x4_splat(lr);
        v128_t veps = wasm_f32x4_splat(eps);
        std::size_t i = 0;
        // Use SIMD to update four parameters per iteration.
        for (; i + 4 <= n; i += 4) {
            v128_t sval = wasm_v128_load(&s[i]);
            v128_t gval = wasm_v128_load(&g[i]);
            v128_t g2 = wasm_f32x4_mul(gval, gval);
            sval = wasm_f32x4_add(wasm_f32x4_mul(vdecay, sval), wasm_f32x4_mul(v1mdecay, g2));
            wasm_v128_store(&s[i], sval);
            v128_t upd = wasm_f32x4_div(gval, wasm_f32x4_add(wasm_f32x4_sqrt(sval), veps));
            v128_t pval = wasm_v128_load(&p[i]);
            pval = wasm_f32x4_sub(pval, wasm_f32x4_mul(vlr, upd));
            wasm_v128_store(&p[i], pval);
        }
        // Finish off any remaining elements with scalar code.
        for (; i < n; ++i) {
            s[i] = decay * s[i] + (1.f - decay) * g[i] * g[i];
            p[i] -= lr * g[i] / (std::sqrt(s[i]) + eps);
        }
#elif HARMONICS_HAS_SSE2
        __m128 vdecay = _mm_set1_ps(decay);
        __m128 v1mdecay = _mm_set1_ps(1.f - decay);
        __m128 vlr = _mm_set1_ps(lr);
        __m128 veps = _mm_set1_ps(eps);
        std::size_t i = 0;
        // Equivalent SSE2 implementation for x86 targets.
        for (; i + 4 <= n; i += 4) {
            __m128 sval = _mm_loadu_ps(&s[i]);
            __m128 gval = _mm_loadu_ps(&g[i]);
            __m128 g2 = _mm_mul_ps(gval, gval);
            sval = _mm_add_ps(_mm_mul_ps(vdecay, sval), _mm_mul_ps(v1mdecay, g2));
            _mm_storeu_ps(&s[i], sval);
            __m128 upd = _mm_div_ps(gval, _mm_add_ps(_mm_sqrt_ps(sval), veps));
            __m128 pval = _mm_loadu_ps(&p[i]);
            pval = _mm_sub_ps(pval, _mm_mul_ps(vlr, upd));
            _mm_storeu_ps(&p[i], pval);
        }
        for (; i < n; ++i) {
            s[i] = decay * s[i] + (1.f - decay) * g[i] * g[i];
            p[i] -= lr * g[i] / (std::sqrt(s[i]) + eps);
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = decay * s[i] + (1.f - decay) * g[i] * g[i];
            p[i] -= lr * g[i] / (std::sqrt(s[i]) + eps);
        }
#endif
        break;
    }
    case HTensor::DType::Float64: {
        auto* p = reinterpret_cast<double*>(param.data().data());
        const auto* g = reinterpret_cast<const double*>(grad.data().data());
        auto* s = reinterpret_cast<double*>(ms.data().data());
        std::size_t n = param.data().size() / sizeof(double);
#if HARMONICS_HAS_SSE2
        __m128d vdecay = _mm_set1_pd(decay);
        __m128d v1mdecay = _mm_set1_pd(1.0 - decay);
        __m128d vlr = _mm_set1_pd(lr);
        __m128d veps = _mm_set1_pd(eps);
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            __m128d sval = _mm_loadu_pd(&s[i]);
            __m128d gval = _mm_loadu_pd(&g[i]);
            __m128d g2 = _mm_mul_pd(gval, gval);
            sval = _mm_add_pd(_mm_mul_pd(vdecay, sval), _mm_mul_pd(v1mdecay, g2));
            _mm_storeu_pd(&s[i], sval);
            __m128d upd = _mm_div_pd(gval, _mm_add_pd(_mm_sqrt_pd(sval), veps));
            __m128d pval = _mm_loadu_pd(&p[i]);
            pval = _mm_sub_pd(pval, _mm_mul_pd(vlr, upd));
            _mm_storeu_pd(&p[i], pval);
        }
        for (; i < n; ++i) {
            s[i] = decay * s[i] + (1.0 - decay) * g[i] * g[i];
            p[i] -= lr * g[i] / (std::sqrt(s[i]) + eps);
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = decay * s[i] + (1.0 - decay) * g[i] * g[i];
            p[i] -= lr * g[i] / (std::sqrt(s[i]) + eps);
        }
#endif
        break;
    }
    default:
        break;
    }
}

/**
 * Update a parameter using the AdamW optimisation algorithm.
 */
inline void apply_adamw_update(HTensor& param, const HTensor& grad, HTensor& m, HTensor& v,
                               std::size_t t, float lr, float beta1, float beta2, float eps,
                               float weight_decay) {
    if (param.dtype() != grad.dtype() || param.shape() != grad.shape())
        return;

    if (param.data().empty())
        param.data().assign(grad.data().size(), std::byte{});
    if (m.data().empty())
        m.data().assign(grad.data().size(), std::byte{});
    if (v.data().empty())
        v.data().assign(grad.data().size(), std::byte{});

    switch (param.dtype()) {
    case HTensor::DType::Float32: {
        auto* p = reinterpret_cast<float*>(param.data().data());
        const auto* g = reinterpret_cast<const float*>(grad.data().data());
        auto* mdat = reinterpret_cast<float*>(m.data().data());
        auto* vdat = reinterpret_cast<float*>(v.data().data());
        std::size_t n = param.data().size() / sizeof(float);
        float bias1 = 1.f - std::pow(beta1, static_cast<float>(t));
        float bias2 = 1.f - std::pow(beta2, static_cast<float>(t));
        for (std::size_t i = 0; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.f - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.f - beta2) * g[i] * g[i];
            float m_hat = mdat[i] / bias1;
            float v_hat = vdat[i] / bias2;
            float upd = m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i];
            p[i] -= lr * upd;
        }
        break;
    }
    case HTensor::DType::Float64: {
        auto* p = reinterpret_cast<double*>(param.data().data());
        const auto* g = reinterpret_cast<const double*>(grad.data().data());
        auto* mdat = reinterpret_cast<double*>(m.data().data());
        auto* vdat = reinterpret_cast<double*>(v.data().data());
        std::size_t n = param.data().size() / sizeof(double);
        double bias1 = 1.0 - std::pow(beta1, static_cast<double>(t));
        double bias2 = 1.0 - std::pow(beta2, static_cast<double>(t));
        for (std::size_t i = 0; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.0 - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.0 - beta2) * g[i] * g[i];
            double m_hat = mdat[i] / bias1;
            double v_hat = vdat[i] / bias2;
            double upd = m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i];
            p[i] -= lr * upd;
        }
        break;
    }
    default:
        break;
    }
}

/**
 * Update a parameter using the LAMB optimiser.
 */
inline void apply_lamb_update(HTensor& param, const HTensor& grad, HTensor& m, HTensor& v,
                              std::size_t t, float lr, float beta1, float beta2, float eps,
                              float weight_decay) {
    if (param.dtype() != grad.dtype() || param.shape() != grad.shape())
        return;

    if (param.data().empty())
        param.data().assign(grad.data().size(), std::byte{});
    if (m.data().empty())
        m.data().assign(grad.data().size(), std::byte{});
    if (v.data().empty())
        v.data().assign(grad.data().size(), std::byte{});

    auto p_norm = tensor_l2_norm(param);

    switch (param.dtype()) {
    case HTensor::DType::Float32: {
        auto* p = reinterpret_cast<float*>(param.data().data());
        const auto* g = reinterpret_cast<const float*>(grad.data().data());
        auto* mdat = reinterpret_cast<float*>(m.data().data());
        auto* vdat = reinterpret_cast<float*>(v.data().data());
        std::size_t n = param.data().size() / sizeof(float);
        float bias1 = 1.f - std::pow(beta1, static_cast<float>(t));
        float bias2 = 1.f - std::pow(beta2, static_cast<float>(t));
        float upd_norm = 0.f;
        std::vector<float> upd(n);
        for (std::size_t i = 0; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.f - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.f - beta2) * g[i] * g[i];
            float m_hat = mdat[i] / bias1;
            float v_hat = vdat[i] / bias2;
            float u = m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i];
            upd[i] = u;
            upd_norm += u * u;
        }
        upd_norm = std::sqrt(upd_norm);
        float trust = (p_norm == 0.f || upd_norm == 0.f) ? 1.f : p_norm / upd_norm;
        for (std::size_t i = 0; i < n; ++i)
            p[i] -= lr * trust * upd[i];
        break;
    }
    case HTensor::DType::Float64: {
        auto* p = reinterpret_cast<double*>(param.data().data());
        const auto* g = reinterpret_cast<const double*>(grad.data().data());
        auto* mdat = reinterpret_cast<double*>(m.data().data());
        auto* vdat = reinterpret_cast<double*>(v.data().data());
        std::size_t n = param.data().size() / sizeof(double);
        double bias1 = 1.0 - std::pow(beta1, static_cast<double>(t));
        double bias2 = 1.0 - std::pow(beta2, static_cast<double>(t));
        double upd_norm = 0.0;
        std::vector<double> upd(n);
        for (std::size_t i = 0; i < n; ++i) {
            mdat[i] = beta1 * mdat[i] + (1.0 - beta1) * g[i];
            vdat[i] = beta2 * vdat[i] + (1.0 - beta2) * g[i] * g[i];
            double m_hat = mdat[i] / bias1;
            double v_hat = vdat[i] / bias2;
            double u = m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i];
            upd[i] = u;
            upd_norm += u * u;
        }
        upd_norm = std::sqrt(upd_norm);
        double trust = (p_norm == 0.f || upd_norm == 0.0) ? 1.0 : p_norm / upd_norm;
        for (std::size_t i = 0; i < n; ++i)
            p[i] -= lr * trust * upd[i];
        break;
    }
    default:
        break;
    }
}

/**
 * Execute a single forward pass and return the resulting state.
 */
inline CycleState HarmonicGraph::inference(const DeploymentDescriptor& deploy,
                                           std::shared_ptr<PrecisionPolicy> policy) const {
    CycleRuntime rt{*this, policy ? std::move(policy) : make_hardware_policy(), deploy};
    rt.forward();
    return rt.state();
}

/**
 * Train the graph for a fixed wall-clock duration.
 */
template <class Rep, class Period>
inline CycleState HarmonicGraph::fit(std::chrono::duration<Rep, Period> duration,
                                     std::shared_ptr<PrecisionPolicy> policy, FitOptions options,
                                     const DeploymentDescriptor& deploy) const {
    CycleRuntime rt{*this, policy ? std::move(policy) : make_hardware_policy(), deploy};
    std::vector<HTensor> params(rt.state().weights.size());
    std::vector<HTensor> opt1(rt.state().weights.size());
    std::vector<HTensor> opt2(rt.state().weights.size());
    std::vector<HTensor> accum(rt.state().weights.size());
    std::size_t accum_count = 0;
    std::size_t step = 0;
    std::size_t stall = 0;
    float best_norm = std::numeric_limits<float>::max();
    auto end = std::chrono::steady_clock::now() + duration;
    auto apply_update = [&]() {
        if (accum_count == 0)
            return;
        float norm = gradients_l2_norm(accum);
        if (norm + options.early_stop_delta < best_norm) {
            best_norm = norm;
            stall = 0;
        } else {
            ++stall;
        }
        ++step;
        uint8_t lr_shift = options.lr_schedule(step - 1);
        float lr_used = options.learning_rate;
        if (options.lr_schedule_fp)
            lr_used = options.lr_schedule_fp(step - 1);
        for (std::size_t i = 0; i < params.size(); ++i)
            if (!accum[i].shape().empty()) {
                scale_tensor_inplace(accum[i], 1.0f / static_cast<float>(accum_count));
                clip_tensor(accum[i], options.grad_clip);
                switch (options.optimizer) {
                case Optimizer::SGD:
                    if (params[i].dtype() == HTensor::DType::UInt8 &&
                        accum[i].dtype() == HTensor::DType::Int32) {
                        apply_integer_sgd_update(params[i], accum[i], lr_shift);
                        lr_used = 1.0f / static_cast<float>(1u << lr_shift);
                    } else {
                        apply_sgd_update(params[i], accum[i], lr_used);
                    }
                    break;
                case Optimizer::Adam:
                    apply_adam_update(params[i], accum[i], opt1[i], opt2[i], step, lr_used);
                    break;
                case Optimizer::AdamW:
                    apply_adamw_update(params[i], accum[i], opt1[i], opt2[i], step, lr_used, 0.9f,
                                       0.999f, 1e-8f, options.weight_decay);
                    break;
                case Optimizer::LAMB:
                    apply_lamb_update(params[i], accum[i], opt1[i], opt2[i], step, lr_used, 0.9f,
                                      0.999f, 1e-8f, options.weight_decay);
                    break;
                case Optimizer::RMSProp:
                    apply_rmsprop_update(params[i], accum[i], opt1[i], lr_used);
                    break;
                }
                zero_tensor(accum[i]);
            }
        float loss = 0.0f;
        if (!rt.state().weights.empty() && !rt.state().weights[0].data().empty())
            loss = *reinterpret_cast<const float*>(rt.state().weights[0].data().data());
        if (options.progress)
            options.progress(step, norm, loss, lr_used);
        accum_count = 0;
    };
    do {
        rt.forward();
        if (hasTrainingTaps()) {
            for (std::size_t i = 0; i < accum.size(); ++i)
                if (!rt.state().weights[i].shape().empty())
                    add_tensor_inplace(accum[i], rt.state().weights[i]);
            ++accum_count;
            if (accum_count >= options.accumulate_steps)
                apply_update();
            if (options.early_stop_patience > 0 && stall >= options.early_stop_patience)
                break;
        }
    } while (std::chrono::steady_clock::now() < end);
    if (accum_count > 0)
        apply_update();
    if (hasTrainingTaps())
        rt.state().weights = std::move(params);
    return rt.state();
}

/**
 * Fit the graph for a fixed number of epochs.
 *
 * @param epochs number of forward passes to execute; runs at least once even if
 *               zero is specified
 * @param precision precision policy used for bit-width negotiation
 * @return final CycleState after training epochs
 */
inline CycleState HarmonicGraph::fit(std::size_t epochs, std::shared_ptr<PrecisionPolicy> policy,
                                     FitOptions options, const DeploymentDescriptor& deploy) const {
    CycleRuntime rt{*this, policy ? std::move(policy) : make_hardware_policy(), deploy};
    std::vector<HTensor> params(rt.state().weights.size());
    std::vector<HTensor> opt1(rt.state().weights.size());
    std::vector<HTensor> opt2(rt.state().weights.size());
    std::vector<HTensor> accum(rt.state().weights.size());
    std::size_t accum_count = 0;
    std::size_t step = 0;
    std::size_t stall = 0;
    float best_norm = std::numeric_limits<float>::max();
    std::size_t count = epochs == 0 ? 1 : epochs;
    auto apply_update = [&]() {
        if (accum_count == 0)
            return;
        float norm = gradients_l2_norm(accum);
        if (norm + options.early_stop_delta < best_norm) {
            best_norm = norm;
            stall = 0;
        } else {
            ++stall;
        }
        ++step;
        uint8_t lr_shift = options.lr_schedule(step - 1);
        float lr_used = options.learning_rate;
        if (options.lr_schedule_fp)
            lr_used = options.lr_schedule_fp(step - 1);
        for (std::size_t j = 0; j < params.size(); ++j)
            if (!accum[j].shape().empty()) {
                scale_tensor_inplace(accum[j], 1.0f / static_cast<float>(accum_count));
                clip_tensor(accum[j], options.grad_clip);
                switch (options.optimizer) {
                case Optimizer::SGD:
                    if (params[j].dtype() == HTensor::DType::UInt8 &&
                        accum[j].dtype() == HTensor::DType::Int32) {
                        apply_integer_sgd_update(params[j], accum[j], lr_shift);
                        lr_used = 1.0f / static_cast<float>(1u << lr_shift);
                    } else {
                        apply_sgd_update(params[j], accum[j], lr_used);
                    }
                    break;
                case Optimizer::Adam:
                    apply_adam_update(params[j], accum[j], opt1[j], opt2[j], step, lr_used);
                    break;
                case Optimizer::AdamW:
                    apply_adamw_update(params[j], accum[j], opt1[j], opt2[j], step, lr_used, 0.9f,
                                       0.999f, 1e-8f, options.weight_decay);
                    break;
                case Optimizer::LAMB:
                    apply_lamb_update(params[j], accum[j], opt1[j], opt2[j], step, lr_used, 0.9f,
                                      0.999f, 1e-8f, options.weight_decay);
                    break;
                case Optimizer::RMSProp:
                    apply_rmsprop_update(params[j], accum[j], opt1[j], lr_used);
                    break;
                }
                zero_tensor(accum[j]);
            }
        float loss = 0.0f;
        if (!rt.state().weights.empty() && !rt.state().weights[0].data().empty())
            loss = *reinterpret_cast<const float*>(rt.state().weights[0].data().data());
        if (options.progress)
            options.progress(step, norm, loss, lr_used);
        accum_count = 0;
    };
    for (std::size_t i = 0; i < count; ++i) {
        rt.forward();
        if (hasTrainingTaps()) {
            for (std::size_t j = 0; j < accum.size(); ++j)
                if (!rt.state().weights[j].shape().empty())
                    add_tensor_inplace(accum[j], rt.state().weights[j]);
            ++accum_count;
            if (accum_count >= options.accumulate_steps)
                apply_update();
            if (options.early_stop_patience > 0 && stall >= options.early_stop_patience)
                break; // early stopping criterion
        }
    }
    if (accum_count > 0)
        apply_update();
    if (hasTrainingTaps())
        rt.state().weights = std::move(params);
    return rt.state();
}

/**
 * Train until the supplied predicate returns true.
 */
template <class StopPredicate>
inline CycleState
HarmonicGraph::fit_until(StopPredicate stop, std::shared_ptr<PrecisionPolicy> policy,
                         FitOptions options, const DeploymentDescriptor& deploy) const {
    CycleRuntime rt{*this, policy ? std::move(policy) : make_hardware_policy(), deploy};
    std::vector<HTensor> params(rt.state().weights.size());
    std::vector<HTensor> opt1(rt.state().weights.size());
    std::vector<HTensor> opt2(rt.state().weights.size());
    std::vector<HTensor> accum(rt.state().weights.size());
    std::size_t accum_count = 0;
    std::size_t step = 0;
    std::size_t stall = 0;
    float best_norm = std::numeric_limits<float>::max();
    auto apply_update = [&]() {
        if (accum_count == 0)
            return;
        float norm = gradients_l2_norm(accum);
        if (norm + options.early_stop_delta < best_norm) {
            best_norm = norm;
            stall = 0;
        } else {
            ++stall;
        }
        ++step;
        uint8_t lr_shift = options.lr_schedule(step - 1);
        float lr_used = options.learning_rate;
        if (options.lr_schedule_fp)
            lr_used = options.lr_schedule_fp(step - 1);
        for (std::size_t i = 0; i < params.size(); ++i)
            if (!accum[i].shape().empty()) {
                scale_tensor_inplace(accum[i], 1.0f / static_cast<float>(accum_count));
                clip_tensor(accum[i], options.grad_clip);
                switch (options.optimizer) {
                case Optimizer::SGD:
                    if (params[i].dtype() == HTensor::DType::UInt8 &&
                        accum[i].dtype() == HTensor::DType::Int32) {
                        apply_integer_sgd_update(params[i], accum[i], lr_shift);
                        lr_used = 1.0f / static_cast<float>(1u << lr_shift);
                    } else {
                        apply_sgd_update(params[i], accum[i], lr_used);
                    }
                    break;
                case Optimizer::Adam:
                    apply_adam_update(params[i], accum[i], opt1[i], opt2[i], step, lr_used);
                    break;
                case Optimizer::AdamW:
                    apply_adamw_update(params[i], accum[i], opt1[i], opt2[i], step, lr_used, 0.9f,
                                       0.999f, 1e-8f, options.weight_decay);
                    break;
                case Optimizer::LAMB:
                    apply_lamb_update(params[i], accum[i], opt1[i], opt2[i], step, lr_used, 0.9f,
                                      0.999f, 1e-8f, options.weight_decay);
                    break;
                case Optimizer::RMSProp:
                    apply_rmsprop_update(params[i], accum[i], opt1[i], lr_used);
                    break;
                }
                zero_tensor(accum[i]);
            }
        float loss = 0.0f;
        if (!rt.state().weights.empty() && !rt.state().weights[0].data().empty())
            loss = *reinterpret_cast<const float*>(rt.state().weights[0].data().data());
        if (options.progress)
            options.progress(step, norm, loss, lr_used);
        accum_count = 0;
    };
    do {
        rt.forward();
        if (hasTrainingTaps()) {
            for (std::size_t i = 0; i < accum.size(); ++i)
                if (!rt.state().weights[i].shape().empty())
                    add_tensor_inplace(accum[i], rt.state().weights[i]);
            ++accum_count;
            if (accum_count >= options.accumulate_steps)
                apply_update();
            if (options.early_stop_patience > 0 && stall >= options.early_stop_patience)
                break; // early stopping criterion
        }
    } while (!stop(rt.state()));
    if (accum_count > 0)
        apply_update();
    if (hasTrainingTaps())
        rt.state().weights = std::move(params);
    return rt.state();
}

} // namespace harmonics
