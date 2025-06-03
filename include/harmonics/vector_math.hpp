#pragma once

/**
 * @file vector_math.hpp
 * @brief Small collection of vector utility functions.
 *
 * These helpers implement a couple of trivial mathematical operations
 * that are used throughout the examples. They intentionally avoid any
 * heavy dependencies and operate purely on standard library containers.
 *
 * The functions are header only and templated so they can be used with
 * both float and double precision vectors. Each operation follows a
 * consistent pattern which keeps the implementations tiny and easy to
 * understand.
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include "harmonics/config.hpp"
#if HARMONICS_HAS_WASM_SIMD
#include <wasm_simd128.h>
#endif
#if HARMONICS_HAS_SSE2
#include <immintrin.h>
#endif

namespace harmonics {

/** Simple alias used throughout the helpers. */
template <typename T> using Vector = std::vector<T>;

/**
 * @brief Add two vectors element by element.
 *
 * The result has the same length as the input vectors. If the sizes differ
 * the extra elements of the longer vector are ignored. The function is
 * intentionally minimal and performs no additional checking.
 *
 * @param a first input vector
 * @param b second input vector
 * @return Vector containing the element wise sum
 */
template <typename T> inline Vector<T> vector_add(const Vector<T>& a, const Vector<T>& b) {
    std::size_t n = std::min(a.size(), b.size());
    Vector<T> out(n);
    if constexpr (std::is_same_v<T, float>) {
#if HARMONICS_HAS_WASM_SIMD
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            v128_t va = wasm_v128_load(&a[i]);
            v128_t vb = wasm_v128_load(&b[i]);
            wasm_v128_store(&out[i], wasm_f32x4_add(va, vb));
        }
        for (; i < n; ++i)
            out[i] = a[i] + b[i];
#elif HARMONICS_HAS_SSE2
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(&a[i]);
            __m128 vb = _mm_loadu_ps(&b[i]);
            _mm_storeu_ps(&out[i], _mm_add_ps(va, vb));
        }
        for (; i < n; ++i)
            out[i] = a[i] + b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] + b[i];
#endif
    } else if constexpr (std::is_same_v<T, double>) {
#if HARMONICS_HAS_SSE2
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            __m128d va = _mm_loadu_pd(&a[i]);
            __m128d vb = _mm_loadu_pd(&b[i]);
            _mm_storeu_pd(&out[i], _mm_add_pd(va, vb));
        }
        for (; i < n; ++i)
            out[i] = a[i] + b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] + b[i];
#endif
    } else {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] + b[i];
    }
    return out;
}

/**
 * @brief Subtract vector @p b from vector @p a.
 *
 * Just like @ref vector_add this helper truncates to the length of the
 * shorter input. The difference is computed element by element.
 *
 * @param a first input vector
 * @param b second input vector
 * @return Vector containing a - b
 */
template <typename T> inline Vector<T> vector_subtract(const Vector<T>& a, const Vector<T>& b) {
    std::size_t n = std::min(a.size(), b.size());
    Vector<T> out(n);
    if constexpr (std::is_same_v<T, float>) {
#if HARMONICS_HAS_WASM_SIMD
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            v128_t va = wasm_v128_load(&a[i]);
            v128_t vb = wasm_v128_load(&b[i]);
            wasm_v128_store(&out[i], wasm_f32x4_sub(va, vb));
        }
        for (; i < n; ++i)
            out[i] = a[i] - b[i];
#elif HARMONICS_HAS_SSE2
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(&a[i]);
            __m128 vb = _mm_loadu_ps(&b[i]);
            _mm_storeu_ps(&out[i], _mm_sub_ps(va, vb));
        }
        for (; i < n; ++i)
            out[i] = a[i] - b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] - b[i];
#endif
    } else if constexpr (std::is_same_v<T, double>) {
#if HARMONICS_HAS_SSE2
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            __m128d va = _mm_loadu_pd(&a[i]);
            __m128d vb = _mm_loadu_pd(&b[i]);
            _mm_storeu_pd(&out[i], _mm_sub_pd(va, vb));
        }
        for (; i < n; ++i)
            out[i] = a[i] - b[i];
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] - b[i];
#endif
    } else {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] - b[i];
    }
    return out;
}

/**
 * @brief Multiply each element of @p v by a scalar.
 *
 * @param v vector to scale
 * @param s scalar factor applied to every element
 * @return Scaled vector
 */
template <typename T, typename S> inline Vector<T> vector_scale(const Vector<T>& v, S s) {
    Vector<T> out(v.size());
    if constexpr (std::is_same_v<T, float> && std::is_floating_point_v<S>) {
#if HARMONICS_HAS_WASM_SIMD
        v128_t vs = wasm_f32x4_splat(static_cast<float>(s));
        std::size_t i = 0;
        for (; i + 4 <= v.size(); i += 4) {
            v128_t va = wasm_v128_load(&v[i]);
            wasm_v128_store(&out[i], wasm_f32x4_mul(va, vs));
        }
        for (; i < v.size(); ++i)
            out[i] = static_cast<float>(v[i] * s);
#elif HARMONICS_HAS_SSE2
        __m128 vs = _mm_set1_ps(static_cast<float>(s));
        std::size_t i = 0;
        for (; i + 4 <= v.size(); i += 4) {
            __m128 va = _mm_loadu_ps(&v[i]);
            _mm_storeu_ps(&out[i], _mm_mul_ps(va, vs));
        }
        for (; i < v.size(); ++i)
            out[i] = static_cast<float>(v[i] * s);
#else
        for (std::size_t i = 0; i < v.size(); ++i)
            out[i] = static_cast<T>(v[i] * s);
#endif
    } else if constexpr (std::is_same_v<T, double> && std::is_floating_point_v<S>) {
#if HARMONICS_HAS_SSE2
        __m128d vs = _mm_set1_pd(static_cast<double>(s));
        std::size_t i = 0;
        for (; i + 2 <= v.size(); i += 2) {
            __m128d va = _mm_loadu_pd(&v[i]);
            _mm_storeu_pd(&out[i], _mm_mul_pd(va, vs));
        }
        for (; i < v.size(); ++i)
            out[i] = static_cast<double>(v[i] * s);
#else
        for (std::size_t i = 0; i < v.size(); ++i)
            out[i] = static_cast<T>(v[i] * s);
#endif
    } else {
        for (std::size_t i = 0; i < v.size(); ++i)
            out[i] = static_cast<T>(v[i] * s);
    }
    return out;
}

/**
 * @brief Compute the dot product of two vectors.
 *
 * Both vectors are truncated to the length of the shorter one. The
 * computation is performed using a simple loop which keeps the
 * implementation completely portable.
 *
 * @param a first vector
 * @param b second vector
 * @return Sum of element wise products
 */
template <typename T> inline T dot_product(const Vector<T>& a, const Vector<T>& b) {
    std::size_t n = std::min(a.size(), b.size());
    if constexpr (std::is_same_v<T, float>) {
        float result = 0.0f;
#if HARMONICS_HAS_WASM_SIMD
        v128_t acc = wasm_f32x4_splat(0.0f);
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            v128_t va = wasm_v128_load(&a[i]);
            v128_t vb = wasm_v128_load(&b[i]);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(va, vb));
        }
        float tmp[4];
        wasm_v128_store(tmp, acc);
        result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < n; ++i)
            result += a[i] * b[i];
        return result;
#elif HARMONICS_HAS_SSE2
        __m128 acc = _mm_setzero_ps();
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(&a[i]);
            __m128 vb = _mm_loadu_ps(&b[i]);
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, acc);
        result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < n; ++i)
            result += a[i] * b[i];
        return result;
#else
        float result2 = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
            result2 += a[i] * b[i];
        return result2;
#endif
    } else if constexpr (std::is_same_v<T, double>) {
#if HARMONICS_HAS_SSE2
        __m128d acc = _mm_setzero_pd();
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2) {
            __m128d va = _mm_loadu_pd(&a[i]);
            __m128d vb = _mm_loadu_pd(&b[i]);
            acc = _mm_add_pd(acc, _mm_mul_pd(va, vb));
        }
        alignas(16) double tmp[2];
        _mm_store_pd(tmp, acc);
        double result = tmp[0] + tmp[1];
        for (; i < n; ++i)
            result += a[i] * b[i];
        return static_cast<T>(result);
#else
        double result = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            result += a[i] * b[i];
        return static_cast<T>(result);
#endif
    } else {
        T result = T{};
        for (std::size_t i = 0; i < n; ++i)
            result += a[i] * b[i];
        return result;
    }
}

/**
 * @brief Compute the cross product of two 3D vectors.
 *
 * The input vectors must have at least three elements. Any additional
 * components are ignored which mirrors the behaviour found in some
 * linear algebra libraries. The return vector always has length three.
 *
 * @param a first 3D vector
 * @param b second 3D vector
 * @return 3 element vector representing the cross product
 */
template <typename T> inline Vector<T> cross_product(const Vector<T>& a, const Vector<T>& b) {
    Vector<T> out(3);
    if (a.size() < 3 || b.size() < 3)
        return out; // Degenerate case returns zero vector
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
    return out;
}

/**
 * @brief Calculate the Euclidean magnitude of a vector.
 *
 * The result is the square root of the dot product with itself. A
 * small epsilon may optionally be added to avoid returning zero for
 * near zero vectors but this function keeps things simple and performs
 * no such adjustment.
 *
 * @param v input vector
 * @return Vector length measured using the 2-norm
 */
template <typename T> inline T magnitude(const Vector<T>& v) {
    return std::sqrt(dot_product(v, v));
}

/**
 * @brief Normalise a vector so that it has unit length.
 *
 * If the input has zero magnitude the result will be a copy of the
 * original vector to avoid division by zero. The function operates in
 * place to keep memory allocations to a minimum.
 *
 * @param v vector to normalise
 * @return Normalised copy of @p v
 */
template <typename T> inline Vector<T> normalize(const Vector<T>& v) {
    T mag = magnitude(v);
    if (mag == T{})
        return v; // Return unchanged if magnitude is zero
    return vector_scale(v, T{1} / mag);
}

/**
 * @brief Compute the element wise product of two vectors.
 *
 * This is sometimes referred to as the Hadamard product. Just like the
 * other pair wise operations, the result has the length of the shorter
 * input vector.
 *
 * @param a first vector
 * @param b second vector
 * @return Element wise product
 */
template <typename T>
inline Vector<T> elementwise_multiply(const Vector<T>& a, const Vector<T>& b) {
    std::size_t n = std::min(a.size(), b.size());
    Vector<T> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] * b[i];
    return out;
}

/**
 * @brief Sum all elements in the vector.
 *
 * This helper simply loops over the array accumulating each value.
 * The STL provides std::accumulate which could achieve the same
 * result, however writing the loop explicitly keeps dependencies
 * obvious for those unfamiliar with the algorithm header.
 *
 * @param v vector to sum
 * @return Total of all elements
 */
template <typename T> inline T vector_sum(const Vector<T>& v) {
    T out = T{};
    for (const auto& e : v)
        out += e;
    return out;
}

/**
 * @brief Compute the arithmetic mean of a vector.
 *
 * When the vector is empty the returned value defaults to zero. The
 * operation relies on @ref vector_sum to do the heavy lifting.
 *
 * @param v input vector
 * @return Average value of all elements
 */
template <typename T> inline T vector_mean(const Vector<T>& v) {
    if (v.empty())
        return T{};
    return vector_sum(v) / static_cast<T>(v.size());
}

/**
 * @brief Find the largest element in the vector.
 *
 * The standard library includes std::max_element but using an explicit
 * loop keeps this helper consistent with the rest of the file. If the
 * vector is empty the returned value is default constructed.
 *
 * @param v vector to search
 * @return Maximum element found
 */
template <typename T> inline T vector_max(const Vector<T>& v) {
    if (v.empty())
        return T{};
    T best = v[0];
    for (const auto& e : v)
        if (e > best)
            best = e;
    return best;
}

/**
 * @brief Find the smallest element in the vector.
 *
 * Mirrors the behaviour of @ref vector_max but for the minimum value.
 * Again the function gracefully handles empty vectors by returning a
 * default constructed value.
 *
 * @param v vector to search
 * @return Smallest element found
 */
template <typename T> inline T vector_min(const Vector<T>& v) {
    if (v.empty())
        return T{};
    T best = v[0];
    for (const auto& e : v)
        if (e < best)
            best = e;
    return best;
}

/**
 * @brief Clamp all elements in the vector to the given range.
 *
 * Values less than @p lo are replaced with @p lo, those greater than
 * @p hi are replaced with @p hi. The comparison uses the normal less
 * than and greater than operators which means it works for any type
 * that supports them, not just numeric values.
 *
 * @param v vector to clamp
 * @param lo lower bound
 * @param hi upper bound
 * @return Clamped vector
 */
template <typename T> inline Vector<T> clamp_elements(const Vector<T>& v, T lo, T hi) {
    Vector<T> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        T val = v[i];
        if (val < lo)
            val = lo; // below range
        if (val > hi)
            val = hi; // above range
        out[i] = val;
    }
    return out;
}

/**
 * @brief Calculate the angle between two vectors in radians.
 *
 * This function first normalises both vectors and then computes the
 * arc cosine of their dot product. The result is constrained to the
 * range [0, pi] which covers all possible orientations.
 *
 * @param a first vector
 * @param b second vector
 * @return angle in radians between @p a and @p b
 */
template <typename T> inline T angle_between(const Vector<T>& a, const Vector<T>& b) {
    Vector<T> na = normalize(a);
    Vector<T> nb = normalize(b);
    T d = dot_product(na, nb);
    // Clamp the dot product so acos does not produce NaN due to
    // slight numerical issues when the vectors are nearly parallel.
    if (d > T{1})
        d = T{1};
    if (d < T{-1})
        d = T{-1};
    return std::acos(d);
}

/**
 * @brief Generate a linearly spaced vector between two values.
 *
 * The resulting vector includes both end points and contains
 * @p count elements. This helper is commonly used in the examples
 * to generate simple test data without relying on external tools.
 *
 * @param start first value of the sequence
 * @param end last value of the sequence
 * @param count number of elements to produce
 * @return Vector of evenly spaced values
 */
template <typename T> inline Vector<T> linspace(T start, T end, std::size_t count) {
    Vector<T> out(count);
    if (count == 0)
        return out;
    if (count == 1) {
        out[0] = start;
        return out;
    }
    T step = (end - start) / static_cast<T>(count - 1);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = start + static_cast<T>(i) * step;
    return out;
}

/**
 * @brief Concatenate two vectors together.
 *
 * Elements of @p b are appended to a copy of @p a. The operation does
 * not modify the inputs which allows the caller to reuse them if
 * desired. When the inputs are large this helper will allocate a
 * new vector with enough capacity to store the result.
 *
 * @param a first vector
 * @param b second vector
 * @return New vector containing elements of @p a followed by @p b
 */
template <typename T> inline Vector<T> concatenate(const Vector<T>& a, const Vector<T>& b) {
    Vector<T> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end()); // copy a
    out.insert(out.end(), b.begin(), b.end()); // copy b
    return out;
}

/**
 * @brief Reverse the order of elements in a vector.
 *
 * A copy of @p v is created with all elements reversed so the input
 * remains unchanged. The standard library provides std::reverse for
 * in-place modifications, however returning a new vector is often more
 * convenient when composing operations.
 *
 * @param v vector to reverse
 * @return Reversed copy of @p v
 */
template <typename T> inline Vector<T> reverse(const Vector<T>& v) {
    Vector<T> out(v.rbegin(), v.rend());
    return out;
}

/**
 * @brief Compute a prefix sum of the elements.
 *
 * Each element of the returned vector is the sum of all previous elements
 * including itself. This is also known as a scan operation and can be
 * useful when converting counts to offsets.
 *
 * @param v input vector
 * @return Vector containing the prefix sums
 */
template <typename T> inline Vector<T> prefix_sum(const Vector<T>& v) {
    Vector<T> out(v.size());
    T running = T{};
    for (std::size_t i = 0; i < v.size(); ++i) {
        running += v[i];
        out[i] = running;
    }
    return out;
}

/**
 * @brief Element wise comparison of two vectors.
 *
 * The function returns true only if both vectors have the same length
 * and all corresponding elements compare equal using the `==` operator.
 * It can be handy in tests that need to verify the exact output of
 * simple transformations.
 *
 * @param a first vector
 * @param b second vector
 * @return true if @p a and @p b contain the same values
 */
template <typename T> inline bool equal(const Vector<T>& a, const Vector<T>& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!(a[i] == b[i]))
            return false;
    }
    return true;
}

/**
 * @brief Remove elements equal to @p value from a copy of @p v.
 *
 * This is a small wrapper around the standard remove/erase idiom. The
 * operation does not modify the input vector.
 *
 * @param v vector to filter
 * @param value value to remove
 * @return Filtered vector
 */
template <typename T> inline Vector<T> remove_value(const Vector<T>& v, const T& value) {
    Vector<T> out;
    out.reserve(v.size());
    for (const auto& e : v) {
        if (!(e == value))
            out.push_back(e);
    }
    return out;
}

/**
 * @brief Rotate elements left by @p count positions.
 *
 * For example rotating `[1,2,3,4]` by one position yields `[2,3,4,1]`.
 * The helper returns a new vector leaving the input unchanged. Rotation
 * by a count greater than the vector size wraps around automatically.
 *
 * @param v vector to rotate
 * @param count number of positions to rotate by
 * @return Rotated vector
 */
template <typename T> inline Vector<T> rotate_left(const Vector<T>& v, std::size_t count) {
    if (v.empty())
        return v;
    Vector<T> out(v.size());
    count %= v.size(); // wrap around
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::size_t src = (i + count) % v.size();
        out[i] = v[src];
    }
    return out;
}

/**
 * @brief Rotate elements right by @p count positions.
 *
 * This mirrors @ref rotate_left but in the opposite direction. The
 * implementation is nearly identical with only the index calculation
 * changed. Keeping both functions makes their intent explicit when
 * reading example code.
 *
 * @param v vector to rotate
 * @param count number of positions to rotate by
 * @return Rotated vector
 */
template <typename T> inline Vector<T> rotate_right(const Vector<T>& v, std::size_t count) {
    if (v.empty())
        return v;
    Vector<T> out(v.size());
    count %= v.size();
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::size_t src = (i + v.size() - count) % v.size();
        out[i] = v[src];
    }
    return out;
}

} // namespace harmonics
