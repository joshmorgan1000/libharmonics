#pragma once

#include <blake3.h>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "harmonics/deployment.hpp"

namespace harmonics {

/**
 * \brief Saturating addition on 32-bit integers.
 */
inline int32_t saturating_add(int32_t a, int32_t b) {
    int64_t sum = static_cast<int64_t>(a) + static_cast<int64_t>(b);
    if (sum > std::numeric_limits<int32_t>::max())
        return std::numeric_limits<int32_t>::max();
    if (sum < std::numeric_limits<int32_t>::min())
        return std::numeric_limits<int32_t>::min();
    return static_cast<int32_t>(sum);
}

/**
 * \brief Multiply two INT8 values and accumulate with saturation.
 */
inline int32_t mul_accum(int32_t acc, int8_t a, int8_t b) {
    int32_t prod = static_cast<int32_t>(a) * static_cast<int32_t>(b);
    return saturating_add(acc, prod);
}

/**
 * \brief Deterministic INT8 matrix multiply.
 *
 * The inputs use row-major layout with MxK and KxN dimensions respectively.
 * The accumulator saturates on overflow to guarantee identical results
 * across targets.
 */
inline std::vector<int32_t> int8_matmul(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                                        std::size_t M, std::size_t N, std::size_t K) {
    std::vector<int32_t> out(M * N, 0);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (std::size_t k = 0; k < K; ++k) {
                int8_t va = A[m * K + k];
                int8_t vb = B[k * N + n];
                acc = mul_accum(acc, va, vb);
            }
            out[m * N + n] = acc;
        }
    }
    return out;
}

/**
 * \brief Compute a BLAKE3 digest for the given INT8 matmul output.
 */
inline std::string digest(const std::vector<int32_t>& data) {
    return blake3(data.data(), data.size() * sizeof(int32_t));
}

/**
 * \brief Simple step decay schedule for integer SGD learning rates.
 *
 * The shift value increases by one every `step_interval` calls to
 * `operator()`, saturating at `max_shift`.
 */
struct StepDecayShiftSchedule {
    uint8_t initial_shift{0};     ///< Starting right shift applied on step 0.
    std::size_t step_interval{1}; ///< Number of steps between shift increments.
    uint8_t max_shift{7};         ///< Maximum shift value returned.

    /**
     * \brief Compute the shift for the given step index.
     */
    uint8_t operator()(std::size_t step) const noexcept {
        if (step_interval == 0)
            return initial_shift;
        std::size_t increments = step / step_interval;
        uint8_t shift = static_cast<uint8_t>(initial_shift + increments);
        if (shift > max_shift)
            shift = max_shift;
        return shift;
    }
};

/**
 * \brief Apply an integer SGD update using a right-shift learning rate.
 *
 * Each weight is updated as `w[i] -= grad[i] >> lr_shift` with the result
 * clamped to the INT8 range.
 */
inline void apply_integer_sgd_update(std::vector<int8_t>& w, const std::vector<int32_t>& grad,
                                     uint8_t lr_shift) {
    if (w.size() != grad.size())
        return;

    for (std::size_t i = 0; i < w.size(); ++i) {
        int32_t delta = grad[i] >> lr_shift;
        int32_t updated = static_cast<int32_t>(w[i]) - delta;
        if (updated > std::numeric_limits<int8_t>::max())
            updated = std::numeric_limits<int8_t>::max();
        else if (updated < std::numeric_limits<int8_t>::min())
            updated = std::numeric_limits<int8_t>::min();
        w[i] = static_cast<int8_t>(updated);
    }
}

/**
 * \brief Apply an integer SGD update with a shift schedule.
 *
 * @param w       weight vector to modify
 * @param grad    gradient vector
 * @param sched   shift schedule returning the learning rate shift for `step`
 * @param step    training step index starting at 0
 */
inline void apply_integer_sgd_update(std::vector<int8_t>& w, const std::vector<int32_t>& grad,
                                     const StepDecayShiftSchedule& sched, std::size_t step) {
    apply_integer_sgd_update(w, grad, sched(step));
}

} // namespace harmonics
