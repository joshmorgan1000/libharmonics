#pragma once

#include "harmonics/int8_math.hpp"
#include <cmath>
#include <cstddef>
#include <functional>

namespace harmonics {

/** Available optimization algorithms for training. */
enum class Optimizer {
    SGD,     ///< Stochastic gradient descent
    Adam,    ///< Adaptive moment estimation
    RMSProp, ///< RMSProp variant
    AdamW,   ///< Adam with decoupled weight decay
    LAMB     ///< Layer-wise Adaptive Moments
};

/** Simple exponential decay schedule for floating point learning rates. */
struct ExponentialDecaySchedule {
    float initial_lr{0.01f};    ///< Starting learning rate
    float decay_rate{0.99f};    ///< Multiplicative decay factor
    std::size_t decay_steps{1}; ///< Number of steps between decays
    float min_lr{0.0f};         ///< Lower bound for the learning rate

    /** Compute the learning rate for the given step index. */
    float operator()(std::size_t step) const noexcept {
        if (decay_steps == 0)
            return initial_lr;
        std::size_t intervals = step / decay_steps;
        float lr = initial_lr * std::pow(decay_rate, static_cast<float>(intervals));
        return lr < min_lr ? min_lr : lr;
    }
};

/**
 * Options controlling the behaviour of `HarmonicGraph::fit`.
 */
struct FitOptions {
    /** Step size used when updating parameters. */
    float learning_rate = 0.01f;
    /** Algorithm used to apply gradients. */
    Optimizer optimizer = Optimizer::SGD;
    /** Clip gradient values to +/- this threshold. Set to 0 to disable. */
    float grad_clip = 0.0f;
    /** Early stopping patience. Zero disables automatic early stopping. */
    std::size_t early_stop_patience = 0;
    /** Minimum improvement in gradient norm to reset patience. */
    float early_stop_delta = 0.0f;
    /** Weight decay used by AdamW and LAMB optimisers. */
    float weight_decay = 0.0f;
    /** Number of forward passes to accumulate before applying an update. */
    std::size_t accumulate_steps = 1;
    /** Learning rate schedule for integer SGD updates. */
    StepDecayShiftSchedule lr_schedule{};
    /** Optional floating point learning rate schedule. */
    std::function<float(std::size_t)> lr_schedule_fp{};
    /**
     * Optional callback receiving the step index, gradient L2 norm,
     * loss value and the learning rate used for the update.
     */
    std::function<void(std::size_t, float, float, float)> progress{};
};

} // namespace harmonics
