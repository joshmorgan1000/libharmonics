#pragma once

#include "harmonics/core.hpp"
#include "harmonics/cuda_adapter.hpp"
#include "harmonics/function_registry.hpp"
#include "harmonics/gpu_backend.hpp"
#include "harmonics/graph.hpp"
#include "harmonics/precision_policy.hpp"
#include <future>

namespace harmonics {

struct GpuCycleKernels;
struct CycleState;

/** Skeleton scheduler for the CUDA backend.
 *
 * This first revision executes all operations on the CPU using the
 * existing activation functions and loss implementations. Device
 * memory is used only for tensor transfers. A future update will
 * replace these placeholders with real CUDA kernels.
 */
struct CudaScheduler {
    /// Launch the compiled kernels for one cycle.
    static void launch(const GpuCycleKernels& kernels, const HarmonicGraph& g, CycleState& state,
                       PrecisionPolicy& policy);
};

} // namespace harmonics
