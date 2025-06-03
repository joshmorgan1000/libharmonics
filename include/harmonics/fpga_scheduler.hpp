#pragma once

#include "harmonics/fpga_backend.hpp"
#include "harmonics/function_registry.hpp"
#include "harmonics/graph.hpp"
#include "harmonics/opencl_adapter.hpp"
#include "harmonics/precision_policy.hpp"
#include <future>

namespace harmonics {

struct FpgaCycleKernels;
struct CycleState;

/** Scheduler for the OpenCL/FPGA backend. */
struct FpgaScheduler {
    /// Launch the compiled kernels for one cycle.
    static void launch(const FpgaCycleKernels& kernels, const HarmonicGraph& g, CycleState& state,
                       PrecisionPolicy& policy);
};

} // namespace harmonics
