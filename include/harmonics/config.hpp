#pragma once

#ifndef HARMONICS_HAS_VULKAN
#define HARMONICS_HAS_VULKAN 0
#endif

#ifndef HARMONICS_HAS_OPENCL
#define HARMONICS_HAS_OPENCL 0
#endif

#ifndef HARMONICS_HAS_CUDA
#define HARMONICS_HAS_CUDA 0
#endif

#ifndef HARMONICS_HAS_WASM
#define HARMONICS_HAS_WASM 0
#endif

#ifndef HARMONICS_HAS_WASM_SIMD
#if defined(__wasm_simd128__)
#define HARMONICS_HAS_WASM_SIMD 1
#else
#define HARMONICS_HAS_WASM_SIMD 0
#endif
#endif

#ifndef HARMONICS_HAS_SSE2
#if defined(__SSE2__)
#define HARMONICS_HAS_SSE2 1
#else
#define HARMONICS_HAS_SSE2 0
#endif
#endif

#ifndef HARMONICS_HAS_QUANTUM_HW
#define HARMONICS_HAS_QUANTUM_HW 0
#endif

#ifndef HARMONICS_HAS_ARROW_FLIGHT
#define HARMONICS_HAS_ARROW_FLIGHT 0
#endif
