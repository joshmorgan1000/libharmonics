#pragma once

#include "harmonics/cycle.hpp"
#include "harmonics/parser.hpp"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#else
#ifndef EMSCRIPTEN_KEEPALIVE
#define EMSCRIPTEN_KEEPALIVE
#endif
#endif

namespace harmonics {
extern "C" {

EMSCRIPTEN_KEEPALIVE inline HarmonicGraph* wasm_parse_graph(const char* src) {
    Parser parser{src};
    auto ast = parser.parse_declarations();
    return new HarmonicGraph(build_graph(ast));
}

EMSCRIPTEN_KEEPALIVE inline void wasm_destroy_graph(HarmonicGraph* g) { delete g; }

EMSCRIPTEN_KEEPALIVE inline CycleRuntime* wasm_create_runtime(const HarmonicGraph* g) {
    return new CycleRuntime{*g};
}

EMSCRIPTEN_KEEPALIVE inline void wasm_destroy_runtime(CycleRuntime* rt) { delete rt; }

EMSCRIPTEN_KEEPALIVE inline void wasm_forward(CycleRuntime* rt) { rt->forward(); }

} // extern "C"
} // namespace harmonics
