#pragma once

#include "harmonics/cycle.hpp"
#include "harmonics/parser.hpp"

#if defined(HARMONICS_HAS_JS)
#include <v8.h>
#endif

namespace harmonics_js {
#if defined(HARMONICS_HAS_JS)
void register_bindings(v8::Isolate* isolate, v8::Local<v8::Object> exports);
#else
inline void register_bindings(void*, void*) {}
#endif
} // namespace harmonics_js
