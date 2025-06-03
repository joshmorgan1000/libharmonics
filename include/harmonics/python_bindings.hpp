#pragma once

#include "harmonics/cycle.hpp"
#include "harmonics/parser.hpp"

#if defined(HARMONICS_HAS_PY)
#include <Python.h>
extern "C" PyObject* PyInit_harmonics_py();
#else
inline void* PyInit_harmonics_py() { return nullptr; }
#endif
