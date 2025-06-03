#include "harmonics/python_bindings.hpp"

#if defined(HARMONICS_HAS_PY)
#include <memory>

using harmonics::build_graph;
using harmonics::CycleRuntime;
using harmonics::HarmonicGraph;
using harmonics::Parser;

namespace {

static void destroy_graph_capsule(PyObject* capsule) {
    auto* g = static_cast<HarmonicGraph*>(PyCapsule_GetPointer(capsule, "HarmonicGraph"));
    delete g;
}

static void destroy_runtime_capsule(PyObject* capsule) {
    auto* rt = static_cast<CycleRuntime*>(PyCapsule_GetPointer(capsule, "CycleRuntime"));
    delete rt;
}

static PyObject* py_compile_graph(PyObject*, PyObject* args) {
    const char* src = nullptr;
    if (!PyArg_ParseTuple(args, "s", &src))
        return nullptr;
    Parser parser{src};
    auto ast = parser.parse_declarations();
    auto* g = new HarmonicGraph(build_graph(ast));
    return PyCapsule_New(g, "HarmonicGraph", destroy_graph_capsule);
}

static PyObject* py_destroy_graph(PyObject*, PyObject* args) {
    PyObject* cap = nullptr;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return nullptr;
    if (PyCapsule_CheckExact(cap))
        destroy_graph_capsule(cap);
    Py_RETURN_NONE;
}

static PyObject* py_create_runtime(PyObject*, PyObject* args) {
    PyObject* cap = nullptr;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return nullptr;
    auto* g = static_cast<HarmonicGraph*>(PyCapsule_GetPointer(cap, "HarmonicGraph"));
    if (!g)
        return nullptr;
    auto* rt = new CycleRuntime{*g};
    return PyCapsule_New(rt, "CycleRuntime", destroy_runtime_capsule);
}

static PyObject* py_destroy_runtime(PyObject*, PyObject* args) {
    PyObject* cap = nullptr;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return nullptr;
    if (PyCapsule_CheckExact(cap))
        destroy_runtime_capsule(cap);
    Py_RETURN_NONE;
}

static PyObject* py_run_cycle(PyObject*, PyObject* args) {
    PyObject* cap = nullptr;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return nullptr;
    auto* rt = static_cast<CycleRuntime*>(PyCapsule_GetPointer(cap, "CycleRuntime"));
    if (!rt)
        return nullptr;
    rt->forward();
    Py_RETURN_NONE;
}

static PyMethodDef harmonics_methods[] = {
    {"compile_graph", py_compile_graph, METH_VARARGS, "Compile a DSL string"},
    {"destroy_graph", py_destroy_graph, METH_VARARGS, "Destroy a graph"},
    {"create_runtime", py_create_runtime, METH_VARARGS, "Create a runtime"},
    {"destroy_runtime", py_destroy_runtime, METH_VARARGS, "Destroy a runtime"},
    {"run_cycle", py_run_cycle, METH_VARARGS, "Run a single cycle"},
    {nullptr, nullptr, 0, nullptr}};

static PyModuleDef harmonics_module = {
    PyModuleDef_HEAD_INIT, "harmonics_py", "Python bindings for Harmonics", -1, harmonics_methods,
};

} // namespace

extern "C" PyObject* PyInit_harmonics_py() { return PyModule_Create(&harmonics_module); }

#endif // HARMONICS_HAS_PY
