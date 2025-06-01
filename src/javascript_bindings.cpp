#include "harmonics/javascript_bindings.hpp"

#if defined(HARMONICS_HAS_JS)
#include <v8.h>

using namespace v8;
using harmonics::build_graph;
using harmonics::CycleRuntime;
using harmonics::HarmonicGraph;
using harmonics::Parser;

namespace {

void CompileGraph(const FunctionCallbackInfo<Value>& info) {
    Isolate* isolate = info.GetIsolate();
    if (info.Length() < 1 || !info[0]->IsString()) {
        isolate->ThrowException(String::NewFromUtf8Literal(isolate, "Expected graph source"));
        return;
    }
    String::Utf8Value src(isolate, info[0]);
    Parser parser{*src};
    auto ast = parser.parse_declarations();
    auto* g = new HarmonicGraph(build_graph(ast));
    info.GetReturnValue().Set(External::New(isolate, g));
}

void DestroyGraph(const FunctionCallbackInfo<Value>& info) {
    if (info.Length() < 1 || !info[0]->IsExternal())
        return;
    auto* g = static_cast<HarmonicGraph*>(info[0].As<External>()->Value());
    delete g;
}

void CreateRuntime(const FunctionCallbackInfo<Value>& info) {
    Isolate* isolate = info.GetIsolate();
    if (info.Length() < 1 || !info[0]->IsExternal()) {
        isolate->ThrowException(String::NewFromUtf8Literal(isolate, "Expected graph handle"));
        return;
    }
    auto* g = static_cast<HarmonicGraph*>(info[0].As<External>()->Value());
    auto* rt = new CycleRuntime{*g};
    info.GetReturnValue().Set(External::New(isolate, rt));
}

void DestroyRuntime(const FunctionCallbackInfo<Value>& info) {
    if (info.Length() < 1 || !info[0]->IsExternal())
        return;
    auto* rt = static_cast<CycleRuntime*>(info[0].As<External>()->Value());
    delete rt;
}

void RunCycle(const FunctionCallbackInfo<Value>& info) {
    if (info.Length() < 1 || !info[0]->IsExternal())
        return;
    auto* rt = static_cast<CycleRuntime*>(info[0].As<External>()->Value());
    rt->forward();
}

} // namespace

namespace harmonics_js {

void register_bindings(Isolate* isolate, Local<Object> exports) {
    exports
        ->Set(isolate->GetCurrentContext(), String::NewFromUtf8Literal(isolate, "compileGraph"),
              Function::New(isolate->GetCurrentContext(), CompileGraph).ToLocalChecked())
        .Check();
    exports
        ->Set(isolate->GetCurrentContext(), String::NewFromUtf8Literal(isolate, "destroyGraph"),
              Function::New(isolate->GetCurrentContext(), DestroyGraph).ToLocalChecked())
        .Check();
    exports
        ->Set(isolate->GetCurrentContext(), String::NewFromUtf8Literal(isolate, "createRuntime"),
              Function::New(isolate->GetCurrentContext(), CreateRuntime).ToLocalChecked())
        .Check();
    exports
        ->Set(isolate->GetCurrentContext(), String::NewFromUtf8Literal(isolate, "destroyRuntime"),
              Function::New(isolate->GetCurrentContext(), DestroyRuntime).ToLocalChecked())
        .Check();
    exports
        ->Set(isolate->GetCurrentContext(), String::NewFromUtf8Literal(isolate, "runCycle"),
              Function::New(isolate->GetCurrentContext(), RunCycle).ToLocalChecked())
        .Check();
}

} // namespace harmonics_js

#endif // HARMONICS_HAS_JS
