#include "harmonics/jni_ffi.hpp"
#if defined(HARMONICS_HAS_JNI)
#include <chrono>

using harmonics::CycleRuntime;
using harmonics::HarmonicGraph;
using harmonics::Parser;

extern "C" {
JNIEXPORT jlong JNICALL Java_ai_harmonics_Harmonics_parseGraph(JNIEnv* env, jclass, jstring src) {
    const char* csrc = env->GetStringUTFChars(src, nullptr);
    Parser parser{csrc};
    auto ast = parser.parse_declarations();
    env->ReleaseStringUTFChars(src, csrc);
    return reinterpret_cast<jlong>(new HarmonicGraph(harmonics::build_graph(ast)));
}

JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_destroyGraph(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<HarmonicGraph*>(handle);
}

JNIEXPORT jlong JNICALL Java_ai_harmonics_Harmonics_createRuntime(JNIEnv*, jclass,
                                                                  jlong graphHandle) {
    auto* g = reinterpret_cast<HarmonicGraph*>(graphHandle);
    return reinterpret_cast<jlong>(new CycleRuntime{*g});
}

JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_destroyRuntime(JNIEnv*, jclass, jlong rtHandle) {
    delete reinterpret_cast<CycleRuntime*>(rtHandle);
}

JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_forward(JNIEnv*, jclass, jlong rtHandle) {
    auto* rt = reinterpret_cast<CycleRuntime*>(rtHandle);
    rt->forward();
}

JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_fit(JNIEnv*, jclass, jlong graphHandle,
                                                       jlong epochs) {
    auto* g = reinterpret_cast<HarmonicGraph*>(graphHandle);
    g->fit(static_cast<std::size_t>(epochs), nullptr);
}

JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_fitFor(JNIEnv*, jclass, jlong graphHandle,
                                                          jdouble seconds) {
    auto* g = reinterpret_cast<HarmonicGraph*>(graphHandle);
    auto dur = std::chrono::duration<double>(seconds);
    g->fit(dur, nullptr);
}
} // extern "C"
#endif // HARMONICS_HAS_JNI
