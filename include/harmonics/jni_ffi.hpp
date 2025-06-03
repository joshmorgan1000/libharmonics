#pragma once

#include "harmonics/cycle.hpp"
#include "harmonics/parser.hpp"

#if defined(HARMONICS_HAS_JNI)
#include <chrono>
#include <jni.h>
namespace harmonics_jni {
extern "C" {
JNIEXPORT jlong JNICALL Java_ai_harmonics_Harmonics_parseGraph(JNIEnv* env, jclass, jstring src);
JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_destroyGraph(JNIEnv* env, jclass, jlong handle);
JNIEXPORT jlong JNICALL Java_ai_harmonics_Harmonics_createRuntime(JNIEnv* env, jclass,
                                                                  jlong graphHandle);
JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_destroyRuntime(JNIEnv* env, jclass,
                                                                  jlong rtHandle);
JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_forward(JNIEnv* env, jclass, jlong rtHandle);
JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_fit(JNIEnv* env, jclass, jlong graphHandle,
                                                       jlong epochs);
JNIEXPORT void JNICALL Java_ai_harmonics_Harmonics_fitFor(JNIEnv* env, jclass, jlong graphHandle,
                                                          jdouble seconds);
}
} // namespace harmonics_jni
#else
namespace harmonics_jni {
inline void jni_stub() {}
} // namespace harmonics_jni
#endif
