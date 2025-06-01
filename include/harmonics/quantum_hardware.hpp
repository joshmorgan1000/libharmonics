#pragma once

#include "harmonics/config.hpp"
#include "harmonics/quantum_backend.hpp"
#include "harmonics/quantum_stub.hpp"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace harmonics {

enum class QuantumHardwareBackend { None, External };

inline constexpr QuantumHardwareBackend select_quantum_hardware_backend() {
#if HARMONICS_HAS_QUANTUM_HW
    return QuantumHardwareBackend::External;
#else
    return QuantumHardwareBackend::None;
#endif
}

inline constexpr bool quantum_hardware_available() {
    return select_quantum_hardware_backend() != QuantumHardwareBackend::None;
}

inline bool quantum_hardware_runtime_available() {
#if HARMONICS_HAS_QUANTUM_HW
    const char* env = std::getenv("HARMONICS_ENABLE_QUANTUM_HW");
    if (!quantum_hardware_available() || !env || std::strcmp(env, "1") != 0)
        return false;
    const char* lib_env = std::getenv("HARMONICS_QUANTUM_HW_LIB");
    const char* lib = lib_env ? lib_env : "libquantum_hw.so";
    void* handle = dlopen(lib, RTLD_LAZY);
    if (!handle)
        handle = dlopen(std::string("./") + lib, RTLD_LAZY);
    if (handle) {
        dlclose(handle);
        return true;
    }
    return false;
#else
    return false;
#endif
}

inline QuantumResult execute_on_hardware(const QuantumCircuit& qc) {
    if (!quantum_hardware_runtime_available())
        return simulate(qc);
#if HARMONICS_HAS_QUANTUM_HW
    using Fn = QuantumResult (*)(const QuantumCircuit&);
    static Fn fn = nullptr;
    static void* handle = nullptr;
    if (!fn) {
        const char* lib_env = std::getenv("HARMONICS_QUANTUM_HW_LIB");
        const char* lib = lib_env ? lib_env : "libquantum_hw.so";
        handle = dlopen(lib, RTLD_LAZY);
        if (!handle)
            handle = dlopen(std::string("./") + lib, RTLD_LAZY);
        if (handle)
            fn = reinterpret_cast<Fn>(dlsym(handle, "harmonics_quantum_execute"));
    }
    if (fn)
        return fn(qc);
#endif
    return simulate(qc);
}

} // namespace harmonics
