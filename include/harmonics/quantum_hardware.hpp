#pragma once

#include "harmonics/config.hpp"
#include "harmonics/quantum_backend.hpp"
#include "harmonics/quantum_stub.hpp"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <optional>
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

inline std::optional<uint32_t>& quantum_device_override() {
    static std::optional<uint32_t> index;
    return index;
}

inline void set_quantum_device_index(uint32_t index) { quantum_device_override() = index; }

inline uint32_t quantum_device_index() {
#if HARMONICS_HAS_QUANTUM_HW
    if (const char* env = std::getenv("HARMONICS_QUANTUM_HW_DEVICE")) {
        int idx = std::atoi(env);
        if (idx >= 0)
            return static_cast<uint32_t>(idx);
    } else if (quantum_device_override()) {
        return *quantum_device_override();
    }
    return 0;
#else
    return 0;
#endif
}

inline uint32_t quantum_device_count() {
#if HARMONICS_HAS_QUANTUM_HW
    return 1;
#else
    return 0;
#endif
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
