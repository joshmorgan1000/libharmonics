#pragma once

#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "harmonics/function_registry.hpp"

namespace harmonics {

/** Handle for a dynamically loaded plugin. */
struct PluginHandle {
    void* handle{nullptr};
    int version{0};
};

/** Internal cache of loaded plugins indexed by file path. */
inline std::unordered_map<std::string, PluginHandle> plugin_cache;
inline std::mutex plugin_mutex;

/**
 * Load a plugin shared library and register its functions.
 *
 * The library must define an extern "C" function with the signature:
 * `void harmonics_register(harmonics::FunctionRegistry&)`.
 *
 * @param path path to the shared library
 * @return handle for the loaded plugin
 * @throws std::runtime_error if loading fails or the registration symbol is missing
 */
PluginHandle load_plugin(const std::string& path);

/** Reload an already loaded plugin in-place. */
PluginHandle reload_plugin(const std::string& path);

/**
 * Load all plugins located in a given directory.
 * Only files with the `.so` extension are considered.
 */
std::vector<PluginHandle> load_plugins_in_directory(const std::string& directory);

/** Reload all plugins under a directory. */
std::vector<PluginHandle> reload_plugins_in_directory(const std::string& directory);

/**
 * Load plugins from a colon separated search path.
 *
 * Environment variable `HARMONICS_PLUGIN_PATH` is used if the path argument
 * is empty.
 */
std::vector<PluginHandle> load_plugins_from_path(const std::string& path = "");

/** Reload plugins found in a colon separated search path. */
std::vector<PluginHandle> reload_plugins_from_path(const std::string& path = "");

/**
 * Asynchronously load a single plugin.
 *
 * The returned future completes once the plugin has been loaded and
 * registered with the global FunctionRegistry.
 */
std::future<PluginHandle> load_plugin_async(const std::string& path);

/**
 * Asynchronously load all plugins in a directory.
 */
std::future<std::vector<PluginHandle>>
load_plugins_in_directory_async(const std::string& directory);

/**
 * Asynchronously load plugins from a colon separated search path.
 */
std::future<std::vector<PluginHandle>> load_plugins_from_path_async(const std::string& path = "");

/** Asynchronously reload a single plugin. */
std::future<PluginHandle> reload_plugin_async(const std::string& path);

/** Asynchronously reload all plugins in a directory. */
std::future<std::vector<PluginHandle>>
reload_plugins_in_directory_async(const std::string& directory);

/** Asynchronously reload plugins from a colon separated search path. */
std::future<std::vector<PluginHandle>> reload_plugins_from_path_async(const std::string& path = "");

/**
 * Unload a previously loaded plugin.
 *
 * The registered functions remain available in the global registry but any
 * static resources owned by the library are released. No plugin code must be
 * executing when this function is called.
 */
void unload_plugin(PluginHandle handle);

/** Unload multiple plugins returned from the load helpers. */
void unload_plugins(std::vector<PluginHandle>& handles);

/** Asynchronously unload a plugin. */
std::future<void> unload_plugin_async(PluginHandle handle);

/** Asynchronously unload multiple plugins. */
std::future<void> unload_plugins_async(std::vector<PluginHandle>& handles);

} // namespace harmonics

#ifdef HARMONICS_PLUGIN_IMPL
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <vector>

namespace harmonics {

inline PluginHandle load_plugin(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);
    auto it = plugin_cache.find(path);
    if (it != plugin_cache.end())
        return it->second;

    void* h = dlopen(path.c_str(), RTLD_LAZY);
    if (!h)
        throw std::runtime_error(dlerror());
    using RegFn = void (*)(FunctionRegistry&);
    auto* fn = reinterpret_cast<RegFn>(dlsym(h, "harmonics_register"));
    if (!fn) {
        dlclose(h);
        throw std::runtime_error("harmonics_register not found in plugin");
    }
    fn(FunctionRegistry::instance());
    using VerFn = int (*)();
    auto* vfn = reinterpret_cast<VerFn>(dlsym(h, "harmonics_plugin_version"));
    int ver = vfn ? vfn() : 0;
    PluginHandle handle{h, ver};
    plugin_cache.emplace(path, handle);
    return handle;
}

inline PluginHandle reload_plugin(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);
    auto it = plugin_cache.find(path);
    if (it != plugin_cache.end())
        unload_plugin(it->second);
    return load_plugin(path);
}

inline std::vector<PluginHandle> load_plugins_in_directory(const std::string& directory) {
    std::vector<PluginHandle> handles;
    namespace fs = std::filesystem;
    fs::path dir{directory};
    if (!fs::exists(dir) || !fs::is_directory(dir))
        return handles;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".so") {
            try {
                handles.push_back(load_plugin(entry.path().string()));
            } catch (...) {
            }
        }
    }
    return handles;
}

inline std::vector<PluginHandle> reload_plugins_in_directory(const std::string& directory) {
    std::vector<PluginHandle> handles;
    namespace fs = std::filesystem;
    fs::path dir{directory};
    if (!fs::exists(dir) || !fs::is_directory(dir))
        return handles;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".so") {
            try {
                handles.push_back(reload_plugin(entry.path().string()));
            } catch (...) {
            }
        }
    }
    return handles;
}

inline std::vector<PluginHandle> load_plugins_from_path(const std::string& path) {
    const char* env = std::getenv("HARMONICS_PLUGIN_PATH");
    std::string search = path.empty() && env ? env : path;
    std::vector<PluginHandle> handles;
    if (search.empty())
        return handles;
    std::stringstream ss(search);
    std::string dir;
    while (std::getline(ss, dir, ':')) {
        if (!dir.empty()) {
            auto h = load_plugins_in_directory(dir);
            handles.insert(handles.end(), h.begin(), h.end());
        }
    }
    return handles;
}

inline std::vector<PluginHandle> reload_plugins_from_path(const std::string& path) {
    const char* env = std::getenv("HARMONICS_PLUGIN_PATH");
    std::string search = path.empty() && env ? env : path;
    std::vector<PluginHandle> handles;
    if (search.empty())
        return handles;
    std::stringstream ss(search);
    std::string dir;
    while (std::getline(ss, dir, ':')) {
        if (!dir.empty()) {
            auto h = reload_plugins_in_directory(dir);
            handles.insert(handles.end(), h.begin(), h.end());
        }
    }
    return handles;
}

inline void unload_plugin(PluginHandle handle) {
    if (handle.handle) {
        std::lock_guard<std::mutex> lock(plugin_mutex);
        dlclose(handle.handle);
        for (auto it = plugin_cache.begin(); it != plugin_cache.end(); ++it) {
            if (it->second.handle == handle.handle) {
                plugin_cache.erase(it);
                break;
            }
        }
    }
}

inline void unload_plugins(std::vector<PluginHandle>& handles) {
    for (auto& h : handles) {
        if (h.handle) {
            std::lock_guard<std::mutex> lock(plugin_mutex);
            dlclose(h.handle);
            for (auto it = plugin_cache.begin(); it != plugin_cache.end(); ++it) {
                if (it->second.handle == h.handle) {
                    plugin_cache.erase(it);
                    break;
                }
            }
            h.handle = nullptr;
        }
    }
    handles.clear();
}

inline std::future<PluginHandle> load_plugin_async(const std::string& path) {
    return std::async(std::launch::async, [path]() { return load_plugin(path); });
}

inline std::future<std::vector<PluginHandle>>
load_plugins_in_directory_async(const std::string& directory) {
    return std::async(std::launch::async,
                      [directory]() { return load_plugins_in_directory(directory); });
}

inline std::future<std::vector<PluginHandle>>
load_plugins_from_path_async(const std::string& path) {
    return std::async(std::launch::async, [path]() { return load_plugins_from_path(path); });
}

inline std::future<PluginHandle> reload_plugin_async(const std::string& path) {
    return std::async(std::launch::async, [path]() { return reload_plugin(path); });
}

inline std::future<std::vector<PluginHandle>>
reload_plugins_in_directory_async(const std::string& directory) {
    return std::async(std::launch::async,
                      [directory]() { return reload_plugins_in_directory(directory); });
}

inline std::future<std::vector<PluginHandle>>
reload_plugins_from_path_async(const std::string& path) {
    return std::async(std::launch::async, [path]() { return reload_plugins_from_path(path); });
}

inline std::future<void> unload_plugin_async(PluginHandle handle) {
    return std::async(std::launch::async, [handle]() { unload_plugin(handle); });
}

inline std::future<void> unload_plugins_async(std::vector<PluginHandle>& handles) {
    return std::async(std::launch::async, [&handles]() { unload_plugins(handles); });
}

} // namespace harmonics
#endif
