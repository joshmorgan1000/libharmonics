#pragma once

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "harmonics/core.hpp"

namespace harmonics {

/** Simple runtime registry for activation, loss and layer functions. */
class FunctionRegistry {
  public:
    /** Access singleton instance. */
    static FunctionRegistry& instance();

    void register_activation(const std::string& id, std::shared_ptr<ActivationFunction> fn,
                             bool allow_override = true);

    void register_loss(const std::string& id, std::shared_ptr<LossFunction> fn,
                       bool allow_override = true);

    void register_layer(const std::string& id, std::shared_ptr<LayerFunction> fn,
                        bool allow_override = true);

    const ActivationFunction& activation(const std::string& id) const;
    const LossFunction& loss(const std::string& id) const;
    const LayerFunction& layer(const std::string& id) const;

  private:
    std::unordered_map<std::string, std::shared_ptr<ActivationFunction>> activations_{};
    std::unordered_map<std::string, std::shared_ptr<LossFunction>> losses_{};
    std::unordered_map<std::string, std::shared_ptr<LayerFunction>> layers_{};
    mutable std::mutex mutex_{};
};

inline FunctionRegistry& FunctionRegistry::instance() {
    static FunctionRegistry inst;
    return inst;
}

inline void FunctionRegistry::register_activation(const std::string& id,
                                                  std::shared_ptr<ActivationFunction> fn,
                                                  bool allow_override) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = activations_.find(id);
    if (it != activations_.end()) {
        if (!allow_override) {
            throw std::runtime_error("activation already registered: " + id);
        }
        it->second = std::move(fn);
    } else {
        activations_.emplace(id, std::move(fn));
    }
}

inline void FunctionRegistry::register_loss(const std::string& id, std::shared_ptr<LossFunction> fn,
                                            bool allow_override) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = losses_.find(id);
    if (it != losses_.end()) {
        if (!allow_override) {
            throw std::runtime_error("loss already registered: " + id);
        }
        it->second = std::move(fn);
    } else {
        losses_.emplace(id, std::move(fn));
    }
}

inline void FunctionRegistry::register_layer(const std::string& id,
                                             std::shared_ptr<LayerFunction> fn,
                                             bool allow_override) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layers_.find(id);
    if (it != layers_.end()) {
        if (!allow_override) {
            throw std::runtime_error("layer already registered: " + id);
        }
        it->second = std::move(fn);
    } else {
        layers_.emplace(id, std::move(fn));
    }
}

inline const ActivationFunction& FunctionRegistry::activation(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = activations_.find(id);
    if (it == activations_.end()) {
        throw std::runtime_error("unknown activation: " + id);
    }
    return *it->second;
}

inline const LossFunction& FunctionRegistry::loss(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = losses_.find(id);
    if (it == losses_.end()) {
        throw std::runtime_error("unknown loss: " + id);
    }
    return *it->second;
}

inline const LayerFunction& FunctionRegistry::layer(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layers_.find(id);
    if (it == layers_.end()) {
        throw std::runtime_error("unknown layer: " + id);
    }
    return *it->second;
}

/** Helper functions for convenience. */
inline void registerActivation(const std::string& id, std::shared_ptr<ActivationFunction> fn,
                               bool allow_override = true) {
    FunctionRegistry::instance().register_activation(id, std::move(fn), allow_override);
}

inline void registerLoss(const std::string& id, std::shared_ptr<LossFunction> fn,
                         bool allow_override = true) {
    FunctionRegistry::instance().register_loss(id, std::move(fn), allow_override);
}

inline void registerLayer(const std::string& id, std::shared_ptr<LayerFunction> fn,
                          bool allow_override = true) {
    FunctionRegistry::instance().register_layer(id, std::move(fn), allow_override);
}

inline const ActivationFunction& getActivation(const std::string& id) {
    return FunctionRegistry::instance().activation(id);
}

inline const LossFunction& getLoss(const std::string& id) {
    return FunctionRegistry::instance().loss(id);
}

inline const LayerFunction& getLayer(const std::string& id) {
    return FunctionRegistry::instance().layer(id);
}

} // namespace harmonics
