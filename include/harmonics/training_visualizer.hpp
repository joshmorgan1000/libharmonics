#pragma once

#include "harmonics/websocket_io.hpp"
#include <cstring>

namespace harmonics {

/**
 * Helper sending training metrics over WebSocket.
 *
 * The callback operator can be assigned to FitOptions::progress
 * to stream the step index, gradient L2 norm, loss value and
 * learning rate as a four-element float tensor.
 */
class WebSocketTrainingVisualizer {
  public:
    WebSocketTrainingVisualizer(const std::string& host, unsigned short port,
                                const std::string& path = "/")
        : ws_{host, port, path} {}

    void operator()(std::size_t step, float grad_norm, float loss, float lr) {
        HTensor t{HTensor::DType::Float32, {4}};
        t.data().resize(sizeof(float) * 4);
        float vals[4] = {static_cast<float>(step), grad_norm, loss, lr};
        std::memcpy(t.data().data(), vals, sizeof(vals));
        ws_.push(t);
    }

  private:
    WebSocketConsumer ws_;
};

} // namespace harmonics
