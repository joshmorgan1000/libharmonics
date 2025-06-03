#pragma once

#include "harmonics/compression.hpp"
#include "harmonics/cycle.hpp"
#include "harmonics/distributed_io.hpp"
#include "harmonics/flight_io.hpp"
#include "harmonics/graph.hpp"
#include "harmonics/grpc_io.hpp"
#include "harmonics/partition.hpp"
#include "harmonics/tcp_io.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace harmonics {

/** Transport used by RemoteScheduler boundaries. */
enum class RemoteTransport { TCP, GRPC, Flight };

/** Mapping between a graph boundary and a remote endpoint. */
struct RemoteBinding {
    std::string name{};            ///< Producer or consumer name
    std::string host{"127.0.0.1"}; ///< Target host
    unsigned short port{0};        ///< Target port
    RemoteTransport transport{RemoteTransport::TCP};
    bool compress{false};    ///< Enable gradient compression
    int max_message_size{0}; ///< Optional gRPC/Flight message size
};

/**
 * @brief Minimal cross-process scheduler.
 *
 * Each partition runs its own RemoteScheduler instance. Boundary
 * producers and consumers are wired up using the provided bindings so
 * tensors flow across processes via TCP/gRPC/Flight transports.
 */
class RemoteScheduler {
  public:
    RemoteScheduler(HarmonicGraph g, const std::vector<RemoteBinding>& prod_bindings,
                    const std::vector<RemoteBinding>& cons_bindings,
                    const DeploymentDescriptor& deploy = {})
        : graph_{std::move(g)}, deploy_{deploy} {
        bind_producers(prod_bindings);
        bind_consumers(cons_bindings);
        runtime_ = std::make_unique<CycleRuntime>(graph_, make_hardware_policy(), deploy_);
    }

    CycleRuntime& runtime() { return *runtime_; }
    const CycleRuntime& runtime() const { return *runtime_; }

    /// Train the partition for a fixed number of epochs.
    void fit(std::size_t epochs) {
        for (std::size_t e = 0; e < epochs; ++e)
            step();
    }

    /// Execute a single forward pass.
    void step() {
        if (deploy_.secure) {
            for (auto& prod : graph_.producer_bindings) {
                if (!prod)
                    continue;
                if (auto p = std::dynamic_pointer_cast<ProofTcpProducer>(prod)) {
                    p->fetch();
                    runtime_->set_chain(p->proof());
                }
            }
        }
        runtime_->forward();
        for (std::size_t i = 0; i < consumer_bindings_.size(); ++i) {
            auto& cons = consumer_bindings_[i];
            if (!cons)
                continue;
            const auto& tensor = runtime_->state().consumer_tensors[i];
            if (deploy_.secure) {
                if (auto pc = std::dynamic_pointer_cast<ProofTcpConsumer>(cons))
                    pc->push(tensor, runtime_->proof());
                else
                    cons->push(tensor);
            } else {
                cons->push(tensor);
            }
        }
    }

  private:
    std::unique_ptr<Producer> make_producer(const RemoteBinding& b) {
        std::unique_ptr<Producer> prod;
        switch (b.transport) {
        case RemoteTransport::TCP:
            if (deploy_.secure)
                prod = std::make_unique<ProofTcpProducer>(b.host, b.port);
            else
                prod = std::make_unique<TcpProducer>(b.host, b.port);
            break;
        case RemoteTransport::GRPC:
            prod = std::make_unique<GrpcProducer>(b.host, b.port, b.max_message_size);
            break;
        case RemoteTransport::Flight:
            prod = std::make_unique<FlightProducer>(b.host, b.port, b.max_message_size);
            break;
        }
        if (b.compress)
            prod = std::make_unique<ZstdProducer>(std::move(prod));
        return prod;
    }

    std::unique_ptr<Consumer> make_consumer(const RemoteBinding& b) {
        std::unique_ptr<Consumer> cons;
        switch (b.transport) {
        case RemoteTransport::TCP:
            if (deploy_.secure)
                cons = std::make_unique<ProofTcpConsumer>(b.host, b.port);
            else
                cons = std::make_unique<TcpConsumer>(b.host, b.port);
            break;
        case RemoteTransport::GRPC:
            cons = std::make_unique<GrpcConsumer>(b.host, b.port, b.max_message_size);
            break;
        case RemoteTransport::Flight:
            cons = std::make_unique<FlightConsumer>(b.host, b.port, b.max_message_size);
            break;
        }
        if (b.compress)
            cons = std::make_unique<ZstdConsumer>(std::move(cons));
        return cons;
    }

    void bind_producers(const std::vector<RemoteBinding>& bindings) {
        for (const auto& b : bindings)
            graph_.bindProducer(b.name, make_producer(b));
    }

    void bind_consumers(const std::vector<RemoteBinding>& bindings) {
        for (const auto& b : bindings) {
            NodeId id = graph_.find(b.name);
            if (id.kind != NodeKind::Consumer)
                throw std::runtime_error(b.name + " is not a consumer");
            if (id.index >= graph_.consumers.size())
                throw std::runtime_error("invalid consumer index");
            if (consumer_bindings_.size() <= id.index)
                consumer_bindings_.resize(id.index + 1);
            consumer_bindings_[id.index] = make_consumer(b);
        }
    }

    HarmonicGraph graph_{};
    DeploymentDescriptor deploy_{};
    std::vector<std::shared_ptr<Consumer>> consumer_bindings_{};
    std::unique_ptr<CycleRuntime> runtime_{};
};

} // namespace harmonics
