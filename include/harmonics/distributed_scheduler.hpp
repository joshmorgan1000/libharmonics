#pragma once

#include "harmonics/cycle.hpp"
#include "harmonics/partition.hpp"
#include "harmonics/secure_io.hpp"
#include "harmonics/stream_io.hpp"
#include <unordered_map>
#include <variant>

namespace harmonics {

/**
 * Simple scheduler that executes partitioned graphs in sequence while
 * forwarding tensors across boundary nodes using in-memory message buses.
 *
 * Each boundary consumer in one partition is paired with a producer of the
 * same name in another partition. During execution the produced tensor is
 * pushed to a queue so the downstream partition can consume it.
 */
class DistributedScheduler {
  public:
    /// Construct the scheduler with already partitioned graphs.
    explicit DistributedScheduler(std::vector<HarmonicGraph> parts,
                                  const DeploymentDescriptor& deploy = {})
        : graphs_{std::move(parts)}, deploy_{deploy} {
        connect_boundaries();
        for (std::size_t i = 0; i < graphs_.size(); ++i) {
            DeploymentDescriptor d = deploy_;
            if (i < deploy_.partitions.size()) {
                d.backend = deploy_.partitions[i].backend;
                d.gpu_device_index = deploy_.partitions[i].device_index;
            }
            runtimes_.emplace_back(graphs_[i], make_hardware_policy(), d);
        }
    }

    /// Return a reference to the runtime for a partition.
    CycleRuntime& runtime(std::size_t i) { return runtimes_.at(i); }
    const CycleRuntime& runtime(std::size_t i) const { return runtimes_.at(i); }

    /// Bind a producer to a specific partition by name.
    void bindProducer(std::size_t part, const std::string& name, std::shared_ptr<Producer> p) {
        if (part >= graphs_.size())
            throw std::out_of_range("invalid partition index");
        graphs_[part].bindProducer(name, std::move(p));
    }

    /// Execute training for a fixed number of epochs.
    void fit(std::size_t epochs) {
        for (std::size_t e = 0; e < epochs; ++e)
            step();
    }

    /// Execute a single forward pass on all partitions.
    ///
    /// Each runtime is stepped in turn. Boundary tensors produced by one
    /// partition are queued on their corresponding message bus so the next
    /// partition can consume them during the same iteration.
    void step() {
        for (std::size_t i = 0; i < runtimes_.size(); ++i) {
            auto& rt = runtimes_[i];
            if (deploy_.secure) {
                for (auto& prod : graphs_[i].producer_bindings) {
                    if (!prod)
                        continue;
                    if (auto pb = std::dynamic_pointer_cast<ProofBusProducer>(prod)) {
                        pb->fetch();
                        rt.set_chain(pb->proof());
                    }
                }
            }
            rt.forward();
            for (const auto& b : boundaries_) {
                if (b.consumer_part == i) {
                    if (deploy_.secure) {
                        auto pb = std::get<std::shared_ptr<ProofMessageBus>>(b.bus);
                        ProofBusConsumer bus{pb};
                        bus.push(rt.state().consumer_tensors[b.consumer_index], rt.proof());
                    } else {
                        auto mb = std::get<std::shared_ptr<MessageBus>>(b.bus);
                        BusConsumer bus{mb};
                        bus.push(rt.state().consumer_tensors[b.consumer_index]);
                    }
                }
            }
        }
    }

  private:
    struct Boundary {
        std::size_t consumer_part{};
        std::size_t consumer_index{};
        std::size_t producer_part{};
        std::size_t producer_index{};
        std::variant<std::shared_ptr<MessageBus>, std::shared_ptr<ProofMessageBus>> bus{};
    };

    /// Connect matching producers and consumers across partitions.
    ///
    /// A small map of producer names to their originating partition is built
    /// first. Each consumer is then paired with the producer of the same name
    /// and a message bus inserted between them. Secure mode swaps the plain bus
    /// for one that also carries zero-knowledge proofs.
    void connect_boundaries() {
        std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> prod_map;
        for (std::size_t p = 0; p < graphs_.size(); ++p)
            for (std::size_t i = 0; i < graphs_[p].producers.size(); ++i)
                prod_map.emplace(graphs_[p].producers[i].name, std::make_pair(p, i));

        for (std::size_t p = 0; p < graphs_.size(); ++p) {
            for (std::size_t i = 0; i < graphs_[p].consumers.size(); ++i) {
                const auto& name = graphs_[p].consumers[i].name;
                auto it = prod_map.find(name);
                if (it == prod_map.end())
                    continue;
                Boundary b;
                b.consumer_part = p;
                b.consumer_index = i;
                b.producer_part = it->second.first;
                b.producer_index = it->second.second;
                if (deploy_.secure) {
                    auto pb = std::make_shared<ProofMessageBus>();
                    b.bus = pb;
                    graphs_[b.producer_part].bindProducer(
                        graphs_[b.producer_part].producers[b.producer_index].name,
                        std::make_shared<ProofBusProducer>(pb));
                } else {
                    auto mb = std::make_shared<MessageBus>();
                    b.bus = mb;
                    graphs_[b.producer_part].bindProducer(
                        graphs_[b.producer_part].producers[b.producer_index].name,
                        std::make_shared<BusProducer>(mb));
                }
                boundaries_.push_back(std::move(b));
            }
        }
    }

    std::vector<HarmonicGraph> graphs_{};
    DeploymentDescriptor deploy_{};
    std::vector<CycleRuntime> runtimes_{};
    std::vector<Boundary> boundaries_{};
};

} // namespace harmonics
