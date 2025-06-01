#pragma once

#include "harmonics/net_utils.hpp"
#include "harmonics/stream_io.hpp"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>

namespace harmonics {

/**
 * @brief Message carrying a tensor and its proof string.
 */
struct ProofMessage {
    HTensor tensor{};    ///< transmitted tensor
    std::string proof{}; ///< associated proof data
};

/**
 * @brief Simple message bus used for proof-enabled transports.
 *
 * The bus stores messages in an internal queue and provides blocking send
 * and receive operations. It is primarily utilised by the proof aware
 * socket and in-memory producers/consumers during testing.
 */
class ProofMessageBus {
  public:
    void send(const HTensor& t, const std::string& proof) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push({t, proof});
        cv_.notify_all();
    }

    ProofMessage receive() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty(); });
        ProofMessage msg = queue_.front();
        queue_.pop();
        return msg;
    }

  private:
    std::queue<ProofMessage> queue_{};
    std::mutex mutex_{};
    std::condition_variable cv_{};
};

/**
 * @brief Producer that reads messages from a ProofMessageBus.
 *
 * Each call to @ref next returns the tensor from the next queued message
 * while tracking the associated proof string for later retrieval via
 * @ref proof.
 */
class ProofBusProducer : public Producer {
  public:
    explicit ProofBusProducer(std::shared_ptr<ProofMessageBus> bus) : bus_{std::move(bus)} {}

    /** Fetch the next message in advance without consuming it. */
    void fetch() {
        if (!has_next_) {
            next_msg_ = bus_->receive();
            last_proof_ = next_msg_.proof;
            has_next_ = true;
        }
    }

    HTensor next() override {
        if (!has_next_)
            fetch();
        has_next_ = false;
        return std::move(next_msg_.tensor);
    }

    std::size_t size() const override { return 0; }

    const std::string& proof() const { return last_proof_; }

  private:
    std::shared_ptr<ProofMessageBus> bus_{};
    ProofMessage next_msg_{};
    bool has_next_{false};
    std::string last_proof_{};
};

/**
 * @brief Consumer that writes messages to a ProofMessageBus.
 *
 * This class mirrors ProofBusProducer and is mainly used in tests to
 * verify that proof metadata can be sent alongside tensors using a
 * simple in-memory transport.
 */
class ProofBusConsumer : public Consumer {
  public:
    explicit ProofBusConsumer(std::shared_ptr<ProofMessageBus> bus) : bus_{std::move(bus)} {}

    void push(const HTensor& t) override { bus_->send(t, ""); }

    void push(const HTensor& t, const std::string& proof) { bus_->send(t, proof); }

  private:
    std::shared_ptr<ProofMessageBus> bus_{};
};

/**
 * @brief Producer that receives proof-enabled tensors from a socket.
 *
 * The network protocol prefixes each tensor with a proof string. This
 * class reads the proof and serialized tensor from the socket and
 * exposes them via the Producer interface.
 */
class ProofSocketProducer : public Producer {
  public:
    explicit ProofSocketProducer(socket_t fd) : fd_{fd} {}
    ~ProofSocketProducer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    void fetch() {
        if (has_next_)
            return;
        std::uint32_t proof_len = 0;
        auto n = net_read(fd_, &proof_len, sizeof(proof_len));
        if (n != sizeof(proof_len))
            return;
        std::string buf(proof_len, '\0');
        std::size_t off = 0;
        while (off < proof_len) {
            n = net_read(fd_, buf.data() + off, static_cast<int>(proof_len - off));
            if (n <= 0)
                return;
            off += static_cast<std::size_t>(n);
        }
        last_proof_ = std::move(buf);
        std::uint32_t size = 0;
        n = net_read(fd_, &size, sizeof(size));
        if (n != sizeof(size))
            return;
        std::string tbuf(size, '\0');
        off = 0;
        while (off < size) {
            n = net_read(fd_, tbuf.data() + off, static_cast<int>(size - off));
            if (n <= 0)
                return;
            off += static_cast<std::size_t>(n);
        }
        std::istringstream in(tbuf);
        next_tensor_ = read_tensor(in);
        has_next_ = true;
    }

    HTensor next() override {
        if (!has_next_)
            fetch();
        has_next_ = false;
        return std::move(next_tensor_);
    }

    std::size_t size() const override { return 0; }
    const std::string& proof() const { return last_proof_; }

  private:
    socket_t fd_{invalid_socket};
    std::string last_proof_{};
    HTensor next_tensor_{};
    bool has_next_{false};
};

/**
 * @brief Consumer that sends proof-enabled tensors over a socket.
 *
 * The consumer serializes the tensor alongside a proof string using the
 * same lightweight protocol understood by @ref ProofSocketProducer.
 */
class ProofSocketConsumer : public Consumer {
  public:
    explicit ProofSocketConsumer(socket_t fd) : fd_{fd} {}
    ~ProofSocketConsumer() override {
        if (fd_ != invalid_socket)
            net_close(fd_);
    }

    void push(const HTensor& t) override { push(t, ""); }

    void push(const HTensor& t, const std::string& proof) {
        std::uint32_t proof_len = static_cast<std::uint32_t>(proof.size());
        net_write(fd_, &proof_len, sizeof(proof_len));
        if (proof_len)
            net_write(fd_, proof.data(), static_cast<int>(proof.size()));
        std::ostringstream out;
        write_tensor(out, t);
        auto str = out.str();
        std::uint32_t size = static_cast<std::uint32_t>(str.size());
        net_write(fd_, &size, sizeof(size));
        std::size_t off = 0;
        while (off < str.size()) {
            auto n = net_write(fd_, str.data() + off, static_cast<int>(str.size() - off));
            if (n <= 0)
                return;
            off += static_cast<std::size_t>(n);
        }
    }

  private:
    socket_t fd_{invalid_socket};
};

} // namespace harmonics
