#pragma once

#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>

#include "harmonics/net_utils.hpp"

#include "harmonics/core.hpp"
#include "harmonics/serialization.hpp"

namespace harmonics {

/** Producer reading serialized tensors from a stream. */
class StreamProducer : public Producer {
  public:
    explicit StreamProducer(std::shared_ptr<std::istream> in) : in_{std::move(in)} {}
    explicit StreamProducer(std::istream& in) : in_{&in, [](std::istream*) {}} {}
    HTensor next() override {
        if (!in_ || !*in_)
            return {};
        try {
            return read_tensor(*in_);
        } catch (...) {
            return {};
        }
    }
    std::size_t size() const override { return 0; }

  private:
    std::shared_ptr<std::istream> in_{};
};

/** Consumer writing serialized tensors to a stream. */
class StreamConsumer : public Consumer {
  public:
    explicit StreamConsumer(std::shared_ptr<std::ostream> out) : out_{std::move(out)} {}
    explicit StreamConsumer(std::ostream& out) : out_{&out, [](std::ostream*) {}} {}
    void push(const HTensor& t) override {
        if (out_ && *out_)
            write_tensor(*out_, t);
    }

  private:
    std::shared_ptr<std::ostream> out_{};
};

/** Producer that loads tensors from a binary file using the serialization helpers. */
class FileProducer : public Producer {
  public:
    explicit FileProducer(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("failed to open file");
        while (f.peek() != EOF) {
            try {
                records_.push_back(read_tensor(f));
            } catch (...) {
                break;
            }
        }
    }

    HTensor next() override {
        if (records_.empty())
            return {};
        return records_[index_++ % records_.size()];
    }

    std::size_t size() const override { return records_.size(); }

  private:
    std::vector<HTensor> records_{};
    std::size_t index_{0};
};

/** Consumer that writes tensors to a binary file using the serialization helpers. */
class FileConsumer : public Consumer {
  public:
    explicit FileConsumer(const std::string& path) : out_(path, std::ios::binary) {
        if (!out_)
            throw std::runtime_error("failed to open file");
    }

    void push(const HTensor& t) override { write_tensor(out_, t); }

  private:
    std::ofstream out_{};
};

/** Producer that reads tensors from a socket using a length prefix. */
class SocketProducer : public Producer {
  public:
    explicit SocketProducer(socket_t fd) : fd_{fd} {}

    HTensor next() override {
        std::uint32_t size = 0;
        auto n = net_read(fd_, &size, sizeof(size));
        if (n != sizeof(size))
            return {};
        std::string buf(size, '\0');
        std::size_t off = 0;
        while (off < size) {
            n = net_read(fd_, buf.data() + off, static_cast<int>(size - off));
            if (n <= 0)
                return {};
            off += static_cast<std::size_t>(n);
        }
        std::istringstream in(buf);
        return read_tensor(in);
    }

    std::size_t size() const override { return 0; }

  private:
    socket_t fd_{};
};

/** Consumer that writes tensors to a socket using a length prefix. */
class SocketConsumer : public Consumer {
  public:
    explicit SocketConsumer(socket_t fd) : fd_{fd} {}

    void push(const HTensor& t) override {
        std::ostringstream out;
        write_tensor(out, t);
        auto str = out.str();
        std::uint32_t size = static_cast<std::uint32_t>(str.size());
        auto written = net_write(fd_, &size, sizeof(size));
        if (written != sizeof(size))
            return;
        std::size_t off = 0;
        while (off < str.size()) {
            written = net_write(fd_, str.data() + off, static_cast<int>(str.size() - off));
            if (written <= 0)
                return;
            off += static_cast<std::size_t>(written);
        }
    }

  private:
    socket_t fd_{};
};

/** Simple in-memory message bus for tensors. */
class MessageBus {
  public:
    void send(const HTensor& t) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(t);
        cv_.notify_all();
    }

    HTensor receive() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty(); });
        HTensor t = queue_.front();
        queue_.pop();
        return t;
    }

  private:
    std::queue<HTensor> queue_{};
    std::mutex mutex_{};
    std::condition_variable cv_{};
};

/** Producer that receives tensors from a MessageBus. */
class BusProducer : public Producer {
  public:
    explicit BusProducer(std::shared_ptr<MessageBus> bus) : bus_{std::move(bus)} {}
    HTensor next() override { return bus_->receive(); }
    std::size_t size() const override { return 0; }

  private:
    std::shared_ptr<MessageBus> bus_;
};

/** Consumer that sends tensors on a MessageBus. */
class BusConsumer : public Consumer {
  public:
    explicit BusConsumer(std::shared_ptr<MessageBus> bus) : bus_{std::move(bus)} {}
    void push(const HTensor& t) override { bus_->send(t); }

  private:
    std::shared_ptr<MessageBus> bus_;
};

} // namespace harmonics
