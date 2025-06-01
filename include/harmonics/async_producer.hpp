#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "harmonics/core.hpp"

// ---------------------------------------------------------------------------
// AsyncProducer
//
// This small utility is used heavily in the tests to ensure that the
// runtime behaves correctly when data is produced asynchronously. It is
// deliberately simplistic and merely spawns a thread that repeatedly
// calls `next()` on the wrapped producer. A fixed size queue buffers the
// results so that the consumer side can operate without blocking.
// ---------------------------------------------------------------------------

namespace harmonics {

/**
 * @brief Asynchronously prefetch samples from another producer.
 *
 * The wrapped producer is polled on a dedicated background thread and
 * fetched tensors are stored in an internal queue. This hides the
 * latency of generating samples from the main thread which can be
 * particularly useful in tests that simulate slow data sources.
 */
class AsyncProducer : public Producer {
  public:
    /**
     * @param inner    Producer to fetch samples from.
     * @param capacity Maximum number of prefetched samples to keep queued.
     */
    explicit AsyncProducer(std::shared_ptr<Producer> inner, std::size_t capacity = 1)
        : inner_{std::move(inner)}, capacity_{capacity} {
        worker_ = std::thread(&AsyncProducer::run, this);
    }

    ~AsyncProducer() override {
        {
            // Signal the worker thread to exit.
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable())
            worker_.join();
    }

    HTensor next() override {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait until a prefetched sample is available.
        cv_.wait(lock, [&] { return !queue_.empty(); });
        HTensor t = std::move(queue_.front());
        queue_.pop_front();
        cv_.notify_all();
        return t;
    }

    std::size_t size() const override { return inner_->size(); }

  private:
    /// Background thread that continuously prefetches samples.
    void run() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stop_ || queue_.size() < capacity_; });
            if (stop_)
                break;
            lock.unlock();

            HTensor t = inner_->next();

            lock.lock();
            queue_.push_back(std::move(t)); // store for main thread
            lock.unlock();
            cv_.notify_all();
        }
    }

    std::shared_ptr<Producer> inner_;
    std::size_t capacity_{};
    std::thread worker_{};
    std::mutex mutex_{};
    std::condition_variable cv_{};
    std::deque<HTensor> queue_{};
    bool stop_{false};
};

} // namespace harmonics
