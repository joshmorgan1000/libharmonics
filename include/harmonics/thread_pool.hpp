#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace harmonics {

/** Simple thread pool for reusing worker threads. */
class ThreadPool {
  public:
    explicit ThreadPool(std::size_t threads = std::thread::hardware_concurrency()) {
        if (threads == 0)
            threads = 1;
        for (std::size_t i = 0; i < threads; ++i)
            workers_.emplace_back(&ThreadPool::worker_loop, this);
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_)
            if (w.joinable())
                w.join();
    }

    /// Schedule a new task for execution.
    void schedule(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

    /// Wait for all tasks to complete.
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&] { return tasks_.empty() && active_ == 0; });
    }

  private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty())
                    return;
                task = std::move(tasks_.front());
                tasks_.pop();
                ++active_;
            }
            task();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_;
                if (tasks_.empty() && active_ == 0)
                    done_cv_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_{};
    std::queue<std::function<void()>> tasks_{};
    std::mutex mutex_{};
    std::condition_variable cv_{};
    std::condition_variable done_cv_{};
    bool stop_{false};
    std::size_t active_{0};
};

} // namespace harmonics
