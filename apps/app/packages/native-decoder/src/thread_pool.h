#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace apex {

/**
 * Simple bounded thread pool for running decoder operations off the
 * Node.js event loop. Default 4 threads to match typical GPU hardware
 * decode session limits.
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = 4);
    ~ThreadPool();

    // Submit a task to the pool. Returns immediately.
    void submit(std::function<void()> task);

    // Singleton accessor — shared across all decoder instances
    static ThreadPool& instance();

private:
    void workerLoop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
};

} // namespace apex
