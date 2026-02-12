#include "thread_pool.h"

namespace apex {

ThreadPool::ThreadPool(size_t numThreads) {
    workers_.reserve(numThreads);
    for (size_t i = 0; i < numThreads; i++) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void ThreadPool::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return shutdown_ || !tasks_.empty();
            });
            if (shutdown_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool(4);
    return pool;
}

} // namespace apex
