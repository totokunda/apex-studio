#include "frame_pool.h"

namespace apex {

void FramePool::init(uint8_t** buffers, size_t* sizes, int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = false;
    poolSize_ = count;
    buffers_.resize(count);
    sizes_.resize(count);
    available_.resize(count, true);

    for (int i = 0; i < count; i++) {
        buffers_[i] = buffers[i];
        sizes_[i] = sizes[i];
        available_[i] = true;
    }
}

int FramePool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
        if (shutdown_) return true;
        for (int i = 0; i < poolSize_; i++) {
            if (available_[i]) return true;
        }
        return false;
    });

    if (shutdown_) return -1;

    for (int i = 0; i < poolSize_; i++) {
        if (available_[i]) {
            available_[i] = false;
            return i;
        }
    }
    return -1; // should not reach here
}

int FramePool::tryAcquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_) return -1;

    for (int i = 0; i < poolSize_; i++) {
        if (available_[i]) {
            available_[i] = false;
            return i;
        }
    }
    return -1;
}

void FramePool::release(int index) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= 0 && index < poolSize_) {
            available_[index] = true;
        }
    }
    cv_.notify_all();
}

uint8_t* FramePool::getBuffer(int index) const {
    if (index < 0 || index >= poolSize_) return nullptr;
    return buffers_[index];
}

size_t FramePool::getBufferSize(int index) const {
    if (index < 0 || index >= poolSize_) return 0;
    return sizes_[index];
}

void FramePool::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

bool FramePool::waitUntilAllAvailableFor(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] {
        if (shutdown_) return true;
        for (int i = 0; i < poolSize_; i++) {
            if (!available_[i]) return false;
        }
        return true;
    });
}

void FramePool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = false;
    buffers_.clear();
    sizes_.clear();
    available_.clear();
    poolSize_ = 0;
}

} // namespace apex
