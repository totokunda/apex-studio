#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

namespace apex {

/**
 * Manages a pool of pre-allocated frame buffers backed by SharedArrayBuffers.
 * The JS side allocates the SharedArrayBuffers and passes raw pointers to C++.
 * C++ writes decoded RGBA into these buffers, then signals JS which buffer
 * index is ready. JS calls ackFrame() to return the buffer to the pool.
 */
class FramePool {
public:
    FramePool() = default;

    /**
     * Initialize the pool with externally-owned buffer pointers.
     * @param buffers   Array of raw pointers to SharedArrayBuffer data
     * @param sizes     Byte size of each buffer
     * @param count     Number of buffers in the pool
     */
    void init(uint8_t** buffers, size_t* sizes, int count);

    /**
     * Acquire the next available buffer index. Blocks if all buffers are in use.
     * Returns -1 if the pool has been shut down.
     */
    int acquire();

    /**
     * Try to acquire without blocking. Returns -1 if none available.
     */
    int tryAcquire();

    /**
     * Release a buffer back to the pool after JS has finished reading it.
     */
    void release(int index);

    /**
     * Get the raw pointer for a buffer at the given index.
     */
    uint8_t* getBuffer(int index) const;

    /**
     * Get the byte size of a buffer at the given index.
     */
    size_t getBufferSize(int index) const;

    /**
     * Signal shutdown — unblocks any threads waiting in acquire().
     */
    void shutdown();

    /**
     * Wait until all buffers have been released back to the pool.
     * Returns true when all buffers are available, false on timeout.
     */
    bool waitUntilAllAvailableFor(std::chrono::milliseconds timeout);

    /**
     * Reset pool state (called before re-init).
     */
    void reset();

    int poolSize() const { return poolSize_; }

private:
    std::vector<uint8_t*> buffers_;
    std::vector<size_t> sizes_;
    std::vector<bool> available_;
    int poolSize_ = 0;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
};

} // namespace apex
