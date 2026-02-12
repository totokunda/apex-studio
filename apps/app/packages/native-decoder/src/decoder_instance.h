#pragma once

#include <napi.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <functional>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
}

#include "frame_pool.h"

namespace apex {

/**
 * Configuration passed from JS to configure a decoder instance.
 */
struct DecoderConfig {
    std::string filePath;
    int outputWidth = 0;
    int outputHeight = 0;

    // SharedArrayBuffer pool — raw pointers to pre-allocated buffers
    std::vector<uint8_t*> bufferPool;
    std::vector<size_t> bufferSizes;
    int poolSize = 0;

    // Thread-safe callbacks into JS
    Napi::ThreadSafeFunction onFrame;
    Napi::ThreadSafeFunction onError;
    Napi::ThreadSafeFunction onReady;
};

using CompletionCallback = std::function<void(bool success, const std::string& error)>;

/**
 * One DecoderInstance per logical decoder (per clip on the timeline).
 *
 * Owns an FFmpeg demuxer (AVFormatContext), video decoder (AVCodecContext),
 * optional alpha decoder, and a pixel format converter (SwsContext).
 *
 * All heavy work (demux/decode/convert) runs on the shared ThreadPool.
 * Results are delivered back to JS via Napi::ThreadSafeFunction callbacks.
 */
class DecoderInstance {
public:
    DecoderInstance();
    ~DecoderInstance();

    /**
     * Open the media file, find the video stream, initialize the codec
     * with hardware acceleration, and set up pixel format conversion.
     */
    bool configure(const DecoderConfig& config, std::string& error);

    /**
     * Seek to a specific timestamp (seconds). Runs on thread pool.
     * Emits one frame via onFrame callback, then resolves the promise.
     */
    void seek(double timestamp, bool forceAccurate, uint64_t requestId,
              CompletionCallback onComplete);

    /**
     * Iterate all frames in [startTime, endTime]. Runs on thread pool.
     * Emits frames via onFrame callback with backpressure (waits for ackFrame).
     */
    void iterate(double startTime, double endTime, uint64_t requestId,
                 CompletionCallback onComplete);

    /**
     * Release a frame buffer back to the pool (called from JS after GPU upload).
     */
    void ackFrame(int bufferIndex);

    /**
     * Cancel the current in-progress seek or iterate.
     */
    void cancel();

    /**
     * Close all FFmpeg contexts and release resources.
     */
    void dispose();

private:
    // FFmpeg contexts
    AVFormatContext* fmtCtx_ = nullptr;
    AVCodecContext* codecCtx_ = nullptr;
    AVCodecContext* alphaCodecCtx_ = nullptr;
    SwsContext* swsCtx_ = nullptr;
    SwsContext* alphaSwsCtx_ = nullptr;

    // Stream indices
    int videoStreamIdx_ = -1;
    int alphaStreamIdx_ = -1;
    bool hasAlpha_ = false;

    // Output dimensions
    int outputWidth_ = 0;
    int outputHeight_ = 0;

    // Frame buffer pool (SharedArrayBuffer-backed)
    FramePool framePool_;

    // Thread-safe JS callbacks
    Napi::ThreadSafeFunction onFrame_;
    Napi::ThreadSafeFunction onError_;
    Napi::ThreadSafeFunction onReady_;

    // Cancellation & request tracking
    std::atomic<bool> cancelled_{false};
    std::atomic<uint64_t> currentRequestId_{0};

    // Protects AVCodecContext during concurrent operations
    std::mutex decoderMutex_;

    // Disposed flag
    std::atomic<bool> disposed_{false};

    // Internal helpers
    bool openFile(const std::string& filePath, std::string& error);
    bool initVideoDecoder(std::string& error);
    bool initAlphaDecoder(std::string& error);
    bool initSwsContext(std::string& error);
    void detectAlphaStream();

    /**
     * Convert a decoded AVFrame to RGBA and write into a pool buffer.
     * Returns the buffer index, or -1 on failure.
     */
    int convertFrameToRGBA(AVFrame* frame, AVFrame* alphaFrame = nullptr);

    /**
     * Emit a frame to JS via onFrame ThreadSafeFunction.
     */
    void emitFrame(int bufferIndex, double timestamp, double duration, uint64_t requestId);

    /**
     * Emit an error to JS via onError ThreadSafeFunction.
     */
    void emitError(const std::string& message);

    /**
     * Signal ready to JS via onReady ThreadSafeFunction.
     */
    void emitReady();
};

} // namespace apex
