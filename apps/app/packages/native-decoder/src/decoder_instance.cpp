#include "decoder_instance.h"
#include "hwaccel.h"
#include "seek_engine.h"
#include "alpha_merger.h"
#include "thread_pool.h"
#include "apple_vt_reader.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <cstring>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

namespace apex {

// Frame callback data passed through ThreadSafeFunction
struct FrameCallbackData {
    int bufferIndex;
    int width;
    int height;
    double timestamp;
    double duration;
    uint64_t requestId;
};

struct FrameBatchCallbackData {
    std::vector<FrameCallbackData> frames;
};

// Error callback data
struct ErrorCallbackData {
    std::string message;
};

DecoderInstance::DecoderInstance() = default;

DecoderInstance::~DecoderInstance() {
    dispose();
}

void DecoderInstance::dispose() {
    if (disposed_.exchange(true)) return;

    cancel();
    framePool_.shutdown();

    std::lock_guard<std::mutex> lock(decoderMutex_);

    if (swsCtx_) {
        sws_freeContext(swsCtx_);
        swsCtx_ = nullptr;
    }
    if (alphaSwsCtx_) {
        sws_freeContext(alphaSwsCtx_);
        alphaSwsCtx_ = nullptr;
    }
    swsSrcFmt_ = AV_PIX_FMT_NONE;
    swsSrcWidth_ = 0;
    swsSrcHeight_ = 0;
    if (hwTransferFrame_) {
        av_frame_free(&hwTransferFrame_);
    }
    if (codecCtx_) {
        avcodec_free_context(&codecCtx_);
    }
    if (alphaCodecCtx_) {
        avcodec_free_context(&alphaCodecCtx_);
    }
    if (fmtCtx_) {
        avformat_close_input(&fmtCtx_);
    }

    // Release ThreadSafeFunctions
    if (onFrame_) onFrame_.Release();
    if (onFrameBatch_) onFrameBatch_.Release();
    if (onError_) onError_.Release();
    if (onReady_) onReady_.Release();
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

bool DecoderInstance::configure(const DecoderConfig& config, std::string& error) {
    outputWidth_ = config.outputWidth;
    outputHeight_ = config.outputHeight;
    decodeOnly_ = config.decodeOnly;
    outputNV12_ = config.outputNV12;
    sourceFilePath_ = config.filePath;
    useAppleDecodeOnlyFastPath_ = false;

    // Store callbacks
    onFrame_ = config.onFrame;
    onFrameBatch_ = config.onFrameBatch;
    onError_ = config.onError;
    onReady_ = config.onReady;
    frameBatchSize_ = std::max(1, config.frameBatchSize);
    suppressFrameCallbacks_ = config.suppressFrameCallbacks;
    {
        std::lock_guard<std::mutex> lock(frameBatchMutex_);
        pendingFrameBatch_.clear();
    }

    // Initialize frame pool with SharedArrayBuffers from JS
    std::vector<uint8_t*> bufs(config.bufferPool);
    std::vector<size_t> sizes(config.bufferSizes);
    framePool_.init(bufs.data(), sizes.data(), config.poolSize);

    // Open file
    if (!openFile(config.filePath, error)) return false;

    // Find and init video decoder
    if (!initVideoDecoder(error)) return false;

#if defined(__APPLE__)
    if (decodeOnly_ &&
        fmtCtx_ &&
        videoStreamIdx_ >= 0 &&
        videoStreamIdx_ < static_cast<int>(fmtCtx_->nb_streams)) {
        const AVCodecID codecId = fmtCtx_->streams[videoStreamIdx_]->codecpar->codec_id;
        switch (codecId) {
            case AV_CODEC_ID_H264:
            case AV_CODEC_ID_HEVC:
            case AV_CODEC_ID_MPEG2VIDEO:
            case AV_CODEC_ID_VP9:
            case AV_CODEC_ID_AV1:
                useAppleDecodeOnlyFastPath_ = true;
                break;
            default:
                useAppleDecodeOnlyFastPath_ = false;
                break;
        }
    }
#endif

    if (!decodeOnly_) {
        if (outputWidth_ <= 0 || outputHeight_ <= 0) {
            error = "Invalid output dimensions";
            return false;
        }

        const uint64_t requiredBytes64 = outputNV12_
            ? (static_cast<uint64_t>(outputWidth_) * static_cast<uint64_t>(outputHeight_) * 3ULL) / 2ULL
            : static_cast<uint64_t>(outputWidth_) * static_cast<uint64_t>(outputHeight_) * 4ULL;
        if (requiredBytes64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            error = "Output frame size overflows size_t";
            return false;
        }
        const size_t requiredBytes = static_cast<size_t>(requiredBytes64);
        for (size_t i = 0; i < config.bufferSizes.size(); i++) {
            if (config.bufferSizes[i] < requiredBytes) {
                error = "Frame buffer too small for output dimensions";
                return false;
            }
        }
    }

    if (!decodeOnly_) {
        // Detect and optionally init alpha decoder
        detectAlphaStream();
        if (hasAlpha_) {
            if (!initAlphaDecoder(error)) {
                // Non-fatal: fall back to no alpha
                hasAlpha_ = false;
                alphaStreamIdx_ = -1;
            }
        }

        // Init pixel format converter
        if (!initSwsContext(error)) return false;
    }

    // Signal ready to JS
    emitReady();

    return true;
}

bool DecoderInstance::openFile(const std::string& filePath, std::string& error) {
    int ret = avformat_open_input(&fmtCtx_, filePath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        error = std::string("Failed to open file: ") + errBuf;
        return false;
    }

    ret = avformat_find_stream_info(fmtCtx_, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        error = std::string("Failed to find stream info: ") + errBuf;
        return false;
    }

    return true;
}

bool DecoderInstance::initVideoDecoder(std::string& error) {
    // Find best video stream
    videoStreamIdx_ = av_find_best_stream(
        fmtCtx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);

    if (videoStreamIdx_ < 0) {
        error = "No video stream found";
        return false;
    }

    AVStream* stream = fmtCtx_->streams[videoStreamIdx_];
    const AVCodecID codecId = stream->codecpar->codec_id;
    const AVCodec* codec = nullptr;

#if defined(__APPLE__)
    const char* vtDecoderName = nullptr;
    switch (codecId) {
        case AV_CODEC_ID_H264:
            vtDecoderName = "h264_videotoolbox";
            break;
        case AV_CODEC_ID_HEVC:
            vtDecoderName = "hevc_videotoolbox";
            break;
        case AV_CODEC_ID_MPEG2VIDEO:
            vtDecoderName = "mpeg2_videotoolbox";
            break;
        case AV_CODEC_ID_VP9:
            vtDecoderName = "vp9_videotoolbox";
            break;
        case AV_CODEC_ID_AV1:
            vtDecoderName = "av1_videotoolbox";
            break;
        default:
            break;
    }
    if (vtDecoderName) {
        codec = avcodec_find_decoder_by_name(vtDecoderName);
    }
#endif
    if (!codec) {
        codec = avcodec_find_decoder(codecId);
    }
    if (!codec) {
        error = "No decoder found for codec: " +
                std::string(avcodec_get_name(codecId));
        return false;
    }

    codecCtx_ = avcodec_alloc_context3(codec);
    if (!codecCtx_) {
        error = "Failed to allocate codec context";
        return false;
    }

    int ret = avcodec_parameters_to_context(codecCtx_, stream->codecpar);
    if (ret < 0) {
        error = "Failed to copy codec parameters";
        return false;
    }

    const bool usingVideoToolboxDecoder =
        std::strstr(codec->name, "videotoolbox") != nullptr;
    if (!usingVideoToolboxDecoder) {
        // Try hardware acceleration with generic codec decoders.
        AVHWDeviceType hwType = HwAccelManager::getPreferredDeviceType();
        if (hwType != AV_HWDEVICE_TYPE_NONE) {
            ret = HwAccelManager::initHwContext(codecCtx_, hwType);
            if (ret >= 0) {
                codecCtx_->get_format = HwAccelManager::getHwFormat;
            }
            // If hwaccel fails, continue with software decode
        }
    }

    // Enable multi-threaded decoding for software path
    codecCtx_->thread_count = 0; // auto-detect
    codecCtx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    ret = avcodec_open2(codecCtx_, codec, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        error = std::string("Failed to open codec: ") + errBuf;
        return false;
    }

    // Auto-detect output dimensions if not specified
    if (outputWidth_ <= 0 || outputHeight_ <= 0) {
        outputWidth_ = codecCtx_->width;
        outputHeight_ = codecCtx_->height;
    }

    return true;
}

void DecoderInstance::detectAlphaStream() {
    // Look for a secondary video stream that could be alpha
    // Common in WebM VP9 with alpha: Matroska block additional data
    // Also check for streams tagged as alpha

    for (unsigned i = 0; i < fmtCtx_->nb_streams; i++) {
        if (static_cast<int>(i) == videoStreamIdx_) continue;

        AVStream* stream = fmtCtx_->streams[i];
        if (stream->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) continue;

        // Check disposition or metadata for alpha indication
        if (stream->disposition & AV_DISPOSITION_DEPENDENT) {
            alphaStreamIdx_ = static_cast<int>(i);
            hasAlpha_ = true;
            return;
        }

        // Check for codec-level alpha support (VP9 alpha in WebM)
        // FFmpeg handles VP9 alpha internally in some builds
        const AVDictionaryEntry* tag = av_dict_get(
            stream->metadata, "alpha_mode", nullptr, 0);
        if (tag) {
            alphaStreamIdx_ = static_cast<int>(i);
            hasAlpha_ = true;
            return;
        }
    }

    // Check if the main video stream has alpha in its pixel format
    if (codecCtx_) {
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codecCtx_->pix_fmt);
        if (desc && (desc->flags & AV_PIX_FMT_FLAG_ALPHA)) {
            hasAlpha_ = true;
            // Alpha is embedded in the main stream, no separate decoder needed
            alphaStreamIdx_ = -1;
        }
    }
}

bool DecoderInstance::initAlphaDecoder(std::string& error) {
    if (alphaStreamIdx_ < 0) return true; // No separate alpha stream

    AVStream* stream = fmtCtx_->streams[alphaStreamIdx_];
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        error = "No decoder for alpha stream codec";
        return false;
    }

    alphaCodecCtx_ = avcodec_alloc_context3(codec);
    if (!alphaCodecCtx_) {
        error = "Failed to allocate alpha codec context";
        return false;
    }

    int ret = avcodec_parameters_to_context(alphaCodecCtx_, stream->codecpar);
    if (ret < 0) {
        error = "Failed to copy alpha codec parameters";
        return false;
    }

    alphaCodecCtx_->thread_count = 0;
    alphaCodecCtx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    ret = avcodec_open2(alphaCodecCtx_, codec, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        error = std::string("Failed to open alpha codec: ") + errBuf;
        return false;
    }

    return true;
}

bool DecoderInstance::initSwsContext(std::string& error) {
    if (!codecCtx_) {
        error = "No codec context for sws init";
        return false;
    }

    // Main sws context is created lazily from the first decoded frame's actual
    // software format and dimensions. This avoids per-frame sws_getContext
    // churn when hw decode output format differs from codecCtx_->pix_fmt.
    if (swsCtx_) {
        sws_freeContext(swsCtx_);
        swsCtx_ = nullptr;
    }
    swsSrcFmt_ = AV_PIX_FMT_NONE;
    swsSrcWidth_ = 0;
    swsSrcHeight_ = 0;

    // If we have a separate alpha stream, create its sws context
    if (alphaCodecCtx_) {
        if (alphaSwsCtx_) {
            sws_freeContext(alphaSwsCtx_);
            alphaSwsCtx_ = nullptr;
        }
        alphaSwsCtx_ = sws_getContext(
            alphaCodecCtx_->width, alphaCodecCtx_->height,
            alphaCodecCtx_->pix_fmt,
            outputWidth_, outputHeight_, AV_PIX_FMT_GRAY8,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );

        if (!alphaSwsCtx_) {
            error = "Failed to create alpha swscale context";
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Frame conversion
// ---------------------------------------------------------------------------

int DecoderInstance::convertFrameToRGBA(AVFrame* frame, AVFrame* alphaFrame) {
    int bufIdx = framePool_.acquire();
    if (bufIdx < 0) return -1; // Pool shut down

    uint8_t* dst = framePool_.getBuffer(bufIdx);
    const size_t dstSize = framePool_.getBufferSize(bufIdx);
    const uint64_t requiredBytes64 =
        static_cast<uint64_t>(outputWidth_) *
        static_cast<uint64_t>(outputHeight_) * 4ULL;
    if (!dst || requiredBytes64 > static_cast<uint64_t>(dstSize)) {
        emitError("Native decoder buffer too small for frame output");
        framePool_.release(bufIdx);
        return -1;
    }

    // Set up destination pointers for sws_scale
    uint8_t* dstData[4] = {dst, nullptr, nullptr, nullptr};
    int dstLinesize[4] = {outputWidth_ * 4, 0, 0, 0};

    // Handle hwaccel frames: transfer to software first
    AVFrame* srcFrame = frame;
    if (frame->hw_frames_ctx) {
        if (!hwTransferFrame_) {
            hwTransferFrame_ = av_frame_alloc();
            if (!hwTransferFrame_) {
                framePool_.release(bufIdx);
                return -1;
            }
        }
        av_frame_unref(hwTransferFrame_);
        int ret = av_hwframe_transfer_data(hwTransferFrame_, frame, 0);
        if (ret < 0) {
            framePool_.release(bufIdx);
            return -1;
        }
        hwTransferFrame_->pts = frame->pts;
        hwTransferFrame_->duration = frame->duration;
        hwTransferFrame_->best_effort_timestamp = frame->best_effort_timestamp;
        srcFrame = hwTransferFrame_;
    }

    // Determine actual source pixel format (may differ after hw transfer)
    AVPixelFormat actualSrcFmt = static_cast<AVPixelFormat>(srcFrame->format);

    // Recreate sws context only when the effective source characteristics change.
    if (!swsCtx_ ||
        actualSrcFmt != swsSrcFmt_ ||
        srcFrame->width != swsSrcWidth_ ||
        srcFrame->height != swsSrcHeight_) {
        if (swsCtx_) {
            sws_freeContext(swsCtx_);
            swsCtx_ = nullptr;
        }
        swsCtx_ = sws_getContext(
            srcFrame->width, srcFrame->height, actualSrcFmt,
            outputWidth_, outputHeight_, AV_PIX_FMT_RGBA,
            SWS_FAST_BILINEAR, nullptr, nullptr, nullptr
        );
        if (!swsCtx_) {
            framePool_.release(bufIdx);
            return -1;
        }
        swsSrcFmt_ = actualSrcFmt;
        swsSrcWidth_ = srcFrame->width;
        swsSrcHeight_ = srcFrame->height;
    }

    int scaledRows = sws_scale(
        swsCtx_,
        srcFrame->data,
        srcFrame->linesize,
        0,
        srcFrame->height,
        dstData,
        dstLinesize
    );
    if (scaledRows <= 0) {
        framePool_.release(bufIdx);
        return -1;
    }

    // Merge alpha if we have a separate alpha frame
    if (alphaFrame && alphaSwsCtx_) {
        // Scale alpha to grayscale at output dimensions
        // Heap-allocated instead of VLA (not valid C++, unsupported by MSVC,
        // and risks stack overflow on large frames like 1920x1080 = ~2 MB).
        std::vector<uint8_t> alphaPlaneVec(static_cast<size_t>(outputWidth_) * outputHeight_);
        uint8_t* alphaPlane = alphaPlaneVec.data();
        uint8_t* alphaDst[4] = {alphaPlane, nullptr, nullptr, nullptr};
        int alphaLinesize[4] = {outputWidth_, 0, 0, 0};

        sws_scale(alphaSwsCtx_, alphaFrame->data, alphaFrame->linesize,
                  0, alphaFrame->height, alphaDst, alphaLinesize);

        AlphaMerger::merge(dst, alphaPlane, outputWidth_,
                          outputWidth_, outputHeight_);
    }
    // For non-alpha video, sws conversion to RGBA already writes opaque alpha.

    return bufIdx;
}

int DecoderInstance::convertFrameToNV12(AVFrame* frame) {
    int bufIdx = framePool_.acquire();
    if (bufIdx < 0) return -1;

    uint8_t* dst = framePool_.getBuffer(bufIdx);
    const size_t dstSize = framePool_.getBufferSize(bufIdx);
    const uint64_t requiredBytes64 =
        (static_cast<uint64_t>(outputWidth_) * static_cast<uint64_t>(outputHeight_) * 3ULL) / 2ULL;
    if (!dst || requiredBytes64 > static_cast<uint64_t>(dstSize)) {
        emitError("Native decoder buffer too small for NV12 output");
        framePool_.release(bufIdx);
        return -1;
    }

    uint8_t* dstY = dst;
    uint8_t* dstUV = dst + (outputWidth_ * outputHeight_);

    AVFrame* srcFrame = frame;
    if (frame->hw_frames_ctx) {
        if (!hwTransferFrame_) {
            hwTransferFrame_ = av_frame_alloc();
            if (!hwTransferFrame_) {
                framePool_.release(bufIdx);
                return -1;
            }
        }
        av_frame_unref(hwTransferFrame_);
        int ret = av_hwframe_transfer_data(hwTransferFrame_, frame, 0);
        if (ret < 0) {
            framePool_.release(bufIdx);
            return -1;
        }
        hwTransferFrame_->pts = frame->pts;
        hwTransferFrame_->duration = frame->duration;
        hwTransferFrame_->best_effort_timestamp = frame->best_effort_timestamp;
        srcFrame = hwTransferFrame_;
    }

    const AVPixelFormat actualSrcFmt = static_cast<AVPixelFormat>(srcFrame->format);
    const bool canCopyDirectNV12 =
        actualSrcFmt == AV_PIX_FMT_NV12 &&
        srcFrame->width == outputWidth_ &&
        srcFrame->height == outputHeight_;

    if (canCopyDirectNV12) {
        for (int y = 0; y < outputHeight_; y++) {
            std::memcpy(
                dstY + static_cast<size_t>(y) * outputWidth_,
                srcFrame->data[0] + static_cast<size_t>(y) * srcFrame->linesize[0],
                static_cast<size_t>(outputWidth_)
            );
        }
        const int chromaHeight = outputHeight_ / 2;
        for (int y = 0; y < chromaHeight; y++) {
            std::memcpy(
                dstUV + static_cast<size_t>(y) * outputWidth_,
                srcFrame->data[1] + static_cast<size_t>(y) * srcFrame->linesize[1],
                static_cast<size_t>(outputWidth_)
            );
        }
        return bufIdx;
    }

    if (!swsCtx_ ||
        actualSrcFmt != swsSrcFmt_ ||
        srcFrame->width != swsSrcWidth_ ||
        srcFrame->height != swsSrcHeight_) {
        if (swsCtx_) {
            sws_freeContext(swsCtx_);
            swsCtx_ = nullptr;
        }
        swsCtx_ = sws_getContext(
            srcFrame->width, srcFrame->height, actualSrcFmt,
            outputWidth_, outputHeight_, AV_PIX_FMT_NV12,
            SWS_FAST_BILINEAR, nullptr, nullptr, nullptr
        );
        if (!swsCtx_) {
            framePool_.release(bufIdx);
            return -1;
        }
        swsSrcFmt_ = actualSrcFmt;
        swsSrcWidth_ = srcFrame->width;
        swsSrcHeight_ = srcFrame->height;
    }

    uint8_t* dstData[4] = {dstY, dstUV, nullptr, nullptr};
    int dstLinesize[4] = {outputWidth_, outputWidth_, 0, 0};
    int scaledRows = sws_scale(
        swsCtx_,
        srcFrame->data,
        srcFrame->linesize,
        0,
        srcFrame->height,
        dstData,
        dstLinesize
    );
    if (scaledRows <= 0) {
        framePool_.release(bufIdx);
        return -1;
    }

    return bufIdx;
}

int DecoderInstance::convertFrameToOutput(AVFrame* frame, AVFrame* alphaFrame) {
    if (outputNV12_) {
        return convertFrameToNV12(frame);
    }
    return convertFrameToRGBA(frame, alphaFrame);
}

// ---------------------------------------------------------------------------
// Emit callbacks to JS
// ---------------------------------------------------------------------------

void DecoderInstance::emitFrame(
    int bufferIndex, double timestamp, double duration, uint64_t requestId
) {
    if (suppressFrameCallbacks_) {
        if (bufferIndex >= 0) {
            framePool_.release(bufferIndex);
        }
        return;
    }

    if (onFrameBatch_ && frameBatchSize_ > 1) {
        bool shouldFlush = false;
        {
            std::lock_guard<std::mutex> lock(frameBatchMutex_);
            pendingFrameBatch_.push_back(QueuedFrameEvent{
                bufferIndex,
                outputWidth_,
                outputHeight_,
                timestamp,
                duration,
                requestId
            });
            shouldFlush = pendingFrameBatch_.size() >= static_cast<size_t>(frameBatchSize_);
        }
        if (shouldFlush) {
            flushFrameBatch();
        }
        return;
    }

    auto* data = new FrameCallbackData{
        bufferIndex, outputWidth_, outputHeight_,
        timestamp, duration, requestId
    };

    onFrame_.NonBlockingCall(data,
        [](Napi::Env env, Napi::Function jsCallback, FrameCallbackData* data) {
            try {
                jsCallback.Call({
                    Napi::Number::New(env, data->bufferIndex),
                    Napi::Number::New(env, data->width),
                    Napi::Number::New(env, data->height),
                    Napi::Number::New(env, data->timestamp),
                    Napi::Number::New(env, data->duration),
                    Napi::Number::New(env, static_cast<double>(data->requestId)),
                });
            } catch (...) {
                if (env.IsExceptionPending()) {
                    (void)env.GetAndClearPendingException();
                }
            }
            delete data;
        }
    );
}

void DecoderInstance::flushFrameBatch() {
    if (!onFrameBatch_) return;

    auto* data = new FrameBatchCallbackData{};
    {
        std::lock_guard<std::mutex> lock(frameBatchMutex_);
        if (pendingFrameBatch_.empty()) {
            delete data;
            return;
        }
        data->frames.reserve(pendingFrameBatch_.size());
        for (const auto& frame : pendingFrameBatch_) {
            data->frames.push_back(FrameCallbackData{
                frame.bufferIndex,
                frame.width,
                frame.height,
                frame.timestamp,
                frame.duration,
                frame.requestId
            });
        }
        pendingFrameBatch_.clear();
    }

    napi_status status = onFrameBatch_.NonBlockingCall(
        data,
        [](Napi::Env env, Napi::Function jsCallback, FrameBatchCallbackData* data) {
            constexpr size_t kFieldsPerFrame = 6;
            const size_t totalFields = data->frames.size() * kFieldsPerFrame;
            Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(
                env,
                totalFields * sizeof(double)
            );
            auto* out = static_cast<double*>(buffer.Data());
            for (size_t i = 0; i < data->frames.size(); i++) {
                const auto& frame = data->frames[i];
                const size_t base = i * kFieldsPerFrame;
                out[base + 0] = static_cast<double>(frame.bufferIndex);
                out[base + 1] = static_cast<double>(frame.width);
                out[base + 2] = static_cast<double>(frame.height);
                out[base + 3] = frame.timestamp;
                out[base + 4] = frame.duration;
                out[base + 5] = static_cast<double>(frame.requestId);
            }

            try {
                jsCallback.Call({
                    Napi::Float64Array::New(env, totalFields, buffer, 0)
                });
            } catch (...) {
                if (env.IsExceptionPending()) {
                    (void)env.GetAndClearPendingException();
                }
            }
            delete data;
        }
    );

    if (status != napi_ok) {
        delete data;
    }
}

void DecoderInstance::emitError(const std::string& message) {
    auto* data = new ErrorCallbackData{message};

    onError_.NonBlockingCall(data,
        [](Napi::Env env, Napi::Function jsCallback, ErrorCallbackData* data) {
            try {
                jsCallback.Call({Napi::String::New(env, data->message)});
            } catch (...) {
                if (env.IsExceptionPending()) {
                    (void)env.GetAndClearPendingException();
                }
            }
            delete data;
        }
    );
}

void DecoderInstance::emitReady() {
    onReady_.NonBlockingCall(
        [](Napi::Env env, Napi::Function jsCallback) {
            try {
                jsCallback.Call({});
            } catch (...) {
                if (env.IsExceptionPending()) {
                    (void)env.GetAndClearPendingException();
                }
            }
        }
    );
}

// ---------------------------------------------------------------------------
// Seek
// ---------------------------------------------------------------------------

void DecoderInstance::seek(
    double timestamp, bool forceAccurate, uint64_t requestId,
    CompletionCallback onComplete
) {
    cancelled_ = false;
    currentRequestId_ = requestId;

    ThreadPool::instance().submit([this, timestamp, forceAccurate,
                                   requestId, onComplete]() {
        if (disposed_) {
            onComplete(false, "Decoder disposed");
            return;
        }

        std::lock_guard<std::mutex> lock(decoderMutex_);

        // Check if this request is still current
        if (currentRequestId_.load() != requestId) {
            onComplete(true, ""); // Silently succeed (superseded)
            return;
        }

        AVFrame* outFrame = av_frame_alloc();
        if (!outFrame) {
            onComplete(false, "Failed to allocate frame");
            return;
        }

        double outPts = 0;
        double outDuration = 0;

        int ret = SeekEngine::seekToTimestamp(
            fmtCtx_, codecCtx_, videoStreamIdx_,
            timestamp, forceAccurate, cancelled_,
            outFrame, outPts, outDuration
        );

        if (ret < 0) {
            av_frame_free(&outFrame);
            if (cancelled_.load()) {
                onComplete(true, ""); // Cancelled = silent success
            } else {
                char errBuf[256];
                av_strerror(ret, errBuf, sizeof(errBuf));
                emitError(std::string("Seek failed: ") + errBuf);
                onComplete(false, errBuf);
            }
            return;
        }

        // Convert to RGBA and emit
        if (decodeOnly_) {
            av_frame_free(&outFrame);
            if (currentRequestId_.load() == requestId) {
                emitFrame(-1, outPts, outDuration, requestId);
            }
        } else {
            int bufIdx = convertFrameToOutput(outFrame);
            av_frame_free(&outFrame);

            if (bufIdx >= 0 && currentRequestId_.load() == requestId) {
                emitFrame(bufIdx, outPts, outDuration, requestId);
            } else if (bufIdx >= 0) {
                framePool_.release(bufIdx); // Stale request, release buffer
            }
        }

        flushFrameBatch();
        onComplete(true, "");
    });
}

// ---------------------------------------------------------------------------
// Iterate
// ---------------------------------------------------------------------------

void DecoderInstance::iterate(
    double startTime, double endTime, uint64_t requestId,
    CompletionCallback onComplete
) {
    cancelled_ = false;
    currentRequestId_ = requestId;

    ThreadPool::instance().submit([this, startTime, endTime,
                                   requestId, onComplete]() {
        if (disposed_) {
            onComplete(false, "Decoder disposed");
            return;
        }

        std::lock_guard<std::mutex> lock(decoderMutex_);

        if (decodeOnly_ && useAppleDecodeOnlyFastPath_) {
            std::string appleErr;
            const bool ok = AppleVTDecodeOnlyIterate(
                sourceFilePath_,
                startTime,
                endTime,
                cancelled_,
                [&](double ptsSec, double durSec) -> bool {
                    if (cancelled_.load(std::memory_order_relaxed) ||
                        currentRequestId_.load() != requestId) {
                        return false;
                    }
                    emitFrame(-1, ptsSec, durSec, requestId);
                    return true;
                },
                appleErr
            );

            if (ok) {
                flushFrameBatch();
                onComplete(true, "");
                return;
            }

            // Fast path failed for this source/runtime: fall back to FFmpeg
            // path for this and subsequent iterate requests.
            useAppleDecodeOnlyFastPath_ = false;
        }

        AVStream* stream = fmtCtx_->streams[videoStreamIdx_];
        AVRational timeBase = stream->time_base;

        // Seek to start position. For full-file iteration from 0, do not clip
        // negative/reordered initial timestamps; otherwise opening B-frames can
        // be dropped versus reference decoders.
        const int64_t startPts = (startTime <= 0.0)
            ? std::numeric_limits<int64_t>::min()
            : static_cast<int64_t>(startTime * timeBase.den / timeBase.num);
        const int64_t seekPts = (startTime <= 0.0)
            ? 0
            : startPts;
        const int64_t endPts = (!std::isfinite(endTime) || endTime <= 0.0)
            ? std::numeric_limits<int64_t>::max()
            : static_cast<int64_t>(endTime * timeBase.den / timeBase.num);

        int seekFlags = AVSEEK_FLAG_BACKWARD;
        if (startTime <= 0.0) {
            // Allow landing on very early non-key packets when possible.
            seekFlags |= AVSEEK_FLAG_ANY;
        }
        int ret = av_seek_frame(fmtCtx_, videoStreamIdx_, seekPts, seekFlags);
        if (ret < 0 && (seekFlags & AVSEEK_FLAG_ANY)) {
            // Some demuxers reject AVSEEK_FLAG_ANY. Retry with backward-only.
            ret = av_seek_frame(fmtCtx_, videoStreamIdx_, seekPts, AVSEEK_FLAG_BACKWARD);
        }
        if (ret < 0) {
            char errBuf[256];
            av_strerror(ret, errBuf, sizeof(errBuf));
            onComplete(false, errBuf);
            return;
        }

        avcodec_flush_buffers(codecCtx_);
        if (alphaCodecCtx_) avcodec_flush_buffers(alphaCodecCtx_);

        AVFrame* frame = av_frame_alloc();
        AVPacket* pkt = av_packet_alloc();

        if (!frame || !pkt) {
            av_frame_free(&frame);
            av_packet_free(&pkt);
            onComplete(false, "Failed to allocate frame/packet");
            return;
        }

        const double avgFps = av_q2d(stream->avg_frame_rate);
        const bool hasAvgFps = std::isfinite(avgFps) && avgFps > 0.0;
        const double fallbackDuration = hasAvgFps ? (1.0 / avgFps) : 0.0;
        const bool debugDecodePath = std::getenv("APEX_NATIVE_DECODER_DEBUG") != nullptr;
        bool loggedDecodePath = false;
        bool reachedEndRange = false;
        std::string decodeError;

        auto emitDecodedFrame = [&](AVFrame* decoded) -> bool {
            if (debugDecodePath && !loggedDecodePath) {
                const AVPixelFormat decodedFmt = static_cast<AVPixelFormat>(decoded->format);
                const char* decodedFmtName = av_get_pix_fmt_name(decodedFmt);
                const AVPixelFormat ctxFmt = codecCtx_
                    ? codecCtx_->pix_fmt
                    : AV_PIX_FMT_NONE;
                const char* ctxFmtName = av_get_pix_fmt_name(ctxFmt);
                std::fprintf(
                    stderr,
                    "[native-decoder] iterate decode path: codecCtx pix_fmt=%s, "
                    "decoded frame pix_fmt=%s, hw_frames_ctx=%s\n",
                    ctxFmtName ? ctxFmtName : "unknown",
                    decodedFmtName ? decodedFmtName : "unknown",
                    decoded->hw_frames_ctx ? "yes" : "no"
                );
                loggedDecodePath = true;
            }

            int64_t framePts = decoded->best_effort_timestamp;
            if (framePts == AV_NOPTS_VALUE) framePts = decoded->pts;
            if (framePts == AV_NOPTS_VALUE) framePts = 0;

            if (framePts > endPts) {
                reachedEndRange = true;
                return false;
            }

            if (framePts < startPts) {
                return true;
            }

            const int64_t frameDuration = decoded->duration;
            if (currentRequestId_.load() == requestId) {
                const double ptsSec = static_cast<double>(framePts) * av_q2d(timeBase);
                const double durSec = (frameDuration > 0)
                    ? static_cast<double>(frameDuration) * av_q2d(timeBase)
                    : fallbackDuration;
                if (decodeOnly_) {
                    emitFrame(-1, ptsSec, durSec, requestId);
                    return true;
                }
            }

            int bufIdx = convertFrameToOutput(decoded);
            if (bufIdx < 0) {
                if (!cancelled_.load(std::memory_order_relaxed) &&
                    currentRequestId_.load() == requestId) {
                    decodeError = "Failed to convert decoded frame";
                }
                return false;
            }

            if (currentRequestId_.load() == requestId) {
                const double ptsSec = static_cast<double>(framePts) * av_q2d(timeBase);
                const double durSec = (frameDuration > 0)
                    ? static_cast<double>(frameDuration) * av_q2d(timeBase)
                    : fallbackDuration;
                emitFrame(bufIdx, ptsSec, durSec, requestId);
            } else {
                framePool_.release(bufIdx);
            }
            return true;
        };

        int maxPackets = 100000; // Safety limit

        while (!cancelled_.load(std::memory_order_relaxed) &&
               currentRequestId_.load() == requestId &&
               maxPackets-- > 0) {

            ret = av_read_frame(fmtCtx_, pkt);
            if (ret < 0) {
                if (ret != AVERROR_EOF) {
                    char errBuf[256];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    decodeError = std::string("Failed reading packet: ") + errBuf;
                }
                break;
            }

            if (pkt->stream_index != videoStreamIdx_) {
                av_packet_unref(pkt);
                continue;
            }

            bool packetPending = true;
            while (packetPending &&
                   !cancelled_.load(std::memory_order_relaxed) &&
                   currentRequestId_.load() == requestId) {
                ret = avcodec_send_packet(codecCtx_, pkt);
                if (ret == 0) {
                    packetPending = false;
                    av_packet_unref(pkt);
                    break;
                }
                if (ret != AVERROR(EAGAIN)) {
                    av_packet_unref(pkt);
                    char errBuf[256];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    decodeError = std::string("Failed sending packet to decoder: ") + errBuf;
                    break;
                }

                // Decoder input queue is full. Drain one output frame and retry
                // the same packet; do not drop the packet.
                ret = avcodec_receive_frame(codecCtx_, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    // Decoder could not provide output to make progress.
                    // Treat as fatal to avoid spinning and dropping this packet.
                    av_packet_unref(pkt);
                    decodeError = "Decoder stalled while handling send_packet(EAGAIN)";
                    break;
                }
                if (ret < 0) {
                    av_packet_unref(pkt);
                    char errBuf[256];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    decodeError = std::string("Failed receiving decoded frame: ") + errBuf;
                    break;
                }
                const bool keepGoing = emitDecodedFrame(frame);
                av_frame_unref(frame);
                if (!keepGoing) {
                    av_packet_unref(pkt);
                    break;
                }
            }
            if (packetPending) {
                av_packet_unref(pkt);
            }

            if (!decodeError.empty() || reachedEndRange) break;

            while (!cancelled_.load(std::memory_order_relaxed)) {
                ret = avcodec_receive_frame(codecCtx_, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) {
                    char errBuf[256];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    decodeError = std::string("Failed receiving decoded frame: ") + errBuf;
                    break;
                }
                const bool keepGoing = emitDecodedFrame(frame);
                av_frame_unref(frame);
                if (!keepGoing) break;
            }

            if (!decodeError.empty() || reachedEndRange) break;
        }

        // Flush decoder and emit delayed/reordered tail frames.
        if (decodeError.empty()) {
            ret = avcodec_send_packet(codecCtx_, nullptr);
            if (ret < 0 && ret != AVERROR_EOF) {
                char errBuf[256];
                av_strerror(ret, errBuf, sizeof(errBuf));
                decodeError = std::string("Failed flushing decoder: ") + errBuf;
            }
        }
        if (decodeError.empty()) {
            while (!cancelled_.load(std::memory_order_relaxed) &&
                   currentRequestId_.load() == requestId) {
                ret = avcodec_receive_frame(codecCtx_, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) {
                    char errBuf[256];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    decodeError = std::string("Failed receiving flushed frame: ") + errBuf;
                    break;
                }

                const bool keepGoing = emitDecodedFrame(frame);
                av_frame_unref(frame);
                if (!keepGoing) break;
            }
        }

        av_frame_free(&frame);
        av_packet_free(&pkt);
        flushFrameBatch();

        // Ensure all emitted frames have been acknowledged by JS before we
        // resolve iterate(). Without this, callers may dispose immediately and
        // lose trailing callbacks that were queued but not yet delivered.
        if (!decodeOnly_ &&
            decodeError.empty() &&
            !cancelled_.load(std::memory_order_relaxed) &&
            currentRequestId_.load() == requestId) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!cancelled_.load(std::memory_order_relaxed) &&
                   currentRequestId_.load() == requestId) {
                if (framePool_.waitUntilAllAvailableFor(std::chrono::milliseconds(50))) {
                    break;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    decodeError = "Timed out waiting for native frame acknowledgements";
                    break;
                }
            }
        }

        if (!decodeError.empty() &&
            !cancelled_.load(std::memory_order_relaxed) &&
            currentRequestId_.load() == requestId) {
            emitError(std::string("Iterate failed: ") + decodeError);
            onComplete(false, decodeError);
            return;
        }

        if (cancelled_.load() || currentRequestId_.load() != requestId) {
            onComplete(true, ""); // Cancelled/superseded = silent success
        } else {
            onComplete(true, "");
        }
    });
}

// ---------------------------------------------------------------------------
// Flow control
// ---------------------------------------------------------------------------

void DecoderInstance::ackFrame(int bufferIndex) {
    framePool_.release(bufferIndex);
}

void DecoderInstance::cancel() {
    cancelled_ = true;
    currentRequestId_++; // Invalidate in-flight requests
}

} // namespace apex
