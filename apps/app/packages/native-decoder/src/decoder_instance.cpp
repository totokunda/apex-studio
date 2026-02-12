#include "decoder_instance.h"
#include "hwaccel.h"
#include "seek_engine.h"
#include "alpha_merger.h"
#include "thread_pool.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <cstring>

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
    if (onError_) onError_.Release();
    if (onReady_) onReady_.Release();
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

bool DecoderInstance::configure(const DecoderConfig& config, std::string& error) {
    outputWidth_ = config.outputWidth;
    outputHeight_ = config.outputHeight;

    // Store callbacks
    onFrame_ = config.onFrame;
    onError_ = config.onError;
    onReady_ = config.onReady;

    // Initialize frame pool with SharedArrayBuffers from JS
    std::vector<uint8_t*> bufs(config.bufferPool);
    std::vector<size_t> sizes(config.bufferSizes);
    framePool_.init(bufs.data(), sizes.data(), config.poolSize);

    // Open file
    if (!openFile(config.filePath, error)) return false;

    // Find and init video decoder
    if (!initVideoDecoder(error)) return false;

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
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        error = "No decoder found for codec: " +
                std::string(avcodec_get_name(stream->codecpar->codec_id));
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

    // Try hardware acceleration
    AVHWDeviceType hwType = HwAccelManager::getPreferredDeviceType();
    if (hwType != AV_HWDEVICE_TYPE_NONE) {
        ret = HwAccelManager::initHwContext(codecCtx_, hwType);
        if (ret >= 0) {
            codecCtx_->get_format = HwAccelManager::getHwFormat;
        }
        // If hwaccel fails, continue with software decode
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

    AVPixelFormat srcFmt = codecCtx_->pix_fmt;

    // If hwaccel, frames will be transferred to NV12/YUV420P first
    // Determine the actual software format from the hw config
    if (codecCtx_->hw_device_ctx) {
        // For most hwaccel methods, the output after transfer is NV12
        srcFmt = AV_PIX_FMT_NV12;
    }

    swsCtx_ = sws_getContext(
        codecCtx_->width, codecCtx_->height, srcFmt,
        outputWidth_, outputHeight_, AV_PIX_FMT_RGBA,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );

    if (!swsCtx_) {
        error = "Failed to create swscale context";
        return false;
    }

    // If we have a separate alpha stream, create its sws context
    if (alphaCodecCtx_) {
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
    if (!dst) {
        framePool_.release(bufIdx);
        return -1;
    }

    // Set up destination pointers for sws_scale
    uint8_t* dstData[4] = {dst, nullptr, nullptr, nullptr};
    int dstLinesize[4] = {outputWidth_ * 4, 0, 0, 0};

    // Handle hwaccel frames: transfer to software first
    AVFrame* swFrame = av_frame_alloc();
    AVFrame* srcFrame = frame;

    if (frame->hw_frames_ctx) {
        int ret = av_hwframe_transfer_data(swFrame, frame, 0);
        if (ret < 0) {
            av_frame_free(&swFrame);
            framePool_.release(bufIdx);
            return -1;
        }
        srcFrame = swFrame;
    }

    // Determine actual source pixel format (may differ after hw transfer)
    AVPixelFormat actualSrcFmt = static_cast<AVPixelFormat>(srcFrame->format);

    // Recreate sws context if source format changed
    // (e.g., hw decode may produce different format than expected)
    SwsContext* ctx = swsCtx_;
    if (actualSrcFmt != codecCtx_->pix_fmt || srcFrame->width != codecCtx_->width) {
        ctx = sws_getContext(
            srcFrame->width, srcFrame->height, actualSrcFmt,
            outputWidth_, outputHeight_, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
    }

    if (ctx) {
        sws_scale(ctx, srcFrame->data, srcFrame->linesize,
                  0, srcFrame->height, dstData, dstLinesize);
    }

    if (ctx != swsCtx_) {
        sws_freeContext(ctx);
    }

    av_frame_free(&swFrame);

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
    } else if (!hasAlpha_) {
        // No alpha: ensure all pixels are fully opaque
        AlphaMerger::setOpaque(dst, outputWidth_, outputHeight_);
    }
    // If hasAlpha_ but no separate stream (embedded alpha), sws_scale to RGBA
    // already includes the alpha channel.

    return bufIdx;
}

// ---------------------------------------------------------------------------
// Emit callbacks to JS
// ---------------------------------------------------------------------------

void DecoderInstance::emitFrame(
    int bufferIndex, double timestamp, double duration, uint64_t requestId
) {
    auto* data = new FrameCallbackData{
        bufferIndex, outputWidth_, outputHeight_,
        timestamp, duration, requestId
    };

    onFrame_.NonBlockingCall(data,
        [](Napi::Env env, Napi::Function jsCallback, FrameCallbackData* data) {
            jsCallback.Call({
                Napi::Number::New(env, data->bufferIndex),
                Napi::Number::New(env, data->width),
                Napi::Number::New(env, data->height),
                Napi::Number::New(env, data->timestamp),
                Napi::Number::New(env, data->duration),
                Napi::Number::New(env, static_cast<double>(data->requestId)),
            });
            delete data;
        }
    );
}

void DecoderInstance::emitError(const std::string& message) {
    auto* data = new ErrorCallbackData{message};

    onError_.NonBlockingCall(data,
        [](Napi::Env env, Napi::Function jsCallback, ErrorCallbackData* data) {
            jsCallback.Call({Napi::String::New(env, data->message)});
            delete data;
        }
    );
}

void DecoderInstance::emitReady() {
    onReady_.NonBlockingCall(
        [](Napi::Env env, Napi::Function jsCallback) {
            jsCallback.Call({});
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
        int bufIdx = convertFrameToRGBA(outFrame);
        av_frame_free(&outFrame);

        if (bufIdx >= 0 && currentRequestId_.load() == requestId) {
            emitFrame(bufIdx, outPts, outDuration, requestId);
        } else if (bufIdx >= 0) {
            framePool_.release(bufIdx); // Stale request, release buffer
        }

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

        AVStream* stream = fmtCtx_->streams[videoStreamIdx_];
        AVRational timeBase = stream->time_base;

        // Seek to start position
        int64_t startPts = static_cast<int64_t>(
            startTime * timeBase.den / timeBase.num);
        int64_t endPts = static_cast<int64_t>(
            endTime * timeBase.den / timeBase.num);

        int ret = av_seek_frame(fmtCtx_, videoStreamIdx_,
                               startPts, AVSEEK_FLAG_BACKWARD);
        if (ret < 0) {
            char errBuf[256];
            av_strerror(ret, errBuf, sizeof(errBuf));
            onComplete(false, errBuf);
            return;
        }

        avcodec_flush_buffers(codecCtx_);
        if (alphaCodecCtx_) avcodec_flush_buffers(alphaCodecCtx_);

        AVFrame* frame = av_frame_alloc();
        AVFrame* swFrame = av_frame_alloc();
        AVPacket* pkt = av_packet_alloc();

        if (!frame || !swFrame || !pkt) {
            av_frame_free(&frame);
            av_frame_free(&swFrame);
            av_packet_free(&pkt);
            onComplete(false, "Failed to allocate frame/packet");
            return;
        }

        int maxPackets = 100000; // Safety limit

        while (!cancelled_.load(std::memory_order_relaxed) &&
               currentRequestId_.load() == requestId &&
               maxPackets-- > 0) {

            ret = av_read_frame(fmtCtx_, pkt);
            if (ret < 0) break; // EOF or error

            if (pkt->stream_index != videoStreamIdx_) {
                av_packet_unref(pkt);
                continue;
            }

            ret = avcodec_send_packet(codecCtx_, pkt);
            av_packet_unref(pkt);

            if (ret < 0 && ret != AVERROR(EAGAIN)) {
                continue; // Skip bad packets
            }

            while (!cancelled_.load(std::memory_order_relaxed)) {
                ret = avcodec_receive_frame(codecCtx_, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) break;

                // Transfer hw frame if needed
                if (frame->hw_frames_ctx) {
                    ret = av_hwframe_transfer_data(swFrame, frame, 0);
                    if (ret < 0) {
                        av_frame_unref(frame);
                        continue;
                    }
                    swFrame->pts = frame->pts;
                    swFrame->duration = frame->duration;
                    swFrame->best_effort_timestamp = frame->best_effort_timestamp;
                    av_frame_unref(frame);
                    av_frame_move_ref(frame, swFrame);
                }

                // Check PTS bounds
                int64_t framePts = frame->best_effort_timestamp;
                if (framePts == AV_NOPTS_VALUE) framePts = frame->pts;
                if (framePts == AV_NOPTS_VALUE) framePts = 0;

                if (framePts > endPts) {
                    av_frame_unref(frame);
                    goto done; // Past end time
                }

                if (framePts < startPts) {
                    av_frame_unref(frame);
                    continue; // Before start time
                }

                // Convert and emit
                int bufIdx = convertFrameToRGBA(frame);
                int64_t frameDuration = frame->duration; // capture before unref
                av_frame_unref(frame);

                if (bufIdx >= 0 && currentRequestId_.load() == requestId) {
                    double pts = static_cast<double>(framePts) * av_q2d(timeBase);
                    double dur = (frameDuration > 0)
                        ? static_cast<double>(frameDuration) * av_q2d(timeBase)
                        : 1.0 / av_q2d(stream->avg_frame_rate);

                    emitFrame(bufIdx, pts, dur, requestId);
                    // Backpressure: acquire blocks until JS calls ackFrame
                } else if (bufIdx >= 0) {
                    framePool_.release(bufIdx);
                }
            }
        }

    done:
        // Flush decoder
        avcodec_send_packet(codecCtx_, nullptr);
        while (avcodec_receive_frame(codecCtx_, frame) == 0) {
            av_frame_unref(frame);
        }

        av_frame_free(&frame);
        av_frame_free(&swFrame);
        av_packet_free(&pkt);

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
