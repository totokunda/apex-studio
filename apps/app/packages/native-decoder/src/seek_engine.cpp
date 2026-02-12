#include "seek_engine.h"
#include "hwaccel.h"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
}

namespace apex {

int SeekEngine::decodeNextFrame(
    AVFormatContext* fmtCtx,
    AVCodecContext* codecCtx,
    int streamIndex,
    std::atomic<bool>& cancelled,
    AVFrame* outFrame
) {
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return AVERROR(ENOMEM);

    int ret = 0;
    while (!cancelled.load(std::memory_order_relaxed)) {
        // Try to receive a frame first (there may be buffered frames)
        ret = avcodec_receive_frame(codecCtx, outFrame);
        if (ret == 0) {
            av_packet_free(&pkt);
            return 0; // Got a frame
        }
        if (ret != AVERROR(EAGAIN)) {
            // EOF or real error
            av_packet_free(&pkt);
            return ret;
        }

        // Need more data — read next packet
        while (!cancelled.load(std::memory_order_relaxed)) {
            ret = av_read_frame(fmtCtx, pkt);
            if (ret < 0) {
                // EOF — flush decoder
                avcodec_send_packet(codecCtx, nullptr);
                ret = avcodec_receive_frame(codecCtx, outFrame);
                av_packet_free(&pkt);
                return ret;
            }

            if (pkt->stream_index != streamIndex) {
                av_packet_unref(pkt);
                continue;
            }

            ret = avcodec_send_packet(codecCtx, pkt);
            av_packet_unref(pkt);

            if (ret == 0) {
                break; // Packet accepted, try receive again
            }
            if (ret == AVERROR(EAGAIN)) {
                // Decoder full, try receive first
                break;
            }
            // Error sending packet
            av_packet_free(&pkt);
            return ret;
        }
    }

    av_packet_free(&pkt);
    return cancelled.load() ? AVERROR_EXIT : AVERROR_EOF;
}

int SeekEngine::ensureSoftwareFrame(AVFrame* frame, AVFrame* swFrame) {
    if (!frame->hw_frames_ctx) return 0; // Already software

    int ret = av_hwframe_transfer_data(swFrame, frame, 0);
    if (ret < 0) return ret;

    // Copy metadata
    swFrame->pts = frame->pts;
    swFrame->duration = frame->duration;
    swFrame->pkt_dts = frame->pkt_dts;
    swFrame->best_effort_timestamp = frame->best_effort_timestamp;

    // Swap: put software frame data into frame
    av_frame_unref(frame);
    av_frame_move_ref(frame, swFrame);

    return 0;
}

int SeekEngine::seekToKeyframe(
    AVFormatContext* fmtCtx,
    AVCodecContext* codecCtx,
    int streamIndex,
    double timestamp,
    std::atomic<bool>& cancelled,
    AVFrame* outFrame,
    double& outPts,
    double& outDuration
) {
    AVStream* stream = fmtCtx->streams[streamIndex];
    AVRational timeBase = stream->time_base;

    // Convert seconds to stream timebase
    int64_t targetPts = static_cast<int64_t>(timestamp * timeBase.den / timeBase.num);

    // Seek to keyframe at or before target
    int ret = av_seek_frame(fmtCtx, streamIndex, targetPts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) return ret;

    // Flush decoder after seek
    avcodec_flush_buffers(codecCtx);

    // Decode the first frame (should be a keyframe)
    AVFrame* swFrame = av_frame_alloc();
    if (!swFrame) return AVERROR(ENOMEM);

    for (int attempt = 0; attempt < MAX_RESYNC_ATTEMPTS; attempt++) {
        if (cancelled.load(std::memory_order_relaxed)) {
            av_frame_free(&swFrame);
            return AVERROR_EXIT;
        }

        ret = decodeNextFrame(fmtCtx, codecCtx, streamIndex, cancelled, outFrame);
        if (ret < 0) {
            if (ret == AVERROR_EOF) break;
            // Try to resync
            avcodec_flush_buffers(codecCtx);
            continue;
        }

        // Transfer from GPU to CPU if needed
        ret = ensureSoftwareFrame(outFrame, swFrame);
        if (ret < 0) {
            av_frame_unref(outFrame);
            continue;
        }

        // Compute output PTS in seconds
        int64_t pts = outFrame->best_effort_timestamp;
        if (pts == AV_NOPTS_VALUE) pts = outFrame->pts;
        if (pts == AV_NOPTS_VALUE) pts = 0;

        outPts = static_cast<double>(pts) * av_q2d(timeBase);
        outDuration = (outFrame->duration > 0)
            ? static_cast<double>(outFrame->duration) * av_q2d(timeBase)
            : 1.0 / av_q2d(stream->avg_frame_rate);

        av_frame_free(&swFrame);
        return 0;
    }

    av_frame_free(&swFrame);
    return AVERROR(EAGAIN);
}

int SeekEngine::seekToTimestamp(
    AVFormatContext* fmtCtx,
    AVCodecContext* codecCtx,
    int streamIndex,
    double timestamp,
    bool accurate,
    std::atomic<bool>& cancelled,
    AVFrame* outFrame,
    double& outPts,
    double& outDuration
) {
    if (!accurate) {
        return seekToKeyframe(fmtCtx, codecCtx, streamIndex, timestamp,
                             cancelled, outFrame, outPts, outDuration);
    }

    // Accurate seek: seek to keyframe, then decode forward to target
    AVStream* stream = fmtCtx->streams[streamIndex];
    AVRational timeBase = stream->time_base;
    int64_t targetPts = static_cast<int64_t>(timestamp * timeBase.den / timeBase.num);

    int ret = av_seek_frame(fmtCtx, streamIndex, targetPts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) return ret;

    avcodec_flush_buffers(codecCtx);

    AVFrame* swFrame = av_frame_alloc();
    AVFrame* bestFrame = av_frame_alloc();
    if (!swFrame || !bestFrame) {
        av_frame_free(&swFrame);
        av_frame_free(&bestFrame);
        return AVERROR(ENOMEM);
    }

    bool haveBest = false;
    int maxFrames = 500; // Safety limit to prevent infinite decode loops on corrupt files

    while (!cancelled.load(std::memory_order_relaxed) && maxFrames-- > 0) {
        ret = decodeNextFrame(fmtCtx, codecCtx, streamIndex, cancelled, outFrame);
        if (ret < 0) break;

        ret = ensureSoftwareFrame(outFrame, swFrame);
        if (ret < 0) {
            av_frame_unref(outFrame);
            continue;
        }

        int64_t framePts = outFrame->best_effort_timestamp;
        if (framePts == AV_NOPTS_VALUE) framePts = outFrame->pts;
        if (framePts == AV_NOPTS_VALUE) framePts = 0;

        // Keep track of closest frame at or before target
        if (framePts <= targetPts) {
            av_frame_unref(bestFrame);
            av_frame_move_ref(bestFrame, outFrame);
            haveBest = true;
        } else {
            av_frame_unref(outFrame);
        }

        // If we've passed the target, we have our frame
        if (framePts >= targetPts) break;
    }

    av_frame_free(&swFrame);

    if (haveBest) {
        av_frame_move_ref(outFrame, bestFrame);
        av_frame_free(&bestFrame);

        int64_t pts = outFrame->best_effort_timestamp;
        if (pts == AV_NOPTS_VALUE) pts = outFrame->pts;
        if (pts == AV_NOPTS_VALUE) pts = 0;

        outPts = static_cast<double>(pts) * av_q2d(timeBase);
        outDuration = (outFrame->duration > 0)
            ? static_cast<double>(outFrame->duration) * av_q2d(timeBase)
            : 1.0 / av_q2d(stream->avg_frame_rate);

        return 0;
    }

    av_frame_free(&bestFrame);
    return AVERROR(EAGAIN);
}

} // namespace apex
