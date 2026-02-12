#pragma once

#include <cstdint>
#include <atomic>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

namespace apex {

/**
 * Encapsulates keyframe-based seeking with optional accurate (decode-to-target)
 * mode. Mirrors the seek logic from the existing video-decoder.worker.ts.
 */
class SeekEngine {
public:
    static constexpr int MAX_RESYNC_ATTEMPTS = 4;

    /**
     * Seek to the given timestamp.
     *
     * @param fmtCtx       Demuxer context
     * @param codecCtx     Decoder context
     * @param streamIndex  Video stream index
     * @param timestamp    Target time in seconds
     * @param accurate     If true, decode forward from keyframe to exact target
     * @param cancelled    Atomic flag checked between frames to abort early
     * @param outFrame     Output frame — caller must av_frame_unref when done
     * @param outPts       Actual PTS of the output frame (in stream timebase seconds)
     * @param outDuration  Duration of the output frame in seconds
     *
     * @return 0 on success, negative error code on failure
     */
    static int seekToTimestamp(
        AVFormatContext* fmtCtx,
        AVCodecContext* codecCtx,
        int streamIndex,
        double timestamp,
        bool accurate,
        std::atomic<bool>& cancelled,
        AVFrame* outFrame,
        double& outPts,
        double& outDuration
    );

    /**
     * Seek to a keyframe at or before the given timestamp.
     * Does NOT decode forward — returns the keyframe directly.
     */
    static int seekToKeyframe(
        AVFormatContext* fmtCtx,
        AVCodecContext* codecCtx,
        int streamIndex,
        double timestamp,
        std::atomic<bool>& cancelled,
        AVFrame* outFrame,
        double& outPts,
        double& outDuration
    );

private:
    /**
     * Decode one frame from the stream, handling send_packet/receive_frame loop.
     * Returns 0 when a frame is successfully decoded into outFrame.
     */
    static int decodeNextFrame(
        AVFormatContext* fmtCtx,
        AVCodecContext* codecCtx,
        int streamIndex,
        std::atomic<bool>& cancelled,
        AVFrame* outFrame
    );

    /**
     * Transfer hardware frame to software frame if needed.
     */
    static int ensureSoftwareFrame(AVFrame* frame, AVFrame* swFrame);
};

} // namespace apex
