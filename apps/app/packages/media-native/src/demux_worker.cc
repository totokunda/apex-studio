#include "demux_worker.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
}

namespace media_native {

namespace {

constexpr size_t kMaxQueuedRequests = 256;

struct MediaContext {
  AVFormatContext* format_ctx = nullptr;
  AVCodecContext* codec_ctx = nullptr;
  int video_stream_index = -1;
  AVRational time_base{1, 1};
  double fps = 30.0;
  int64_t last_source_frame = 0;
  bool has_position = false;
};

int64_t FrameToStreamPts(const MediaContext& ctx, int64_t source_frame) {
  const double safe_fps = ctx.fps > 0.0 ? ctx.fps : 30.0;
  const double micros_d =
      (static_cast<double>(source_frame) / safe_fps) * 1000000.0;
  const int64_t micros =
      micros_d > 0.0 ? static_cast<int64_t>(std::llround(micros_d)) : 0;
  return av_rescale_q(micros, AVRational{1, 1000000}, ctx.time_base);
}

double RationalToFps(const AVRational& r, double fallback) {
  if (r.num <= 0 || r.den <= 0) return fallback;
  const double v = static_cast<double>(r.num) / static_cast<double>(r.den);
  if (!std::isfinite(v) || v <= 0.0) return fallback;
  return v;
}

void ReleaseNativeAvFrame(void* raw_frame) {
  AVFrame* frame = static_cast<AVFrame*>(raw_frame);
  av_frame_free(&frame);
}

bool PopulateDecodedVideoFrame(const DecodeRequest& req,
                               const AVFrame* frame,
                               int64_t frame_pts,
                               DecodedVideoFrame* out) {
  if (!frame || !out) return false;
  const auto pixel_format = static_cast<AVPixelFormat>(frame->format);
  if (pixel_format == AV_PIX_FMT_NONE) return false;
  if (frame->width <= 0 || frame->height <= 0) return false;
  AVFrame* retained_frame = av_frame_clone(frame);
  if (!retained_frame) return false;

  out->clip_id = req.clip_id;
  out->media_path = req.media_path;
  out->focus_frame = req.focus_frame;
  out->source_frame = req.source_frame;
  out->pts = frame_pts == AV_NOPTS_VALUE ? 0 : frame_pts;
  out->width = frame->width;
  out->height = frame->height;
  out->pixel_format = frame->format;
  out->z_index = req.z_index;
  out->native_frame = NativeOwnedHandle(retained_frame, ReleaseNativeAvFrame);
  return true;
}

}  // namespace

struct DemuxWorker::Impl {
  mutable std::mutex mu;
  std::condition_variable cv;
  bool stop_requested = false;
  bool reset_requested = false;
  std::deque<DecodeRequest> queue;
  std::unordered_map<std::string, MediaContext> contexts;
  DemuxWorkerStats stats{};
  DecodedFrameCallback on_frame;
  std::thread thread;

  explicit Impl(DecodedFrameCallback cb)
      : on_frame(std::move(cb)), thread(&Impl::ThreadMain, this) {}

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mu);
      stop_requested = true;
      queue.clear();
      reset_requested = false;
    }
    cv.notify_all();
    if (thread.joinable()) thread.join();
    CloseAllContexts();
  }

  void Submit(std::vector<DecodeRequest>&& requests) {
    if (requests.empty()) return;
    {
      std::lock_guard<std::mutex> lock(mu);
      if (stop_requested) return;
      for (auto& req : requests) {
        bool replaced = false;
        if (!req.clip_id.empty()) {
          for (auto queued_it = queue.rbegin(); queued_it != queue.rend();
               ++queued_it) {
            if (queued_it->clip_id == req.clip_id &&
                queued_it->focus_frame == req.focus_frame) {
              *queued_it = std::move(req);
              replaced = true;
              break;
            }
          }
        }

        if (!replaced) {
          if (queue.size() >= kMaxQueuedRequests) {
            queue.pop_front();
            stats.dropped_requests += 1;
          }
          queue.push_back(std::move(req));
        }
      }
      stats.submitted_requests += static_cast<uint64_t>(requests.size());
      stats.queue_depth = static_cast<uint64_t>(queue.size());
    }
    cv.notify_one();
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mu);
    stats.dropped_requests += static_cast<uint64_t>(queue.size());
    queue.clear();
    stats.queue_depth = 0;
    reset_requested = true;
    cv.notify_one();
  }

  DemuxWorkerStats GetStats() const {
    std::lock_guard<std::mutex> lock(mu);
    DemuxWorkerStats out = stats;
    out.queue_depth = static_cast<uint64_t>(queue.size());
    return out;
  }

  void ThreadMain() {
    for (;;) {
      DecodeRequest req{};
      bool has_request = false;
      bool should_reset = false;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&]() {
          return stop_requested || reset_requested || !queue.empty();
        });
        if (stop_requested) break;
        should_reset = reset_requested;
        reset_requested = false;
        if (!queue.empty()) {
          req = std::move(queue.front());
          queue.pop_front();
          stats.queue_depth = static_cast<uint64_t>(queue.size());
          has_request = true;
        }
      }

      if (should_reset) {
        ResetContextsForSeek();
      }
      if (!has_request) {
        continue;
      }

      const bool ok = ProcessRequest(req);
      {
        std::lock_guard<std::mutex> lock(mu);
        stats.processed_requests += 1;
        if (ok) {
          stats.succeeded_requests += 1;
        } else {
          stats.failed_requests += 1;
        }
      }
    }

    std::lock_guard<std::mutex> lock(mu);
    stats.dropped_requests += static_cast<uint64_t>(queue.size());
    queue.clear();
    stats.queue_depth = 0;
  }

  void ResetContextsForSeek() {
    for (auto& entry : contexts) {
      entry.second.has_position = false;
      entry.second.last_source_frame = 0;
      if (entry.second.codec_ctx) {
        avcodec_flush_buffers(entry.second.codec_ctx);
      }
    }
  }

  bool ProcessRequest(const DecodeRequest& req) {
    if (req.clip_kind != static_cast<uint32_t>(ClipKind::kVideo)) {
      return false;
    }
    if (req.media_path.empty()) return false;
    MediaContext* ctx = GetOrOpenContext(req);
    if (!ctx) return false;

    DecodedVideoFrame decoded{};
    const bool ok = SeekAndDecode(*ctx, req, &decoded);
    if (ok && on_frame) {
      on_frame(std::move(decoded));
    }
    return ok;
  }

  MediaContext* GetOrOpenContext(const DecodeRequest& req) {
    if (req.media_path.empty()) return nullptr;
    const std::string context_key =
        req.clip_id.empty() ? req.media_path : (req.clip_id + "|" + req.media_path);
    auto it = contexts.find(context_key);
    if (it != contexts.end()) return &it->second;

    const std::string& media_path = req.media_path;

    AVFormatContext* format_ctx = nullptr;
    if (avformat_open_input(&format_ctx, media_path.c_str(), nullptr, nullptr) <
        0) {
      return nullptr;
    }
    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    const int video_stream_index =
        av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index < 0) {
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    AVStream* stream = format_ctx->streams[video_stream_index];
    if (!stream) {
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
      avformat_close_input(&format_ctx);
      return nullptr;
    }
    if (avcodec_parameters_to_context(codec_ctx, stream->codecpar) < 0) {
      avcodec_free_context(&codec_ctx);
      avformat_close_input(&format_ctx);
      return nullptr;
    }
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
      avcodec_free_context(&codec_ctx);
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    MediaContext ctx{};
    ctx.format_ctx = format_ctx;
    ctx.codec_ctx = codec_ctx;
    ctx.video_stream_index = video_stream_index;
    ctx.time_base = stream->time_base;
    ctx.fps = RationalToFps(stream->avg_frame_rate, 0.0);
    ctx.fps = RationalToFps(stream->r_frame_rate, ctx.fps > 0.0 ? ctx.fps : 30.0);

    auto [inserted_it, inserted] = contexts.emplace(context_key, ctx);
    if (!inserted) {
      avcodec_free_context(&codec_ctx);
      avformat_close_input(&format_ctx);
      return &inserted_it->second;
    }
    return &inserted_it->second;
  }

  bool SeekAndDecode(MediaContext& ctx,
                     const DecodeRequest& req,
                     DecodedVideoFrame* out) {
    if (!ctx.format_ctx || !ctx.codec_ctx || ctx.video_stream_index < 0) {
      return false;
    }

    const int64_t target_pts = FrameToStreamPts(ctx, req.source_frame);
    const double safe_fps = req.timeline_fps > 0.0
                                ? req.timeline_fps
                                : (ctx.fps > 0.0 ? ctx.fps : 30.0);
    const int64_t sequential_threshold_frames =
        static_cast<int64_t>(std::llround(safe_fps * 2.0));
    const bool need_seek =
        !ctx.has_position || req.source_frame < ctx.last_source_frame ||
        (req.source_frame - ctx.last_source_frame) > sequential_threshold_frames;

    if (need_seek) {
      if (av_seek_frame(ctx.format_ctx, ctx.video_stream_index, target_pts,
                        AVSEEK_FLAG_BACKWARD) < 0) {
        return false;
      }
      avcodec_flush_buffers(ctx.codec_ctx);
    }

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    if (!packet || !frame) {
      if (packet) av_packet_free(&packet);
      if (frame) av_frame_free(&frame);
      return false;
    }

    bool decoded = false;
    constexpr int kPacketReadBudget = 300;
    int packet_budget = kPacketReadBudget;

    while (packet_budget-- > 0 && av_read_frame(ctx.format_ctx, packet) >= 0) {
      if (packet->stream_index != ctx.video_stream_index) {
        av_packet_unref(packet);
        continue;
      }

      const int send_rc = avcodec_send_packet(ctx.codec_ctx, packet);
      av_packet_unref(packet);
      if (send_rc < 0 && send_rc != AVERROR(EAGAIN)) {
        break;
      }

      for (;;) {
        const int recv_rc = avcodec_receive_frame(ctx.codec_ctx, frame);
        if (recv_rc == AVERROR(EAGAIN) || recv_rc == AVERROR_EOF) {
          break;
        }
        if (recv_rc < 0) {
          packet_budget = -1;
          break;
        }

        int64_t frame_pts = frame->best_effort_timestamp;
        if (frame_pts == AV_NOPTS_VALUE) {
          frame_pts = frame->pts;
        }

        if (need_seek && frame_pts != AV_NOPTS_VALUE && frame_pts < target_pts) {
          av_frame_unref(frame);
          continue;
        }

        if (!PopulateDecodedVideoFrame(req, frame, frame_pts, out)) {
          av_frame_unref(frame);
          packet_budget = -1;
          break;
        }

        decoded = true;
        av_frame_unref(frame);

        if (frame_pts == AV_NOPTS_VALUE || frame_pts >= target_pts) {
          packet_budget = -1;
          break;
        }
      }

      if (decoded && packet_budget < 0) {
        break;
      }
    }

    av_packet_free(&packet);
    av_frame_free(&frame);
    if (decoded) {
      ctx.last_source_frame = req.source_frame;
      ctx.has_position = true;
    }
    return decoded;
  }

  void CloseAllContexts() {
    for (auto& entry : contexts) {
      auto& ctx = entry.second;
      if (ctx.codec_ctx) {
        avcodec_free_context(&ctx.codec_ctx);
      }
      if (ctx.format_ctx) {
        avformat_close_input(&ctx.format_ctx);
      }
    }
    contexts.clear();
  }
};

DemuxWorker::DemuxWorker(DecodedFrameCallback on_frame)
    : impl_(std::make_unique<Impl>(std::move(on_frame))) {}
DemuxWorker::~DemuxWorker() = default;

void DemuxWorker::Submit(std::vector<DecodeRequest>&& requests) {
  impl_->Submit(std::move(requests));
}

void DemuxWorker::Reset() { impl_->Reset(); }

DemuxWorkerStats DemuxWorker::Stats() const { return impl_->GetStats(); }

}  // namespace media_native
