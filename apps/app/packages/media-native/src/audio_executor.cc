#include "audio_executor.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/rational.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace media_native {

namespace {

constexpr int kOutputSampleRate = 48000;
constexpr int kOutputChannels = 2;
constexpr AVSampleFormat kOutputSampleFormat = AV_SAMPLE_FMT_FLT;
constexpr size_t kMaxQueuedRequests = 256;

struct AudioContext {
  AVFormatContext* format_ctx = nullptr;
  AVCodecContext* codec_ctx = nullptr;
  SwrContext* swr_ctx = nullptr;
  AVChannelLayout output_layout{};
  int stream_index = -1;
  AVRational time_base{1, 1};
  int64_t last_source_frame = 0;
  bool has_position = false;
  std::deque<float> queued_interleaved_f32;
};

int64_t FrameToAudioPts(const AudioContext& ctx,
                        int64_t source_frame,
                        double timeline_fps) {
  const double safe_fps = timeline_fps > 0.0 ? timeline_fps : 24.0;
  const double micros_d =
      (static_cast<double>(source_frame) / safe_fps) * 1000000.0;
  const int64_t micros =
      micros_d > 0.0 ? static_cast<int64_t>(std::llround(micros_d)) : 0;
  return av_rescale_q(micros, AVRational{1, 1000000}, ctx.time_base);
}

bool EnsureResampler(AudioContext* ctx) {
  if (!ctx || !ctx->codec_ctx) return false;
  if (ctx->swr_ctx) return true;

  AVCodecContext* codec_ctx = ctx->codec_ctx;
  if (codec_ctx->sample_rate <= 0) return false;
  if (codec_ctx->sample_fmt == AV_SAMPLE_FMT_NONE) return false;
  if (codec_ctx->ch_layout.nb_channels <= 0) return false;

  av_channel_layout_default(&ctx->output_layout, kOutputChannels);

  SwrContext* swr_ctx = nullptr;
  const int rc = swr_alloc_set_opts2(
      &swr_ctx, &ctx->output_layout, kOutputSampleFormat, kOutputSampleRate,
      &codec_ctx->ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate, 0,
      nullptr);
  if (rc < 0 || !swr_ctx) {
    if (swr_ctx) swr_free(&swr_ctx);
    av_channel_layout_uninit(&ctx->output_layout);
    return false;
  }

  if (swr_init(swr_ctx) < 0) {
    swr_free(&swr_ctx);
    av_channel_layout_uninit(&ctx->output_layout);
    return false;
  }

  ctx->swr_ctx = swr_ctx;
  return true;
}

bool PopulateDecodedAudioChunk(AudioContext& ctx,
                               const DecodeRequest& req,
                               const AVFrame* frame,
                               int64_t frame_pts,
                               DecodedAudioChunk* out) {
  if (!frame || !out) return false;
  if (!ctx.swr_ctx) return false;
  if (frame->nb_samples <= 0) return false;

  const int64_t delayed = swr_get_delay(ctx.swr_ctx, frame->sample_rate);
  const int dst_nb_samples = static_cast<int>(av_rescale_rnd(
      delayed + frame->nb_samples, kOutputSampleRate, frame->sample_rate,
      AV_ROUND_UP));
  if (dst_nb_samples <= 0) return false;

  std::vector<float> converted(
      static_cast<size_t>(dst_nb_samples) * static_cast<size_t>(kOutputChannels));

  uint8_t* out_data[1] = {
      reinterpret_cast<uint8_t*>(converted.data()),
  };

  const int produced_samples =
      swr_convert(ctx.swr_ctx, out_data, dst_nb_samples,
                  const_cast<const uint8_t**>(frame->extended_data),
                  frame->nb_samples);
  if (produced_samples <= 0) {
    return false;
  }

  const size_t produced_floats =
      static_cast<size_t>(produced_samples) * static_cast<size_t>(kOutputChannels);
  converted.resize(produced_floats);

  out->clip_id = req.clip_id;
  out->media_path = req.media_path;
  out->focus_frame = req.focus_frame;
  out->source_frame = req.source_frame;
  out->pts = frame_pts == AV_NOPTS_VALUE ? 0 : frame_pts;
  out->sample_rate = kOutputSampleRate;
  out->channels = kOutputChannels;
  out->sample_format = static_cast<int32_t>(kOutputSampleFormat);
  out->nb_samples = produced_samples;
  out->planar = false;
  out->z_index = req.z_index;
  out->data.resize(produced_floats * sizeof(float));
  std::memcpy(out->data.data(), converted.data(), out->data.size());
  return true;
}

}  // namespace

struct AudioExecutor::Impl {
  mutable std::mutex mu;
  std::condition_variable cv;
  bool stop_requested = false;
  bool reset_requested = false;
  std::deque<DecodeRequest> queue;
  std::unordered_map<std::string, AudioContext> contexts;
  std::unordered_set<std::string> no_audio_media_paths;
  AudioExecutorStats stats{};
  DecodedChunkCallback on_chunk;
  std::thread thread;

  explicit Impl(DecodedChunkCallback cb)
      : on_chunk(std::move(cb)), thread(&Impl::ThreadMain, this) {}

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

  AudioExecutorStats GetStats() const {
    std::lock_guard<std::mutex> lock(mu);
    AudioExecutorStats out = stats;
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
      entry.second.queued_interleaved_f32.clear();
      if (entry.second.codec_ctx) {
        avcodec_flush_buffers(entry.second.codec_ctx);
      }
      if (entry.second.swr_ctx) {
        swr_close(entry.second.swr_ctx);
        swr_init(entry.second.swr_ctx);
      }
    }
  }

  bool ProcessRequest(const DecodeRequest& req) {
    const auto kind = static_cast<ClipKind>(req.clip_kind);
    if (kind != ClipKind::kAudio && kind != ClipKind::kVideo) {
      return false;
    }
    if (req.media_path.empty()) return false;

    AudioContext* ctx = GetOrOpenContext(req);
    if (!ctx) return false;
    if (!EnsureResampler(ctx)) return false;

    DecodedAudioChunk decoded{};
    const bool ok = DecodeForRequest(*ctx, req, &decoded);
    if (ok && on_chunk) {
      on_chunk(std::move(decoded));
    }
    return ok;
  }

  AudioContext* GetOrOpenContext(const DecodeRequest& req) {
    if (req.media_path.empty()) return nullptr;
    const std::string& media_path = req.media_path;
    if (no_audio_media_paths.find(media_path) != no_audio_media_paths.end()) {
      return nullptr;
    }

    const std::string context_key =
        req.clip_id.empty() ? media_path : (req.clip_id + "|" + media_path);
    auto it = contexts.find(context_key);
    if (it != contexts.end()) return &it->second;

    AVFormatContext* format_ctx = nullptr;
    if (avformat_open_input(&format_ctx, media_path.c_str(), nullptr, nullptr) <
        0) {
      no_audio_media_paths.insert(media_path);
      return nullptr;
    }
    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
      no_audio_media_paths.insert(media_path);
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    const int stream_index =
        av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream_index < 0) {
      no_audio_media_paths.insert(media_path);
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    AVStream* stream = format_ctx->streams[stream_index];
    if (!stream) {
      no_audio_media_paths.insert(media_path);
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
      no_audio_media_paths.insert(media_path);
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
      no_audio_media_paths.insert(media_path);
      avformat_close_input(&format_ctx);
      return nullptr;
    }
    if (avcodec_parameters_to_context(codec_ctx, stream->codecpar) < 0) {
      no_audio_media_paths.insert(media_path);
      avcodec_free_context(&codec_ctx);
      avformat_close_input(&format_ctx);
      return nullptr;
    }
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
      no_audio_media_paths.insert(media_path);
      avcodec_free_context(&codec_ctx);
      avformat_close_input(&format_ctx);
      return nullptr;
    }

    AudioContext ctx{};
    ctx.format_ctx = format_ctx;
    ctx.codec_ctx = codec_ctx;
    ctx.stream_index = stream_index;
    ctx.time_base = stream->time_base;

    auto [inserted_it, inserted] = contexts.emplace(context_key, std::move(ctx));
    if (!inserted) {
      avcodec_free_context(&codec_ctx);
      avformat_close_input(&format_ctx);
      return &inserted_it->second;
    }
    return &inserted_it->second;
  }

  bool DecodeForRequest(AudioContext& ctx,
                        const DecodeRequest& req,
                        DecodedAudioChunk* out) {
    if (!ctx.format_ctx || !ctx.codec_ctx || ctx.stream_index < 0) return false;

    const int64_t target_pts =
        FrameToAudioPts(ctx, req.source_frame, req.timeline_fps);
    const double safe_fps = req.timeline_fps > 0.0 ? req.timeline_fps : 24.0;
    const int target_samples_per_channel = std::max(
        1, static_cast<int>(std::llround(
               static_cast<double>(kOutputSampleRate) / safe_fps)));
    const size_t target_interleaved_samples =
        static_cast<size_t>(target_samples_per_channel) *
        static_cast<size_t>(kOutputChannels);
    const int64_t sequential_threshold_frames =
        static_cast<int64_t>(std::llround(safe_fps * 2.0));

    const bool need_seek =
        !ctx.has_position || req.source_frame < ctx.last_source_frame ||
        (req.source_frame - ctx.last_source_frame) > sequential_threshold_frames;

    if (need_seek) {
      if (av_seek_frame(ctx.format_ctx, ctx.stream_index, target_pts,
                        AVSEEK_FLAG_BACKWARD) < 0) {
        return false;
      }
      avcodec_flush_buffers(ctx.codec_ctx);
      if (ctx.swr_ctx) {
        swr_close(ctx.swr_ctx);
        swr_init(ctx.swr_ctx);
      }
      ctx.queued_interleaved_f32.clear();
    }

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    if (!packet || !frame) {
      if (packet) av_packet_free(&packet);
      if (frame) av_frame_free(&frame);
      return false;
    }

    bool decoded_any = false;
    int64_t first_decoded_pts = target_pts;
    bool has_decoded_pts = false;
    constexpr int kPacketReadBudget = 400;
    int packet_budget = kPacketReadBudget;

    while (ctx.queued_interleaved_f32.size() < target_interleaved_samples &&
           packet_budget-- > 0 && av_read_frame(ctx.format_ctx, packet) >= 0) {
      if (packet->stream_index != ctx.stream_index) {
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

        DecodedAudioChunk decoded_chunk{};
        if (!PopulateDecodedAudioChunk(ctx, req, frame, frame_pts,
                                       &decoded_chunk)) {
          av_frame_unref(frame);
          packet_budget = -1;
          break;
        }

        if (!has_decoded_pts) {
          first_decoded_pts = decoded_chunk.pts;
          has_decoded_pts = true;
        }
        decoded_any = true;

        const size_t float_count =
            decoded_chunk.data.size() / static_cast<size_t>(sizeof(float));
        const float* src =
            reinterpret_cast<const float*>(decoded_chunk.data.data());
        for (size_t i = 0; i < float_count; ++i) {
          ctx.queued_interleaved_f32.push_back(src[i]);
        }

        av_frame_unref(frame);
        if (ctx.queued_interleaved_f32.size() >= target_interleaved_samples) {
          packet_budget = -1;
          break;
        }
      }

      if (packet_budget < 0) {
        break;
      }
    }

    av_packet_free(&packet);
    av_frame_free(&frame);

    if (ctx.queued_interleaved_f32.empty()) {
      return false;
    }

    const size_t copy_count =
        std::min(target_interleaved_samples, ctx.queued_interleaved_f32.size());
    std::vector<float> block(target_interleaved_samples, 0.0f);
    for (size_t i = 0; i < copy_count; ++i) {
      block[i] = ctx.queued_interleaved_f32.front();
      ctx.queued_interleaved_f32.pop_front();
    }

    out->clip_id = req.clip_id;
    out->media_path = req.media_path;
    out->focus_frame = req.focus_frame;
    out->source_frame = req.source_frame;
    out->pts = has_decoded_pts ? first_decoded_pts : target_pts;
    out->sample_rate = kOutputSampleRate;
    out->channels = kOutputChannels;
    out->sample_format = static_cast<int32_t>(kOutputSampleFormat);
    out->nb_samples = target_samples_per_channel;
    out->planar = false;
    out->z_index = req.z_index;
    out->data.resize(block.size() * sizeof(float));
    std::memcpy(out->data.data(), block.data(), out->data.size());

    if (decoded_any || copy_count > 0) {
      ctx.last_source_frame = req.source_frame;
      ctx.has_position = true;
    }
    return true;
  }

  void CloseAllContexts() {
    for (auto& entry : contexts) {
      auto& ctx = entry.second;
      if (ctx.swr_ctx) {
        swr_free(&ctx.swr_ctx);
      }
      av_channel_layout_uninit(&ctx.output_layout);
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

AudioExecutor::AudioExecutor(DecodedChunkCallback on_chunk)
    : impl_(std::make_unique<Impl>(std::move(on_chunk))) {}
AudioExecutor::~AudioExecutor() = default;

void AudioExecutor::Submit(std::vector<DecodeRequest>&& requests) {
  impl_->Submit(std::move(requests));
}

void AudioExecutor::Reset() { impl_->Reset(); }

AudioExecutorStats AudioExecutor::Stats() const { return impl_->GetStats(); }

}  // namespace media_native
