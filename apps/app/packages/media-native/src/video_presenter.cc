#include "video_presenter.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <utility>

namespace media_native {

namespace {

constexpr size_t kMaxQueuedFrames = 128;
constexpr int64_t kStaleFrameWindow = 2;

}  // namespace

struct VideoPresenter::Impl {
  mutable std::mutex mu;
  std::deque<DecodedVideoFrame> queue;
  DecodedVideoFrame presented_frame;
  bool has_presented_frame_payload = false;
  VideoPresenterStats stats{};

  void SubmitFrame(DecodedVideoFrame&& frame) {
    std::lock_guard<std::mutex> lock(mu);
    if (queue.size() >= kMaxQueuedFrames) {
      queue.pop_front();
      stats.dropped_frames += 1;
    }
    queue.push_back(std::move(frame));
    stats.queued_frames = static_cast<uint64_t>(queue.size());
  }

  void PresentForTargetFrame(int64_t target_focus_frame,
                             double audio_clock_seconds,
                             bool is_playing) {
    std::lock_guard<std::mutex> lock(mu);
    stats.target_focus_frame = target_focus_frame;
    stats.audio_clock_seconds = audio_clock_seconds;

    if (queue.empty()) {
      stats.queued_frames = 0;
      return;
    }

    // Keep queue roughly bounded around the target frame.
    while (!queue.empty() &&
           queue.front().focus_frame <
               (target_focus_frame - kStaleFrameWindow)) {
      queue.pop_front();
      stats.dropped_frames += 1;
    }

    if (queue.empty()) {
      stats.queued_frames = 0;
      return;
    }

    size_t selected_index = std::numeric_limits<size_t>::max();

    if (is_playing) {
      // While playing, present the latest frame not ahead of audio time.
      for (size_t i = 0; i < queue.size(); ++i) {
        if (queue[i].focus_frame <= target_focus_frame) {
          selected_index = i;
        } else {
          break;
        }
      }
      if (selected_index == std::numeric_limits<size_t>::max()) {
        selected_index = 0;
      }
    } else {
      // While paused/scrubbing, present nearest frame to requested focus frame.
      int64_t best_distance = std::numeric_limits<int64_t>::max();
      for (size_t i = 0; i < queue.size(); ++i) {
        const int64_t distance =
            std::llabs(queue[i].focus_frame - target_focus_frame);
        if (distance < best_distance) {
          best_distance = distance;
          selected_index = i;
        }
      }
      if (selected_index == std::numeric_limits<size_t>::max()) {
        selected_index = 0;
      }
    }

    // Drop any older frames prior to selected frame.
    for (size_t i = 0; i < selected_index; ++i) {
      queue.pop_front();
      stats.dropped_frames += 1;
    }

    if (queue.empty()) {
      stats.queued_frames = 0;
      return;
    }

    presented_frame = std::move(queue.front());
    has_presented_frame_payload = true;
    queue.pop_front();

    stats.has_presented_frame = true;
    stats.last_presented_focus_frame = presented_frame.focus_frame;
    stats.last_presented_source_frame = presented_frame.source_frame;
    stats.last_presented_pts = presented_frame.pts;
    stats.last_presented_width = presented_frame.width;
    stats.last_presented_height = presented_frame.height;
    stats.last_presented_pixel_format = presented_frame.pixel_format;
    stats.last_presented_z_index = presented_frame.z_index;
    stats.presented_frames += 1;
    stats.queued_frames = static_cast<uint64_t>(queue.size());
  }

  bool DrainPresentedFrame(DecodedVideoFrame* out_frame) {
    if (!out_frame) return false;
    std::lock_guard<std::mutex> lock(mu);
    if (!has_presented_frame_payload) return false;
    *out_frame = std::move(presented_frame);
    presented_frame = DecodedVideoFrame{};
    has_presented_frame_payload = false;
    return true;
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mu);
    queue.clear();
    presented_frame = DecodedVideoFrame{};
    has_presented_frame_payload = false;
    stats.has_presented_frame = false;
    stats.last_presented_focus_frame = -1;
    stats.last_presented_source_frame = -1;
    stats.last_presented_pts = 0;
    stats.last_presented_width = 0;
    stats.last_presented_height = 0;
    stats.last_presented_pixel_format = 0;
    stats.last_presented_z_index = 0;
    stats.target_focus_frame = -1;
    stats.audio_clock_seconds = 0.0;
    stats.queued_frames = 0;
  }

  VideoPresenterStats GetStats() const {
    std::lock_guard<std::mutex> lock(mu);
    VideoPresenterStats out = stats;
    out.queued_frames = static_cast<uint64_t>(queue.size());
    return out;
  }
};

VideoPresenter::VideoPresenter() : impl_(std::make_unique<Impl>()) {}
VideoPresenter::~VideoPresenter() = default;
VideoPresenter::VideoPresenter(VideoPresenter&&) noexcept = default;
VideoPresenter& VideoPresenter::operator=(VideoPresenter&&) noexcept = default;

void VideoPresenter::SubmitFrame(DecodedVideoFrame&& frame) {
  impl_->SubmitFrame(std::move(frame));
}

void VideoPresenter::PresentForTargetFrame(int64_t target_focus_frame,
                                           double audio_clock_seconds,
                                           bool is_playing) {
  impl_->PresentForTargetFrame(target_focus_frame, audio_clock_seconds,
                               is_playing);
}

bool VideoPresenter::DrainPresentedFrame(DecodedVideoFrame* out_frame) {
  return impl_->DrainPresentedFrame(out_frame);
}

void VideoPresenter::Reset() { impl_->Reset(); }

VideoPresenterStats VideoPresenter::Stats() const { return impl_->GetStats(); }

}  // namespace media_native
