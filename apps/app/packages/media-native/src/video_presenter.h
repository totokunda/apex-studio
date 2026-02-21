#pragma once

#include <cstdint>
#include <memory>

#include "media_types.h"

namespace media_native {

struct VideoPresenterStats {
  uint64_t queued_frames = 0;
  uint64_t presented_frames = 0;
  uint64_t dropped_frames = 0;
  bool has_presented_frame = false;
  int64_t last_presented_focus_frame = -1;
  int64_t last_presented_source_frame = -1;
  int64_t last_presented_pts = 0;
  int32_t last_presented_width = 0;
  int32_t last_presented_height = 0;
  int32_t last_presented_pixel_format = 0;
  int32_t last_presented_z_index = 0;
  int64_t target_focus_frame = -1;
  double audio_clock_seconds = 0.0;
};

class VideoPresenter {
 public:
  VideoPresenter();
  ~VideoPresenter();

  VideoPresenter(const VideoPresenter&) = delete;
  VideoPresenter& operator=(const VideoPresenter&) = delete;
  VideoPresenter(VideoPresenter&&) noexcept;
  VideoPresenter& operator=(VideoPresenter&&) noexcept;

  void SubmitFrame(DecodedVideoFrame&& frame);
  void PresentForTargetFrame(int64_t target_focus_frame,
                             double audio_clock_seconds,
                             bool is_playing);
  bool DrainPresentedFrame(DecodedVideoFrame* out_frame);
  void Reset();
  VideoPresenterStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace media_native
