#pragma once

#include <cstdint>
#include <memory>

#include "media_types.h"

namespace media_native {

struct AudioSinkStats {
  bool started = false;
  bool playing = false;
  int32_t sample_rate = 0;
  int32_t channels = 0;
  uint64_t queued_samples = 0;
  uint64_t submitted_blocks = 0;
  uint64_t submitted_samples = 0;
  uint64_t consumed_samples = 0;
  uint64_t dropped_samples = 0;
  double playback_seconds = 0.0;
};

class AudioSink {
 public:
  AudioSink();
  ~AudioSink();

  AudioSink(const AudioSink&) = delete;
  AudioSink& operator=(const AudioSink&) = delete;
  AudioSink(AudioSink&&) noexcept;
  AudioSink& operator=(AudioSink&&) noexcept;

  bool Start();
  void Stop();
  void Reset();
  void SetPlaying(bool is_playing);
  void SubmitMixedBlock(MixedAudioBlock&& block);
  AudioSinkStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace media_native
