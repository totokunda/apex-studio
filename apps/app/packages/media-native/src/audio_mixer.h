#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "media_types.h"

namespace media_native {

struct AudioMixerStats {
  uint64_t submitted_chunks = 0;
  uint64_t mixed_blocks_enqueued = 0;
  uint64_t mixed_blocks_drained = 0;
  uint64_t dropped_chunks = 0;
  uint64_t input_queue_depth = 0;
};

class AudioMixer {
 public:
  AudioMixer();
  ~AudioMixer();

  AudioMixer(const AudioMixer&) = delete;
  AudioMixer& operator=(const AudioMixer&) = delete;
  AudioMixer(AudioMixer&&) noexcept;
  AudioMixer& operator=(AudioMixer&&) noexcept;

  void SubmitChunk(DecodedAudioChunk&& chunk);
  std::vector<MixedAudioBlock> DrainMixedBlocks();
  void Reset();
  AudioMixerStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace media_native
