#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "media_types.h"

namespace media_native {

struct AudioExecutorStats {
  uint64_t submitted_requests = 0;
  uint64_t processed_requests = 0;
  uint64_t succeeded_requests = 0;
  uint64_t failed_requests = 0;
  uint64_t dropped_requests = 0;
  uint64_t queue_depth = 0;
};

class AudioExecutor {
 public:
  using DecodedChunkCallback = std::function<void(DecodedAudioChunk&&)>;

  explicit AudioExecutor(DecodedChunkCallback on_chunk = {});
  ~AudioExecutor();

  AudioExecutor(const AudioExecutor&) = delete;
  AudioExecutor& operator=(const AudioExecutor&) = delete;

  void Submit(std::vector<DecodeRequest>&& requests);
  void Reset();
  AudioExecutorStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace media_native
