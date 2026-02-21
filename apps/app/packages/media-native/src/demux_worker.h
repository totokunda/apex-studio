#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "media_types.h"

namespace media_native {

struct DemuxWorkerStats {
  uint64_t submitted_requests = 0;
  uint64_t processed_requests = 0;
  uint64_t succeeded_requests = 0;
  uint64_t failed_requests = 0;
  uint64_t dropped_requests = 0;
  uint64_t queue_depth = 0;
};

// Background demux consumer. It receives decode requests from the timeline
// evaluator and translates source-frame targets into demuxed packet reads.
class DemuxWorker {
 public:
  using DecodedFrameCallback = std::function<void(DecodedVideoFrame&&)>;

  explicit DemuxWorker(DecodedFrameCallback on_frame = {});
  ~DemuxWorker();

  DemuxWorker(const DemuxWorker&) = delete;
  DemuxWorker& operator=(const DemuxWorker&) = delete;

  void Submit(std::vector<DecodeRequest>&& requests);
  void Reset();
  DemuxWorkerStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace media_native
