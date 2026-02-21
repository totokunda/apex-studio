#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "media_types.h"

namespace media_native {

// Playback evaluator + demux handoff queue.
// addon.cc updates clip/timeline state; this module decides what to decode.
class DemuxerQueue {
 public:
  void Reset();

  void EvaluateAtTime(const std::unordered_map<std::string, ClipRecord>& clips,
                      int64_t focus_frame,
                      double timeline_fps);

  void RemoveClip(const std::string& clip_id);

  std::vector<DecodeRequest> DrainPendingDecodeRequests();

  size_t ActiveClipCount() const;
  size_t PendingDecodeCount() const;
  uint64_t TotalDecodeRequestsEnqueued() const;

 private:
  int64_t last_evaluated_focus_frame_ = -1;
  std::vector<std::string> active_clip_ids_;
  std::unordered_map<std::string, DecodeRequest> pending_decode_requests_;
  std::unordered_map<std::string, int64_t> last_requested_source_frame_;
  uint64_t total_decode_requests_enqueued_ = 0;
};

}  // namespace media_native
