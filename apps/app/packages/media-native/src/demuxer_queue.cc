#include "demuxer_queue.h"

#include <cmath>
#include <unordered_set>
#include <utility>

namespace media_native {

namespace {

bool IsDecodeEligibleClipKind(uint32_t clip_kind) {
  const auto kind = static_cast<ClipKind>(clip_kind);
  switch (kind) {
    case ClipKind::kVideo:
    case ClipKind::kImage:
    case ClipKind::kAudio:
      return true;
    default:
      return false;
  }
}

bool IsClipActiveAtFrame(const ClipRecord& clip, int64_t focus_frame) {
  if (!clip.visible) return false;
  if (clip.end_frame <= clip.start_frame) return false;
  return focus_frame >= clip.start_frame && focus_frame < clip.end_frame;
}

int64_t ComputeSourceFrameForClip(const ClipRecord& clip, int64_t focus_frame) {
  const int64_t timeline_offset = focus_frame - clip.start_frame;
  const double safe_speed = clip.speed > 0.0 ? clip.speed : 1.0;
  const double scaled = std::floor(static_cast<double>(timeline_offset) * safe_speed);
  const int64_t source_offset = scaled >= 0.0 ? static_cast<int64_t>(scaled) : 0;
  const int64_t source_frame = clip.trim_start + source_offset;
  return source_frame > 0 ? source_frame : 0;
}

}  // namespace

void DemuxerQueue::Reset() {
  active_clip_ids_.clear();
  pending_decode_requests_.clear();
  last_requested_source_frame_.clear();
  last_evaluated_focus_frame_ = -1;
}

void DemuxerQueue::EvaluateAtTime(
    const std::unordered_map<std::string, ClipRecord>& clips,
    int64_t focus_frame,
    double timeline_fps) {
  active_clip_ids_.clear();

  if (clips.empty()) {
    pending_decode_requests_.clear();
    last_requested_source_frame_.clear();
    last_evaluated_focus_frame_ = focus_frame;
    return;
  }

  std::unordered_set<std::string> seen_active;
  seen_active.reserve(clips.size());

  for (const auto& clip_entry : clips) {
    const ClipRecord& clip = clip_entry.second;
    if (!IsClipActiveAtFrame(clip, focus_frame)) continue;

    seen_active.insert(clip.clip_id);
    active_clip_ids_.push_back(clip.clip_id);

    if (!IsDecodeEligibleClipKind(clip.clip_kind)) continue;
    if (clip.media_path.empty()) continue;

    const int64_t source_frame = ComputeSourceFrameForClip(clip, focus_frame);
    const auto last_request_it = last_requested_source_frame_.find(clip.clip_id);
    const bool same_request =
        last_request_it != last_requested_source_frame_.end() &&
        last_request_it->second == source_frame;
    if (same_request) {
      continue;
    }

    DecodeRequest req{};
    req.clip_id = clip.clip_id;
    req.clip_kind = clip.clip_kind;
    req.media_path = clip.media_path;
    req.focus_frame = focus_frame;
    req.source_frame = source_frame;
    req.timeline_fps = timeline_fps > 0.0 ? timeline_fps : 24.0;
    req.z_index = clip.z_index;
    pending_decode_requests_[clip.clip_id] = std::move(req);
    last_requested_source_frame_[clip.clip_id] = source_frame;
    total_decode_requests_enqueued_ += 1;
  }

  for (auto it = pending_decode_requests_.begin();
       it != pending_decode_requests_.end();) {
    if (seen_active.find(it->first) == seen_active.end()) {
      it = pending_decode_requests_.erase(it);
      continue;
    }
    ++it;
  }

  for (auto it = last_requested_source_frame_.begin();
       it != last_requested_source_frame_.end();) {
    if (clips.find(it->first) == clips.end()) {
      it = last_requested_source_frame_.erase(it);
      continue;
    }
    ++it;
  }

  last_evaluated_focus_frame_ = focus_frame;
}

void DemuxerQueue::RemoveClip(const std::string& clip_id) {
  pending_decode_requests_.erase(clip_id);
  last_requested_source_frame_.erase(clip_id);
}

std::vector<DecodeRequest> DemuxerQueue::DrainPendingDecodeRequests() {
  std::vector<DecodeRequest> drained;
  drained.reserve(pending_decode_requests_.size());
  for (auto& entry : pending_decode_requests_) {
    drained.push_back(std::move(entry.second));
  }
  pending_decode_requests_.clear();
  return drained;
}

size_t DemuxerQueue::ActiveClipCount() const { return active_clip_ids_.size(); }

size_t DemuxerQueue::PendingDecodeCount() const {
  return pending_decode_requests_.size();
}

uint64_t DemuxerQueue::TotalDecodeRequestsEnqueued() const {
  return total_decode_requests_enqueued_;
}

}  // namespace media_native
