#include "audio_mixer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace media_native {

namespace {

constexpr int kTargetSampleRate = 48000;
constexpr int kTargetChannels = 2;
constexpr size_t kMaxPendingChunks = 512;

float ClampToUnit(float v) {
  if (v > 1.0f) return 1.0f;
  if (v < -1.0f) return -1.0f;
  return v;
}

}  // namespace

struct AudioMixer::Impl {
  mutable std::mutex mu;
  std::deque<DecodedAudioChunk> pending_chunks;
  AudioMixerStats stats{};

  void SubmitChunk(DecodedAudioChunk&& chunk) {
    std::lock_guard<std::mutex> lock(mu);
    if (pending_chunks.size() >= kMaxPendingChunks) {
      pending_chunks.pop_front();
      stats.dropped_chunks += 1;
    }
    pending_chunks.push_back(std::move(chunk));
    stats.submitted_chunks += 1;
    stats.input_queue_depth = static_cast<uint64_t>(pending_chunks.size());
  }

  std::vector<MixedAudioBlock> DrainMixedBlocks() {
    std::deque<DecodedAudioChunk> drained_chunks;
    {
      std::lock_guard<std::mutex> lock(mu);
      drained_chunks.swap(pending_chunks);
      stats.input_queue_depth = 0;
    }

    if (drained_chunks.empty()) {
      return {};
    }

    struct MixAccumulator {
      MixedAudioBlock block;
      std::vector<uint16_t> contributor_counts;
    };
    std::map<int64_t, MixAccumulator> accumulators_by_frame;
    std::unordered_set<std::string> seen_clip_focus_pairs;
    seen_clip_focus_pairs.reserve(drained_chunks.size());

    // Traverse latest -> oldest so we keep most recent decode result per clip/frame.
    for (auto it = drained_chunks.rbegin(); it != drained_chunks.rend(); ++it) {
      auto& chunk = *it;
      if (chunk.sample_rate != kTargetSampleRate) continue;
      if (chunk.channels != kTargetChannels) continue;
      if (chunk.planar) continue;
      if (chunk.nb_samples <= 0) continue;

      if (!chunk.clip_id.empty()) {
        const std::string key =
            std::to_string(chunk.focus_frame) + "|" + chunk.clip_id;
        if (!seen_clip_focus_pairs.insert(key).second) {
          continue;
        }
      }

      const size_t sample_count =
          static_cast<size_t>(chunk.nb_samples) * static_cast<size_t>(kTargetChannels);
      const size_t expected_bytes = sample_count * sizeof(float);
      if (chunk.data.size() < expected_bytes) continue;

      auto [acc_it, inserted] =
          accumulators_by_frame.try_emplace(chunk.focus_frame, MixAccumulator{});
      MixAccumulator& acc = acc_it->second;
      MixedAudioBlock& block = acc.block;
      if (inserted) {
        block.focus_frame = chunk.focus_frame;
        block.sample_rate = kTargetSampleRate;
        block.channels = kTargetChannels;
        block.nb_samples = chunk.nb_samples;
        block.interleaved_f32.assign(sample_count, 0.0f);
        acc.contributor_counts.assign(sample_count, 0);
      } else if (chunk.nb_samples > block.nb_samples) {
        const size_t next_sample_count =
            static_cast<size_t>(chunk.nb_samples) * static_cast<size_t>(kTargetChannels);
        block.nb_samples = chunk.nb_samples;
        block.interleaved_f32.resize(next_sample_count, 0.0f);
        acc.contributor_counts.resize(next_sample_count, 0);
      }

      const float* src = reinterpret_cast<const float*>(chunk.data.data());
      const size_t mix_count =
          std::min(sample_count, block.interleaved_f32.size());
      for (size_t i = 0; i < mix_count; ++i) {
        block.interleaved_f32[i] += src[i];
        if (acc.contributor_counts[i] < std::numeric_limits<uint16_t>::max()) {
          acc.contributor_counts[i] += 1;
        }
      }
    }

    std::vector<MixedAudioBlock> out;
    out.reserve(accumulators_by_frame.size());
    for (auto& [_, acc] : accumulators_by_frame) {
      MixedAudioBlock& block = acc.block;
      for (size_t i = 0; i < block.interleaved_f32.size(); ++i) {
        const uint16_t contributors = acc.contributor_counts[i];
        if (contributors > 1) {
          block.interleaved_f32[i] /= static_cast<float>(contributors);
        }
      }
      for (float& sample : block.interleaved_f32) {
        sample = ClampToUnit(sample);
      }
      out.push_back(std::move(block));
    }

    {
      std::lock_guard<std::mutex> lock(mu);
      stats.mixed_blocks_enqueued += static_cast<uint64_t>(out.size());
      stats.mixed_blocks_drained += static_cast<uint64_t>(out.size());
    }

    return out;
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mu);
    pending_chunks.clear();
    stats.input_queue_depth = 0;
  }

  AudioMixerStats GetStats() const {
    std::lock_guard<std::mutex> lock(mu);
    AudioMixerStats out = stats;
    out.input_queue_depth = static_cast<uint64_t>(pending_chunks.size());
    return out;
  }
};

AudioMixer::AudioMixer() : impl_(std::make_unique<Impl>()) {}
AudioMixer::~AudioMixer() = default;
AudioMixer::AudioMixer(AudioMixer&&) noexcept = default;
AudioMixer& AudioMixer::operator=(AudioMixer&&) noexcept = default;

void AudioMixer::SubmitChunk(DecodedAudioChunk&& chunk) {
  impl_->SubmitChunk(std::move(chunk));
}

std::vector<MixedAudioBlock> AudioMixer::DrainMixedBlocks() {
  return impl_->DrainMixedBlocks();
}

void AudioMixer::Reset() { impl_->Reset(); }

AudioMixerStats AudioMixer::Stats() const { return impl_->GetStats(); }

}  // namespace media_native
