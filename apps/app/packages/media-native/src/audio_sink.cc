#include "audio_sink.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

#if defined(MEDIA_NATIVE_USE_SDL)
#include <SDL.h>
#endif

namespace media_native {

namespace {

constexpr int kDesiredSampleRate = 48000;
constexpr int kDesiredChannels = 2;
constexpr uint64_t kMaxQueuedSamples =
    static_cast<uint64_t>(kDesiredSampleRate) * static_cast<uint64_t>(kDesiredChannels) * 4ULL;

}  // namespace

struct AudioSink::Impl {
  mutable std::mutex mu;
  std::deque<float> queue;
  AudioSinkStats stats{};

#if defined(MEDIA_NATIVE_USE_SDL)
  SDL_AudioDeviceID device_id = 0;
  bool owns_audio_subsystem = false;
#endif

  bool Start() {
#if !defined(MEDIA_NATIVE_USE_SDL)
    return false;
#else
    if (stats.started) return true;

    if ((SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO) == 0) {
      if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
        return false;
      }
      owns_audio_subsystem = true;
    }

    SDL_AudioSpec desired{};
    desired.freq = kDesiredSampleRate;
    desired.format = AUDIO_F32SYS;
    desired.channels = static_cast<Uint8>(kDesiredChannels);
    desired.samples = 1024;
    desired.callback = &Impl::AudioCallback;
    desired.userdata = this;

    SDL_AudioSpec obtained{};
    device_id = SDL_OpenAudioDevice(nullptr, 0, &desired, &obtained, 0);
    if (device_id == 0) {
      if (owns_audio_subsystem) {
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        owns_audio_subsystem = false;
      }
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mu);
      stats.started = true;
      stats.playing = false;
      stats.sample_rate = obtained.freq;
      stats.channels = static_cast<int32_t>(obtained.channels);
    }

    SDL_PauseAudioDevice(device_id, 1);
    return true;
#endif
  }

  void Stop() {
#if defined(MEDIA_NATIVE_USE_SDL)
    if (device_id != 0) {
      SDL_PauseAudioDevice(device_id, 1);
      SDL_CloseAudioDevice(device_id);
      device_id = 0;
    }
    if (owns_audio_subsystem) {
      SDL_QuitSubSystem(SDL_INIT_AUDIO);
      owns_audio_subsystem = false;
    }
#endif
    std::lock_guard<std::mutex> lock(mu);
    queue.clear();
    stats.started = false;
    stats.playing = false;
    stats.queued_samples = 0;
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mu);
    queue.clear();
    stats.queued_samples = 0;
    stats.submitted_blocks = 0;
    stats.submitted_samples = 0;
    stats.consumed_samples = 0;
    stats.dropped_samples = 0;
    stats.playback_seconds = 0.0;
  }

  void SetPlaying(bool is_playing) {
#if defined(MEDIA_NATIVE_USE_SDL)
    if (device_id != 0) {
      SDL_PauseAudioDevice(device_id, is_playing ? 0 : 1);
    }
#endif
    std::lock_guard<std::mutex> lock(mu);
    stats.playing = is_playing;
  }

  void SubmitMixedBlock(MixedAudioBlock&& block) {
    if (block.interleaved_f32.empty()) return;

    std::lock_guard<std::mutex> lock(mu);
    stats.submitted_blocks += 1;
    stats.submitted_samples += static_cast<uint64_t>(block.interleaved_f32.size());

    for (float sample : block.interleaved_f32) {
      queue.push_back(sample);
    }

    while (queue.size() > kMaxQueuedSamples) {
      queue.pop_front();
      stats.dropped_samples += 1;
    }

    stats.queued_samples = static_cast<uint64_t>(queue.size());
  }

  AudioSinkStats GetStats() const {
    std::lock_guard<std::mutex> lock(mu);
    AudioSinkStats out = stats;
    out.queued_samples = static_cast<uint64_t>(queue.size());
    const double denom =
        static_cast<double>(std::max(1, out.sample_rate * std::max(1, out.channels)));
    out.playback_seconds = static_cast<double>(out.consumed_samples) / denom;
    return out;
  }

#if defined(MEDIA_NATIVE_USE_SDL)
  static void AudioCallback(void* userdata, Uint8* stream, int len) {
    if (!userdata || !stream || len <= 0) return;
    auto* self = static_cast<Impl*>(userdata);

    const size_t sample_count = static_cast<size_t>(len) / sizeof(float);
    float* out = reinterpret_cast<float*>(stream);
    std::memset(out, 0, sample_count * sizeof(float));

    std::lock_guard<std::mutex> lock(self->mu);
    const size_t to_copy = std::min(sample_count, self->queue.size());
    for (size_t i = 0; i < to_copy; ++i) {
      out[i] = self->queue.front();
      self->queue.pop_front();
    }
    self->stats.consumed_samples += static_cast<uint64_t>(to_copy);
    self->stats.queued_samples = static_cast<uint64_t>(self->queue.size());
  }
#endif
};

AudioSink::AudioSink() : impl_(std::make_unique<Impl>()) {}
AudioSink::~AudioSink() {
  if (impl_) {
    impl_->Stop();
  }
}
AudioSink::AudioSink(AudioSink&&) noexcept = default;
AudioSink& AudioSink::operator=(AudioSink&&) noexcept = default;

bool AudioSink::Start() {
  if (!impl_) return false;
  return impl_->Start();
}

void AudioSink::Stop() {
  if (!impl_) return;
  impl_->Stop();
}

void AudioSink::Reset() {
  if (!impl_) return;
  impl_->Reset();
}

void AudioSink::SetPlaying(bool is_playing) {
  if (!impl_) return;
  impl_->SetPlaying(is_playing);
}

void AudioSink::SubmitMixedBlock(MixedAudioBlock&& block) {
  if (!impl_) return;
  impl_->SubmitMixedBlock(std::move(block));
}

AudioSinkStats AudioSink::Stats() const {
  if (!impl_) return {};
  return impl_->GetStats();
}

}  // namespace media_native
