#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace media_native {

struct NativeOwnedHandle {
  NativeOwnedHandle() = default;
  NativeOwnedHandle(void* next_ptr, void (*next_release)(void*))
      : ptr(next_ptr), release(next_release) {}

  ~NativeOwnedHandle() { Reset(); }

  NativeOwnedHandle(const NativeOwnedHandle&) = delete;
  NativeOwnedHandle& operator=(const NativeOwnedHandle&) = delete;

  NativeOwnedHandle(NativeOwnedHandle&& other) noexcept
      : ptr(other.ptr), release(other.release) {
    other.ptr = nullptr;
    other.release = nullptr;
  }

  NativeOwnedHandle& operator=(NativeOwnedHandle&& other) noexcept {
    if (this == &other) return *this;
    Reset();
    ptr = other.ptr;
    release = other.release;
    other.ptr = nullptr;
    other.release = nullptr;
    return *this;
  }

  void Reset() {
    if (ptr && release) {
      release(ptr);
    }
    ptr = nullptr;
    release = nullptr;
  }

  bool HasValue() const { return ptr != nullptr; }

  void* ptr = nullptr;
  void (*release)(void*) = nullptr;
};

enum class ClipKind : uint32_t {
  kUnknown = 0,
  kVideo = 1,
  kImage = 2,
  kModel = 3,
  kShape = 4,
  kText = 5,
  kDrawing = 6,
  kAudio = 7,
};

struct ClipRecord {
  std::string clip_id;
  uint32_t clip_kind = 0;
  std::string asset_id;
  std::string media_path;
  int64_t start_frame = 0;
  int64_t end_frame = 0;
  int64_t trim_start = 0;
  int64_t trim_end = 0;
  double speed = 1.0;
  double x = 0.0;
  double y = 0.0;
  double width = 0.0;
  double height = 0.0;
  double scale_x = 1.0;
  double scale_y = 1.0;
  double rotation_deg = 0.0;
  double opacity = 1.0;
  double corner_radius = 0.0;
  bool visible = true;
  bool has_crop = false;
  double crop_x = 0.0;
  double crop_y = 0.0;
  double crop_width = 1.0;
  double crop_height = 1.0;
  int32_t z_index = 0;
};

struct DecodeRequest {
  std::string clip_id;
  uint32_t clip_kind = 0;
  std::string media_path;
  int64_t focus_frame = 0;
  int64_t source_frame = 0;
  double timeline_fps = 24.0;
  int32_t z_index = 0;
};

struct DecodedVideoFrame {
  std::string clip_id;
  std::string media_path;
  int64_t focus_frame = 0;
  int64_t source_frame = 0;
  int64_t pts = 0;
  int32_t width = 0;
  int32_t height = 0;
  int32_t pixel_format = 0;
  int32_t z_index = 0;
  NativeOwnedHandle native_frame;
};

struct DecodedAudioChunk {
  std::string clip_id;
  std::string media_path;
  int64_t focus_frame = 0;
  int64_t source_frame = 0;
  int64_t pts = 0;
  int32_t sample_rate = 0;
  int32_t channels = 0;
  int32_t sample_format = 0;
  int32_t nb_samples = 0;
  bool planar = false;
  int32_t z_index = 0;
  std::vector<uint8_t> data;
};

struct MixedAudioBlock {
  int64_t focus_frame = 0;
  int32_t sample_rate = 48000;
  int32_t channels = 2;
  int32_t nb_samples = 0;
  std::vector<float> interleaved_f32;
};

}  // namespace media_native
