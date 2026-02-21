#pragma once

#include <cstdint>

#include "media_types.h"

namespace media_native {

struct HoleRendererStats {
  bool attached = false;
  bool visible = false;
  uint64_t presented_frames = 0;
  uint64_t failed_frames = 0;
  int32_t width = 0;
  int32_t height = 0;
};

class HoleRenderer {
 public:
  HoleRenderer();
  ~HoleRenderer();

  HoleRenderer(const HoleRenderer&) = delete;
  HoleRenderer& operator=(const HoleRenderer&) = delete;
  HoleRenderer(HoleRenderer&&) noexcept;
  HoleRenderer& operator=(HoleRenderer&&) noexcept;

  bool AttachSurface(uintptr_t surface_handle);
  void SetRect(int x, int y, int width, int height, bool visible);
  void PresentFrame(DecodedVideoFrame&& frame);
  void Reset();
  HoleRendererStats Stats() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace media_native
