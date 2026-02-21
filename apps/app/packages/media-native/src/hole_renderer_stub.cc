#include "hole_renderer.h"

namespace media_native {

struct HoleRenderer::Impl {
  HoleRendererStats stats{};
};

HoleRenderer::HoleRenderer() : impl_(new Impl()) {}
HoleRenderer::~HoleRenderer() { delete impl_; }
HoleRenderer::HoleRenderer(HoleRenderer&& other) noexcept : impl_(other.impl_) {
  other.impl_ = nullptr;
}
HoleRenderer& HoleRenderer::operator=(HoleRenderer&& other) noexcept {
  if (this == &other) return *this;
  delete impl_;
  impl_ = other.impl_;
  other.impl_ = nullptr;
  return *this;
}

bool HoleRenderer::AttachSurface(uintptr_t) {
  if (!impl_) return false;
  impl_->stats.attached = false;
  return false;
}

void HoleRenderer::SetRect(int, int, int, int, bool) {}

void HoleRenderer::PresentFrame(DecodedVideoFrame&&) {
  if (!impl_) return;
  impl_->stats.failed_frames += 1;
}

void HoleRenderer::Reset() {
  if (!impl_) return;
  impl_->stats.visible = false;
}

HoleRendererStats HoleRenderer::Stats() const {
  if (!impl_) return {};
  return impl_->stats;
}

}  // namespace media_native
