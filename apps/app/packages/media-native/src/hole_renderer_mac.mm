#include "hole_renderer.h"

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#import <dispatch/dispatch.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

namespace media_native {

namespace {

void RunOnMainSync(const std::function<void()>& fn) {
  if ([NSThread isMainThread]) {
    fn();
    return;
  }
  dispatch_sync(dispatch_get_main_queue(), ^{
    fn();
  });
}

bool ResolveHostView(uintptr_t raw_handle, NSView*& out_host_view) {
  out_host_view = nil;
  if (raw_handle == 0) return false;

  id candidate = (__bridge id)(reinterpret_cast<void*>(raw_handle));
  if ([candidate isKindOfClass:[NSWindow class]]) {
    NSWindow* window = static_cast<NSWindow*>(candidate);
    out_host_view = window.contentView;
  } else if ([candidate isKindOfClass:[NSView class]]) {
    NSView* view = static_cast<NSView*>(candidate);
    out_host_view = view.window.contentView ? view.window.contentView : view;
  }

  return out_host_view != nil;
}

CGImageRef CreateImageFromBGRA(const std::vector<uint8_t>& bgra,
                               int width,
                               int height,
                               int stride) {
  if (bgra.empty() || width <= 0 || height <= 0 || stride <= 0) {
    return nullptr;
  }

  CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
  if (!color_space) return nullptr;

  CFDataRef data_ref = CFDataCreate(
      kCFAllocatorDefault, reinterpret_cast<const UInt8*>(bgra.data()),
      static_cast<CFIndex>(bgra.size()));
  if (!data_ref) {
    CGColorSpaceRelease(color_space);
    return nullptr;
  }

  CGDataProviderRef provider = CGDataProviderCreateWithCFData(data_ref);
  CFRelease(data_ref);
  if (!provider) {
    CGColorSpaceRelease(color_space);
    return nullptr;
  }

  const CGBitmapInfo bitmap_info =
      kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
  CGImageRef image = CGImageCreate(
      width, height, 8, 32, static_cast<size_t>(stride), color_space,
      bitmap_info, provider, nullptr, false, kCGRenderingIntentDefault);

  CGDataProviderRelease(provider);
  CGColorSpaceRelease(color_space);
  return image;
}

}  // namespace

struct HoleRenderer::Impl {
  mutable std::mutex mu;
  HoleRendererStats stats{};

  NSView* host_view = nil;
  CALayer* hole_layer = nil;
  CALayer* video_layer = nil;

  SwsContext* sws_ctx = nullptr;
  int sws_width = 0;
  int sws_height = 0;
  int sws_format = -1;

  void EnsureLayersCreated() {
    if (!host_view) return;

    [host_view setWantsLayer:YES];
    if (!host_view.layer) {
      return;
    }

    if (!hole_layer) {
      hole_layer = [CALayer layer];
      hole_layer.frame = CGRectMake(0, 0, 1, 1);
      hole_layer.backgroundColor = NSColor.blackColor.CGColor;
      hole_layer.hidden = YES;
      [host_view.layer insertSublayer:hole_layer atIndex:0];
    }

    if (!video_layer) {
      video_layer = [CALayer layer];
      video_layer.frame = hole_layer.bounds;
      video_layer.hidden = YES;
      video_layer.contentsGravity = kCAGravityResizeAspectFill;
      video_layer.contentsScale = NSScreen.mainScreen.backingScaleFactor > 0
                                      ? NSScreen.mainScreen.backingScaleFactor
                                      : 1.0;
      [hole_layer addSublayer:video_layer];
    }
  }

  bool AttachSurface(uintptr_t surface_handle) {
    NSView* resolved_host = nil;
    if (!ResolveHostView(surface_handle, resolved_host)) {
      std::lock_guard<std::mutex> lock(mu);
      stats.attached = false;
      return false;
    }

    RunOnMainSync([&]() {
      host_view = resolved_host;
      EnsureLayersCreated();
      if (hole_layer && host_view.layer) {
        [hole_layer removeFromSuperlayer];
        [host_view.layer insertSublayer:hole_layer atIndex:0];
      }
    });

    std::lock_guard<std::mutex> lock(mu);
    stats.attached = true;
    return true;
  }

  void SetRect(int x, int y, int width, int height, bool visible) {
    RunOnMainSync([&]() {
      if (!host_view) return;
      EnsureLayersCreated();
      if (!hole_layer) return;

      if (!visible) {
        hole_layer.hidden = YES;
        if (video_layer) video_layer.hidden = YES;
        return;
      }

      const CGFloat nx = static_cast<CGFloat>(x);
      const CGFloat top_y = static_cast<CGFloat>(y);
      const CGFloat nw = std::max(1, width);
      const CGFloat nh = std::max(1, height);

      const CGFloat ny = host_view.isFlipped
                             ? top_y
                             : host_view.bounds.size.height - top_y - nh;
      hole_layer.frame = CGRectMake(nx, ny, nw, nh);
      if (video_layer) {
        video_layer.frame = hole_layer.bounds;
      }
      hole_layer.hidden = NO;
      if (video_layer) {
        video_layer.hidden = NO;
      }
      if (host_view.layer) {
        [hole_layer removeFromSuperlayer];
        [host_view.layer insertSublayer:hole_layer atIndex:0];
      }
    });

    std::lock_guard<std::mutex> lock(mu);
    stats.visible = visible;
  }

  void PresentFrame(DecodedVideoFrame&& frame) {
    AVFrame* av_frame = static_cast<AVFrame*>(frame.native_frame.ptr);
    if (!av_frame) {
      std::lock_guard<std::mutex> lock(mu);
      stats.failed_frames += 1;
      return;
    }

    const int width = av_frame->width;
    const int height = av_frame->height;
    const int src_format = av_frame->format;
    if (width <= 0 || height <= 0 || src_format < 0) {
      std::lock_guard<std::mutex> lock(mu);
      stats.failed_frames += 1;
      return;
    }

    if (!sws_ctx || sws_width != width || sws_height != height ||
        sws_format != src_format) {
      sws_ctx = sws_getCachedContext(
          sws_ctx, width, height, static_cast<AVPixelFormat>(src_format), width,
          height, AV_PIX_FMT_BGRA, SWS_BILINEAR, nullptr, nullptr, nullptr);
      sws_width = width;
      sws_height = height;
      sws_format = src_format;
    }

    if (!sws_ctx) {
      std::lock_guard<std::mutex> lock(mu);
      stats.failed_frames += 1;
      return;
    }

    const int dst_stride = width * 4;
    std::vector<uint8_t> bgra(static_cast<size_t>(dst_stride) *
                              static_cast<size_t>(height));
    uint8_t* dst_data[4] = {bgra.data(), nullptr, nullptr, nullptr};
    int dst_linesize[4] = {dst_stride, 0, 0, 0};

    const int scaled = sws_scale(
        sws_ctx, av_frame->data, av_frame->linesize, 0, height, dst_data,
        dst_linesize);
    if (scaled <= 0) {
      std::lock_guard<std::mutex> lock(mu);
      stats.failed_frames += 1;
      return;
    }

    CGImageRef image = CreateImageFromBGRA(bgra, width, height, dst_stride);
    if (!image) {
      std::lock_guard<std::mutex> lock(mu);
      stats.failed_frames += 1;
      return;
    }

    RunOnMainSync([&]() {
      if (!host_view) {
        return;
      }
      EnsureLayersCreated();
      if (!video_layer) {
        return;
      }
      video_layer.contents = (__bridge id)image;
      video_layer.hidden = NO;
      if (hole_layer) {
        hole_layer.hidden = NO;
      }
    });

    CGImageRelease(image);

    std::lock_guard<std::mutex> lock(mu);
    stats.presented_frames += 1;
    stats.width = width;
    stats.height = height;
  }

  void Reset() {
    RunOnMainSync([&]() {
      if (video_layer) {
        video_layer.contents = nil;
        [video_layer removeFromSuperlayer];
        video_layer = nil;
      }
      if (hole_layer) {
        [hole_layer removeFromSuperlayer];
        hole_layer = nil;
      }
      host_view = nil;
    });

    if (sws_ctx) {
      sws_freeContext(sws_ctx);
      sws_ctx = nullptr;
      sws_width = 0;
      sws_height = 0;
      sws_format = -1;
    }

    std::lock_guard<std::mutex> lock(mu);
    stats.attached = false;
    stats.visible = false;
  }

  HoleRendererStats GetStats() const {
    std::lock_guard<std::mutex> lock(mu);
    return stats;
  }
};

HoleRenderer::HoleRenderer() : impl_(new Impl()) {}

HoleRenderer::~HoleRenderer() {
  if (!impl_) return;
  impl_->Reset();
  delete impl_;
  impl_ = nullptr;
}

HoleRenderer::HoleRenderer(HoleRenderer&& other) noexcept : impl_(other.impl_) {
  other.impl_ = nullptr;
}

HoleRenderer& HoleRenderer::operator=(HoleRenderer&& other) noexcept {
  if (this == &other) return *this;
  if (impl_) {
    impl_->Reset();
    delete impl_;
  }
  impl_ = other.impl_;
  other.impl_ = nullptr;
  return *this;
}

bool HoleRenderer::AttachSurface(uintptr_t surface_handle) {
  if (!impl_) return false;
  return impl_->AttachSurface(surface_handle);
}

void HoleRenderer::SetRect(int x, int y, int width, int height, bool visible) {
  if (!impl_) return;
  impl_->SetRect(x, y, width, height, visible);
}

void HoleRenderer::PresentFrame(DecodedVideoFrame&& frame) {
  if (!impl_) return;
  impl_->PresentFrame(std::move(frame));
}

void HoleRenderer::Reset() {
  if (!impl_) return;
  impl_->Reset();
}

HoleRendererStats HoleRenderer::Stats() const {
  if (!impl_) return {};
  return impl_->GetStats();
}

}  // namespace media_native
