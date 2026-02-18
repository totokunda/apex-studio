#include <napi.h>
#include <memory>
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <utility>

extern "C" {
  #include <libavformat/avformat.h>
  #include <libavcodec/avcodec.h>
  #include <libavutil/avutil.h>
  #include <libavutil/imgutils.h>
  #include <libavutil/hwcontext.h>
}

#include <libyuv.h>

extern "C" {
  #include <libswscale/swscale.h>
}

#if defined(__APPLE__)
  #include <CoreMedia/CoreMedia.h>
  #include <CoreVideo/CoreVideo.h>
  #include <VideoToolbox/VideoToolbox.h>
#endif

#if defined(__APPLE__)
// Direct VT path can produce non-monotonic sequential timestamps for some
// GOP/B-frame layouts; keep it opt-in until fully stabilized.
static bool allowDirectVideoToolboxPath() {
  static int cached = -1;
  if (cached != -1) return cached == 1;
  const char* env = std::getenv("APEX_ENABLE_DIRECT_VIDEOTOOLBOX");
  if (!env) {
    cached = 0;
    return false;
  }
  cached = (!std::strcmp(env, "1") || !std::strcmp(env, "true") ||
            !std::strcmp(env, "TRUE"))
             ? 1
             : 0;
  return cached == 1;
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
//  HW accel config
// ═══════════════════════════════════════════════════════════════════════════

static const AVHWDeviceType kHWDeviceTypes[] = {
#if defined(__APPLE__)
  AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
#elif defined(_WIN32)
  AV_HWDEVICE_TYPE_D3D11VA,
  AV_HWDEVICE_TYPE_CUDA,
  AV_HWDEVICE_TYPE_QSV,
#else
  AV_HWDEVICE_TYPE_CUDA,
  AV_HWDEVICE_TYPE_VAAPI,
  AV_HWDEVICE_TYPE_QSV,
#endif
  AV_HWDEVICE_TYPE_NONE
};

static AVPixelFormat g_hwPixFmt = AV_PIX_FMT_NONE;

static enum AVPixelFormat get_hw_format(AVCodecContext*, const enum AVPixelFormat* fmts) {
  for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; p++) {
    if (*p == g_hwPixFmt) return *p;
  }
  return fmts[0];
}

// ═══════════════════════════════════════════════════════════════════════════
//  Color conversion: YUV/etc → RGBA via libyuv (SIMD) + swscale fallback
// ═══════════════════════════════════════════════════════════════════════════

static int convertFrameToRGBA(AVFrame* srcFrame, uint8_t* dstData, int width, int height) {
  AVPixelFormat srcFmt = static_cast<AVPixelFormat>(srcFrame->format);
  int dstStride = width * 4;
  int result = -1;
  const bool sameDimensions = (srcFrame->width == width && srcFrame->height == height);

  // libyuv fast paths are only safe for 1:1 conversion. For resizing, use swscale.
  if (sameDimensions) {
    // libyuv naming on little-endian:
    //   "ARGB" = B,G,R,A in memory = BGRA
    //   "ABGR" = R,G,B,A in memory = RGBA  ← what we want
    if (srcFmt == AV_PIX_FMT_YUV420P || srcFmt == AV_PIX_FMT_YUVJ420P) {
      result = libyuv::I420ToABGR(
          srcFrame->data[0], srcFrame->linesize[0],
          srcFrame->data[1], srcFrame->linesize[1],
          srcFrame->data[2], srcFrame->linesize[2],
          dstData, dstStride, width, height);
    } else if (srcFmt == AV_PIX_FMT_NV12) {
      result = libyuv::NV12ToABGR(
          srcFrame->data[0], srcFrame->linesize[0],
          srcFrame->data[1], srcFrame->linesize[1],
          dstData, dstStride, width, height);
    } else if (srcFmt == AV_PIX_FMT_NV21) {
      result = libyuv::NV21ToABGR(
          srcFrame->data[0], srcFrame->linesize[0],
          srcFrame->data[1], srcFrame->linesize[1],
          dstData, dstStride, width, height);
    } else if (srcFmt == AV_PIX_FMT_YUV422P || srcFmt == AV_PIX_FMT_YUVJ422P) {
      result = libyuv::I422ToABGR(
          srcFrame->data[0], srcFrame->linesize[0],
          srcFrame->data[1], srcFrame->linesize[1],
          srcFrame->data[2], srcFrame->linesize[2],
          dstData, dstStride, width, height);
    } else if (srcFmt == AV_PIX_FMT_YUV444P || srcFmt == AV_PIX_FMT_YUVJ444P) {
      result = libyuv::I444ToABGR(
          srcFrame->data[0], srcFrame->linesize[0],
          srcFrame->data[1], srcFrame->linesize[1],
          srcFrame->data[2], srcFrame->linesize[2],
          dstData, dstStride, width, height);
    } else if (srcFmt == AV_PIX_FMT_UYVY422) {
      // No direct UYVYToABGR in libyuv — go through ARGB then swap
      result = libyuv::UYVYToARGB(
          srcFrame->data[0], srcFrame->linesize[0],
          dstData, dstStride, width, height);
      if (result == 0) {
        result = libyuv::ARGBToABGR(
            dstData, dstStride, dstData, dstStride, width, height);
      }
    } else if (srcFmt == AV_PIX_FMT_YUYV422) {
      // No direct YUY2ToABGR in libyuv — go through ARGB then swap
      result = libyuv::YUY2ToARGB(
          srcFrame->data[0], srcFrame->linesize[0],
          dstData, dstStride, width, height);
      if (result == 0) {
        result = libyuv::ARGBToABGR(
            dstData, dstStride, dstData, dstStride, width, height);
      }
    } else if (srcFmt == AV_PIX_FMT_RGBA) {
      // Already RGBA — just copy
      libyuv::ARGBCopy(
          srcFrame->data[0], srcFrame->linesize[0],
          dstData, dstStride, width, height);
      result = 0;
    } else if (srcFmt == AV_PIX_FMT_BGRA) {
      // BGRA → RGBA: swap R and B
      result = libyuv::ARGBToABGR(
          srcFrame->data[0], srcFrame->linesize[0],
          dstData, dstStride, width, height);
    }
  }

  if (result != 0) {
    SwsContext* swsCtx = sws_getContext(
        srcFrame->width, srcFrame->height, srcFmt,
        width, height, AV_PIX_FMT_RGBA,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!swsCtx) return -1;
    uint8_t* dst[] = { dstData };
    int dstStrides[] = { dstStride };
    sws_scale(swsCtx, srcFrame->data, srcFrame->linesize, 0, srcFrame->height, dst, dstStrides);
    sws_freeContext(swsCtx);
    result = 0;
  }
  return result;
}

#if defined(__APPLE__)
static constexpr int64_t kCMTimescale = 1000000;

static uint16_t readBE16(const uint8_t* p) {
  return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

static size_t findAnnexBStartCode(const uint8_t* data, size_t size, size_t offset, size_t* scSize) {
  for (size_t i = offset; i + 3 < size; ++i) {
    if (data[i] != 0 || data[i + 1] != 0) continue;
    if (data[i + 2] == 1) {
      *scSize = 3;
      return i;
    }
    if (i + 3 < size && data[i + 2] == 0 && data[i + 3] == 1) {
      *scSize = 4;
      return i;
    }
  }
  return SIZE_MAX;
}

static bool extractAnnexBNalus(const uint8_t* data, size_t size,
                               std::vector<std::pair<const uint8_t*, size_t>>& out) {
  out.clear();
  size_t pos = 0;
  while (true) {
    size_t scSize = 0;
    size_t start = findAnnexBStartCode(data, size, pos, &scSize);
    if (start == SIZE_MAX) break;
    size_t nalStart = start + scSize;
    size_t nextScSize = 0;
    size_t next = findAnnexBStartCode(data, size, nalStart, &nextScSize);
    size_t nalEnd = (next == SIZE_MAX) ? size : next;
    if (nalEnd > nalStart) out.emplace_back(data + nalStart, nalEnd - nalStart);
    if (next == SIZE_MAX) break;
    pos = next;
  }
  return !out.empty();
}

static bool looksLikeAnnexB(const uint8_t* data, size_t size) {
  if (!data || size < 4) return false;
  size_t scSize = 0;
  size_t start = findAnnexBStartCode(data, size, 0, &scSize);
  return start != SIZE_MAX && start <= 4;
}

static bool annexBToLengthPrefixed(const uint8_t* data, size_t size, int nalLenSize,
                                   std::vector<uint8_t>& out) {
  if (nalLenSize < 1 || nalLenSize > 4) return false;
  std::vector<std::pair<const uint8_t*, size_t>> nalus;
  if (!extractAnnexBNalus(data, size, nalus)) return false;
  out.clear();
  for (const auto& nal : nalus) {
    const size_t len = nal.second;
    const uint64_t maxLen = (nalLenSize == 4) ? 0xFFFFFFFFULL : ((1ULL << (8 * nalLenSize)) - 1);
    if (len > maxLen) return false;
    uint8_t prefix[4] = {0, 0, 0, 0};
    for (int i = 0; i < nalLenSize; ++i) {
      prefix[nalLenSize - 1 - i] = static_cast<uint8_t>((len >> (8 * i)) & 0xFF);
    }
    out.insert(out.end(), prefix, prefix + nalLenSize);
    out.insert(out.end(), nal.first, nal.first + nal.second);
  }
  return !out.empty();
}

static bool parseH264ParameterSets(const uint8_t* extra, size_t extraSize, int* nalLenSize,
                                   std::vector<std::vector<uint8_t>>& sets) {
  sets.clear();
  if (!extra || extraSize < 7) return false;

  // avcC
  if (extra[0] == 1) {
    *nalLenSize = (extra[4] & 0x03) + 1;
    size_t off = 6;
    int numSps = extra[5] & 0x1F;
    for (int i = 0; i < numSps; ++i) {
      if (off + 2 > extraSize) return false;
      uint16_t len = readBE16(extra + off);
      off += 2;
      if (off + len > extraSize || len == 0) return false;
      sets.emplace_back(extra + off, extra + off + len);
      off += len;
    }
    if (off + 1 > extraSize) return false;
    int numPps = extra[off++];
    for (int i = 0; i < numPps; ++i) {
      if (off + 2 > extraSize) return false;
      uint16_t len = readBE16(extra + off);
      off += 2;
      if (off + len > extraSize || len == 0) return false;
      sets.emplace_back(extra + off, extra + off + len);
      off += len;
    }
    return !sets.empty();
  }

  // Annex B
  std::vector<std::pair<const uint8_t*, size_t>> nalus;
  if (!extractAnnexBNalus(extra, extraSize, nalus)) return false;
  bool hasSps = false, hasPps = false;
  for (const auto& nal : nalus) {
    if (nal.second < 1) continue;
    uint8_t type = nal.first[0] & 0x1F;
    if (type == 7 && !hasSps) {
      sets.emplace_back(nal.first, nal.first + nal.second);
      hasSps = true;
    } else if (type == 8 && !hasPps) {
      sets.emplace_back(nal.first, nal.first + nal.second);
      hasPps = true;
    }
  }
  *nalLenSize = 4;
  return hasSps && hasPps;
}

static bool parseHEVCParameterSets(const uint8_t* extra, size_t extraSize, int* nalLenSize,
                                   std::vector<std::vector<uint8_t>>& sets) {
  sets.clear();
  if (!extra || extraSize < 23) return false;

  // hvcC
  if (extra[0] == 1) {
    *nalLenSize = (extra[21] & 0x03) + 1;
    size_t off = 23;
    uint8_t numArrays = extra[22];
    for (uint8_t i = 0; i < numArrays; ++i) {
      if (off + 3 > extraSize) return false;
      uint8_t nalType = extra[off++] & 0x3F;
      uint16_t numNalus = readBE16(extra + off);
      off += 2;
      for (uint16_t j = 0; j < numNalus; ++j) {
        if (off + 2 > extraSize) return false;
        uint16_t len = readBE16(extra + off);
        off += 2;
        if (off + len > extraSize || len == 0) return false;
        if (nalType == 32 || nalType == 33 || nalType == 34) {
          sets.emplace_back(extra + off, extra + off + len);
        }
        off += len;
      }
    }
    return !sets.empty();
  }

  // Annex B
  std::vector<std::pair<const uint8_t*, size_t>> nalus;
  if (!extractAnnexBNalus(extra, extraSize, nalus)) return false;
  bool hasVps = false, hasSps = false, hasPps = false;
  for (const auto& nal : nalus) {
    if (nal.second < 2) continue;
    uint8_t type = (nal.first[0] >> 1) & 0x3F;
    if (type == 32 && !hasVps) {
      sets.emplace_back(nal.first, nal.first + nal.second);
      hasVps = true;
    } else if (type == 33 && !hasSps) {
      sets.emplace_back(nal.first, nal.first + nal.second);
      hasSps = true;
    } else if (type == 34 && !hasPps) {
      sets.emplace_back(nal.first, nal.first + nal.second);
      hasPps = true;
    }
  }
  *nalLenSize = 4;
  return hasVps && hasSps && hasPps;
}

static bool copyPixelBufferToRGBA(CVPixelBufferRef pixelBuffer, uint8_t* dstData, int width, int height,
                                  bool srcIsBGRA) {
  if (!pixelBuffer || !dstData || width <= 0 || height <= 0) return false;
  if (CVPixelBufferGetPlaneCount(pixelBuffer) != 0) return false;

  CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  uint8_t* src = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pixelBuffer));
  size_t srcStride = CVPixelBufferGetBytesPerRow(pixelBuffer);
  int srcWidth = static_cast<int>(CVPixelBufferGetWidth(pixelBuffer));
  int srcHeight = static_cast<int>(CVPixelBufferGetHeight(pixelBuffer));

  bool ok = true;
  int dstStride = width * 4;
  const bool sameDimensions = (srcWidth == width && srcHeight == height);
  if (!src || srcWidth <= 0 || srcHeight <= 0) {
    ok = false;
  } else if (sameDimensions) {
    if (srcIsBGRA) {
      ok = libyuv::ARGBToABGR(src, static_cast<int>(srcStride), dstData, dstStride, width, height) == 0;
    } else {
      for (int y = 0; y < height; ++y) {
        std::memcpy(dstData + y * dstStride, src + y * srcStride, static_cast<size_t>(width) * 4);
      }
    }
  } else {
    const size_t srcBytes = static_cast<size_t>(srcWidth) * static_cast<size_t>(srcHeight) * 4;
    std::vector<uint8_t> srcRgba(srcBytes);
    if (srcRgba.empty()) {
      ok = false;
    } else {
      const int srcRgbaStride = srcWidth * 4;
      if (srcIsBGRA) {
        ok = libyuv::ARGBToABGR(
                 src,
                 static_cast<int>(srcStride),
                 srcRgba.data(),
                 srcRgbaStride,
                 srcWidth,
                 srcHeight) == 0;
      } else {
        for (int y = 0; y < srcHeight; ++y) {
          std::memcpy(
              srcRgba.data() + y * srcRgbaStride,
              src + y * srcStride,
              static_cast<size_t>(srcWidth) * 4);
        }
      }
      if (ok) {
        ok = libyuv::ARGBScale(
                 srcRgba.data(),
                 srcRgbaStride,
                 srcWidth,
                 srcHeight,
                 dstData,
                 dstStride,
                 width,
                 height,
                 libyuv::kFilterBilinear) == 0;
      }
    }
  }

  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  return ok;
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
//  VideoDecoder: FFmpeg resources + stateful decode cursor
//
//  Key insight for editor-grade performance:
//  The decoder tracks its current position (lastDecodedPts). When the user
//  seeks to a nearby timestamp, we skip the expensive seek+flush and just
//  decode forward. This makes scrubbing feel instant.
// ═══════════════════════════════════════════════════════════════════════════

class VideoDecoder {
private:
  AVFormatContext* formatCtx = nullptr;
  AVCodecContext* videoCodecCtx = nullptr;
  AVCodecContext* audioCodecCtx = nullptr;
  const AVCodec* videoCodec = nullptr;
  const AVCodec* audioCodec = nullptr;
  AVBufferRef* hwDeviceCtx = nullptr;
  int videoStreamIndex = -1;
  int audioStreamIndex = -1;
  bool usingHwAccel = false;

  // ── Decode cursor state ──
  int64_t lastDecodedPts = AV_NOPTS_VALUE;
  int64_t lastServedPts = AV_NOPTS_VALUE;
  bool    decoderFlushed = true;
  bool    decoderEofSignaled = false;

  // ── Reusable allocations (avoid malloc per frame) ──
  AVPacket* pkt = nullptr;
  AVFrame*  frame = nullptr;
  AVFrame*  swFrame = nullptr;

  struct CachedFrame {
    int64_t pts = AV_NOPTS_VALUE;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;
  };

  std::deque<CachedFrame> historyCache;
  std::deque<CachedFrame> aheadCache;
  size_t historyCacheBytes = 0;
  size_t aheadCacheBytes = 0;

  static constexpr size_t kHistoryMaxFrames = 3;
  static constexpr size_t kAheadMaxFrames = 4;
  static constexpr size_t kHistoryBudgetBytes = 96 * 1024 * 1024;
  static constexpr size_t kAheadBudgetBytes = 128 * 1024 * 1024;

#if defined(__APPLE__)
  struct VTDecodeOutput {
    OSStatus status = noErr;
    CVPixelBufferRef pixelBuffer = nullptr;
    CMTime pts = kCMTimeInvalid;
  };

  bool usingDirectVideoToolbox = false;
  bool vtCodecIsHEVC = false;
  int vtNALLengthSize = 4;
  bool vtOutputIsBGRA = true;
  int vtConsecutiveDecodeFailures = 0;
  CMVideoFormatDescriptionRef vtFormatDesc = nullptr;
  VTDecompressionSessionRef vtSession = nullptr;
  CVPixelBufferRef vtLastPixelBuffer = nullptr;
  std::vector<uint8_t> vtPacketBuffer;

  static void vtDecodeCallback(void* /*decompressionOutputRefCon*/,
                               void* sourceFrameRefCon,
                               OSStatus status,
                               VTDecodeInfoFlags /*infoFlags*/,
                               CVImageBufferRef imageBuffer,
                               CMTime presentationTimeStamp,
                               CMTime /*presentationDuration*/) {
    auto* out = static_cast<VTDecodeOutput*>(sourceFrameRefCon);
    if (!out) return;
    out->status = status;
    if (status != noErr || !imageBuffer) return;
    if (out->pixelBuffer) CFRelease(out->pixelBuffer);
    out->pixelBuffer = static_cast<CVPixelBufferRef>(imageBuffer);
    CFRetain(out->pixelBuffer);
    out->pts = presentationTimeStamp;
  }

  void clearVTFrame() {
    if (vtLastPixelBuffer) {
      CFRelease(vtLastPixelBuffer);
      vtLastPixelBuffer = nullptr;
    }
  }

  void resetVTSession() {
    clearVTFrame();
    if (vtSession) {
      VTDecompressionSessionInvalidate(vtSession);
      CFRelease(vtSession);
      vtSession = nullptr;
    }
    if (vtFormatDesc) {
      CFRelease(vtFormatDesc);
      vtFormatDesc = nullptr;
    }
  }

  bool createVTSession(OSType pixelFormat) {
    if (!vtFormatDesc) return false;
    CFMutableDictionaryRef attrs = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (!attrs) return false;

    int32_t pf = static_cast<int32_t>(pixelFormat);
    CFNumberRef pfNum = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pf);
    if (!pfNum) {
      CFRelease(attrs);
      return false;
    }
    CFDictionarySetValue(attrs, kCVPixelBufferPixelFormatTypeKey, pfNum);
    CFRelease(pfNum);

    VTDecompressionOutputCallbackRecord cb {};
    cb.decompressionOutputCallback = &VideoDecoder::vtDecodeCallback;
    cb.decompressionOutputRefCon = this;

    OSStatus status = VTDecompressionSessionCreate(
        kCFAllocatorDefault, vtFormatDesc, nullptr, attrs, &cb, &vtSession);
    CFRelease(attrs);
    if (status != noErr || !vtSession) return false;

    vtOutputIsBGRA = (pixelFormat == kCVPixelFormatType_32BGRA);
    return true;
  }

  bool initDirectVideoToolbox(AVCodecParameters* cp) {
    if (!cp || !cp->extradata || cp->extradata_size <= 0) return false;
    if (cp->codec_id != AV_CODEC_ID_H264 && cp->codec_id != AV_CODEC_ID_HEVC) return false;

    std::vector<std::vector<uint8_t>> paramSets;
    vtCodecIsHEVC = (cp->codec_id == AV_CODEC_ID_HEVC);
    bool parsed = vtCodecIsHEVC
      ? parseHEVCParameterSets(cp->extradata, static_cast<size_t>(cp->extradata_size), &vtNALLengthSize, paramSets)
      : parseH264ParameterSets(cp->extradata, static_cast<size_t>(cp->extradata_size), &vtNALLengthSize, paramSets);
    if (!parsed || paramSets.empty()) return false;

    std::vector<const uint8_t*> setPtrs;
    std::vector<size_t> setSizes;
    setPtrs.reserve(paramSets.size());
    setSizes.reserve(paramSets.size());
    for (const auto& s : paramSets) {
      setPtrs.push_back(s.data());
      setSizes.push_back(s.size());
    }

    OSStatus status = noErr;
    if (vtCodecIsHEVC) {
      status = CMVideoFormatDescriptionCreateFromHEVCParameterSets(
          kCFAllocatorDefault,
          static_cast<int>(setPtrs.size()),
          setPtrs.data(),
          setSizes.data(),
          vtNALLengthSize,
          nullptr,
          &vtFormatDesc);
    } else {
      status = CMVideoFormatDescriptionCreateFromH264ParameterSets(
          kCFAllocatorDefault,
          static_cast<int>(setPtrs.size()),
          setPtrs.data(),
          setSizes.data(),
          vtNALLengthSize,
          &vtFormatDesc);
    }
    if (status != noErr || !vtFormatDesc) {
      resetVTSession();
      return false;
    }

    // BGRA is the most broadly supported VT output pixel format.
    if (!createVTSession(kCVPixelFormatType_32BGRA)) {
      resetVTSession();
      return false;
    }
    return true;
  }

  bool buildVTCompressedSample(const AVPacket* packet, CMSampleBufferRef* outSample) {
    *outSample = nullptr;
    if (!packet || packet->size <= 0 || !packet->data || !vtFormatDesc) return false;

    const uint8_t* data = packet->data;
    size_t size = static_cast<size_t>(packet->size);
    vtPacketBuffer.clear();
    if (looksLikeAnnexB(data, size)) {
      if (!annexBToLengthPrefixed(data, size, vtNALLengthSize, vtPacketBuffer)) return false;
      data = vtPacketBuffer.data();
      size = vtPacketBuffer.size();
    }
    if (size == 0) return false;

    CMBlockBufferRef block = nullptr;
    OSStatus status = CMBlockBufferCreateWithMemoryBlock(
        kCFAllocatorDefault, nullptr, size, kCFAllocatorDefault, nullptr, 0, size, 0, &block);
    if (status != kCMBlockBufferNoErr || !block) return false;

    status = CMBlockBufferReplaceDataBytes(data, block, 0, size);
    if (status != kCMBlockBufferNoErr) {
      CFRelease(block);
      return false;
    }

    int64_t packetPts = (packet->pts != AV_NOPTS_VALUE) ? packet->pts : packet->dts;
    CMTime pts = kCMTimeInvalid;
    if (packetPts != AV_NOPTS_VALUE) {
      double seconds = ptsToTs(packetPts);
      pts = CMTimeMakeWithSeconds(seconds, kCMTimescale);
    }
    CMSampleTimingInfo timing = {kCMTimeInvalid, pts, kCMTimeInvalid};
    size_t sampleSize = size;

    status = CMSampleBufferCreateReady(kCFAllocatorDefault, block, vtFormatDesc,
                                       1, 1, &timing, 1, &sampleSize, outSample);
    CFRelease(block);
    if (status != noErr || !*outSample) return false;

    if (!(packet->flags & AV_PKT_FLAG_KEY)) {
      CFArrayRef attachments = CMSampleBufferGetSampleAttachmentsArray(*outSample, true);
      if (attachments && CFArrayGetCount(attachments) > 0) {
        auto* dict = const_cast<CFMutableDictionaryRef>(
            static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(attachments, 0)));
        if (dict) CFDictionarySetValue(dict, kCMSampleAttachmentKey_NotSync, kCFBooleanTrue);
      }
    }
    return true;
  }

  bool decodePacketWithVT(const AVPacket* packet, int64_t* outPts) {
    if (!vtSession || !vtFormatDesc) return false;
    VTDecodeOutput out {};
    CMSampleBufferRef sample = nullptr;
    if (!buildVTCompressedSample(packet, &sample)) return false;

    VTDecodeInfoFlags infoFlags = 0;
    OSStatus status = VTDecompressionSessionDecodeFrame(
        vtSession, sample, kVTDecodeFrame_EnableAsynchronousDecompression, &out, &infoFlags);
    CFRelease(sample);
    if (status != noErr) return false;

    VTDecompressionSessionWaitForAsynchronousFrames(vtSession);
    if (out.status != noErr || !out.pixelBuffer) {
      if (out.pixelBuffer) CFRelease(out.pixelBuffer);
      return false;
    }

    clearVTFrame();
    vtLastPixelBuffer = out.pixelBuffer;
    out.pixelBuffer = nullptr;

    int64_t packetPts = (packet->pts != AV_NOPTS_VALUE) ? packet->pts : packet->dts;
    if (CMTIME_IS_VALID(out.pts) && out.pts.timescale > 0) {
      AVRational cmTb = {1, static_cast<int>(out.pts.timescale)};
      *outPts = av_rescale_q(out.pts.value, cmTb, videoTimeBase());
    } else {
      *outPts = packetPts;
    }
    if (out.pixelBuffer) CFRelease(out.pixelBuffer);
    return true;
  }

  bool writeCurrentVTFrameToRGBA(uint8_t* dstData, int width, int height) {
    return copyPixelBufferToRGBA(vtLastPixelBuffer, dstData, width, height, vtOutputIsBGRA);
  }
#endif

  bool tryHwAccel(AVCodecContext* ctx, const AVCodec* codec) {
    for (int i = 0; kHWDeviceTypes[i] != AV_HWDEVICE_TYPE_NONE; i++) {
      AVHWDeviceType deviceType = kHWDeviceTypes[i];
      AVPixelFormat hwFmt = AV_PIX_FMT_NONE;
      for (int j = 0;; j++) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(codec, j);
        if (!config) break;
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == deviceType) {
          hwFmt = config->pix_fmt; break;
        }
      }
      if (hwFmt == AV_PIX_FMT_NONE) continue;
      AVBufferRef* deviceCtx = nullptr;
      if (av_hwdevice_ctx_create(&deviceCtx, deviceType, nullptr, nullptr, 0) < 0) continue;
      ctx->hw_device_ctx = av_buffer_ref(deviceCtx);
      hwDeviceCtx = deviceCtx;
      g_hwPixFmt = hwFmt;
      ctx->get_format = get_hw_format;
      usingHwAccel = true;
      return true;
    }
    return false;
  }

  AVFrame* transferIfNeeded(AVFrame* decoded) {
    if (decoded->format == g_hwPixFmt && usingHwAccel) {
      if (av_hwframe_transfer_data(swFrame, decoded, 0) < 0) return nullptr;
      swFrame->width = decoded->width;
      swFrame->height = decoded->height;
      return swFrame;
    }
    return decoded;
  }

  bool writeCurrentFrameToRGBA(uint8_t* dstData, int width, int height) {
#if defined(__APPLE__)
    if (usingDirectVideoToolbox) return writeCurrentVTFrameToRGBA(dstData, width, height);
#endif
    AVFrame* src = transferIfNeeded(frame);
    if (!src) return false;
    return convertFrameToRGBA(src, dstData, width, height) == 0;
  }

  static size_t rgbaByteSize(int width, int height) {
    if (width <= 0 || height <= 0) return 0;
    return static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
  }

  void clearFrameCaches() {
    historyCache.clear();
    aheadCache.clear();
    historyCacheBytes = 0;
    aheadCacheBytes = 0;
  }

  void trimHistoryCache() {
    while (!historyCache.empty() &&
           (historyCache.size() > kHistoryMaxFrames ||
            historyCacheBytes > kHistoryBudgetBytes)) {
      historyCacheBytes -= historyCache.front().rgba.size();
      historyCache.pop_front();
    }
  }

  void trimAheadCache() {
    while (!aheadCache.empty() &&
           (aheadCache.size() > kAheadMaxFrames ||
            aheadCacheBytes > kAheadBudgetBytes)) {
      aheadCacheBytes -= aheadCache.front().rgba.size();
      aheadCache.pop_front();
    }
  }

  void pushHistoryEntry(CachedFrame&& entry) {
    if (entry.pts == AV_NOPTS_VALUE || entry.rgba.empty()) return;
    if (!historyCache.empty() && historyCache.back().pts == entry.pts) return;
    historyCacheBytes += entry.rgba.size();
    historyCache.emplace_back(std::move(entry));
    trimHistoryCache();
  }

  void pushAheadEntry(CachedFrame&& entry) {
    if (entry.pts == AV_NOPTS_VALUE || entry.rgba.empty()) return;
    if (!aheadCache.empty() && aheadCache.back().pts == entry.pts) return;
    aheadCacheBytes += entry.rgba.size();
    aheadCache.emplace_back(std::move(entry));
    trimAheadCache();
  }

  void noteServedFrameCopy(int64_t pts, const uint8_t* rgba, int width, int height) {
    const size_t bytes = rgbaByteSize(width, height);
    if (pts == AV_NOPTS_VALUE || !rgba || bytes == 0) return;
    CachedFrame entry;
    entry.pts = pts;
    entry.width = width;
    entry.height = height;
    entry.rgba.assign(rgba, rgba + bytes);
    pushHistoryEntry(std::move(entry));
    lastServedPts = pts;
  }

  bool serveHistoryFrameNear(int64_t targetPts, int64_t tolerancePts,
                             uint8_t* dstData, int width, int height, int64_t* outPts) {
    if (historyCache.empty()) return false;
    size_t bestIndex = historyCache.size();
    int64_t bestDiff = INT64_MAX;
    for (size_t i = 0; i < historyCache.size(); ++i) {
      const CachedFrame& entry = historyCache[i];
      if (entry.width != width || entry.height != height || entry.rgba.empty()) continue;
      const int64_t diff = std::llabs(entry.pts - targetPts);
      if (diff > tolerancePts) continue;
      if (diff < bestDiff) {
        bestDiff = diff;
        bestIndex = i;
      }
    }
    if (bestIndex == historyCache.size()) return false;

    const CachedFrame& entry = historyCache[bestIndex];
    std::memcpy(dstData, entry.rgba.data(), entry.rgba.size());
    *outPts = entry.pts;
    lastServedPts = entry.pts;
    return true;
  }

  bool consumeAheadForTarget(int64_t targetPts, int64_t tolerancePts,
                             uint8_t* dstData, int width, int height, int64_t* outPts) {
    if (aheadCache.empty()) return false;
    size_t selected = aheadCache.size();
    for (size_t i = 0; i < aheadCache.size(); ++i) {
      const CachedFrame& entry = aheadCache[i];
      if (entry.width != width || entry.height != height || entry.rgba.empty()) continue;
      if (entry.pts + tolerancePts >= targetPts) {
        selected = i;
        break;
      }
    }
    if (selected == aheadCache.size()) return false;

    for (size_t i = 0; i < selected; ++i) {
      CachedFrame consumed = std::move(aheadCache.front());
      aheadCacheBytes -= consumed.rgba.size();
      aheadCache.pop_front();
      pushHistoryEntry(std::move(consumed));
    }

    CachedFrame frameHit = std::move(aheadCache.front());
    aheadCacheBytes -= frameHit.rgba.size();
    aheadCache.pop_front();
    if (frameHit.width != width || frameHit.height != height || frameHit.rgba.empty()) {
      return false;
    }

    std::memcpy(dstData, frameHit.rgba.data(), frameHit.rgba.size());
    *outPts = frameHit.pts;
    lastServedPts = frameHit.pts;
    pushHistoryEntry(std::move(frameHit));
    return true;
  }

  bool consumeNextAheadFrame(uint8_t* dstData, int width, int height, int64_t* outPts) {
    while (!aheadCache.empty()) {
      CachedFrame frameHit = std::move(aheadCache.front());
      aheadCacheBytes -= frameHit.rgba.size();
      aheadCache.pop_front();
      if (frameHit.width != width || frameHit.height != height || frameHit.rgba.empty()) continue;
      std::memcpy(dstData, frameHit.rgba.data(), frameHit.rgba.size());
      *outPts = frameHit.pts;
      lastServedPts = frameHit.pts;
      pushHistoryEntry(std::move(frameHit));
      return true;
    }
    return false;
  }

  void prefetchAheadFrames(int width, int height, int maxFrames) {
    if (maxFrames <= 0) return;
    const size_t frameBytes = rgbaByteSize(width, height);
    if (frameBytes == 0 || frameBytes > kAheadBudgetBytes) return;

    int produced = 0;
    while (produced < maxFrames &&
           aheadCache.size() < kAheadMaxFrames &&
           aheadCacheBytes + frameBytes <= kAheadBudgetBytes) {
      int64_t pts = decodeOneFrame();
      if (pts == AV_NOPTS_VALUE) break;
      CachedFrame entry;
      entry.pts = pts;
      entry.width = width;
      entry.height = height;
      entry.rgba.resize(frameBytes);
      if (!writeCurrentFrameToRGBA(entry.rgba.data(), width, height)) break;
      pushAheadEntry(std::move(entry));
      produced++;
    }
  }

public:
  ~VideoDecoder() { cleanup(); }

  void cleanup() {
#if defined(__APPLE__)
    resetVTSession();
    usingDirectVideoToolbox = false;
    vtCodecIsHEVC = false;
    vtNALLengthSize = 4;
    vtOutputIsBGRA = true;
    vtConsecutiveDecodeFailures = 0;
    vtPacketBuffer.clear();
#endif
    if (pkt)           av_packet_free(&pkt);
    if (frame)         av_frame_free(&frame);
    if (swFrame)       av_frame_free(&swFrame);
    if (videoCodecCtx) avcodec_free_context(&videoCodecCtx);
    if (audioCodecCtx) avcodec_free_context(&audioCodecCtx);
    if (hwDeviceCtx) { av_buffer_unref(&hwDeviceCtx); hwDeviceCtx = nullptr; }
    if (formatCtx)     avformat_close_input(&formatCtx);
    videoCodec = nullptr;
    audioCodec = nullptr;
    videoStreamIndex = -1;
    audioStreamIndex = -1;
    usingHwAccel = false;
    lastDecodedPts = AV_NOPTS_VALUE;
    lastServedPts = AV_NOPTS_VALUE;
    decoderFlushed = true;
    decoderEofSignaled = false;
    clearFrameCaches();
  }

  bool open(const std::string& filename) {
    if (avformat_open_input(&formatCtx, filename.c_str(), nullptr, nullptr) < 0) return false;
    if (avformat_find_stream_info(formatCtx, nullptr) < 0) { cleanup(); return false; }

    for (unsigned int i = 0; i < formatCtx->nb_streams; i++) {
      AVCodecParameters* cp = formatCtx->streams[i]->codecpar;
      if (cp->codec_type == AVMEDIA_TYPE_VIDEO && videoStreamIndex == -1) {
        videoStreamIndex = i;
        videoCodec = avcodec_find_decoder(cp->codec_id);
        if (!videoCodec) { cleanup(); return false; }
        videoCodecCtx = avcodec_alloc_context3(videoCodec);
        if (!videoCodecCtx) { cleanup(); return false; }
        if (avcodec_parameters_to_context(videoCodecCtx, cp) < 0) { cleanup(); return false; }
#if defined(__APPLE__)
        usingDirectVideoToolbox =
          allowDirectVideoToolboxPath() && initDirectVideoToolbox(cp);
        if (usingDirectVideoToolbox) usingHwAccel = true;
#endif
        if (
#if defined(__APPLE__)
            !usingDirectVideoToolbox
#else
            true
#endif
        ) {
          tryHwAccel(videoCodecCtx, videoCodec);
          if (avcodec_open2(videoCodecCtx, videoCodec, nullptr) < 0) {
            if (usingHwAccel) {
              avcodec_free_context(&videoCodecCtx);
              if (hwDeviceCtx) { av_buffer_unref(&hwDeviceCtx); hwDeviceCtx = nullptr; }
              usingHwAccel = false;
              videoCodecCtx = avcodec_alloc_context3(videoCodec);
              if (!videoCodecCtx || avcodec_parameters_to_context(videoCodecCtx, cp) < 0 ||
                  avcodec_open2(videoCodecCtx, videoCodec, nullptr) < 0)
              { cleanup(); return false; }
            } else { cleanup(); return false; }
          }
        } else if (avcodec_open2(videoCodecCtx, videoCodec, nullptr) < 0) {
          // Direct VideoToolbox path still uses stream metadata even if software decode isn't available.
          avcodec_free_context(&videoCodecCtx);
          videoCodecCtx = nullptr;
        }
      }
      else if (cp->codec_type == AVMEDIA_TYPE_AUDIO && audioStreamIndex == -1) {
        audioStreamIndex = i;
        audioCodec = avcodec_find_decoder(cp->codec_id);
        if (!audioCodec) continue;
        audioCodecCtx = avcodec_alloc_context3(audioCodec);
        if (!audioCodecCtx) continue;
        if (avcodec_parameters_to_context(audioCodecCtx, cp) < 0) {
          avcodec_free_context(&audioCodecCtx); continue;
        }
        if (avcodec_open2(audioCodecCtx, audioCodec, nullptr) < 0) {
          avcodec_free_context(&audioCodecCtx); audioCodecCtx = nullptr;
        }
      }
    }

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    swFrame = av_frame_alloc();
    if (!pkt || !frame || !swFrame) { cleanup(); return false; }
    return videoStreamIndex >= 0;
  }

  // ─── Time helpers ───

  AVRational videoTimeBase() const { return formatCtx->streams[videoStreamIndex]->time_base; }

  int64_t tsToPts(double seconds) const {
    const int64_t micros = static_cast<int64_t>(std::llround(seconds * AV_TIME_BASE));
    return av_rescale_q(micros, AV_TIME_BASE_Q, videoTimeBase());
  }

  double ptsToTs(int64_t pts) const { return pts * av_q2d(videoTimeBase()); }

  double fps() const {
    AVRational r = formatCtx->streams[videoStreamIndex]->avg_frame_rate;
    return (r.den > 0) ? av_q2d(r) : 30.0;
  }

  int gopSize() const {
    if (!videoCodecCtx) return 30;
    int g = videoCodecCtx->gop_size;
    return (g > 0) ? g : 30;
  }

  int64_t frameDurationPts() const {
    return av_rescale_q(
      static_cast<int64_t>((1.0 / fps()) * AV_TIME_BASE), AV_TIME_BASE_Q, videoTimeBase());
  }

  int64_t maxSeekablePts() const {
    if (!formatCtx || videoStreamIndex < 0) return AV_NOPTS_VALUE;
    const AVStream* stream = formatCtx->streams[videoStreamIndex];
    int64_t endPts = AV_NOPTS_VALUE;
    if (stream && stream->duration != AV_NOPTS_VALUE && stream->duration > 0) {
      endPts = stream->duration;
    } else if (formatCtx->duration > 0) {
      endPts = av_rescale_q(formatCtx->duration, AV_TIME_BASE_Q, videoTimeBase());
    }
    if (endPts == AV_NOPTS_VALUE || endPts <= 0) return AV_NOPTS_VALUE;

    const int64_t fdur = std::max<int64_t>(1, frameDurationPts());
    int64_t maxPts = endPts - std::max<int64_t>(1, fdur / 2);
    if (maxPts < 0) maxPts = 0;
    return maxPts;
  }

  int64_t clampSeekPts(int64_t pts) const {
    if (pts < 0) pts = 0;
    const int64_t maxPts = maxSeekablePts();
    if (maxPts != AV_NOPTS_VALUE && pts > maxPts) pts = maxPts;
    return pts;
  }

  // ─── Core decode operations ───

  bool hardSeek(int64_t pts) {
    pts = clampSeekPts(pts);
    if (av_seek_frame(formatCtx, videoStreamIndex, pts, AVSEEK_FLAG_BACKWARD) < 0) return false;
#if defined(__APPLE__)
    if (usingDirectVideoToolbox && vtSession) {
      clearVTFrame();
      VTDecompressionSessionWaitForAsynchronousFrames(vtSession);
      VTDecompressionSessionFinishDelayedFrames(vtSession);
      vtConsecutiveDecodeFailures = 0;
    } else
#endif
    if (videoCodecCtx) {
      avcodec_flush_buffers(videoCodecCtx);
    }
    lastDecodedPts = AV_NOPTS_VALUE;
    lastServedPts = AV_NOPTS_VALUE;
    decoderFlushed = false;
    decoderEofSignaled = false;
    clearFrameCaches();
    return true;
  }

  int64_t decodeOneFrame() {
#if defined(__APPLE__)
    if (usingDirectVideoToolbox) {
      while (av_read_frame(formatCtx, pkt) >= 0) {
        if (pkt->stream_index != videoStreamIndex) { av_packet_unref(pkt); continue; }

        bool gotFrame = false;
        int64_t pts = AV_NOPTS_VALUE;
        if (decodePacketWithVT(pkt, &pts)) {
          vtConsecutiveDecodeFailures = 0;
          gotFrame = true;
        } else {
          vtConsecutiveDecodeFailures++;
          // Runtime safety net: if VT fails for this stream, fall back to FFmpeg decode.
          if (videoCodecCtx) {
            int ret = avcodec_send_packet(videoCodecCtx, pkt);
            if (ret >= 0 && avcodec_receive_frame(videoCodecCtx, frame) == 0) {
              usingDirectVideoToolbox = false;
              resetVTSession();
              vtConsecutiveDecodeFailures = 0;
              usingHwAccel = videoCodecCtx->hw_device_ctx != nullptr;
              pts = frame->best_effort_timestamp;
              if (pts == AV_NOPTS_VALUE) pts = frame->pts;
              gotFrame = true;
            }
          }
          if (!gotFrame && vtConsecutiveDecodeFailures >= 8 && videoCodecCtx) {
            usingDirectVideoToolbox = false;
            resetVTSession();
            vtConsecutiveDecodeFailures = 0;
            usingHwAccel = videoCodecCtx->hw_device_ctx != nullptr;
          }
        }

        av_packet_unref(pkt);
        if (gotFrame) {
          lastDecodedPts = pts;
          decoderFlushed = false;
          decoderEofSignaled = false;
          return lastDecodedPts;
        }
      }
      return AV_NOPTS_VALUE;
    }
#endif

    if (!videoCodecCtx) return AV_NOPTS_VALUE;

    for (;;) {
      // Drain already-decoded frames before feeding more packets.
      int ret = avcodec_receive_frame(videoCodecCtx, frame);
      if (ret == 0) {
        int64_t pts = frame->best_effort_timestamp;
        if (pts == AV_NOPTS_VALUE) pts = frame->pts;
        lastDecodedPts = pts;
        decoderFlushed = false;
        decoderEofSignaled = false;
        return lastDecodedPts;
      }

      if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
        // Non-fatal decode error: continue by feeding more packets.
      }

      // Need more input, or decoder reached EOF and has no more buffered frames.
      bool fedDecoder = false;
      while (av_read_frame(formatCtx, pkt) >= 0) {
        if (pkt->stream_index != videoStreamIndex) {
          av_packet_unref(pkt);
          continue;
        }

        ret = avcodec_send_packet(videoCodecCtx, pkt);
        av_packet_unref(pkt);

        if (ret == 0 || ret == AVERROR(EAGAIN)) {
          fedDecoder = true;
          break;
        }

        // Bad packet: keep scanning for the next video packet.
      }

      if (fedDecoder) continue;

      // Input EOF: flush decoder once, then drain remaining buffered frames.
      if (!decoderEofSignaled) {
        ret = avcodec_send_packet(videoCodecCtx, nullptr);
        if (ret == 0 || ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) {
          decoderEofSignaled = true;
          continue;
        }
      }

      return AV_NOPTS_VALUE;
    }
  }

  // ─── Smart seek: editor-grade approach ───
  //
  // Decision tree:
  //   1. Target is 0..N frames ahead of cursor → decode forward (no seek!)
  //   2. Target is behind cursor → hard seek + decode forward to target
  //   3. Target is far ahead → hard seek + decode forward to target
  //   4. keyframeOnly=true → hard seek, return first frame (the keyframe)
  //
  // This means sequential scrubbing forward never seeks — it's as fast as
  // a memcpy + color convert per frame.

  int64_t seekAndDecode(int64_t targetPts, uint8_t* dstData, int width, int height,
                        bool keyframeOnly = false) {
    targetPts = clampSeekPts(targetPts);
    const int64_t fdur = std::max<int64_t>(1, frameDurationPts());
    if (!keyframeOnly) {
      int64_t cachedPts = AV_NOPTS_VALUE;
      if (consumeAheadForTarget(targetPts, fdur / 2, dstData, width, height, &cachedPts)) {
        prefetchAheadFrames(width, height, 1);
        return cachedPts;
      }
      if (serveHistoryFrameNear(targetPts, fdur, dstData, width, height, &cachedPts)) {
        return cachedPts;
      }
    }

    const double fpsValue = std::max(1.0, fps());
    const int64_t minForwardFrames = std::max<int64_t>(
        1, static_cast<int64_t>(std::llround(fpsValue * 3.0)));
    const int64_t forwardThreshold = std::max<int64_t>(
        fdur * std::max<int64_t>(1, gopSize() / 2), fdur * minForwardFrames);

    bool needSeek = true;
    int64_t forwardDistancePts = 0;
    if (!decoderFlushed && lastDecodedPts != AV_NOPTS_VALUE) {
      int64_t delta = targetPts - lastDecodedPts;
      if (delta >= 0) {
        forwardDistancePts = delta;
        if (delta <= forwardThreshold) needSeek = false;
      } else if (-delta <= std::max<int64_t>(1, fdur / 2)) {
        // Scrub jitter near clip boundaries often oscillates by <1 frame.
        // Avoid expensive backward seeks for these micro-movements.
        needSeek = false;
      }
    }

    if (needSeek) {
      if (!hardSeek(targetPts)) return AV_NOPTS_VALUE;
    }

    if (!needSeek && lastDecodedPts != AV_NOPTS_VALUE) {
      // Micro backward scrub jitter: reuse current frame instead of reseeking.
      if (targetPts <= lastDecodedPts) {
        if (!writeCurrentFrameToRGBA(dstData, width, height)) return AV_NOPTS_VALUE;
        noteServedFrameCopy(lastDecodedPts, dstData, width, height);
        return lastDecodedPts;
      }
      // Near EOF, the next forward frame may not exist. Clamp to current.
      const int64_t maxPts = maxSeekablePts();
      if (maxPts != AV_NOPTS_VALUE &&
          lastDecodedPts >= maxPts - fdur &&
          targetPts <= lastDecodedPts + fdur) {
        if (!writeCurrentFrameToRGBA(dstData, width, height)) return AV_NOPTS_VALUE;
        noteServedFrameCopy(lastDecodedPts, dstData, width, height);
        return lastDecodedPts;
      }
    }

    if (keyframeOnly) {
      int64_t pts = decodeOneFrame();
      if (pts == AV_NOPTS_VALUE) return AV_NOPTS_VALUE;
      if (!writeCurrentFrameToRGBA(dstData, width, height)) return AV_NOPTS_VALUE;
      noteServedFrameCopy(pts, dstData, width, height);
      return pts;
    }

    // Decode budget scales with target distance so long-GOP files (e.g. 8s keyframes)
    // still reach exact seek targets without returning stale pre-target frames.
    int64_t estimatedDistancePts = forwardDistancePts;
    if (needSeek) estimatedDistancePts = std::max<int64_t>(0, targetPts);
    const int safetyFrames = std::max(120, gopSize() * 2);
    const int estimatedFrames = static_cast<int>(estimatedDistancePts / fdur) + safetyFrames;
    const int maxFrames = std::clamp(estimatedFrames, 120, 12000);

    int64_t fallbackPts = AV_NOPTS_VALUE;
    for (int i = 0; i < maxFrames; i++) {
      int64_t pts = decodeOneFrame();
      if (pts == AV_NOPTS_VALUE) break;

      if (pts + fdur / 2 < targetPts) {
        fallbackPts = pts;
        continue;
      }

      if (!writeCurrentFrameToRGBA(dstData, width, height)) return AV_NOPTS_VALUE;
      noteServedFrameCopy(pts, dstData, width, height);
      return pts;
    }

    if (fallbackPts != AV_NOPTS_VALUE) {
      if (!writeCurrentFrameToRGBA(dstData, width, height)) return AV_NOPTS_VALUE;
      noteServedFrameCopy(fallbackPts, dstData, width, height);
      return fallbackPts;
    }
    return AV_NOPTS_VALUE;
  }

  // Sequential decode — next frame in stream order.
  // If endPts is set (!= AV_NOPTS_VALUE), returns AV_NOPTS_VALUE when
  // the decoded frame's PTS exceeds the end boundary.
  int64_t decodeNextFrameInto(uint8_t* dstData, int width, int height,
                              int64_t endPts = AV_NOPTS_VALUE) {
    int64_t cachedPts = AV_NOPTS_VALUE;
    if (consumeNextAheadFrame(dstData, width, height, &cachedPts)) {
      if (endPts != AV_NOPTS_VALUE && cachedPts > endPts + frameDurationPts() / 2)
        return AV_NOPTS_VALUE;
      prefetchAheadFrames(width, height, 1);
      return cachedPts;
    }

    int64_t pts = decodeOneFrame();
    if (pts == AV_NOPTS_VALUE) return AV_NOPTS_VALUE;
    // Check end boundary (with half-frame tolerance so we don't drop the last frame)
    if (endPts != AV_NOPTS_VALUE && pts > endPts + frameDurationPts() / 2)
      return AV_NOPTS_VALUE;
    if (!writeCurrentFrameToRGBA(dstData, width, height)) return AV_NOPTS_VALUE;
    noteServedFrameCopy(pts, dstData, width, height);
    prefetchAheadFrames(width, height, 1);
    return pts;
  }

  // Position the decoder at a given start PTS using seekAndDecode.
  // The first frame at/after startPts is decoded into the buffer.
  // Returns the PTS of that first frame, or AV_NOPTS_VALUE on error.
  int64_t seekToStart(int64_t startPts, uint8_t* dstData, int width, int height) {
    const int64_t fdur = std::max<int64_t>(1, frameDurationPts());
    int64_t cachedPts = AV_NOPTS_VALUE;
    if (consumeAheadForTarget(startPts, fdur / 2, dstData, width, height, &cachedPts)) {
      prefetchAheadFrames(width, height, 2);
      return cachedPts;
    }
    if (!decoderFlushed && lastServedPts != AV_NOPTS_VALUE &&
        std::llabs(lastServedPts - startPts) <= fdur &&
        serveHistoryFrameNear(startPts, fdur, dstData, width, height, &cachedPts)) {
      prefetchAheadFrames(width, height, 2);
      return cachedPts;
    }
    // Clip boundaries are often contiguous (next start ~= previous end). If we
    // already decoded a frame within one frame of the new start, reuse it
    // instead of forcing a backward seek + GOP walk.
    if (!decoderFlushed && lastDecodedPts != AV_NOPTS_VALUE &&
        lastDecodedPts >= startPts && lastDecodedPts - startPts <= fdur) {
      if (!writeCurrentFrameToRGBA(dstData, width, height)) return AV_NOPTS_VALUE;
      noteServedFrameCopy(lastDecodedPts, dstData, width, height);
      prefetchAheadFrames(width, height, 2);
      return lastDecodedPts;
    }
    int64_t pts = seekAndDecode(startPts, dstData, width, height, false);
    if (pts != AV_NOPTS_VALUE) prefetchAheadFrames(width, height, 2);
    return pts;
  }

  int videoWidth() const {
    if (videoCodecCtx && videoCodecCtx->width > 0) return videoCodecCtx->width;
    if (formatCtx && videoStreamIndex >= 0) return formatCtx->streams[videoStreamIndex]->codecpar->width;
    return 0;
  }

  int videoHeight() const {
    if (videoCodecCtx && videoCodecCtx->height > 0) return videoCodecCtx->height;
    if (formatCtx && videoStreamIndex >= 0) return formatCtx->streams[videoStreamIndex]->codecpar->height;
    return 0;
  }

  std::string outputPixelFormat() const {
#if defined(__APPLE__)
    if (usingDirectVideoToolbox) return "rgba";
#endif
    if (!videoCodecCtx) return "unknown";
    const char* name = av_get_pix_fmt_name(videoCodecCtx->pix_fmt);
    return name ? name : "unknown";
  }

  std::string videoCodecName() const {
    if (videoCodecCtx) return avcodec_get_name(videoCodecCtx->codec_id);
    if (formatCtx && videoStreamIndex >= 0) {
      return avcodec_get_name(formatCtx->streams[videoStreamIndex]->codecpar->codec_id);
    }
    return "unknown";
  }

  std::string decodeBackend() const {
#if defined(__APPLE__)
    if (usingDirectVideoToolbox) return "videotoolbox_direct";
#endif
    if (usingHwAccel) return "ffmpeg_hwaccel";
    return "ffmpeg_software";
  }

  bool isHwAccelerated() const { return usingHwAccel; }
  AVFormatContext* getFormatContext() { return formatCtx; }
  AVCodecContext* getVideoCodecContext() { return videoCodecCtx; }
  AVCodecContext* getAudioCodecContext() { return audioCodecCtx; }
  int getVideoStreamIndex() const { return videoStreamIndex; }
  int getAudioStreamIndex() const { return audioStreamIndex; }
};

// ═══════════════════════════════════════════════════════════════════════════
//  Decoder pool
// ═══════════════════════════════════════════════════════════════════════════

struct DecoderEntry {
  std::unique_ptr<VideoDecoder> decoder;
  std::chrono::steady_clock::time_point lastUsed;
};

static std::map<std::string, DecoderEntry> g_decoderPool;
static const size_t kMaxDecoders = 16;

static void evictIfNeeded() {
  while (g_decoderPool.size() >= kMaxDecoders) {
    auto oldest = g_decoderPool.begin();
    for (auto it = g_decoderPool.begin(); it != g_decoderPool.end(); ++it)
      if (it->second.lastUsed < oldest->second.lastUsed) oldest = it;
    g_decoderPool.erase(oldest->first);
  }
}

static std::string makeDecoderPoolKey(const std::string& filename, const std::string& decoderKey) {
  if (decoderKey.empty()) return filename;
  return decoderKey + "|" + filename;
}

static VideoDecoder* getDecoder(Napi::Env env, const std::string& filename, const std::string& decoderKey) {
  const std::string poolKey = makeDecoderPoolKey(filename, decoderKey);
  auto it = g_decoderPool.find(poolKey);
  if (it == g_decoderPool.end()) {
    evictIfNeeded();
    auto decoder = std::make_unique<VideoDecoder>();
    if (!decoder->open(filename)) {
      Napi::Error::New(env, "Failed to open file").ThrowAsJavaScriptException();
      return nullptr;
    }
    DecoderEntry entry;
    entry.decoder = std::move(decoder);
    entry.lastUsed = std::chrono::steady_clock::now();
    it = g_decoderPool.emplace(poolKey, std::move(entry)).first;
  } else {
    it->second.lastUsed = std::chrono::steady_clock::now();
  }
  return it->second.decoder.get();
}

// ═══════════════════════════════════════════════════════════════════════════
//  N-API exports
// ═══════════════════════════════════════════════════════════════════════════

// loadFile(path) → file info object
Napi::Object LoadFile(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Expected filename").ThrowAsJavaScriptException();
    return Napi::Object::New(env);
  }
  std::string filename = info[0].As<Napi::String>().Utf8Value();
  std::string decoderKey = filename;
  if (info.Length() > 1 && info[1].IsString()) {
    decoderKey = info[1].As<Napi::String>().Utf8Value();
  }
  VideoDecoder* dec = getDecoder(env, filename, decoderKey);
  if (!dec) return Napi::Object::New(env);

  AVFormatContext* fmt = dec->getFormatContext();
  AVCodecContext* ac = dec->getAudioCodecContext();

  Napi::Object r = Napi::Object::New(env);
  r.Set("format", Napi::String::New(env, fmt->iformat->name));
  r.Set("duration", Napi::Number::New(env, fmt->duration / (double)AV_TIME_BASE));
  r.Set("bitrate", Napi::Number::New(env, fmt->bit_rate));
  r.Set("nb_streams", Napi::Number::New(env, fmt->nb_streams));
  r.Set("hw_accelerated", Napi::Boolean::New(env, dec->isHwAccelerated()));
  r.Set("decode_backend", Napi::String::New(env, dec->decodeBackend()));

  if (dec->getVideoStreamIndex() >= 0) {
    Napi::Object v = Napi::Object::New(env);
    v.Set("width", Napi::Number::New(env, dec->videoWidth()));
    v.Set("height", Napi::Number::New(env, dec->videoHeight()));
    v.Set("codec", Napi::String::New(env, dec->videoCodecName()));
    v.Set("pixel_format", Napi::String::New(env, dec->outputPixelFormat()));
    v.Set("fps", Napi::Number::New(env, dec->fps()));
    v.Set("stream_index", Napi::Number::New(env, dec->getVideoStreamIndex()));
    r.Set("video", v);
  }
  if (ac) {
    Napi::Object a = Napi::Object::New(env);
    a.Set("codec", Napi::String::New(env, avcodec_get_name(ac->codec_id)));
    a.Set("sample_rate", Napi::Number::New(env, ac->sample_rate));
    a.Set("channels", Napi::Number::New(env, ac->ch_layout.nb_channels));
    a.Set("stream_index", Napi::Number::New(env, dec->getAudioStreamIndex()));
    r.Set("audio", a);
  }
  return r;
}

// decodeFrameInto(path, buffer, timestamp, keyframeOnly?, outWidth?, outHeight?, decoderKey?) → { timestamp }
//
// Smart seeking with cursor awareness:
//  - Scrubbing forward frame-by-frame: no seek needed, just decode forward
//  - Scrubbing backward or jumping: seek to keyframe, decode to target
//  - keyframeOnly=true: instant scrub preview (nearest keyframe)
Napi::Value DecodeFrameInto(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3 || !info[0].IsString() || !info[1].IsTypedArray() || !info[2].IsNumber()) {
    Napi::TypeError::New(env, "Expected (path, buffer, timestamp, keyframeOnly?)").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string filename = info[0].As<Napi::String>().Utf8Value();
  auto arr = info[1].As<Napi::TypedArrayOf<uint8_t>>();
  double ts = info[2].As<Napi::Number>().DoubleValue();
  bool kfOnly = (info.Length() > 3 && info[3].IsBoolean()) ? info[3].As<Napi::Boolean>().Value() : false;
  std::string decoderKey = filename;
  if (info.Length() > 4 && info[4].IsString()) {
    decoderKey = info[4].As<Napi::String>().Utf8Value();
  }
  if (info.Length() > 6 && info[6].IsString()) {
    decoderKey = info[6].As<Napi::String>().Utf8Value();
  }

  VideoDecoder* dec = getDecoder(env, filename, decoderKey);
  if (!dec) return env.Undefined();
  int srcW = dec->videoWidth();
  int srcH = dec->videoHeight();
  if (srcW <= 0 || srcH <= 0) {
    Napi::Error::New(env, "No video stream").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  int outW = srcW;
  int outH = srcH;
  if (info.Length() > 4 && info[4].IsNumber()) {
    if (info.Length() < 6 || !info[4].IsNumber() || !info[5].IsNumber()) {
      Napi::TypeError::New(env, "Expected output width/height as numbers").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    outW = info[4].As<Napi::Number>().Int32Value();
    outH = info[5].As<Napi::Number>().Int32Value();
  }
  if (outW <= 0 || outH <= 0) {
    Napi::TypeError::New(env, "Output width/height must be > 0").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (arr.ByteLength() < static_cast<size_t>(outW) * static_cast<size_t>(outH) * 4) {
    Napi::Error::New(env, "Buffer too small").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  int64_t resultPts = dec->seekAndDecode(dec->tsToPts(ts), arr.Data(), outW, outH, kfOnly);
  if (resultPts == AV_NOPTS_VALUE) {
    Napi::Error::New(env, "Could not decode frame").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Object result = Napi::Object::New(env);
  result.Set("timestamp", Napi::Number::New(env, dec->ptsToTs(resultPts)));
  return result;
}

// decodeNextFrame(path, buffer, startTime?, endTime?, outWidth?, outHeight?, decoderKey?) → { timestamp } | null
//
// Sequential decode for playback / batch extraction.
//
// Usage patterns:
//   decodeNextFrame(path, buf)             — next frame from current position
//   decodeNextFrame(path, buf, 5.0)        — seek to 5s, return that frame
//   decodeNextFrame(path, buf, 5.0, 10.0)  — seek to 5s, return frame (call
//                                            again without start to continue;
//                                            returns null when past 10s)
//   decodeNextFrame(path, buf, -1, 10.0)   — continue from current pos, stop at 10s
//
// On first call (or when startTime is provided), seeks to the start position.
// Subsequent calls without startTime decode sequentially — no seeking.
// Returns null at EOF or when the frame exceeds endTime.
Napi::Value DecodeNextFrame(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsString() || !info[1].IsTypedArray()) {
    Napi::TypeError::New(env, "Expected (path, buffer, startTime?, endTime?)").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string filename = info[0].As<Napi::String>().Utf8Value();
  auto arr = info[1].As<Napi::TypedArrayOf<uint8_t>>();
  std::string decoderKey = filename;
  if (info.Length() > 4 && info[4].IsString()) {
    decoderKey = info[4].As<Napi::String>().Utf8Value();
  }
  if (info.Length() > 6 && info[6].IsString()) {
    decoderKey = info[6].As<Napi::String>().Utf8Value();
  }

  // Optional start time: if provided and >= 0, seek to this position
  double startSec = -1.0;
  if (info.Length() > 2 && info[2].IsNumber())
    startSec = info[2].As<Napi::Number>().DoubleValue();

  // Optional end time: if provided and >= 0, stop when frame exceeds this
  double endSec = -1.0;
  if (info.Length() > 3 && info[3].IsNumber())
    endSec = info[3].As<Napi::Number>().DoubleValue();

  VideoDecoder* dec = getDecoder(env, filename, decoderKey);
  if (!dec) return env.Undefined();
  int srcW = dec->videoWidth();
  int srcH = dec->videoHeight();
  if (srcW <= 0 || srcH <= 0) {
    Napi::Error::New(env, "No video stream").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  int outW = srcW;
  int outH = srcH;
  if (info.Length() > 4 && info[4].IsNumber()) {
    if (info.Length() < 6 || !info[4].IsNumber() || !info[5].IsNumber()) {
      Napi::TypeError::New(env, "Expected output width/height as numbers").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    outW = info[4].As<Napi::Number>().Int32Value();
    outH = info[5].As<Napi::Number>().Int32Value();
  }
  if (outW <= 0 || outH <= 0) {
    Napi::TypeError::New(env, "Output width/height must be > 0").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (arr.ByteLength() < static_cast<size_t>(outW) * static_cast<size_t>(outH) * 4) {
    Napi::Error::New(env, "Buffer too small").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  int64_t endPts = (endSec >= 0.0) ? dec->tsToPts(endSec) : AV_NOPTS_VALUE;
  int64_t pts;

  if (startSec >= 0.0) {
    // Seek to start position (uses smart seek — skips if already nearby)
    pts = dec->seekToStart(dec->tsToPts(startSec), arr.Data(), outW, outH);
    // Check end boundary on the first frame too
    if (pts != AV_NOPTS_VALUE && endPts != AV_NOPTS_VALUE &&
        pts > endPts + dec->frameDurationPts() / 2)
      return env.Null();
  } else {
    // Continue sequential decode from current position
    pts = dec->decodeNextFrameInto(arr.Data(), outW, outH, endPts);
  }

  if (pts == AV_NOPTS_VALUE) return env.Null();

  Napi::Object result = Napi::Object::New(env);
  result.Set("timestamp", Napi::Number::New(env, dec->ptsToTs(pts)));
  return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("loadFile", Napi::Function::New(env, LoadFile));
  exports.Set("decodeFrameInto", Napi::Function::New(env, DecodeFrameInto));
  exports.Set("decodeNextFrame", Napi::Function::New(env, DecodeNextFrame));
  return exports;
}

NODE_API_MODULE(addon, Init)
