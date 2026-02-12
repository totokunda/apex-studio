#include "hwaccel.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace apex {

bool HwAccelManager::probed_ = false;
std::vector<AVHWDeviceType> HwAccelManager::available_;
AVHWDeviceType HwAccelManager::preferred_ = AV_HWDEVICE_TYPE_NONE;

void HwAccelManager::probeOnce() {
    if (probed_) return;
    probed_ = true;

    // Platform-preferred order
    std::vector<AVHWDeviceType> candidates;

#if defined(_WIN32)
    candidates = {AV_HWDEVICE_TYPE_D3D11VA, AV_HWDEVICE_TYPE_DXVA2};
#elif defined(__APPLE__)
    candidates = {AV_HWDEVICE_TYPE_VIDEOTOOLBOX};
#elif defined(__linux__)
    candidates = {AV_HWDEVICE_TYPE_VAAPI};
#endif

    // Also try CUDA on all platforms (NVIDIA GPUs)
    candidates.push_back(AV_HWDEVICE_TYPE_CUDA);

    for (auto type : candidates) {
        AVBufferRef* hwDeviceCtx = nullptr;
        int ret = av_hwdevice_ctx_create(&hwDeviceCtx, type, nullptr, nullptr, 0);
        if (ret >= 0) {
            available_.push_back(type);
            av_buffer_unref(&hwDeviceCtx);

            if (preferred_ == AV_HWDEVICE_TYPE_NONE) {
                preferred_ = type;
            }
        }
    }
}

AVHWDeviceType HwAccelManager::getPreferredDeviceType() {
    probeOnce();
    return preferred_;
}

std::string HwAccelManager::getPreferredMethodName() {
    probeOnce();
    if (preferred_ == AV_HWDEVICE_TYPE_NONE) return "software";
    const char* name = av_hwdevice_get_type_name(preferred_);
    return name ? name : "unknown";
}

std::vector<std::string> HwAccelManager::getAvailableMethods() {
    probeOnce();
    std::vector<std::string> result;
    result.reserve(available_.size());
    for (auto type : available_) {
        const char* name = av_hwdevice_get_type_name(type);
        if (name) result.push_back(name);
    }
    if (result.empty()) {
        result.push_back("software");
    }
    return result;
}

int HwAccelManager::initHwContext(AVCodecContext* codecCtx, AVHWDeviceType type) {
    if (type == AV_HWDEVICE_TYPE_NONE) return -1;

    AVBufferRef* hwDeviceCtx = nullptr;
    int ret = av_hwdevice_ctx_create(&hwDeviceCtx, type, nullptr, nullptr, 0);
    if (ret < 0) return ret;

    codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx);
    av_buffer_unref(&hwDeviceCtx);

    return 0;
}

AVPixelFormat HwAccelManager::getHwFormat(
    AVCodecContext* ctx,
    const AVPixelFormat* pixFmts
) {
    // The expected hw pixel format is stored in the codec context's opaque field
    AVPixelFormat expectedFmt = AV_PIX_FMT_NONE;

    if (ctx->hw_device_ctx) {
        auto* hwFramesCtx = reinterpret_cast<AVHWDeviceContext*>(
            ctx->hw_device_ctx->data);
        if (hwFramesCtx) {
            expectedFmt = hwPixelFormat(hwFramesCtx->type);
        }
    }

    for (const AVPixelFormat* p = pixFmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == expectedFmt) return *p;
    }

    // Fallback to first software format
    return pixFmts[0];
}

AVPixelFormat HwAccelManager::hwPixelFormat(AVHWDeviceType type) {
    switch (type) {
#if defined(_WIN32)
        case AV_HWDEVICE_TYPE_D3D11VA: return AV_PIX_FMT_D3D11;
        case AV_HWDEVICE_TYPE_DXVA2:   return AV_PIX_FMT_DXVA2_VLD;
#endif
#if defined(__APPLE__)
        case AV_HWDEVICE_TYPE_VIDEOTOOLBOX: return AV_PIX_FMT_VIDEOTOOLBOX;
#endif
#if defined(__linux__)
        case AV_HWDEVICE_TYPE_VAAPI:   return AV_PIX_FMT_VAAPI;
#endif
        case AV_HWDEVICE_TYPE_CUDA:    return AV_PIX_FMT_CUDA;
        default: return AV_PIX_FMT_NONE;
    }
}

} // namespace apex
