#pragma once

#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

namespace apex {

/**
 * Manages hardware-accelerated video decoding across platforms.
 *
 * Probes available hardware acceleration methods on first use and provides
 * helpers to initialize hardware device contexts for AVCodecContext instances.
 *
 * Platform strategy:
 *   Windows:  D3D11VA → DXVA2 → software
 *   macOS:    VideoToolbox → software
 *   Linux:    VA-API → software
 */
class HwAccelManager {
public:
    /**
     * Get the best available hardware device type for this platform.
     * Returns AV_HWDEVICE_TYPE_NONE if no hardware acceleration is available.
     */
    static AVHWDeviceType getPreferredDeviceType();

    /**
     * Get a human-readable name for the preferred method.
     */
    static std::string getPreferredMethodName();

    /**
     * Get a list of all available hardware acceleration method names.
     */
    static std::vector<std::string> getAvailableMethods();

    /**
     * Initialize a hardware device context on the given codec context.
     * Returns 0 on success, negative on failure.
     *
     * On failure, the caller should fall back to software decoding.
     */
    static int initHwContext(AVCodecContext* codecCtx, AVHWDeviceType type);

    /**
     * get_format callback for AVCodecContext. Selects the hardware pixel format
     * if available, otherwise falls back to software formats.
     */
    static AVPixelFormat getHwFormat(
        AVCodecContext* ctx,
        const AVPixelFormat* pixFmts
    );

    /**
     * Returns the expected hardware pixel format for the given device type.
     */
    static AVPixelFormat hwPixelFormat(AVHWDeviceType type);

private:
    static void probeOnce();
    static bool probed_;
    static std::vector<AVHWDeviceType> available_;
    static AVHWDeviceType preferred_;
};

} // namespace apex
