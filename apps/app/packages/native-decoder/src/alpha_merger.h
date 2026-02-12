#pragma once

#include <cstdint>
#include <cstddef>

namespace apex {

/**
 * Merges a grayscale alpha plane into an RGBA buffer's alpha channel.
 * Uses SIMD intrinsics (SSE4.1 on x86_64, NEON on ARM64) for
 * maximum throughput when processing large frames.
 *
 * The alpha plane is expected to be a Y-plane (luminance) from a
 * decoded alpha video stream. The red channel (or luminance) of each
 * pixel is written into the 4th byte (A) of each RGBA pixel.
 */
class AlphaMerger {
public:
    /**
     * Merge alpha data into an RGBA buffer.
     *
     * @param rgba          RGBA buffer (4 bytes per pixel, modified in place)
     * @param alphaPlane    Grayscale alpha plane (1 byte per pixel, Y channel)
     * @param alphaStride   Stride (bytes per row) of the alpha plane
     * @param width         Frame width in pixels
     * @param height        Frame height in pixels
     */
    static void merge(
        uint8_t* rgba,
        const uint8_t* alphaPlane,
        int alphaStride,
        int width,
        int height
    );

    /**
     * Set all alpha bytes to 255 (fully opaque) in an RGBA buffer.
     */
    static void setOpaque(uint8_t* rgba, int width, int height);

private:
    static void mergeSIMD(
        uint8_t* rgba,
        const uint8_t* alphaPlane,
        int alphaStride,
        int width,
        int height
    );

    static void mergeScalar(
        uint8_t* rgba,
        const uint8_t* alphaPlane,
        int alphaStride,
        int width,
        int height
    );
};

} // namespace apex
