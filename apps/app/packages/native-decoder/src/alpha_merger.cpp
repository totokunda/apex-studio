#include "alpha_merger.h"
#include <cstring>

// Platform-specific SIMD headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define APEX_HAS_SSE 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define APEX_HAS_NEON 1
#endif

namespace apex {

void AlphaMerger::merge(
    uint8_t* rgba,
    const uint8_t* alphaPlane,
    int alphaStride,
    int width,
    int height
) {
#if defined(APEX_HAS_SSE) || defined(APEX_HAS_NEON)
    mergeSIMD(rgba, alphaPlane, alphaStride, width, height);
#else
    mergeScalar(rgba, alphaPlane, alphaStride, width, height);
#endif
}

void AlphaMerger::mergeScalar(
    uint8_t* rgba,
    const uint8_t* alphaPlane,
    int alphaStride,
    int width,
    int height
) {
    for (int y = 0; y < height; y++) {
        const uint8_t* alphaRow = alphaPlane + y * alphaStride;
        uint8_t* rgbaRow = rgba + y * width * 4;

        for (int x = 0; x < width; x++) {
            rgbaRow[x * 4 + 3] = alphaRow[x];
        }
    }
}

void AlphaMerger::mergeSIMD(
    uint8_t* rgba,
    const uint8_t* alphaPlane,
    int alphaStride,
    int width,
    int height
) {
#if defined(APEX_HAS_SSE)
    // SSE4.1 implementation: process 16 pixels at a time
    // Each pixel is 4 bytes (RGBA). We need to scatter 16 alpha bytes
    // into every 4th byte of 64 RGBA bytes.
    for (int y = 0; y < height; y++) {
        const uint8_t* alphaRow = alphaPlane + y * alphaStride;
        uint8_t* rgbaRow = rgba + y * width * 4;

        int x = 0;

        // Process 4 pixels at a time (simplest scatter pattern)
        for (; x + 4 <= width; x += 4) {
            // Load 4 alpha bytes
            uint8_t a0 = alphaRow[x + 0];
            uint8_t a1 = alphaRow[x + 1];
            uint8_t a2 = alphaRow[x + 2];
            uint8_t a3 = alphaRow[x + 3];

            // Load 16 bytes of RGBA (4 pixels)
            __m128i rgbaVec = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(rgbaRow + x * 4));

            // Create alpha mask: [_, _, _, a0, _, _, _, a1, _, _, _, a2, _, _, _, a3]
            __m128i alphaMask = _mm_set_epi8(
                a3, 0, 0, 0,
                a2, 0, 0, 0,
                a1, 0, 0, 0,
                a0, 0, 0, 0
            );

            // Blend: keep RGB from original, take A from alpha mask
            // Alpha channel mask: 0x00000080 per pixel
            __m128i blendMask = _mm_set_epi8(
                static_cast<char>(0x80), 0, 0, 0,
                static_cast<char>(0x80), 0, 0, 0,
                static_cast<char>(0x80), 0, 0, 0,
                static_cast<char>(0x80), 0, 0, 0
            );

            __m128i result = _mm_blendv_epi8(rgbaVec, alphaMask, blendMask);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(rgbaRow + x * 4), result);
        }

        // Handle remaining pixels
        for (; x < width; x++) {
            rgbaRow[x * 4 + 3] = alphaRow[x];
        }
    }

#elif defined(APEX_HAS_NEON)
    // NEON implementation: process 8 pixels at a time
    for (int y = 0; y < height; y++) {
        const uint8_t* alphaRow = alphaPlane + y * alphaStride;
        uint8_t* rgbaRow = rgba + y * width * 4;

        int x = 0;

        for (; x + 8 <= width; x += 8) {
            // Load 8 alpha bytes
            uint8x8_t alpha = vld1_u8(alphaRow + x);

            // Load 8 RGBA pixels (32 bytes) as 4-channel interleaved
            uint8x8x4_t rgbaPixels = vld4_u8(rgbaRow + x * 4);

            // Replace alpha channel
            rgbaPixels.val[3] = alpha;

            // Store back
            vst4_u8(rgbaRow + x * 4, rgbaPixels);
        }

        // Handle remaining pixels
        for (; x < width; x++) {
            rgbaRow[x * 4 + 3] = alphaRow[x];
        }
    }

#else
    mergeScalar(rgba, alphaPlane, alphaStride, width, height);
#endif
}

void AlphaMerger::setOpaque(uint8_t* rgba, int width, int height) {
    const int totalPixels = width * height;

#if defined(APEX_HAS_SSE)
    int i = 0;
    for (; i + 4 <= totalPixels; i += 4) {
        uint8_t* base = rgba + i * 4;
        __m128i rgbaVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base));

        // OR with 0xFF in alpha positions
        __m128i alphaMask = _mm_set_epi8(
            static_cast<char>(0xFF), 0, 0, 0,
            static_cast<char>(0xFF), 0, 0, 0,
            static_cast<char>(0xFF), 0, 0, 0,
            static_cast<char>(0xFF), 0, 0, 0
        );

        __m128i result = _mm_or_si128(rgbaVec, alphaMask);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(base), result);
    }
    for (; i < totalPixels; i++) {
        rgba[i * 4 + 3] = 0xFF;
    }

#elif defined(APEX_HAS_NEON)
    int i = 0;
    uint8x8_t opaqueAlpha = vdup_n_u8(0xFF);
    for (; i + 8 <= totalPixels; i += 8) {
        uint8x8x4_t pixels = vld4_u8(rgba + i * 4);
        pixels.val[3] = opaqueAlpha;
        vst4_u8(rgba + i * 4, pixels);
    }
    for (; i < totalPixels; i++) {
        rgba[i * 4 + 3] = 0xFF;
    }

#else
    for (int i = 0; i < totalPixels; i++) {
        rgba[i * 4 + 3] = 0xFF;
    }
#endif
}

} // namespace apex
