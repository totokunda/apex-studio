/**
 * SharedArrayBuffer pool allocation for the native video decoder.
 *
 * The C++ addon writes decoded RGBA frames directly into these buffers.
 * The renderer reads from them via NativeFrameUploader (WebGL texImage2D)
 * and then releases them back to the pool via ackNativeFrame().
 */

/** Number of frame buffers in each pool — matches MAX_ITERATION_IN_FLIGHT. */
export const NATIVE_POOL_SIZE = 4;

/**
 * Allocate a pool of SharedArrayBuffers sized for RGBA frames at the given dimensions.
 */
export function createBufferPool(width: number, height: number): SharedArrayBuffer[] {
    const bytesPerFrame = width * height * 4; // RGBA: 4 bytes per pixel
    return Array.from({ length: NATIVE_POOL_SIZE }, () => new SharedArrayBuffer(bytesPerFrame));
}
