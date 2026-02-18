/**
 * MLT Backend - Zero-copy YUV420p frame delivery via SharedArrayBuffer
 *
 * Workflow:
 *   1. probe(filePath) → { width, height, fps, frameSize }
 *   2. Allocate SharedArrayBuffer
 *   3. load(filePath, sab, slotCount) — starts decoding into shared memory
 *   4. Read frames via FrameReader (poll or Atomics.wait)
 *   5. stop() — tears everything down
 */

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

const HEADER_INTS = 8;
const HEADER_BYTES = HEADER_INTS * 4;

interface ProbeResult {
  width: number;
  height: number;
  fps: number;
  frameSize: number;
}

interface NativeAddon {
  probe: (path: string) => ProbeResult;
  load: (path: string, sab: SharedArrayBuffer, slotCount: number) => void;
  stop: () => void;
}

function loadAddon(): NativeAddon {
  const addonPath = join(__dirname, "..", "build", "Release", "addon.node");
  return createRequire(import.meta.url)(addonPath);
}

let addon: NativeAddon | null = null;
function getAddon(): NativeAddon {
  if (!addon) addon = loadAddon();
  return addon;
}

// ─── Public API ─────────────────────────────────────────────────────────────

export function probe(path: string): ProbeResult {
  return getAddon().probe(path);
}

export interface FrameReader {
  /** Header view — read write_index, read_index, width, height, etc. */
  header: Int32Array;
  /** Raw frame slot data (YUV420p planes packed contiguously) */
  slots: Uint8Array;
  /** The SharedArrayBuffer backing everything */
  sab: SharedArrayBuffer;
  /** Frame dimensions */
  width: number;
  height: number;
  frameSize: number;
  slotCount: number;
  /**
   * Try to read the next frame.  Returns a Uint8Array subview into the
   * slot (zero-copy) or null if no new frame is available.
   *
   * IMPORTANT: the returned view is only valid until you call readFrame()
   * again or the slot is overwritten.  If you need to keep it, copy it.
   */
  readFrame(): Uint8Array | null;
  /** Number of frames C++ dropped because JS wasn't reading fast enough */
  droppedFrames(): number;
}

export function load(path: string, slotCount: number = 3): FrameReader {
  const info = probe(path);
  const sabSize = HEADER_BYTES + slotCount * info.frameSize;
  const sab = new SharedArrayBuffer(sabSize);

  const header = new Int32Array(sab, 0, HEADER_INTS);
  const slots = new Uint8Array(sab, HEADER_BYTES);

  getAddon().load(path, sab, slotCount);

  return {
    header,
    slots,
    sab,
    width: info.width,
    height: info.height,
    frameSize: info.frameSize,
    slotCount,

    readFrame(): Uint8Array | null {
      const wi = Atomics.load(header, 0);
      const ri = Atomics.load(header, 1);
      if (wi === ri) return null; // no new frame

      const slot = ri % slotCount;
      const offset = slot * info.frameSize;
      const view = slots.subarray(offset, offset + info.frameSize);

      // Advance read index
      Atomics.store(header, 1, ri + 1);
      return view;
    },

    droppedFrames(): number {
      return Atomics.load(header, 6);
    },
  };
}

export function stop() {
  return getAddon().stop();
}