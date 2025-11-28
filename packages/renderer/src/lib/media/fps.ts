type Rounding = "floor" | "round" | "ceil";
type RangeMode = "openEnd" | "closed";

interface ConvertRangeOptions {
  /** Treat input as [start, end) if "openEnd" (default) or [start, end] if "closed". */
  rangeMode?: RangeMode;
  /** Rounding for the converted start frame (default: "floor"). */
  startRounding?: Rounding;
  /** Rounding for the converted end frame (default: "ceil"). */
  endRounding?: Rounding;
  /**
   * If you know the true duration (in seconds), provide it for maximum accuracy.
   * When set, it overrides end time derived from frames.
   * Example: 81 frames @ 16fps → durationSeconds = 81 / 16 = 5.0625
   */
  durationSeconds?: number;
}

/**
 * Convert a frame range from src FPS to dst FPS.
 * Defaults to half-open ranges: [start, end).
 * If `durationSeconds` is provided, it is used instead of deriving the end from frames.
 */
export function convertFrameRange(
  startFrame: number,
  endFrame: number,
  srcFps: number,
  dstFps: number,
  options: ConvertRangeOptions = {},
): { start: number; end: number } {
  if (srcFps <= 0 || dstFps <= 0) throw new Error("FPS must be positive.");
  if (!Number.isFinite(startFrame) || !Number.isFinite(endFrame)) {
    throw new Error("Frames must be finite numbers.");
  }

  const {
    rangeMode = "openEnd",
    startRounding = "floor",
    endRounding = "ceil",
    durationSeconds,
  } = options;

  const apply = (v: number, mode: Rounding) =>
    mode === "floor"
      ? Math.floor(v)
      : mode === "ceil"
        ? Math.ceil(v)
        : Math.round(v);

  // Start time from start frame
  const startSec = startFrame / srcFps;

  // End time:
  //   - If real duration provided, use start + durationSeconds (most accurate).
  //   - Else derive from frames; for closed ranges, +1 frame to make it exclusive.
  const derivedEndSec =
    (rangeMode === "closed" ? endFrame + 1 : endFrame) / srcFps;

  const endSec =
    durationSeconds != null ? startSec + durationSeconds : derivedEndSec;

  // Map to destination frames with chosen rounding
  const startOut = apply(startSec * dstFps, startRounding);
  const endOut = apply(endSec * dstFps, endRounding);

  return { start: Math.max(0, startOut), end: Math.max(startOut, endOut) };
}

/** Convert a single frame index between FPS (point mapping). */
export function convertFrameIndex(
  frame: number,
  srcFps: number,
  dstFps: number,
  rounding: Rounding = "round",
): number {
  if (srcFps <= 0 || dstFps <= 0) throw new Error("FPS must be positive.");
  const t = frame / srcFps;
  const f = t * dstFps;
  return rounding === "floor"
    ? Math.floor(f)
    : rounding === "ceil"
      ? Math.ceil(f)
      : Math.round(f);
}

export const toFrameRange = (
  startFrame: number,
  endFrame: number,
  srcFps: number,
  dstFps: number,
  durationSeconds: number,
): { start: number; end: number } => {
  const startSec = startFrame / srcFps;
  const endSec = endFrame / srcFps;
  const maxEnd = Math.round(durationSeconds * dstFps);
  // we have it converted to seconds,
  const startOut = Math.round(startSec * dstFps);
  const endOut = Math.min(Math.round(endSec * dstFps), maxEnd);
  return { start: startOut, end: endOut };
};
