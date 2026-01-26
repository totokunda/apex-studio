import { PreprocessorClipProps } from "./types";

/**
 * Converts an x position on the timeline to a frame number
 */
export function calculateFrameFromX(
  xPosition: number,
  timelinePadding: number,
  timelineWidth: number,
  timelineDuration: [number, number],
): number {
  const timelineX = xPosition - timelinePadding;
  const [startFrame, endFrame] = timelineDuration;
  const framePosition =
    (timelineX / timelineWidth) * (endFrame - startFrame) + startFrame;
  return Math.round(framePosition);
}

/**
 * Gets all preprocessors for a clip excluding a specific one
 */
export function getOtherPreprocessors(
  allPreprocessors: PreprocessorClipProps[],
  excludeId: string,
): PreprocessorClipProps[] {
  return allPreprocessors
    .filter((p) => p.id !== excludeId)
    .sort((a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0));
}

/**
 * Detects collisions between a target range and other preprocessors
 */
export function detectCollisions(
  targetStart: number,
  targetEnd: number,
  otherPreprocessors: PreprocessorClipProps[],
  clipDuration: number,
): PreprocessorClipProps[] {
  return otherPreprocessors.filter((p) => {
    const pStart = p.startFrame ?? 0;
    const pEnd = p.endFrame ?? clipDuration;
    return !(targetEnd <= pStart || targetStart >= pEnd);
  });
}

/**
 * Finds a gap after a block of preprocessors where a preprocessor can fit
 */
export function findGapAfterBlock(
  collidingPreprocessors: PreprocessorClipProps[],
  direction: "left" | "right",
  preprocessorDuration: number,
  clipDuration: number,
): number | null {
  if (collidingPreprocessors.length === 0) return null;

  const sorted = [...collidingPreprocessors].sort(
    (a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0),
  );

  let blockStart = sorted[0].startFrame ?? 0;
  let blockEnd = sorted[0].endFrame ?? clipDuration;

  for (let i = 1; i < sorted.length; i++) {
    const current = sorted[i];
    const currentStart = current.startFrame ?? 0;
    const currentEnd = current.endFrame ?? clipDuration;

    if (currentStart <= blockEnd) {
      blockEnd = Math.max(blockEnd, currentEnd);
    }
  }

  if (direction === "right") {
    const gapStart = blockEnd;
    const gapEnd = clipDuration;
    const availableSpace = gapEnd - gapStart;

    if (availableSpace >= preprocessorDuration) {
      return gapStart;
    }
  } else {
    const gapStart = 0;
    const gapEnd = blockStart;
    const availableSpace = gapEnd - gapStart;

    if (availableSpace >= preprocessorDuration) {
      return gapEnd - preprocessorDuration;
    }
  }
  return null;
}

/**
 * Validates that a preprocessor's updated frames are valid
 * Returns an object with isValid boolean and optional error message
 */
export function validatePreprocessorFrames(
  startFrame: number,
  endFrame: number,
  preprocessorId: string,
  allPreprocessors: PreprocessorClipProps[],
  clipDuration: number,
): { isValid: boolean; error?: string } {
  // Check minimum duration
  if (endFrame - startFrame < 1) {
    return {
      isValid: false,
      error: "Preprocessor must be at least 1 frame long",
    };
  }

  // Check bounds
  if (startFrame < 0) {
    return { isValid: false, error: "Start frame cannot be before clip start" };
  }

  if (endFrame > clipDuration) {
    return { isValid: false, error: "End frame cannot exceed clip duration" };
  }

  // Check for collisions with other preprocessors
  const otherPreprocessors = getOtherPreprocessors(
    allPreprocessors,
    preprocessorId,
  );
  const collisions = detectCollisions(
    startFrame,
    endFrame,
    otherPreprocessors,
    clipDuration,
  );

  if (collisions.length > 0) {
    return { isValid: false, error: "Overlaps with another preprocessor" };
  }

  return { isValid: true };
}
