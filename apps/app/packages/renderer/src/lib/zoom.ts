import { ZoomLevel } from "./types";

interface ZoomLevelConfig {
  minorTickInterval: number;
  majorTickInterval: number;
  minorTickFormat: "frame" | "second";
  majorTickFormat: "frame" | "second";
}

// Calculate dynamic tick intervals based on duration and zoom level
export const getZoomLevelConfig = (
  zoomLevel: ZoomLevel,
  timelineDuration: [number, number],
  fps: number = 24,
  maxZoomLevel: ZoomLevel = 10,
  minZoomLevel: ZoomLevel = 1,
): ZoomLevelConfig => {
  // Target tick counts: 50 ticks at zoom 1, 10 ticks at zoom 10
  const targetTicks = 50 - (zoomLevel - 1) * 4.4; // Linear progression from 50 to ~10
  const [startFrame, endFrame] = timelineDuration;
  // Convert duration from frames to seconds for calculation
  const durationInSeconds = (endFrame - startFrame) / fps;

  // Calculate minor tick interval to achieve target tick count
  let minorTickInterval = durationInSeconds / targetTicks;
  let format: "frame" | "second" = "second";

  // For higher zoom levels (7-10), switch to frame-based ticks
  const seventyPercentThresh =
    minZoomLevel + (maxZoomLevel - minZoomLevel) * 0.7;
  if (zoomLevel >= seventyPercentThresh) {
    format = "frame";
    minorTickInterval = Math.max(
      1,
      Math.round((endFrame - startFrame) / targetTicks),
    );
  } else {
    // For second-based ticks, ensure intervals convert to whole frames
    // Calculate what the interval would be in frames
    const intervalInFrames = minorTickInterval * fps;

    // Round to nice frame intervals that divide evenly
    if (intervalInFrames >= fps) {
      // 1+ seconds - round to whole seconds
      minorTickInterval = Math.round(minorTickInterval);
    } else if (intervalInFrames >= fps / 2) {
      // 0.5 seconds = 12 frames at 24fps
      minorTickInterval = 0.5;
    } else if (intervalInFrames >= fps / 4) {
      // 0.25 seconds = 6 frames at 24fps
      minorTickInterval = 0.25;
    } else if (intervalInFrames >= fps / 8) {
      // 0.125 seconds = 3 frames at 24fps - problematic, skip to larger interval
      minorTickInterval = 0.25; // Use fewer ticks with better spacing
    } else {
      // Very small intervals - use 0.25 seconds for cleaner spacing
      minorTickInterval = 0.25;
    }
  }

  // Major tick interval is typically 5-10x the minor interval
  const majorTickInterval =
    format === "frame"
      ? Math.max(2, Math.round(minorTickInterval * 5))
      : minorTickInterval * 10;

  return {
    minorTickInterval,
    majorTickInterval,
    minorTickFormat: format,
    majorTickFormat: format,
  };
};
