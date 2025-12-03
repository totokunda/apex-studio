import React, { useState, useRef, useEffect } from "react";
import { useControlsStore } from "@/lib/control";
import { getZoomLevelConfig } from "@/lib/zoom";
import { cn } from "@/lib/utils";

interface ScrubControlProps {
  stageHeight: number;
  stageWidth: number;
}

export const ScrubControl: React.FC<ScrubControlProps> = ({
  stageHeight,
  stageWidth,
}) => {
  const {
    timelineDuration,
    fps,
    zoomLevel,
    focusFrame,
    setFocusFrame,
    setFocusAnchorRatio,
    isPlaying,
    pause,
    play,
    isFullscreen,
    setPossibleKeyFocusFrames,
    setIsAccurateSeekNeeded,
  } = useControlsStore();
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState(18);
  const [currentFrame, setCurrentFrame] = useState(0);
  // Geometry constants: keep in sync with timeline ticks padding and handle width
  const startPadding = 24; // ticks start at x = 24
  const handleHalf = 6; // half of top handle width (w-3 ≈ 12px)
  const containerRef = useRef<HTMLDivElement>(null);
  const [wasPlaying, setWasPlaying] = useState(false);

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
    setIsAccurateSeekNeeded(false);
    if (isPlaying) {
      setWasPlaying(true);
      pause();
    }
    // pause the video
    // Initialize anchor and focus based on current position
    const centerX = position + handleHalf;
    const progress = (centerX - startPadding) / stageWidth;
    const [startFrame, endFrame] = timelineDuration;
    const framePosition = startFrame + progress * (endFrame - startFrame);
    setFocusFrame(Math.round(framePosition));
    setFocusAnchorRatio(Math.max(0, Math.min(1, progress)));
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging || !containerRef.current) return;

    const container = containerRef.current.parentElement;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    // Compute desired center X in the same coordinate system as ticks (includes startPadding)
    const rawCenterX = e.clientX - rect.left;
    const minCenterX = startPadding;
    const maxCenterX = stageWidth - containerRef.current.offsetWidth / 2;
    const centerX = Math.max(minCenterX, Math.min(rawCenterX, maxCenterX));

    // Move handle smoothly by setting left to center - half width
    setPosition(centerX - handleHalf);
    // Persist anchor ratio so zoom can preserve relative alignment
    const anchorRatio = (centerX - startPadding) / stageWidth;
    setFocusAnchorRatio(anchorRatio);

    // Track nearest frame under the cursor (no snapping yet)
    const progress = (centerX - startPadding) / stageWidth;
    const [startFrame, endFrame] = timelineDuration;
    const framePosition = startFrame + progress * (endFrame - startFrame);
    const snapped = Math.round(framePosition);
    setCurrentFrame(snapped);
    setFocusFrame(snapped);
  };

  const handleMouseUp = () => {
    setIsDragging(false);

    // Determine snapping strategy based on zoom config (frame vs second)
    const config = getZoomLevelConfig(zoomLevel, timelineDuration, fps);
    const isFrameMode =
      config.minorTickFormat === "frame" || config.majorTickFormat === "frame";

    // Current center in pixels
    const currentCenterX = position + handleHalf;
    const minCenterX = startPadding;
    const maxCenterX = startPadding + stageWidth;
    let targetCenterX = currentCenterX;
    const [startFrame, endFrame] = timelineDuration;

    if (isFrameMode) {
      // Build frame tick positions (use minor interval for density)
      const frameStep = Math.max(
        1,
        Math.round(
          config.minorTickInterval *
            (config.minorTickFormat === "second" ? fps : 1),
        ),
      );
      let nearest = currentCenterX;
      let nearestDist = Number.POSITIVE_INFINITY;

      for (let f = startFrame; f <= endFrame; f += frameStep) {
        const progress = (f - startFrame) / (endFrame - startFrame);
        const tickCenterX = startPadding + progress * stageWidth;
        if (tickCenterX < minCenterX - 1 || tickCenterX > maxCenterX + 1)
          continue;
        const d = Math.abs(tickCenterX - currentCenterX);
        if (d < nearestDist) {
          nearestDist = d;
          nearest = tickCenterX;
        }
      }
      targetCenterX = nearest;
    } else {
      // Approximate nearest frame position when ticks are in seconds
      const progress = (currentFrame - startFrame) / (endFrame - startFrame);
      targetCenterX = startPadding + progress * stageWidth;
    }

    // Clamp and apply (align handle center to target)
    const clampedCenterX = Math.max(
      minCenterX,
      Math.min(targetCenterX, maxCenterX),
    );
    setPosition(clampedCenterX - handleHalf);
    // Update focus frame and anchor to snapped values
    const progress = (clampedCenterX - startPadding) / stageWidth;
    const framePosition = startFrame + progress * (endFrame - startFrame);
    
    setIsAccurateSeekNeeded(true);
    setFocusFrame(Math.round(framePosition));

    setFocusAnchorRatio(progress);

    if (wasPlaying) {
      play();
      setWasPlaying(false);
    }
  };

  // Reposition scrubber when the visible range or focus frame changes (e.g., zoom in/out)
  useEffect(() => {
    if (!containerRef.current) return;
    // Keep the scrubber centered over the focusFrame in the new viewport
    const minCenterX = startPadding;
    const maxCenterX = startPadding + stageWidth;
    const [startFrame, endFrame] = timelineDuration;
    const centerX =
      startPadding +
      (stageWidth * (focusFrame - startFrame)) / (endFrame - startFrame);
    const clampedCenterX = Math.max(minCenterX, Math.min(centerX, maxCenterX));
    if (!isDragging) {
      setPosition(clampedCenterX - handleHalf);
    }
  }, [timelineDuration, stageWidth, focusFrame, isDragging]);

  // Add global event listeners for mouse move and up
  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, timelineDuration, stageWidth, currentFrame]);

  useEffect(() => {
    // Only calculate if we have valid dimensions
    if (stageWidth <= 0) return;

    const [startFrame, endFrame] = timelineDuration;
    // We can interact with at most stageWidth pixels (assuming 1px resolution)
    // The startPadding determines where the timeline effectively starts in the coordinate system
    // The handle movement is clamped between startPadding and stageWidth - startPadding (roughly)
    // However, the mouse move logic maps [startPadding, startPadding + stageWidth] to [startFrame, endFrame]
    // See handleMouseMove: progress = (centerX - startPadding) / stageWidth
    
    // We want to simulate every possible integer pixel position for centerX
    // The range of valid centerX values in handleMouseMove is effectively constrained by the container
    // But conceptually, the progress goes from 0 to 1 across 'stageWidth' pixels
    
    const frames = new Set<number>();
    
    // Iterate through every possible pixel offset relative to the start of the timeline area
    for (let pixelOffset = 0; pixelOffset <= stageWidth; pixelOffset++) {
        const progress = pixelOffset / stageWidth;
        const framePosition = startFrame + progress * (endFrame - startFrame);
        const snapped = Math.round(framePosition);
        frames.add(snapped);
    }
    
    setPossibleKeyFocusFrames(Array.from(frames).sort((a, b) => a - b));
  }, [timelineDuration, stageWidth, setPossibleKeyFocusFrames]);

  if (isFullscreen) return null;

  return (
    <div
      ref={containerRef}
      onMouseDown={handleMouseDown}
      style={{ left: Math.min(position, stageWidth) }}
      className="absolute flex z-50 top-0 cursor-col-resize"
    >
      <div className="flex flex-col items-center">
        <div
          className={cn(
            "w-3 h-4 rounded-t-sm rounded-b-lg border-2 border-white shadow-md bg-white",
          )}
        ></div>
        <div
          style={{ height: stageHeight }}
          className={cn(
            "rounded-b-full flex items-center w-[0.10rem] bg-white shadow-md",
          )}
        ></div>
      </div>
    </div>
  );
};
