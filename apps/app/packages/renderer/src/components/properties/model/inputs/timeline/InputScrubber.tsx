import React, { useState, useRef, useEffect } from "react";
import { useAssetControlsStore } from "@/lib/assetControl";
import { getZoomLevelConfig } from "@/lib/zoom";
import { cn } from "@/lib/utils";

interface AssetScrubControlProps {
  stageHeight: number;
  stageWidth: number;
}

const AssetScrubControl: React.FC<AssetScrubControlProps> = ({
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
  } = useAssetControlsStore();
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState(16);
  const [currentFrame, setCurrentFrame] = useState(0);
  // Slightly smaller geometry than the main scrubber
  const startPadding = 24; // keep aligned with ticks
  const handleHalf = 5; // smaller top handle half-width (~10px total)
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
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
    const rawCenterX = e.clientX - rect.left;
    const minCenterX = startPadding;
    const maxCenterX = stageWidth - containerRef.current.offsetWidth / 2;
    const centerX = Math.max(minCenterX, Math.min(rawCenterX, maxCenterX));
    setPosition(centerX - handleHalf);
    const anchorRatio = (centerX - startPadding) / stageWidth;
    setFocusAnchorRatio(anchorRatio);
    const progress = (centerX - startPadding) / stageWidth;
    const [startFrame, endFrame] = timelineDuration;
    const framePosition = startFrame + progress * (endFrame - startFrame);
    const snapped = Math.round(framePosition);
    setCurrentFrame(snapped);
    setFocusFrame(snapped);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    const config = getZoomLevelConfig(zoomLevel, timelineDuration, fps);
    const isFrameMode =
      config.minorTickFormat === "frame" || config.majorTickFormat === "frame";
    const currentCenterX = position + handleHalf;
    const minCenterX = startPadding;
    const maxCenterX = startPadding + stageWidth;
    let targetCenterX = currentCenterX;
    const [startFrame, endFrame] = timelineDuration;
    if (isFrameMode) {
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
      const progress = (currentFrame - startFrame) / (endFrame - startFrame);
      targetCenterX = startPadding + progress * stageWidth;
    }
    const clampedCenterX = Math.max(
      minCenterX,
      Math.min(targetCenterX, maxCenterX),
    );
    setPosition(clampedCenterX - handleHalf);
    const progress = (clampedCenterX - startPadding) / stageWidth;
    const framePosition = startFrame + progress * (endFrame - startFrame);
    setFocusFrame(Math.round(framePosition));
    setFocusAnchorRatio(progress);
  };

  useEffect(() => {
    if (!containerRef.current) return;
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
            "w-2.5 h-3 rounded-t-sm rounded-b-[6px] border-2 border-white shadow-md bg-white",
          )}
        ></div>
        <div
          style={{ height: stageHeight }}
          className={cn(
            "rounded-b-full flex items-center w-[0.08rem] bg-white shadow-md",
          )}
        ></div>
      </div>
    </div>
  );
};

export default AssetScrubControl;
