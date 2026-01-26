import { Rect } from "react-konva";
import { useControlsStore } from "@/lib/control";
import React, { useRef, useMemo, useEffect, useState } from "react";

const SCROLLBAR_HW = 8;

interface ScrollbarProps {
  stageWidth: number;
  stageHeight: number;
  isScrollbarHovered: boolean;
  setIsScrollbarHovered: (isHovered: boolean) => void;
}

const Scrollbar: React.FC<ScrollbarProps> = ({
  stageWidth,
  stageHeight,
  isScrollbarHovered,
  setIsScrollbarHovered,
}) => {
  const controlStore = useControlsStore();
  const scrollbarHeight = SCROLLBAR_HW;
  const { totalTimelineFrames, timelineDuration, zoomLevel, focusFrame } =
    useControlsStore();
  const horizontalScrollStateRef = useRef({
    startX: 0,
    lastX: 0,
    startFrame: 0,
    fractionalFrames: 0,
  });
  const [scrollbarX, setScrollbarX] = useState(0);
  const isDraggingRef = useRef(false);
  const [wasPlaying, setWasPlaying] = useState(false);

  const { scrollbarWidth, maxScrollbarX } = useMemo(() => {
    const [startFrame, endFrame] = timelineDuration;
    const visibleDuration = endFrame - startFrame;
    const scrollbarWidth = Math.max(
      20,
      (visibleDuration / totalTimelineFrames) * stageWidth,
    );
    const maxScrollbarX = Math.max(0, stageWidth - scrollbarWidth);

    return {
      scrollbarWidth,
      maxScrollbarX,
    };
  }, [
    stageWidth,
    stageHeight,
    zoomLevel,
    timelineDuration,
    totalTimelineFrames,
  ]);

  useEffect(() => {
    if (!isDraggingRef.current) {
      const [startFrame, endFrame] = timelineDuration;
      const visibleDuration = endFrame - startFrame;
      const scrollableFrames = totalTimelineFrames - visibleDuration;

      const scrollbarPosition =
        maxScrollbarX > 0 && scrollableFrames > 0
          ? (startFrame / scrollableFrames) * maxScrollbarX
          : 0;

      setScrollbarX(scrollbarPosition);
    }
  }, [focusFrame, timelineDuration, totalTimelineFrames, maxScrollbarX]);

  useEffect(() => {
    horizontalScrollStateRef.current.startFrame = timelineDuration[0];
  }, [timelineDuration]);

  return (
    <Rect
      fill={
        isScrollbarHovered ? "rgba(227,227,227,0.4)" : "rgba(227,227,227,0.1)"
      }
      height={scrollbarHeight}
      x={scrollbarX}
      y={stageHeight - scrollbarHeight}
      cornerRadius={scrollbarHeight}
      width={scrollbarWidth}
      onMouseEnter={() => setIsScrollbarHovered(true)}
      onMouseLeave={() => setIsScrollbarHovered(false)}
      draggable
      dragBoundFunc={(pos) => {
        const clampedX = Math.max(0, Math.min(maxScrollbarX, pos.x));
        return { x: clampedX, y: stageHeight - scrollbarHeight };
      }}
      onDragStart={(e) => {
        isDraggingRef.current = true;
        const x = e.target.x();
        horizontalScrollStateRef.current.startX = x;
        horizontalScrollStateRef.current.startFrame = timelineDuration[0];
        horizontalScrollStateRef.current.lastX = x;
        setWasPlaying(controlStore.isPlaying);
        controlStore.pause();
      }}
      onDragMove={(e) => {
        const x = e.target.x();
        setScrollbarX(x);

        const [startFrame, endFrame] = timelineDuration;
        const visibleDuration = endFrame - startFrame;
        const scrollableFrames = Math.max(
          0,
          totalTimelineFrames - visibleDuration,
        );

        if (maxScrollbarX <= 0 || scrollableFrames <= 0) {
          horizontalScrollStateRef.current.lastX = x;
          return;
        }

        const targetStartFrame = Math.round(
          (x / maxScrollbarX) * scrollableFrames,
        );
        const clampedTarget = Math.max(
          0,
          Math.min(scrollableFrames, targetStartFrame),
        );

        const currentStart = horizontalScrollStateRef.current.startFrame;
        const framesToShift = clampedTarget - currentStart;

        if (framesToShift !== 0) {
          controlStore.shiftTimelineDuration(framesToShift, true);
          horizontalScrollStateRef.current.startFrame =
            currentStart + framesToShift;
        }

        horizontalScrollStateRef.current.lastX = x;
      }}
      onDragEnd={(e) => {
        const x = e.target.x();
        setScrollbarX(x);
        isDraggingRef.current = false;
        if (wasPlaying) {
          controlStore.play();
          setWasPlaying(false);
        }
      }}
    />
  );
};

export default Scrollbar;
