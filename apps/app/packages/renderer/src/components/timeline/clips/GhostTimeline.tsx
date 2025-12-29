import { useClipStore, getClipWidth, getClipX } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { TimelineProps } from "@/lib/types";
import React, { useMemo } from "react";
import { Rect, Line } from "react-konva";

const GhostTimeline: React.FC<TimelineProps> = ({
  timelineY,
  timelineHeight,
  timelineId,
  timelineWidth,
  timelinePadding = 24,
}) => {
  const {
    ghostStartEndFrame,
    ghostX,
    ghostTimelineId,
    getClipsForTimeline,
    draggingClipId,
    ghostInStage,
  } = useClipStore();

  
  const { timelineDuration } = useControlsStore();
  const ghostWidth = useMemo(
    () =>
      getClipWidth(
        ghostStartEndFrame[0],
        ghostStartEndFrame[1],
        timelineWidth!,
        timelineDuration,
      ),
    [ghostStartEndFrame, timelineWidth, timelineDuration],
  );
  const { getClipById } = useClipStore();
  const currentClip = getClipById(draggingClipId!);
  const cornerRadius = useMemo(() => {
    return 1;
  }, [currentClip?.type]);

  const gapBoundaryPadding = useMemo(() => {
    return timelineDuration[0] === 0 ? timelinePadding : 0;
  }, [currentClip?.type]);

  const computedGuideLines = useMemo(() => {
    if (!ghostInStage || ghostTimelineId !== timelineId) return null;
    const [tStart, tEnd] = timelineDuration;
    const stageWidth = timelineWidth || 0;
    // Build merged occupied intervals in px (inner coordinates)
    const clips = getClipsForTimeline(timelineId).filter(
      (c) => c.clipId !== draggingClipId,
    );
    const intervals = clips
      .map((c) => {
        const sx = getClipX(c.startFrame || 0, c.endFrame || 0, stageWidth, [
          tStart,
          tEnd,
        ]);
        const sw = getClipWidth(
          c.startFrame || 0,
          c.endFrame || 0,
          stageWidth,
          [tStart, tEnd],
        );
        const lo = Math.max(0, Math.min(stageWidth, sx));
        const hi = Math.max(0, Math.min(stageWidth, sx + sw));
        return hi > lo ? ([lo, hi] as [number, number]) : null;
      })
      .filter(Boolean) as [number, number][];

    intervals.sort((a, b) => a[0] - b[0]);
    const merged: [number, number][] = [];
    for (const [lo, hi] of intervals) {
      if (merged.length === 0) merged.push([lo, hi]);
      else {
        const last = merged[merged.length - 1];
        if (lo <= last[1]) last[1] = Math.max(last[1], hi);
        else merged.push([lo, hi]);
      }
    }
    const gaps: [number, number][] = [];
    let prev = 0;
    for (const [lo, hi] of merged) {
      if (lo > prev) gaps.push([prev, lo]);
      prev = Math.max(prev, hi);
    }
    if (prev < stageWidth) gaps.push([prev, stageWidth]);
    const validGaps = gaps.filter(([lo, hi]) => hi - lo >= ghostWidth);
    // Choose gap such that the ghost's LEFT edge (ghostX) lies within [gLo, gHi - ghostWidth]
    let chosen: [number, number] | null = null;
    const left = ghostX || 0;
    for (const g of validGaps) {
      if (left >= g[0] && left <= g[1] - ghostWidth) {
        chosen = g;
        break;
      }
    }
    if (!chosen) {
      // No fitting gap: prevent overriding on the left. If the pointer is inside an
      // occupied interval, snap gLo to that interval's hi; otherwise to the last hi before pointer.
      let gLo = 0;
      for (const [lo, hi] of merged) {
        if (left < lo) break;
        gLo = Math.max(gLo, hi);
        if (left <= hi) break;
      }
      chosen = [gLo, stageWidth];
    }
    return chosen;
  }, [
    ghostTimelineId,
    timelineId,
    ghostX,
    ghostWidth,
    getClipsForTimeline,
    timelineDuration,
    timelineWidth,
    draggingClipId,
    ghostInStage,
  ]);

  // For preprocessor timelines, use ghostGuideLines from store; for media timelines, use computed guidelines
  const guideLines = computedGuideLines;

  return (
    <React.Fragment>
      {ghostInStage && ghostTimelineId === timelineId && guideLines && (
        <>
          {/* Highlight the valid drop gap */}
          <Rect
            listening={false}
            x={guideLines[0] + gapBoundaryPadding}
            y={timelineY! - timelineHeight!}
            width={guideLines[1] - guideLines[0]}
            height={timelineHeight}
            fill={"rgba(255,255,255,0.12)"}
            cornerRadius={cornerRadius}
          />
        </>
      )}
      {ghostInStage && ghostTimelineId === timelineId && (
        <Rect
          x={ghostX + gapBoundaryPadding}
          y={timelineY! - timelineHeight!}
          cornerRadius={cornerRadius}
          width={ghostWidth}
          height={timelineHeight}
          fill={"rgba(255, 255, 255, 0.3)"}
          stroke={"#FFFFFF"}
          strokeWidth={2}
          shadowColor={"#FFFFFF"}
          shadowBlur={8}
          shadowOpacity={0.35}
        />
      )}
      {ghostInStage && ghostTimelineId === timelineId && guideLines && (
        <>
          {/* Gap boundary guidelines - extend 10px above and below */}
          <Line
            listening={false}
            points={[
              guideLines[0] + gapBoundaryPadding,
              timelineY! - timelineHeight!,
              guideLines[0] + gapBoundaryPadding,
              timelineY!,
            ]}
            stroke={"#FFFFFF"}
            strokeWidth={1.5}
            dash={[4, 3]}
            shadowColor={"#FFFFFF"}
            shadowBlur={6}
            shadowOpacity={0.4}
          />
          <Line
            listening={false}
            points={[
              guideLines[1] + gapBoundaryPadding,
              timelineY! - timelineHeight!,
              guideLines[1] + gapBoundaryPadding,
              timelineY!,
            ]}
            stroke={"#FFFFFF"}
            strokeWidth={1.5}
            dash={[4, 3]}
            shadowColor={"#FFFFFF"}
            shadowBlur={6}
            shadowOpacity={0.4}
          />
        </>
      )}
    </React.Fragment>
  );
};

export default GhostTimeline;
