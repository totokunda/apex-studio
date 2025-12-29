import React, { useEffect, useMemo, useRef, useState } from "react";
import { useClipStore, getTimelineTypeForClip } from "@/lib/clip";
import { ClipType } from "@/lib/types";
import { Stage, Layer, Line, Rect } from "react-konva";
import Timeline from "@/components/timeline/Timeline";
import Konva from "konva";
import TimelineMoments from "./InputTimelineMoments";
import Scrollbar from "@/components/properties/model/inputs/timeline/Scrollbar";
import AssetScrubControl from "@/components/properties/model/inputs/timeline/InputScrubber";
import TimelineZoom from "./InputTimelineZoom";
import { useAssetControlsStore } from "@/lib/assetControl";
// import { getZoomLevelConfig } from '@/lib/zoom';

interface TimelineSearchProps {
  types: ClipType[];
  width?: number;
  height?: number;
  excludeClipId?: string | null;
  isAssetSelected?: (clipId: string) => boolean;
}

const SCROLLBAR_HW = 8;

const TimelineSearch: React.FC<TimelineSearchProps> = ({
  types,
  width = 580,
  height = 300,
  excludeClipId = null,
  isAssetSelected = () => false,
}) => {
  const timelinesLayerRef = useRef<Konva.Layer>(null);
  const [clampedScroll, setClampedScroll] = useState<number>(0);
  const [verticalScroll, setVerticalScroll] = useState<number>(0);
  const [isScrollbarHovered, setIsScrollbarHovered] = useState<boolean>(false);
  const [snapGuideX] = useState<number | null>(null);
  const originalTimelines = useClipStore((s) => s.timelines);
  const getClipsForTimeline = useClipStore((s) => s.getClipsForTimeline);

  const allowedTimelineTypes = useMemo(() => {
    const set = new Set<string>();
    (types || []).forEach((t) => set.add(getTimelineTypeForClip(t)));
    return set;
  }, [types]);

  const timelines = useMemo(() => {
    const SCALE = 0.8; // shrink timelines for search view
    if (!originalTimelines || originalTimelines.length === 0)
      return [] as typeof originalTimelines;

    // Filter timelines to only include those matching allowed types (if any provided)
    const typeFiltered =
      allowedTimelineTypes.size === 0
        ? originalTimelines
        : originalTimelines.filter((tl) => allowedTimelineTypes.has(tl.type));

    // Remove timeline if, after excluding the clip, no clips remain
    const filtered = typeFiltered.filter((tl) => {
      try {
        const clips = getClipsForTimeline(tl.timelineId) || [];
        if (!excludeClipId) return clips.length > 0;
        const remaining = clips.filter((c) => c.clipId !== excludeClipId);
        return remaining.length > 0;
      } catch {
        return true;
      }
    });

    if (filtered.length === 0) return [] as typeof originalTimelines;

    const result: typeof originalTimelines = [];
    for (let i = 0; i < filtered.length; i++) {
      const current = filtered[i];
      const prev = result[i - 1];
      const newHeight = Math.max(
        36,
        Math.round((current.timelineHeight || 0) * SCALE),
      );
      const newY =
        i === 0 ? 0 : (prev?.timelineY || 0) + (prev?.timelineHeight || 0);
      result.push({
        ...current,
        timelineHeight: newHeight,
        timelineY: newY,
      });
    }
    return result;
  }, [originalTimelines, allowedTimelineTypes, excludeClipId]);

  const dimensions = useMemo(
    () => ({
      stageWidth: width,
      stageHeight: height,
    }),
    [width, height],
  );

  const contentHeight = useMemo(() => {
    return timelines.reduce((acc, timeline) => {
      return Math.max(
        acc,
        (timeline.timelineY || 0) + (timeline.timelineHeight || 0),
      );
    }, 0);
  }, [timelines]);

  const hasClips = timelines.length > 0;
  const maxScroll = Math.max(0, contentHeight - dimensions.stageHeight);
  const { totalTimelineFrames, timelineDuration } = useAssetControlsStore();
  const canScrollHorizontal = useMemo(() => {
    const [startFrame, endFrame] = timelineDuration;
    const visible = endFrame - startFrame;
    return visible < totalTimelineFrames;
  }, [timelineDuration, totalTimelineFrames]);

  useEffect(() => {
    setClampedScroll(Math.max(0, Math.min(verticalScroll, maxScroll)));
  }, [verticalScroll, maxScroll]);

  return (
    <>
      <div className="bg-brand rounded-t py-2.5 px-3 flex items-center justify-between gap-x-2 mt-2 border border-x border-b-brand-light/50 border-x-brand-light/10">
        <div className="text-brand-light/70 text-[10px] font-medium">
          Timeline
        </div>
        <TimelineZoom />
      </div>
      <div className="relative">
        <Stage
          width={dimensions.stageWidth}
          height={dimensions.stageHeight}
          className=" z-10 relative border-b border-x border-brand-light/10 bg-brand rounded-b"
        >
          <Layer ref={timelinesLayerRef} visible={hasClips} y={-clampedScroll}>
            {timelines.map((timeline, index) => (
              <Timeline
                assetMode
                isAssetSelected={isAssetSelected}
                excludeClipId={excludeClipId || null}
                cornerRadius={4}
                key={timeline.timelineId}
                scrollY={clampedScroll}
                timelinePadding={timeline.timelinePadding}
                index={index}
                type={timeline.type}
                muted={timeline.muted}
                hidden={timeline.hidden}
                timelineWidth={dimensions.stageWidth}
                timelineY={timeline.timelineY}
                timelineHeight={timeline.timelineHeight}
                timelineId={timeline.timelineId}
              />
            ))}
          </Layer>
          <Layer listening={false} visible={hasClips}>
            {/* Time labels along the top */}
            <TimelineMoments
              stageWidth={dimensions.stageWidth}
              startPadding={24}
              maxScroll={maxScroll}
              thumbY={() => {
                const trackTop = 24;
                const trackBottomPad = 8;
                const trackHeight = Math.max(
                  0,
                  dimensions.stageHeight - trackTop - trackBottomPad,
                );
                const ratio = Math.max(
                  0,
                  Math.min(
                    1,
                    dimensions.stageHeight / Math.max(1, contentHeight),
                  ),
                );
                const minThumb = 24;
                const thumbHeight = Math.max(
                  minThumb,
                  Math.round(trackHeight * ratio),
                );
                const maxThumbY = Math.max(0, trackHeight - thumbHeight);
                const thumbY =
                  trackTop +
                  (maxScroll > 0
                    ? Math.round((clampedScroll / maxScroll) * maxThumbY)
                    : 0);
                return thumbY;
              }}
            />
          </Layer>
          {/* Snap guideline overlay */}
          <Layer listening={false}>
            {typeof snapGuideX === "number" && (
              <Line
                points={[snapGuideX, 0, snapGuideX, dimensions.stageHeight]}
                stroke={"#FFFFFF"}
                strokeWidth={1.5}
                dash={[4, 4]}
                shadowColor={"#FFFFFF"}
                shadowBlur={6}
                shadowOpacity={0.4}
              />
            )}
          </Layer>
          {/* Virtual vertical scrollbar */}
          {hasClips && maxScroll > 0 && (
            <Layer>
              {/* Invisible hover track to detect mouseover across full height on the right margin */}
              <Rect
                x={Math.max(0, dimensions.stageWidth - SCROLLBAR_HW)}
                y={0}
                width={SCROLLBAR_HW}
                height={dimensions.stageHeight}
                fill={"transparent"}
                listening
                onMouseEnter={() => setIsScrollbarHovered(true)}
                onMouseLeave={() => setIsScrollbarHovered(false)}
              />
              {(() => {
                const scrollbarWidth = SCROLLBAR_HW;
                const trackTop = 24;
                const trackBottomPad = 8;
                const trackHeight = Math.max(
                  0,
                  dimensions.stageHeight - trackTop - trackBottomPad,
                );
                const ratio = Math.max(
                  0,
                  Math.min(
                    1,
                    dimensions.stageHeight / Math.max(1, contentHeight),
                  ),
                );
                const minThumb = 24;
                const thumbHeight = Math.max(
                  minThumb,
                  Math.round(trackHeight * ratio),
                );
                const maxThumbY = Math.max(0, trackHeight - thumbHeight);
                const thumbY =
                  trackTop +
                  (maxScroll > 0
                    ? Math.round((clampedScroll / maxScroll) * maxThumbY)
                    : 0);
                const thumbX = Math.max(
                  0,
                  dimensions.stageWidth - scrollbarWidth,
                );
                return (
                  <Rect
                    x={thumbX}
                    y={thumbY}
                    width={scrollbarWidth}
                    height={thumbHeight}
                    cornerRadius={scrollbarWidth}
                    fill={
                      isScrollbarHovered
                        ? "rgba(227,227,227,0.4)"
                        : "rgba(227,227,227,0.1)"
                    }
                    draggable
                    dragBoundFunc={(pos) => {
                      const clampedY = Math.max(
                        trackTop,
                        Math.min(trackTop + maxThumbY, pos.y),
                      );
                      return { x: thumbX, y: clampedY };
                    }}
                    onDragMove={(e) => {
                      const y = e.target.y();
                      const rel =
                        maxThumbY > 0 ? (y - trackTop) / maxThumbY : 0;
                      const next = rel * maxScroll;
                      setVerticalScroll(next);
                    }}
                    onDragEnd={(e) => {
                      const y = e.target.y();
                      const rel =
                        maxThumbY > 0 ? (y - trackTop) / maxThumbY : 0;
                      const next = rel * maxScroll;
                      setVerticalScroll(next);
                    }}
                    onMouseEnter={() => setIsScrollbarHovered(true)}
                    onMouseLeave={() => setIsScrollbarHovered(false)}
                  />
                );
              })()}
            </Layer>
          )}
          {canScrollHorizontal && (
            <Layer>
              <Scrollbar
                stageWidth={dimensions.stageWidth}
                stageHeight={dimensions.stageHeight}
                isScrollbarHovered={isScrollbarHovered}
                setIsScrollbarHovered={setIsScrollbarHovered}
              />
            </Layer>
          )}
        </Stage>
        {hasClips && (
          <AssetScrubControl
            stageHeight={dimensions.stageHeight - 24}
            stageWidth={dimensions.stageWidth}
          />
        )}
      </div>
    </>
  );
};

export default TimelineSearch;
