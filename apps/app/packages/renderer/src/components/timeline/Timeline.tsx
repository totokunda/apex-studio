import React, { useMemo } from "react";
import { Rect, Line, Group } from "react-konva";
import { ImageClipProps, ModelClipProps, TimelineProps, VideoClipProps } from "@/lib/types";
import { useClipStore, getTimelineX } from "@/lib/clip";
import TimelineClip from "./clips/TimelineClip";
import GhostTimeline from "./clips/GhostTimeline";
import { useControlsStore } from "@/lib/control";
import { useContextMenuStore } from "@/lib/context-menu";
import { calculateFrameFromX } from "@/lib/preprocessorHelpers";
import { Rect as KonvaRect } from "konva/lib/shapes/Rect";
import {
  useAssetControlsStore,
  updateAssetZoomLevel,
} from "@/lib/assetControl";

const Timeline: React.FC<
  TimelineProps & {
    index: number;
    scrollY: number;
    cornerRadius?: number;
    assetMode?: boolean;
    excludeClipId?: string | null;
    isAssetSelected?: (clipId: string) => boolean;
  }
> = ({
  timelineWidth,
  timelineY,
  timelineHeight = 54,
  timelinePadding = 24,
  timelineId,
  index,
  scrollY,
  type,
  muted,
  hidden,
  cornerRadius = 1,
  assetMode = false,
  excludeClipId = null,
  isAssetSelected = () => false,
}) => {


  const { hoveredTimelineId, getClipsForTimeline } = useClipStore();
  const hiddenRectRef = React.useRef<KonvaRect>(null);

  const clipsAll = getClipsForTimeline(timelineId);
  const clips = useMemo(() => {
    if (!excludeClipId) return clipsAll;
    return clipsAll
      .filter((c) => c.clipId !== excludeClipId)
      .filter((c) =>
        assetMode ? (c.type === "model" && !(c as ModelClipProps).assetId ? false : true) : true,
      );
  }, [clipsAll, excludeClipId]);
  // Stable signature to detect only meaningful clip-set changes (global across all timelines)
  const allClips = useClipStore((s) => s.clips);
  const allVisibleClips = useMemo(
    () => allClips.filter((c) => !c.hidden),
    [allClips],
  );

  const allClipsSignature = useMemo(() => {
    if (!allVisibleClips || allVisibleClips.length === 0) return "0:0";
    let longest = 0;
    for (let i = 0; i < allVisibleClips.length; i++) {
      const end = Math.max(0, allVisibleClips[i].endFrame || 0);
      if (end > longest) longest = end;
    }
    return `${allVisibleClips.length}:${longest}`;
  }, [allVisibleClips]);
  const clipDuration = useClipStore((s) => s.clipDuration);

  const ctrl = useControlsStore();
  const asset = useAssetControlsStore();
  const timelineDuration = assetMode
    ? asset.timelineDuration
    : ctrl.timelineDuration;

  const timelineHasPreprocessors = useMemo(() => {
    if (type !== "media") return false;
    return clips.some(
      (clip) =>
        (clip as VideoClipProps | ImageClipProps).preprocessors?.length > 0,
    );
  }, [clips, type]);

  const renderedTimelineHeight = useMemo(() => {
    return timelineHeight - 8;
  }, [timelineHasPreprocessors, timelineHeight]);

  // Compute exact edges of the visible timeline rect
  const timelineTopY = (timelineY ?? 0) + 32;
  const timelineBottomY = timelineTopY + renderedTimelineHeight;

  // Timelines are spaced by their full height but rects are 8px shorter, creating an 8px gap
  const gapBetweenTimelines = 8;
  // Hover target spans the full gap for easy interaction
  const underGroupHeight = gapBetweenTimelines;

  // Position groups so the line (drawn at underGroupHeight/2) sits exactly in the middle of the gap
  const topDashGroupY =
    timelineTopY - gapBetweenTimelines / 2 - underGroupHeight / 2;
  const bottomDashGroupY =
    timelineBottomY + gapBetweenTimelines / 2 - underGroupHeight / 2;

  const timelineX = useMemo(
    () => getTimelineX(timelineWidth!, timelinePadding, timelineDuration),
    [timelineWidth, timelinePadding, timelineDuration],
  );

  // When viewing in assetMode, calibrate baseline/zoom from ALL visible clips once (top row only)
  React.useEffect(() => {
    if (!assetMode || index !== 0) return;
    updateAssetZoomLevel(allVisibleClips, clipDuration);
  }, [assetMode, index, allClipsSignature]);

  // Ensure the hidden/disabled overlay rect always stays on top visually
  React.useEffect(() => {
    if (hiddenRectRef.current) {
      setTimeout(() => {
        hiddenRectRef.current?.moveToTop();
        hiddenRectRef.current?.getLayer()?.batchDraw();
      }, 100);
    }
  }, [hidden, muted, type]);



  
  return (
    <>
      {index === 0 && (
        <Group
          id={`dashed-${timelineId}-top`}
          name={"timeline-dashed"}
          height={underGroupHeight}
          x={timelinePadding}
          y={topDashGroupY}
        >
          <Line
            points={[
              0,
              underGroupHeight / 2,
              timelineWidth!,
              underGroupHeight / 2,
            ]}
            stroke={"rgba(255, 255, 255, 0.75)"}
            strokeWidth={
              hoveredTimelineId === `dashed-${timelineId}-top` ? 1.2 : 0
            }
          />
        </Group>
      )}
      <Rect
        id={timelineId}
        x={timelineX}
        y={timelineY! + 32}
        cornerRadius={4}
        width={timelineWidth! - timelineX + 8}
        height={renderedTimelineHeight}
        fill={"rgba(11, 11, 13, 0.25)"}
        onContextMenu={(e) => {
          e.evt.preventDefault();
          const stage = e.target.getStage();
          const pos = stage?.getPointerPosition();
          if (!pos) return;
          const [startFrame, endFrame] = timelineDuration;
          const innerWidth = timelineWidth!; // stage width used in calculateFrameFromX expects total stage width
          const frame = calculateFrameFromX(
            pos.x,
            timelinePadding,
            innerWidth,
            [startFrame, endFrame],
          );
          const progress = Math.max(
            0,
            Math.min(1, (pos.x - timelinePadding) / Math.max(1, innerWidth)),
          );
          if (assetMode) {
            useAssetControlsStore.getState().setFocusAnchorRatio(progress);
            useAssetControlsStore.getState().setFocusFrame(frame);
          } else {
            useControlsStore.getState().setFocusAnchorRatio(progress);
            useControlsStore.getState().setFocusFrame(frame, false);
          }
          useContextMenuStore.getState().openMenu({
            position: { x: e.evt.clientX, y: e.evt.clientY },
            target: { type: "timeline", timelineId },
            groups: [
              {
                id: "edit",
                items: [
                  {
                    id: "paste",
                    label: "Paste at Playhead",
                    action: "paste",
                    shortcut: "⌘V",
                  },
                ],
              },
            ],
          });
        }}
      />
      {clips.map((clip) => (
        <TimelineClip
          key={clip.clipId}
          muted={muted}
          hidden={assetMode ? false : hidden}
          cornerRadius={cornerRadius}
          timelineId={timelineId}
          clipId={clip.clipId}
          timelinePadding={timelinePadding}
          timelineWidth={timelineWidth}
          timelineY={timelineBottomY}
          timelineHeight={renderedTimelineHeight}
          clipType={clip.type}
          type={type}
          scrollY={scrollY}
          isAssetSelected={isAssetSelected}
          assetMode={assetMode}
        />
      ))}
      <GhostTimeline
        timelineId={timelineId}
        timelineY={timelineBottomY}
        timelineHeight={renderedTimelineHeight}
        timelinePadding={timelinePadding}
        timelineWidth={timelineWidth}
        type={type}
        muted={muted}
        hidden={assetMode ? false : hidden}
      />
      <Group
        id={`dashed-${timelineId}`}
        name={"timeline-dashed"}
        height={underGroupHeight}
        x={timelinePadding}
        y={bottomDashGroupY}
      >
        <Line
          points={[
            0,
            underGroupHeight / 2,
            timelineWidth!,
            underGroupHeight / 2,
          ]}
          stroke={"rgba(255, 255, 255, 0.75)"}
          strokeWidth={hoveredTimelineId === `dashed-${timelineId}` ? 1.2 : 0}
        />
      </Group>
      {(!assetMode && (hidden || (muted && type === "audio"))) && (
        <Rect
          ref={hiddenRectRef}
          id={`hidden-${timelineId}`}
          x={timelineX}
          y={timelineY! + 32}
          cornerRadius={0}
          width={timelineWidth! - timelineX + 8}
          height={timelineHeight - 8}
          fill={"rgba(11, 11, 13, 0.60)"}
        />
      )}
    </>
  );
};

export default Timeline;
