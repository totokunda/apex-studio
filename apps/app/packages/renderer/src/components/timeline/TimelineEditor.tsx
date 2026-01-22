import { getZoomLevelConfig } from "@/lib/zoom";
import { useControlsStore } from "@/lib/control";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Stage, Layer, Text, Line, Rect } from "react-konva";
import Konva from "konva";
import { ScrubControl } from "./Scrubber";
import Timeline from "./Timeline";
import {
  useClipStore,
  getClipWidth,
  getClipX,
  isValidTimelineForClip,
  getTimelineTypeForClip,
  getTimelineHeightForClip,
} from "@/lib/clip";
import { GoFileMedia } from "react-icons/go";
import Droppable from "../dnd/Droppable";
import { DragEndEvent, useDndMonitor } from "@dnd-kit/core";
import { MediaItem } from "../media/Item";
import { v4 as uuidv4 } from "uuid";
import {
  AnyClipProps,
  Asset,
  Filter,
  FilterClipProps,
  ImageClipProps,
  ModelClipProps,
  PreprocessorClipProps,
  TimelineProps,
  VideoClipProps,
} from "@/lib/types";
import TimelineSidebar from "./TimelineSidebar";
import Scrollbar from "./Scrollbar";
import { useWebGLHaldClut } from "../preview/webgl-filters";
import { ensureFullMediaStats } from "@/lib/media/utils";
import { Preprocessor } from "@/lib/preprocessor";
import {
  calculateFrameFromX,
  getOtherPreprocessors,
} from "@/lib/preprocessorHelpers";
import { convertFrameRange } from "@/lib/media/fps";
import { getOffloadDefaultsForManifest } from "@app/preload";

import { ManifestWithType } from "@/lib/manifest/api";
import { useViewportStore } from "@/lib/viewport";

interface TimelineEditorProps {}

interface TimelineMomentsProps {
  stageWidth: number;
  startPadding: number;
  maxScroll: number;
  thumbY: () => number;
}

interface TickMark {
  x: number;
  text: string;
  type: "major" | "minor";
  format: "frame" | "second";
}

const SCROLLBAR_HW = 8;
const SCROLL_BOTTOM_PADDING = 48; // extra space at bottom for easier drag-in

const getMajorZoomConfigFormat = (
  zoomConfig: {
    majorTickFormat: "frame" | "second";
    minorTickFormat: "frame" | "second";
  },
  fps: number,
  frameInterval: number,
) => {
  if (zoomConfig.majorTickFormat == "second") {
    return "second";
  } else if (
    zoomConfig.majorTickFormat == "frame" &&
    frameInterval % fps == 0
  ) {
    return "second";
  } else {
    return "frame";
  }
};

const TimelineMoments: React.FC<TimelineMomentsProps> = React.memo(
  ({ stageWidth, startPadding, maxScroll, thumbY }) => {
    const {
      timelineDuration,
      fps,
      zoomLevel,
      focusFrame,
      shiftTimelineDuration,
      maxZoomLevel,
      minZoomLevel,
    } = useControlsStore();
    const [startFrame, endFrame] = timelineDuration;

    // We will basically render from startDuration to startDuration + duration.
    useEffect(() => {
      // ensure focusframe is always within the timeline duration
      if (focusFrame < startFrame) {
        shiftTimelineDuration(focusFrame - startFrame);
      }
      if (focusFrame > endFrame) {
        let duration = endFrame - startFrame;
        const additionalDuration =
          duration + endFrame > focusFrame ? 0 : focusFrame - endFrame;
        shiftTimelineDuration(duration + additionalDuration);
      }
    }, [startFrame, endFrame, focusFrame]);
    // Convert duration to milliseconds if needed for consistent calculations
    const tickMark: TickMark[] = useMemo(() => {
      let ticks: TickMark[] = [];
      const zoomConfig = getZoomLevelConfig(
        zoomLevel,
        timelineDuration,
        fps,
        maxZoomLevel,
        minZoomLevel,
      );
      const majorTickInterval =
        zoomConfig.majorTickInterval *
        (zoomConfig.majorTickFormat === "second" ? fps : 1);
      const minorTickInterval =
        zoomConfig.minorTickInterval *
        (zoomConfig.minorTickFormat === "second" ? fps : 1);
      const [startFrame, endFrame] = timelineDuration;

      for (let i = startFrame; i <= endFrame; i += majorTickInterval) {
        ticks.push({
          x: Math.round(i),
          text: Math.round(i).toString(),
          type: "major",
          format: getMajorZoomConfigFormat(zoomConfig, fps, majorTickInterval),
        });
      }

      for (let i = startFrame; i <= endFrame; i += minorTickInterval) {
        ticks.push({
          x: Math.round(i),
          text: Math.round(i).toString(),
          type: "minor",
          format: zoomConfig.minorTickFormat,
        });
      }
      ticks.sort((a, b) => a.x - b.x);
      // remove duplicates where x is the same and minor is true
      ticks = ticks.filter(
        (tick, index, self) =>
          index === 0 || tick.x !== self[index - 1].x || tick.type !== "minor",
      );

      return ticks;
    }, [timelineDuration, zoomLevel, fps]);

    // Helper function to format tick labels
    const formatTickLabel = (
      value: number,
      format: "frame" | "second",
    ): string => {
      if (format === "frame") {
        return `${value}f`;
      } else {
        // Convert frames to seconds
        const seconds = value / fps;
        const totalSeconds = Math.floor(seconds);
        const minutes = Math.floor(totalSeconds / 60);
        const remainingSeconds = totalSeconds % 60;
        return `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
      }
    };

    return (
      <>
        <Rect
          x={0}
          y={0}
          width={stageWidth}
          height={28}
          fill={maxScroll > 0 && thumbY() > 24 ? "#222124" : undefined}
          listening={false}
        />
        {tickMark.map((tick, index) => {
          // Calculate x position based on timeline progress
          const progress = (tick.x - startFrame) / (endFrame - startFrame);
          const xPosition = progress * stageWidth;

          // Only render ticks that are visible on screen
          if (xPosition < -50 || xPosition > stageWidth + 50) {
            return null;
          }

          const isMajor = tick.type === "major";
          const tickHeight = isMajor ? 21 : 7;
          const tickY = 0;

          return (
            <React.Fragment key={`${tick.x}-${tick.type}-${index}`}>
              {/* Tick line */}
              <Line
                points={[
                  xPosition + startPadding,
                  tickY,
                  xPosition + startPadding,
                  tickY + tickHeight,
                ]}
                stroke={
                  isMajor
                    ? "rgba(255, 255, 255, 0.3)"
                    : "rgba(255, 255, 255, 0.1)"
                }
                strokeWidth={1}
                listening={false}
              />

              {/* Label for major ticks only */}
              {isMajor && (
                <Text
                  x={xPosition + 4 + startPadding}
                  y={tickY + tickHeight - 8}
                  text={formatTickLabel(tick.x, tick.format)}
                  fontSize={8.5}
                  fill="rgba(255, 255, 255, 0.4)"
                  fontFamily="Poppins, system-ui, sans-serif"
                  listening={false}
                />
              )}
            </React.Fragment>
          );
        })}
      </>
    );
  },
);

const TimelineEditor: React.FC<TimelineEditorProps> = React.memo(() => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  // Dev-only: if timeline HMR updates "stick" visually (react-konva/Electron), we remount the Stage.
  const [hmrStageKey, setHmrStageKey] = useState(0);
  // Always mount timeline elements; avoid conditional unmounts that can
  // cause rendering glitches in Tauri's WebView. Toggle visibility via data.

  const controlStore = useControlsStore();
  const {
    clips,
    timelines,
    addPreprocessorToClip,
    removePreprocessorFromClip,
    updatePreprocessor,
    getClipsForTimeline,
    getTimelinePosition,
    getClipPosition,
    addTimeline,
    addClip,
    setGhostStartEndFrame,
    setGhostX,
    setGhostGuideLines,
    setGhostTimelineId,
    setActiveMediaItem,
    removeTimeline,
    setGhostInStage,
    setHoveredTimelineId,
    hoveredTimelineId,
    snapGuideX,
    setSelectedPreprocessorId,
    setIsDragging,
    addAsset
  } = useClipStore();
  // const scrollBarRef = useRef<any>(null);
  // const isSyncingScrollRef = useRef(false);
  const [isRulerDragging, setIsRulerDragging] = useState(false);
  const panStateRef = useRef({
    startX: 0,
    lastX: 0,
    startFrame: 0,
    fractionalFrames: 0,
  });
  const wheelRemainderRef = useRef(0);
  const timelinesLayerRef = useRef<Konva.Layer | null>(null);
  const haldClutRef = useWebGLHaldClut();
  const [verticalScroll, setVerticalScroll] = useState(0);
  const verticalScrollRef = useRef(0);
  const [isScrollbarHovered, setIsScrollbarHovered] = useState(false);
  const [canScrollHorizontal, setCanScrollHorizontal] = useState(false);
  const { totalTimelineFrames, timelineDuration } = useControlsStore();
  const preprocessorDragClipRef = useRef<PreprocessorClipProps | null>(null);
  const preprocessorDragKonvaNodeRef = useRef<Konva.Node | null>(null);

  useEffect(() => {
    if (!import.meta.hot) return;
    const onTimelineHmr = () => setHmrStageKey((k) => k + 1);
    import.meta.hot.on("apex:timeline-hmr", onTimelineHmr);
    return () => {
      import.meta.hot?.off("apex:timeline-hmr", onTimelineHmr);
    };
  }, []);

  useEffect(() => {
    const [startFrame, endFrame] = timelineDuration;
    const totalDuration = endFrame - startFrame;
    if (totalDuration < totalTimelineFrames) {
      setCanScrollHorizontal(true);
    } else {
      setCanScrollHorizontal(false);
    }
  }, [totalTimelineFrames, timelineDuration]);

  const getBoundsForTimeline = (timelineId: string) => {
    const timeline = timelines.find((t) => t.timelineId === timelineId);
    if (!timeline) return 8;
    if (timeline.type === "media") return 16;
    return 8;
  };

  useDndMonitor({
    onDragStart: (event) => {
      const data = event.active?.data?.current as unknown as
        | MediaItem
        | Preprocessor
        | ManifestWithType
        | undefined;
      if (!data) return;

      if (data.type === "preprocessor") {
        const clipFrames = controlStore.fps * controlStore.defaultClipLength;
        setGhostStartEndFrame(0, clipFrames);
        return;
      }

      setIsDragging(true);

      if (data.type === "model") {
        const clipFrames =
          ((data as ManifestWithType).spec?.default_duration_secs ??
          controlStore.defaultClipLength) * controlStore.fps;
        setGhostStartEndFrame(0, clipFrames);
        return;
      }

      const mediaInfo = (data as MediaItem).mediaInfo;
      setActiveMediaItem(data as MediaItem);
      const clipFrames = (() => {
        const fps = controlStore.fps;
        if (data.type === "image") return fps * controlStore.defaultClipLength;
        if (data.type === "video") {
          const realFps = mediaInfo?.stats.video?.averagePacketRate ?? fps;
          const realEnd = (mediaInfo?.duration ?? 0) * realFps;
          const { end: endFrameReal } = convertFrameRange(
            0,
            realEnd,
            realFps,
            fps,
          );
          return endFrameReal;
        }
        if (data.type === "audio") {
          return Math.round((mediaInfo?.duration ?? 0) * fps);
        }
        if (data.type === "filter") {
          return fps * controlStore.defaultClipLength;
        }
        return 0;
      })();
      setGhostStartEndFrame(0, clipFrames);
    },
    onDragMove: (event) => {
      const container = containerRef.current;
      let pointerX: number | null = null;
      let pointerY: number | null = null;
      const data = event.active?.data?.current as unknown as
        | MediaItem
        | Preprocessor
        | ManifestWithType
        | undefined;

      if (clips.length === 0) return;

      if (container) {
        const rect = container.getBoundingClientRect();
        const activeRect =
          (event as any)?.active?.rect?.current?.translated ||
          (event as any)?.active?.rect?.current;
        if (activeRect) {
          const centerX = (activeRect.left ?? 0) + (activeRect.width ?? 0) / 2;
          const centerY = (activeRect.top ?? 0) + (activeRect.height ?? 0) / 2;
          pointerX = centerX - rect.left;
          pointerY = centerY - rect.top;
        }
      }

      if (pointerX == null || pointerY == null) {
        return;
      }

      // Edge auto-scroll for external DnD moves
      edgeAutoScroll(pointerY);

      if (data?.type === "preprocessor") {
        // get pointer position
        handlePreprocessorDragMove(data as Preprocessor, pointerY, pointerX);
      } else if (
        data?.type === "image" ||
        data?.type === "video" ||
        data?.type === "audio" ||
        data?.type === "filter" ||
        data?.type === "model"
      ) {
        handleDragMove(data as MediaItem, pointerY, pointerX);
      }
    },
    onDragEnd: (event) => {
      const data = event.active?.data?.current as unknown as
        | MediaItem
        | Preprocessor
        | undefined;

      if (!data) {
        setIsDragging(false);
        return;
      }

      // Route to appropriate handler based on item type
      if (data.type === "preprocessor") {
        handlePreprocessorDrop(event, data as Preprocessor);
      } else if (
        data?.type === "image" ||
        data?.type === "video" ||
        data?.type === "audio" ||
        data?.type === "filter" ||
        data?.type === "model"
      ) {
        handleDrop(event, data);
      }

      setIsDragging(false);
    },
  });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Throttle resize updates to improve performance
    let timeoutId: NodeJS.Timeout;
    const observer = new ResizeObserver((entries) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        const entry = entries[0];
        if (!entry) return;
        const { width, height } = entry.contentRect;
        setSize({ width, height });
      }, 16); // ~60fps throttle
    });

    observer.observe(el);
    return () => {
      clearTimeout(timeoutId);
      observer.disconnect();
    };
  }, []);

  // monitor all timelines and if any are empty, remove them
  useEffect(() => {
    const emptyTimelines = timelines.filter(
      (t) => getClipsForTimeline(t.timelineId).length === 0,
    );
    emptyTimelines.forEach((t) => {
      removeTimeline(t.timelineId);
    });
  }, [clips]);

  // Memoize dimensions to prevent unnecessary recalculations
  const dimensions = useMemo(() => {
    const stageWidth = Math.max(1, size.width);
    const stageHeight = Math.max(1, size.height);
    const dropZonePadding = 24;
    const dropZoneHeight = 80;
    const dropZoneX = dropZonePadding;
    const dropZoneY = Math.max(0, stageHeight / 2 - dropZoneHeight / 2);
    const dropZoneWidth = Math.max(0, stageWidth - dropZonePadding * 2);

    return {
      stageWidth,
      stageHeight,
      dropZoneX,
      dropZoneY,
      dropZoneWidth,
      dropZoneHeight,
    };
  }, [size.width, size.height]);

  const handleDragMove = useCallback(
    (data: MediaItem | undefined, pointerY: number, pointerX: number) => {
      if (!data) return;
      // Record last pointer X for later use on drop when creating a new timeline via dashed hover
      try {
        (window as any).__apex_lastPointerX = pointerX;
      } catch {}
      // check if the pointer is over a dashed line
      const stage = timelinesLayerRef.current;
      const children = stage?.children || [];
      let hoveredTimelineIdCurrent: string | null = null;

      for (const child of children) {
        const id = child.id();
        if (id.startsWith("dashed-")) {
          // get the rect of the child
          const rect = child.getClientRect();

          const rectY = rect.y;
          const rectX = rect.x;
          const rectWidth = rect.width;
          const boundNumber = getBoundsForTimeline(id);
          const boundNumberTop = id?.endsWith("-top") ? 36 : boundNumber;
          const boundNumberBottom = boundNumber;
          const boundsY = [rectY - boundNumberTop, rectY + boundNumberBottom];

          const boundsX = [rectX, rectX + rectWidth];
          if (
            pointerY >= boundsY[0] &&
            pointerY <= boundsY[1] &&
            pointerX >= boundsX[0] &&
            pointerX <= boundsX[1]
          ) {
            hoveredTimelineIdCurrent = id;
            break;
          }
        }
      }

      setHoveredTimelineId(hoveredTimelineIdCurrent);

      if (hoveredTimelineIdCurrent) {
        // make everything else null
        setGhostTimelineId(null);
        setGhostInStage(false);
        return;
      }

      const timelinePadding = 24;
      const stageWidth = dimensions.stageWidth;
      const timelines = useClipStore.getState().timelines;

      let activeTimelineId: string | null = null;
      for (let i = 0; i < timelines.length; i++) {
        const t = timelines[i];
        const top = (t.timelineY ?? 0) + 8 - (verticalScrollRef.current || 0);
        const height = t.timelineHeight ?? 54;
        const left = timelinePadding;
        const right = left + stageWidth;
        const bottom = top + height;

        const isInside =
          pointerY >= top &&
          pointerY <= bottom &&
          pointerX >= left &&
          pointerX <= right;
        if (isInside) {
          // check for same type
          if (isValidTimelineForClip(t, data!)) {
            activeTimelineId = t.timelineId!;
            break;
          }
        }
      }

      if (!activeTimelineId) {
        setGhostTimelineId(null);
        setGhostInStage(false);
        return;
      }

      // Center ghost under pointer and validate against bounds/overlaps
      const pointerLocalX = pointerX - timelinePadding; // pointer relative to inner timeline
      let [visibleStartFrame, visibleEndFrame] =
        useControlsStore.getState().timelineDuration;
      const ghostFrames = useClipStore.getState().ghostStartEndFrame;
      const ghostFramesLen = Math.max(
        1,
        (ghostFrames[1] ?? 0) - (ghostFrames[0] ?? 0),
      );

      // Map pointer directly to ghost's left edge in PX to keep exact alignment
      const desiredLeftPx = pointerLocalX;
      const ghostWidthPx = getClipWidth(0, ghostFramesLen, stageWidth, [
        visibleStartFrame,
        visibleEndFrame,
      ]);
      let desiredLeft = desiredLeftPx;

      // Build occupied intervals (in px, inner coordinates)
      const getClipsForTimeline = useClipStore.getState().getClipsForTimeline;
      const existingClips = getClipsForTimeline(activeTimelineId);
      let maxRight = 0;
      const occupied = existingClips
        .map((c) => {
          const sx = getClipX(c.startFrame || 0, c.endFrame || 0, stageWidth, [
            visibleStartFrame,
            visibleEndFrame,
          ]);
          const sw = getClipWidth(
            c.startFrame || 0,
            c.endFrame || 0,
            stageWidth,
            [visibleStartFrame, visibleEndFrame],
          );
          const lo = Math.max(0, sx);
          const hi = Math.max(0, sx + sw);
          maxRight = Math.max(maxRight, hi);
          return hi > lo ? ([lo, hi] as [number, number]) : null;
        })
        .filter(Boolean) as [number, number][];

      occupied.sort((a, b) => a[0] - b[0]);
      // Merge overlapping/touching intervals
      const merged: [number, number][] = [];
      for (const [lo, hi] of occupied) {
        if (merged.length === 0) {
          merged.push([lo, hi]);
        } else {
          const last = merged[merged.length - 1];
          if (lo <= last[1]) {
            last[1] = Math.max(last[1], hi);
          } else {
            merged.push([lo, hi]);
          }
        }
      }

      // Compute gaps
      const gaps: [number, number][] = [];
      let prev = 0;
      for (const [lo, hi] of merged) {
        if (lo > prev) gaps.push([prev, lo]);
        prev = Math.max(prev, hi);
      }
      // add the last gap
      if (prev < stageWidth) gaps.push([prev, Infinity]);

      const validGaps = gaps.filter(([lo, hi]) => hi - lo >= ghostWidthPx);
      const pointerCenter = desiredLeft;
      let chosenGap: [number, number] | null = null;

      for (const gap of validGaps) {
        if (pointerCenter >= gap[0] && pointerCenter <= gap[1]) {
          chosenGap = gap;
          break;
        }
      }
      if (!chosenGap && validGaps.length > 0) {
        chosenGap = validGaps.reduce((best, gap) => {
          const gc = (gap[0] + gap[1]) / 2;
          const bc = (best[0] + best[1]) / 2;
          return Math.abs(pointerCenter - gc) < Math.abs(pointerCenter - bc)
            ? gap
            : best;
        });
      }

      let validatedLeft = desiredLeft;

      if (chosenGap) {
        const [gLo, gHi] = chosenGap;
        validatedLeft = Math.min(
          Math.max(desiredLeft, gLo),
          gHi - ghostWidthPx,
        );
      } else {
        // Ensure validated Left
        validatedLeft = Math.max(validatedLeft, maxRight);
      }

      validatedLeft = Math.max(0, validatedLeft);

      setGhostTimelineId(activeTimelineId);
      setGhostInStage(true);
      setGhostX(validatedLeft);
    },
    [
      dimensions.stageWidth,
      dimensions.stageHeight,
      verticalScrollRef.current,
      timelinesLayerRef.current,
      hoveredTimelineId,
      setGhostTimelineId,
      setGhostInStage,
      setGhostX,
      setGhostStartEndFrame,
      setGhostInStage,
      setGhostX,
    ],
  );

  const handleDrop = useCallback(
    (_event: any, data: MediaItem | ManifestWithType) => {
      const timelines = useClipStore.getState().timelines;
      let timelineId: string | undefined = undefined;
      const timelineHeight = getTimelineHeightForClip(data);

      if (timelines.length === 0) {
        if (!_event.over) {
          return;
        }
        timelineId = uuidv4();
        const newTimeline = {
          type: getTimelineTypeForClip(data),
          timelineId,
          timelinePadding: 24,
          timelineWidth: size.width,
          timelineY: 0,
          timelineHeight: timelineHeight,
        };
        addTimeline(newTimeline);
      } else if (hoveredTimelineId) {
        // we add a new timeline at the hovered timeline id
        const timelines = useClipStore.getState().timelines;
        const hoveredTimelineIdx = timelines.findIndex(
          (t) => t.timelineId === hoveredTimelineId.replace("dashed-", ""),
        );
        const hoveredTimeline =
          hoveredTimelineIdx !== -1 ? timelines[hoveredTimelineIdx] : null;
        timelineId = uuidv4();
        const newTimeline = {
          type: getTimelineTypeForClip(data),
          timelineId,
          timelinePadding: 24,
          timelineWidth: size.width,
          timelineY: (hoveredTimeline?.timelineY ?? 0) + timelineHeight,
          timelineHeight: timelineHeight,
        };
        addTimeline(newTimeline, hoveredTimelineIdx);
      }

      const mediaInfo = (data as MediaItem)?.mediaInfo;

      if (!mediaInfo && data.type !== "filter" && data.type !== "model") {
        setActiveMediaItem(null);
        setGhostTimelineId(null);
        setGhostStartEndFrame(0, 0);
        setGhostX(0);
        return;
      }

      let numFrames: number = 0;
      let trimEnd = 0;
      let trimStart = 0;
      let height: number | undefined = undefined;
      let width: number | undefined = undefined;

      if (data.type === "video") {
        const duration = mediaInfo?.duration ?? 0;
        const fps = controlStore.fps; // Use project fps for clip duration
        height = mediaInfo?.video?.displayHeight;
        width = mediaInfo?.video?.displayWidth;
        numFrames = Math.round(duration * fps);
      } else if (data.type === "audio") {
        const duration = mediaInfo?.duration ?? 0;
        const fps = controlStore.fps;
        numFrames = Math.round(duration * fps);
      } else if (data.type === "image") {
        numFrames = controlStore.fps * controlStore.defaultClipLength;
        trimEnd = -Infinity;
        trimStart = Infinity;
        height = mediaInfo?.image?.height;
        width = mediaInfo?.image?.width;
      } else if (data.type === "filter") {
        numFrames = controlStore.fps * controlStore.defaultClipLength;
        trimEnd = -Infinity;
        trimStart = Infinity;
        // Start with smallPath
        void haldClutRef?.preloadClut((data as unknown as Filter).smallPath);
      } else if (data.type === "model") {
        numFrames =
          ((data as ManifestWithType).spec?.default_duration_secs ??
          controlStore.defaultClipLength)  * controlStore.fps;
        trimEnd = -Infinity;
        trimStart = Infinity;
      }

      // Use validated ghost position to compute frames
      const state = useClipStore.getState();
      const ghostTimelineId = state.ghostTimelineId;
      // Use ghostX when we have a ghost target; if we are creating a new timeline via dashed hover,
      // compute the X from the last pointer position captured in onDragMove (center of draggable)
      let ghostX = state.ghostTimelineId ? state.ghostX : 0;

      // If we're creating a new timeline (hovered dashed), align start using current mouse position
      if (
        hoveredTimelineId &&
        typeof (window as any).__apex_lastPointerX === "number"
      ) {
        const pointerX = Number((window as any).__apex_lastPointerX);
        const innerX = Math.max(
          0,
          Math.min(dimensions.stageWidth, pointerX - 24),
        );
        ghostX = Math.round(innerX);
      }

      const dropTimelineId = hoveredTimelineId
        ? timelineId
        : ghostTimelineId || timelineId;
      setHoveredTimelineId(null);

      if (!dropTimelineId) {
        setActiveMediaItem(null);
        setGhostTimelineId(null);
        setGhostStartEndFrame(0, 0);
        setGhostX(0);
        return;
      }

      // If this is the first media (image/video) clip being added, set the preview aspect ratio
      try {
        const existingClips = useClipStore.getState().clips;
        const hasMediaAlready = existingClips.some(
          (c) => c.type === "video" || c.type === "image" || (c.type === "model" && c.assetId),
        );
        const isMediaIncoming = data.type === "video" || data.type === "image";
        if (!hasMediaAlready && isMediaIncoming) {
          const w = Number(width);
          const h = Number(height);
          if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
            const gcd = (a: number, b: number): number =>
              b === 0 ? a : gcd(b, a % b);
            const g = gcd(Math.round(w), Math.round(h)) || 1;
            const id = `${Math.round(w / g)}:${Math.round(h / g)}`;
            useViewportStore
              .getState()
              .setAspectRatio({
                width: Math.round(w),
                height: Math.round(h),
                id,
              });
          }
        }
      } catch {}

      let [tStart, tEnd] = controlStore.timelineDuration;
      const stageWidth = dimensions.stageWidth;
      const visibleDuration = tEnd - tStart;
      const clipLen = Math.max(1, numFrames);
      // Map ghostX (left edge) to frame in visible window

      let startFrame = Math.round(
        tStart +
          (Math.max(0, Math.min(stageWidth, ghostX)) / stageWidth) *
            visibleDuration,
      );
      let endFrame = startFrame + clipLen;

      // Validate against overlaps on the target timeline across ALL frames
      // and choose a feasible placement. If no gap at desired position, append at end.
      const getClipsForTimeline = state.getClipsForTimeline;
      const existingClips = getClipsForTimeline(dropTimelineId)
        .map((c) => ({ lo: c.startFrame || 0, hi: c.endFrame || 0 }))
        .filter((iv) => iv.hi > iv.lo)
        .sort((a, b) => a.lo - b.lo);

      // Merge intervals globally
      const merged: { lo: number; hi: number }[] = [];
      for (const iv of existingClips) {
        if (merged.length === 0) merged.push({ ...iv });
        else {
          const last = merged[merged.length - 1];
          if (iv.lo <= last.hi) last.hi = Math.max(last.hi, iv.hi);
          else merged.push({ ...iv });
        }
      }

      // Try to place at or after requested startFrame without overlap
      let placementStart = Math.max(0, startFrame);
      if (merged.length === 0) {
        // No existing clips; keep startFrame as selected (0 if first clip logic below)
      } else {
        // Advance placementStart past any overlapping intervals
        for (const iv of merged) {
          if (placementStart + clipLen <= iv.lo) {
            // Fits before this interval
            break;
          }
          if (placementStart < iv.hi) {
            // Overlaps; move right after this interval and continue
            placementStart = iv.hi;
          }
        }
      }
      startFrame =
        existingClips.length === 0
          ? existingClips.length === 0
            ? startFrame
            : placementStart
          : placementStart;
      endFrame = startFrame + clipLen;


      let asset: Asset | null = null

      if (data.type === 'audio' || data.type === 'video' || data.type === 'image') {
        asset = addAsset({ path: (data as MediaItem)?.assetUrl });
      }

      // @ts-ignore
      const newClip: AnyClipProps = {
        timelineId: dropTimelineId,
        clipId: uuidv4(),
        startFrame: existingClips.length === 0 ? startFrame : startFrame,
        endFrame,
        // @ts-ignore
        type: data.type,
        trimEnd: trimEnd,
        trimStart: trimStart,
        speed: 1.0,
      };

      if (asset) {
        (newClip as VideoClipProps | ImageClipProps).assetId = asset.id;
      }

      // Ensure intrinsic media dimensions are captured for image/video clips
      if (data.type === "image" || data.type === "video") {
        const mw = Number(width);
        const mh = Number(height);
        if (Number.isFinite(mw) && Number.isFinite(mh) && mw > 0 && mh > 0) {
          (newClip as VideoClipProps | ImageClipProps).mediaWidth = mw;
          (newClip as VideoClipProps | ImageClipProps).mediaHeight = mh;
          (newClip as VideoClipProps | ImageClipProps).mediaAspectRatio =
            mw / mh;
        }
      }

      if (data.type === "filter") {
        (newClip as FilterClipProps).name = (data as unknown as Filter).name;
        (newClip as FilterClipProps).smallPath = (
          data as unknown as Filter
        ).smallPath;
        (newClip as FilterClipProps).fullPath = (
          data as unknown as Filter
        ).fullPath;
        (newClip as FilterClipProps).category = (
          data as unknown as Filter
        ).category;
        (newClip as FilterClipProps).examplePath = (
          data as unknown as Filter
        ).examplePath;
        (newClip as FilterClipProps).exampleAssetUrl = (
          data as unknown as Filter
        ).exampleAssetUrl;
      }

      if (data.type === "image" || data.type === "video") {
        (newClip as VideoClipProps | ImageClipProps).preprocessors = [];
        (newClip as VideoClipProps | ImageClipProps).masks = [];
      }

      if (data.type === "model") {
        // We need to get the manifest document from the data
        const manifest = data as ManifestWithType;
        const cleanup = () => {
          setActiveMediaItem(null);
          setGhostTimelineId(null);
          setGhostStartEndFrame(0, 0);
          setGhostX(0);
        };

        (newClip as ModelClipProps).manifest = manifest;
        addClip(newClip as AnyClipProps);

        // Best-effort: hydrate global offload defaults for this manifest id.
        // Do this *after* addClip so we can safely update the stored clip without
        // needing to block the DnD event on IPC.
        try {
          const mfId = String((manifest as any)?.metadata?.id || "").trim();
          if (mfId) {
            const newClipId = String((newClip as any)?.clipId || "");
            void (async () => {
              try {
                const defaults = await getOffloadDefaultsForManifest(mfId);
                if (!defaults) return;
                const store = useClipStore.getState();
                const existing = store.getClipById(newClipId) as ModelClipProps | undefined;
                if (!existing || existing.type !== "model") return;
                const currentOffload = (existing as any).offload;
                if (currentOffload && typeof currentOffload === "object" && Object.keys(currentOffload).length > 0) {
                  return;
                }
                store.updateClip(newClipId, { offload: defaults } as any);
              } catch {
                // ignore
              }
            })();
          }
        } catch {
          // ignore
        }
        cleanup();
      } else {
        addClip(newClip as AnyClipProps);

        // Upgrade to full stats when clip is added to timeline (for media clips only)
        if (
          (data as MediaItem)?.assetUrl &&
          ((data as MediaItem)?.type === "video" ||
            (data as MediaItem)?.type === "audio")
        ) {
          void ensureFullMediaStats((data as MediaItem)?.assetUrl);
        }

        setActiveMediaItem(null);
        setGhostTimelineId(null);
        setGhostStartEndFrame(0, 0);
        setGhostX(0);
      }

      // Baseline and zoom level recalibration happen within addClip -> _updateZoomLevel
    },
    [
      size.width,
      hoveredTimelineId,
      controlStore,
      dimensions.stageWidth,
      haldClutRef,
      addTimeline,
      setActiveMediaItem,
      setGhostTimelineId,
      setGhostStartEndFrame,
      setGhostX,
      setHoveredTimelineId,
      addClip,
    ],
  );

  const handlePreprocessorDrop = useCallback(
    (_event: DragEndEvent, _data: Preprocessor) => {
      let draggedPreprocessor = preprocessorDragClipRef.current;
      if (!draggedPreprocessor) {
        setGhostInStage(false);
        return;
      }
      const clipId = draggedPreprocessor.clipId;
      if (!clipId) {
        setGhostInStage(false);
        preprocessorDragClipRef.current = null;
        return;
      }
      const clip = clips.find((c) => c.clipId === clipId);
      if (!clip || (clip.type !== "video" && clip.type !== "image")) {
        setGhostInStage(false);
        preprocessorDragClipRef.current = null;
        return;
      }

      const clipDuration = (clip.endFrame ?? 0) - (clip.startFrame ?? 0);
      const allPreprocessors = clip.preprocessors || [];
      const otherPreprocessors = getOtherPreprocessors(
        allPreprocessors,
        draggedPreprocessor.id,
      );
      const startFrame = draggedPreprocessor.startFrame ?? 0;
      const endFrame = draggedPreprocessor.endFrame ?? clipDuration;
      const preprocessorDuration = endFrame - startFrame;

      // Check if preprocessor is within clip bounds
      const isOutOfBounds =
        startFrame < 0 ||
        endFrame > clipDuration ||
        startFrame >= clipDuration ||
        endFrame <= 0;

      // Check for collisions with other preprocessors
      const collisions = otherPreprocessors.filter((p) => {
        const pStart = p.startFrame ?? 0;
        const pEnd = p.endFrame ?? clipDuration;
        return !(endFrame <= pStart || startFrame >= pEnd);
      });

      // If no collisions and within bounds, keep it where it is
      if (collisions.length === 0 && !isOutOfBounds) {
        setGhostInStage(false);
        preprocessorDragClipRef.current = null;
        preprocessorDragKonvaNodeRef.current = null;
        setSelectedPreprocessorId(draggedPreprocessor.id);
        return;
      }

      let targetStartFrame: number | null = null;
      const sortedOthers = [...otherPreprocessors].sort(
        (a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0),
      );
      const gaps: [number, number][] = [];
      let prevEnd = 0;
      for (const p of sortedOthers) {
        const pStart = p.startFrame ?? 0;
        const pEnd = p.endFrame ?? clipDuration;
        if (pStart > prevEnd) gaps.push([prevEnd, pStart]);
        prevEnd = Math.max(prevEnd, pEnd);
      }
      if (prevEnd < clipDuration) gaps.push([prevEnd, clipDuration]);
      const validGaps = gaps.filter(
        ([lo, hi]) => hi - lo >= preprocessorDuration,
      );
      if (validGaps.length === 0) {
        removePreprocessorFromClip(clipId, draggedPreprocessor.id);
        setGhostInStage(false);
        preprocessorDragClipRef.current = null;
        preprocessorDragKonvaNodeRef.current = null;
        return;
      }
      const currentCenter = startFrame + preprocessorDuration / 2;
      let bestGap = validGaps[0];
      let bestDistance = Infinity;
      for (const gap of validGaps) {
        const gapCenter = (gap[0] + gap[1]) / 2;
        const distance = Math.abs(currentCenter - gapCenter);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestGap = gap;
        }
      }
      const [gapStart, gapEnd] = bestGap;
      const maxStart = gapEnd - preprocessorDuration;
      targetStartFrame = Math.max(gapStart, Math.min(startFrame, maxStart));
      const targetEndFrame = targetStartFrame + preprocessorDuration;

      // Ensure the target position is within clip bounds
      if (targetStartFrame < 0 || targetEndFrame > clipDuration) {
        // Clamp to clip bounds
        targetStartFrame = Math.max(
          0,
          Math.min(targetStartFrame, clipDuration - preprocessorDuration),
        );
        const clampedEndFrame = targetStartFrame + preprocessorDuration;

        // If still doesn't fit, remove it
        if (targetStartFrame < 0 || clampedEndFrame > clipDuration) {
          removePreprocessorFromClip(clipId, draggedPreprocessor.id);
          setGhostInStage(false);
          preprocessorDragClipRef.current = null;
          preprocessorDragKonvaNodeRef.current = null;
          return;
        }
      }

      // Update the preprocessor data
      updatePreprocessor(clipId, draggedPreprocessor.id, {
        startFrame: targetStartFrame,
        endFrame: targetEndFrame,
      });

      // Synchronize the visual X position with the updated frame position
      if (preprocessorDragKonvaNodeRef.current) {
        const [visibleStartFrame, visibleEndFrame] =
          useControlsStore.getState().timelineDuration;
        const timelineWidth = dimensions.stageWidth;
        const timelinePadding = 24;

        // Convert relative frame (within clip) to absolute frame (on timeline)
        const absoluteStartFrame = (clip.startFrame ?? 0) + targetStartFrame;

        // Calculate X position from absolute frame
        const progress =
          (absoluteStartFrame - visibleStartFrame) /
          (visibleEndFrame - visibleStartFrame);
        const targetX = progress * timelineWidth + timelinePadding;

        // Update Konva node position
        preprocessorDragKonvaNodeRef.current.x(targetX);
      }

      // Reset ghost state
      setGhostInStage(false);
      preprocessorDragClipRef.current = null;
      preprocessorDragKonvaNodeRef.current = null;
    },
    [
      clips,
      getOtherPreprocessors,
      removePreprocessorFromClip,
      updatePreprocessor,
      dimensions.stageWidth,
      setGhostInStage,
    ],
  );

  const handlePreprocessorDragMove = useCallback(
    (_data: Preprocessor, pointerY: number, pointerX: number) => {
      const timelinePadding = 24;
      const stageWidth = dimensions.stageWidth;
      const timelines = useClipStore.getState().timelines;
      const [visibleStartFrame, visibleEndFrame] =
        useControlsStore.getState().timelineDuration;
      const fps = controlStore.fps;

      // Find media clip (video/image) under pointer and calculate distance
      let targetClip: AnyClipProps | null = null;
      let targetTimeline: TimelineProps | null = null;

      let minDistance = Infinity;
      for (let timeline of timelines) {
        if (timeline.type !== "media") continue;

        const { top, bottom, left, right } = getTimelinePosition(
          timeline.timelineId,
          verticalScrollRef.current,
        );

        // Calculate distance to timeline bounds
        const dx = Math.max(left - pointerX, 0, pointerX - right);
        const dy = Math.max(top - pointerY, 0, pointerY - bottom);
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < minDistance) {
          minDistance = distance;
          targetTimeline = timeline;
        }
      }

      if (!targetTimeline) {
        // Only allow removal if the pointer leaves vertically (top/bottom) of the original clip
        const existing = preprocessorDragClipRef.current;
        if (existing) {
          const allTimelines = useClipStore.getState().timelines;
          let originalClip: AnyClipProps | null = null;
          for (const t of allTimelines) {
            const tClips = getClipsForTimeline(t.timelineId);
            const found = tClips.find((c) => c.clipId === existing.clipId);
            if (found) {
              originalClip = found;
              break;
            }
          }
          if (originalClip) {
            const { top, bottom } = getClipPosition(
              originalClip.clipId,
              verticalScrollRef.current,
            );
            const isVerticallyAligned = pointerY >= top && pointerY <= bottom;
            if (isVerticallyAligned) {
              // Keep using the original clip without removing
              targetTimeline =
                timelines.find((t) =>
                  getClipsForTimeline(t.timelineId).some(
                    (c) => c.clipId === originalClip!.clipId,
                  ),
                ) || null;
              targetClip = originalClip;
            }
          }
        }

        if (!targetClip) {
          setGhostInStage(false);
          if (preprocessorDragClipRef.current) {
            removePreprocessorFromClip(
              preprocessorDragClipRef.current.clipId ?? "",
              preprocessorDragClipRef.current.id,
            );
            preprocessorDragClipRef.current = null;
            try {
              preprocessorDragKonvaNodeRef.current?.remove();
            } catch (error) {
              console.error(
                "Error removing preprocessor drag konva node",
                error,
              );
            } finally {
              preprocessorDragKonvaNodeRef.current = null;
            }
          }
          return;
        }
      }
      // only look at clips on the target timeline
      if (!targetTimeline) {
        return;
      }

      const clips = getClipsForTimeline(targetTimeline.timelineId);
      for (const clip of clips) {
        const { top, bottom, left, right } = getClipPosition(
          clip.clipId,
          verticalScrollRef.current,
        );
        const isInside =
          pointerY >= top &&
          pointerY <= bottom &&
          pointerX >= left &&
          pointerX <= right;
        if (isInside && (clip.type === "image" || clip.type === "video")) {
          targetClip = clip;
          break;
        }
      }

      if (!targetClip) {
        // If we're horizontally outside the clip but still within the original clip's vertical band,
        // treat this as dragging within the original clip instead of removing.
        const existing = preprocessorDragClipRef.current;
        if (existing) {
          // Find the original clip for the dragging preprocessor
          const allTimelines = useClipStore.getState().timelines;
          let originalClip: AnyClipProps | null = null;
          for (const t of allTimelines) {
            const tClips = getClipsForTimeline(t.timelineId);
            const found = tClips.find((c) => c.clipId === existing.clipId);
            if (found) {
              originalClip = found;
              break;
            }
          }
          if (originalClip) {
            const { top, bottom } = getClipPosition(
              originalClip.clipId,
              verticalScrollRef.current,
            );
            const isVerticallyAligned = pointerY >= top && pointerY <= bottom;
            if (isVerticallyAligned) {
              targetClip = originalClip;
            }
          }
        }

        if (!targetClip) {
          setGhostInStage(false);
          if (preprocessorDragClipRef.current) {
            removePreprocessorFromClip(
              preprocessorDragClipRef.current.clipId ?? "",
              preprocessorDragClipRef.current.id,
            );
            preprocessorDragClipRef.current = null;
            try {
              preprocessorDragKonvaNodeRef.current?.remove();
            } catch (error) {
              console.error(
                "Error removing preprocessor drag konva node",
                error,
              );
            } finally {
              preprocessorDragKonvaNodeRef.current = null;
            }
          }
          return;
        }
      }

      // Hide the ghost overlay when dragging over a valid clip
      setGhostInStage(true);

      // Calculate start frame from mouse position
      const timelineWidth = stageWidth;
      const absoluteStartFrame = calculateFrameFromX(
        pointerX,
        timelinePadding,
        timelineWidth,
        [visibleStartFrame, visibleEndFrame],
      );

      // Convert to relative frame within the clip
      const relativeStartFrame = Math.max(
        0,
        absoluteStartFrame - targetClip.startFrame!,
      );
      const clipDuration = targetClip.endFrame! - targetClip.startFrame!;

      // Get existing preprocessors to find the largest gap anywhere in the clip
      const existingPreprocessors =
        (targetClip as VideoClipProps | ImageClipProps).preprocessors || [];
      const otherPreprocessors = getOtherPreprocessors(
        existingPreprocessors,
        "",
      );

      // Find all gaps in the clip timeline
      const sortedPreprocessors = [...otherPreprocessors].sort(
        (a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0),
      );
      const gaps: [number, number][] = [];
      let prevEnd = 0;

      for (const preprocessor of sortedPreprocessors) {
        const pStart = preprocessor.startFrame ?? 0;
        const pEnd = preprocessor.endFrame ?? clipDuration;
        if (pStart > prevEnd) {
          gaps.push([prevEnd, pStart]);
        }
        prevEnd = Math.max(prevEnd, pEnd);
      }
      if (prevEnd < clipDuration) {
        gaps.push([prevEnd, clipDuration]);
      }

      // Find the largest gap anywhere in the clip
      let largestGapSize = 0; // Default to full clip if no preprocessors
      if (gaps.length > 0) {
        largestGapSize = Math.max(...gaps.map(([start, end]) => end - start));
      }

      // If there's no space available in the clip (completely full), don't add the preprocessor
      if (largestGapSize <= 0) {
        setGhostInStage(false);
        return;
      }

      // Calculate desired duration (5 seconds or largest gap, whichever is smaller)
      const desiredDuration = 5 * fps;
      const actualDuration = Math.min(
        desiredDuration,
        largestGapSize,
        clipDuration - relativeStartFrame,
      );
      
      const relativeEndFrame = Math.min(
        relativeStartFrame + actualDuration,
        clipDuration,
      );

      // If the calculated duration is too small (less than 1 frame), don't add the preprocessor
      if (actualDuration < 1) {
        setGhostInStage(false);
        return;
      }

      // fill the values with the default values
      const values: Record<string, any> = {};
      for (const param of _data.parameters || []) {
        values[param.name] = param.default;
      }

      // For image clips, always span the entire duration
      const finalStartFrame =
        targetClip.type === "image" ? 0 : relativeStartFrame;
      const finalEndFrame =
        targetClip.type === "image" ? clipDuration : relativeEndFrame;

      const newPreprocessorClip: PreprocessorClipProps = {
        clipId: targetClip.clipId,
        id: uuidv4(),
        preprocessor: _data,
        startFrame: finalStartFrame,
        endFrame: finalEndFrame,
        values: values,
      };

      if (targetClip.type === "image") {
        if (!_data.supports_image) {
          return;
        }
      }
      if (targetClip.type === "video") {
        if (!_data.supports_video) {
          return;
        }
      }

      if (preprocessorDragClipRef.current) {
        const stage = timelinesLayerRef.current?.getStage();
        if (!stage) return;
        const konvaNode = stage?.findOne(
          `#${preprocessorDragClipRef.current.id}`,
        );
        // Now you have the Konva node
        if (konvaNode && !preprocessorDragKonvaNodeRef.current) {
          preprocessorDragKonvaNodeRef.current = konvaNode;
          preprocessorDragKonvaNodeRef.current.moveToTop();
        }

        // Check if clip has changed
        const hasClipChanged =
          preprocessorDragClipRef.current.clipId !== targetClip.clipId;

        if (hasClipChanged) {
          // Remove from old clip
          removePreprocessorFromClip(
            preprocessorDragClipRef.current.clipId ?? "",
            preprocessorDragClipRef.current.id,
          );

          // Calculate the absolute timeline frame from pointer position
          const absoluteFrame = calculateFrameFromX(
            pointerX,
            timelinePadding,
            timelineWidth,
            [visibleStartFrame, visibleEndFrame],
          );

          // Convert to relative frame within the new clip
          const rawStart = absoluteFrame - targetClip.startFrame!;
          const rawDuration =
            (preprocessorDragClipRef.current.endFrame ?? 0) -
            (preprocessorDragClipRef.current.startFrame ?? 0);
          const desiredDuration = Math.min(
            Math.max(rawDuration, 0),
            clipDuration,
          );
          const maxStart = Math.max(0, clipDuration - desiredDuration);
          const newRelativeStartFrame = Math.max(
            0,
            Math.min(rawStart, maxStart),
          );
          const newRelativeEndFrame = newRelativeStartFrame + desiredDuration;

          // Update clipId and add to new clip
          preprocessorDragClipRef.current.clipId = targetClip.clipId;
          preprocessorDragClipRef.current.startFrame = newRelativeStartFrame;
          preprocessorDragClipRef.current.endFrame = newRelativeEndFrame;

          addPreprocessorToClip(
            targetClip.clipId,
            preprocessorDragClipRef.current,
          );
        } else {
          // Same clip - just update position
          // Calculate the absolute timeline frame from pointer position
          const absoluteFrame = calculateFrameFromX(
            pointerX,
            timelinePadding,
            timelineWidth,
            [visibleStartFrame, visibleEndFrame],
          );

          // Convert to relative frame within the clip
          const rawStart = absoluteFrame - targetClip.startFrame!;
          const clipDuration = targetClip.endFrame! - targetClip.startFrame!;
          const rawDuration =
            (preprocessorDragClipRef.current.endFrame ?? 0) -
            (preprocessorDragClipRef.current.startFrame ?? 0);
          const desiredDuration = Math.min(
            Math.max(rawDuration, 0),
            clipDuration,
          );
          const maxStart = Math.max(0, clipDuration - desiredDuration);
          const newRelativeStartFrame = Math.max(
            0,
            Math.min(rawStart, maxStart),
          );
          const newRelativeEndFrame = newRelativeStartFrame + desiredDuration;

          // Calculate the visual X position from the frame data (ensure consistency)
          const finalAbsoluteFrame =
            targetClip.startFrame! + newRelativeStartFrame;
          const progress =
            (finalAbsoluteFrame - visibleStartFrame) /
            (visibleEndFrame - visibleStartFrame);
          const visualX = progress * stageWidth + timelinePadding;

          // Update visual position
          preprocessorDragKonvaNodeRef.current?.x(visualX);

          // Update data
          preprocessorDragClipRef.current.startFrame = newRelativeStartFrame;
          preprocessorDragClipRef.current.endFrame = newRelativeEndFrame;
          updatePreprocessor(
            targetClip.clipId,
            preprocessorDragClipRef.current.id,
            {
              startFrame: newRelativeStartFrame,
              endFrame: newRelativeEndFrame,
            },
          );
        }
        return;
      }

      preprocessorDragClipRef.current = newPreprocessorClip;

      // we only want to add the preprocessor to the clip if it's not already there
      if (
        (targetClip as VideoClipProps | ImageClipProps).preprocessors?.some(
          (p) => p.id === newPreprocessorClip.id,
        )
      ) {
        return;
      } else {
        addPreprocessorToClip(targetClip.clipId, newPreprocessorClip);
      }
    },
    [
      dimensions.stageWidth,
      controlStore.fps,
      verticalScrollRef.current,
      addTimeline,
      removeTimeline,
      setGhostStartEndFrame,
      setGhostTimelineId,
      setGhostInStage,
      setGhostX,
      setGhostGuideLines,
      clips,
      timelines,
      addPreprocessorToClip,
      removePreprocessorFromClip,
      updatePreprocessor,
      getOtherPreprocessors,
    ],
  );

  // Compute vertical content height for scrollbar logic
  const contentHeight = useMemo(() => {
    if (!timelines || timelines.length === 0) return dimensions.stageHeight;
    const bottoms = timelines.map((t) => {
      const y = t.timelineY || 0;
      const h = t.timelineHeight || 54;
      return y + h + 32; // include padding/offsets in Timeline rendering
    });
    return Math.max(...bottoms, 0) + SCROLL_BOTTOM_PADDING;
  }, [timelines, dimensions.stageHeight]);

  const maxScroll = useMemo(
    () => Math.max(0, contentHeight - dimensions.stageHeight),
    [contentHeight, dimensions.stageHeight],
  );
  const clampedScroll = useMemo(
    () => Math.max(0, Math.min(verticalScroll, maxScroll)),
    [verticalScroll, maxScroll],
  );

  useEffect(() => {
    setVerticalScroll((prev) => Math.max(0, Math.min(prev, maxScroll)));
  }, [maxScroll]);

  useEffect(() => {
    verticalScrollRef.current = clampedScroll;
  }, [clampedScroll]);

  // Auto-scroll when dragging near top/bottom edges (both external DnD and internal clip drags)
  const edgeAutoScroll = useCallback(
    (pointerY: number) => {
      if (maxScroll <= 0) return;
      const stageHeight = dimensions.stageHeight || 0;
      const TOP_MARGIN = 24; // ruler height
      const EDGE_ZONE = 36; // px threshold near edges
      const MAX_DELTA = 28; // px per invocation

      // Scroll up when within top edge zone (below the ruler)
      const topEdgeY = TOP_MARGIN + EDGE_ZONE;
      if (pointerY <= topEdgeY) {
        const intensity = Math.max(0, (topEdgeY - pointerY) / EDGE_ZONE);
        const delta = Math.max(6, Math.round(intensity * MAX_DELTA));
        setVerticalScroll((prev) => Math.max(0, prev - delta));
        return;
      }

      // Scroll down when within bottom edge zone
      const bottomEdgeStart = Math.max(0, stageHeight - EDGE_ZONE);
      if (pointerY >= bottomEdgeStart) {
        const intensity = Math.max(0, (pointerY - bottomEdgeStart) / EDGE_ZONE);
        const delta = Math.max(6, Math.round(intensity * MAX_DELTA));
        setVerticalScroll((prev) => Math.min(maxScroll, prev + delta));
      }
    },
    [dimensions.stageHeight, maxScroll],
  );

  // Listen for internal clip drag auto-scroll events
  useEffect(() => {
    const handler = (e: Event) => {
      try {
        const anyEvt = e as any;
        const y = Number(anyEvt?.detail?.y);
        if (!Number.isFinite(y)) return;
        edgeAutoScroll(y);
      } catch {}
    };
    window.addEventListener("timeline-editor-autoscroll", handler as any);
    return () => {
      window.removeEventListener("timeline-editor-autoscroll", handler as any);
    };
  }, [edgeAutoScroll]);

  // Horizontal scroll/pan with mouse wheel or trackpad
  const handleWheelScroll = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      if (!dimensions.stageWidth) return;
      const absX = Math.abs(e.deltaX);
      const absY = Math.abs(e.deltaY);
      const isHorizontalIntent = absX >= absY || e.shiftKey;

      if (isHorizontalIntent) {
        const [startFrame, endFrame] = controlStore.timelineDuration;
        const duration = endFrame - startFrame;
        const framesPerPixel = duration / dimensions.stageWidth;
        const speedMultiplier = e.shiftKey ? 3 : 1;
        const deltaFramesFloat =
          e.deltaX * framesPerPixel * speedMultiplier +
          wheelRemainderRef.current;
        const integerShift = Math.trunc(deltaFramesFloat);
        wheelRemainderRef.current = deltaFramesFloat - integerShift;
        if (integerShift !== 0) {
          if (controlStore.canTimelineDurationBeShifted(integerShift)) {
            controlStore.shiftTimelineDuration(integerShift, true, true);
          }
        }
      } else if (maxScroll > 0) {
        const pxDelta = e.deltaY;
        if (pxDelta !== 0) {
          setVerticalScroll((prev) =>
            Math.max(0, Math.min(prev + pxDelta, maxScroll)),
          );
        }
      }
    },
    [dimensions.stageWidth, controlStore.timelineDuration, maxScroll],
  );

  // Drag to pan when clicking the top ruler area
  const handleStageMouseDown = useCallback(
    (e: any) => {
      const stage = e?.target?.getStage?.();
      if (!stage) return;
      const pos = stage.getPointerPosition();
      if (!pos) return;
      const tickAreaHeight = 24; // top ruler area height
      if (pos.y <= tickAreaHeight) {
        setIsRulerDragging(true);
        panStateRef.current.startX = pos.x;
        panStateRef.current.lastX = pos.x;
        panStateRef.current.startFrame = controlStore.timelineDuration[0];
        panStateRef.current.fractionalFrames = 0;
        if (e?.evt?.preventDefault) e.evt.preventDefault();
      }
    },
    [controlStore.timelineDuration],
  );

  const handleStageMouseMove = useCallback(
    (e: any) => {
      if (!isRulerDragging) return;
      const stage = e?.target?.getStage?.();
      if (!stage) return;
      const pos = stage.getPointerPosition();
      if (!pos) return;
      const deltaX = pos.x - panStateRef.current.lastX;
      const [startFrame, endFrame] = controlStore.timelineDuration;
      const duration = endFrame - startFrame;
      const framesPerPixel = duration / Math.max(1, dimensions.stageWidth);
      const deltaFramesFloat =
        deltaX * framesPerPixel + panStateRef.current.fractionalFrames;
      const integerShift = Math.trunc(deltaFramesFloat);
      panStateRef.current.fractionalFrames = deltaFramesFloat - integerShift;
      if (integerShift !== 0) {
        // update focus frame

        if (controlStore.canTimelineDurationBeShifted(integerShift)) {
          controlStore.shiftTimelineDuration(integerShift, true);
        }
      }
      panStateRef.current.lastX = pos.x;
    },
    [isRulerDragging, controlStore.timelineDuration, dimensions.stageWidth],
  );

  const endRulerDrag = useCallback(() => {
    if (isRulerDragging) setIsRulerDragging(false);
  }, [isRulerDragging]);

  const hasClips = useMemo(() => clips.length > 0, [clips]);

  const handleStageClick = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      // Check if the click target is the stage itself (background click)
      if (e.target === e.target.getStage()) {
        controlStore.clearSelection();
        setSelectedPreprocessorId(null);
        const stage = e.target.getStage();
        const pos = stage?.getPointerPosition();
        if (!pos) return;
        // Map pointer X to frame within the visible window [startFrame, endFrame]
        const timelinePadding = 24; // left padding used by timeline
        const [startFrame, endFrame] = controlStore.timelineDuration;
        const innerX = Math.max(
          0,
          Math.min(pos.x - timelinePadding, dimensions.stageWidth),
        );
        const progress =
          dimensions.stageWidth > 0 ? innerX / dimensions.stageWidth : 0;
        const targetFrame = Math.round(
          startFrame + progress * (endFrame - startFrame),
        );

        // Pause if playing, then set focus without auto-resume
        if (controlStore.isPlaying) controlStore.pause();
        controlStore.setFocusAnchorRatio(Math.max(0, Math.min(1, progress)));
        controlStore.setFocusFrame(targetFrame, false);
        controlStore.setIsAccurateSeekNeeded(true);
      }
    },
    [controlStore, setSelectedPreprocessorId],
  );

  return (
    <div className="relative h-full flex flex-row overflow-hidden">
      {hasClips && <TimelineSidebar clampedScroll={clampedScroll} />}
      <div
        className="relative h-full w-full overflow-hidden"
        ref={containerRef}
        onWheel={handleWheelScroll}
      >
        {dimensions.stageWidth > 0 && dimensions.stageHeight > 0 && (
          <>
            {hasClips && (
              <>
                <Stage
                  key={hmrStageKey}
                  width={dimensions.stageWidth}
                  height={dimensions.stageHeight}
                  className="border-b border-brand-light/10 bg-brand z-10 relative"
                  onClick={handleStageClick}
                  onMouseDown={handleStageMouseDown}
                  onMouseMove={handleStageMouseMove}
                  onMouseUp={endRulerDrag}
                  onMouseLeave={endRulerDrag}
                  style={{ cursor: isRulerDragging ? "grabbing" : "default" }}
                >
                  <Layer
                    ref={timelinesLayerRef}
                    visible={hasClips}
                    y={-clampedScroll}
                  >
                    {timelines.map((timeline, index) => (
                      <Timeline
                      
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
                        assetMode={false}
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
                        const maxThumbY = Math.max(
                          0,
                          trackHeight - thumbHeight,
                        );
                        const thumbY =
                          trackTop +
                          (maxScroll > 0
                            ? Math.round(
                                (clampedScroll / maxScroll) * maxThumbY,
                              )
                            : 0);
                        return thumbY;
                      }}
                    />
                  </Layer>
                  {/* Snap guideline overlay */}
                  <Layer listening={false}>
                    {typeof snapGuideX === "number" && (
                      <Line
                        points={[
                          snapGuideX,
                          0,
                          snapGuideX,
                          dimensions.stageHeight,
                        ]}
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
                        const maxThumbY = Math.max(
                          0,
                          trackHeight - thumbHeight,
                        );
                        const thumbY =
                          trackTop +
                          (maxScroll > 0
                            ? Math.round(
                                (clampedScroll / maxScroll) * maxThumbY,
                              )
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
              </>
            )}
            {!hasClips && (
              <>
                {/* Overlay helper text/icon centered */}

                <div
                  style={{ display: clips.length > 0 ? "none" : "flex" }}
                  className="h-full items-center justify-center w-full p-8"
                >
                  <Droppable
                    id="timeline"
                    className="w-full rounded-lg bg-brand-background/60 text-brand-light/90 duration-100 ease-in-out "
                    accepts={["media"]}
                    highlight={{
                      borderColor: "#A477C4",
                      textColor: "#FFFFFF",
                      bgColor: "#A477C4",
                    }}
                  >
                    <div className="w-full group py-6 px-10  rounded-lg  flex items-center ">
                      <div className=" mx-auto w-full flex items-center font-sans pointer-events-auto">
                        <h4 className=" flex items-center flex-row leading-none gap-x-3.5">
                          <span className="flex items-center justify-center leading-none">
                            <GoFileMedia className="w-5 h-5 " />
                          </span>
                          <span className="text-[12px] font-light font-poppins leading-none ">
                            Drag and drop media to start creating
                          </span>
                        </h4>
                      </div>
                    </div>
                  </Droppable>
                </div>
              </>
            )}
          </>
        )}
        <div style={{ display: hasClips ? undefined : "none" }}>
          <ScrubControl
            stageHeight={dimensions.stageHeight - 30}
            stageWidth={dimensions.stageWidth}
          />
        </div>
      </div>
    </div>
  );
});

export default TimelineEditor;
