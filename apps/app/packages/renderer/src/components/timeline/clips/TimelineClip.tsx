import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  useClipStore,
  getClipWidth,
  getClipX,
  isValidTimelineForClip,
  getTimelineTypeForClip,
} from "@/lib/clip";
import { generatePosterCanvas } from "@/lib/media/timeline";
import { useControlsStore } from "@/lib/control";
import { useAssetControlsStore } from "@/lib/assetControl";
import { Image, Group, Rect, Text, Line } from "react-konva";
import Konva from "konva";
import { sanitizeCornerRadius } from "@/lib/konva/sanitizeCornerRadius";
import {
  MediaInfo,
  ShapeClipProps,
  TextClipProps,
  TimelineProps,
  VideoClipProps,
  ImageClipProps,
  ClipType,
  FilterClipProps,
  MaskClipProps,
  PreprocessorClipType,
  GroupClipProps,
  ModelClipProps,
  AudioClipProps,
} from "@/lib/types";
import { v4 as uuidv4 } from "uuid";
import {
  generateAudioWaveformCanvas,
  getMediaInfoCached,
} from "@/lib/media/utils";
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import { useWebGLMask } from "@/components/preview/mask/useWebGLMask";
import { PreprocessorClip } from "./PreprocessorClip";
import MaskKeyframes from "./MaskKeyframes";
import { useViewportStore } from "@/lib/viewport";
import { useContextMenuStore, ContextMenuItem } from "@/lib/context-menu";
import { renderToStaticMarkup } from "react-dom/server";
import { RxText as RxTextIcon } from "react-icons/rx";
import {
  MdOutlineDraw as MdOutlineDrawIcon,
  MdMovie as MdMovieIcon,
  MdImage as MdImageIcon,
  MdAudiotrack as MdAudiotrackIcon,
} from "react-icons/md";
import {
  LuShapes as LuShapeIcon,
  LuBox as LuBoxIcon,
  LuCheck as LuCheckIcon,
  LuPointer,
} from "react-icons/lu";
import { MdPhotoFilter as MdFilterIcon } from "react-icons/md";
import { ManifestDocument } from "@/lib/manifest/api";
import ModelClip from "./ModelClip";

import {
  generateTimelineThumbnailAudio,
  generateTimelineThumbnailImage,
  generateTimelineThumbnailVideo,
  generateTimelineThumbnailShape,
  generateTimelineThumbnailText,
  generateTimelineThumbnailFilter,
  generateTimelineThumbnailDrawing,
} from "./thumbnails";
import { Html } from "react-konva-utils";
import { cn } from "@/lib/utils";

/**
 * text rx/RxText
 * image fa6/FaRegFileImage
 * video fa6/FaRegFileVideo
 * audio fa/FaRegFileAudio
 * image+mask fa6/FaRegFileImage + tb/TbMask
 * video+mask fa6/FaRegFileVideo2 + tb/TbMask
 * image+preprocessor ri/RiImageAiLine
 * video+preprocessor ri/RiVideoAiLine
 * image_list lu/LuImages
 * video_list bi/BiSolidVideos
 */

const TimelineClip: React.FC<
  TimelineProps & {
    clipId: string;
    clipType: ClipType;
    scrollY: number;
    cornerRadius?: number;
    assetMode?: boolean;
    isAssetSelected?: (clipId: string) => boolean;
  }
> = ({
  timelineWidth = 0,
  timelineY = 0,
  timelineHeight = 54,
  timelinePadding = 24,
  clipId,
  timelineId,
  clipType,
  scrollY,
  cornerRadius = 1,
  assetMode = false,
  isAssetSelected = () => false,
}) => {
  // Select only what we need to avoid unnecessary rerenders
  const ctrlTimelineDuration = useControlsStore((s) => s.timelineDuration);
  const ctrlSelectedClipIds = useControlsStore((s) => s.selectedClipIds);
  const ctrlToggleClipSelection = useControlsStore(
    (s) => s.toggleClipSelection,
  );
  const ctrlZoomLevel = useControlsStore((s) => s.zoomLevel);
  const assetTimelineDuration = useAssetControlsStore(
    (s) => s.timelineDuration,
  );
  const assetZoomLevel = useAssetControlsStore((s) => s.zoomLevel);
  const assetSelectedAssetClipId = useAssetControlsStore(
    (s) => s.selectedAssetClipId,
  );
  const setSelectedAssetClipId = useAssetControlsStore(
    (s) => s.setSelectedAssetClipId,
  );

  const timelineDuration = assetMode
    ? assetTimelineDuration
    : ctrlTimelineDuration;
  const zoomLevel = assetMode ? assetZoomLevel : ctrlZoomLevel;
  const resizeClip = useClipStore((s) => s.resizeClip);
  const moveClipToEnd = useClipStore((s) => s.moveClipToEnd);
  const updateClip = useClipStore((s) => s.updateClip);
  const setGhostX = useClipStore((s) => s.setGhostX);
  const setGhostTimelineId = useClipStore((s) => s.setGhostTimelineId);
  const setGhostStartEndFrame = useClipStore((s) => s.setGhostStartEndFrame);
  const getClipsForGroup = useClipStore((s) => s.getClipsForGroup);
  const setGhostInStage = useClipStore((s) => s.setGhostInStage);
  const setDraggingClipId = useClipStore((s) => s.setDraggingClipId);
  const getClipsForTimeline = useClipStore((s) => s.getClipsForTimeline);
  const getTimelineById = useClipStore((s) => s.getTimelineById);
  const setHoveredTimelineId = useClipStore((s) => s.setHoveredTimelineId);
  const setSnapGuideX = useClipStore((s) => s.setSnapGuideX);
  const addTimeline = useClipStore((s) => s.addTimeline);
  const setSelectedPreprocessorId = useClipStore(
    (s) => s.setSelectedPreprocessorId,
  );
  const getAssetById = useClipStore((s) => s.getAssetById);
  const setIsDraggingGlobal = useClipStore((s) => s.setIsDragging);
  const ctrlFocusFrame = useControlsStore((s) => s.focusFrame);
  const tool = useViewportStore((s) => s.tool);
  const getClipById = useClipStore((s) => s.getClipById);
  const [groupedCanvases, setGroupedCanvases] = useState<HTMLCanvasElement[]>(
    [],
  );
  const [groupCounts, setGroupCounts] = useState<{
    video: number;
    image: number;
    audio: number;
    text: number;
    draw: number;
    filter: number;
    shape: number;
    model: number;
  }>({
    video: 0,
    image: 0,
    audio: 0,
    text: 0,
    draw: 0,
    filter: 0,
    shape: 0,
    model: 0,
  });
  // Subscribe directly to this clip's data
  const currentClip = useClipStore((s) =>
    s.clips.find(
      (c) =>
        c.clipId === clipId &&
        (timelineId ? c.timelineId === timelineId : true),
    ),
  );

  const assetFocusFrame = useAssetControlsStore((s) => s.focusFrame);
  const { applyMask } = useWebGLMask({
    focusFrame: assetMode ? assetFocusFrame : ctrlFocusFrame,
    masks:
      (currentClip as PreprocessorClipType & { masks: MaskClipProps[] })
        ?.masks || [],
    disabled:
      tool === "mask" ||
      (currentClip?.type !== "video" && currentClip?.type !== "image"),
    clip: currentClip,
  });

  // Check if clip has preprocessors (only for video/image clips)
  const hasPreprocessors = useMemo(() => {
    if (currentClip?.type !== "video" && currentClip?.type !== "image")
      return false;
    const preprocessors = (currentClip as VideoClipProps | ImageClipProps)
      ?.preprocessors;
    return preprocessors && preprocessors.length > 0;
  }, [currentClip]);

  // Total height including preprocessor bar if needed
  const totalClipHeight = useMemo(() => {
    return timelineHeight;
  }, [hasPreprocessors, timelineHeight]);

  const currentStartFrame = currentClip?.startFrame ?? 0;
  const currentEndFrame = currentClip?.endFrame ?? 0;

  const clipWidth = useMemo(
    () =>
      Math.max(
        getClipWidth(
          currentStartFrame,
          currentEndFrame,
          timelineWidth,
          timelineDuration,
        ),
        3,
      ),
    [currentStartFrame, currentEndFrame, timelineWidth, timelineDuration],
  );

  const safeCornerRadius = useMemo(
    () => sanitizeCornerRadius(cornerRadius, clipWidth, timelineHeight) as number,
    [cornerRadius, clipWidth, timelineHeight],
  );
  const clipX = useMemo(
    () =>
      getClipX(
        currentStartFrame,
        currentEndFrame,
        timelineWidth,
        timelineDuration,
      ),
    [
      currentStartFrame,
      currentEndFrame,
      timelineWidth,
      timelineDuration,
      timelineId,
    ],
  );
  const clipRef = useRef<Konva.Line>(null);
  const [resizeSide, setResizeSide] = useState<"left" | "right" | null>(null);
  const [imageCanvas] = useState<HTMLCanvasElement>(() =>
    document.createElement("canvas"),
  );
  const mediaInfoRef = useRef<MediaInfo | undefined>(
    getMediaInfoCached((currentClip as VideoClipProps | ImageClipProps)?.assetId!),
  );
  const dragInitialWindowRef = useRef<[number, number] | null>(null);
  const thumbnailClipWidth = useRef<number>(0);
  const maxTimelineWidth = useMemo(
    () => timelineWidth,
    [timelineWidth, timelinePadding],
  );
  const groupRef = useRef<Konva.Group>(null);
  const rootGroupRef = useRef<Konva.Group>(null);
  const exactVideoUpdateTimerRef = useRef<number | null>(null);
  const exactVideoUpdateSeqRef = useRef(0);
  const lastExactRequestKeyRef = useRef<string | null>(null);
  const textRef = useRef<Konva.Text>(null);
  const [textWidth, setTextWidth] = useState(0);
  const modelNameRef = useRef<Konva.Text>(null);
  const [modelNameWidth, setModelNameWidth] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const rectRefLeft = useRef<Konva.Rect>(null);
  const rectRefRight = useRef<Konva.Rect>(null);
  // global context menu used instead of local state
  const { applyFilters } = useWebGLFilters();
  const [forceRerenderCounter, setForceRerenderCounter] = useState(0);
  // Manifest data for model clips
  
  const [modelUiCounts, setModelUiCounts] = useState<Record<
    string,
    number
  > | null>(null);

  // Track engine job status for model clip to disable resizing while generating
  const isModelRunning = (currentClip as ModelClipProps | undefined)?.modelStatus === 'running' || (currentClip as ModelClipProps | undefined)?.modelStatus === 'pending';

  // Sizing for stacked canvases inside group clips
  const groupCardHeight = useMemo(
    () => Math.max(1, timelineHeight - 24),
    [timelineHeight],
  );
  const groupCardWidth = useMemo(
    () =>
      Math.max(
        1,
        Math.min(clipWidth - 24, Math.round((timelineHeight - 24) * 1.35)),
      ),
    [timelineHeight, clipWidth],
  );

  // (moved) image positioning is computed after clipPosition is defined

  useEffect(() => {
    if (!(currentClip as VideoClipProps | ImageClipProps)?.assetId) {
      mediaInfoRef.current = undefined;
      return;
    }
    mediaInfoRef.current = getMediaInfoCached((currentClip as VideoClipProps | ImageClipProps)?.assetId!);
  }, [(currentClip as VideoClipProps | ImageClipProps)?.assetId]);

  useEffect(() => {
    imageCanvas.width = Math.min(clipWidth, maxTimelineWidth);
    imageCanvas.height = timelineHeight;
  }, [
    zoomLevel,
    clipWidth,
    timelineHeight,
    timelineWidth,
    timelinePadding,
    maxTimelineWidth,
    clipType,
    imageCanvas,
  ]);

  useEffect(() => {
    return () => {
      if (exactVideoUpdateTimerRef.current != null) {
        window.clearTimeout(exactVideoUpdateTimerRef.current);
        exactVideoUpdateTimerRef.current = null;
      }
    };
  }, []);

  // Load manifest UI schema for model clip and compute input type counts
  useEffect(() => {
    if (!currentClip || currentClip.type !== "model") {
      setModelUiCounts(null);
      return;
    }
    const manifestId = (currentClip as ModelClipProps)?.manifest?.metadata?.id;
    if (!manifestId) {
      setModelUiCounts(null);
      return;
    }
    const computeCounts = (doc: ManifestDocument) => {
      const ui = doc?.ui || doc?.spec?.ui;
      if (!ui || !Array.isArray(ui.inputs)) {
        setModelUiCounts(null);
        return;
      }
      const counts: Record<string, number> = {};
      for (const inp of ui.inputs) {
        const t = String(inp?.type || "").toLowerCase();
        counts[t] = (counts[t] || 0) + 1;
      }
      setModelUiCounts(counts);
    };

    computeCounts((currentClip as ModelClipProps)?.manifest);
  }, [currentClip]);

  const restoreWindowIfChanged = useCallback((anchorStartFrame: number) => {
    try {
      const initial = dragInitialWindowRef.current;
      if (!initial) return;
      const [initStart, initEnd] = initial;
      const controls = useControlsStore.getState();
      const [curStart, curEnd] = controls.timelineDuration;
      if (initStart === curStart && initEnd === curEnd) return;
      const originalWindowLen = Math.max(1, initEnd - initStart);
      const desiredStart = anchorStartFrame;
      const desiredEnd = desiredStart + originalWindowLen;
      if (controls.totalTimelineFrames < desiredEnd) {
        controls.incrementTotalTimelineFrames(
          desiredEnd - controls.totalTimelineFrames,
        );
      }
      controls.setTimelineDuration(desiredStart, desiredEnd);
      if ((controls as any).setFocusFrame) {
        (controls as any).setFocusFrame(desiredStart);
      }
    } finally {
      dragInitialWindowRef.current = null;
    }
  }, []);

  // Use global selection state instead of local state
  const currentClipId = clipId;
  const isSelected = assetMode
    ? isAssetSelected(currentClipId) || assetSelectedAssetClipId === currentClipId
    : ctrlSelectedClipIds.includes(currentClipId);

  const showMaskKeyframes = useMemo(() => {
    if (!isSelected) return false;
    if (tool !== "mask") return false;
    if (!currentClip || currentClip.type !== "video") return false;
    if (isDragging) return false; // hide while dragging current clip
    const masks = (currentClip as VideoClipProps).masks ?? [];
    return masks.length > 0;
  }, [currentClip, isSelected, tool, isDragging]);

  const [clipPosition, setClipPosition] = useState<{ x: number; y: number }>({
    x: clipX + timelinePadding,
    y: timelineY - totalClipHeight,
  });
  const [tempClipPosition, setTempClipPosition] = useState<{
    x: number;
    y: number;
  }>({
    x: clipX + timelinePadding,
    y: timelineY - totalClipHeight,
  });
  const fixedYRef = useRef(timelineY - totalClipHeight);

  // Width used for the thumbnail image we render inside the clip group.
  const imageWidth = useMemo(
    () => Math.min(clipWidth, maxTimelineWidth),
    [clipWidth, maxTimelineWidth],
  );

  // Compute image x so that the image stays centered over the portion of the
  // group that is currently visible inside the stage viewport. This allows us
  // to "virtually" pan across long clips without rendering an infinitely wide image.
  const overHang = useMemo(() => {
    let overhang = 0;
    const positionX =
      clipPosition.x == 24 || clipWidth <= imageWidth ? 0 : -clipPosition.x;

    if (clipWidth - positionX <= timelineWidth && positionX > 0) {
      overhang = timelineWidth - (clipWidth - positionX);
    }
    return overhang;
  }, [clipPosition.x, clipWidth, timelineWidth]);

  const imageX = useMemo(() => {
    let overhang = 0;
    // Default behavior for clips that fit within timeline or are at the start
    const positionX =
      clipPosition.x == 24 || clipWidth <= imageWidth ? 0 : -clipPosition.x;
    if (clipWidth - positionX <= timelineWidth && positionX > 0) {
      overhang = timelineWidth - (clipWidth - positionX);
    }

    const imageX = positionX - overhang;
    return Math.max(0, imageX);
  }, [clipPosition.x, clipWidth, timelinePadding, timelineWidth, imageWidth]);

  // Clamp HTML overlay to timeline bounds and offset to keep it visible
  const absClipX = isDragging ? tempClipPosition.x : clipPosition.x;
  const timelineLeft = assetMode ? 0 : timelinePadding;
  const timelineRight = timelineLeft + timelineWidth;
  const htmlLocalX = Math.max(0, Math.round(timelineLeft - absClipX));
  const htmlVisibleWidth = Math.max(
    0,
    Math.min(clipWidth, Math.round(timelineRight - absClipX)) - htmlLocalX,
  );

  useEffect(() => {
    thumbnailClipWidth.current = Math.max(
      getClipWidth(
        currentStartFrame,
        currentEndFrame,
        timelineWidth,
        timelineDuration,
      ),
      3,
    );
  }, [
    zoomLevel,
    timelineWidth,
    clipType,
    timelineDuration,
    currentClip?.trimStart,
    currentClip?.trimEnd,
    currentClip?.startFrame,
    currentClip?.endFrame,
  ]);

  useEffect(() => {
    const newY = timelineY - totalClipHeight;
    setClipPosition({ x: clipX + timelinePadding, y: newY });
    fixedYRef.current = newY;
  }, [timelinePadding, timelineY, timelineId, clipX, totalClipHeight]);

  useEffect(() => {
    if (textRef.current) {
      setTextWidth(textRef.current.width());
    }
  }, [(currentClip as ShapeClipProps)?.shapeType]);

  useEffect(() => {
    if (modelNameRef.current) {
      setModelNameWidth(modelNameRef.current.width());
    }
  }, [
    clipWidth,
    timelineHeight,
    (currentClip as ModelClipProps)?.manifest?.metadata?.name,
  ]);

  useEffect(() => {
    if (!currentClip) return;

    if (clipType === "audio") {
      generateTimelineThumbnailAudio(
        clipType,
        currentClip as AudioClipProps,
        currentClipId,
        mediaInfoRef.current ?? null,
        imageCanvas,
        timelineHeight,
        currentStartFrame,
        currentEndFrame,
        timelineDuration,
        timelineWidth,
        timelinePadding,
        groupRef,
      );
    } else if (clipType === "image") {
      generateTimelineThumbnailImage(
        clipType,
        currentClip as ImageClipProps,
        currentClipId,
        mediaInfoRef.current ?? null,
        imageCanvas,
        timelineHeight,
        thumbnailClipWidth.current,
        maxTimelineWidth,
        applyMask,
        applyFilters,
        groupRef,
        moveClipToEnd,
        resizeSide,
      );
    } else if (clipType === "video") {
      generateTimelineThumbnailVideo(
        clipType,
        currentClip as VideoClipProps,
        currentClipId,
        mediaInfoRef.current ?? null,
        imageCanvas,
        timelineHeight,
        thumbnailClipWidth.current,
        maxTimelineWidth,
        timelineWidth,
        timelineDuration,
        currentStartFrame,
        currentEndFrame,
        overHang,
        applyMask,
        applyFilters,
        groupRef,
        resizeSide,
        exactVideoUpdateTimerRef,
        exactVideoUpdateSeqRef,
        lastExactRequestKeyRef,
        setForceRerenderCounter,
      );
    } else if (clipType === "shape") {
      generateTimelineThumbnailShape(clipType, imageCanvas, groupRef);
    } else if (clipType === "text") {
      generateTimelineThumbnailText(clipType, imageCanvas, groupRef);
    } else if (clipType === "filter") {
      generateTimelineThumbnailFilter(clipType, imageCanvas, groupRef);
    } else if (clipType === "draw") {
      generateTimelineThumbnailDrawing(clipType, imageCanvas, clipRef);
    }
  }, [
    zoomLevel,
    clipWidth,
    clipType,
    currentClip,
    tool,
    resizeSide,
    thumbnailClipWidth,
    maxTimelineWidth,
    timelineDuration[0],
    timelineDuration[1],
    overHang,
    resizeSide,
    forceRerenderCounter,
  ]);

  const calculateFrameFromX = useCallback(
    (xPosition: number) => {
      // Remove padding to get actual timeline position
      const timelineX = xPosition - timelinePadding;
      // Calculate the frame based on the position within the visible timeline
      const [startFrame, endFrame] = timelineDuration;
      const framePosition =
        (timelineX / timelineWidth) * (endFrame - startFrame) + startFrame;
      return Math.round(framePosition);
    },
    [timelinePadding, timelineWidth, timelineDuration],
  );

  const handleDragMove = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      if (assetMode) return;
      const halfStroke = isSelected ? 1.5 : 0;
      // For preprocessor clips, only update X position to prevent vertical drift
      setClipPosition({
        x: e.target.x() - halfStroke,
        y: e.target.y() - halfStroke,
      });
      const stage = e.target.getStage();
      const container = stage?.container();
      let pointerX: number | null = null;
      let pointerY: number | null = null;
      if (container) {
        const rect = container.getBoundingClientRect();
        pointerX = e.evt.clientX - rect.left;
        pointerY = e.evt.clientY - rect.top;
      }

      if (pointerX == null || pointerY == null) {
        return;
      }

      // Notify editor to auto-scroll if near top/bottom edges
      try {
        window.dispatchEvent(
          new CustomEvent("timeline-editor-autoscroll", {
            detail: { y: pointerY },
          }),
        );
      } catch {}

      // We only need the pointer position for proximity checks

      const children = stage?.children[0]?.children || [];
      const timelineState = useClipStore.getState();
      const timelineMap = new Map(
        timelineState.timelines.map((t) => [t.timelineId, t]),
      );

      // Determine proximity to dashed line(s) vs the actual timeline track center
      let nearestDashedId: string | null = null;
      let minDashedDistance = Infinity;
      let nearestTimelineForGhost: string | null = null;
      let minTimelineDistance = Infinity;

      for (const child of children) {
        const childId = child.id();
        if (!childId?.startsWith("dashed-")) continue;

        const timelineKey = childId.replace("dashed-", "").replace("-top", "");
        const timelineData = timelineMap.get(timelineKey);
        const timelineFullHeight =
          timelineData?.timelineHeight ?? timelineHeight + 16;
        const timelineBottom =
          (timelineData?.timelineY ?? 0) + timelineFullHeight + 24;
        const trackHeight = Math.max(1, timelineFullHeight - 16);
        const timelineTop = timelineBottom - trackHeight;
        const timelineCenterY = timelineTop + trackHeight / 2;

        const rect = child.getClientRect();
        const rectLeft = rect.x;
        const rectRight = rect.x + rect.width;
        const pointerInX = pointerX >= rectLeft && pointerX <= rectRight;
        if (!pointerInX) continue;

        // Dashed line is visually a horizontal line within this group's bounds.
        // Use the vertical center of the group as the line Y and check ±15px.
        const lineY = rect.y + rect.height / 2;
        const dashedDistance = Math.abs(pointerY - lineY);
        if (dashedDistance < minDashedDistance) {
          minDashedDistance = dashedDistance;
          nearestDashedId = childId;
        }

        // Also compute distance to the actual timeline track center for comparison
        const timelineDistance = Math.abs(pointerY + scrollY - timelineCenterY);
        if (timelineDistance < minTimelineDistance) {
          minTimelineDistance = timelineDistance;
          nearestTimelineForGhost = timelineKey;
        }
      }

      const dashedWins =
        !!nearestDashedId &&
        minDashedDistance <= 16 &&
        minDashedDistance < minTimelineDistance;
      setHoveredTimelineId(dashedWins ? nearestDashedId : null);

      if (clipRef.current) {
        clipRef.current.x(e.target.x());
        clipRef.current.y(e.target.y());
      }

      // If user is within ±15px of a dashed line and closer to it than the timeline,
      // don't render the ghost overlay (we are "in between" timelines)
      if (dashedWins) {
        setSnapGuideX(null);
        setGhostTimelineId(null);
        setGhostInStage(false);
        return;
      }

      // Update ghost overlay with validated position while dragging
      // Convert the dragged group's stage X into "inner timeline" X (0..timelineWidth)
      // so all snapping/ghost math is independent of the left gutter/padding.
      const rectLeft = Math.round(e.target.x() - halfStroke - timelinePadding);
      const [visibleStartFrame, visibleEndFrame] = timelineDuration;
      const clipLen = Math.max(1, currentEndFrame - currentStartFrame);
      const ghostWidthPx = getClipWidth(0, clipLen, timelineWidth, [
        visibleStartFrame,
        visibleEndFrame,
      ]);
      const desiredLeft = rectLeft;

      // Determine target timeline for ghost purely by vertical proximity to timeline centers,
      // independent of layer scroll or dashed bounds
      {
        const timelinesArr = timelineState.timelines || [];
        let bestId: string | null = null;
        let bestDist = Infinity;
        for (const t of timelinesArr) {
          const fullH = t.timelineHeight ?? timelineHeight + 16;
          // Center Y in content coordinates: top is y + 40, track height is fullH - 16
          const centerY = (t.timelineY ?? 0) + 40 + Math.max(1, fullH - 16) / 2;
          const d = Math.abs(pointerY + scrollY - centerY);
          if (d < bestDist) {
            bestDist = d;
            bestId = t.timelineId!;
          }
        }
        nearestTimelineForGhost = bestId || nearestTimelineForGhost;
      }
      const targetTimelineId = nearestTimelineForGhost || timelineId!;
      const targetTimeline = getTimelineById(targetTimelineId);
      if (!isValidTimelineForClip(targetTimeline!, currentClip!)) return;
      if (!targetTimeline) return;

      // Build occupied intervals on the target timeline excluding this clip (visible-window pixels)
      const otherClips = getClipsForTimeline(targetTimelineId).filter(
        (c) => c.clipId !== currentClipId,
      );
      let maxRight = 0;
      const occupied = otherClips
        .map((c) => {
          const sx = getClipX(
            c.startFrame || 0,
            c.endFrame || 0,
            timelineWidth,
            [visibleStartFrame, visibleEndFrame],
          );
          const sw = getClipWidth(
            c.startFrame || 0,
            c.endFrame || 0,
            timelineWidth,
            [visibleStartFrame, visibleEndFrame],
          );
          const lo = Math.max(0, sx);
          const hi = Math.max(0, sx + sw);
          maxRight = Math.max(maxRight, hi);
          return hi > lo ? ([lo, hi] as [number, number]) : null;
        })
        .filter(Boolean) as [number, number][];
      occupied.sort((a, b) => a[0] - b[0]);
      const merged: [number, number][] = [];
      for (const [lo, hi] of occupied) {
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
      // add the last gap
      if (prev < timelineWidth) gaps.push([prev, Infinity]);

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

      // Cross-timeline edge snapping against other timelines' clip edges
      const SNAP_THRESHOLD_PX = 6;
      let appliedSnap = false;
      let snapStageX: number | null = null;

      if (chosenGap) {
        const [gLo, gHi] = chosenGap;
        const allTimelines = useClipStore.getState().timelines || [];
        const [sStart, sEnd] = useControlsStore.getState().timelineDuration;
        const edgeCandidates: number[] = [];
        for (const t of allTimelines) {
          if (!t?.timelineId) continue;
          if (t.timelineId === targetTimelineId) continue;
          const tClips = getClipsForTimeline(t.timelineId);
          for (const c of tClips) {
            const sx = getClipX(
              c.startFrame || 0,
              c.endFrame || 0,
              timelineWidth,
              [sStart, sEnd],
            );
            const sw = getClipWidth(
              c.startFrame || 0,
              c.endFrame || 0,
              timelineWidth,
              [sStart, sEnd],
            );
            const lo = Math.max(0, Math.min(timelineWidth, sx));
            const hi = Math.max(0, Math.min(timelineWidth, sx + sw));
            if (hi > lo) {
              edgeCandidates.push(lo, hi);
            }
          }
        }
        if (edgeCandidates.length > 0) {
          const leftEdge = validatedLeft;
          const rightEdge = validatedLeft + ghostWidthPx;
          let bestDist = Infinity;
          let bestEdge: number | null = null;
          let bestSide: "left" | "right" | null = null;
          for (const edge of edgeCandidates) {
            const dL = Math.abs(edge - leftEdge);
            const dR = Math.abs(edge - rightEdge);
            if (dL < bestDist) {
              bestDist = dL;
              bestEdge = edge;
              bestSide = "left";
            }
            if (dR < bestDist) {
              bestDist = dR;
              bestEdge = edge;
              bestSide = "right";
            }
          }
          if (bestEdge != null && bestDist <= SNAP_THRESHOLD_PX) {
            let snappedLeft =
              bestSide === "left" ? bestEdge : bestEdge - ghostWidthPx;
            // keep within chosen gap
            if (snappedLeft < gLo) snappedLeft = gLo;
            if (snappedLeft + ghostWidthPx > gHi)
              snappedLeft = gHi - ghostWidthPx;
            const finalLeft = snappedLeft;
            const finalRight = snappedLeft + ghostWidthPx;
            const finalDist =
              bestSide === "left"
                ? Math.abs(bestEdge - finalLeft)
                : Math.abs(bestEdge - finalRight);
            if (finalDist <= SNAP_THRESHOLD_PX) {
              validatedLeft = finalLeft;
              appliedSnap = true;
              snapStageX = timelinePadding + bestEdge;
            }
          }
        }
      }

      setGhostTimelineId(targetTimelineId);
      setGhostInStage(true);
      setGhostX(Math.round(validatedLeft));
      setSnapGuideX(
        appliedSnap && snapStageX != null ? Math.round(snapStageX) : null,
      );
      setGhostStartEndFrame(0, clipLen);
    },
    [
      assetMode,
      clipRef,
      clipWidth,
      timelineHeight,
      isSelected,
      timelinePadding,
      timelineDuration,
      currentEndFrame,
      currentStartFrame,
      timelineWidth,
      getClipsForTimeline,
      timelineId,
      currentClipId,
      setGhostTimelineId,
      setGhostInStage,
      setGhostX,
      setGhostStartEndFrame,
      setHoveredTimelineId,
      scrollY,
      clipType,
      currentClip,
      setSnapGuideX,
    ],
  );

  const handleDragEnd = useCallback(
    (_e: Konva.KonvaEventObject<MouseEvent>) => {
      if (assetMode) return;
      rectRefLeft.current?.moveToTop();
      rectRefRight.current?.moveToTop();
      setIsDragging(false);
      setIsDraggingGlobal(false);
      // Compute validated frames from ghost state
      const [tStart, tEnd] = timelineDuration;
      const stageWidth = timelineWidth;
      const visibleDuration = tEnd - tStart;
      const clipLen = Math.max(1, currentEndFrame - currentStartFrame);

      // Use ghost state target timeline and position, but if hovered dashed line exists,
      // create a new timeline at that location and drop onto it (similar to DnD flow)
      const state = useClipStore.getState();
      const hoveredId = state.hoveredTimelineId;
      let dropTimelineId = state.ghostTimelineId || timelineId!;
      let gX = state.ghostX;

      if (hoveredId) {
        const timelines = state.timelines;
        const hoveredKey = hoveredId.replace("dashed-", "");
        const hoveredIdx = timelines.findIndex(
          (t) => t.timelineId === hoveredKey,
        );
        const hoveredTimeline =
          hoveredIdx !== -1 ? timelines[hoveredIdx] : null;
        const newTimelineId = uuidv4();
        // get the timeline this clip is on
        const timeline = state.getTimelineById(timelineId!);
        const newTimeline = {
          type: getTimelineTypeForClip(currentClip!),
          timelineId: newTimelineId,
          timelineWidth: stageWidth,
          timelineY: (hoveredTimeline?.timelineY ?? 0) + 54,
          timelineHeight: timeline?.timelineHeight ?? 54,
        };
        // check if idx is the same as the timelineId
        const currentIdx = state.timelines.findIndex(
          (t) => t.timelineId === timelineId,
        );

        // Check if trying to create a new timeline in the same position with only one clip
        const clipsOnCurrentTimeline = getClipsForTimeline(timelineId!);
        const isOnlyClipOnTimeline = clipsOnCurrentTimeline.length === 1;

        // Check if creating timeline at same position or adjacent position would result in no-op
        // hoveredIdx === currentIdx: inserting at same position (pushes current down, then deletes)
        // hoveredIdx === currentIdx + 1: inserting right below (after deletion, ends up at same position)
        const wouldBeNoOp =
          isOnlyClipOnTimeline &&
          (hoveredIdx === currentIdx || hoveredIdx === currentIdx - 1);

        if (wouldBeNoOp) {
          // Snap back to original position - treat as no-op
          setClipPosition({
            x: clipX + timelinePadding,
            y: timelineY - totalClipHeight,
          });
          setHoveredTimelineId(null);
          setGhostInStage(false);
          setGhostTimelineId(null);
          setGhostStartEndFrame(0, 0);
          setGhostX(0);
          setDraggingClipId(null);
          setSnapGuideX(null);
          return;
        }

        addTimeline(newTimeline, hoveredIdx);
        dropTimelineId = newTimelineId;
        // When creating a new timeline via dashed hover, place based on the clip's left edge
        // Use the dragged group's X position rather than the pointer position
        const groupLeftX = Math.max(
          0,
          Math.min(stageWidth, _e.target.x() - timelinePadding),
        );
        gX = Math.round(groupLeftX);
      }

      setHoveredTimelineId(null);
      let startFrame = Math.round(tStart + (gX / stageWidth) * visibleDuration);
      let newTEnd = tEnd;
      let endFrame = startFrame + clipLen;

      // Validate against overlaps on target timeline (in frame units), excluding this clip if present
      const existing = getClipsForTimeline(dropTimelineId)
        .filter((c) => c.clipId !== currentClipId)
        .map((c) => ({ lo: c.startFrame || 0, hi: c.endFrame || 0 }))
        .filter((iv) => iv.hi > iv.lo)
        .sort((a, b) => a.lo - b.lo);
      const merged: { lo: number; hi: number }[] = [];
      for (const iv of existing) {
        if (merged.length === 0) merged.push({ ...iv });
        else {
          const last = merged[merged.length - 1];
          if (iv.lo <= last.hi) last.hi = Math.max(last.hi, iv.hi);
          else merged.push({ ...iv });
        }
      }

      const overlapsExisting = existing.some(
        (iv) => startFrame + clipLen > iv.lo && startFrame < iv.hi,
      );

      if (!overlapsExisting) {
        // If no overlap, allow placement even if it's outside original window
        const startFrameClamped = Math.max(0, startFrame);
        const endFrameClamped = startFrameClamped + clipLen;

        if (dropTimelineId === timelineId) {
          setClipPosition({
            x: gX + timelinePadding,
            y: timelineY - totalClipHeight,
          });
        }

        updateClip(clipId, {
          timelineId: dropTimelineId,
          startFrame: startFrameClamped,
          endFrame: endFrameClamped,
        });

        // Conditionally restore window back to original length anchored at new clip start
        restoreWindowIfChanged(startFrameClamped);

        // Clear ghost state
        setGhostInStage(false);
        setGhostTimelineId(null);
        setGhostStartEndFrame(0, 0);
        setGhostX(0);
        setDraggingClipId(null);
        setSnapGuideX(null);
        return;
      }

      // Clamp into visible window for gap-based placement
      startFrame = Math.max(tStart, Math.min(newTEnd - clipLen, startFrame));

      const gaps: { lo: number; hi: number }[] = [];
      let prev = tStart;
      for (const iv of merged) {
        const lo = Math.max(tStart, iv.lo);
        const hi = Math.min(tEnd, iv.hi);
        if (lo > prev) gaps.push({ lo: prev, hi: lo });
        prev = Math.max(prev, hi);
      }
      if (prev < tEnd) gaps.push({ lo: prev, hi: tEnd });
      const validGaps = gaps.filter((g) => g.hi - g.lo >= clipLen);
      let chosen =
        validGaps.find(
          (g) => startFrame >= g.lo && startFrame + clipLen <= g.hi,
        ) || null;
      if (!chosen && validGaps.length > 0) {
        const desiredCenter = startFrame + clipLen / 2;
        chosen = validGaps.reduce((best, g) => {
          const gCenter = (g.lo + g.hi) / 2;
          const bCenter = (best.lo + best.hi) / 2;
          return Math.abs(desiredCenter - gCenter) <
            Math.abs(desiredCenter - bCenter)
            ? g
            : best;
        });
      }

      if (chosen) {
        startFrame = Math.min(
          Math.max(startFrame, chosen.lo),
          chosen.hi - clipLen,
        );
        endFrame = startFrame + clipLen;
      }

      if (dropTimelineId === timelineId) {
        setClipPosition({
          x: gX + timelinePadding,
          y: timelineY - totalClipHeight,
        });
      }
      updateClip(clipId, { timelineId: dropTimelineId, startFrame, endFrame });

      // Conditionally restore window back to original length anchored at new clip start
      restoreWindowIfChanged(startFrame);

      // Clear ghost state
      setGhostInStage(false);
      setGhostTimelineId(null);
      setGhostStartEndFrame(0, 0);
      setGhostX(0);
      setDraggingClipId(null);
      setSnapGuideX(null);
    },
    [
      assetMode,
      timelineWidth,
      timelineDuration,
      currentEndFrame,
      currentStartFrame,
      getClipsForTimeline,
      timelineId,
      currentClipId,
      updateClip,
      setGhostInStage,
      setGhostTimelineId,
      setGhostStartEndFrame,
      setGhostX,
      setDraggingClipId,
      timelinePadding,
      clipId,
      restoreWindowIfChanged,
      clipType,
      currentClip,
      clipX,
      timelineY,
      totalClipHeight,
      setSnapGuideX,
      setHoveredTimelineId,
      addTimeline,
      getTimelineById,
      setIsDraggingGlobal,
    ],
  );

  useEffect(() => {
    const newY = timelineY - totalClipHeight;
    setClipPosition({ x: clipX + timelinePadding, y: newY });
    fixedYRef.current = newY;
  }, [clipX, timelinePadding, timelineY, totalClipHeight]);

  const handleClick = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      const isShiftClick = e.evt?.shiftKey || false;
      if (assetMode) {
        setSelectedAssetClipId(currentClipId);
      } else {
        ctrlToggleClipSelection(currentClipId, isShiftClick);
      }
      setSelectedPreprocessorId(null);
    },
    [
      assetMode,
      currentClipId,
      ctrlToggleClipSelection,
      setSelectedAssetClipId,
      moveClipToEnd,
    ],
  );

  const handleContextMenu = 
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      e.evt.preventDefault();
      const stage = e.target.getStage();
      const container = stage?.container();
      if (!container) return;
      // Select this clip if it's not already part of the selection
      const sel = useControlsStore.getState().selectedClipIds || [];
      if (!sel.includes(currentClipId)) {
        useControlsStore.getState().setSelectedClipIds([currentClipId]);
      }
      // Use global context menu store
      const controls = useControlsStore.getState();
      const clipsState = useClipStore.getState();
      const clip = clipsState.getClipById(currentClipId);
      const isVideo = clip?.type === "video";
      const isSeparated = (() => {
        if (!clip || clip.type !== "video") return false;
        try {
          const asset = clipsState.getAssetById(clip.assetId);
          if (!asset) return false;
          const url = new URL(asset.path);
          if ((url.hash || "").replace("#", "") === "video") return true;
          const audioURL = new URL(asset.path);
          audioURL.hash = "audio";
          return (clipsState.clips || []).some(
            (c) => c.type === "audio" && c.assetId === asset.id,
          );
        } catch {
          return false;
        }
      })();
      const targetIds = (controls.selectedClipIds || []).includes(currentClipId)
        ? controls.selectedClipIds
        : [currentClipId];
      const aiCommands: ContextMenuItem[] = [];

      /**AI Commands, we will have to implement with our AI system.
       * Coming Soon
       *
       */
      if (
        isVideo &&
        clip.masks &&
        clip.masks.length === 0 &&
        clip.preprocessors &&
        clip.preprocessors.length === 0
      ) {
        //aiCommands.push({ id: 'extend', label: 'Extend', action: 'extend', });
        //aiCommands.push({ id: 'stabilize', label: 'Stabilize', action: 'stabilize', });
        //aiCommands.push({ id: 'editVideo', label: 'Edit Video', action: 'editVideo', });
      } else if (
        clip?.type === "image" &&
        clip.masks &&
        clip.masks.length === 0 &&
        clip.preprocessors &&
        clip.preprocessors.length === 0
      ) {
        //aiCommands.push({ id: 'animate', label: 'Animate', action: 'animate', });
        //aiCommands.push({ id: 'editImage', label: 'Edit Image', action: 'editImage', });
      }
      if (
        (clip?.type === "video" || clip?.type === "image") &&
        clip.masks &&
        clip.masks.length > 0
      ) {
        //aiCommands.push({ id: 'inpaint', label: 'Inpaint', action: 'inpaint', });
        //aiCommands.push({ id: 'outpaint', label: 'Outpaint', action: 'outpaint', });
      }
      if (
        (clip?.type === "video" || clip?.type === "image") &&
        clip.preprocessors &&
        clip.preprocessors.length > 0
      ) {
        //aiCommands.push({ id: 'control', label: 'Use as Control', action: 'control', });
      }

      const isGroup = clip?.type === "group";
      const isModelWithSrc =
        clip?.type === "model" && typeof clip.assetId === "string";

      const otherCommands: ContextMenuItem[] = [];

      if (clip?.type === "model" && !isModelRunning && clip?.assetId) {
        otherCommands.push({
          id: "export",
          label: "Export as Media",
          action: "export",
        });
      }

      if (clip?.type === "image") {
        otherCommands.push({
          id: "export",
          label: "Export as Image",
          action: "export",
        });
      }
      if (clip?.type === "video") {
        const fps = useControlsStore.getState().fps;
        const duration = (currentEndFrame - currentStartFrame) / fps;
        if (duration <= 60) {
          otherCommands.push({
            id: "export",
            label: "Export as Video",
            action: "export",
          });
        }
      } else if (clip?.type === "audio" && clip.assetId) {
        otherCommands.push({
          id: "export",
          label: "Export as Audio",
          action: "export",
        });
      }
      // check if any of the selected clips are groups if so, we cannot group it
      const isAnyGroup = (controls.selectedClipIds || []).some((clipId) => {
        const clip = clipsState.getClipById(clipId);
        return clip?.type === "group";
      });
      if (isGroup) {
        otherCommands.push({
          id: "ungroup",
          label: "Ungroup",
          action: "ungroup",
          shortcut: "⌘⇧G",
        });
      } else if (!isAnyGroup) {
        otherCommands.push({
          id: "group",
          label: "Group…",
          action: "group",
          disabled: (controls.selectedClipIds || []).length < 2,
          shortcut: "⌘G",
        });
      }

      const clipActions: ContextMenuItem[] = [];

      if (clipType === "model" && isModelWithSrc) {
        clipActions.push({
          id: "convertToMedia",
          label: "Convert to Media",
          action: "convertToMedia",
          disabled: isModelRunning,
          shortcut: "⌘⇧M",
        });
      } else {
        clipActions.push(
          {
            id: "split",
            label: "Split at Playhead",
            action: "split",
            disabled: isGroup || clipType === "model",
          },
          {
            id: "separate",
            label: "Detach Audio",
            action: "separateAudio",
            disabled: !isVideo || isSeparated,
          },
        );
      }

      useContextMenuStore.getState().openMenu({
        position: { x: e.evt.clientX, y: e.evt.clientY },
        target: {
          type: "clip",
          clipIds: targetIds,
          primaryClipId: currentClipId,
          isVideo: !!isVideo,
        },
        groups: [
          {
            id: "edit",
            items: [
              { id: "copy", label: "Copy", action: "copy", shortcut: "⌘C" },
              { id: "cut", label: "Cut", action: "cut", shortcut: "⌘X" },
              { id: "paste", label: "Paste", action: "paste", shortcut: "⌘V" },
              {
                id: "delete",
                label: "Delete",
                action: "delete",
                shortcut: "Del",
              },
            ],
          },
          {
            id: "ai",
            label: "AI",
            items: [...aiCommands],
          },
          {
            id: "clip-actions",
            items: [...clipActions],
          },
          {
            id: "other",
            items: [...otherCommands],
          },
        ],
      });
    }
   

  useEffect(() => {
    if (isSelected) {
      rectRefLeft.current?.moveToTop();
      rectRefRight.current?.moveToTop();
    }
  }, [isSelected]);

  // fixed width pill; no dynamic measurement needed

  // Ensure no heavy filters/caching remain (fixes blank/slow rendering when zooming in asset mode)
  useEffect(() => {
    const grp = groupRef.current as any;
    if (!grp) return;
    try {
      grp.clearCache();
    } catch {}
    try {
      grp.filters([]);
      grp.blurRadius?.(0);
      grp.brightness?.(0);
    } catch {}
  }, [assetMode, isSelected, clipWidth, timelineHeight]);

  // Handle resizing via global mouse move/up while a handle is being dragged
  useEffect(() => {
    if (assetMode) return;
    if (!resizeSide) return;
    const stage = clipRef.current?.getStage();
    if (!stage) return;

    const handleMouseMove = (e: MouseEvent) => {
      stage.container().style.cursor = "col-resize";
      const rect = stage.container().getBoundingClientRect();
      const stageX = e.clientX - rect.left;
      const newFrame = calculateFrameFromX(stageX);

      if (resizeSide === "right") {
        let targetFrame = newFrame;

        // Check for preprocessor boundaries - prevent resizing past preprocessors
        if (currentClip && currentClip.type === "video") {
          const preprocessors =
            (currentClip as VideoClipProps | ImageClipProps).preprocessors ||
            [];
          if (preprocessors.length > 0) {
            // Find the rightmost preprocessor end position (in absolute frames)
            const rightmostPreprocessorEnd = Math.max(
              ...preprocessors.map(
                (p) => (currentStartFrame || 0) + (p.endFrame ?? 0),
              ),
            );
            // Limit resize to not go below the rightmost preprocessor end
            targetFrame = Math.max(targetFrame, rightmostPreprocessorEnd);
          }
        }

        // Cross-timeline edge snapping when resizing right
        const [tStart, tEnd] = useControlsStore.getState().timelineDuration;
        const stageWidth = timelineWidth;
        const pointerEdgeInnerX = Math.max(
          0,
          Math.min(stageWidth, stageX - timelinePadding),
        );
        const allTimelines = useClipStore.getState().timelines || [];
        const existingEdges: number[] = [];
        for (const t of allTimelines) {
          if (!t?.timelineId) continue;
          const tClips = getClipsForTimeline(t.timelineId).filter(
            (c) => c.clipId !== clipId,
          );
          for (const c of tClips) {
            const sx = getClipX(
              c.startFrame || 0,
              c.endFrame || 0,
              stageWidth,
              [tStart, tEnd],
            );
            const sw = getClipWidth(
              c.startFrame || 0,
              c.endFrame || 0,
              stageWidth,
              [tStart, tEnd],
            );
            const lo = Math.max(0, Math.min(stageWidth, sx));
            const hi = Math.max(0, Math.min(stageWidth, sx + sw));
            if (hi > lo) {
              existingEdges.push(lo, hi);
            }
          }
        }

        const SNAP_THRESHOLD_PX = 6;
        let best: { edge: number; dist: number } | null = null;
        for (const edge of existingEdges) {
          const dist = Math.abs(edge - pointerEdgeInnerX);
          if (!best || dist < best.dist) best = { edge, dist };
        }
        if (best && best.dist <= SNAP_THRESHOLD_PX) {
          const snappedFrame = Math.round(
            tStart + (best.edge / Math.max(1, stageWidth)) * (tEnd - tStart),
          );
          targetFrame = Math.max((currentStartFrame || 0) + 1, snappedFrame);
          setSnapGuideX(Math.round(timelinePadding + best.edge));
        } else {
          setSnapGuideX(null);
        }
        if (targetFrame !== currentEndFrame) {
          // Use the new contiguous resize method - local state will update via useEffect
          resizeClip(clipId, "right", targetFrame);
        }
      } else if (resizeSide === "left") {
        let targetFrame = newFrame;

        // Check for preprocessor boundaries - prevent resizing past preprocessors
        if (currentClip && currentClip.type === "video") {
          const preprocessors =
            (currentClip as VideoClipProps | ImageClipProps).preprocessors ||
            [];
          if (preprocessors.length > 0) {
            // Find the leftmost preprocessor start position (in absolute frames)
            const leftmostPreprocessorStart = Math.min(
              ...preprocessors.map(
                (p) => (currentStartFrame || 0) + (p.startFrame ?? 0),
              ),
            );
            // Limit resize to not go above the leftmost preprocessor start
            targetFrame = Math.min(targetFrame, leftmostPreprocessorStart);
          }
        }

        // Cross-timeline edge snapping when resizing left
        const [tStart, tEnd] = useControlsStore.getState().timelineDuration;
        const stageWidth = timelineWidth;
        const pointerEdgeInnerX = Math.max(
          0,
          Math.min(stageWidth, stageX - timelinePadding),
        );
        const allTimelines = useClipStore.getState().timelines || [];
        const existingEdges: number[] = [];
        for (const t of allTimelines) {
          if (!t?.timelineId) continue;
          const tClips = getClipsForTimeline(t.timelineId).filter(
            (c) => c.clipId !== clipId,
          );
          for (const c of tClips) {
            const sx = getClipX(
              c.startFrame || 0,
              c.endFrame || 0,
              stageWidth,
              [tStart, tEnd],
            );
            const sw = getClipWidth(
              c.startFrame || 0,
              c.endFrame || 0,
              stageWidth,
              [tStart, tEnd],
            );
            const lo = Math.max(0, Math.min(stageWidth, sx));
            const hi = Math.max(0, Math.min(stageWidth, sx + sw));
            if (hi > lo) {
              existingEdges.push(lo, hi);
            }
          }
        }
        const SNAP_THRESHOLD_PX = 6;
        let best: { edge: number; dist: number } | null = null;
        for (const edge of existingEdges) {
          const dist = Math.abs(edge - pointerEdgeInnerX);
          if (!best || dist < best.dist) best = { edge, dist };
        }
        if (best && best.dist <= SNAP_THRESHOLD_PX) {
          const snappedFrame = Math.round(
            tStart + (best.edge / Math.max(1, stageWidth)) * (tEnd - tStart),
          );
          targetFrame = Math.min((currentEndFrame || 0) - 1, snappedFrame);
          setSnapGuideX(Math.round(timelinePadding + best.edge));
        } else {
          setSnapGuideX(null);
        }

        // Prevent resizing below frame 0
        targetFrame = Math.max(0, targetFrame);

        if (targetFrame !== currentStartFrame) {
          // Use the new contiguous resize method - local state will update via useEffect
          resizeClip(clipId, "left", targetFrame);
        }
      }
    };

    const handleMouseUp = () => {
      setResizeSide(null);
      stage.container().style.cursor = "default";
      setSnapGuideX(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [
    assetMode,
    resizeSide,
    currentStartFrame,
    currentEndFrame,
    calculateFrameFromX,
    clipId,
    resizeClip,
    clipType,
    currentClip,
    updateClip,
    timelinePadding,
    timelineWidth,
    getClipsForTimeline,
    setSnapGuideX,
    isModelRunning,
  ]);

  const handleDragStart = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      if (assetMode) return;
      rootGroupRef.current?.moveToTop();
      groupRef.current?.moveToTop();

      setSelectedPreprocessorId(null);

      setIsDragging(true);
      setIsDraggingGlobal(true);
      setTempClipPosition({ x: clipPosition.x, y: clipPosition.y });
      // Store fixed Y position at drag start for preprocessor clips
      fixedYRef.current = clipPosition.y;

      rectRefLeft.current?.moveToBottom();
      rectRefRight.current?.moveToBottom();
      // If this clip isn't already selected, select it (without shift behavior during drag)
      if (!isSelected) {
        ctrlToggleClipSelection(currentClipId, false);
      }

      // Capture initial window at the start of drag
      dragInitialWindowRef.current = [
        ...useControlsStore.getState().timelineDuration,
      ] as [number, number];

      // Initialize ghost overlay for this clip
      const clipLen = Math.max(1, currentEndFrame - currentStartFrame);
      setDraggingClipId(currentClipId);
      setGhostTimelineId(timelineId!);
      setGhostStartEndFrame(0, clipLen);
      setGhostInStage(true);

      const stage = e.target.getStage();
      const pos = stage?.getPointerPosition();
      if (pos) {
        const [visibleStartFrame, visibleEndFrame] =
          useControlsStore.getState().timelineDuration;
        const ghostWidthPx = getClipWidth(0, clipLen, timelineWidth, [
          visibleStartFrame,
          visibleEndFrame,
        ]);
        const pointerLocalX = pos.x - timelinePadding;
        const desiredLeft = pointerLocalX;
        let validated = Math.max(
          0,
          Math.min(timelineWidth - ghostWidthPx, desiredLeft),
        );
        setGhostX(Math.round(validated));
      }
    },
    [
      assetMode,
      currentClipId,
      moveClipToEnd,
      isSelected,
      currentEndFrame,
      currentStartFrame,
      setDraggingClipId,
      setGhostTimelineId,
      setGhostStartEndFrame,
      setGhostInStage,
      setGhostX,
      timelineId,
      timelineWidth,
      timelinePadding,
      clipPosition,
      setIsDraggingGlobal,
      ctrlToggleClipSelection,
    ],
  );

  const handleMouseOver = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      //moveClipToEnd(currentClipId);
      e.target.getStage()!.container().style.cursor = "pointer";
    },
    [isSelected],
  );
  const handleMouseLeave = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      e.target.getStage()!.container().style.cursor = "default";
    },
    [isSelected],
  );

  useEffect(() => {
    (async () => {
      if (clipType === "group") {
        // get the children of the group
        const children = getClipsForGroup(
          (currentClip as GroupClipProps).children,
        ).reverse();
        // Compute per-type counts for badge row
        const counts = {
          video: 0,
          image: 0,
          audio: 0,
          text: 0,
          draw: 0,
          filter: 0,
          shape: 0,
          model: 0,
        } as {
          video: number;
          image: number;
          audio: number;
          text: number;
          draw: number;
          filter: number;
          shape: number;
          model: number;
        };
        for (const ch of children) {
          if (!ch) continue;
          if (ch.type === "video") counts.video++;
          else if (ch.type === "image") counts.image++;
          else if (ch.type === "audio") counts.audio++;
          else if (ch.type === "text") counts.text++;
          else if (ch.type === "draw") counts.draw++;
          else if (ch.type === "filter") counts.filter++;
          else if (ch.type === "shape") counts.shape++;
          else if (ch.type === "model") counts.model++;
        }
        setGroupCounts(counts);
        const childrenToUse = [...children].slice(0, 3);
        const canvases = await Promise.all(
          childrenToUse.map(async (child) => {
            if (
              child?.type === "video" ||
              (child?.type === "image" && child?.assetId)
            ) {
              const asset = getAssetById(child.assetId);
              if (!asset) return null;
              const mediaInfo = getMediaInfoCached(asset.path);
              if (!mediaInfo) return null;
              const masks =
                (child as VideoClipProps | ImageClipProps).masks || [];
              const preprocessors =
                (child as VideoClipProps | ImageClipProps).preprocessors || [];
              const poster = await generatePosterCanvas(
                asset.path,
                undefined,
                undefined,
                { mediaInfo, masks, preprocessors },
              );
              if (!poster) return null;
              return poster;
            } else if (child?.type === "audio" && child?.assetId) {
              const asset = getAssetById(child.assetId);
              if (!asset) return null;
              const mediaInfo = getMediaInfoCached(asset.path);
              if (!mediaInfo) return null;
              const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
              const cssWidth = 64;
              const cssHeight = Math.round((cssWidth * 9) / 16);
              const width = cssWidth * dpr;
              const height = cssHeight * dpr;
              // make the height and width small like max and use that ratio to scale the width and height
              const waveform = await generateAudioWaveformCanvas(
                asset.path,
                width,
                height,
                { color: "#7791C4", mediaInfo: mediaInfo },
              );
              if (!waveform) return null;
              return waveform;
            } else if (
              child?.type === "text" ||
              child?.type === "draw" ||
              child?.type === "filter" ||
              child?.type === "shape" ||
              child?.type === "model"
            ) {
              const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
              const cssWidth = timelineWidth || 240;
              const cssHeight = Math.round((cssWidth * 9) / 16);
              const width = cssWidth * dpr;
              const height = cssHeight * dpr;
              const canvas = document.createElement("canvas");
              canvas.width = Math.max(1, width);
              canvas.height = Math.max(1, height);
              const ctx = canvas.getContext("2d");
              if (!ctx) return null;
              // Set background color based on clip type (matching timeline thumbnails)
              let bg = "#E3E3E3";
              if (child.type === "draw") bg = "#9B59B6";
              if (child.type === "filter") bg = "#00BFFF";
              if (child.type === "shape") bg = "#894c30";
              if (child.type === "model") bg = "#6247aa";
              ctx.fillStyle = bg;
              ctx.fillRect(0, 0, canvas.width, canvas.height);
              // Prepare the icon SVG
              const iconSize = Math.floor(
                Math.min(canvas.width, canvas.height) * 0.35,
              );
              let iconSvg = "";
              if (child.type === "text") {
                iconSvg = renderToStaticMarkup(
                  React.createElement(RxTextIcon, {
                    size: iconSize,
                    color: "#222124",
                  }),
                );
              } else if (child.type === "draw") {
                iconSvg = renderToStaticMarkup(
                  React.createElement(MdOutlineDrawIcon, {
                    size: iconSize,
                    color: "#FFFFFF",
                  }),
                );
              } else if (child.type === "filter") {
                iconSvg = renderToStaticMarkup(
                  React.createElement(MdFilterIcon, {
                    size: iconSize,
                    color: "#FFFFFF",
                  }),
                );
              } else if (child.type === "shape") {
                iconSvg = renderToStaticMarkup(
                  React.createElement(LuShapeIcon, {
                    size: iconSize,
                    color: "#FFFFFF",
                  }),
                );
              } else if (child.type === "model") {
                iconSvg = renderToStaticMarkup(
                  React.createElement(LuBoxIcon, {
                    size: iconSize,
                    color: "#FFFFFF",
                  }),
                );
              }
              if (iconSvg) {
                const img = new (window as any).Image() as HTMLImageElement;
                img.crossOrigin = "anonymous";
                // Ensure an SVG wrapper if not present
                const svgWrapped = iconSvg.startsWith("<svg")
                  ? iconSvg
                  : `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${iconSize} ${iconSize}">${iconSvg}</svg>`;
                img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgWrapped)}`;
                await new Promise<void>((resolve) => {
                  img.onload = () => {
                    const x = Math.floor((canvas.width - iconSize) / 2);
                    const y = Math.floor((canvas.height - iconSize) / 2);
                    ctx.drawImage(img, x, y, iconSize, iconSize);
                    resolve();
                  };
                  img.onerror = () => resolve();
                });
              }
              return canvas;
            }
            return null;
          }),
        );
        setGroupedCanvases(
          canvases.reverse().filter((c) => c !== null) as HTMLCanvasElement[],
        );
      }
    })();
  }, [currentClip, getClipById, clipType]);

  // (removed) shimmer animation for model clip background gradient

  return (
    <Group ref={rootGroupRef}>
      <Group
        onClick={handleClick}
        draggable={!assetMode && resizeSide === null}
        onDragEnd={handleDragEnd}
        onDragMove={handleDragMove}
        onDragStart={handleDragStart}
        onContextMenu={handleContextMenu}
        x={isDragging ? tempClipPosition.x : clipPosition.x}
        y={isDragging ? tempClipPosition.y : clipPosition.y}
        width={clipWidth}
        height={timelineHeight}
        clipX={0}
        clipY={0}
        clipWidth={clipWidth}
        clipHeight={timelineHeight}
        clipFunc={(ctx) => {
          const w = Math.max(1, clipWidth);
          const h = Math.max(1, timelineHeight);
          const rRaw = Number(cornerRadius || 0);
          const r = Math.max(0, Math.min(rRaw, Math.min(w, h) / 2));
          ctx.beginPath();
          ctx.moveTo(r, 0);
          ctx.lineTo(w - r, 0);
          ctx.quadraticCurveTo(w, 0, w, r);
          ctx.lineTo(w, h - r);
          ctx.quadraticCurveTo(w, h, w - r, h);
          ctx.lineTo(r, h);
          ctx.quadraticCurveTo(0, h, 0, h - r);
          ctx.lineTo(0, r);
          ctx.quadraticCurveTo(0, 0, r, 0);
          ctx.closePath();
        }}
        onMouseOver={handleMouseOver}
        onMouseLeave={handleMouseLeave}
      >
        <Group ref={groupRef} width={clipWidth} height={timelineHeight}>
          {clipType === "group" ? (
            <Group>
              <Rect
                x={0}
                y={0}
                width={clipWidth}
                height={timelineHeight}
                cornerRadius={safeCornerRadius}
                fillLinearGradientStartPoint={{ x: 0, y: 0 }}
                fillLinearGradientEndPoint={{ x: clipWidth, y: 0 }}
                fillLinearGradientColorStops={[0, "#AE81CE", 1, "#6A5ACD"]}
                opacity={0.9}
              />
              {/* Stacked preview cards */}
              {groupedCanvases && groupedCanvases.length > 0 && (
                <Group>
                  {(() => {
                    const configs = [
                      {
                        rotation: 16,
                        dx: 12,
                        dy: 6,
                        opacity: 0.75,
                        scale: 0.8,
                      }, // back
                      { rotation: 8, dx: 6, dy: 3, opacity: 0.9, scale: 0.9 }, // middle
                      { rotation: 0, dx: 0, dy: 0, opacity: 1, scale: 1.0 }, // front
                    ];
                    const count = Math.min(3, groupedCanvases.length);
                    const startIdx = configs.length - count;
                    const used = configs.slice(startIdx);
                    const baseX = 10 + groupCardWidth / 2;
                    const baseY = timelineHeight / 2;
                    // Render back-to-front with object-fit: cover
                    return used.map((cfg, i) => {
                      const canvas = groupedCanvases[i];
                      const iw = Math.max(
                        1,
                        (canvas as any).width ||
                          (canvas as any).naturalWidth ||
                          1,
                      );
                      const ih = Math.max(
                        1,
                        (canvas as any).height ||
                          (canvas as any).naturalHeight ||
                          1,
                      );
                      const cardW = Math.max(
                        1,
                        Math.round(groupCardWidth * (cfg as any).scale),
                      );
                      const cardH = Math.max(
                        1,
                        Math.round(groupCardHeight * (cfg as any).scale),
                      );
                      const targetRatio = Math.max(
                        0.0001,
                        cardW / Math.max(1, cardH),
                      );
                      const sourceRatio = iw / ih;
                      let cropX = 0,
                        cropY = 0,
                        cropW = iw,
                        cropH = ih;
                      if (sourceRatio > targetRatio) {
                        // source is wider: crop left/right
                        cropW = Math.max(1, Math.round(ih * targetRatio));
                        cropX = Math.max(0, Math.round((iw - cropW) / 2));
                        cropY = 0;
                        cropH = ih;
                      } else {
                        // source is taller: crop top/bottom
                        cropH = Math.max(1, Math.round(iw / targetRatio));
                        cropY = Math.max(0, Math.round((ih - cropH) / 2));
                        cropX = 0;
                        cropW = iw;
                      }
                      return (
                        <Image
                          key={`group-card-${i}`}
                          image={canvas}
                          x={baseX + cfg.dx}
                          y={baseY + cfg.dy}
                          width={cardW}
                          height={cardH}
                          fill={"#1A2138"}
                          crop={{
                            x: cropX,
                            y: cropY,
                            width: cropW,
                            height: cropH,
                          }}
                          offsetX={cardW / 2}
                          offsetY={cardH / 2}
                          rotation={cfg.rotation}
                          opacity={cfg.opacity}
                          cornerRadius={1}
                          shadowColor={"#000000"}
                          shadowBlur={8}
                          shadowOpacity={0.18}
                        />
                      );
                    });
                  })()}
                  <Text
                    x={groupCardWidth + 28}
                    y={assetMode ? 12 : 16}
                    text={"Group"}
                    fontSize={assetMode ? 9.5 : 10.5}
                    fontFamily="Poppins"
                    fontStyle="500"
                    fill="white"
                    align="left"
                  />

                  {(() => {
                    const items: { Icon: any; count: number }[] = [
                      { Icon: MdMovieIcon, count: groupCounts.video },
                      { Icon: MdImageIcon, count: groupCounts.image },
                      { Icon: MdAudiotrackIcon, count: groupCounts.audio },
                      { Icon: RxTextIcon, count: groupCounts.text },
                      { Icon: MdOutlineDrawIcon, count: groupCounts.draw },
                      { Icon: MdFilterIcon, count: groupCounts.filter },
                      { Icon: LuShapeIcon, count: groupCounts.shape },
                      { Icon: LuBoxIcon, count: groupCounts.model },
                    ].filter((i) => i.count > 0);

                    const startX = groupCardWidth + 28;
                    const startY = assetMode ? 9 + 18 : 16 + 18; // below label
                    let curX = startX;
                    return items.map((it, idx) => {
                      const Ico = it.Icon;
                      const group = (
                        <Group key={`gstat-${idx}`}>
                          {/* icon */}
                          <Image
                            x={curX}
                            y={startY - 1}
                            width={assetMode ? 10 : 12}
                            height={assetMode ? 10 : 12}
                            image={(() => {
                              const svg = renderToStaticMarkup(
                                React.createElement(Ico, {
                                  size: 12,
                                  color: "#FFFFFF",
                                }),
                              );
                              const img = new (window as any).Image();
                              img.crossOrigin = "anonymous";
                              img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
                              return img as any;
                            })()}
                            opacity={0.85}
                          />
                          {/* count */}
                          <Text
                            x={curX + (assetMode ? 14 : 16)}
                            y={startY - 1}
                            text={`${it.count}`}
                            fontSize={assetMode ? 9.5 : 11}
                            fontFamily="Poppins"
                            fill="rgba(255,255,255,0.82)"
                          />
                        </Group>
                      );
                      curX += assetMode ? 24 : 28; // spacing between icon+count pairs
                      return group;
                    });
                  })()}
                </Group>
              )}
            </Group>
          ) : (
            <>
              {clipType === "model" ? (
                <ModelClip
                  clipWidth={clipWidth}
                  timelineHeight={timelineHeight}
                  cornerRadius={safeCornerRadius}
                  currentClip={currentClip as ModelClipProps}
                  modelUiCounts={modelUiCounts}
                  modelNameRef={modelNameRef}
                  modelNameWidth={modelNameWidth}
                  clipPosition={clipPosition}
                  resizeSide={resizeSide}
                  timelineWidth={timelineWidth}
                  imageWidth={imageWidth}
                  clipId={currentClipId}
                  zoomLevel={zoomLevel}
                  clipType={clipType as "video" | "image"}
                  tool={tool as "move" | "resize" | null}
                  thumbnailClipWidth={thumbnailClipWidth.current}
                  maxTimelineWidth={timelineWidth}
                  timelineDuration={timelineDuration}
                />
              ) : (
                <Image
                  x={imageX}
                  y={0}
                  image={imageCanvas}
                  width={imageWidth}
                  height={timelineHeight}
                  cornerRadius={safeCornerRadius}
                  fill={clipType === "audio" ? "#1A2138" : "#FFFFFF"}
                />
              )}
            </>
          )}
          {/* Lightweight dim overlay for asset selection (replaces slow blur cache) */}
          {assetMode && (
            <Group
              x={htmlLocalX}
              offsetX={1}
              y={0}
              width={htmlVisibleWidth + 2}
              height={timelineHeight}
            >
              <Html>
                <div
                  style={{
                    width: htmlVisibleWidth + 2,
                    height: timelineHeight,
                    borderRadius: cornerRadius,
                  }}
                  onClick={() => {
                    if (isSelected) {
                      setSelectedAssetClipId(null);
                    } else {
                      setSelectedAssetClipId(currentClipId);
                    }
                  }}
                  className={cn(
                    "cursor-pointer w-full h-full bg-brand-background-light/50 backdrop-blur-sm rounded-md z-20 hover:opacity-100 transition-all duration-200 flex items-center justify-center",
                    isSelected ? "opacity-100" : "opacity-0",
                  )}
                >
                  {htmlVisibleWidth < 8 ? null : (
                    <div
                      className={cn(
                        "rounded-full py-1 px-3   flex items-center justify-center font-medium text-[10px] w-fit",

                        htmlVisibleWidth > 64 && isSelected
                          ? "bg-brand-light/10"
                          : "",
                      )}
                    >
                      {htmlVisibleWidth > 64 ? (
                        <>
                          {isSelected
                            ? "Selected"
                            : htmlVisibleWidth > 84
                              ? "Use as Input"
                              : "Use Input"}
                        </>
                      ) : (
                        <>
                          {isSelected ? (
                            <>
                              <LuCheckIcon />
                            </>
                          ) : (
                            <LuPointer />
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              </Html>
            </Group>
          )}
          {clipType === "shape" &&
            (currentClip as ShapeClipProps)?.shapeType && (
              <Group>
                <Rect
                  x={12 - 4}
                  y={timelineHeight / 2}
                  width={textWidth + 8}
                  height={14}
                  cornerRadius={2}
                  fill="rgba(255, 255, 255, 0.0)"
                  offsetY={7.5}
                />
                <Text
                  ref={textRef}
                  x={9}
                  y={timelineHeight / 2}
                  text={
                    ((currentClip as ShapeClipProps)?.shapeType
                      ?.charAt(0)
                      .toUpperCase() ?? "") +
                    ((currentClip as ShapeClipProps)?.shapeType?.slice(1) ?? "")
                  }
                  fontSize={9.5}
                  fontFamily="Poppins"
                  fontStyle="500"
                  fill="white"
                  align="left"
                  verticalAlign="middle"
                  offsetY={5}
                />
              </Group>
            )}
          {clipType === "text" && (currentClip as TextClipProps)?.text && (
            <Group>
              <Rect
                x={12 - 4}
                y={timelineHeight / 2}
                width={textWidth + 8}
                height={14}
                cornerRadius={2}
                fill="rgba(0, 0, 0, 0.0)"
                offsetY={7.5}
              />
              <Text
                ref={textRef}
                x={9}
                y={timelineHeight / 2}
                text={
                  (currentClip as TextClipProps)?.text?.replace("\n", " ") ?? ""
                }
                fontSize={10}
                fontFamily={
                  (currentClip as TextClipProps)?.fontFamily ?? "Poppins"
                }
                fontStyle="500"
                fill="#151517"
                align="left"
                verticalAlign="middle"
                offsetY={5}
              />
            </Group>
          )}
          {clipType === "filter" && (currentClip as FilterClipProps)?.name && (
            <Group>
              <Rect
                x={12 - 4}
                y={timelineHeight / 2}
                width={textWidth + 8}
                height={14}
                cornerRadius={2}
                fill="rgba(0, 0, 0, 0.0)"
                offsetY={7.5}
              />
              <Text
                ref={textRef}
                x={9}
                y={timelineHeight / 2}
                text={(currentClip as FilterClipProps)?.name ?? ""}
                fontSize={9.5}
                fontFamily={"Poppins"}
                fontStyle="500"
                fill="#ffffff"
                align="left"
                verticalAlign="middle"
                offsetY={5}
              />
            </Group>
          )}
          {clipType === "draw" && (
            <Group>
              <Rect
                x={12 - 4}
                y={timelineHeight / 2}
                width={textWidth + 8}
                height={14}
                cornerRadius={2}
                fill="rgba(255, 255, 255, 0.0)"
                offsetY={7.5}
              />
              <Text
                ref={textRef}
                x={9}
                y={timelineHeight / 2}
                text="Drawing"
                fontSize={9.5}
                fontFamily="Poppins"
                fontStyle="500"
                fill="white"
                align="left"
                verticalAlign="middle"
                offsetY={5}
              />
            </Group>
          )}
        </Group>
      </Group>

      {/* Per-clip menu component retained (optional); global menu now handles rendering */}
      {hasPreprocessors &&
        (currentClip?.type === "video" || currentClip?.type === "image") && (
          <Group clipX={clipPosition.x} clipY={clipPosition.y}>
            {(currentClip as VideoClipProps | ImageClipProps).preprocessors.map(
              (preprocessor) => {
                return (
                  <PreprocessorClip
                    assetMode={assetMode}
                    key={preprocessor.id}
                    preprocessor={preprocessor}
                    currentStartFrame={currentStartFrame}
                    currentEndFrame={currentEndFrame}
                    timelineWidth={timelineWidth}
                    clipPosition={clipPosition}
                    timelineHeight={timelineHeight}
                    isDragging={isDragging}
                    clipId={currentClipId}
                    cornerRadius={safeCornerRadius}
                    timelinePadding={timelinePadding}
                  />
                );
              },
            )}
          </Group>
        )}
      {showMaskKeyframes && currentClip?.type === "video" && !assetMode && (
        <MaskKeyframes
          clip={currentClip as VideoClipProps}
          clipPosition={clipPosition}
          clipWidth={clipWidth}
          timelineHeight={timelineHeight}
          isDragging={isDragging}
          currentStartFrame={currentStartFrame}
          currentEndFrame={currentEndFrame}
        />
      )}
      <Rect
        ref={rectRefRight}
        x={
          isDragging
            ? clipPosition.x + clipWidth - 1
            : clipPosition.x + clipWidth - 2.5
        }
        y={isDragging ? clipPosition.y + 1.5 : clipPosition.y}
        width={3}
        visible={
          !assetMode &&
          isSelected &&
          clipType !== "group" &&
          !(clipType === "model" && isModelRunning)
        }
        height={timelineHeight}
        cornerRadius={[0, safeCornerRadius, safeCornerRadius, 0]}
        fill={isSelected ? "#FFFFFF" : "transparent"}
        onMouseOver={(e) => {
          if (isSelected && !(clipType === "model" && isModelRunning)) {
            e.target.getStage()!.container().style.cursor = "col-resize";
          }
        }}
        onMouseDown={(e) => {
          e.cancelBubble = true;
          if (assetMode) return;
          if (clipType === "model" && isModelRunning) return;
          if (!isSelected) {
            ctrlToggleClipSelection(currentClipId, false);
          }
          if (currentClipId) {
            //moveClipToEnd(currentClipId);
          }
          if (assetMode) return;
          setResizeSide("right");
          e.target.getStage()!.container().style.cursor = "col-resize";
        }}
        onMouseLeave={(e) => {
          if (isSelected) {
            e.target.getStage()!.container().style.cursor = "default";
          }
        }}
      />

      <Rect
        ref={rectRefLeft}
        x={isDragging ? clipPosition.x + 1.5 : clipPosition.x}
        y={isDragging ? clipPosition.y + 1.5 : clipPosition.y}
        width={3}
        visible={
          !assetMode &&
          isSelected &&
          clipType !== "group" &&
          !(clipType === "model" && isModelRunning)
        }
        height={timelineHeight}
        cornerRadius={[safeCornerRadius, 0, 0, safeCornerRadius]}
        fill={isSelected ? "#FFFFFF" : "transparent"}
        onMouseOver={(e) => {
          if (isSelected && !(clipType === "model" && isModelRunning)) {
            e.target.getStage()!.container().style.cursor = "col-resize";
          }
        }}
        onMouseDown={(e) => {
          e.cancelBubble = true;
          if (assetMode) return;
          if (clipType === "model" && isModelRunning) return;
          if (!isSelected) {
            ctrlToggleClipSelection(currentClipId, false);
          }
          if (currentClipId) {
            moveClipToEnd(currentClipId);
          }
          if (assetMode) return;
          setResizeSide("left");
          e.target.getStage()!.container().style.cursor = "col-resize";
        }}
        onMouseLeave={(e) => {
          if (isSelected) {
            e.target.getStage()!.container().style.cursor = "default";
          }
        }}
      />
      {isSelected && !assetMode && (
        <Line
          ref={clipRef}
          x={clipPosition.x}
          y={clipPosition.y}
          points={[
            6,
            0,
            clipWidth - 6,
            0,
            clipWidth,
            0,
            clipWidth,
            6,
            clipWidth,
            timelineHeight - 6,
            clipWidth,
            timelineHeight,
            clipWidth - 6,
            timelineHeight,
            6,
            timelineHeight,
            0,
            timelineHeight,
            0,
            timelineHeight - 6,
            0,
            6,
            0,
            0,
          ]}
          stroke={"#FFFFFF"}
          strokeWidth={2.0}
          lineCap="round"
          lineJoin="round"
          listening={false}
          bezier={false}
          closed
        />
      )}
    </Group>
  );
};

export default TimelineClip;
