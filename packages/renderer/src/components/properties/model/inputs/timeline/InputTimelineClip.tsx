import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useClipStore } from "@/lib/clip";
import { generatePosterCanvas } from "@/lib/media/timeline";
import { Image, Group, Rect, Text } from "react-konva";
import Konva from "konva";
import {
  MediaInfo,
  ShapeClipProps,
  TextClipProps,
  TimelineProps,
  VideoClipProps,
  ImageClipProps,
  FilterClipProps,
  MaskClipProps,
  PreprocessorClipType,
  GroupClipProps,
  ModelClipProps,
  AnyClipProps,
  AudioClipProps,
} from "@/lib/types";
import {
  generateAudioWaveformCanvas,
  getMediaInfoCached,
} from "@/lib/media/utils";
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import { useWebGLMask } from "@/components/preview/mask/useWebGLMask";
import { PreprocessorClip } from "@/components/timeline/clips/PreprocessorClip";
import { useViewportStore } from "@/lib/viewport";
import { renderToStaticMarkup } from "react-dom/server";
import { RxText as RxTextIcon } from "react-icons/rx";
import {
  MdOutlineDraw as MdOutlineDrawIcon,
  MdMovie as MdMovieIcon,
  MdImage as MdImageIcon,
  MdAudiotrack as MdAudiotrackIcon,
} from "react-icons/md";
import { LuShapes as LuShapeIcon, LuBox as LuBoxIcon } from "react-icons/lu";
import {
  FaRegFileImage as FaRegFileImageIcon,
  FaRegFileVideo as FaRegFileVideoIcon,
  FaRegFileAudio as FaRegFileAudioIcon,
} from "react-icons/fa6";
import { TbMask as TbMaskIcon } from "react-icons/tb";
import {
  RiImageAiLine as RiImageAiLineIcon,
  RiVideoAiLine as RiVideoAiLineIcon,
} from "react-icons/ri";
import { LuImages as LuImagesIcon } from "react-icons/lu";
import { BiSolidVideos as BiSolidVideosIcon } from "react-icons/bi";
import { useManifestStore } from "@/lib/manifest/store";
import { MdPhotoFilter as MdFilterIcon } from "react-icons/md";
import RotatingCube from "@/components/common/RotatingCube";
import { ManifestDocument } from "@/lib/manifest/api";
import { TbFileTextSpark } from "react-icons/tb";
import {
  generateTimelineThumbnailAudio,
  generateTimelineThumbnailImage,
  generateTimelineThumbnailVideo,
  generateTimelineThumbnailShape,
  generateTimelineThumbnailText,
  generateTimelineThumbnailFilter,
  generateTimelineThumbnailDrawing,
} from "@/components/timeline/clips/thumbnails";
import { useInputControlsStore } from "@/lib/inputControl";


const TimelineClip: React.FC<
  TimelineProps & {
    clip: AnyClipProps;
    cornerRadius?: number;
    selectionMode?: "frame" | "range";
    mode?: "frame" | "range";
    inputId?: string;
    // Optional max duration (in frames) for the selectable range
    maxDuration?: number;
  }
> = ({
  timelineWidth = 0,
  timelineY = 0,
  timelineHeight = 54,
  timelinePadding = 24,
  clip,
  cornerRadius = 1,
  selectionMode,
  mode,
  inputId,
  maxDuration,
}) => {
  const effectiveSelectionMode = selectionMode ?? mode ?? "range";
  // Select only what we need to avoid unnecessary rerenders
  const timelineDuration = useInputControlsStore((s) =>
    s.getTimelineDuration(inputId ?? ""),
  );
  const zoomLevel = useInputControlsStore((s) => s.getZoomLevel(inputId ?? ""));
  const tool = useViewportStore((s) => s.tool);
  const getClipById = useClipStore((s) => s.getClipById);
  const getClipsForGroup = useClipStore((s) => s.getClipsForGroup);
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
  const currentClip = clip;

  const clipType = useMemo(() => currentClip?.type, [currentClip]);

  const { focusFrameByInputId, selectedRangeByInputId } =
    useInputControlsStore();
  const setFocusFrame = useInputControlsStore((s) => s.setFocusFrame);
  const setFocusAnchorRatio = useInputControlsStore(
    (s) => s.setFocusAnchorRatio,
  );
  const selectedRange = selectedRangeByInputId[inputId ?? ""] ?? [0, 1];
  const focusFrame = focusFrameByInputId[inputId ?? ""] ?? 0;
  const setSelectedRange = useInputControlsStore((s) => s.setSelectedRange);
  const { applyMask } = useWebGLMask({
    focusFrame: focusFrame,
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

  const clipSpan = useMemo(
    () => Math.max(1, currentEndFrame - currentStartFrame),
    [currentStartFrame, currentEndFrame],
  );
  const visibleSpan = useMemo(
    () => Math.max(1, timelineDuration[1] - timelineDuration[0]),
    [timelineDuration],
  );
  const clipWidth = useMemo(
    () => Math.max(3, Math.round((clipSpan / visibleSpan) * timelineWidth)),
    [clipSpan, visibleSpan, timelineWidth],
  );

  const clipX = useMemo(() => 0, []);

  // Frame selection overlay (only in frame mode)
  const frameWidthPx = useMemo(() => {
    const span = Math.max(1, timelineDuration[1] - timelineDuration[0]);
    return timelineWidth / span;
  }, [timelineWidth, timelineDuration]);
  const frameWidthSafe = useMemo(
    () => Math.max(1e-6, frameWidthPx),
    [frameWidthPx],
  );

  const visibleStart = timelineDuration[0] ?? 0;
  const frameToX = useCallback(
    (frame: number) => (frame - visibleStart) * frameWidthSafe,
    [visibleStart, frameWidthSafe],
  );

  const constrainedFocusFrame = useMemo(() => {
    // Focus frame in store is clip-local; convert to absolute for overlay
    const absFocus = currentStartFrame + Math.max(0, Math.round(focusFrame));
    const minF = Math.max(0, currentStartFrame);
    const maxF = Math.max(minF, currentEndFrame - 1);
    return Math.max(minF, Math.min(maxF, absFocus));
  }, [focusFrame, currentStartFrame, currentEndFrame]);

  const focusXLocal = useMemo(() => {
    // Position the focus frame relative to the currently visible window
    return frameToX(constrainedFocusFrame);
  }, [constrainedFocusFrame, frameToX]);

  const [rangeStart, rangeEnd] = useMemo<[number, number]>(() => {
    // selectedRange is stored clip-local; convert to absolute for rendering
    const rawStartLocal = Math.round(selectedRange?.[0] ?? 0);
    const rawEndLocal = Math.round(selectedRange?.[1] ?? rawStartLocal + 1);
    const rawStart = currentStartFrame + rawStartLocal;
    const rawEnd = currentStartFrame + rawEndLocal;
    const clampedStart = Math.max(
      currentStartFrame,
      Math.min(currentEndFrame - 1, rawStart),
    );
    const minEnd = clampedStart + 1;
    const clampedEnd = Math.max(minEnd, Math.min(currentEndFrame, rawEnd));
    return [clampedStart, clampedEnd];
  }, [selectedRange, currentStartFrame, currentEndFrame]);

  const rangeSpanFrames = useMemo(
    () => Math.max(1, rangeEnd - rangeStart),
    [rangeStart, rangeEnd],
  );

  const maxSpanFrames = useMemo(() => {
    if (typeof maxDuration !== "number" || maxDuration <= 0) return null;
    return Math.max(1, Math.floor(maxDuration));
  }, [maxDuration]);

  const effectiveSpanFrames = useMemo(
    () =>
      maxSpanFrames ? Math.min(rangeSpanFrames, maxSpanFrames) : rangeSpanFrames,
    [rangeSpanFrames, maxSpanFrames],
  );

  // Base X positions for the absolute range, then clipped to the visible window
  const baseRangeStartX = useMemo(
    () => frameToX(rangeStart),
    [rangeStart, frameToX],
  );
  const baseRangeEndX = useMemo(() => frameToX(rangeEnd), [rangeEnd, frameToX]);

  const rangeStartLocal = useMemo(
    () => Math.max(0, Math.min(timelineWidth, baseRangeStartX)),
    [baseRangeStartX, timelineWidth],
  );

  const rangeEndLocal = useMemo(
    () => Math.max(0, Math.min(timelineWidth, baseRangeEndX)),
    [baseRangeEndX, timelineWidth],
  );

  const rangeWidthPx = useMemo(
    () => Math.max(frameWidthSafe, rangeEndLocal - rangeStartLocal),
    [frameWidthSafe, rangeEndLocal, rangeStartLocal],
  );
  
  const maxRangeStart = useMemo(
    () => Math.max(currentStartFrame, currentEndFrame - effectiveSpanFrames),
    [currentStartFrame, currentEndFrame, effectiveSpanFrames],
  );
  const rangeHandleWidth = useMemo(() => 3, []);

  const commitRange = useCallback(
    (startFrame: number, endFrame: number) => {
      let clampedStart = Math.max(
        currentStartFrame,
        Math.min(currentEndFrame - 1, Math.round(startFrame)),
      );
      const minEnd = clampedStart + 1;
      let clampedEnd = Math.max(
        minEnd,
        Math.min(currentEndFrame, Math.round(endFrame)),
      );

      if (maxSpanFrames) {
        const maxSpan = maxSpanFrames;
        const span = clampedEnd - clampedStart;
        if (span > maxSpan) {
          clampedEnd = clampedStart + maxSpan;
        }
      }

      // Store selection as clip-local
      const localStart = clampedStart - currentStartFrame;
      const localEnd = clampedEnd - currentStartFrame;
      const [existingStart, existingEnd] = useInputControlsStore
        .getState()
        .getSelectedRange(inputId ?? "");
      if (existingStart === localStart && existingEnd === localEnd) {
        // Range already matches; still ensure focus is clamped into this range if needed.
      } else {
        setSelectedRange(localStart, localEnd, inputId ?? "");
      }

      // Ensure focus frame does not jump to range start when moving range.
      // Keep previous focus if it stays inside the new range, otherwise clamp to range edges.
      const prevAbsFocus =
        currentStartFrame + Math.max(0, Math.round(focusFrame));
      let newAbsFocus = prevAbsFocus;

      if (prevAbsFocus < clampedStart) {
        newAbsFocus = clampedStart;
      } else if (prevAbsFocus >= clampedEnd) {
        newAbsFocus = clampedEnd - 1;
      }

      const newLocalFocus = newAbsFocus - currentStartFrame;
      setFocusFrame(newLocalFocus, inputId ?? "");

      const [winStart, winEnd] = timelineDuration;
      const winSpan = Math.max(1, winEnd - winStart);
      const anchor = (newLocalFocus - winStart) / winSpan;
      setFocusAnchorRatio(Math.max(0, Math.min(1, anchor)), inputId ?? "");
    },
    [
      currentStartFrame,
      currentEndFrame,
      focusFrame,
      inputId,
      setSelectedRange,
      setFocusFrame,
      setFocusAnchorRatio,
      timelineDuration,
      maxSpanFrames,
    ],
  );

  // Smooth drag: directly move Konva nodes and throttle store updates
  const rangeRectRef = useRef<Konva.Rect>(null);
  const leftHandleRef = useRef<Konva.Rect>(null);
  const rightHandleRef = useRef<Konva.Rect>(null);
  const scrubberRef = useRef<Konva.Rect>(null);
  const isDraggingRangeRef = useRef(false);

  const updateRangeVisuals = useCallback(
    (startFrame: number, endFrame: number) => {
      // Convert absolute frames to local X, then clip to the visible timeline window
      const rawStartLocal = frameToX(startFrame);
      const rawEndLocal = frameToX(endFrame);

      const visStartX = 0;
      const visEndX = timelineWidth;

      const clippedStart = Math.max(
        visStartX,
        Math.min(visEndX, rawStartLocal),
      );
      const clippedEnd = Math.max(visStartX, Math.min(visEndX, rawEndLocal));

      const widthPx = Math.max(frameWidthSafe, clippedEnd - clippedStart);
      const rect = rangeRectRef.current;
      const left = leftHandleRef.current;
      const right = rightHandleRef.current;
      if (rect) {
        rect.x(clippedStart);
        rect.width(Math.max(0, Math.min(widthPx, visEndX - clippedStart)));
        rect.y(0);
      }
      if (left) {
        left.x(clippedStart - rangeHandleWidth / 2);
        left.y(0);
      }
      if (right) {
        right.x(clippedEnd - rangeHandleWidth / 2);
        right.y(0);
      }
      const layer = rect?.getLayer() || left?.getLayer() || right?.getLayer();
      if (layer) layer.batchDraw();
    },
    [frameToX, frameWidthSafe, rangeHandleWidth, timelineWidth],
  );

  // No vertical movement: selector uses full timeline height at y=0
  const clipRef = useRef<Konva.Line>(null);
  const [resizeSide] = useState<"left" | "right" | null>(null);
  const [imageCanvas] = useState<HTMLCanvasElement>(() =>
    document.createElement("canvas"),
  );
  const mediaInfoRef = useRef<MediaInfo | undefined>(
    getMediaInfoCached((currentClip as VideoClipProps | ImageClipProps).assetId!),
  );
  useEffect(() => {
    if (!(currentClip as VideoClipProps | ImageClipProps)?.assetId) {
      mediaInfoRef.current = undefined;
      return;
    }
    mediaInfoRef.current = getMediaInfoCached((currentClip as VideoClipProps | ImageClipProps).assetId!);
  }, [(currentClip as VideoClipProps | ImageClipProps)?.assetId]);
  const thumbnailClipWidth = useRef<number>(0);
  const maxTimelineWidth = useMemo(
    () => timelineWidth,
    [timelineWidth, timelinePadding],
  );
  const groupRef = useRef<Konva.Group>(null);
  const exactVideoUpdateTimerRef = useRef<number | null>(null);
  const exactVideoUpdateSeqRef = useRef(0);
  const lastExactRequestKeyRef = useRef<string | null>(null);
  const textRef = useRef<Konva.Text>(null);
  const [textWidth, setTextWidth] = useState(0);
  const modelNameRef = useRef<Konva.Text>(null);
  const [modelNameWidth, setModelNameWidth] = useState(0);
  const getAssetById = useClipStore((s) => s.getAssetById);
  // global context menu used instead of local state
  const { applyFilters } = useWebGLFilters();
  const [forceRerenderCounter, setForceRerenderCounter] = useState(0);
  // Manifest data for model clips
  const loadManifest = useManifestStore((s) => s.loadManifest);
  const getLoadedManifest = useManifestStore((s) => s.getLoadedManifest);
  const [modelUiCounts, setModelUiCounts] = useState<Record<
    string,
    number
  > | null>(null);

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

  const moveClipToEnd = useCallback(() => {}, []);

  // (moved) image positioning is computed after clipPosition is defined

  useEffect(() => {
    imageCanvas.width = timelineWidth;
    imageCanvas.height = timelineHeight;
  }, [
    zoomLevel,
    timelineHeight,
    timelineWidth,
    timelinePadding,
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
  }, [currentClip, loadManifest, getLoadedManifest]);

  const [clipPosition, setClipPosition] = useState<{ x: number; y: number }>({
    x: clipX + timelinePadding,
    y: timelineY - totalClipHeight,
  });

  const fixedYRef = useRef(timelineY - totalClipHeight);

  // Width used for the thumbnail image we render inside the clip group.
  const imageWidth = useMemo(
    () => Math.max(3, Math.min(clipWidth, maxTimelineWidth)),
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

  useEffect(() => {
    thumbnailClipWidth.current = timelineWidth; //Math.max(getClipWidth(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), 3);
  }, [zoomLevel, timelineWidth, clipType]);

  useEffect(() => {
    const newY = timelineY - totalClipHeight;
    setClipPosition({ x: clipX + timelinePadding, y: newY });
    fixedYRef.current = newY;
  }, [timelinePadding, timelineY, clipX, totalClipHeight]);

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
    if (!currentClip || !(currentClip as VideoClipProps | ImageClipProps | AudioClipProps)?.assetId) return;

    if (clipType === "audio") {
      generateTimelineThumbnailAudio(
        clipType,
        currentClip as AudioClipProps,
        currentClip.clipId,
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
        currentClip.clipId,
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
        currentClip.clipId,
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
    mediaInfoRef.current,
    maxTimelineWidth,
    timelineDuration[0],
    timelineDuration[1],
    overHang,
    resizeSide,
    forceRerenderCounter,
  ]);

  useEffect(() => {
    const newY = timelineY - totalClipHeight;
    setClipPosition({ x: clipX + timelinePadding, y: newY });
    fixedYRef.current = newY;
  }, [clipX, timelinePadding, timelineY, totalClipHeight]);

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
              (child?.type === "image" && (child as ImageClipProps).assetId)
            ) {
              const asset = getAssetById((child as VideoClipProps | ImageClipProps).assetId);
              const src = asset?.path;
              if (!src) return null;
              const mediaInfo = getMediaInfoCached(src);
              if (!mediaInfo) return null;
              const masks =
                (child as VideoClipProps | ImageClipProps).masks || [];
              const preprocessors =
                (child as VideoClipProps | ImageClipProps).preprocessors || [];
              const poster = await generatePosterCanvas(
                src,
                undefined,
                undefined,
                { mediaInfo, masks, preprocessors },
              );
              if (!poster) return null;
              return poster;
            } else if (child?.type === "audio" && (child as AudioClipProps).assetId) {
              const asset = getAssetById((child as AudioClipProps).assetId);
              const src = asset?.path;
              if (!src) return null;
              const mediaInfo = getMediaInfoCached(src);
              if (!mediaInfo) return null;
              const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
              const cssWidth = 64;
              const cssHeight = Math.round((cssWidth * 9) / 16);
              const width = cssWidth * dpr;
              const height = cssHeight * dpr;
              // make the height and width small like max and use that ratio to scale the width and height
              const waveform = await generateAudioWaveformCanvas(
                src,
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
    <>
      <Group
        x={clipPosition.x}
        y={clipPosition.y}
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
      >
        <Group ref={groupRef} width={clipWidth} height={timelineHeight}>
          {clipType === "group" ? (
            <Group>
              <Rect
                x={0}
                y={0}
                width={clipWidth}
                height={timelineHeight}
                cornerRadius={cornerRadius}
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
                    y={12}
                    text={"Group"}
                    fontSize={9.5}
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
                    const startY = 9 + 18; // below label
                    let curX = startX;
                    return items.map((it, idx) => {
                      const Ico = it.Icon;
                      const group = (
                        <Group key={`gstat-${idx}`}>
                          {/* icon */}
                          <Image
                            x={curX}
                            y={startY - 1}
                            width={10}
                            height={10}
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
                            x={curX + 14}
                            y={startY - 1}
                            text={`${it.count}`}
                            fontSize={9.5}
                            fontFamily="Poppins"
                            fill="rgba(255,255,255,0.82)"
                          />
                        </Group>
                      );
                      curX += 24; // spacing between icon+count pairs
                      return group;
                    });
                  })()}
                </Group>
              )}
            </Group>
          ) : (
            <>
              {clipType === "model" ? (
                <>
                  <Rect
                    x={0}
                    y={0}
                    width={clipWidth}
                    height={timelineHeight}
                    cornerRadius={cornerRadius}
                    fillLinearGradientStartPoint={{ x: 0, y: 0 }}
                    fillLinearGradientEndPoint={{ x: 0, y: timelineHeight }}
                    fillLinearGradientColorStops={[
                      0,
                      "#6F56C6",
                      0.08,
                      "#6A50C0",
                      0.5,
                      "#5A40B2",
                      1,
                      "#4A329E",
                    ]}
                    shadowColor={"#000000"}
                    shadowBlur={8}
                    shadowOffsetY={2}
                    shadowOpacity={0.22}
                  />

                  {(() => {
                    const size = Math.max(
                      10,
                      Math.min(18, Math.floor(timelineHeight * 0.55)),
                    );
                    const cx = Math.floor(size / 2) + 4;
                    const cy = timelineHeight - 14;
                    return (
                      <>
                        <RotatingCube
                          baseColors={[
                            "#ffffff",
                            "#6247AA",
                            "#6247AA",
                            "#6247AA",
                            "#6247AA",
                            "#ffffff",
                          ]}
                          x={cx}
                          y={cy}
                          size={8}
                          opacity={1}
                          stroke="#ffffff"
                          strokeWidth={1}
                          phaseKey={`${timelineDuration[0]}-${timelineDuration[1]}`}
                          listening={false}
                        />
                        <Text
                          ref={modelNameRef}
                          x={size + 7}
                          y={timelineHeight - 19}
                          text={
                            (currentClip as ModelClipProps)?.manifest?.metadata
                              ?.name ?? ""
                          }
                          fontSize={10}
                          fontFamily="Poppins"
                          fontStyle="500"
                          fill="white"
                          align="left"
                        />
                        {(() => {
                          const counts = modelUiCounts || {};
                          const ordered: { Icon: any; count: number }[] = [
                            {
                              Icon: FaRegFileImageIcon,
                              count: counts["image"] || 0,
                            },
                            {
                              Icon: FaRegFileVideoIcon,
                              count: counts["video"] || 0,
                            },
                            {
                              Icon: FaRegFileAudioIcon,
                              count: counts["audio"] || 0,
                            },
                            {
                              Icon: TbFileTextSpark,
                              count: counts["text"] || 0,
                            },
                            {
                              Icon: TbMaskIcon,
                              count:
                                (counts["image+mask"] || 0) +
                                (counts["video+mask"] || 0),
                            },
                            {
                              Icon: RiImageAiLineIcon,
                              count: counts["image+preprocessor"] || 0,
                            },
                            {
                              Icon: RiVideoAiLineIcon,
                              count: counts["video+preprocessor"] || 0,
                            },
                            {
                              Icon: LuImagesIcon,
                              count: counts["image_list"] || 0,
                            },
                            {
                              Icon: BiSolidVideosIcon,
                              count: counts["video_list"] || 0,
                            },
                          ].filter((i) => i.count > 0);
                          if (ordered.length === 0) return null;
                          const iconSlotWidth = 28;
                          const totalIconsWidth =
                            ordered.length * iconSlotWidth;
                          const rightPadding = 0;
                          const modelName =
                            (currentClip as ModelClipProps)?.manifest?.metadata
                              ?.name ?? "";
                          if (modelName && modelNameWidth === 0) return null;
                          // hide counts if there isn't enough space to the right of the model name text
                          const leftOccupied = size + 7 + modelNameWidth + 6; // cube + gap + text + small gap
                          const availableRightWidth = Math.max(
                            0,
                            clipWidth - leftOccupied,
                          );
                          if (availableRightWidth < totalIconsWidth)
                            return null;
                          const startX = Math.max(
                            6,
                            clipWidth - totalIconsWidth - rightPadding,
                          );
                          const startY = timelineHeight - 19;
                          let curX = startX;
                          return ordered.map((it, idx) => {
                            const Ico = it.Icon;
                            const group = (
                              <Group key={`mstat-${idx}`}>
                                <Image
                                  x={curX}
                                  y={startY - 1}
                                  width={12}
                                  height={12}
                                  image={(() => {
                                    const svg = renderToStaticMarkup(
                                      React.createElement(Ico, {
                                        size: 11,
                                        color: "#FFFFFF",
                                      }),
                                    );
                                    const img = new (window as any).Image();
                                    img.crossOrigin = "anonymous";
                                    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
                                    return img as any;
                                  })()}
                                  opacity={1}
                                />
                                <Text
                                  x={curX + 16}
                                  y={startY - 1}
                                  text={`${it.count}`}
                                  fontSize={11}
                                  fontStyle="500"
                                  fontFamily="Poppins"
                                  fill="rgba(255,255,255,0.82)"
                                />
                              </Group>
                            );
                            curX += iconSlotWidth;
                            return group;
                          });
                        })()}
                      </>
                    );
                  })()}
                </>
              ) : (
                <Image
                  x={imageX}
                  y={0}
                  image={imageCanvas}
                  width={imageWidth}
                  height={timelineHeight}
                  cornerRadius={cornerRadius}
                  fill={clipType === "audio" ? "#1A2138" : "#FFFFFF"}
                />
              )}
            </>
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
          <Group
            clipX={clipPosition.x}
            clipY={clipPosition.y}
            clipWidth={clipWidth}
            clipHeight={timelineHeight}
          >
            {(currentClip as VideoClipProps | ImageClipProps).preprocessors.map(
              (preprocessor) => {
                return (
                  <PreprocessorClip
                    inputMode
                    key={preprocessor.id}
                    preprocessor={preprocessor}
                    currentStartFrame={currentStartFrame}
                    currentEndFrame={currentEndFrame}
                    timelineWidth={timelineWidth}
                    clipPosition={clipPosition}
                    timelineHeight={timelineHeight}
                    timelineDuration={timelineDuration}
                    isDragging={false}
                    clipId={currentClip.clipId}
                    cornerRadius={cornerRadius}
                    timelinePadding={timelinePadding}
                  />
                );
              },
            )}
          </Group>
        )}
      {effectiveSelectionMode === "range" && rangeWidthPx > 0 && (
        <Group
          x={clipPosition.x + 1}
          y={clipPosition.y}
          width={timelineWidth}
          height={timelineHeight}
          clipX={0}
          clipY={0}
          //clipWidth={clipWidth}
          //clipHeight={timelineHeight}
        >
          <Rect
            ref={rangeRectRef}
            x={rangeStartLocal}
            y={0}
            width={rangeWidthPx}
            height={Math.max(1, timelineHeight)}
            fill={"rgba(255, 255, 255, 0.35)"}
            stroke={"rgba(255, 255, 255, 0.9)"}
            strokeWidth={3}
            cornerRadius={1}
            draggable
            dragBoundFunc={(pos) => {
              const minX = 0;
              const maxX = Math.max(0, timelineWidth - rangeWidthPx);
              const boundedX = Math.max(minX, Math.min(maxX, pos.x));
              return { x: boundedX, y: 0 };
            }}
            onMouseEnter={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "grab";
            }}
            onMouseLeave={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "default";
            }}
            onDragStart={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "grabbing";
              e.target.y(0);
              isDraggingRangeRef.current = true;
            }}
            onDragMove={(e) => {
              e.target.y(0);
              const rawX = Math.max(
                0,
                Math.min(timelineWidth - rangeWidthPx, e.target.x()),
              );
              const frameOffset = Math.round(rawX / frameWidthSafe);
              const proposedStart = visibleStart + frameOffset;
              const clampedStart = Math.max(
                currentStartFrame,
                Math.min(maxRangeStart, proposedStart),
              );
              const nextStartLocal = frameToX(clampedStart);
              e.target.x(nextStartLocal);
              const span = effectiveSpanFrames;
              const endFrame = clampedStart + span;
              updateRangeVisuals(clampedStart, endFrame);
            }}
            onDragEnd={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "grab";
              e.target.y(0);
              const rawX = Math.max(
                0,
                Math.min(timelineWidth - rangeWidthPx, e.target.x()),
              );
              const frameOffset = Math.round(rawX / frameWidthSafe);
              const proposedStart = visibleStart + frameOffset;
              const clampedStart = Math.max(
                currentStartFrame,
                Math.min(maxRangeStart, proposedStart),
              );
              const finalLocal = frameToX(clampedStart);
              e.target.x(finalLocal);
              const span = effectiveSpanFrames;
              const endFrame = clampedStart + span;
              updateRangeVisuals(clampedStart, endFrame);
              commitRange(clampedStart, endFrame);
              isDraggingRangeRef.current = false;
            }}
          />

          {/* Left handle */}
          <Rect
            ref={leftHandleRef}
            x={rangeStartLocal - rangeHandleWidth / 2}
            y={0}
            width={rangeHandleWidth}
            height={Math.max(1, timelineHeight)}
            fill={"rgba(255, 255, 255, 0.9)"}
            //cornerRadius={[2, 0, 2, 0]}
            draggable
            stroke={"rgba(255, 255, 255, 0.9)"}
            strokeWidth={0}
            dragBoundFunc={(pos) => ({ x: pos.x, y: 0 })}
            onDragStart={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "ew-resize";
              e.target.y(0);
              isDraggingRangeRef.current = true;
            }}
            onMouseEnter={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "ew-resize";
            }}
            onMouseLeave={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "default";
            }}
            onDragMove={(e) => {
              e.target.y(0);
              const minCenter = 0;
              const maxCenter = rangeEndLocal - frameWidthSafe;
              const center = e.target.x() + rangeHandleWidth / 2;
              const boundedCenter = Math.max(
                minCenter,
                Math.min(maxCenter, center),
              );
              const frameOffset = Math.round(boundedCenter / frameWidthSafe);
              const proposedStart = visibleStart + frameOffset;
              const clampedStart = Math.max(
                currentStartFrame,
                Math.min(rangeEnd - 1, proposedStart),
              );
              const nextLocal = frameToX(clampedStart);
              e.target.x(nextLocal - rangeHandleWidth / 2);
              let endFrame = Math.max(clampedStart + 1, rangeEnd);
              if (maxSpanFrames) {
                const maxEnd = clampedStart + maxSpanFrames;
                endFrame = Math.min(endFrame, maxEnd);
              }
              updateRangeVisuals(clampedStart, endFrame);
            }}
            onDragEnd={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "ew-resize";
              e.target.y(0);
              const center = e.target.x() + rangeHandleWidth / 2;
              const minCenter = 0;
              const maxCenter = rangeEndLocal - frameWidthSafe;
              const boundedCenter = Math.max(
                minCenter,
                Math.min(maxCenter, center),
              );
              const frameOffset = Math.round(boundedCenter / frameWidthSafe);
              const proposedStart = visibleStart + frameOffset;
              const clampedStart = Math.max(
                currentStartFrame,
                Math.min(rangeEnd - 1, proposedStart),
              );
              const nextLocal = frameToX(clampedStart);
              e.target.x(nextLocal - rangeHandleWidth / 2);
              let endFrame = Math.max(clampedStart + 1, rangeEnd);
              if (maxSpanFrames) {
                const maxEnd = clampedStart + maxSpanFrames;
                endFrame = Math.min(endFrame, maxEnd);
              }
              updateRangeVisuals(clampedStart, endFrame);
              commitRange(clampedStart, endFrame);
              isDraggingRangeRef.current = false;
            }}
          />

          {/* Right handle */}
          <Rect
            ref={rightHandleRef}
            x={rangeEndLocal - rangeHandleWidth / 2}
            y={0}
            width={rangeHandleWidth}
            height={Math.max(1, timelineHeight)}
            fill={"rgba(255, 255, 255, 0.9)"}
            stroke={"rgba(255, 255, 255, 0.9)"}
            strokeWidth={0}
            //cornerRadius={[0, 2, 0, 2]}
            draggable
            dragBoundFunc={(pos) => ({ x: pos.x, y: 0 })}
            onDragStart={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "ew-resize";
              e.target.y(0);
              isDraggingRangeRef.current = true;
            }}
            onMouseEnter={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "ew-resize";
            }}
            onMouseLeave={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "default";
            }}
            onDragMove={(e) => {
              e.target.y(0);
              const minCenter = rangeStartLocal + frameWidthSafe;
              const maxCenter = timelineWidth;
              const center = e.target.x() + rangeHandleWidth / 2;
              const boundedCenter = Math.max(
                minCenter,
                Math.min(maxCenter, center),
              );
              const frameOffset = Math.round(boundedCenter / frameWidthSafe);
              const proposedEnd = visibleStart + frameOffset;
              const clampedEnd = Math.max(
                rangeStart + 1,
                Math.min(currentEndFrame, proposedEnd),
              );
              let limitedEnd = clampedEnd;
              if (maxSpanFrames) {
                const maxEnd = rangeStart + maxSpanFrames;
                limitedEnd = Math.min(limitedEnd, maxEnd);
              }
              const nextLocal = frameToX(limitedEnd);
              e.target.x(nextLocal - rangeHandleWidth / 2);
              updateRangeVisuals(rangeStart, limitedEnd);
            }}
            onDragEnd={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "ew-resize";
              e.target.y(0);
              const center = e.target.x() + rangeHandleWidth / 2;
              const minCenter = rangeStartLocal + frameWidthSafe;
              const maxCenter = timelineWidth;
              const boundedCenter = Math.max(
                minCenter,
                Math.min(maxCenter, center),
              );
              const frameOffset = Math.round(boundedCenter / frameWidthSafe);
              const proposedEnd = visibleStart + frameOffset;
              const clampedEnd = Math.max(
                rangeStart + 1,
                Math.min(currentEndFrame, proposedEnd),
              );
              let limitedEnd = clampedEnd;
              if (maxSpanFrames) {
                const maxEnd = rangeStart + maxSpanFrames;
                limitedEnd = Math.min(limitedEnd, maxEnd);
              }
              const nextLocal = frameToX(limitedEnd);
              e.target.x(nextLocal - rangeHandleWidth / 2);
              updateRangeVisuals(rangeStart, limitedEnd);
              commitRange(rangeStart, limitedEnd);
              isDraggingRangeRef.current = false;
            }}
          />

          {/* Range scrubber */}
          <Rect
            ref={scrubberRef}
            x={focusXLocal}
            y={0}
            width={5}
            height={Math.max(1, timelineHeight)}
            cornerRadius={5}
            fill={"rgba(255, 255, 255, 1)"}
            stroke={"rgba(255, 255, 255, 1)"}
            strokeWidth={1}
            shadowBlur={4}
            shadowColor={"#666666"}
            shadowOpacity={0.8}
            draggable
            dragBoundFunc={(pos) => {
              const minX = rangeStartLocal;
              // Allow selecting the last frame in range
              const maxFrameIndex = Math.max(0, rangeEnd - 1 - visibleStart);
              const maxX = Math.max(minX, maxFrameIndex * frameWidthSafe);

              const boundedX = Math.max(minX, Math.min(maxX, pos.x));
              return { x: boundedX, y: 0 };
            }}
            onMouseEnter={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "grab";
            }}
            onMouseLeave={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "default";
            }}
            onDragStart={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "grabbing";
              e.target.y(0);
            }}
            onDragMove={(e) => {
              e.target.y(0);
              const minX = rangeStartLocal;
              const maxFrameIndex = Math.max(0, rangeEnd - 1 - visibleStart);
              const maxX = Math.max(minX, maxFrameIndex * frameWidthSafe);
              const localX = Math.max(minX, Math.min(maxX, e.target.x()));

              const frameOffset = Math.round(localX / frameWidthSafe);
              const newFocus = Math.max(
                currentStartFrame,
                Math.min(currentEndFrame - 1, visibleStart + frameOffset),
              );

              setFocusFrame(newFocus - currentStartFrame, inputId ?? "");

              const [winStart, winEnd] = timelineDuration;
              const winSpan = Math.max(1, winEnd - winStart);
              const newLocalFocus = newFocus - currentStartFrame;
              const anchor = (newLocalFocus - winStart) / winSpan;
              setFocusAnchorRatio(Math.max(0, Math.min(1, anchor)), inputId ?? "");
            }}
            onDragEnd={(e) => {
              const container = e.target.getStage()?.container();
              if (container) container.style.cursor = "grab";
              e.target.y(0);
            }}
          />
        </Group>
      )}

      {effectiveSelectionMode === "frame" && (
        <Group
          x={clipPosition.x}
          y={clipPosition.y}
          width={clipWidth}
          height={timelineHeight}
          onMouseOver={(e) => {
            // set cursor to grab
            const container = e.target.getStage()?.container();
            if (container) {
              container.style.cursor = "grab";
            }
          }}
          onMouseOut={(e) => {
            // set cursor to default
            const container = e.target.getStage()?.container();
            if (container) {
              container.style.cursor = "default";
            }
          }}
        >
          {/* Draggable frame selector overlay */}
          <Rect
            x={Math.max(
              0,
              Math.min(clipWidth - Math.max(12, frameWidthPx), focusXLocal),
            )}
            y={0}
            width={Math.max(8, frameWidthPx)}
            height={Math.max(1, timelineHeight)}
            fill={"rgba(255, 255, 255, 0.6)"}
            stroke={"rgba(255, 255, 255, 0.9)"}
            strokeWidth={2}
            draggable
            cornerRadius={1}
            shadowBlur={4}
            shadowColor={"#6247AA"}
            shadowOpacity={0.8}
            dragBoundFunc={(pos) => {
              const rectWidth = Math.max(8, frameWidthPx);
              const minX = 0;
              const maxX = Math.max(0, clipWidth - rectWidth);
              const snapped =
                Math.round(pos.x / Math.max(1e-6, frameWidthPx)) * frameWidthPx;
              return {
                x: Math.max(minX, Math.min(maxX, snapped)),
                y: 0,
              };
            }}
            onDragStart={(e) => {
              e.target.y(0);
              e.target.height(Math.max(1, timelineHeight));
            }}
            onDragMove={(e) => {
              e.target.y(0);
              const rectWidth = Math.max(8, frameWidthPx);
              const localX = Math.max(
                0,
                Math.min(clipWidth - rectWidth, e.target.x()),
              );
              const frameOffset = Math.round(
                localX / Math.max(1e-6, frameWidthPx),
              );
              const newFocus =
                Math.max(
                  currentStartFrame,
                  Math.min(
                    currentEndFrame - 1,
                    currentStartFrame + frameOffset,
                  ),
                ) + timelineDuration[0];
              setFocusFrame(newFocus - currentStartFrame, inputId ?? "");
              const [winStart, winEnd] = timelineDuration;
              const winSpan = Math.max(1, winEnd - winStart);
              const newLocalFocus = newFocus - currentStartFrame;
              const anchor = (newLocalFocus - winStart) / winSpan;
              setFocusAnchorRatio(Math.max(0, Math.min(1, anchor)), inputId ?? "");
            }}
          />
        </Group>
      )}
    </>
  );
};

export default TimelineClip;
