import { MediaInfo, VideoClipProps } from "@/lib/types";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Image, Transformer, Group, Line} from "react-konva";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import { useControlsStore } from "@/lib/control";
import Konva from "konva";
import { useViewportStore } from "@/lib/viewport";
import { DEFAULT_FPS } from "@/lib/settings";
import { useClipStore } from "@/lib/clip";
import { WrappedCanvas } from "mediabunny";
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import { BaseClipApplicator } from "./apply/base";
import _ from "lodash";
import { useWebGLMask } from "../mask/useWebGLMask";
import { useInputControlsStore } from "@/lib/inputControl";
import { useVideoDecoderManager } from "@/lib/media/VideoDecoderManagerContext";
import { useProjectsStore } from "@/lib/projects";
import { generatePosterCanvas } from "@/lib/media/timeline";
import { sanitizeCornerRadius } from "@/lib/konva/sanitizeCornerRadius";
// (prefetch helper removed by request; timeline-driven rendering only)

const calculateIterateRange = (
  currentFrame: number,
  trimStart: number | undefined,
  frameOffset: number,
  speed: number,
  clipFps: number,
  projectFps: number,
  mediaInfo: MediaInfo,
  selectedAssetId: string,
  assetId: string
) => {
    const isUsingPreprocessorSrc = selectedAssetId !== assetId;
    const adjustedCurrentFrame = isUsingPreprocessorSrc
      ? currentFrame - (trimStart || 0)
      : currentFrame;
    const idealStartFrame =
      Math.max(0, adjustedCurrentFrame - frameOffset) * Math.max(0.1, speed);
    const actualStartFrame = Math.round(
      (idealStartFrame / projectFps) * clipFps,
    );
    const totalFrames = Math.max(
      0,
      Math.floor((mediaInfo.duration || 0) * clipFps),
    );
    const startIdx =
      Math.max(0, Math.min(totalFrames, actualStartFrame)) +
      Math.round(((mediaInfo.startFrame || 0) / projectFps) * clipFps);
      
    const targetEndFrame = mediaInfo.endFrame
      ? Math.round(((mediaInfo.endFrame || 0) / projectFps) * clipFps)
      : undefined;

    const startTime = startIdx / clipFps;
    const endTime = targetEndFrame !== undefined 
        ? targetEndFrame / clipFps 
        : (mediaInfo.duration || 0);
        
    return { startTime, endTime, startIdx };
};

const getAspectFitSize = (
  info: MediaInfo | null | undefined,
  rectWidth: number,
  rectHeight: number,
) => {
  const originalWidth = info?.video?.displayWidth || 0;
  const originalHeight = info?.video?.displayHeight || 0;
  if (!originalWidth || !originalHeight || !rectWidth || !rectHeight) {
    return { displayWidth: 0, displayHeight: 0, offsetX: 0, offsetY: 0 };
  }
  const aspectRatio = originalWidth / originalHeight;
  let dw = rectWidth;
  let dh = rectHeight;
  if (rectWidth / rectHeight > aspectRatio) {
    dw = rectHeight * aspectRatio;
  } else {
    dh = rectWidth / aspectRatio;
  }
  const ox = (rectWidth - dw) / 2;
  const oy = (rectHeight - dh) / 2;
  return { displayWidth: dw, displayHeight: dh, offsetX: ox, offsetY: oy };
};


const VideoPreview: React.FC<
  VideoClipProps & {
    framesToPrefetch?: number;
    rectWidth: number;
    rectHeight: number;
    applicators: BaseClipApplicator[];
    overlap: boolean;
    overrideClip?: VideoClipProps;
    inputMode?: boolean;
    inputId?: string;
    focusFrameOverride?: number;
    currentLocalFrameOverride?: number;
    offscreenFast?: boolean;
    /**
     * If true, keep decoders warm and update the backing canvas, but do not
     * render/interact in Konva. This is used to prewarm the next clip segment
     * (e.g. after a split) to avoid a visible flicker on boundary transitions.
     */
    hidden?: boolean;
    /**
     * Optional logical key to scope decoder state so multiple previews of the same
     * asset/clip (e.g. media dialog vs. timeline poster) don't override each other.
     */
    decoderKey?: string;
  }
> = ({
  assetId,
  clipId,
  startFrame = 0,
  framesToPrefetch: _framesToPrefetch = 32,
  rectWidth,
  rectHeight,
  trimStart,
  speed: _speed,
  applicators,
  overlap,
  overrideClip,
  inputMode = false,
  inputId,
  focusFrameOverride,
  currentLocalFrameOverride,
  offscreenFast = false,
  decoderKey,
  hidden = false,
}) => {

  const mediaInfo = useRef<MediaInfo | null>(getMediaInfoCached(assetId) || null);
  // `mediaInfo` is stored in a ref for fast access by decoder callbacks, but ref updates
  // don't trigger React renders. We bump this version whenever `mediaInfo.current` changes
  // so aspect-fit sizing and Konva props update immediately (no "wait until drag" issues).
  const [mediaInfoVersion, setMediaInfoVersion] = useState(0);
  const setMediaInfoAndBump = useCallback((info: MediaInfo | null) => {
    mediaInfo.current = info;
    setMediaInfoVersion((v) => v + 1);
  }, []);
  const decoderManager = useVideoDecoderManager();
  const focusFrameFromControls = useControlsStore((state) => state.focusFrame);
  const focusFrameFromInputs = useInputControlsStore((s) =>
    s.getFocusFrame(inputId ?? ""),
  );
  const getActiveProject = useProjectsStore((s) => s.getActiveProject);
  const useInputScopedControls = inputMode && !!inputId;
  const focusFrame =
    typeof focusFrameOverride === "number"
      ? focusFrameOverride
      : useInputScopedControls
        ? focusFrameFromInputs
        : focusFrameFromControls;
  
  
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const displaySizeRef = useRef<{
    displayWidth: number;
    displayHeight: number;
    offsetX: number;
    offsetY: number;
  }>({ displayWidth: 0, displayHeight: 0, offsetX: 0, offsetY: 0 });
  const [imageSource, setImageSource] = useState<HTMLCanvasElement | null>(
    null,
  );
  const originalFrameRef = useRef<HTMLCanvasElement | null>(null); // Store unfiltered frame
  const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<Konva.Image>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const drawTokenRef = useRef(0);
  const posterRequestRef = useRef(0);
  const lastPosterKeyRef = useRef<string | null>(null);
  const suppressUntilRef = useRef<number>(0);
  const { applyFilters } = useWebGLFilters();
  // Resolve clip early so timing math can reference grouping info
  const clipFromStore = useClipStore((s) =>
    s.getClipById(clipId),
  ) as VideoClipProps;
  const clip = (overrideClip as VideoClipProps) || clipFromStore;
  // In input mode, when a clip is part of a group, offset by the group's start so playback is contiguous
  const groupStartForClip = useMemo(() => {
    const grpId = (clip as any)?.groupId as string | undefined;
    if (!grpId) return 0;
    try {
      const groupClip = useClipStore.getState().getClipById(grpId) as any;
      return groupClip?.startFrame ?? 0;
    } catch {
      return 0;
    }
  }, [clip]);
  const startFrameUsed = useMemo(() => {
    if (!inputMode) return startFrame;
    const s = (clip as any)?.startFrame as number | undefined;
    const hasGroup = Boolean((clip as any)?.groupId);
    if (hasGroup && typeof s === "number") {
      const rel = s - (groupStartForClip || 0);
      return Math.max(0, rel);
    }
    return 0;
  }, [inputMode, startFrame, clip, groupStartForClip]);

  // Mirror `startFrameUsed` semantics for end-frame checks (important in input mode where we
  // normalize non-grouped clips to a 0-based local window).
  const endFrameUsed = useMemo(() => {
    const rawEnd = (clip as any)?.endFrame as number | undefined;
    if (!inputMode) return typeof rawEnd === "number" ? rawEnd : undefined;

    const rawStart = (clip as any)?.startFrame as number | undefined;
    const hasGroup = Boolean((clip as any)?.groupId);

    if (hasGroup && typeof rawEnd === "number") {
      const rel = rawEnd - (groupStartForClip || 0);
      return Math.max(0, rel);
    }

    // Non-grouped input previews use `startFrameUsed = 0`. If the clip provides absolute
    // start/end frames, convert that into a duration window [0..(end-start)].
    if (typeof rawEnd === "number" && typeof rawStart === "number") {
      return Math.max(0, rawEnd - rawStart);
    }

    return typeof rawEnd === "number" ? rawEnd : undefined;
  }, [clip, groupStartForClip, inputMode]);

  // Gate Konva node rendering: keep decode/effects hooks running, but do not mount Konva
  // nodes unless the clip is actually in frame for the current focus frame.
  const isInFrame = useMemo(() => {
    const f = Number(focusFrame);
    if (!Number.isFinite(f)) return true;
    const s = Number(startFrameUsed ?? 0);
    if (!Number.isFinite(s)) return true;
    const e =
      typeof endFrameUsed === "number" && Number.isFinite(endFrameUsed)
        ? endFrameUsed
        : Infinity;
    
    return f >= s && f <= e;
  }, [focusFrame, startFrameUsed, endFrameUsed]);


  const currentFrame = useMemo(
    () => focusFrame - startFrameUsed + (trimStart || 0),
    [focusFrame, startFrameUsed, trimStart],
  );
  const speed = useMemo(() => {
    const s = Number(_speed ?? 1);
    return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
  }, [_speed]);
  const tool = useViewportStore((s) => s.tool);
  const scale = useViewportStore((s) => s.scale);
  const position = useViewportStore((s) => s.position);
  const setClipTransform = useClipStore((s) => s.setClipTransform);
  const clipTransform = overrideClip
    ? overrideClip.transform
    : useClipStore((s) => s.getClipTransform(clipId));

  const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
  const addClipSelection = useControlsStore((s) => s.addClipSelection);
  const clearSelection = useControlsStore((s) => s.clearSelection);
  const { selectedClipIds, isFullscreen, fps: srcFps, isAccurateSeekNeeded } = useControlsStore();
  const isSelected = useMemo(
    () => selectedClipIds.includes(clipId),
    [clipId, selectedClipIds],
  );

  // const setFocusFrame = useControlsStore((s) => s.setFocusFrame);
  const getAssetById = useClipStore((s) => s.getAssetById);
  const lastSelectedAssetIdRef = useRef<string | null>(null);
  const cachedPreprocessorRangeRef = useRef<{
    startFrame: number;
    endFrame: number;
    selectedAssetId: string;
    frameOffset: number;
  } | null>(null);
  const addedTimestampRef = useRef<number | undefined>(undefined); // last timestamp rendered

  const activeDecoderAssetIdRef = useRef<string | null>(null);
  // Use a logical decoder id so multiple clips can share the same underlying
  // asset while keeping independent decoder state and handlers.
  const makeDecoderId = useCallback(
    (id: string) => {
      const logicalClipKey = decoderKey ?? clipId;
      // For input-mode previews (model inputs, media dialogs, etc.), scope the
      // decoder id by inputId so they never override the main timeline's
      // onFrame/onError handlers for the same clip.
      if (inputMode && inputId) {
        return `${id}::${logicalClipKey}::input::${inputId}`;
      }
      // For normal timeline playback, keep the legacy id so it matches the
      // preconfigured decoders from VideoDecoderManagerProvider.
      return `${id}::${logicalClipKey}`;
    },
    [clipId, decoderKey, inputMode, inputId],
  );
  
  const { applyMask } = useWebGLMask({
    focusFrame: focusFrame,
    masks: clip?.masks || [],
    disabled: tool === "mask" && !inputMode,
    clip: clip,
  });

  const { selectedAssetId, frameOffset } = useMemo(() => {
    // Check if we can use the cached result

    // Cache miss - recalculate
    if (
      !_.has(clip, "preprocessors") ||
      !clip.preprocessors ||
      clip.preprocessors.length === 0
    ) {
      cachedPreprocessorRangeRef.current = null;
      addedTimestampRef.current = 0;
      return { selectedAssetId: assetId, frameOffset: 0 };
    }

    if (
      cachedPreprocessorRangeRef.current &&
      currentFrame >= cachedPreprocessorRangeRef.current.startFrame &&
      currentFrame <= cachedPreprocessorRangeRef.current.endFrame
    ) {
      return {
        selectedAssetId: cachedPreprocessorRangeRef.current.selectedAssetId,
        frameOffset: cachedPreprocessorRangeRef.current.frameOffset,
      };
    }

    // go through the preprocessors and find the one that is within the focus frame
    // adjust preprocessor ranges by trimStart to match currentFrame's reference frame
    const cliptrimStart = trimStart || 0;
    for (const preprocessor of clip.preprocessors) {
      if (
        preprocessor.startFrame !== undefined &&
        preprocessor.endFrame !== undefined &&
        // Only apply preprocessor outputs in-place when explicitly requested.
        // When createNewClip is enabled (default), the parent clip should render as-is.
        preprocessor.createNewClip === false &&
        preprocessor.status === "complete" &&
        preprocessor.assetId
      ) {
        const adjustedStartFrame = preprocessor.startFrame + cliptrimStart;
        const adjustedEndFrame = preprocessor.endFrame + cliptrimStart;

        if (
          currentFrame >= adjustedStartFrame &&
          currentFrame <= adjustedEndFrame
        ) {
          const startSec = preprocessor.startFrame / srcFps;
          addedTimestampRef.current = startSec;

          cachedPreprocessorRangeRef.current = {
            startFrame: adjustedStartFrame,
            endFrame: adjustedEndFrame,
            selectedAssetId: preprocessor.assetId,
            frameOffset: preprocessor.startFrame,
          };

          return {
            selectedAssetId: preprocessor.assetId,
            frameOffset: preprocessor.startFrame,
          };
        }
      }
    }

    cachedPreprocessorRangeRef.current = null;
    addedTimestampRef.current = 0;
    return { selectedAssetId: assetId, frameOffset: 0 };
  }, [clip?.preprocessors, assetId, currentFrame, trimStart]);

  const posterPreprocessors = useMemo(() => {
    const preprocessors = clip?.preprocessors ?? [];
    return preprocessors.filter(
      (p) =>
        p?.assetId &&
        p.createNewClip === false &&
        p.status === "complete" &&
        (typeof p.startFrame === "number" || typeof p.endFrame === "number"),
    );
  }, [clip?.preprocessors]);

  const posterMasks = useMemo(() => clip?.masks ?? [], [clip?.masks]);

  // (seekInProgressRef removed; was unused and could cause confusion)

  // Use refs to store current filter values to avoid callback recreation
  const filterParamsRef = useRef({
    brightness: clip?.brightness,
    contrast: clip?.contrast,
    hue: clip?.hue,
    saturation: clip?.saturation,
    blur: clip?.blur,
    sharpness: clip?.sharpness,
    noise: clip?.noise,
    vignette: clip?.vignette,
  });

  // Use ref to store current applicators to avoid callback recreation
  const applicatorsRef = useRef(applicators);

  const toolRef = useRef(tool);
  useEffect(() => {
    toolRef.current = tool;
  }, [tool]);

  const applyFiltersRef = useRef(applyFilters);
  useEffect(() => {
    applyFiltersRef.current = applyFilters;
  }, [applyFilters]);

  const applyMaskRef = useRef(applyMask);
  useEffect(() => {
    applyMaskRef.current = applyMask;
  }, [applyMask]);

  const maskFrameForCurrentFocus = useMemo(() => {
    const speedFactor = Math.max(0.1, speed);
    if (clip) {
      if (inputMode) {
        const local = Math.max(0, focusFrame + (trimStart || 0));
        return Math.max(0, Math.floor(local * speedFactor));
      }
      const isUsingPreprocessorSrc = selectedAssetId !== assetId;
      const baseLocal = Math.max(0, focusFrame - startFrameUsed);
      const derivedLocal = isUsingPreprocessorSrc
        ? Math.max(0, baseLocal - Math.max(0, frameOffset))
        : Math.max(0, baseLocal + (trimStart || 0));
      return Math.max(0, Math.floor(derivedLocal * speedFactor));
    }
    return Math.max(0, Math.floor(Math.max(0, currentFrame) * speedFactor));
  }, [
    clip,
    focusFrame,
    currentFrame,
    inputMode,
    trimStart,
    speed,
    selectedAssetId,
    assetId,
    frameOffset,
    startFrameUsed,
  ]);

  const aspectRatio = useMemo(() => {
    const originalWidth = mediaInfo.current?.video?.displayWidth || 0;
    const originalHeight = mediaInfo.current?.video?.displayHeight || 0;
    if (!originalWidth || !originalHeight) return 16 / 9;
    const aspectRatio = originalWidth / originalHeight;

    return aspectRatio;
  }, [
    mediaInfo.current?.video?.displayWidth,
    mediaInfo.current?.video?.displayHeight,
  ]);

  const groupRef = useRef<Konva.Group>(null);
  const SNAP_THRESHOLD_PX = 4; // pixels at screen scale
  const [guides, setGuides] = useState({
    vCenter: false,
    hCenter: false,
    v25: false,
    v75: false,
    h25: false,
    h75: false,
    left: false,
    right: false,
    top: false,
    bottom: false,
  });
  const [isInteracting, setIsInteracting] = useState(false);
  const [isRotating, setIsRotating] = useState(false);
  const [isTransforming, setIsTransforming] = useState(false);
  const iteratorRef = useRef<AsyncIterable<WrappedCanvas | null> | null>(null);
  const isPlayingFromControls = useControlsStore((s) => s.isPlaying);
  const isPlayingFromInputs = useInputControlsStore((s) =>
    s.getIsPlaying(inputId ?? ""),
  );
  // IMPORTANT:
  // - `isPlaying` must be reactive (derived from store selectors), otherwise playback
  //   can get stuck in "paused" mode and force per-frame seeks.
  // - We still use refs (e.g. focusFrameRef) for fast access inside decoder callbacks.
  const isPlaying = offscreenFast
    ? true
    : useInputScopedControls
      ? isPlayingFromInputs
      : isPlayingFromControls;
  const focusFrameRef = useRef(focusFrame);
  useEffect(() => {
    focusFrameRef.current = focusFrame;
  }, [focusFrame]);
  // Use a ref for isPlaying to avoid triggering seeks when pausing
  // This prevents the frame from jumping when transitioning from play to pause
  const isPlayingRef = useRef(isPlaying);
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);
  // When input playback reaches the end, the store can "rewind" focusFrame back to the
  // range start and immediately resume playing. If `startRendering()` kicked off using
  // the old end-frame before that rewind propagated, the iterator will be out of sync
  // and can appear frozen until the user scrubs (seek).
  //
  // Detect backwards jumps while playing and restart the iterator from the new position.
  const prevFocusFrameWhilePlayingRef = useRef<number | null>(null);
  const fpsFromControls = useControlsStore((s) => s.fps);
  const fpsFromInputs = useInputControlsStore((s) => s.getFps(inputId ?? ""));
  const fps = useInputScopedControls ? fpsFromInputs : fpsFromControls;
  const currentStartFrameRef = useRef<number>(0);
  const lastRenderedFrameRef = useRef<number>(-1);


  // Update refs when values change
  useEffect(() => {
    filterParamsRef.current = {
      brightness: clip?.brightness,
      contrast: clip?.contrast,
      hue: clip?.hue,
      saturation: clip?.saturation,
      blur: clip?.blur,
      sharpness: clip?.sharpness,
      noise: clip?.noise,
      vignette: clip?.vignette,
    };
    applicatorsRef.current = applicators;
  }, [
    clip?.brightness,
    clip?.contrast,
    clip?.hue,
    clip?.saturation,
    clip?.blur,
    clip?.sharpness,
    clip?.noise,
    clip?.vignette,
    applicators,
    applicators.length,
  ]);

  const updateGuidesAndMaybeSnap = useCallback(
    (opts: { snap: boolean }) => {
      if (isRotating) return; // disable guides/snapping while rotating
      const node = imageRef.current;
      const group = groupRef.current;
      if (!node || !group) return;
      const thresholdLocal = SNAP_THRESHOLD_PX / Math.max(0.0001, scale);
      const client = node.getClientRect({
        skipShadow: true,
        skipStroke: true,
        relativeTo: group as any,
      });
      const centerX = client.x + client.width / 2;
      const centerY = client.y + client.height / 2;
      const dxToVCenter = rectWidth / 2 - centerX;
      const dyToHCenter = rectHeight / 2 - centerY;
      const dxToV25 = rectWidth * 0.25 - centerX;
      const dxToV75 = rectWidth * 0.75 - centerX;
      const dyToH25 = rectHeight * 0.25 - centerY;
      const dyToH75 = rectHeight * 0.75 - centerY;
      const distVCenter = Math.abs(dxToVCenter);
      const distHCenter = Math.abs(dyToHCenter);
      const distV25 = Math.abs(dxToV25);
      const distV75 = Math.abs(dxToV75);
      const distH25 = Math.abs(dyToH25);
      const distH75 = Math.abs(dyToH75);
      const distLeft = Math.abs(client.x - 0);
      const distRight = Math.abs(client.x + client.width - rectWidth);
      const distTop = Math.abs(client.y - 0);
      const distBottom = Math.abs(client.y + client.height - rectHeight);

      const nextGuides = {
        vCenter: distVCenter <= thresholdLocal,
        hCenter: distHCenter <= thresholdLocal,
        v25: distV25 <= thresholdLocal,
        v75: distV75 <= thresholdLocal,
        h25: distH25 <= thresholdLocal,
        h75: distH75 <= thresholdLocal,
        left: distLeft <= thresholdLocal,
        right: distRight <= thresholdLocal,
        top: distTop <= thresholdLocal,
        bottom: distBottom <= thresholdLocal,
      };
      setGuides(nextGuides);

      if (opts.snap) {
        let deltaX = 0;
        let deltaY = 0;
        if (nextGuides.vCenter) {
          deltaX += dxToVCenter;
        } else if (nextGuides.v25) {
          deltaX += dxToV25;
        } else if (nextGuides.v75) {
          deltaX += dxToV75;
        } else if (nextGuides.left) {
          deltaX += -client.x;
        } else if (nextGuides.right) {
          deltaX += rectWidth - (client.x + client.width);
        }
        if (nextGuides.hCenter) {
          deltaY += dyToHCenter;
        } else if (nextGuides.h25) {
          deltaY += dyToH25;
        } else if (nextGuides.h75) {
          deltaY += dyToH75;
        } else if (nextGuides.top) {
          deltaY += -client.y;
        } else if (nextGuides.bottom) {
          deltaY += rectHeight - (client.y + client.height);
        }
        if (deltaX !== 0 || deltaY !== 0) {
          node.x(node.x() + deltaX);
          node.y(node.y() + deltaY);
          setClipTransform(clipId, { x: node.x(), y: node.y() });
        }
      }
    },
    [rectWidth, rectHeight, scale, setClipTransform, clipId, isRotating],
  );

  const transformerBoundBoxFunc = useCallback(
    (_oldBox: any, newBox: any) => {
      if (isRotating) return newBox; // do not snap bounds while rotating
      // Convert absolute newBox to local coordinates of the content group (rect space)
      const invScale = 1 / Math.max(0.0001, scale);
      const local = {
        x: (newBox.x - position.x) * invScale,
        y: (newBox.y - position.y) * invScale,
        width: newBox.width * invScale,
        height: newBox.height * invScale,
      };
      const thresholdLocal = SNAP_THRESHOLD_PX * invScale;

      const left = local.x;
      const right = local.x + local.width;
      const top = local.y;
      const bottom = local.y + local.height;
      const v25 = rectWidth * 0.25;
      const v75 = rectWidth * 0.75;
      const h25 = rectHeight * 0.25;
      const h75 = rectHeight * 0.75;

      // Snap left edge to 0, 25%, 75%
      if (Math.abs(left - 0) <= thresholdLocal) {
        local.x = 0;
        local.width = right - local.x;
      } else if (Math.abs(left - v25) <= thresholdLocal) {
        local.x = v25;
        local.width = right - local.x;
      } else if (Math.abs(left - v75) <= thresholdLocal) {
        local.x = v75;
        local.width = right - local.x;
      }
      // Snap right edge to rectWidth, 75%, 25%
      if (Math.abs(rectWidth - right) <= thresholdLocal) {
        local.width = rectWidth - local.x;
      } else if (Math.abs(v75 - right) <= thresholdLocal) {
        local.width = v75 - local.x;
      } else if (Math.abs(v25 - right) <= thresholdLocal) {
        local.width = v25 - local.x;
      }
      // Snap top edge to 0, 25%, 75%
      if (Math.abs(top - 0) <= thresholdLocal) {
        local.y = 0;
        local.height = bottom - local.y;
      } else if (Math.abs(top - h25) <= thresholdLocal) {
        local.y = h25;
        local.height = bottom - local.y;
      } else if (Math.abs(top - h75) <= thresholdLocal) {
        local.y = h75;
        local.height = bottom - local.y;
      }
      // Snap bottom edge to rectHeight, 75%, 25%
      if (Math.abs(rectHeight - bottom) <= thresholdLocal) {
        local.height = rectHeight - local.y;
      } else if (Math.abs(h75 - bottom) <= thresholdLocal) {
        local.height = h75 - local.y;
      } else if (Math.abs(h25 - bottom) <= thresholdLocal) {
        local.height = h25 - local.y;
      }

      // Convert back to absolute space
      let adjusted = {
        ...newBox,
        x: position.x + local.x * scale,
        y: position.y + local.y * scale,
        width: local.width * scale,
        height: local.height * scale,
      };

      // Prevent negative or zero sizes in absolute space just in case
      const MIN_SIZE_ABS = 1e-3;
      if (adjusted.width < MIN_SIZE_ABS) adjusted.width = MIN_SIZE_ABS;
      if (adjusted.height < MIN_SIZE_ABS) adjusted.height = MIN_SIZE_ABS;

      return adjusted;
    },
    [
      rectWidth,
      rectHeight,
      scale,
      position.x,
      position.y,
      isRotating,
      aspectRatio,
    ],
  );

  // Create canvas once and expose to Konva Image via state so initial render receives it
  useEffect(() => {
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
      setImageSource(canvasRef.current);
    } else {
      setImageSource(canvasRef.current);
    }
    return () => {
      canvasRef.current = null;
      originalFrameRef.current = null;
      processingCanvasRef.current = null;
      setImageSource(null);
    };
  }, []);

  useEffect(() => {
    if (!isSelected) return;
    const tr = transformerRef.current;
    const img = imageRef.current;
    if (!tr || !img) return;
    const raf = requestAnimationFrame(() => {
      tr.nodes([img]);
      if (typeof (tr as any).forceUpdate === "function") {
        (tr as any).forceUpdate();
      }
      tr.getLayer()?.batchDraw?.();
    });
    return () => cancelAnimationFrame(raf);
  }, [isSelected]);

  useEffect(() => {
    if (lastSelectedAssetIdRef.current === selectedAssetId) return;
    lastSelectedAssetIdRef.current = selectedAssetId;
    // Force redraw on source switch: reset last rendered frame and clear cached original frame
    lastRenderedFrameRef.current = -1;
    lastPosterKeyRef.current = null;
    originalFrameRef.current = null;
    processingCanvasRef.current = null;
    // @ts-ignore
    iteratorRef.current?.return?.();
    iteratorRef.current = null;
    let info = getMediaInfoCached(selectedAssetId);
    if (!info) {
      return;
    } else {
      setMediaInfoAndBump(info);
      // Update the "current" aspect-fit size for drawWrappedCanvas immediately so the
      // very first frame of the new asset can't render into a stale-sized canvas.
      displaySizeRef.current = getAspectFitSize(info, rectWidth, rectHeight);
      // Have cached info; force immediate redraw
      lastRenderedFrameRef.current = -1;
    }
  }, [selectedAssetId, rectWidth, rectHeight, setMediaInfoAndBump]);

  // Compute aspect-fit display size and offsets within the preview rect
  const { displayWidth, displayHeight, offsetX, offsetY } = useMemo(() => {
    return getAspectFitSize(mediaInfo.current, rectWidth, rectHeight);
  }, [
    mediaInfoVersion,
    mediaInfo.current?.video?.displayWidth,
    mediaInfo.current?.video?.displayHeight,
    rectWidth,
    rectHeight,
  ]);

  // Keep a ref version for drawWrappedCanvas (which may run before React re-renders
  // after an asset switch) so it always knows the latest target canvas size.
  useEffect(() => {
    displaySizeRef.current = { displayWidth, displayHeight, offsetX, offsetY };
  }, [displayWidth, displayHeight, offsetX, offsetY]);

  // Initialize default transform if missing or invalid (zero-sized),
  // always recentering the clip in the preview rect.
  useEffect(() => {
    if (!overrideClip && displayWidth > 0 && displayHeight > 0) {
      const hasTransform = !!clipTransform;
      const width = clipTransform?.width ?? 0;
      const height = clipTransform?.height ?? 0;
      const needsInit = !hasTransform || width <= 0 || height <= 0;

      if (needsInit) {
        setClipTransform(clipId, {
          x: offsetX,
          y: offsetY,
          width: displayWidth,
          height: displayHeight,
          scaleX: 1,
          scaleY: 1,
          rotation: 0,
        });
      }
    }
  }, [
    clipTransform,
    displayWidth,
    displayHeight,
    offsetX,
    offsetY,
    clipId,
    setClipTransform,
    overrideClip,
  ]);

  // Hard guarantee: clip transform width/height are never zero or negative.
  // If we ever see an invalid size, immediately normalize it to a sane value.
  useEffect(() => {
    if (!clipTransform) return;
    // Do not mutate store transforms when rendering an override-only clip.
    if (overrideClip) return;

    const currentWidth = clipTransform.width ?? 0;
    const currentHeight = clipTransform.height ?? 0;

    if (currentWidth > 0 && currentHeight > 0) return;

    const fallbackWidth =
      (displayWidth && displayWidth > 0 ? displayWidth : currentWidth) || 1;
    const fallbackHeight =
      (displayHeight && displayHeight > 0 ? displayHeight : currentHeight) || 1;

    setClipTransform(clipId, {
      ...clipTransform,
      // When we normalize an invalid transform, also recenter the clip
      // within the preview rect so it remains visually centered.
      x: offsetX,
      y: offsetY,
      width: Math.max(fallbackWidth, 1),
      height: Math.max(fallbackHeight, 1),
    });
  }, [
    clipTransform,
    displayWidth,
    displayHeight,
    offsetX,
    offsetY,
    clipId,
    setClipTransform,
    overrideClip,
  ]);

  // Ensure canvas matches display size for crisp rendering
  useEffect(() => {
    if (!canvasRef.current) return;
    if (!displayWidth || !displayHeight) return;
    const canvas = canvasRef.current;
    const w = Math.floor(displayWidth);
    const h = Math.floor(displayHeight);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }, [displayWidth, displayHeight]);

  const ensureProcessingCanvas = useCallback(
    (width: number, height: number) => {
      let canvas = processingCanvasRef.current;
      if (!canvas) {
        canvas = document.createElement("canvas");
        processingCanvasRef.current = canvas;
      }
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      return canvas;
    },
    [],
  );

  const drawWrappedCanvas = useCallback(
    (wc: { canvas: HTMLCanvasElement | OffscreenCanvas | VideoFrame; timestamp: number; duration: number }, maskFrame?: number) => {
      let canvas = canvasRef.current;
      if (!canvas) return;

      // If the active source asset changes (assetId/selectedAssetId switch) and the
      // aspect-fit size is different, ensure we resize our backing canvas before drawing.
      // This prevents drawing new frames into a stale-sized canvas.
      const targetW = Math.floor(displaySizeRef.current.displayWidth || 0);
      const targetH = Math.floor(displaySizeRef.current.displayHeight || 0);
      if (targetW > 0 && targetH > 0) {
        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
          // Any cached intermediate canvases must be reset to match the new size.
          originalFrameRef.current = null;
          processingCanvasRef.current = null;
        }
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = true;
      // @ts-ignore
      ctx.imageSmoothingQuality = "high";
      try {
        ctx.drawImage(wc.canvas, 0, 0, canvas.width, canvas.height);
      } catch {}

      // Store the original unfiltered frame for filter adjustments while paused
      if (!originalFrameRef.current) {
        originalFrameRef.current = document.createElement("canvas");
      }
      if (
        originalFrameRef.current.width !== canvas.width ||
        originalFrameRef.current.height !== canvas.height
      ) {
        originalFrameRef.current.width = canvas.width;
        originalFrameRef.current.height = canvas.height;
      }

      const origCtx = originalFrameRef.current.getContext("2d");
      if (origCtx) {
        origCtx.clearRect(0, 0, canvas.width, canvas.height);
        origCtx.drawImage(canvas, 0, 0);
      }

      const workingCanvas = ensureProcessingCanvas(canvas.width, canvas.height);
      const workingCtx = workingCanvas.getContext("2d");
      if (!workingCtx) return;

      workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
      workingCtx.drawImage(canvas, 0, 0);

      // Apply masks before running filters/applicators so downstream operations see masked pixels

      const maskedCanvas = toolRef.current !== "mask" ? applyMaskRef.current(workingCanvas, maskFrame) : workingCanvas;
      if (maskedCanvas !== workingCanvas) {
        workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
        try {
          workingCtx.drawImage(
            maskedCanvas,
            0,
            0,
            workingCanvas.width,
            workingCanvas.height,
          );
        } catch {}
      }

      // Apply WebGL filters for better performance (fast enough for real-time playback)
      // Use ref values to avoid callback recreation on filter/applicator changes
      applyFiltersRef.current(workingCanvas, filterParamsRef.current);

      // Apply applicators to canvas
      let processedCanvas = workingCanvas;

      for (const applicator of applicatorsRef.current) {
        const result = applicator.apply(processedCanvas);
        // Ensure result is copied back to working canvas for chaining
        if (result !== processedCanvas) {
          workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
          workingCtx.drawImage(
            result,
            0,
            0,
            workingCanvas.width,
            workingCanvas.height,
          );
          processedCanvas = workingCanvas;
        }
      }

      // Always draw the final processed result back to display canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(processedCanvas, 0, 0, canvas.width, canvas.height);
      imageRef.current?.getLayer()?.batchDraw?.();
    },
    [ensureProcessingCanvas],
  );

  const decoderMaskFrameRef = useRef(0);
  useEffect(() => {
    decoderMaskFrameRef.current = maskFrameForCurrentFocus;
  }, [maskFrameForCurrentFocus]);

  const getTargetFrameInfo = useCallback(() => {
    if (!mediaInfo.current) return null;
    const clipFps =
      mediaInfo.current.stats.video?.averagePacketRate || fps || DEFAULT_FPS;
    const projectFps = fps || DEFAULT_FPS;
    if (!Number.isFinite(clipFps) || clipFps <= 0) return null;
    if (!Number.isFinite(projectFps) || projectFps <= 0) return null;

    const isUsingPreprocessorSrc = selectedAssetId !== assetId;
    const currentFrameForSeek = focusFrame - startFrameUsed + (trimStart || 0);
    const adjustedCurrentFrame = isUsingPreprocessorSrc
      ? currentFrameForSeek - (trimStart || 0)
      : currentFrameForSeek;
    const idealFrame =
      Math.max(0, adjustedCurrentFrame - frameOffset) * Math.max(0.1, speed);
    const actualFrame = Math.round((idealFrame / projectFps) * clipFps);
    const totalFrames = Math.max(
      0,
      Math.floor((mediaInfo.current.duration || 0) * clipFps),
    );
    const targetFrame =
      Math.max(0, Math.min(totalFrames, actualFrame)) +
      Math.round(((mediaInfo.current.startFrame || 0) / projectFps) * clipFps);

    return { timestamp: targetFrame / clipFps, targetFrame };
  }, [
    mediaInfo,
    fps,
    selectedAssetId,
    assetId,
    focusFrame,
    startFrameUsed,
    trimStart,
    frameOffset,
    speed,
  ]);

  const renderPosterFallback = useCallback(
    async (opts?: { force?: boolean }) => {

      if (!opts?.force && isPlaying) return;
      if (hidden || !isInFrame) return;
      if (!canvasRef.current) return;
      if (!selectedAssetId) return;
      if (originalFrameRef.current) return;

      const info = mediaInfo.current;
      if (!info) return;


      let { displayWidth: targetW, displayHeight: targetH } =
        displaySizeRef.current;
      if (!targetW || !targetH) {
        const fallback = getAspectFitSize(info, rectWidth, rectHeight);
        displaySizeRef.current = fallback;
        targetW = fallback.displayWidth;
        targetH = fallback.displayHeight;
      }

      const width = Math.max(1, Math.floor(targetW || 0));
      const height = Math.max(1, Math.floor(targetH || 0));
      if (!width || !height) return;

      const targetInfo = getTargetFrameInfo();
      const frameIndex = Math.max(
        0,
        Math.floor(targetInfo?.targetFrame ?? 0),
      );
      const posterKey = `${selectedAssetId}|${frameIndex}|${width}x${height}`;
      if (lastPosterKeyRef.current === posterKey) return;

      const asset = getAssetById(selectedAssetId);
      if (!asset?.path) return;

      const token = ++posterRequestRef.current;
      lastPosterKeyRef.current = posterKey;

      const poster = await generatePosterCanvas(asset.path, width, height, {
        mediaInfo: info,
        frameIndex,
        masks: posterMasks.length ? posterMasks : undefined,
        preprocessors: posterPreprocessors.length
          ? posterPreprocessors
          : undefined,
      });

      if (posterRequestRef.current !== token) return;
      if (!poster) return;
      if (originalFrameRef.current || lastRenderedFrameRef.current >= 0) {
        return;
      }

      const fallbackCanvas = document.createElement("canvas");
      fallbackCanvas.width = width;
      fallbackCanvas.height = height;
      const ctx = fallbackCanvas.getContext("2d");
      if (!ctx) return;
      try {
        ctx.drawImage(poster, 0, 0, width, height);
      } catch {
        return;
      }

      const maskFrame = maskFrameForCurrentFocus;
      decoderMaskFrameRef.current = maskFrame;
      drawWrappedCanvas(
        {
          canvas: fallbackCanvas,
          timestamp: targetInfo?.timestamp ?? 0,
          duration: 0,
        },
        maskFrame,
      );
    },
    [
      isPlaying,
      hidden,
      isInFrame,
      selectedAssetId,
      rectWidth,
      rectHeight,
      getAssetById,
      getTargetFrameInfo,
      drawWrappedCanvas,
      posterMasks,
      posterPreprocessors,
      maskFrameForCurrentFocus,
    ],
  );

  const seekToCurrentFrame = useCallback(
    async (isAccurateSeekNeededInput: boolean = false) => {
      

      // NOTE: Do NOT use `useInputControlsStore.getState()` here:
      // it reads the global fallback store (wrong clip scope) and will return false
      // during input playback, causing us to seek every frame.
      // Use ref to avoid triggering seeks when pausing - this prevents frame jumping
      if (isPlayingRef.current) return;


    const info = getTargetFrameInfo();


    if (!info) {
      void renderPosterFallback();
      return;
    }

    const { timestamp, targetFrame } = info;

    // Update the mask frame ref immediately before seeking to ensure sync
    decoderMaskFrameRef.current = maskFrameForCurrentFocus;

    // Cancel any ongoing paused seek operations (do not interfere with live decode token)
    const myToken = ++drawTokenRef.current;
    if (isAccurateSeekNeeded) {
      isAccurateSeekNeededInput = true;
    }

    try {
      const targetAssetId = selectedAssetId;
      if (!targetAssetId) return;

      const logicalId = makeDecoderId(targetAssetId);

      await decoderManager.seek(logicalId, timestamp, isAccurateSeekNeededInput);
      activeDecoderAssetIdRef.current = logicalId;

      if (myToken === drawTokenRef.current) {
         lastRenderedFrameRef.current = targetFrame;
      }
    } catch (e) {
      console.warn("[video] seek failed", e);
      void renderPosterFallback({ force: true });
    }
    },
    [
      decoderManager,
      getTargetFrameInfo,
      //renderPosterFallback,
      maskFrameForCurrentFocus,
      isAccurateSeekNeeded,
      selectedAssetId,
      makeDecoderId,
      isInFrame
    ],
  );
  

  useEffect(() => {
    
    const configureDecoders = async () => {
      const ids = new Set<string>();
      if (assetId) ids.add(assetId);
      clip?.preprocessors?.forEach((p) => {
        if (p.assetId) ids.add(p.assetId);
      });
    
      for (const id of ids) {
        try {
          let info = getMediaInfoCached(id);
          const asset = getAssetById(id);
          if (!info && asset?.path) {
            info = await getMediaInfo(asset.path, {
              sourceDir: clip.type === "video" ? "user-data" : "apex-cache",
            });
          }

          if (!info || !asset) continue;

          // If this is the currently-selected source, publish the mediaInfo into React
          // state immediately so sizing updates before the first frame renders.
          if (id === selectedAssetId) {
            setMediaInfoAndBump(info);
            displaySizeRef.current = getAspectFitSize(info, rectWidth, rectHeight);
          }

          const config = info.videoDecoderConfig;
          if (!config) continue;


          const logicalId = makeDecoderId(id);

          const onFrame = (data: {
            canvas: VideoFrame;
            timestamp: number;
            duration: number;
          }) => {
            drawWrappedCanvas(data, decoderMaskFrameRef.current);
          };

          const onError = (e: Error) =>
            console.error("[VideoDecoderManager] Error", id, e);

          if (decoderManager.hasAsset(logicalId)) {
            
            decoderManager.updateAssetHandlers(logicalId, { onFrame, onError });
            // Trigger seek for existing decoders since onReady won't be called

            try {
            await seekToCurrentFrame(true);
            } catch (e) {
              console.error("Error seeking to current frame", id, e);
            }
          } else {
            const activeProject = getActiveProject();
            decoderManager.addAsset(asset, {
              mediaInfo: info,
              videoDecoderConfig: config,
              folderUuid: activeProject?.folderUuid,
              onFrame,
              onError,
              logicalId,
              onReady: async () => {
                await seekToCurrentFrame(true);
              },
            });
          }
        } catch (e) {
          console.error("Error configuring decoder for", id, e);
        }
      }
    };

    void configureDecoders();

   
  }, [
    assetId,
    clip?.preprocessors,
    decoderManager,
    getAssetById,
    drawWrappedCanvas,
    makeDecoderId
  ]);

  useEffect(() => {
    void seekToCurrentFrame();
  }, [seekToCurrentFrame]);

  useEffect(() => {
    void renderPosterFallback();
  }, [
    renderPosterFallback,
    isPlaying,
    hidden,
    isInFrame,
    mediaInfoVersion,
    selectedAssetId,
  ]);

  const startRendering = useCallback(async () => {
    if (!canvasRef.current) return;
    if (!mediaInfo.current) return;
    if (!displayWidth || !displayHeight) return;
    const clipFps =
      mediaInfo.current?.stats.video?.averagePacketRate || fps || DEFAULT_FPS;
    const projectFps = fps || DEFAULT_FPS;

    if (!Number.isFinite(clipFps) || clipFps <= 0) return;
    if (!Number.isFinite(projectFps) || projectFps <= 0) return;

    const { startTime, endTime, startIdx } = calculateIterateRange(
      currentFrame,
      trimStart,
      frameOffset,
      speed,
      clipFps,
      projectFps,
      mediaInfo.current,
      selectedAssetId,
      assetId
    );

    currentStartFrameRef.current = startIdx;
    lastRenderedFrameRef.current = startIdx - 1;

    const myToken = ++drawTokenRef.current;
    // @ts-ignore
    iteratorRef.current?.return?.();

    const activeAssetId = selectedAssetId;
    if (!activeAssetId) return;

    const logicalId = makeDecoderId(activeAssetId);

    const checkCancel = () => {
        if (!offscreenFast && myToken !== drawTokenRef.current) return false;
        if (!isPlaying) return false;
        return true;
    };

    try {
      // Ensure the decoder is positioned at our intended start time before iterating.
      // This is particularly important when replaying after hitting end-of-stream, where
      // the underlying decoder/worker may not automatically rewind for a backwards range.
      if (!checkCancel()) return;
      try {
        await decoderManager.seek(logicalId, startTime, true);
      } catch {}
      if (!checkCancel()) return;

      await decoderManager.iterate(
        logicalId,
        startTime,
        endTime,
        async (ts) => {
          if (!checkCancel()) return;

          let sampleIdx = Number.isFinite(ts)
            ? Math.floor(ts * clipFps + 1e-4)
            : lastRenderedFrameRef.current + 1;

          const isUsingPreprocessorSrc = selectedAssetId !== assetId;
          const computeLocalFocusMedia = () => {
            const focusFrameValue = focusFrameRef.current;
            // Base timeline-local frames relative to clip start (no give-start applied)
            const baseLocal = Math.max(
              0,
              (focusFrameValue ?? 0) - startFrameUsed,
            );
            // When using preprocessor src, align to its own frame space by subtracting its start offset.
            // Otherwise, include trimStart to match the main clip's reference frame.
            const derivedLocal = isUsingPreprocessorSrc
              ? Math.max(0, baseLocal - Math.max(0, frameOffset))
              : Math.max(0, baseLocal + (trimStart || 0));
            const localProjectFrames =
              typeof currentLocalFrameOverride === "number"
                ? Math.max(0, currentLocalFrameOverride)
                : derivedLocal;
            const speedAdjusted = Math.max(
              0,
              localProjectFrames * Math.max(0.1, speed),
            );
            // Map from project fps to native fps using floor to reduce jitter
            const actualFrameIdx = Math.floor(
              (speedAdjusted / projectFps) * clipFps + 1e-4,
            );
            return (
              actualFrameIdx +
              Math.round(
                ((mediaInfo.current?.startFrame || 0) / projectFps) * clipFps,
              )
            );
          };

          if (!offscreenFast) {
            // Skip stale frames that are behind the timeline by more than 1 frame
            let localFocus = computeLocalFocusMedia();
            if (sampleIdx < localFocus - 1) {
              lastRenderedFrameRef.current = sampleIdx;
              return;
            }
            // If we're ahead of the timeline, wait until the timeline catches up (sync to rAF)
            while (sampleIdx > (localFocus = computeLocalFocusMedia())) {
              if (!checkCancel()) return;
              await new Promise<void>((resolve) =>
                requestAnimationFrame(() => resolve()),
              );
            }
          }

          const focusFrameForMask = focusFrameRef.current;
          const speedFactor = Math.max(0.1, speed);
          let maskFrame: number;
          if (clip) {
            if (inputMode) {
              const local = Math.max(0, focusFrameForMask + (trimStart || 0));
              maskFrame = Math.max(0, Math.floor(local * speedFactor));
            } else {
              const isUsingPreprocessorSrc = selectedAssetId !== assetId;
              const baseLocal = Math.max(0, focusFrameForMask - startFrameUsed);
              const derivedLocal = isUsingPreprocessorSrc
                ? Math.max(0, baseLocal - Math.max(0, frameOffset))
                : Math.max(0, baseLocal + (trimStart || 0));
              maskFrame = Math.max(0, Math.floor(derivedLocal * speedFactor));
            }
          } else {
            const local = Math.max(
              0,
              focusFrameForMask - startFrameUsed + (trimStart || 0),
            );
            maskFrame = Math.max(0, Math.floor(local * speedFactor));
          }
          decoderMaskFrameRef.current = maskFrame;
          lastRenderedFrameRef.current = sampleIdx;
        },
        checkCancel
      );
    } catch (e: any) {
      console.log("startRendering error", e);
      void renderPosterFallback({ force: true });
    }
  }, [
    mediaInfo,
    fps,
    selectedAssetId,
    assetId,
    displayWidth,
    displayHeight,
    currentFrame,
    drawWrappedCanvas,
    speed,
    startFrameUsed,
    frameOffset,
    trimStart,
    clip,
    isPlaying,
    inputMode,
    makeDecoderId,
    renderPosterFallback,
  ]);

  useEffect(() => {
    if (isPlaying) {
      void startRendering();
    }
    return () => {
      drawTokenRef.current++;
      // @ts-ignore
      iteratorRef.current?.return?.();
    };
  }, [
    isPlaying,
    offscreenFast,
    selectedAssetId,
    assetId,
    mediaInfo,
    displayWidth,
    displayHeight,
    fps,
    speed,
    frameOffset,
    applicators.length,
    inputId,
    inputMode,
  ]);

  // Restart iteration if focusFrame jumps backwards during playback (e.g. replay from end).
  useEffect(() => {
    if (!isPlaying) {
      prevFocusFrameWhilePlayingRef.current = focusFrame;
      return;
    }
    const prev = prevFocusFrameWhilePlayingRef.current;
    prevFocusFrameWhilePlayingRef.current = focusFrame;
    if (typeof prev === "number" && Number.isFinite(prev)) {
      // Any backwards jump indicates a discontinuity (scrub or replay). Restart decode.
      if (focusFrame < prev) {
        void startRendering();
      }
    }
  }, [focusFrame, isPlaying, startRendering]);

  // If video is paused, reapply filters and applicators when they change
  useEffect(() => {
    if (!isPlaying && canvasRef.current && imageRef.current) {
      // If we have an original frame cached, use it for fast reapplication
      if (originalFrameRef.current) {
        let canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const workingCanvas = ensureProcessingCanvas(
            canvas.width,
            canvas.height,
          );
          const workingCtx = workingCanvas.getContext("2d");
          if (!workingCtx) return;

          // Start with the original unfiltered frame
          workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
          workingCtx.drawImage(originalFrameRef.current, 0, 0);

          // Apply masks before filters so masked pixels feed the rest of the pipeline
          const maskedCanvas = toolRef.current !== "mask" ? applyMaskRef.current(workingCanvas, maskFrameForCurrentFocus) : workingCanvas;
          if (maskedCanvas !== workingCanvas) {
            workingCtx.clearRect(
              0,
              0,
              workingCanvas.width,
              workingCanvas.height,
            );
            workingCtx.drawImage(
              maskedCanvas,
              0,
              0,
              workingCanvas.width,
              workingCanvas.height,
            );
          }

          // Apply filters to the clean frame
          applyFilters(workingCanvas, filterParamsRef.current);

          // Apply applicators (filter clips from layers above)
          let processedCanvas = workingCanvas;
          for (const applicator of applicatorsRef.current) {
            const result = applicator.apply(processedCanvas);
            if (result !== processedCanvas) {
              workingCtx.clearRect(
                0,
                0,
                workingCanvas.width,
                workingCanvas.height,
              );
              workingCtx.drawImage(
                result,
                0,
                0,
                workingCanvas.width,
                workingCanvas.height,
              );
              processedCanvas = workingCanvas;
            }
          }

          // Always draw final result back to display canvas
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(processedCanvas, 0, 0, canvas.width, canvas.height);

          imageRef.current.getLayer()?.batchDraw();
        }
      } else {
        // If no cached frame exists, decode the current frame
        // Force re-decode even if we already rendered this frame index
        lastRenderedFrameRef.current = -1;
      }
    }
  }, [
    clip?.brightness,
    clip?.contrast,
    clip?.hue,
    clip?.saturation,
    clip?.blur,
    clip?.sharpness,
    clip?.noise,
    clip?.vignette,
    isPlaying,
    applyFilters,
    applicators,
    applicators.length,
    applyMask,
    maskFrameForCurrentFocus,
    ensureProcessingCanvas,
    inputId,
    inputMode,
  ]);

  // Ensure any CLUTs needed by filter applicators are preloaded before drawing
  useEffect(() => {
    let cancelled = false;
    const maybePreload = async () => {
      const preloadTasks: Promise<void>[] = [];
      for (const app of applicatorsRef.current) {
        const maybeEnsure = (app as any)?.ensureResources as
          | (() => Promise<void>)
          | undefined;
        if (typeof maybeEnsure === "function") {
          preloadTasks.push(maybeEnsure());
        }
      }
      if (preloadTasks.length) {
        try {
          await Promise.all(preloadTasks);
        } catch {}
      }
      if (cancelled) return;
      // After resources are ready, force redraw immediately
      if (canvasRef.current) {
        lastRenderedFrameRef.current = -1;
       
        imageRef.current?.getLayer()?.batchDraw?.();
      }
    };
    void maybePreload();
    return () => {
      cancelled = true;
    };
  }, [applicators, applicators.length, isPlaying, inputId, inputMode]);



  // In offscreen/single-frame scenarios, ensure immediate seek when explicit overrides change (no debounce)
  useEffect(() => {
    if (isPlaying) return;
    // Force a draw whenever caller overrides the exact frame to display
    if (
      typeof focusFrameOverride === "number" ||
      typeof currentLocalFrameOverride === "number"
    ) {
      lastRenderedFrameRef.current = -1;
    }
  }, [focusFrameOverride, currentLocalFrameOverride, isPlaying]);

  // Force re-init when the selected clip changes (clipId) or overrideClip identity changes
  useEffect(() => {
    // reset caches to guarantee re-render of first frame for new selection
    lastSelectedAssetIdRef.current = null;
    lastRenderedFrameRef.current = -1;
    lastPosterKeyRef.current = null;
    originalFrameRef.current = null;
    // @ts-ignore
    iteratorRef.current?.return?.();
    iteratorRef.current = null;
  }, [clipId, overrideClip]);


  const handleDragMove = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      updateGuidesAndMaybeSnap({ snap: true });
      const node = imageRef.current;
      if (node) {
        setClipTransform(clipId, { x: node.x(), y: node.y() });
      } else {
        setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
      }
    },
    [setClipTransform, clipId, updateGuidesAndMaybeSnap],
  );

  const handleDragStart = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      e.target.getStage()!.container().style.cursor = "grab";
      addClipSelection(clipId);
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
      setIsInteracting(true);
      updateGuidesAndMaybeSnap({ snap: true });
    },
    [clipId, addClipSelection, updateGuidesAndMaybeSnap],
  );

  const handleDragEnd = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      e.target.getStage()!.container().style.cursor = "default";
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
      setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
      setIsInteracting(false);
      setGuides({
        vCenter: false,
        hCenter: false,
        v25: false,
        v75: false,
        h25: false,
        h75: false,
        left: false,
        right: false,
        top: false,
        bottom: false,
      });
    },
    [setClipTransform, clipId],
  );

  const handleClick = useCallback(() => {
    if (isFullscreen) return;
    if (hidden) return;
    clearSelection();
    addClipSelection(clipId);
  }, [addClipSelection, clipId, isFullscreen, hidden]);

  // If we become visible after being hidden (prewarmed), force a redraw so the
  // already-decoded backing canvas is displayed immediately.
  useEffect(() => {
    if (hidden) return;
    try {
      imageRef.current?.getLayer()?.batchDraw?.();
    } catch {}
  }, [hidden]);

  useEffect(() => {
    const transformer = transformerRef.current;
    if (!transformer) return;
    const bumpSuppress = () => {
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 300);
    };
    const onTransformStart = () => {
      bumpSuppress();
      setIsTransforming(true);
      const active = (transformer as any)?.getActiveAnchor?.();
      const rotating = typeof active === "string" && active.includes("rotater");
      setIsRotating(!!rotating);
      setIsInteracting(true);
      if (!rotating) {
        updateGuidesAndMaybeSnap({ snap: false });
      } else {
        setGuides({
          vCenter: false,
          hCenter: false,
          v25: false,
          v75: false,
          h25: false,
          h75: false,
          left: false,
          right: false,
          top: false,
          bottom: false,
        });
      }
    };
    const persistTransform = () => {
      const node = imageRef.current;
      if (!node) return;
      const newWidth = node.width() * node.scaleX();
      const newHeight = node.height() * node.scaleY();
      setClipTransform(clipId, {
        x: node.x(),
        y: node.y(),
        width: newWidth,
        height: newHeight,
        scaleX: 1,
        scaleY: 1,
        rotation: node.rotation(),
      }, true, true);
      node.width(newWidth);
      node.height(newHeight);
      node.scaleX(1);
      node.scaleY(1);
    };
    const onTransform = () => {
      bumpSuppress();
      if (!isRotating) {
        updateGuidesAndMaybeSnap({ snap: false });
      }
      persistTransform();
    };

    const onTransformEnd = () => {
      bumpSuppress();
      setIsTransforming(false);
      setIsInteracting(false);
      setIsRotating(false);
      setGuides({
        vCenter: false,
        hCenter: false,
        v25: false,
        v75: false,
        h25: false,
        h75: false,
        left: false,
        right: false,
        top: false,
        bottom: false,
      });
      persistTransform();
    };
    transformer.on("transformstart", onTransformStart);
    transformer.on("transform", onTransform);
    transformer.on("transformend", onTransformEnd);
    return () => {
      transformer.off("transformstart", onTransformStart);
      transformer.off("transform", onTransform);
      transformer.off("transformend", onTransformEnd);
    };
  }, [
    transformerRef.current,
    updateGuidesAndMaybeSnap,
    setClipTransform,
    clipId,
    isRotating,
  ]);

  useEffect(() => {
    if (inputMode) return;
    const handleWindowClick = (e: MouseEvent) => {
      if (!isSelected) return;
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      if (now < suppressUntilRef.current) return;
      const stage = imageRef.current?.getStage();
      const container = stage?.container();
      // check that node is inside container
      const node = e.target;
      if (!container?.contains(node as Node)) return;
      if (!stage || !container || !imageRef.current) return;
      const containerRect = container.getBoundingClientRect();
      const pointerX = e.clientX - containerRect.left;
      const pointerY = e.clientY - containerRect.top;
      const imgRect = imageRef.current.getClientRect({
        skipShadow: true,
        skipStroke: true,
      });
      const insideImage =
        pointerX >= imgRect.x &&
        pointerX <= imgRect.x + imgRect.width &&
        pointerY >= imgRect.y &&
        pointerY <= imgRect.y + imgRect.height;

      if (!insideImage) {
        removeClipSelection(clipId);
      }
    };
    window.addEventListener("click", handleWindowClick);
    return () => {
      window.removeEventListener("click", handleWindowClick);
    };
  }, [clipId, isSelected, removeClipSelection, inputMode]);

  // Calculate pixel crop from normalized crop for Konva Image
  const pixelCrop = useMemo(() => {
    const c = clipTransform?.crop;
    if (!c || !displayWidth || !displayHeight) return undefined;
    return {
      x: c.x * displayWidth,
      y: c.y * displayHeight,
      width: c.width * displayWidth,
      height: c.height * displayHeight,
    };
  }, [clipTransform?.crop, displayWidth, displayHeight]);

  const nodeWidth = useMemo(
    () =>
      clipTransform?.width && clipTransform.width > 0
        ? clipTransform.width
        : displayWidth || 1,
    [clipTransform?.width, displayWidth],
  );
  const nodeHeight = useMemo(
    () =>
      clipTransform?.height && clipTransform.height > 0
        ? clipTransform.height
        : displayHeight || 1,
    [clipTransform?.height, displayHeight],
  );
  const safeCornerRadius = useMemo(
    () =>
      sanitizeCornerRadius(clipTransform?.cornerRadius, nodeWidth, nodeHeight),
    [clipTransform?.cornerRadius, nodeWidth, nodeHeight],
  );

  

  // Only render Konva nodes when the clip is active in the current frame and not explicitly hidden.
  if (hidden || !isInFrame) {
    return null;
  }



  return (
    <React.Fragment>
      <Group
        ref={groupRef}
        clipX={0}
        clipY={0}
        clipWidth={rectWidth}
        clipHeight={rectHeight}
      >
        
        <Image
          
          visible={!hidden}
          listening={!hidden}
          draggable={tool === "pointer" && !isTransforming && !inputMode && !hidden}
          ref={imageRef}
          image={imageSource || undefined}
          x={clipTransform?.x ?? offsetX}
          y={clipTransform?.y ?? offsetY}
          width={nodeWidth}
          height={nodeHeight}
          scaleX={clipTransform?.scaleX ?? 1}
          scaleY={clipTransform?.scaleY ?? 1}
          rotation={clipTransform?.rotation ?? 0}
          cornerRadius={safeCornerRadius}
          opacity={(clipTransform?.opacity ?? 100) / 100}
          crop={pixelCrop}
          onDragMove={handleDragMove}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
          onClick={handleClick}
        />
        {tool === "pointer" &&
          isSelected &&
          isInteracting &&
          !isRotating &&
          !isFullscreen && (
            <React.Fragment>
              {guides.vCenter && (
                <Line
                  listening={false}
                  points={[rectWidth / 2, 0, rectWidth / 2, rectHeight]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.v25 && (
                <Line
                  listening={false}
                  points={[rectWidth * 0.25, 0, rectWidth * 0.25, rectHeight]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.v75 && (
                <Line
                  listening={false}
                  points={[rectWidth * 0.75, 0, rectWidth * 0.75, rectHeight]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.hCenter && (
                <Line
                  listening={false}
                  points={[0, rectHeight / 2, rectWidth, rectHeight / 2]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.h25 && (
                <Line
                  listening={false}
                  points={[0, rectHeight * 0.25, rectWidth, rectHeight * 0.25]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.h75 && (
                <Line
                  listening={false}
                  points={[0, rectHeight * 0.75, rectWidth, rectHeight * 0.75]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.left && (
                <Line
                  listening={false}
                  points={[0, 0, 0, rectHeight]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.right && (
                <Line
                  listening={false}
                  points={[rectWidth, 0, rectWidth, rectHeight]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.top && (
                <Line
                  listening={false}
                  points={[0, 0, rectWidth, 0]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.bottom && (
                <Line
                  listening={false}
                  points={[0, rectHeight, rectWidth, rectHeight]}
                  stroke={"#AE81CE"}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
            </React.Fragment>
          )}
      </Group>
      <Transformer
        borderStroke="#AE81CE"
        anchorCornerRadius={8}
        anchorStroke="#E3E3E3"
        anchorStrokeWidth={1}
        borderStrokeWidth={2}
        visible={
          !hidden &&
          tool === "pointer" &&
          isSelected &&
          !isFullscreen &&
          overlap &&
          !inputMode
        }
        listening={!hidden}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]}
        boundBoxFunc={transformerBoundBoxFunc as any}
        ref={(node) => {
          transformerRef.current = node;
          if (node && imageRef.current) {
            node.nodes([imageRef.current]);
            if (typeof (node as any).forceUpdate === "function") {
              (node as any).forceUpdate();
            }
            node.getLayer()?.batchDraw?.();
          }
        }}
        enabledAnchors={[
          "top-left",
          "bottom-right",
          "top-right",
          "bottom-left",
        ]}
      />
    </React.Fragment>
  );
};

export default VideoPreview;
