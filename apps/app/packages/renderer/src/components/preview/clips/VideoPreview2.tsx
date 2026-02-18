import { MediaInfo, VideoClipProps } from "@/lib/types";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Image, Transformer, Group, Line} from "react-konva";

import { useControlsStore } from "@/lib/control";
import Konva from "konva";
import { useViewportStore } from "@/lib/viewport";
import { DEFAULT_FPS } from "@/lib/settings";
import { useClipStore } from "@/lib/clip";
import { WrappedCanvas } from "mediabunny";
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import { FrameBlitter } from "@/components/preview/webgl/FrameBlitter";
import { BaseClipApplicator } from "./apply/base";
import _ from "lodash";
import { useWebGLMask } from "../mask/useWebGLMask";
import { useInputControlsStore } from "@/lib/inputControl";
import { sanitizeCornerRadius } from "@/lib/konva/sanitizeCornerRadius";
import { SELECTION_STROKE_COLOR } from "@/lib/selectionStroke";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
// (prefetch helper removed by request; timeline-driven rendering only)



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
  hidden = false,
}) => {

  const mediaInfo = useRef<MediaInfo | null>(getMediaInfoCached(assetId) || null);
  const mediaInfoAssetIdRef = useRef<string | null>(
    mediaInfo.current ? assetId : null,
  );
  // `mediaInfo` is stored in a ref for fast access by decoder callbacks, but ref updates
  // don't trigger React renders. We bump this version whenever `mediaInfo.current` changes
  // so aspect-fit sizing and Konva props update immediately (no "wait until drag" issues).
  const [mediaInfoVersion, setMediaInfoVersion] = useState(0);
  const setMediaInfoAndBump = useCallback((info: MediaInfo | null) => {
    mediaInfo.current = info;
    setMediaInfoVersion((v) => v + 1);
  }, []);

  const focusFrameFromControls = useControlsStore((state) => state.focusFrame);
  const focusFrameFromInputs = useInputControlsStore((s) =>
    s.getFocusFrame(inputId ?? ""),
  );


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
  const frameBlitterRef = useRef<FrameBlitter | null>(null);
  const use2dPresentRef = useRef(false);
  const imageRef = useRef<Konva.Image>(null);
  const transformerRef = useRef<Konva.Transformer>(null);

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

    
      if (inputMode) {
        if (!clip.groupId) {
          return true;
        }
      }

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
  const speedRef = useRef(speed);
  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);
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
  const { selectedClipIds, isFullscreen, fps: srcFps } = useControlsStore();
  const isSelected = useMemo(
    () => selectedClipIds.includes(clipId),
    [clipId, selectedClipIds],
  );

  const getAssetById = useClipStore((s) => s.getAssetById);
  const lastSelectedAssetIdRef = useRef<string | null>(null);
  const cachedPreprocessorRangeRef = useRef<{
    startFrame: number;
    endFrame: number;
    selectedAssetId: string;
    frameOffset: number;
  } | null>(null);
  const addedTimestampRef = useRef<number | undefined>(undefined); // last timestamp rendered


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
    colorTintColor: clip?.colorTintColor,
    colorTintIntensity: clip?.colorTintIntensity,
    scanLines: clip?.scanLines,
    chromaticAberration: clip?.chromaticAberration,
    interlace: clip?.interlace,
    pixelate: clip?.pixelate,
    jitter: clip?.jitter,
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
  const allowPlayback = offscreenFast || (!hidden && isInFrame);
  const allowPlaybackRef = useRef(allowPlayback);
  useEffect(() => {
    allowPlaybackRef.current = allowPlayback;
  }, [allowPlayback]);
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

  const fpsFromControls = useControlsStore((s) => s.fps);
  const fpsFromInputs = useInputControlsStore((s) => s.getFps(inputId ?? ""));
  const fps = useInputScopedControls ? fpsFromInputs : fpsFromControls;
  const lastRenderedFrameRef = useRef<number>(-1);
  const lastDrawnFocusFrameRef = useRef<number | null>(null);
  const skipDrawRef = useRef(false);
  const suppressSeekFramesRef = useRef(false);
  const resumeGateFrameRef = useRef<number | null>(null);
  const pendingSeekTargetRef = useRef<{ frame: number; strict: boolean } | null>(
    null,
  );
  const seekLockStartedAtMsRef = useRef(0);
  const seekLockSkippedFramesRef = useRef(0);
  const fpsRef = useRef(fps);
  useEffect(() => {
    fpsRef.current = fps;
  }, [fps]);



  const videoRef = useRef<HTMLVideoElement | null>(null);
  const pendingVideoFrameCallbackRef = useRef<number | null>(null);
  const inVideoFrameCallbackRef = useRef(false);
  const pendingPlayAfterSeekRef = useRef(false);
  const clearPendingVideoFrameCallback = useCallback(() => {
    const video = videoRef.current;
    const callbackId = pendingVideoFrameCallbackRef.current;
    if (
      video &&
      callbackId !== null &&
      typeof (video as any).cancelVideoFrameCallback === "function"
    ) {
      try {
        (video as any).cancelVideoFrameCallback(callbackId);
      } catch {}
    }
    pendingVideoFrameCallbackRef.current = null;
  }, []);
  const clearPendingSeekTarget = useCallback(() => {
    pendingSeekTargetRef.current = null;
    seekLockStartedAtMsRef.current = 0;
    seekLockSkippedFramesRef.current = 0;
  }, []);

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
      colorTintColor: clip?.colorTintColor,
      colorTintIntensity: clip?.colorTintIntensity,
      scanLines: clip?.scanLines,
      chromaticAberration: clip?.chromaticAberration,
      interlace: clip?.interlace,
      pixelate: clip?.pixelate,
      jitter: clip?.jitter,
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
      frameBlitterRef.current?.dispose();
      frameBlitterRef.current = null;
      use2dPresentRef.current = false;
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
    lastDrawnFocusFrameRef.current = null;
    resumeGateFrameRef.current = null;
    clearPendingSeekTarget();
    pendingPlayAfterSeekRef.current = false;
    clearPendingVideoFrameCallback();
    processingCanvasRef.current = null;
    // @ts-ignore
    iteratorRef.current?.return?.();
    iteratorRef.current = null;

    let cancelled = false;
    const hydrateMediaInfo = async () => {
      let info = getMediaInfoCached(selectedAssetId);

      // Cold page refresh can start before media cache is hydrated.
      // Fallback to async metadata load so first frame can render immediately after mount.
      if (!info) {
        const asset = getAssetById(selectedAssetId);
        const mediaPath = asset?.path || selectedAssetId;
        if (mediaPath) {
          try {
            info = await getMediaInfo(mediaPath, {
              sourceDir: mediaPath.startsWith("app://apex-cache/")
                ? "apex-cache"
                : "user-data",
            });
          } catch {}
        }
      }

      if (!info || cancelled) return;

      mediaInfoAssetIdRef.current = selectedAssetId;
      setMediaInfoAndBump(info);
      // Update the "current" aspect-fit size for drawWrappedCanvas immediately so the
      // very first frame of the new asset can't render into a stale-sized canvas.
      displaySizeRef.current = getAspectFitSize(info, rectWidth, rectHeight);

      // Have media info; force immediate redraw
      lastRenderedFrameRef.current = -1;
    };

    void hydrateMediaInfo();
    return () => {
      cancelled = true;
    };
  }, [
    selectedAssetId,
    rectWidth,
    rectHeight,
    setMediaInfoAndBump,
    getAssetById,
    clearPendingSeekTarget,
    clearPendingVideoFrameCallback,
  ]);


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

  const presentCanvas = useCallback((sourceCanvas: HTMLCanvasElement) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (
      canvas.width !== sourceCanvas.width ||
      canvas.height !== sourceCanvas.height
    ) {
      canvas.width = sourceCanvas.width;
      canvas.height = sourceCanvas.height;
    }

    if (!use2dPresentRef.current) {
      if (!frameBlitterRef.current) {
        const nextBlitter = new FrameBlitter(canvas);
        if (nextBlitter.isReady()) {
          frameBlitterRef.current = nextBlitter;
        } else {
          nextBlitter.dispose();
          use2dPresentRef.current = true;
        }
      }

      if (frameBlitterRef.current) {
        frameBlitterRef.current.resize(canvas.width, canvas.height);
        if (frameBlitterRef.current.blit(sourceCanvas)) {
          return;
        }
      }
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    // @ts-ignore
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
  }, []);

  const drawWrappedCanvas = useCallback(
    (
      wc: {
        canvas: HTMLCanvasElement | OffscreenCanvas | VideoFrame | HTMLVideoElement;
        timestamp: number;
        duration: number;
      },
      maskFrame?: number,
      opts?: { recordFrame?: boolean },
    ) => {
      const displayCanvas = canvasRef.current;
      if (!displayCanvas) return;

      if (isPlayingRef.current && suppressSeekFramesRef.current) {
        skipDrawRef.current = false;
        return;
      }

      if (isPlayingRef.current && skipDrawRef.current) {
        skipDrawRef.current = false;
        return;
      }

      const info = mediaInfo.current;
      const clipFps =
        info?.stats.video?.averagePacketRate || fpsRef.current || DEFAULT_FPS;
      const frameIdx =
        Number.isFinite(clipFps) && clipFps > 0
          ? Math.floor(wc.timestamp * clipFps + 1e-4)
          : null;
      const pendingSeek = pendingSeekTargetRef.current;
      if (
        pendingSeek &&
        pendingSeek.strict &&
        frameIdx !== null
      ) {
        const frameDelta = frameIdx - pendingSeek.frame;
        if (Math.abs(frameDelta) > 2) {
          if (!isPlayingRef.current) {
            return;
          }
          if (frameDelta > 2) {
            clearPendingSeekTarget();
          } else {
            const now =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            if (seekLockStartedAtMsRef.current <= 0) {
              seekLockStartedAtMsRef.current = now;
            }
            seekLockSkippedFramesRef.current += 1;
            const lockAgeMs = now - seekLockStartedAtMsRef.current;
            const shouldReleaseLock =
              lockAgeMs > 250 || seekLockSkippedFramesRef.current > 10;
            if (!shouldReleaseLock) {
              return;
            }
            clearPendingSeekTarget();
          }
        }
      }
      const resumeGate = resumeGateFrameRef.current;
      if (
        isPlayingRef.current &&
        typeof resumeGate === "number" &&
        frameIdx !== null &&
        frameIdx < resumeGate
      ) {
        const now =
          typeof performance !== "undefined" && performance.now
            ? performance.now()
            : Date.now();
        const gateAgeMs =
          seekLockStartedAtMsRef.current > 0
            ? now - seekLockStartedAtMsRef.current
            : 0;
        if (gateAgeMs <= 350) {
          return;
        }
        resumeGateFrameRef.current = null;
      }

      skipDrawRef.current = false;

      // If the active source asset changes (assetId/selectedAssetId switch) and the
      // aspect-fit size is different, ensure we resize our backing canvas before drawing.
      // This prevents drawing new frames into a stale-sized canvas.
      const targetW = Math.floor(displaySizeRef.current.displayWidth || 0);
      const targetH = Math.floor(displaySizeRef.current.displayHeight || 0);
      if (targetW > 0 && targetH > 0) {
        if (
          displayCanvas.width !== targetW ||
          displayCanvas.height !== targetH
        ) {
          displayCanvas.width = targetW;
          displayCanvas.height = targetH;
          // Any cached intermediate canvases must be reset to match the new size.
          originalFrameRef.current = null;
          processingCanvasRef.current = null;
        }
      }

      const workingCanvas = ensureProcessingCanvas(
        displayCanvas.width,
        displayCanvas.height,
      );
      const workingCtx = workingCanvas.getContext("2d");
      if (!workingCtx) return;

      workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
      try {
        workingCtx.drawImage(wc.canvas, 0, 0, workingCanvas.width, workingCanvas.height);
      } catch {
        return;
      }

      // Cache the unfiltered frame only while paused to avoid extra frame copies during playback.
      if (!isPlayingRef.current) {
        if (!originalFrameRef.current) {
          originalFrameRef.current = document.createElement("canvas");
        }
        if (
          originalFrameRef.current.width !== workingCanvas.width ||
          originalFrameRef.current.height !== workingCanvas.height
        ) {
          originalFrameRef.current.width = workingCanvas.width;
          originalFrameRef.current.height = workingCanvas.height;
        }
        const origCtx = originalFrameRef.current.getContext("2d");
        if (origCtx) {
          origCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
          origCtx.drawImage(workingCanvas, 0, 0);
        }
      }

      // Apply masks before running filters/applicators so downstream operations see masked pixels
      const maskedCanvas =
        toolRef.current !== "mask"
          ? applyMaskRef.current(workingCanvas, maskFrame)
          : workingCanvas;
      let processedCanvas = maskedCanvas;

      // Apply WebGL filters for better performance (fast enough for real-time playback)
      // Use passthrough mode to avoid an in-place copy after each filter chain.
      processedCanvas = applyFiltersRef.current(
        processedCanvas,
        filterParamsRef.current,
        { output: "passthrough" },
      );

      for (const applicator of applicatorsRef.current) {
        const result = applicator.apply(processedCanvas);
        if (result) processedCanvas = result;
      }

      // Final present to the Konva source canvas uses a GPU blit when available.
      presentCanvas(processedCanvas);
      if (opts?.recordFrame !== false) {
        if (frameIdx !== null) {
          lastRenderedFrameRef.current = frameIdx;
          if (
            typeof resumeGate === "number" &&
            frameIdx >= resumeGate
          ) {
            resumeGateFrameRef.current = null;
          }
        }
        if (
          pendingSeek &&
          pendingSeek.strict &&
          frameIdx !== null &&
          Math.abs(frameIdx - pendingSeek.frame) <= 2
        ) {
          clearPendingSeekTarget();
        }
        lastDrawnFocusFrameRef.current = focusFrameRef.current;
      }
      imageRef.current?.getLayer()?.batchDraw?.();
    },
    [ensureProcessingCanvas, displayWidth, displayHeight, presentCanvas, clearPendingSeekTarget],
  );

  const drawWrappedCanvasRef = useRef(drawWrappedCanvas);
  useEffect(() => {
    drawWrappedCanvasRef.current = drawWrappedCanvas;
  }, [drawWrappedCanvas]);

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
    const actualFrame = Math.floor((idealFrame / projectFps) * clipFps + 1e-4);
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
  const getTargetFrameInfoRef = useRef(getTargetFrameInfo);
  useEffect(() => {
    getTargetFrameInfoRef.current = getTargetFrameInfo;
  }, [getTargetFrameInfo]);
  const seekVideoToTargetFrame = useCallback(
    (
      video: HTMLVideoElement,
      targetInfo: { timestamp: number; targetFrame: number },
      opts?: {
        strict?: boolean;
        gatePlayback?: boolean;
        thresholdFrames?: number;
      },
    ) => {
      const { timestamp, targetFrame } = targetInfo;
      if (!Number.isFinite(timestamp)) return false;

      const clipFps =
        mediaInfo.current?.stats.video?.averagePacketRate || fpsRef.current || DEFAULT_FPS;
      const currentFrame =
        Number.isFinite(clipFps) && clipFps > 0
          ? Math.floor(video.currentTime * clipFps + 1e-4)
          : null;
      const thresholdFrames = Math.max(0, opts?.thresholdFrames ?? 0);
      if (
        currentFrame !== null &&
        Number.isFinite(currentFrame) &&
        Math.abs(currentFrame - targetFrame) <= thresholdFrames
      ) {
        if (opts?.strict) {
          clearPendingSeekTarget();
        }
        return false;
      }

      pendingSeekTargetRef.current = {
        frame: targetFrame,
        strict: opts?.strict ?? true,
      };

      seekLockStartedAtMsRef.current =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      seekLockSkippedFramesRef.current = 0;
    
      if (opts?.gatePlayback) {
        resumeGateFrameRef.current = targetFrame;
      }

      try {
        video.currentTime = timestamp;
        return true;
      } catch {
        clearPendingSeekTarget();
        if (opts?.gatePlayback) {
          resumeGateFrameRef.current = null;
        }
        return false;
      }
    },
    [clearPendingSeekTarget],
  );

  // Video frame callback: draw current video frame to canvas and trigger Konva redraw
  const onVideoFrameCallback = useCallback(
    (_now: number, _metadata: { mediaTime: number; presentedFrames: number }) => {
      pendingVideoFrameCallbackRef.current = null;
      inVideoFrameCallbackRef.current = true;
      const video = videoRef.current;
      const draw = drawWrappedCanvasRef.current;
      if (!video || !draw || !canvasRef.current || video.readyState < 2) {
        if (
          video &&
          isPlayingRef.current &&
          pendingVideoFrameCallbackRef.current === null
        ) {
          pendingVideoFrameCallbackRef.current =
            video.requestVideoFrameCallback(onVideoFrameCallback);
        }
        inVideoFrameCallbackRef.current = false;
        return;
      }
      if (!isPlayingRef.current) {
        inVideoFrameCallbackRef.current = false;
        return;
      }
      if (!allowPlaybackRef.current) {
        inVideoFrameCallbackRef.current = false;
        return;
      }
      try {
        draw(
          {
            canvas: video,
            timestamp: video.currentTime,
            duration: 0,
          },
          decoderMaskFrameRef.current,
        );
      } catch {
        // ignore draw errors (e.g. during seek)
      }
      if (
        isPlayingRef.current &&
        allowPlaybackRef.current &&
        pendingVideoFrameCallbackRef.current === null
      ) {
        pendingVideoFrameCallbackRef.current =
          video.requestVideoFrameCallback(onVideoFrameCallback);
      }
      inVideoFrameCallbackRef.current = false;
    },
    [],
  );

  // Setup video element and wire frame rendering to Konva
  useEffect(() => {
    if (!selectedAssetId || !displayWidth || !displayHeight) return;

    const info =
      getMediaInfoCached(selectedAssetId) ||
      (mediaInfoAssetIdRef.current === selectedAssetId ? mediaInfo.current : null);
    const asset = getAssetById(selectedAssetId);
    const url =
      (info?.video as any)?.input?.source?._url?.href ??
      info?.path ??
      asset?.path;
    if (!url || !info?.video) return;

    let video = videoRef.current;
    if (!video) {
      video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.setAttribute("playsinline", "true");
      videoRef.current = video;
    }
    const initialPlaybackRate = Math.max(0.1, speedRef.current || 1);
    try {
      video.defaultPlaybackRate = initialPlaybackRate;
      video.playbackRate = initialPlaybackRate;
    } catch {}

    const scheduleNextFrame = () => {
      if (!video) return;
      if (inVideoFrameCallbackRef.current) return;
      if (pendingVideoFrameCallbackRef.current !== null) return;
      pendingVideoFrameCallbackRef.current =
        video.requestVideoFrameCallback(onVideoFrameCallback);
    };

    const drawCurrentFrame = () => {
      const draw = drawWrappedCanvasRef.current;
      if (!video || !draw || !canvasRef.current || video.readyState < 2) return;
      try {
        draw(
          {
            canvas: video,
            timestamp: video.currentTime,
            duration: 0,
          },
          decoderMaskFrameRef.current,
        );
      } catch {}
    };

    const seekToFocusFrame = () => {
      const targetInfo = getTargetFrameInfoRef.current();
      if (!targetInfo || !video) return false;
      return seekVideoToTargetFrame(video, targetInfo, {
        strict: true,
        gatePlayback: isPlayingRef.current,
        thresholdFrames: 0,
      });
    };

    const onLoadedData = () => {
      const didSeek = seekToFocusFrame();
      if (!didSeek) {
        drawCurrentFrame();
      }
      if (isPlayingRef.current && didSeek) {
        pendingPlayAfterSeekRef.current = true;
      } else if (isPlayingRef.current) {
        pendingPlayAfterSeekRef.current = false;
        video?.play().catch(() => {});
        scheduleNextFrame();
      }
    };

    const onSeeked = () => {
      drawCurrentFrame();
      const pendingSeek = pendingSeekTargetRef.current;
      if (pendingSeek?.strict) {
        const clipFps =
          mediaInfo.current?.stats.video?.averagePacketRate || fpsRef.current || DEFAULT_FPS;
        const frameIdx =
          Number.isFinite(clipFps) && clipFps > 0
            ? Math.floor(video.currentTime * clipFps + 1e-4)
            : null;
        if (
          frameIdx !== null &&
          Number.isFinite(frameIdx) &&
          Math.abs(frameIdx - pendingSeek.frame) <= 2
        ) {
          clearPendingSeekTarget();
        }
      }
      if (isPlayingRef.current || pendingPlayAfterSeekRef.current) {
        pendingPlayAfterSeekRef.current = false;
        video?.play().catch(() => {});
        scheduleNextFrame();
      }
    };

    video.addEventListener("loadeddata", onLoadedData);
    video.addEventListener("seeked", onSeeked);

    video.src = url;
    video.load();
    if (video.readyState >= 2) {
      onLoadedData();
    }

    if (isPlayingRef.current) {
      video.play().catch(() => {});
      scheduleNextFrame();
    }

    return () => {
      video?.removeEventListener("loadeddata", onLoadedData);
      video?.removeEventListener("seeked", onSeeked);
      pendingPlayAfterSeekRef.current = false;
      clearPendingSeekTarget();
      clearPendingVideoFrameCallback();
      video?.pause();
      videoRef.current = null;
    };
  }, [
    selectedAssetId,
    displayWidth,
    displayHeight,
    getAssetById,
    clearPendingSeekTarget,
    clearPendingVideoFrameCallback,
    onVideoFrameCallback,
    seekVideoToTargetFrame,
  ]);

  // Seek video to current frame when paused and focusFrame changes
  useEffect(() => {
    if (isPlaying) return;
    const video = videoRef.current;
    const targetInfo = getTargetFrameInfo();
    if (!video || !targetInfo || video.readyState < 2) return;

    const clipFps =
      mediaInfo.current?.stats.video?.averagePacketRate || fpsRef.current || DEFAULT_FPS;
    const frameIdx =
      Number.isFinite(clipFps) && clipFps > 0
        ? Math.floor(video.currentTime * clipFps + 1e-4)
        : null;
    const frameDelta =
      frameIdx === null ? Infinity : Math.abs(frameIdx - targetInfo.targetFrame);
    const pendingSeek = pendingSeekTargetRef.current;
    if (
      video.seeking &&
      pendingSeek?.strict &&
      pendingSeek.frame === targetInfo.targetFrame
    ) {
      return;
    }
    if (!Number.isFinite(frameDelta) || frameDelta > 0) {
      seekVideoToTargetFrame(video, targetInfo, {
        strict: true,
        gatePlayback: false,
        thresholdFrames: 0,
      });
      return;
    }
    clearPendingSeekTarget();

    // If we're already at the target time while paused, browsers may not emit
    // another frame callback. Draw once explicitly to avoid initial black frames.
    if (lastDrawnFocusFrameRef.current !== focusFrame) {
      const draw = drawWrappedCanvasRef.current;
      if (!draw || !canvasRef.current) return;
      try {
        draw(
          {
            canvas: video,
            timestamp: video.currentTime,
            duration: 0,
          },
          decoderMaskFrameRef.current,
        );
      } catch {}
    }
  }, [isPlaying, focusFrame, getTargetFrameInfo, seekVideoToTargetFrame, clearPendingSeekTarget]);

  // Play/pause video based on timeline
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying && allowPlayback) {
      const targetInfo = getTargetFrameInfoRef.current();
      if (targetInfo && video.readyState >= 2) {
        const didSeek = seekVideoToTargetFrame(video, targetInfo, {
          strict: true,
          gatePlayback: true,
          thresholdFrames: 0,
        });
        if (didSeek) {
          pendingPlayAfterSeekRef.current = true;
          return;
        }
      }
      pendingPlayAfterSeekRef.current = false;
      video.play().catch(() => {});
      if (
        pendingVideoFrameCallbackRef.current === null &&
        !inVideoFrameCallbackRef.current
      ) {
        pendingVideoFrameCallbackRef.current =
          video.requestVideoFrameCallback(onVideoFrameCallback);
      }
    } else {
      pendingPlayAfterSeekRef.current = false;
      clearPendingVideoFrameCallback();
      video.pause();
    }
  }, [
    isPlaying,
    allowPlayback,
    onVideoFrameCallback,
    clearPendingVideoFrameCallback,
    seekVideoToTargetFrame,
  ]);

  // Keep the HTML video element playback speed aligned with clip speed.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const playbackRate = Math.max(0.1, speed || 1);
    try {
      video.defaultPlaybackRate = playbackRate;
      video.playbackRate = playbackRate;
    } catch {}
  }, [speed]);


  // If video is paused, reapply filters and applicators when they change
  useEffect(() => {
    if (!isPlaying && canvasRef.current && imageRef.current) {
      // If we have an original frame cached, use it for fast reapplication
      if (originalFrameRef.current) {
        const baseWidth = Math.max(1, originalFrameRef.current.width);
        const baseHeight = Math.max(1, originalFrameRef.current.height);
        const workingCanvas = ensureProcessingCanvas(baseWidth, baseHeight);
        const workingCtx = workingCanvas.getContext("2d");
        if (!workingCtx) return;

        // Start with the original unfiltered frame
        workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
        workingCtx.drawImage(originalFrameRef.current, 0, 0);

        // Apply masks before filters so masked pixels feed the rest of the pipeline
        const maskedCanvas =
          toolRef.current !== "mask"
            ? applyMaskRef.current(workingCanvas, maskFrameForCurrentFocus)
            : workingCanvas;
        let processedCanvas = maskedCanvas;

        // Apply filters to the clean frame
        processedCanvas = applyFilters(processedCanvas, filterParamsRef.current, {
          output: "passthrough",
        });

        // Apply applicators (filter clips from layers above)
        for (const applicator of applicatorsRef.current) {
          const result = applicator.apply(processedCanvas);
          if (result) processedCanvas = result;
        }

        presentCanvas(processedCanvas);
        imageRef.current.getLayer()?.batchDraw();
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
    clip?.colorTintColor,
    clip?.colorTintIntensity,
    clip?.scanLines,
    clip?.chromaticAberration,
    clip?.interlace,
    clip?.pixelate,
    clip?.jitter,
    isPlaying,
    applyFilters,
    applicators,
    applicators.length,
    applyMask,
    maskFrameForCurrentFocus,
    ensureProcessingCanvas,
    presentCanvas,
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
    lastDrawnFocusFrameRef.current = null;
    resumeGateFrameRef.current = null;
    clearPendingSeekTarget();
    pendingPlayAfterSeekRef.current = false;
    clearPendingVideoFrameCallback();
    // @ts-ignore
    iteratorRef.current?.return?.();
    iteratorRef.current = null;
  }, [clipId, overrideClip, clearPendingVideoFrameCallback, clearPendingSeekTarget]);


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
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.v25 && (
                <Line
                  listening={false}
                  points={[rectWidth * 0.25, 0, rectWidth * 0.25, rectHeight]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.v75 && (
                <Line
                  listening={false}
                  points={[rectWidth * 0.75, 0, rectWidth * 0.75, rectHeight]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.hCenter && (
                <Line
                  listening={false}
                  points={[0, rectHeight / 2, rectWidth, rectHeight / 2]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.h25 && (
                <Line
                  listening={false}
                  points={[0, rectHeight * 0.25, rectWidth, rectHeight * 0.25]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.h75 && (
                <Line
                  listening={false}
                  points={[0, rectHeight * 0.75, rectWidth, rectHeight * 0.75]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.left && (
                <Line
                  listening={false}
                  points={[0, 0, 0, rectHeight]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.right && (
                <Line
                  listening={false}
                  points={[rectWidth, 0, rectWidth, rectHeight]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.top && (
                <Line
                  listening={false}
                  points={[0, 0, rectWidth, 0]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
              {guides.bottom && (
                <Line
                  listening={false}
                  points={[0, rectHeight, rectWidth, rectHeight]}
                  stroke={SELECTION_STROKE_COLOR}
                  strokeWidth={1}
                  dash={[6, 4]}
                />
              )}
            </React.Fragment>
          )}
      </Group>
      <Transformer
        borderStroke={SELECTION_STROKE_COLOR}
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
