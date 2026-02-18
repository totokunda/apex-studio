import { MediaInfo, VideoClipProps } from "@/lib/types";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { getMediaInfoCached } from "@/lib/media/utils";
import { useControlsStore } from "@/lib/control";
import Konva from "konva";
import { useViewportStore } from "@/lib/viewport";
import { DEFAULT_FPS } from "@/lib/settings";
import { useClipStore } from "@/lib/clip";
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import { BaseClipApplicator } from "./apply/base";
import _ from "lodash";
import { useWebGLMask } from "../mask/useWebGLMask";
import { useInputControlsStore } from "@/lib/inputControl";
import { generatePosterCanvas } from "@/lib/media/timeline";
import SharedClipCanvasSurface, {
  getAspectFitSize,
} from "./shared/SharedClipCanvasSurface";

import {
  iterate as iterateDecodedFrames,
  loadFile,
  seek as seekFrame,
} from "@app/decoder";

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
    const actualStartFrame = Math.floor(
      (idealStartFrame / projectFps) * clipFps + 1e-4,
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
  const imageRef = useRef<Konva.Image>(null);
  const drawTokenRef = useRef(0);
  const posterRequestRef = useRef(0);
  const lastPosterKeyRef = useRef<string | null>(null);
  const lastSeekFrameRef = useRef<number>(0);
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
  const tool = useViewportStore((s) => s.tool);
  const clipTransform = overrideClip
    ? overrideClip.transform
    : useClipStore((s) => s.getClipTransform(clipId));
  const srcFps = useControlsStore((s) => s.fps);
  const isAccurateSeekNeeded = useControlsStore((s) => s.isAccurateSeekNeeded);

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

  const decoderLoadedAssetIdRef = useRef<string | null>(null);
  const isPlayingFromControls = useControlsStore((s) => s.isPlaying);
  const isPlayingFromInputs = useInputControlsStore((s) =>
    s.getIsPlaying(inputId ?? ""),
  );

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

  const prevFocusFrameWhilePlayingRef = useRef<number | null>(null);
  const fpsFromControls = useControlsStore((s) => s.fps);
  const fpsFromInputs = useInputControlsStore((s) => s.getFps(inputId ?? ""));
  const fps = useInputScopedControls ? fpsFromInputs : fpsFromControls;
  const currentStartFrameRef = useRef<number>(0);
  const lastRenderedFrameRef = useRef<number>(-1);
  const lastDrawnFocusFrameRef = useRef<number | null>(null);
  const skipDrawRef = useRef(false);
  const suppressSeekFramesRef = useRef(false);
  const resumeGateFrameRef = useRef<number | null>(null);
  const pendingSeekTargetRef = useRef<{ frame: number; strict: boolean } | null>(
    null,
  );

  const fpsRef = useRef(fps);
  useEffect(() => {
    fpsRef.current = fps;
  }, [fps]);

  const ensureDecoderLoaded = useCallback(
    async (targetAssetId: string): Promise<boolean> => {
      if (!targetAssetId) return false;
      if (decoderLoadedAssetIdRef.current === targetAssetId) return true;

      const asset = getAssetById(targetAssetId);
      const pathOrUrl =
        asset?.path ??
        (targetAssetId === assetId ? mediaInfo.current?.path : undefined);
      if (!pathOrUrl) return false;

      try {
        await loadFile(pathOrUrl);
        decoderLoadedAssetIdRef.current = targetAssetId;
        return true;
      } catch (e) {
        console.warn("[decoder] loadFile failed:", e);
        return false;
      }
    },
    [assetId, getAssetById],
  );

  useEffect(() => {
    let cancelled = false;
    const configure = async () => {
      if (!selectedAssetId) return;
      const ok = await ensureDecoderLoaded(selectedAssetId);
      if (cancelled || !ok) return;
      if (lastRenderedFrameRef.current < 0) return;
      lastRenderedFrameRef.current = -1;
    };
    void configure();
    return () => {
      cancelled = true;
    };
  }, [selectedAssetId, ensureDecoderLoaded]);


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
    if (lastSelectedAssetIdRef.current === selectedAssetId) return;
    lastSelectedAssetIdRef.current = selectedAssetId;
    // Force redraw on source switch: reset last rendered frame and clear cached original frame
    lastRenderedFrameRef.current = -1;
    lastPosterKeyRef.current = null;
    originalFrameRef.current = null;
    lastDrawnFocusFrameRef.current = null;
    resumeGateFrameRef.current = null;
    pendingSeekTargetRef.current = null;
    processingCanvasRef.current = null;
    decoderLoadedAssetIdRef.current = null;
    let info = getMediaInfoCached(selectedAssetId);
    if (!info) {
      return;
    } else {
      setMediaInfoAndBump(info);
      // Update the "current" aspect-fit size for drawWrappedCanvas immediately so the
      // very first frame of the new asset can't render into a stale-sized canvas.
      displaySizeRef.current = getAspectFitSize(
        info.video?.displayWidth || 0,
        info.video?.displayHeight || 0,
        rectWidth,
        rectHeight,
      );
      
      // Have cached info; force immediate redraw
      lastRenderedFrameRef.current = -1;
    }
  }, [selectedAssetId, rectWidth, rectHeight, setMediaInfoAndBump]);


  // Compute aspect-fit display size and offsets within the preview rect
  const { displayWidth, displayHeight, offsetX, offsetY } = useMemo(() => {
    return getAspectFitSize(
      mediaInfo.current?.video?.displayWidth || 0,
      mediaInfo.current?.video?.displayHeight || 0,
      rectWidth,
      rectHeight,
    );
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
    (
      wc: {
        canvas: HTMLCanvasElement | OffscreenCanvas | VideoFrame;
        timestamp: number;
        duration: number;
      },
      maskFrame?: number,
      opts?: { recordFrame?: boolean },
    ) => {

      
      let canvas = canvasRef.current;


      if (!canvas) return;

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
        !isPlayingRef.current &&
        pendingSeek &&
        pendingSeek.strict &&
        frameIdx !== null &&
        Math.abs(frameIdx - pendingSeek.frame) > 2
      ) {
        pendingSeekTargetRef.current = { ...pendingSeek, strict: false };
      }
      const resumeGate = resumeGateFrameRef.current;
      if (
        isPlayingRef.current &&
        typeof resumeGate === "number" &&
        frameIdx !== null &&
        frameIdx < resumeGate
      ) {
        return;
      }

      skipDrawRef.current = false;

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
      let processedCanvas = maskedCanvas;

      // Apply WebGL filters for better performance (fast enough for real-time playback)
      // Use passthrough mode to keep filter output on GPU/canvas chain.
      processedCanvas = applyFiltersRef.current(
        processedCanvas,
        filterParamsRef.current,
        { output: "passthrough" },
      );

      for (const applicator of applicatorsRef.current) {
        const result = applicator.apply(processedCanvas);
        if (result) processedCanvas = result;
      }

      // Always draw the final processed result back to display canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(processedCanvas, 0, 0, canvas.width, canvas.height);
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
          pendingSeekTargetRef.current = null;
        }
        lastDrawnFocusFrameRef.current = focusFrameRef.current;
      }
      imageRef.current?.getLayer()?.batchDraw?.();
    },
    [ensureProcessingCanvas, displayWidth, displayHeight, ],
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
        const fallback = getAspectFitSize(
          info.video?.displayWidth || 0,
          info.video?.displayHeight || 0,
          rectWidth,
          rectHeight,
        );
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
      pendingSeekTargetRef.current = null;
      drawWrappedCanvas(
        {
          canvas: fallbackCanvas,
          timestamp: targetInfo?.timestamp ?? 0,
          duration: 0,
        },
        maskFrame,
        { recordFrame: false },
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

  const pausedSeekTokenRef = useRef(0);

  const performWorkerSeek = useCallback(
    async ({
      targetAssetId,
      timestamp,
      forceAccurate,
      token,
    }: {
      targetAssetId: string;
      timestamp: number;
      forceAccurate: boolean;
      token: number;
    }) => {
      if (token !== pausedSeekTokenRef.current) return;
      const ready = await ensureDecoderLoaded(targetAssetId);
      if (token !== pausedSeekTokenRef.current) return;
      if (!ready) {
        pendingSeekTargetRef.current = null;
        if (forceAccurate) {
          void renderPosterFallback({ force: true });
        }
        return;
      }

      const logicalId = makeDecoderId(targetAssetId);
      const decoded = await seekFrame(timestamp, forceAccurate);
      console.log("decoded", decoded);
      if (token !== pausedSeekTokenRef.current) {
        if (decoded?.frame) decoded.frame.close();
        return;
      }
      if (!decoded?.frame) {
        if (!forceAccurate) {
          const accurate = await seekFrame(timestamp, true);
          if (token !== pausedSeekTokenRef.current) {
            if (accurate?.frame) accurate.frame.close();
            return;
          }
          if (accurate?.frame) {
            try {
              drawWrappedCanvas(
                {
                  canvas: accurate.frame,
                  timestamp: accurate.timestamp,
                  duration: accurate.duration,
                },
                decoderMaskFrameRef.current,
              );
            } finally {
              accurate.frame.close();
            }
            activeDecoderAssetIdRef.current = logicalId;
            return;
          }
        }
        pendingSeekTargetRef.current = null;
        if (forceAccurate) {
          void renderPosterFallback({ force: true });
        }
        return;
      }

      try {
        drawWrappedCanvas(
          {
            canvas: decoded.frame,
            timestamp: decoded.timestamp,
            duration: decoded.duration,
          },
          decoderMaskFrameRef.current,
        );
      } finally {
        decoded.frame.close();
      }
      activeDecoderAssetIdRef.current = logicalId;
    },
    [drawWrappedCanvas, ensureDecoderLoaded, makeDecoderId, renderPosterFallback],
  );

  const seekToCurrentFrame = useCallback(
    async (isAccurateSeekNeededInput: boolean = false) => {


      // NOTE: Do NOT use `useInputControlsStore.getState()` here:
      // it reads the global fallback store (wrong clip scope) and will return false
      // during input playback, causing us to seek every frame.
      // During active playback, rely on the iterate loop unless this is an explicit jump.


    const info = getTargetFrameInfo();

    const nextTargetFrame = info?.targetFrame ?? 0;
    const previousTargetFrame = lastSeekFrameRef.current;
    lastSeekFrameRef.current = nextTargetFrame;
    const clipFpsForJump =
      mediaInfo.current?.stats.video?.averagePacketRate || fpsRef.current || DEFAULT_FPS;
    const largeJumpThreshold = Math.max(30, Math.floor(clipFpsForJump * 1.25));
    if (Math.abs(nextTargetFrame - previousTargetFrame) >= largeJumpThreshold) {
      isAccurateSeekNeededInput = true;
    }

    if (isPlayingRef.current && !isAccurateSeekNeededInput) {
      return;
    }
 

    if (!info) {
      void renderPosterFallback();
      return;
    }

    const { timestamp, targetFrame } = info;
    pendingSeekTargetRef.current = {
      frame: targetFrame,
      strict: isAccurateSeekNeededInput,
    };


    // Update the mask frame ref immediately before seeking to ensure sync
    decoderMaskFrameRef.current = maskFrameForCurrentFocus;

    // Cancel any ongoing paused seek operations (do not interfere with live decode token)
    drawTokenRef.current++;
    if (isAccurateSeekNeeded) {
      isAccurateSeekNeededInput = true;
    }

    try {
      const targetAssetId = selectedAssetId;
      if (!targetAssetId) return;
      const token = ++pausedSeekTokenRef.current;
      await performWorkerSeek({
        targetAssetId,
        timestamp,
        forceAccurate: isAccurateSeekNeededInput,
        token,
      });

    } catch (e) {
      console.warn("[video] seek failed", e);
      pendingSeekTargetRef.current = null;
      void renderPosterFallback({ force: true });
    }
    },
    [
      getTargetFrameInfo,
      renderPosterFallback,
      maskFrameForCurrentFocus,
      isAccurateSeekNeeded,
      selectedAssetId,
      performWorkerSeek,
    ],
  );



  useEffect(() => {
    void seekToCurrentFrame();
  }, [
    focusFrame,
    currentLocalFrameOverride,
    focusFrameOverride,
    selectedAssetId,
    isPlaying,
    seekToCurrentFrame,
  ]);

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
    pausedSeekTokenRef.current += 1;

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
    const lastFrame = lastRenderedFrameRef.current;
    if (
      Number.isFinite(lastFrame) &&
      lastFrame >= 0 &&
      Math.abs(lastFrame - startIdx) <= 1
    ) {
      resumeGateFrameRef.current = lastFrame;
    } else {
      resumeGateFrameRef.current = null;
    }
    skipDrawRef.current = false;
    pendingSeekTargetRef.current = null;

    const myToken = ++drawTokenRef.current;

    const activeAssetId = selectedAssetId;
    if (!activeAssetId) return;

    const logicalId = makeDecoderId(activeAssetId);

    const checkCancel = () => {
        if (!offscreenFast && myToken !== drawTokenRef.current) return false;
        if (!isPlaying) return false;
        return true;
    };

    try {
      if (!checkCancel()) return;
      const ready = await ensureDecoderLoaded(activeAssetId);
      if (!ready || !checkCancel()) {
        void renderPosterFallback({ force: true });
        return;
      }
      activeDecoderAssetIdRef.current = logicalId;

      await iterateDecodedFrames(
        startTime,
        endTime,
        async ({ frame, timestamp, duration }) => {
          try {
          if (!checkCancel()) return;

          let sampleIdx = Number.isFinite(timestamp)
            ? Math.floor(timestamp * clipFps + 1e-4)
            : lastRenderedFrameRef.current + 1;
          skipDrawRef.current = false;
          const resumeGate = resumeGateFrameRef.current;
          if (
            !offscreenFast &&
            typeof resumeGate === "number" &&
            Number.isFinite(resumeGate) &&
            sampleIdx < resumeGate
          ) {
            skipDrawRef.current = true;
            return;
          }

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
              skipDrawRef.current = true;
              return;
            }
            // If we're ahead of the timeline, wait until the timeline catches up.
            // On Windows the DWM compositor can delay RAF callbacks by one or
            // more vsync intervals (16-33ms), causing the decoder to stall
            // indefinitely when a frame arrives just before a compositor pause.
            // We use a deadline-aware loop: if the RAF hasn't fired within
            // 1.5× the expected frame interval, we break out anyway so playback
            // never stalls under compositor jitter.
            if (sampleIdx > (localFocus = computeLocalFocusMedia())) {
              const frameBudgetMs = (1000 / Math.max(1, projectFps)) * 1.5;
              const deadline = performance.now() + frameBudgetMs;
              await new Promise<void>((resolve) => {
                const check = () => {
                  if (
                    !checkCancel() ||
                    sampleIdx <= computeLocalFocusMedia() ||
                    performance.now() >= deadline
                  ) {
                    resolve();
                  } else {
                    requestAnimationFrame(check);
                  }
                };
                requestAnimationFrame(check);
              });
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
          drawWrappedCanvas(
            {
              canvas: frame,
              timestamp,
              duration,
            },
            maskFrame,
          );
          } finally {
            frame.close();
          }
        },
        checkCancel,
      );

    } catch (e) {
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
    ensureDecoderLoaded,
  ]);

  useEffect(() => {
    if (isPlaying) {
      void startRendering();
    }
    return () => {
      skipDrawRef.current = false;
      suppressSeekFramesRef.current = false;
      resumeGateFrameRef.current = null;
      pendingSeekTargetRef.current = null;
      drawTokenRef.current++;
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
      resumeGateFrameRef.current = null;
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
    pendingSeekTargetRef.current = null;
    decoderLoadedAssetIdRef.current = null;
  }, [clipId, overrideClip]);


  // If we become visible after being hidden (prewarmed), force a redraw so the
  // already-decoded backing canvas is displayed immediately.
  useEffect(() => {
    if (hidden) return;
    try {
      imageRef.current?.getLayer()?.batchDraw?.();
    } catch {}
  }, [hidden]);

  return (
    <SharedClipCanvasSurface
      clipId={clipId}
      rectWidth={rectWidth}
      rectHeight={rectHeight}
      displayWidth={displayWidth}
      displayHeight={displayHeight}
      offsetX={offsetX}
      offsetY={offsetY}
      clipTransform={clipTransform}
      canvasRef={canvasRef}
      imageRef={imageRef}
      imageSource={imageSource || canvasRef.current}
      overlap={overlap}
      inputMode={inputMode}
      isInFrame={isInFrame}
      hidden={hidden}
      overrideClip={!!overrideClip}
    />
  );
};

export default VideoPreview;
