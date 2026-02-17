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
import { WrappedCanvas } from "mediabunny";
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
  fileURLToPath,
  previewMseAbort,
  previewMseRead,
  previewMseStart,
} from "@app/preload";

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
  hidden = false,
  decoderKey,
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
  const isAccurateSeekNeededFromControls = useControlsStore(
    (s) => s.isAccurateSeekNeeded,
  );
  const isAccurateSeekNeeded = useInputScopedControls
    ? false
    : isAccurateSeekNeededFromControls;

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

  const activeAssetPath =
    (getAssetById(selectedAssetId || assetId)?.path as string) || "";
  const mediaSessionScope = useMemo(() => {
    const scopeClipKey = decoderKey ?? clipId;
    const inputScope = inputMode ? `input:${inputId ?? ""}` : "timeline";
    return `${selectedAssetId || assetId}:${scopeClipKey}:${inputScope}`;
  }, [assetId, clipId, decoderKey, inputId, inputMode, selectedAssetId]);
  const filePath =
    activeAssetPath.startsWith("file:") || activeAssetPath.startsWith("file://")
      ? fileURLToPath(activeAssetPath)
      : activeAssetPath;


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
  const lastRenderedTimestampRef = useRef<number | null>(null);
  const playbackResyncCooldownUntilRef = useRef<number>(0);
  const wasPlayingRef = useRef<boolean>(false);
  const lastDrawnFocusFrameRef = useRef<number | null>(null);
  const skipDrawRef = useRef(false);
  const suppressSeekFramesRef = useRef(false);
  const resumeGateFrameRef = useRef<number | null>(null);
  const pendingSeekTargetRef = useRef<{ frame: number; strict: boolean } | null>(
    null,
  );
  const lastSeekFrameRef = useRef<number>(-1);
  const fpsRef = useRef(fps);
  const playbackVideoRef = useRef<HTMLVideoElement | null>(null);
  const mediaSourceSessionIdRef = useRef<string | null>(null);
  const mediaSourceAbortRef = useRef<AbortController | null>(null);
  const mediaSourceObjectUrlRef = useRef<string | null>(null);
  const playbackRafRef = useRef<number | null>(null);
  const mediaSourceFailureCountRef = useRef(0);
  const mediaSourceBackoffUntilRef = useRef(0);
  const mediaSourceDisabledUntilRef = useRef(0);
  const mediaSourceStartInFlightRef = useRef(false);
  const mediaSourceRetryTimerRef = useRef<number | null>(null);
  const lastPlaybackRestartRequestAtRef = useRef(0);
  const playingLatchRef = useRef(false);
  useEffect(() => {
    fpsRef.current = fps;
  }, [fps]);


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

  const ensurePlaybackVideo = useCallback(() => {
    if (playbackVideoRef.current) return playbackVideoRef.current;
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "true");
    video.preload = "auto";
    playbackVideoRef.current = video;
    return video;
  }, []);

  const cleanupMediaSourcePlayback = useCallback(
    (opts?: { keepVideo?: boolean }) => {
      if (playbackRafRef.current != null) {
        cancelAnimationFrame(playbackRafRef.current);
        playbackRafRef.current = null;
      }
      if (mediaSourceRetryTimerRef.current != null) {
        window.clearTimeout(mediaSourceRetryTimerRef.current);
        mediaSourceRetryTimerRef.current = null;
      }

      mediaSourceAbortRef.current?.abort();
      mediaSourceAbortRef.current = null;

      const sessionId = mediaSourceSessionIdRef.current;
      mediaSourceSessionIdRef.current = null;
      if (sessionId) {
        void previewMseAbort(sessionId).catch(() => {});
      }

      const video = playbackVideoRef.current;
      if (video) {
        try {
          video.pause();
        } catch {}
        try {
          video.removeAttribute("src");
          video.load();
        } catch {}
      }

      const objectUrl = mediaSourceObjectUrlRef.current;
      if (objectUrl) {
        try {
          URL.revokeObjectURL(objectUrl);
        } catch {}
        mediaSourceObjectUrlRef.current = null;
      }

      if (!opts?.keepVideo) {
        playbackVideoRef.current = null;
      }
      playingLatchRef.current = false;
    },
    [],
  );

  const waitForMediaVideoReady = useCallback(
    (video: HTMLVideoElement, signal: AbortSignal) =>
      new Promise<void>((resolve, reject) => {
        if (video.readyState >= 2) {
          resolve();
          return;
        }

        const onCanPlay = () => {
          cleanup();
          resolve();
        };
        const onError = () => {
          cleanup();
          reject(new Error("MediaSource video failed to load"));
        };
        const onAbort = () => {
          cleanup();
          reject(new Error("MediaSource start aborted"));
        };
        const cleanup = () => {
          video.removeEventListener("canplay", onCanPlay);
          video.removeEventListener("error", onError);
          signal.removeEventListener("abort", onAbort);
        };

        video.addEventListener("canplay", onCanPlay, { once: true });
        video.addEventListener("error", onError, { once: true });
        signal.addEventListener("abort", onAbort, { once: true });
      }),
    [],
  );

  const startMediaSourceStream = useCallback(
    async (
      seekTo: number,
      opts?: { fps?: number; size?: number; mode?: "seek" | "playback" },
    ) => {
      if (!filePath) return null;

      const nowMs = () =>
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();

      const isSeekMode = opts?.mode === "seek";
      let now = nowMs();
      if (!isSeekMode && now < mediaSourceBackoffUntilRef.current) return null;
      if (!isSeekMode && now < mediaSourceDisabledUntilRef.current) return null;

      if (mediaSourceStartInFlightRef.current) {
        if (!isSeekMode) return null;
        const waitStartedAt = now;
        while (mediaSourceStartInFlightRef.current) {
          await new Promise((resolve) => setTimeout(resolve, 12));
          now = nowMs();
          if (now - waitStartedAt > 350) return null;
        }
      }

      mediaSourceStartInFlightRef.current = true;
      try {
        cleanupMediaSourcePlayback({ keepVideo: true });
        const video = ensurePlaybackVideo();
        const abortController = new AbortController();
        mediaSourceAbortRef.current = abortController;

        const startResult = await previewMseStart({
          path: filePath,
          seekTo,
          fps: opts?.fps,
          size: opts?.size,
          scope: mediaSessionScope,
        });
        const startError =
          typeof startResult?.error === "string" ? startResult.error : null;
        if (startError) {
          throw new Error(startError);
        }

        const sessionId =
          typeof startResult?.sessionId === "string"
            ? startResult.sessionId
            : undefined;
        const mimeCodec =
          typeof startResult?.mimeCodec === "string"
            ? startResult.mimeCodec
            : undefined;
        if (!sessionId || !mimeCodec) {
          throw new Error("Failed to start MediaSource session");
        }
        if (!MediaSource.isTypeSupported(mimeCodec)) {
          throw new Error(`Unsupported MediaSource codec: ${mimeCodec}`);
        }
        if (abortController.signal.aborted) {
          await previewMseAbort(sessionId).catch(() => {});
          return null;
        }
        mediaSourceSessionIdRef.current = sessionId;

        const mediaSource = new MediaSource();
        const objectUrl = URL.createObjectURL(mediaSource);
        mediaSourceObjectUrlRef.current = objectUrl;
        video.src = objectUrl;

        await new Promise<void>((resolve, reject) => {
          const onOpen = () => {
            cleanup();
            resolve();
          };
          const onError = () => {
            cleanup();
            reject(new Error("MediaSource failed to open"));
          };
          const onAbort = () => {
            cleanup();
            reject(new Error("MediaSource start aborted"));
          };
          const cleanup = () => {
            mediaSource.removeEventListener("sourceopen", onOpen);
            mediaSource.removeEventListener("error", onError);
            abortController.signal.removeEventListener("abort", onAbort);
          };
          mediaSource.addEventListener("sourceopen", onOpen, { once: true });
          mediaSource.addEventListener("error", onError, { once: true });
          abortController.signal.addEventListener("abort", onAbort, { once: true });
        });

        if (abortController.signal.aborted) return null;

        const sourceBuffer = mediaSource.addSourceBuffer(mimeCodec);
        sourceBuffer.timestampOffset =
          Math.max(0, seekTo) -
          1 / Math.max(1, opts?.fps || fpsRef.current || DEFAULT_FPS);

        let streamEnded = false;
        let readInFlight = false;
        let pendingChunk: Uint8Array | null = null;
        let processChunkTimeout: number | null = null;
        const bufferThrottleSec = 10;
        const bufferMaxSec = 70;
        const bufferKeepSec = 30;

        const clearProcessChunkTimeout = () => {
          if (processChunkTimeout != null) {
            window.clearTimeout(processChunkTimeout);
            processChunkTimeout = null;
          }
        };

        const getBufferedRange = () => {
          if (sourceBuffer.buffered.length === 0) return null;
          const endIdx = sourceBuffer.buffered.length - 1;
          return {
            start: sourceBuffer.buffered.start(0),
            end: sourceBuffer.buffered.end(endIdx),
          };
        };

        const maybeTrimBuffer = () => {
          if (abortController.signal.aborted) return false;
          if (sourceBuffer.updating) return false;
          const range = getBufferedRange();
          if (!range) return false;
          const bufferedDuration = range.end - range.start;
          if (bufferedDuration <= bufferMaxSec) return false;
          const removeTo = Math.min(
            range.end - bufferKeepSec,
            Math.max(range.start, video.currentTime - 1),
          );
          if (!(removeTo > range.start + 0.5)) return false;
          try {
            sourceBuffer.remove(0, removeTo);
            return true;
          } catch {
            return false;
          }
        };

        const maybeEndStream = () => {
          if (!streamEnded) return;
          if (abortController.signal.aborted) return;
          if (mediaSource.readyState !== "open") return;
          if (sourceBuffer.updating) return;
          try {
            mediaSource.endOfStream();
          } catch {}
        };

        const scheduleProcessChunk = (delayMs = 0) => {
          if (abortController.signal.aborted || streamEnded) return;
          clearProcessChunkTimeout();
          processChunkTimeout = window.setTimeout(() => {
            processChunkTimeout = null;
            void processChunk();
          }, delayMs);
        };

        const processChunk = async () => {
          if (abortController.signal.aborted || streamEnded || readInFlight) return;
          if (sourceBuffer.updating) return;
          if (maybeTrimBuffer()) return;

          const range = getBufferedRange();
          if (range) {
            const bufferAhead = range.end - video.currentTime;
            if (bufferAhead > bufferThrottleSec) {
              scheduleProcessChunk(120);
              return;
            }
          }

          let chunk = pendingChunk;
          pendingChunk = null;
          if (!chunk) {
            readInFlight = true;
            try {
              chunk = await previewMseRead(sessionId);
            } finally {
              readInFlight = false;
            }
          }

          if (abortController.signal.aborted) return;
          if (!chunk || chunk.byteLength === 0) {
            streamEnded = true;
            maybeEndStream();
            return;
          }

          try {
            const chunkBuffer = chunk.buffer.slice(
              chunk.byteOffset,
              chunk.byteOffset + chunk.byteLength,
            ) as ArrayBuffer;
            sourceBuffer.appendBuffer(chunkBuffer);
          } catch {
            pendingChunk = chunk;
            if (maybeTrimBuffer()) return;
            scheduleProcessChunk(80);
          }
        };

        const onUpdateEnd = () => {
          if (maybeTrimBuffer()) return;
          maybeEndStream();
          if (abortController.signal.aborted || streamEnded) return;
          scheduleProcessChunk();
        };
        const onSourceBufferError = () => {
          if (abortController.signal.aborted || streamEnded) return;
          scheduleProcessChunk(250);
        };

        sourceBuffer.addEventListener("updateend", onUpdateEnd);
        sourceBuffer.addEventListener("error", onSourceBufferError);
        abortController.signal.addEventListener(
          "abort",
          () => {
            sourceBuffer.removeEventListener("updateend", onUpdateEnd);
            sourceBuffer.removeEventListener("error", onSourceBufferError);
            clearProcessChunkTimeout();
          },
          { once: true },
        );
        scheduleProcessChunk();

        await waitForMediaVideoReady(video, abortController.signal);
        mediaSourceFailureCountRef.current = 0;
        mediaSourceBackoffUntilRef.current = 0;
        mediaSourceDisabledUntilRef.current = 0;
        return video;
      } catch (err) {
        mediaSourceFailureCountRef.current += 1;
        const penaltyMs = Math.min(
          isSeekMode ? 4000 : 30000,
          (isSeekMode ? 200 : 600) *
            Math.pow(2, Math.min(5, mediaSourceFailureCountRef.current)),
        );
        const stamp = nowMs();
        mediaSourceBackoffUntilRef.current = stamp + penaltyMs;
        if (!isSeekMode && mediaSourceFailureCountRef.current >= 4) {
          mediaSourceDisabledUntilRef.current = stamp + 60_000;
        }
        if (mediaSourceFailureCountRef.current <= 2) {
          console.warn(
            "[video] MediaSource start failed",
            err instanceof Error ? err.message : err,
          );
        }
        cleanupMediaSourcePlayback({ keepVideo: true });
        return null;
      } finally {
        mediaSourceStartInFlightRef.current = false;
      }
    },
    [
      cleanupMediaSourcePlayback,
      ensurePlaybackVideo,
      filePath,
      mediaSessionScope,
      waitForMediaVideoReady,
    ],
  );

  const getMediaSourcePreviewParams = useCallback(() => {
    const projectFps = Number(fpsRef.current || DEFAULT_FPS);
    const targetFps = Number.isFinite(projectFps)
      ? Math.max(15, Math.min(30, Math.round(projectFps)))
      : 24;
    const largestSide = Math.round(Math.max(displayWidth || 0, displayHeight || 0));
    const targetSize =
      Number.isFinite(largestSide) && largestSide > 0
        ? Math.max(240, Math.min(800, largestSide))
        : undefined;
    return { targetFps, targetSize };
  }, [displayWidth, displayHeight]);

  useEffect(() => {
    return () => {
      cleanupMediaSourcePlayback();
    };
  }, [cleanupMediaSourcePlayback]);

  const previousMediaScopeRef = useRef(mediaSessionScope);
  useEffect(() => {
    if (previousMediaScopeRef.current === mediaSessionScope) return;
    previousMediaScopeRef.current = mediaSessionScope;
    cleanupMediaSourcePlayback({ keepVideo: true });
  }, [cleanupMediaSourcePlayback, mediaSessionScope]);

  useEffect(() => {
    const video = playbackVideoRef.current;
    if (!video) return;
    try {
      video.playbackRate = Math.max(0.1, speed);
    } catch {
      // ignore rate update errors
    }
  }, [speed]);

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
        return;
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
      if (Number.isFinite(wc.timestamp)) {
        lastRenderedTimestampRef.current = wc.timestamp;
      }
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
  const getTargetFrameInfoRef = useRef(getTargetFrameInfo);
  useEffect(() => {
    getTargetFrameInfoRef.current = getTargetFrameInfo;
  }, [getTargetFrameInfo]);

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

  const computeMaskFrameForFocus = useCallback(
    (focusFrameValue: number) => {
      const speedFactor = Math.max(0.1, speed);
      if (clip) {
        if (inputMode) {
          const local = Math.max(0, focusFrameValue + (trimStart || 0));
          return Math.max(0, Math.floor(local * speedFactor));
        }
        const isUsingPreprocessorSrc = selectedAssetId !== assetId;
        const baseLocal = Math.max(0, focusFrameValue - startFrameUsed);
        const derivedLocal = isUsingPreprocessorSrc
          ? Math.max(0, baseLocal - Math.max(0, frameOffset))
          : Math.max(0, baseLocal + (trimStart || 0));
        return Math.max(0, Math.floor(derivedLocal * speedFactor));
      }
      const local = Math.max(0, focusFrameValue - startFrameUsed + (trimStart || 0));
      return Math.max(0, Math.floor(local * speedFactor));
    },
    [
      speed,
      clip,
      inputMode,
      trimStart,
      selectedAssetId,
      assetId,
      startFrameUsed,
      frameOffset,
    ],
  );

  const seekToCurrentFrame = useCallback(
    async (isAccurateSeekNeededInput: boolean = false) => {
      if (isPlaying) return;
      if (isPlayingRef.current) return;
      if (hidden || !isInFrame) return;

      const info = getTargetFrameInfo();
      if (
        Math.abs(lastSeekFrameRef.current - (info?.targetFrame ?? 0)) > 8
      ) {
        isAccurateSeekNeededInput = true;
      }
      lastSeekFrameRef.current = info?.targetFrame ?? 0;

      if (!info) {
        void renderPosterFallback();
        return;
      }

      const focusFrameValue = focusFrameRef.current;
      if (
        !isAccurateSeekNeededInput &&
        originalFrameRef.current &&
        lastDrawnFocusFrameRef.current === focusFrameValue
      ) {
        return;
      }

      const { timestamp, targetFrame } = info;
      pendingSeekTargetRef.current = {
        frame: targetFrame,
        strict: isAccurateSeekNeededInput,
      };
      decoderMaskFrameRef.current = maskFrameForCurrentFocus;

      drawTokenRef.current++;
      if (isAccurateSeekNeeded) {
        isAccurateSeekNeededInput = true;
      }

      try {
        const clipFps =
          mediaInfo.current?.stats.video?.averagePacketRate || fps || DEFAULT_FPS;
        const { targetFps, targetSize } = getMediaSourcePreviewParams();
        const video = await startMediaSourceStream(timestamp, {
          fps: targetFps,
          size: targetSize,
          mode: "seek",
        });
        if (!video) {
          pendingSeekTargetRef.current = null;
          void renderPosterFallback({ force: true });
          return;
        }

        if (
          Number.isFinite(video.currentTime) &&
          Math.abs(video.currentTime - timestamp) > 1 / Math.max(1, clipFps)
        ) {
          await new Promise<void>((resolve) => {
            const onSeeked = () => {
              cleanup();
              resolve();
            };
            const onTimeout = () => {
              cleanup();
              resolve();
            };
            const cleanup = () => {
              video.removeEventListener("seeked", onSeeked);
              clearTimeout(timeout);
            };
            const timeout = setTimeout(onTimeout, 200);
            video.addEventListener("seeked", onSeeked, { once: true });
            try {
              video.currentTime = timestamp;
            } catch {
              cleanup();
              resolve();
            }
          });
        }

        try {
          video.pause();
        } catch {}

        const maskFrame = computeMaskFrameForFocus(focusFrameRef.current);
        decoderMaskFrameRef.current = maskFrame;
        drawWrappedCanvas(
          {
            canvas: video,
            timestamp: Number.isFinite(video.currentTime)
              ? video.currentTime
              : timestamp,
            duration: 1 / Math.max(1, clipFps),
          },
          maskFrame,
        );
      } catch (e) {
        console.warn("[video] seek failed", e);
        pendingSeekTargetRef.current = null;
        void renderPosterFallback({ force: true });
      }
    },
    [
      getTargetFrameInfo,
      maskFrameForCurrentFocus,
      isAccurateSeekNeeded,
      isInFrame,
      isPlaying,
      hidden,
      renderPosterFallback,
      drawWrappedCanvas,
      mediaInfo,
      fps,
      startMediaSourceStream,
      getMediaSourcePreviewParams,
      computeMaskFrameForFocus,
    ],
  );

  useEffect(() => {
    void seekToCurrentFrame();
  }, [seekToCurrentFrame]);

  const startRendering = useCallback(async (opts?: { force?: boolean }) => {
    if (!canvasRef.current) return;
    if (!mediaInfo.current) return;
    if (!displayWidth || !displayHeight) return;
    if (
      !opts?.force &&
      playingLatchRef.current &&
      mediaSourceSessionIdRef.current &&
      playbackVideoRef.current
    ) {
      return;
    }
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
    // @ts-ignore
    iteratorRef.current?.return?.();

    const checkCancel = () => {
      if (!offscreenFast && myToken !== drawTokenRef.current) return false;
      if (!isPlayingRef.current) return false;
      return true;
    };

    const frameDuration = 1 / clipFps;

    try {
      const { targetFps, targetSize } = getMediaSourcePreviewParams();
      const video = await startMediaSourceStream(startTime, {
        fps: targetFps,
        size: targetSize,
        mode: "playback",
      });
      if (!video || !checkCancel()) {
        if (!video && isPlayingRef.current) {
          void renderPosterFallback({ force: true });
          const stamp =
            typeof performance !== "undefined" && performance.now
              ? performance.now()
              : Date.now();
          const retryAt = Math.max(
            mediaSourceBackoffUntilRef.current,
            mediaSourceDisabledUntilRef.current,
          );
          const retryIn = Math.max(250, Math.min(3000, retryAt - stamp + 25));
          if (mediaSourceRetryTimerRef.current != null) {
            window.clearTimeout(mediaSourceRetryTimerRef.current);
          }
          mediaSourceRetryTimerRef.current = window.setTimeout(() => {
            mediaSourceRetryTimerRef.current = null;
            if (isPlayingRef.current) {
              void startRenderingRef.current({ force: true });
            }
          }, retryIn);
        }
        return;
      }
      playingLatchRef.current = true;

      try {
        video.playbackRate = Math.max(0.1, speed);
      } catch {}
      await video.play().catch(() => {});
      if (!checkCancel()) return;
      let previousVideoTime = Number.NaN;
      let stagnantTicks = 0;

      const tick = () => {
        if (!checkCancel()) return;

        if (video.readyState < 2) {
          playbackRafRef.current = requestAnimationFrame(tick);
          return;
        }

        if (Number.isFinite(previousVideoTime)) {
          const delta = Math.abs(video.currentTime - previousVideoTime);
          if (delta < 1 / Math.max(1, clipFps * 4)) {
            stagnantTicks += 1;
          } else {
            stagnantTicks = 0;
          }
        }
        previousVideoTime = video.currentTime;
        if (stagnantTicks > 24 && !video.paused && !video.ended) {
          const nowStamp =
            typeof performance !== "undefined" && performance.now
              ? performance.now()
              : Date.now();
          if (nowStamp - lastPlaybackRestartRequestAtRef.current > 600) {
            lastPlaybackRestartRequestAtRef.current = nowStamp;
            void startRenderingRef.current({ force: true });
            return;
          }
        }

        if (video.currentTime >= endTime) {
          try {
            video.pause();
          } catch {}
          playingLatchRef.current = false;
          return;
        }

        const maskFrame = computeMaskFrameForFocus(focusFrameRef.current);
        decoderMaskFrameRef.current = maskFrame;
        drawWrappedCanvas(
          {
            canvas: video,
            timestamp: video.currentTime,
            duration: frameDuration,
          },
          maskFrame,
        );

        playbackRafRef.current = requestAnimationFrame(tick);
      };

      playbackRafRef.current = requestAnimationFrame(tick);
    } catch (e: any) {
      playingLatchRef.current = false;
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
    renderPosterFallback,
    startMediaSourceStream,
    getMediaSourcePreviewParams,
    computeMaskFrameForFocus,
    offscreenFast,
    currentLocalFrameOverride,
  ]);

  const startRenderingRef = useRef(startRendering);
  useEffect(() => {
    startRenderingRef.current = startRendering;
  }, [startRendering]);

  const playbackScopeRef = useRef(mediaSessionScope);
  useEffect(() => {
    if (playbackScopeRef.current === mediaSessionScope) return;
    playbackScopeRef.current = mediaSessionScope;
    if (isPlayingRef.current) {
      void startRenderingRef.current({ force: true });
    }
  }, [mediaSessionScope]);

  useEffect(() => {
    if (isPlaying) {
      void startRenderingRef.current();
    }
    return () => {
      cleanupMediaSourcePlayback({ keepVideo: true });
      skipDrawRef.current = false;
      suppressSeekFramesRef.current = false;
      resumeGateFrameRef.current = null;
      pendingSeekTargetRef.current = null;
      playingLatchRef.current = false;
      if (mediaSourceRetryTimerRef.current != null) {
        window.clearTimeout(mediaSourceRetryTimerRef.current);
        mediaSourceRetryTimerRef.current = null;
      }
      drawTokenRef.current++;
      // @ts-ignore
      iteratorRef.current?.return?.();
    };
  }, [isPlaying, cleanupMediaSourcePlayback]);

  // Keep playback synchronized with the timeline clock (master/slave style).
  // If decode output drifts too far from the timeline target, restart iteration
  // from the current focus frame to recover smoothly.
  useEffect(() => {
    if (!isPlaying) return;

    const interval = window.setInterval(() => {
      if (!isPlayingRef.current) return;
      if (pendingSeekTargetRef.current?.strict) return;

      const targetInfo = getTargetFrameInfoRef.current();
      const renderedTimestamp = lastRenderedTimestampRef.current;
      if (!targetInfo || renderedTimestamp == null) return;

      const driftSec = targetInfo.timestamp - renderedTimestamp;
      if (Math.abs(driftSec) < 0.35) return;

      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      if (now < playbackResyncCooldownUntilRef.current) return;

      playbackResyncCooldownUntilRef.current = now + 250;
      void startRenderingRef.current({ force: true });
    }, 30);

    return () => {
      window.clearInterval(interval);
    };
  }, [isPlaying]);

  // On play -> pause transition, settle to an accurate frame so scrubbing and
  // paused edits start from a deterministic visual frame.
  useEffect(() => {
    if (wasPlayingRef.current && !isPlaying) {
      void seekToCurrentFrame(true);
    }
    wasPlayingRef.current = isPlaying;
  }, [isPlaying, seekToCurrentFrame]);

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
        void startRenderingRef.current({ force: true });
      }
    }
  }, [focusFrame, isPlaying]);

  // If video is paused, reapply filters/applicators/masks immediately.
  useEffect(() => {
    if (!isPlaying && canvasRef.current && imageRef.current) {
      if (originalFrameRef.current) {
        let canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const workingCanvas = ensureProcessingCanvas(canvas.width, canvas.height);
        const workingCtx = workingCanvas.getContext("2d");
        if (!workingCtx) return;

        workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
        workingCtx.drawImage(originalFrameRef.current, 0, 0);

        const maskedCanvas =
          toolRef.current !== "mask"
            ? applyMaskRef.current(workingCanvas, maskFrameForCurrentFocus)
            : workingCanvas;
        let processedCanvas = maskedCanvas;

        processedCanvas = applyFilters(processedCanvas, filterParamsRef.current, {
          output: "passthrough",
        });

        for (const applicator of applicatorsRef.current) {
          const result = applicator.apply(processedCanvas);
          if (result) processedCanvas = result;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(processedCanvas, 0, 0, canvas.width, canvas.height);

        imageRef.current.getLayer()?.batchDraw();
      } else {
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

  // Force re-init when selected clip context changes.
  useEffect(() => {
    lastSelectedAssetIdRef.current = null;
    lastRenderedFrameRef.current = -1;
    lastRenderedTimestampRef.current = null;
    lastPosterKeyRef.current = null;
    originalFrameRef.current = null;
    lastDrawnFocusFrameRef.current = null;
    resumeGateFrameRef.current = null;
    pendingSeekTargetRef.current = null;
    // @ts-ignore
    iteratorRef.current?.return?.();
    iteratorRef.current = null;
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
