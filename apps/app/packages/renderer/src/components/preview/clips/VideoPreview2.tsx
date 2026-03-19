import { AnyClipProps, FilterClipProps, MediaInfo, VideoClipProps, clipSignature } from "@/lib/types";
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
import { DEFAULT_FPS } from "@/lib/settings";
import { useClipStore } from "@/lib/clip";

import { CompositorShader } from "@/components/preview/webgl-filters";
import { BaseClipApplicator } from "./apply/base";
import _ from "lodash";
import { useInputControlsStore } from "@/lib/inputControl";
import SharedClipCanvasSurface, {
  getAspectFitSize,
} from "./shared/SharedClipCanvasSurface";

import {useVideoDecoder} from "@/lib/video-decode/context";


import { useViewportStore } from "@/lib/viewport";

type FrameRange = {
  startFrame: number;
  endFrame?: number;
};

const toFiniteFrame = (value: unknown): number | undefined => {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  return value;
};

const getFrameRangeFromPath = (path?: string): Partial<FrameRange> => {
  if (!path) return {};
  try {
    const u = new URL(path);
    const startRaw = u.searchParams.get("startFrame");
    const endRaw = u.searchParams.get("endFrame");
    const startFrame = startRaw == null ? undefined : toFiniteFrame(Number(startRaw));
    const endFrame = endRaw == null ? undefined : toFiniteFrame(Number(endRaw));
    return { startFrame, endFrame };
  } catch {
    return {};
  }
};

const toSourceProjectFrame = (
  timelineLocalFrame: number,
  trimStart: number | undefined,
  frameOffset: number,
  speed: number,
  isUsingPreprocessorSrc: boolean,
): number => {
  const speedFactor = Math.max(0.1, speed);
  if (isUsingPreprocessorSrc) {
    return Math.max(0, timelineLocalFrame - Math.max(0, frameOffset)) * speedFactor;
  }
  // Keep trimStart as a fixed source offset; only scale timeline progression.
  return Math.max(0, (trimStart || 0) + timelineLocalFrame * speedFactor);
};

function filtersSignature(filters: FilterClipProps[]): string {
  if (!filters?.length) return "";
  return filters
    .map((f) =>
      [
        f.clipId ?? "",
        f.smallPath ?? "",
        f.fullPath ?? "",
        f.intensity ?? 100,
        f.startFrame ?? 0,
        f.endFrame ?? 0,
      ].join(",")
    )
    .join("|");
}

function clipRenderSignature(clip: VideoClipProps): string {
  const base = clipSignature(clip as AnyClipProps);
  const adj = [
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
  ].join(",");
  return `${base};${adj}`;
}

const createUpdateSignature = (
  maskFrame: number,
  clip: VideoClipProps,
  focusFrame: number,
  filters: FilterClipProps[],
  useMask: boolean
): string => {
  return [
    maskFrame,
    focusFrame,
    useMask ? "1" : "0",
    clipRenderSignature(clip),
    filtersSignature(filters),
  ].join("::");
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
  framesToPrefetch = 4,
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
  offscreenFast = false,
  decoderKey,
  hidden = false,
}) => {

  
  // `mediaInfo` is stored in a ref for fast access by decoder callbacks, but ref updates
  // don't trigger React renders. We bump this version whenever `mediaInfo.current` changes
  // so aspect-fit sizing and Konva props update immediately (no "wait until drag" issues).
  const videoDecoder = useVideoDecoder();
  const focusFrameFromControls = useControlsStore((state) => state.focusFrame);
  const focusFrameFromInputs = useInputControlsStore((s) =>
    s.getFocusFrame(inputId ?? ""),
  );

  const initCompleteRef = useRef(false);

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

  

  const compositorRef = useRef<CompositorShader | null>(null);
  if (!compositorRef.current) {
    compositorRef.current = new CompositorShader();
  }
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

    return f >= s && f < e;
  }, [focusFrame, startFrameUsed, endFrameUsed]);


  const currentFrame = useMemo(
    () => focusFrame - startFrameUsed + (trimStart || 0),
    [focusFrame, startFrameUsed, trimStart],
  );
  const speed = useMemo(() => {
    const s = Number(_speed ?? 1);
    return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
  }, [_speed]);

  const clipTransform = overrideClip
    ? overrideClip.transform
    : useClipStore((s) => s.getClipTransform(clipId));
  const srcFps = useControlsStore((s) => s.fps);

  const getAssetById = useClipStore((s) => s.getAssetById);

  const cachedPreprocessorRangeRef = useRef<{
    startFrame: number;
    endFrame: number;
    selectedAssetId: string;
    frameOffset: number;
  } | null>(null);
  const addedTimestampRef = useRef<number | undefined>(undefined); // last timestamp rendered

  
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

  const decoderId = useMemo(() => {
    return makeDecoderId(selectedAssetId);
  }, [selectedAssetId, clipId, decoderKey, inputMode, inputId]);

  const mediaInfo = useRef<MediaInfo | null>(getMediaInfoCached(selectedAssetId) || null);

  useEffect(() => {
    mediaInfo.current = getMediaInfoCached(selectedAssetId ?? "") || null;
  }, [selectedAssetId]);

  const maskFrameForCurrentFocus = useMemo(() => {
    if (clip) {
      if (inputMode) {
        return Math.max(
          0,
          Math.floor(
            toSourceProjectFrame(
              focusFrame,
              trimStart,
              0,
              speed,
              false,
            ),
          ),
        );
      }
      const isUsingPreprocessorSrc = selectedAssetId !== assetId;
      const baseLocal = Math.max(0, focusFrame - startFrameUsed);
      return Math.max(
        0,
        Math.floor(
          toSourceProjectFrame(
            baseLocal,
            trimStart,
            frameOffset,
            speed,
            isUsingPreprocessorSrc,
          ),
        ),
      );
    }
    return Math.max(
      0,
      Math.floor(
        toSourceProjectFrame(
          Math.max(0, focusFrame - startFrameUsed),
          trimStart,
          0,
          speed,
          false,
        ),
      ),
    );
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

  const effectiveSourceRange = useMemo<FrameRange>(() => {
    const info = mediaInfo.current;
    const fallbackStart = Math.max(0, toFiniteFrame(info?.startFrame) ?? 0);
    const fallbackEnd = toFiniteFrame(info?.endFrame);
    const assetPath = getAssetById(selectedAssetId)?.path || info?.path;
    const fromPath = getFrameRangeFromPath(assetPath);

    if (fromPath.startFrame !== undefined || fromPath.endFrame !== undefined) {
      return {
        startFrame: Math.max(0, fromPath.startFrame ?? 0),
        endFrame: toFiniteFrame(fromPath.endFrame),
      };
    }

    const clipTrimStart = trimStart || 0;
    const activePreprocessor = (clip?.preprocessors ?? []).find((p) => {
      if (
        p.startFrame === undefined ||
        p.endFrame === undefined ||
        p.createNewClip !== false ||
        p.status !== "complete" ||
        p.assetId !== selectedAssetId
      ) {
        return false;
      }
      const adjustedStartFrame = p.startFrame + clipTrimStart;
      const adjustedEndFrame = p.endFrame + clipTrimStart;
      return (
        currentFrame >= adjustedStartFrame && currentFrame <= adjustedEndFrame
      );
    });

    if (activePreprocessor) {
      const preStart = Math.max(
        0,
        toFiniteFrame(activePreprocessor.startFrame) ?? 0,
      );
      const preEnd = toFiniteFrame(activePreprocessor.endFrame);
      return {
        startFrame: preStart,
        endFrame:
          preEnd !== undefined ? Math.max(preStart + 1, preEnd) : undefined,
      };
    }

    return { startFrame: fallbackStart, endFrame: fallbackEnd };
  }, [
    clip?.preprocessors,
    currentFrame,
    getAssetById,
    selectedAssetId,
    trimStart,
  ]);

  // (seekInProgressRef removed; was unused and could cause confusion)

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

  const fpsFromControls = useControlsStore((s) => s.fps);
  const fpsFromInputs = useInputControlsStore((s) => s.getFps(inputId ?? ""));
  const fps = useInputScopedControls ? fpsFromInputs : fpsFromControls;
  const prewarmLeadFrames = useMemo(() => {
    const value = Number(framesToPrefetch ?? 0);
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.floor(value));
  }, [framesToPrefetch]);
  const tool = useViewportStore((s) => s.tool);

  const currentSignatureRef = useRef<string | null>(null);
  const prewarmedSeekKeyRef = useRef<string | null>(null);

  useEffect(() => {
    currentSignatureRef.current = createUpdateSignature(maskFrameForCurrentFocus, clip, focusFrame, applicators.map((applicator) => applicator.getClip()), tool !== "mask");
  }, [maskFrameForCurrentFocus, clip, focusFrame, applicators, tool]);

  const fpsRef = useRef(fps);
  useEffect(() => {
    fpsRef.current = fps;
  }, [fps]);
  
  const speedRef = useRef(speed);
  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  const isInFrameRef = useRef(isInFrame);
  useEffect(() => {
    isInFrameRef.current = isInFrame;
  }, [isInFrame]);

  // Create canvas once and expose to Konva Image via state so initial render receives it
  useEffect(() => {
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
      setImageSource(canvasRef.current);
    } else {
      setImageSource(canvasRef.current);
    }
    return () => {
      compositorRef.current?.dispose();
      compositorRef.current = null;
      canvasRef.current = null;
      originalFrameRef.current = null;
      processingCanvasRef.current = null;
      setImageSource(null);
    };
  }, []);

  // Compute aspect-fit display size and offsets within the preview rect
  const { displayWidth, displayHeight, offsetX, offsetY } = useMemo(() => {
    return getAspectFitSize(
      mediaInfo.current?.video?.displayWidth || 0,
      mediaInfo.current?.video?.displayHeight || 0,
      rectWidth,
      rectHeight,
    );
  }, [
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

  const getTargetFrameInfo = useCallback(() => {
    if (!mediaInfo.current) return null;
    const clipFps =
      mediaInfo.current.stats.video?.averagePacketRate || fps || DEFAULT_FPS;
    const projectFps = fps || DEFAULT_FPS;
    if (!Number.isFinite(clipFps) || clipFps <= 0) return null;
    if (!Number.isFinite(projectFps) || projectFps <= 0) return null;

    const isUsingPreprocessorSrc = selectedAssetId !== assetId;
    const currentFrameForSeek = focusFrame - startFrameUsed + (trimStart || 0);
    const timelineLocalFrame = currentFrameForSeek - (trimStart || 0);
    const idealFrame = toSourceProjectFrame(
      timelineLocalFrame,
      trimStart,
      frameOffset,
      speed,
      isUsingPreprocessorSrc,
    );

    
    const actualFrame = Math.floor((idealFrame / projectFps) * clipFps + 1e-4);
    const totalFrames = Math.max(
      0,
      Math.floor((mediaInfo.current.duration || 0) * clipFps),
    );

    const targetFrame =
      Math.max(0, Math.min(totalFrames, actualFrame)) +
      Math.round((effectiveSourceRange.startFrame / projectFps) * clipFps);

      const timelineLocalEndFrame = Math.max(0, (endFrameUsed ?? 0) - startFrameUsed);
      const endFrameProject = toSourceProjectFrame(
        timelineLocalEndFrame,
        trimStart,
        frameOffset,
        speed,
        isUsingPreprocessorSrc,
      );
      const endTimestamp =
        endFrameProject > 0
          ? endFrameProject / projectFps
          : (mediaInfo.current?.duration ?? undefined);
      
    
    return { timestamp: targetFrame / clipFps, targetFrame, endTimestamp };
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
    effectiveSourceRange,
    endFrameUsed,
  ]);

  const getTargetFrameInfoRef = useRef(getTargetFrameInfo);
  useEffect(() => {
    getTargetFrameInfoRef.current = getTargetFrameInfo;
  }, [getTargetFrameInfo]);

  const onUpdateComplete = useCallback((data: { id: string, success: boolean }) => {
    
    if (!isPlayingRef.current && data.success && isInFrameRef.current) {
      const targetFrameInfo = getTargetFrameInfoRef.current();
      
      if (targetFrameInfo) {
        videoDecoder.seek({
          id: data.id,
          timestamp: targetFrameInfo.timestamp,
          speed: speed,
          targetFps: fps,
        });
      }
    }
  }, [videoDecoder]);

  const onInitComplete = useCallback((data: { id: string, duration: number }) => {
    initCompleteRef.current = true;
    const targetFrameInfo = getTargetFrameInfoRef.current();
    if (!isInFrameRef.current) return;
    
    if (targetFrameInfo) {
      if (isPlayingRef.current) {
        // we want to continue iterating from the current frame
        videoDecoder.iterate({
          id: data.id,
          startTimestamp: targetFrameInfo.timestamp,
          endTimestamp: targetFrameInfo.endTimestamp,
          speed: speed,
          targetFps: fps,
        });
        
      } else {

        videoDecoder.pause({
          id: data.id,
        });
        
        videoDecoder.seek({
          id: data.id,
          timestamp: targetFrameInfo.timestamp,
          speed: speed,
          targetFps: fps,
        });
      }
      
    }
  }, [videoDecoder]);

  const decoderSources = useMemo(() => {
    // get all assetIDs from the clip and preprocessors
    const assetIds = [selectedAssetId, ...(clip?.preprocessors ?? []).map((p) => p.assetId)];
    return assetIds.filter((id): id is string => id !== undefined);
  }, [clip?.preprocessors, selectedAssetId]);


  useEffect(() => {
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }

    for (const source of decoderSources) {
      const currentDecoderId = makeDecoderId(source);
      const currentMediaInfo = getMediaInfoCached(source) || mediaInfo.current;
      const {displayWidth: currentDisplayWidth, displayHeight: currentDisplayHeight} = getAspectFitSize(
        currentMediaInfo?.video?.displayWidth || 0,
        currentMediaInfo?.video?.displayHeight || 0,
        rectWidth,
        rectHeight,
      );

      videoDecoder.init({
         canvasId: clipId,
         canvas: canvasRef.current,
         sourceOrPath: currentMediaInfo?.path || "",
         renderer: "2d",
         width: currentDisplayWidth,
         height: currentDisplayHeight,
         id: currentDecoderId,
         onUpdateComplete: onUpdateComplete,
         onInitComplete: onInitComplete,
         onFrame() {
          requestAnimationFrame(() => {
            imageRef.current?.getLayer()?.batchDraw?.();
          });
         },
        })
    }

    return () => {
      for (const source of decoderSources) {
        const currentDecoderId = makeDecoderId(source);
        videoDecoder.destroy({
          id: currentDecoderId,
          canvasId: clipId,
        });
      }
    }

  }, [clipId, decoderSources]);

  useEffect(() => {
    if (!initCompleteRef.current) return;
    if (isPlaying && isInFrame) {
      const targetFrameInfo = getTargetFrameInfoRef.current();

      videoDecoder.iterate({
        id: decoderId,
        startTimestamp: targetFrameInfo?.timestamp ?? 0,
        endTimestamp: targetFrameInfo?.endTimestamp ?? 0,
        speed: speed,
        targetFps: fps
      });

    } else {
      videoDecoder.pause({
        id: decoderId,
      });
    }
  }, [decoderId, fps, isInFrame, isPlaying, speed, videoDecoder]);

  useEffect(() => {
    if (!initCompleteRef.current) return;

    if (!isPlaying) {
      prewarmedSeekKeyRef.current = null;
      return;
    }

    // Prewarm slightly before clip entry so the first visible frame is ready.
    const framesUntilStart = startFrameUsed - focusFrame;
    if (!Number.isFinite(framesUntilStart) || framesUntilStart <= 0) {
      prewarmedSeekKeyRef.current = null;
      return;
    }

    const clipEnd =
      typeof endFrameUsed === "number" && Number.isFinite(endFrameUsed)
        ? endFrameUsed
        : Infinity;
    if (focusFrame >= clipEnd) {
      prewarmedSeekKeyRef.current = null;
      return;
    }

    if (framesUntilStart > prewarmLeadFrames) {
      prewarmedSeekKeyRef.current = null;
      return;
    }

    const targetFrameInfo = getTargetFrameInfoRef.current();
    if (!targetFrameInfo) return;

    const seekKey = `${decoderId}:${Math.floor(targetFrameInfo.targetFrame)}`;
    if (prewarmedSeekKeyRef.current === seekKey) return;

    videoDecoder.seek({
      id: decoderId,
      timestamp: targetFrameInfo.timestamp,
      speed: speed,
      targetFps: fps,
    });
    prewarmedSeekKeyRef.current = seekKey;
  }, [
    decoderId,
    endFrameUsed,
    focusFrame,
    fps,
    isPlaying,
    prewarmLeadFrames,
    speed,
    startFrameUsed,
    videoDecoder,
  ]);


  // Update refs when values change
  useEffect(() => {
    if (!initCompleteRef.current) return;

    videoDecoder.updateRenderer({
      id: decoderId,
      maskFrame: maskFrameForCurrentFocus,
      clip: clip,
      focusFrame: focusFrame,
      filters: applicators.map((applicator) => applicator.getClip()),
      useMask: tool !== "mask",
    });

  }, [
    decoderId,
    focusFrame,
    maskFrameForCurrentFocus,
    clip,
    applicators,
    tool,
    speed,
    fps,
    isPlaying,
  ]);

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
