import { ImageClipProps, MediaInfo } from "@/lib/types";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { fetchImage } from "@/lib/media/image";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import { useControlsStore } from "@/lib/control";
import Konva from "konva";
import { useViewportStore } from "@/lib/viewport";
// (useClipStore already imported above)
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import { BaseClipApplicator } from "./apply/base";
import { useClipStore } from "@/lib/clip";
import { useWebGLMask } from "../mask/useWebGLMask";
import { useInputControlsStore } from "@/lib/inputControl";
import SharedClipCanvasSurface, {
  getAspectFitSize,
} from "./shared/SharedClipCanvasSurface";

const ImagePreview: React.FC<
  ImageClipProps & {
    rectWidth: number;
    rectHeight: number;
    applicators: BaseClipApplicator[];
    overlap: boolean;
    overrideClip?: ImageClipProps;
    inputMode?: boolean;
    inputId?: string;
    focusFrameOverride?: number;
    currentLocalFrameOverride?: number;
  }
> = ({
  assetId,
  clipId,
  rectWidth,
  rectHeight,
  applicators,
  overlap,
  overrideClip,
  inputMode = false,
  inputId,
  focusFrameOverride,
  currentLocalFrameOverride,
}) => {
  const mediaInfoRef = useRef<MediaInfo | null>(
    getMediaInfoCached(assetId) || null,
  );


  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<Konva.Image>(null);
  const { applyFilters } = useWebGLFilters();
  const clipsState = useClipStore((s) => s.clips);
  const tool = useViewportStore((s) => s.tool);
  const clipTransform = overrideClip
    ? overrideClip.transform
    : useClipStore((s) => s.getClipTransform(clipId));
  const focusFrameFromControls = useControlsStore((s) => s.focusFrame);
  const focusFrameFromInputs = useInputControlsStore((s) =>
    s.getFocusFrame(inputId ?? ""),
  );
  const clipInfo = useMemo(() => {
    try {
      const overrideAny = overrideClip as any | undefined;
      const base =
        overrideAny ?? (useClipStore.getState().getClipById(clipId) as any);
      return {
        groupId: base?.groupId as string | undefined,
        startFrame: base?.startFrame ?? 0,
      };
    } catch {
      return { groupId: undefined, startFrame: 0 };
    }
  }, [overrideClip, clipId]);
  const groupStartForClip = useMemo(() => {
    const grpId = clipInfo.groupId;
    if (!grpId) return 0;
    try {
      const groupClip = useClipStore.getState().getClipById(grpId) as any;
      return groupClip?.startFrame ?? 0;
    } catch {
      return 0;
    }
  }, [clipInfo.groupId]);

  const syntheticGlobalFromLocal =
    typeof currentLocalFrameOverride === "number"
      ? Math.max(
          0,
          clipInfo.startFrame +
            groupStartForClip +
            Math.max(0, currentLocalFrameOverride),
        )
      : undefined;
  const focusFrame =
    typeof focusFrameOverride === "number"
      ? focusFrameOverride
      : typeof syntheticGlobalFromLocal === "number"
        ? syntheticGlobalFromLocal
        : inputMode
          ? focusFrameFromInputs
          : focusFrameFromControls;


  const clipFromStore = useClipStore((s) =>
    s.getClipById(clipId),
  ) as ImageClipProps;
  const clip = (overrideClip as ImageClipProps) || clipFromStore;

  // Determine whether our `focusFrame` is expressed in global timeline frames or in input-local
  // frames. In input mode we sometimes synthesize a global focus frame via
  // `currentLocalFrameOverride`, so we must keep the in-frame check in the same space.
  const isGlobalFocusFrame =
    !inputMode ||
    typeof focusFrameOverride === "number" ||
    typeof syntheticGlobalFromLocal === "number";

  // Mirror VideoPreview's input-mode semantics: grouped clips render in a 0-based group-local
  // frame space; non-grouped input previews render in a 0-based local window.
  const startFrameUsed = useMemo(() => {
    const rawStart = (clip as any)?.startFrame ?? 0;
    if (isGlobalFocusFrame) return rawStart;
    const hasGroup = Boolean(clipInfo.groupId);
    if (hasGroup) {
      return Math.max(0, rawStart - (groupStartForClip || 0));
    }
    return 0;
  }, [clip, clipInfo.groupId, groupStartForClip, isGlobalFocusFrame]);

  const endFrameUsed = useMemo(() => {
    const rawEnd = (clip as any)?.endFrame as number | undefined;
    const rawStart = (clip as any)?.startFrame as number | undefined;
    if (isGlobalFocusFrame) return typeof rawEnd === "number" ? rawEnd : undefined;

    const hasGroup = Boolean(clipInfo.groupId);
    if (hasGroup && typeof rawEnd === "number") {
      return Math.max(0, rawEnd - (groupStartForClip || 0));
    }

    // For non-grouped input previews, normalize absolute [start..end] to a 0-based window.
    if (typeof rawEnd === "number" && typeof rawStart === "number") {
      return Math.max(0, rawEnd - rawStart);
    }

    return typeof rawEnd === "number" ? rawEnd : undefined;
  }, [clip, clipInfo.groupId, groupStartForClip, isGlobalFocusFrame]);

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

  // Stable signature for masks so we don't retrigger effects on array identity changes
  const masksSignature = useMemo(() => {
    const masks = clip?.masks;
    if (!masks || masks.length === 0) return "none";
    try {
      return masks
        .map((m) => {
          const keyframes = m.keyframes as
            | Map<number, any>
            | Record<number, any>
            | undefined;
          const keyframeKeys = keyframes
            ? keyframes instanceof Map
              ? Array.from(keyframes.keys()).join(",")
              : Object.keys(keyframes).join(",")
            : "none";
          return [
            m.id,
            m.tool,
            m.isTracked ? "tracked" : "static",
            m.lastModified,
            keyframeKeys,
            // Operation / rendering settings that affect the masked output
            m.featherAmount,
            m.brushSize ?? "na",
            m.inverted ? "inv" : "norm",
            m.maskColorEnabled ?? true,
            m.maskColor ?? "na",
            m.maskOpacity ?? "na",
            m.backgroundColorEnabled ?? true,
            m.backgroundColor ?? "na",
            m.backgroundOpacity ?? "na",
            // Tracking-related knobs that can influence which keyframes exist / are used
            m.trackingDirection ?? "na",
            m.confidenceThreshold ?? "na",
            m.maxTrackingFrames ?? "na",
            // Transform (mask application depends on this)
            (() => {
              const t = m.transform as any;
              if (!t) return "t:none";
              return `t:${[
                t.x,
                t.y,
                t.width,
                t.height,
                t.scaleX,
                t.scaleY,
                t.rotation,
              ].join(",")}`;
            })(),
          ].join("#");
        })
        .join("|");
    } catch {
      return `len:${masks?.length ?? 0}`;
    }
  }, [clip?.masks]);

  const { applyMask } = useWebGLMask({
    focusFrame: focusFrame,
    masks: clip?.masks || [],
    disabled: tool === "mask" && !inputMode,
    clip: clip
  });

  const selectedAssetId = useMemo(() => {
    return (
      // Only apply preprocessor outputs in-place when explicitly requested.
      // When createNewClip is enabled (default), the parent clip should render as-is.
      clip?.preprocessors?.find(
        (p) => p.createNewClip === false && p.status === "complete",
      )?.assetId ?? assetId
    );
  }, [assetId, clip?.preprocessors]);

  const getAssetById = useClipStore((s) => s.getAssetById);
  const [, forceRerenderForMediaInfo] = useState(0);

  // Create canvas once
  useEffect(() => {
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }
    return () => {
      canvasRef.current = null;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const asset = getAssetById(assetId);
        if (!asset) return;
        const info = await getMediaInfo(asset.path);
        if (!cancelled) {
          mediaInfoRef.current = info;
          // Trigger a safe re-render so dimensions recompute, the draw effect will run then
          forceRerenderForMediaInfo((v) => v + 1);
        }
      } catch (e) {
        console.error(e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [assetId]);

  const { displayWidth, displayHeight, offsetX, offsetY } = useMemo(() => {
    const originalWidth = mediaInfoRef.current?.image?.width || 0;
    const originalHeight = mediaInfoRef.current?.image?.height || 0;
    return getAspectFitSize(originalWidth, originalHeight, rectWidth, rectHeight);
  }, [
    mediaInfoRef.current?.image?.height,
    mediaInfoRef.current?.image?.width,
    rectWidth,
    rectHeight,
  ]);

  // Stable applicators ref to avoid recreating draw on array ref changes
  const applicatorsRef = useRef<BaseClipApplicator[]>(applicators);
  useEffect(() => {
    applicatorsRef.current = applicators;
  }, [applicators]);

  // Timeline-aware applicator signature (type + clipId + start-end)
  const applicatorsSignature = useMemo(() => {
    if (!applicators || applicators.length === 0) return "none";
    try {
      return applicators
        .map((a) => {
          const type = a?.constructor?.name || "Unknown";
          const start = (a)?.getStartFrame?.() ?? "u";
          const end = (a)?.getEndFrame?.() ?? "u";
          const intensity = (a)?.getIntensity?.() ?? "u";
          const owner = (a as any)?.getClip?.()?.clipId ?? "u";
          return `${type}#${owner}@${start}-${end}@${intensity}`;
        })
        .join("|");
    } catch {
      return `len:${applicators.length}`;
    }
  }, [applicators]);

  // Store-driven active flag for current focus frame
  const applicatorsActiveStore = useMemo(() => {
    const apps = applicators || [];
    if (!apps.length) return false;
    const getClipById = useClipStore.getState().getClipById;
    const frame = typeof focusFrame === "number" ? focusFrame : 0;
    return apps.some((a) => {
      const owned = (a as any)?.getClip?.();
      const id = owned?.clipId;
      if (!id) return false;
      const sc = getClipById(id) as any;
      if (!sc) return false;
      const start = sc.startFrame ?? 0;
      const end = sc.endFrame ?? 0;
      return frame >= start && frame <= end;
    });
  }, [clipsState, focusFrame, applicatorsSignature]);


  // Stabilize applyMask across focusFrame changes; we'll pass frame explicitly when drawing
  const applyMaskRef = useRef<typeof applyMask | null>(applyMask);
  useEffect(() => {
    applyMaskRef.current = applyMask;
  }, [applyMask]);

  const draw = useCallback(async () => {
    if (!isInFrame) return;
    if (!canvasRef.current) return;
    if (!mediaInfoRef.current) return;
    if (!displayWidth || !displayHeight) return;

    try {
      const targetWidth = Math.max(1, Math.floor(displayWidth));
      const targetHeight = Math.max(1, Math.floor(displayHeight));
      const asset = getAssetById(selectedAssetId);
      if (!asset) return;

      let canvas = canvasRef.current;
      if (!canvas) return;
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }

      const image = await fetchImage(asset.path, targetWidth, targetHeight, {
        mediaInfo: mediaInfoRef.current,
      });

      if (!image) return;

      canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = true;
      // @ts-ignore
      ctx.imageSmoothingQuality = "high";
      // clear the canvas FIRST to ensure clean slate
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Create a fresh working canvas completely isolated from cached canvas
      const workingCanvas = document.createElement("canvas");
      workingCanvas.width = canvas.width;
      workingCanvas.height = canvas.height;
      const workingCtx = workingCanvas.getContext("2d");
      if (!workingCtx) return;

      // Copy the original image to working canvas (never mutate image.canvas!)
      workingCtx.imageSmoothingEnabled = true;
      // @ts-ignore
      workingCtx.imageSmoothingQuality = "high";
      workingCtx.drawImage(
        image.canvas as HTMLCanvasElement,
        0,
        0,
        canvas.width,
        canvas.height,
      );

      // Apply mask to working canvas (may return same or different canvas)
      // Pass the current focusFrame explicitly; do not depend on applyMask identity
      let processedCanvas =
        applyMaskRef.current?.(workingCanvas, focusFrame) ?? workingCanvas;

      // Apply WebGL filters and keep output in passthrough mode.
      processedCanvas = applyFilters(
        processedCanvas,
        {
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
        },
        { output: "passthrough" },
      );

      // Ensure resources (e.g., CLUTs) are preloaded for applicators before applying
      const preloadTasks: Promise<void>[] = [];
      for (const app of applicatorsRef.current || []) {
        const ensure = (app as any)?.ensureResources as
          | (() => Promise<void>)
          | undefined;
        if (typeof ensure === "function") {
          preloadTasks.push(ensure());
        }
      }
      if (preloadTasks.length) {
        try {
          await Promise.all(preloadTasks);
        } catch {}
      }

      // Apply applicators to canvas
      let finalCanvas = processedCanvas;
      for (const applicator of applicatorsRef.current || []) {
        const result = applicator.apply(finalCanvas);
        if (result) finalCanvas = result;
      }

      // Draw final result to display canvas
      ctx.drawImage(finalCanvas, 0, 0, canvas.width, canvas.height);

      imageRef.current?.getLayer()?.batchDraw?.();
    } catch (e) {
      console.log("error", e);
      console.error(e);
    }
  }, [
    mediaInfoRef,
    selectedAssetId,
    displayWidth,
    displayHeight,
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
    masksSignature,
    applicatorsSignature,
    applicatorsActiveStore,
    applyFilters,
    tool,
    isInFrame,
  ]);

  useEffect(() => {
    draw();
  }, [draw]);

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
      overlap={overlap}
      inputMode={inputMode}
      isInFrame={isInFrame}
      overrideClip={!!overrideClip}
    />
  );
};

export default ImagePreview;
