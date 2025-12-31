import { ImageClipProps, MediaInfo } from "@/lib/types";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Image, Transformer, Group, Line } from "react-konva";
import { fetchImage } from "@/lib/media/image";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import { useControlsStore } from "@/lib/control";
import Konva from "konva";
import { useViewportStore } from "@/lib/viewport";
// (useClipStore already imported above)
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import { BaseClipApplicator } from "./apply/base";
import { useClipStore, getLocalFrame } from "@/lib/clip";
import { useWebGLMask } from "../mask/useWebGLMask";
import { useInputControlsStore } from "@/lib/inputControl";

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
  const transformerRef = useRef<Konva.Transformer>(null);
  const suppressUntilRef = useRef<number>(0);
  const { applyFilters } = useWebGLFilters();
  const clipsState = useClipStore((s) => s.clips);
  const tool = useViewportStore((s) => s.tool);
  const scale = useViewportStore((s) => s.scale);
  const position = useViewportStore((s) => s.position);
  const setClipTransform = useClipStore((s) => s.setClipTransform);
  const clipTransform = overrideClip
    ? overrideClip.transform
    : useClipStore((s) => s.getClipTransform(clipId));
  const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
  const clearSelection = useControlsStore((s) => s.clearSelection);
  const addClipSelection = useControlsStore((s) => s.addClipSelection);
  const { selectedClipIds, isFullscreen } = useControlsStore();
  const isSelected = useMemo(
    () => selectedClipIds.includes(clipId),
    [clipId, selectedClipIds],
  );
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
            m.inverted ? "inv" : "norm",
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

  const aspectRatio = useMemo(() => {
    const originalWidth = mediaInfoRef.current?.image?.width;
    const originalHeight = mediaInfoRef.current?.image?.height;
    if (!originalWidth || !originalHeight) return 16 / 9;
    const aspectRatio = originalWidth / originalHeight;
    return aspectRatio;
  }, [mediaInfoRef.current?.image?.width, mediaInfoRef.current?.image?.height]);

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
  const getAssetById = useClipStore((s) => s.getAssetById);

  const [, forceRerenderForMediaInfo] = useState(0);

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

  // Compute aspect-fit display size and offsets within the preview rect
  const { displayWidth, displayHeight, offsetX, offsetY } = useMemo(() => {
    const originalWidth = mediaInfoRef.current?.image?.width || 0;
    const originalHeight = mediaInfoRef.current?.image?.height || 0;
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
  }, [
    mediaInfoRef.current?.image?.height,
    mediaInfoRef.current?.image?.width,
    rectWidth,
    rectHeight,
  ]);

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
      const height = mediaInfoRef.current.image?.height;
      const width = mediaInfoRef.current.image?.width;
      const asset = getAssetById(selectedAssetId);
      if (!asset) return;
      const image = await fetchImage(asset.path, height, width, {
        mediaInfo: mediaInfoRef.current,
      });

      if (!image) return;

      let canvas = canvasRef.current;
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

      // If mask returned a different canvas, copy it back to working canvas to maintain single reference
      if (processedCanvas !== workingCanvas) {
        workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
        workingCtx.drawImage(
          processedCanvas,
          0,
          0,
          workingCanvas.width,
          workingCanvas.height,
        );
        processedCanvas = workingCanvas;
      }

      // Apply WebGL filters (modifies canvas in place)
      applyFilters(processedCanvas, {
        brightness: clip?.brightness,
        contrast: clip?.contrast,
        hue: clip?.hue,
        saturation: clip?.saturation,
        blur: clip?.blur,
        sharpness: clip?.sharpness,
        noise: clip?.noise,
        vignette: clip?.vignette,
      });

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
        if (result !== finalCanvas) {
          workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
          workingCtx.drawImage(
            result,
            0,
            0,
            workingCanvas.width,
            workingCanvas.height,
          );
          finalCanvas = workingCanvas;
        }
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
    // deselect all other clips
    clearSelection();
    addClipSelection(clipId);
  }, [addClipSelection, clipId, isFullscreen]);

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

  // Only mount Konva nodes when the clip is active for the current focus frame.
  if (!isInFrame) {
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
          draggable={tool === "pointer" && !isTransforming && !inputMode}
          ref={imageRef}
          cornerRadius={clipTransform?.cornerRadius ?? 0}
          opacity={(clipTransform?.opacity ?? 100) / 100}
          image={canvasRef.current || undefined}
          x={clipTransform?.x ?? offsetX}
          y={clipTransform?.y ?? offsetY}
          width={
            clipTransform?.width && clipTransform.width > 0
              ? clipTransform.width
              : displayWidth || 1
          }
          height={
            clipTransform?.height && clipTransform.height > 0
              ? clipTransform.height
              : displayHeight || 1
          }
          scaleX={clipTransform?.scaleX ?? 1}
          scaleY={clipTransform?.scaleY ?? 1}
          rotation={clipTransform?.rotation ?? 0}
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
          tool === "pointer" &&
          isSelected &&
          !isFullscreen &&
          overlap &&
          !inputMode
        }
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

export default ImagePreview;
