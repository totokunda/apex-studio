import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Stage,
  Layer,
  Group,
  Rect,
  Line as KonvaLine,
  Ellipse as KonvaEllipse,
  RegularPolygon,
  Star as KonvaStar,
  Circle,
  Transformer,
} from "react-konva";
import { useViewportStore } from "@/lib/viewport";
import { sortClipsForStacking } from "@/lib/clipOrdering";
import { getLocalFrame, useClipStore } from "@/lib/clip";
import { KonvaEventObject } from "konva/lib/Node";
import { BASE_LONG_SIDE, DEFAULT_FPS } from "@/lib/settings";

import _ from "lodash";
import VideoPreview from "./clips/VideoPreview";
import AudioPreview from "./clips/AudioPreview";
import ImagePreview from "./clips/ImagePreview";
import ShapePreview from "./clips/ShapePreview";
import TextPreview from "./clips/TextPreview";
import DrawingPreview from "./clips/DrawingPreview";
import MaskPreview from "./clips/MaskPreview";
import { useControlsStore } from "@/lib/control";
import {
  AnyClipProps,
  PolygonClipProps,
  ShapeClipProps,
  TextClipProps,
  DrawingClipProps,
  DrawingLine,
  PreprocessorClipType,
  MaskData,
  ClipTransform,
} from "@/lib/types";
import { v4 as uuidv4 } from "uuid";
import { SlSizeFullscreen } from "react-icons/sl";
import FullscreenPreview from "./FullscreenPreview";
import { getApplicatorsForClip } from "@/lib/applicator-utils";
import { useWebGLHaldClut } from "./webgl-filters";
import DynamicModelPreview from "./clips/DynamicModelPreview";
import { useDrawingStore } from "@/lib/drawing";
import { useMaskStore } from "@/lib/mask";
import {
  erasePolylineByEraser,
  transformEraserToLocal,
  mergeConnectedLines,
} from "@/lib/eraser-utils";
import {
  calculateArea,
  doPolygonsIntersect,
  isPolygonInsidePolygon,
  mergePolygons,
} from "@/lib/polygon";

const getFiniteNumber = (value: number | undefined, fallback = 0): number =>
  Number.isFinite(value) ? (value as number) : fallback;

// Compute the pixel offset introduced by a normalized crop on a clip transform.
// We intentionally use the clip's base width/height (pre-scale) to match how
// mask previews compensate for crop inside `MaskPreview`.
const getCropOffset = (
  transform?: ClipTransform | null,
): { offsetX: number; offsetY: number } => {
  if (!transform) {
    return { offsetX: 0, offsetY: 0 };
  }

  const anyTransform = transform as any;
  const crop = anyTransform?.crop as
    | { x?: number; y?: number; width?: number; height?: number }
    | undefined;

  const hasValidCrop =
    crop &&
    Number.isFinite(crop.x) &&
    Number.isFinite(crop.y) &&
    Number.isFinite(transform.width) &&
    Number.isFinite(transform.height);

  if (!hasValidCrop) {
    return { offsetX: 0, offsetY: 0 };
  }

  const cropW = crop!.width && crop!.width > 0 ? crop!.width : 1;
  const cropH = crop!.height && crop!.height > 0 ? crop!.height : 1;

  const baseWidth = (transform.width as number) || 0;
  const baseHeight = (transform.height as number) || 0;
  const offsetX = (baseWidth / cropW) * (crop!.x as number);
  const offsetY = (baseHeight / cropH) * (crop!.y as number);

  return {
    offsetX,
    offsetY,
  };
};

const normalizeMaskPointForRotation = (
  x: number,
  y: number,
  transform?: ClipTransform | null,
): { x: number; y: number } => {
  if (!transform) return { x, y };

  const { offsetX, offsetY } = getCropOffset(transform);
  const rotation = getFiniteNumber(transform.rotation);
  const originX = getFiniteNumber(transform.x);
  const originY = getFiniteNumber(transform.y);

  // No rotation: we only need to encode the crop offset into mask space
  if (Math.abs(rotation) < 1e-6) {
    return {
      x: x - originX + offsetX,
      y: y - originY + offsetY,
    };
  }

  const angleRad = (rotation * Math.PI) / 180;
  const dx = x - originX;
  const dy = y - originY;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  const unrotatedX = cos * dx + sin * dy;
  const unrotatedY = -sin * dx + cos * dy;

  return {
    x: unrotatedX + offsetX,
    y: unrotatedY + offsetY,
  };
};

const normalizeFlatPointsForRotation = (
  points: number[],
  transform?: ClipTransform | null,
): number[] => {
  if (!transform || !points.length) {
    return points.slice();
  }

  const normalized: number[] = new Array(points.length);
  for (let i = 0; i < points.length; i += 2) {
    const x = points[i];
    const y = points[i + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      normalized[i] = x;
      normalized[i + 1] = y;
      continue;
    }
    const point = normalizeMaskPointForRotation(x, y, transform);
    normalized[i] = point.x;
    normalized[i + 1] = point.y;
  }
  return normalized;
};

const denormalizeMaskPointForRotation = (
  x: number,
  y: number,
  transform?: ClipTransform | null,
): { x: number; y: number } => {
  if (!transform) return { x, y };

  // Inverse of `normalizeMaskPointForRotation`: first remove the crop offset
  // from mask-space, then re-apply rotation about the clip origin.
  const { offsetX, offsetY } = getCropOffset(transform);
  const baseX = x - offsetX;
  const baseY = y - offsetY;
  const originX = getFiniteNumber(transform.x);
  const originY = getFiniteNumber(transform.y);

  const rotation = getFiniteNumber(transform.rotation);
  if (Math.abs(rotation) < 1e-6) {
    return { x: baseX + originX, y: baseY + originY };
  }
  const angleRad = (rotation * Math.PI) / 180;
  const dx = baseX;
  const dy = baseY;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  const rotatedX = originX + cos * dx - sin * dy;
  const rotatedY = originY + sin * dx + cos * dy;
  return { x: rotatedX, y: rotatedY };
};

interface PreviewProps {}

const Preview: React.FC<PreviewProps> = () => {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<any>(null);
  const layerRef = useRef<any>(null);
  const aspectRectRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  const prevRectRef = useRef<{ w: number; h: number } | null>(null);
  const liveAspectUpdateRafRef = useRef<number | null>(null);
  const scale = useViewportStore((s) => s.scale);
  const position = useViewportStore((s) => s.position);
  const zoomAtScreenPoint = useViewportStore((s) => s.zoomAtScreenPoint);
  const panBy = useViewportStore((s) => s.panBy);
  const tool = useViewportStore((s) => s.tool);
  const shape = useViewportStore((s) => s.shape);
  const setViewportSize = useViewportStore((s) => s.setViewportSize);
  const setContentBounds = useViewportStore((s) => s.setContentBounds);
  
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const isAspectEditing = useViewportStore((s) => s.isAspectEditing);
  const shouldUpdateViewport = useViewportStore((s) => s.shouldUpdateViewport);
  
  const setAspectRatio = useViewportStore((s) => s.setAspectRatio);
  const {
    clips,
    clipWithinFrame,
    timelines,
    addClip,
    addTimeline,
    updateClip,
  } = useClipStore();
  const getClipTransform = useClipStore((s) => s.getClipTransform);
  // Note: we use imperative store access in the recenter effect to avoid rerender loops
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const totalTimelineFrames = useControlsStore((s) => s.totalTimelineFrames);
  // Shape creation state
  const [isDrawingShape, setIsDrawingShape] = useState(false);
  const [isDrawingText, setIsDrawingText] = useState(false);
  const [textStart, setTextStart] = useState<{ x: number; y: number } | null>(
    null,
  );
  const [textCurrent, setTextCurrent] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [shapeStart, setShapeStart] = useState<{ x: number; y: number } | null>(
    null,
  );
  const [shapeCurrent, setShapeCurrent] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const isFullscreen = useControlsStore((s) => s.isFullscreen);
  const setIsFullscreen = useControlsStore((s) => s.setIsFullscreen);
  const haldClutInstance = useWebGLHaldClut();
  const setSelectedMaskId = useControlsStore((s) => s.setSelectedMaskId);
  const setSelectedClipIds = useControlsStore((s) => s.setSelectedClipIds);
  const selectedClipIds = useControlsStore((s) => s.selectedClipIds);
  const {
    tool: drawingTool,
    color: drawingColor,
    opacity: drawingOpacity,
    smoothing: drawingSmoothing,
    getCurrentSize,
    setCurrentLineId,
  } = useDrawingStore();
  const isDrawingRef = useRef(false);
  const eraserPointsRef = useRef<Array<{ x: number; y: number }>>([]);
  const lastEraserPosRef = useRef<{ x: number; y: number } | null>(null);
  const [cursorPreview, setCursorPreview] = useState<{
    x: number;
    y: number;
  } | null>(null);
  // Temp buffered drawing (brush/highlighter) before committing on mouse up
  const [tempDrawingPoints, setTempDrawingPoints] = useState<number[] | null>(
    null,
  );
  const tempDrawingStyleRef = useRef<{
    tool: "brush" | "highlighter";
    stroke: string;
    strokeWidth: number;
    opacity: number;
    smoothing: any;
  } | null>(null);
  // Temp erased lines preview for eraser (immediate visual, persist on mouse up)
  const [tempErased, setTempErased] = useState<{
    clipId: string;
    lines: DrawingLine[];
  } | null>(null);

  // Mask tool state
  const maskTool = useMaskStore((s) => s.tool);
  const maskShape = useMaskStore((s) => s.shape);
  const maskBrushSize = useMaskStore((s) => s.brushSize);
  const featherAmount = useMaskStore((s) => s.featherAmount);
  const touchLabel = useMaskStore((s) => s.touchLabel);
  const touchDrawMode = useMaskStore((s) => s.touchDrawMode);
  const setTouchMaskRefetchToken = useMaskStore(
    (s) => s.setTouchMaskRefetchToken,
  );
  const [isDrawingLasso, setIsDrawingLasso] = useState(false);
  const [lassoPoints, setLassoPoints] = useState<number[]>([]);
  const [isDrawingMaskRect, setIsDrawingMaskRect] = useState(false);
  const [maskRectStart, setMaskRectStart] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [maskRectCurrent, setMaskRectCurrent] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [isDrawingMaskBrush, setIsDrawingMaskBrush] = useState(false);
  const [maskBrushPoints, setMaskBrushPoints] = useState<number[]>([]);
  const [maskCursorPreview, setMaskCursorPreview] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [isDrawingTouchLasso, setIsDrawingTouchLasso] = useState(false);
  const [touchLassoPoints, setTouchLassoPoints] = useState<number[]>([]);
  const isMaskDragging = useMaskStore((s) => s.isMaskDragging);
  const isOverMask = useMaskStore((s) => s.isOverMask);
  const maskShapeTargetClipRef = useRef<PreprocessorClipType | null>(null);
  const maskShapeTransformRef = useRef<ClipTransform | null>(null);
  const maskRectStartNormalizedRef = useRef<{ x: number; y: number } | null>(
    null,
  );

  const resetMaskShapeDrawingRefs = useCallback(() => {
    maskShapeTargetClipRef.current = null;
    maskShapeTransformRef.current = null;
    maskRectStartNormalizedRef.current = null;
  }, []);
  // Memoize applicator factory configuration
  const applicatorConfig = useMemo(
    () => ({
      haldClutInstance,
      // Add more shared resources here as needed
    }),
    [haldClutInstance],
  );

  // Get applicators for a specific clip using the factory
  const getClipApplicators = useCallback(
    (clipId: string) => {
      return getApplicatorsForClip(clipId, applicatorConfig);
    },
    [applicatorConfig],
  );

  // Preload all resources for applicator clips
  useEffect(() => {
    if (!haldClutInstance) return;

    // Preload filter CLUTs
    const filterClips = clips.filter((c) => c.type === "filter");

    if (filterClips.length === 0) {
      return;
    }

    let loadedCount = 0;
    const loadPromises = filterClips.map(async (clip: any) => {
      const filterPath = clip.fullPath || clip.smallPath;
      if (filterPath) {
        try {
          await haldClutInstance.preloadClut(filterPath);
          loadedCount++;
        } catch (e) {
          console.warn("Failed to preload CLUT:", filterPath, e);
        }
      }
    });

    // Add preloading for other applicator types here as they are added
    // e.g., mask textures, processor models, etc.

    Promise.all(loadPromises).catch(console.error);
  }, [clips, haldClutInstance]);

  const sortClips = useCallback(
    (inputClips: AnyClipProps[]) => {
      return sortClipsForStacking(inputClips, timelines);
    },
    [timelines],
  );

  const filterClips = useCallback(
    (clips: AnyClipProps[], audio: boolean = false) => {
      const filteredClips = clips.filter((clip: AnyClipProps) => {
        const timeline = timelines.find(
          (t) => t.timelineId === clip.timelineId,
        );
        if (audio) {
          if (timeline?.muted) {
            return false;
          }
        }
        if (timeline?.hidden) {
          return false;
        }
        if (clip.groupId) {
          const clipGroup = clips.find(
            (c) => c.clipId === clip.groupId,
          );
          const groupTimeline = timelines.find(
            (t) => t.timelineId === clipGroup?.timelineId,
          );
          if (groupTimeline?.hidden) {
            return false;
          }
        }
        return true;
      });

      return filteredClips;
    },
    [timelines],
  );

  // Find the topmost rendered clip at the current focus frame
  const getClipForMask = useCallback(() => {
    // Or get currently selected clip
    if (selectedClipIds.length > 0) {
      // get the last clip
      const lastClipId = selectedClipIds[selectedClipIds.length - 1];
      const lastClip = clips.find(
        (clip: AnyClipProps) => clip.clipId === lastClipId,
      );
      if (lastClip) {
        return lastClip;
      }
    }
    const visibleClips = sortClips(filterClips(clips)).filter(
      (clip: AnyClipProps) => {
        const clipAtFrame = clipWithinFrame(clip, focusFrame);
        // Only consider clips that can have masks (video and image)
        const timeline = timelines.find(
          (t) => t.timelineId === clip.timelineId,
        );
        if (timeline?.hidden) return false;
        return clipAtFrame && (clip.type === "video" || clip.type === "image");
      },
    );

    // Return the first clip (topmost due to sortClips)
    return visibleClips.length > 0
      ? visibleClips[visibleClips.length - 1]
      : null;
  }, [
    clips,
    timelines,
    focusFrame,
    sortClips,
    filterClips,
    clipWithinFrame,
    timelines,
  ]);

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
        setViewportSize({ width, height });
      }, 16); // ~60fps throttle
    });

    observer.observe(el);
    return () => {
      clearTimeout(timeoutId);
      observer.disconnect();
    };
  }, [containerRef.current?.offsetWidth, containerRef.current?.offsetHeight]);

  // Force size update when exiting fullscreen
  useEffect(() => {
    if (!isFullscreen && containerRef.current) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        if (containerRef.current) {
          const el = containerRef.current;
          const rect = el.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) {
            setSize({ width: rect.width, height: rect.height });
            setViewportSize({ width: rect.width, height: rect.height });
          }
        }
      });
    }
  }, [isFullscreen, setViewportSize]);

  // Compute rect dimensions based on aspect ratio
  const { rectWidth, rectHeight } = useMemo(() => {
    const ratio = aspectRatio.width / aspectRatio.height;
    const baseShortSide = BASE_LONG_SIDE; // keep short side constant
    if (!Number.isFinite(ratio) || ratio <= 0) {
      return { rectWidth: 0, rectHeight: 0 };
    }
    // Use a fixed height for consistency; scale width by aspect ratio.
    // This keeps portrait sizes from becoming excessively tall.
    return { rectWidth: baseShortSide * ratio, rectHeight: baseShortSide };
  }, [aspectRatio.width, aspectRatio.height]);

  // Helper: recenter all clips for a given rect size
  const recenterAllClips = useCallback(
    (targetRectWidth: number, targetRectHeight: number) => {
      const { clips: currentClips } = useClipStore.getState();
      currentClips.forEach((clip: AnyClipProps) => {
        const t = useClipStore.getState().getClipTransform(clip.clipId);
        const w = t?.width ?? 0;
        const h = t?.height ?? 0;
        const cx = Math.max(0, (targetRectWidth - w) / 2);
        const cy = Math.max(0, (targetRectHeight - h) / 2);
        useClipStore.getState().setClipTransform(clip.clipId, { x: cx, y: cy });
      });
    },
    [],
  );

  // Center the rect initially and whenever aspect ratio or viewport changes
  useEffect(() => {
    if (!stageRef.current || isFullscreen || isAspectEditing) return;
    if (!shouldUpdateViewport)  {
      return;
    }
    const rectBounds = { x: 0, y: 0, width: rectWidth, height: rectHeight };
    setContentBounds(rectBounds);
    const rectWorld = {
      x: rectBounds.x + rectBounds.width / 2,
      y: rectBounds.y + rectBounds.height / 2,
    };
    useViewportStore
      .getState()
      .centerOnWorldPoint(rectWorld, {
        width: size.width,
        height: size.height,
      });
  }, [
    size.width,
    size.height,
    rectWidth,
    rectHeight,
    isFullscreen,
    isAspectEditing,
  ]);

  // Attach transformer to aspect rect when editing (attach once when toggled)
  useEffect(() => {
    if (!isAspectEditing) return;
    const tr = transformerRef.current;
    const node = aspectRectRef.current;
    if (tr && node) {
      tr.nodes([node]);
      tr.getLayer()?.batchDraw();
    }
  }, [isAspectEditing]);

  // Keep transformer visuals updated when dimensions change without reattaching
  useEffect(() => {
    const tr = transformerRef.current;
    if (tr) {
      tr.getLayer()?.batchDraw();
    }
  }, [rectWidth, rectHeight]);

  // Cleanup any pending rAF when leaving aspect edit mode
  useEffect(() => {
    if (!isAspectEditing && liveAspectUpdateRafRef.current) {
      cancelAnimationFrame(liveAspectUpdateRafRef.current);
      liveAspectUpdateRafRef.current = null;
    }
  }, [isAspectEditing]);

  // Recenter all clips when rect size changes (but not on initial render), skip while editing to prevent jitter
  useEffect(() => {
    if (!rectWidth || !rectHeight || isFullscreen || isAspectEditing) return;
    const prev = prevRectRef.current;
    if (!prev || prev.w === 0 || prev.h === 0) {
      prevRectRef.current = { w: rectWidth, h: rectHeight };
      return; // skip initial mount
    }
    if (prev.w === rectWidth && prev.h === rectHeight) return; // no change

    recenterAllClips(rectWidth, rectHeight);

    prevRectRef.current = { w: rectWidth, h: rectHeight };
  }, [rectWidth, rectHeight, isFullscreen, isAspectEditing, recenterAllClips]);

  const handleWheel = useCallback(
    (e: any) => {
      e.evt.preventDefault();
      const isZoom = e.evt.ctrlKey || e.evt.metaKey; // ctrl (win) or cmd (mac)
      const { offsetX, offsetY, deltaY, deltaX } = e.evt;
      if (isZoom) {
        const zoomIntensity = 0.0015; // sensitivity
        const nextScale = scale * (1 - deltaY * zoomIntensity);
        zoomAtScreenPoint(nextScale, { x: offsetX, y: offsetY });
      } else {
        // pan with wheel: shift to horizontal when shiftKey is pressed
        const dx = e.evt.shiftKey ? deltaY : deltaX;
        const dy = e.evt.shiftKey ? 0 : deltaY;
        panBy(dx, dy);
      }
    },
    [scale, zoomAtScreenPoint, panBy],
  );

  const isPanning = useRef(false);
  const lastPointer = useRef<{ x: number; y: number } | null>(null);
  const currentDrawingClipId = useRef<string | null>(null);
  const activeDrawingToolRef = useRef<
    "brush" | "highlighter" | "eraser" | null
  >(null);

  const onMouseDown = useCallback(
    (e: any) => {
      if (tool === "hand") {
        isPanning.current = true;
        lastPointer.current = { x: e.evt.offsetX, y: e.evt.offsetY };
        return;
      }

      if (tool === "mask" && maskTool === "lasso") {
        const stage = e.target.getStage();
        if (!stage) return;

        if (isMaskDragging || isOverMask) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        setIsDrawingLasso(true);
        setLassoPoints([worldX, worldY]);
        return;
      }

      if (tool === "mask" && maskTool === "shape") {
        const stage = e.target.getStage();
        if (!stage) return;

        if (isMaskDragging || isOverMask) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;
        const targetClip = getClipForMask() as PreprocessorClipType;

        if (
          targetClip &&
          (targetClip.type === "video" || targetClip.type === "image")
        ) {
          maskShapeTargetClipRef.current = targetClip;
          const clipTransform =
            getClipTransform(targetClip.clipId) ?? targetClip.transform;
          maskShapeTransformRef.current = clipTransform || null;
          maskRectStartNormalizedRef.current = clipTransform
            ? normalizeMaskPointForRotation(worldX, worldY, clipTransform)
            : { x: worldX, y: worldY };
        } else {
          maskShapeTargetClipRef.current = null;
          maskShapeTransformRef.current = null;
          maskRectStartNormalizedRef.current = { x: worldX, y: worldY };
        }

        setIsDrawingMaskRect(true);
        setMaskRectStart({ x: worldX, y: worldY });
        setMaskRectCurrent({ x: worldX, y: worldY });
        return;
      }

      if (tool === "mask" && maskTool === "draw") {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        setIsDrawingMaskBrush(true);
        setMaskBrushPoints([worldX, worldY]);
        return;
      }

      if (tool === "mask" && maskTool === "touch") {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        // Handle draw mode: start lasso drawing
        if (touchDrawMode === "draw") {
          setIsDrawingTouchLasso(true);
          setTouchLassoPoints([worldX, worldY]);
          return;
        }

        // Handle point mode: add touch points
        // Find the topmost clip at the current frame
        const targetClip = getClipForMask() as PreprocessorClipType;

        if (
          !targetClip ||
          (targetClip.type !== "video" && targetClip.type !== "image")
        )
          return;

        const localFrame =
          targetClip.type === "image"
            ? 0
            : getLocalFrame(focusFrame, targetClip);
        const clipTransform =
          getClipTransform(targetClip.clipId) ?? targetClip.transform;
        const normalizedPoint = normalizeMaskPointForRotation(
          worldX,
          worldY,
          clipTransform,
        );

        // Get existing masks
        const currentMasks = targetClip.masks || [];

        // Check if there's already a touch mask at this frame for this clip
        let existingTouchMask = currentMasks.find(
          (mask) => mask.tool === "touch",
        );

        if (existingTouchMask) {
          // Add point to existing mask
          const keyframes =
            existingTouchMask.keyframes instanceof Map
              ? existingTouchMask.keyframes
              : (existingTouchMask.keyframes as Record<number, MaskData>);

          const currentFrameData =
            keyframes instanceof Map
              ? keyframes.get(localFrame)
              : keyframes[localFrame];

          const touchPoints = currentFrameData?.touchPoints || [];
          // Add new point with label
          const updatedTouchPoints = [
            ...touchPoints,
            {
              x: normalizedPoint.x,
              y: normalizedPoint.y,
              label: touchLabel === "positive" ? (1 as const) : (0 as const),
            },
          ];

          // Update keyframes
          if (keyframes instanceof Map) {
            keyframes.set(localFrame, { touchPoints: updatedTouchPoints });
            existingTouchMask.keyframes = keyframes;
          } else {
            existingTouchMask.keyframes = {
              ...keyframes,
              [localFrame]: { touchPoints: updatedTouchPoints },
            };
          }

          existingTouchMask.lastModified = Date.now();
          // Signal touch mask preview to refetch mask for updated points
          setTouchMaskRefetchToken(Date.now());
          updateClip(targetClip.clipId, {
            masks: currentMasks,
          });

          // Ensure the related clip and mask are selected
          setSelectedClipIds([targetClip.clipId]);
          setSelectedMaskId(existingTouchMask.id);
        } else {
          // Create new touch mask

          const newMask = {
            id: uuidv4(),
            clipId: targetClip.clipId,
            tool: "touch" as const,
            featherAmount: featherAmount,
            keyframes: {
              [localFrame]: {
                touchPoints: [
                  {
                    x: normalizedPoint.x,
                    y: normalizedPoint.y,
                    label:
                      touchLabel === "positive" ? (1 as const) : (0 as const),
                  },
                ],
              },
            },
            isTracked: false,
            createdAt: Date.now(),
            lastModified: Date.now(),
            maskColor: "#FFFFFF",
            maskOpacity: 100,
            maskColorEnabled: true,
            backgroundColor: "#000000",
            backgroundOpacity: 100,
            backgroundColorEnabled: true,
            transform: clipTransform ? { ...clipTransform } : undefined,
          };

          currentMasks.push(newMask);

          updateClip(targetClip.clipId, {
            masks: currentMasks,
          });

          // Signal touch mask preview to refetch mask for new mask points
          setTouchMaskRefetchToken(Date.now());

          // Ensure the related clip and mask are selected
          setSelectedClipIds([targetClip.clipId]);
          setSelectedMaskId(newMask.id);
        }

        return;
      }

      if (tool === "shape") {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        setIsDrawingShape(true);
        setShapeStart({ x: worldX, y: worldY });
        setShapeCurrent({ x: worldX, y: worldY });
        return;
      }

      if (tool === "text") {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        setIsDrawingText(true);
        setTextStart({ x: worldX, y: worldY });
        setTextCurrent({ x: worldX, y: worldY });
        return;
      }

      if (tool === "draw") {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        isDrawingRef.current = true;
        activeDrawingToolRef.current = drawingTool;

        // Check if there's already a drawing clip at the current frame
        const existingDrawingClip = clips.find(
          (c) => c.type === "draw" && clipWithinFrame(c, focusFrame),
        ) as DrawingClipProps | undefined;

        // Handle eraser tool separately - live erasing, no buffering
        if (drawingTool === "eraser") {
          if (!existingDrawingClip) {
            // Eraser needs existing clip to work with
            isDrawingRef.current = false;
            currentDrawingClipId.current = null;
            activeDrawingToolRef.current = null;
            return;
          }

          currentDrawingClipId.current = existingDrawingClip.clipId;
          eraserPointsRef.current = [{ x: worldX, y: worldY }];
          lastEraserPosRef.current = { x: worldX, y: worldY };
          return;
        }

        // Buffer brush/highlighter points and style; commit on mouse up
        setTempErased(null);
        const lineId = uuidv4();
        setCurrentLineId(lineId);
        tempDrawingStyleRef.current = {
          tool: drawingTool === "highlighter" ? "highlighter" : "brush",
          stroke: drawingTool === "highlighter" ? "#FFFF00" : drawingColor,
          strokeWidth: getCurrentSize(),
          opacity: drawingTool === "highlighter" ? 50 : drawingOpacity,
          smoothing: drawingSmoothing,
        };
        setTempDrawingPoints([worldX, worldY]);
        return;
      }
    },
    [
      tool,
      position,
      scale,
      clips,
      clipWithinFrame,
      focusFrame,
      drawingTool,
      drawingColor,
      drawingOpacity,
      getCurrentSize,
      setCurrentLineId,
      isMaskDragging,
      isOverMask,
      timelines,
      addTimeline,
      addClip,
      totalTimelineFrames,
      maskTool,
      touchLabel,
      touchDrawMode,
      featherAmount,
      getClipForMask,
    ],
  );

  const onMouseMove = useCallback(
    (e: any) => {
      // Handle lasso drawing
      if (tool === "mask" && maskTool === "lasso" && isDrawingLasso) {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        // Add point to lasso path
        setLassoPoints((prev) => [...prev, worldX, worldY]);
        return;
      }

      // Handle shape mask drawing
      if (
        tool === "mask" &&
        maskTool === "shape" &&
        isDrawingMaskRect &&
        maskRectStart
      ) {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;
        const clipTransform = maskShapeTransformRef.current;
        const startNormalized =
          maskRectStartNormalizedRef.current ??
          (clipTransform
            ? normalizeMaskPointForRotation(
                maskRectStart.x,
                maskRectStart.y,
                clipTransform,
              )
            : { x: maskRectStart.x, y: maskRectStart.y });
        maskRectStartNormalizedRef.current = startNormalized;

        if (clipTransform) {
          let normalizedCurrent = normalizeMaskPointForRotation(
            worldX,
            worldY,
            clipTransform,
          );

          if (maskShape === "star") {
            const dx = normalizedCurrent.x - startNormalized.x;
            const dy = normalizedCurrent.y - startNormalized.y;
            const size = Math.max(Math.abs(dx), Math.abs(dy));
            const sx = dx >= 0 ? 1 : -1;
            const sy = dy >= 0 ? 1 : -1;
            normalizedCurrent = {
              x: startNormalized.x + sx * size,
              y: startNormalized.y + sy * size,
            };
          } else if (maskShape === "polygon") {
            const dx = normalizedCurrent.x - startNormalized.x;
            const dy = normalizedCurrent.y - startNormalized.y;
            const ratio = 1.1543665517482078;
            let width = Math.abs(dx);
            let height = Math.abs(dy);
            if (Math.abs(height) < 1e-6) {
              height = width / ratio;
            } else if (Math.abs(width) < 1e-6) {
              width = height * ratio;
            }
            if (width / height >= ratio) {
              height = width / ratio;
            } else {
              width = height * ratio;
            }
            const sx = dx >= 0 ? 1 : -1;
            const sy = dy >= 0 ? 1 : -1;
            normalizedCurrent = {
              x: startNormalized.x + sx * width,
              y: startNormalized.y + sy * height,
            };
          }

          const constrainedWorld = denormalizeMaskPointForRotation(
            normalizedCurrent.x,
            normalizedCurrent.y,
            clipTransform,
          );
          setMaskRectCurrent(constrainedWorld);
        } else {
          if (maskShape === "star") {
            const dx = worldX - maskRectStart.x;
            const dy = worldY - maskRectStart.y;
            const size = Math.max(Math.abs(dx), Math.abs(dy));
            const sx = dx >= 0 ? 1 : -1;
            const sy = dy >= 0 ? 1 : -1;
            const constrainedX = maskRectStart.x + sx * size;
            const constrainedY = maskRectStart.y + sy * size;
            setMaskRectCurrent({ x: constrainedX, y: constrainedY });
          } else if (maskShape === "polygon") {
            const dx = worldX - maskRectStart.x;
            const dy = worldY - maskRectStart.y;
            const ratio = 1.1543665517482078;
            let width = Math.abs(dx);
            let height = Math.abs(dy);
            if (width / (height || 1) >= ratio) {
              height = width / ratio;
            } else {
              width = height * ratio;
            }
            const sx = dx >= 0 ? 1 : -1;
            const sy = dy >= 0 ? 1 : -1;
            const constrainedX = maskRectStart.x + sx * width;
            const constrainedY = maskRectStart.y + sy * height;
            setMaskRectCurrent({ x: constrainedX, y: constrainedY });
          } else {
            setMaskRectCurrent({ x: worldX, y: worldY });
          }
        }
        return;
      }

      // Handle brush mask drawing
      if (tool === "mask" && maskTool === "draw" && isDrawingMaskBrush) {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        // Add point to brush path only if it's far enough from the last point
        setMaskBrushPoints((prev) => {
          if (prev.length < 2) return [...prev, worldX, worldY];

          const lastX = prev[prev.length - 2];
          const lastY = prev[prev.length - 1];
          const dx = worldX - lastX;
          const dy = worldY - lastY;
          const distance = Math.sqrt(dx * dx + dy * dy);

          // Only add point if it's at least 1 pixel away from the last point
          if (distance >= 1) {
            return [...prev, worldX, worldY];
          }
          return prev;
        });
        return;
      }

      // Handle touch lasso drawing
      if (
        tool === "mask" &&
        maskTool === "touch" &&
        touchDrawMode === "draw" &&
        isDrawingTouchLasso
      ) {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        // Add point to touch lasso path
        setTouchLassoPoints((prev) => [...prev, worldX, worldY]);
        return;
      }

      // Update cursor preview for mask draw tool
      if (tool === "mask" && maskTool === "draw") {
        const stage = e.target.getStage();
        if (stage) {
          const pointerPos = stage.getPointerPosition();
          if (pointerPos) {
            const worldX = (pointerPos.x - position.x) / scale;
            const worldY = (pointerPos.y - position.y) / scale;
            setMaskCursorPreview({ x: worldX, y: worldY });
          }
        }
      } else {
        setMaskCursorPreview(null);
      }

      // Update cursor preview for draw tool
      if (tool === "draw") {
        const stage = e.target.getStage();
        if (stage) {
          const pointerPos = stage.getPointerPosition();
          if (pointerPos) {
            const worldX = (pointerPos.x - position.x) / scale;
            const worldY = (pointerPos.y - position.y) / scale;
            setCursorPreview({ x: worldX, y: worldY });
          }
        }
      } else {
        setCursorPreview(null);
      }

      if (tool === "hand" && isPanning.current) {
        const current = { x: e.evt.offsetX, y: e.evt.offsetY };
        const last = lastPointer.current || current;
        const dx = current.x - last.x;
        const dy = current.y - last.y;
        panBy(-dx, -dy);
        lastPointer.current = current;
        return;
      }

      if (tool === "shape" && isDrawingShape && shapeStart) {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        let worldX = (pointerPos.x - position.x) / scale;
        let worldY = (pointerPos.y - position.y) / scale;

        // Enforce constraints for star and polygon shape tools
        if (shape === "star") {
          const dx = worldX - shapeStart.x;
          const dy = worldY - shapeStart.y;
          const size = Math.max(Math.abs(dx), Math.abs(dy));
          const sx = dx >= 0 ? 1 : -1;
          const sy = dy >= 0 ? 1 : -1;
          worldX = shapeStart.x + sx * size;
          worldY = shapeStart.y + sy * size;
        } else if (shape === "polygon") {
          const dx = worldX - shapeStart.x;
          const dy = worldY - shapeStart.y;
          const ratio = 1.1543665517482078;
          let width = Math.abs(dx);
          let height = Math.abs(dy);
          if (width / (height || 1) >= ratio) {
            height = width / ratio;
          } else {
            width = height * ratio;
          }
          const sx = dx >= 0 ? 1 : -1;
          const sy = dy >= 0 ? 1 : -1;
          worldX = shapeStart.x + sx * width;
          worldY = shapeStart.y + sy * height;
        }

        setShapeCurrent({ x: worldX, y: worldY });
      }
      if (tool === "text" && isDrawingText && textStart) {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        setTextCurrent({ x: worldX, y: worldY });
      }

      if (tool === "draw" && isDrawingRef.current) {
        const stage = e.target.getStage();
        if (!stage) return;

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) return;

        // Convert screen coordinates to world coordinates
        const worldX = (pointerPos.x - position.x) / scale;
        const worldY = (pointerPos.y - position.y) / scale;

        // Handle eraser tool - compute immediate visual preview only; persist on mouse up
        if (activeDrawingToolRef.current === "eraser") {
          // Interpolate points between last position and current position to avoid gaps
          const lastPos = lastEraserPosRef.current;
          if (lastPos) {
            const dx = worldX - lastPos.x;
            const dy = worldY - lastPos.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > 2) {
              const steps = Math.ceil(dist / 2);
              for (let i = 1; i <= steps; i++) {
                const t = i / steps;
                eraserPointsRef.current.push({
                  x: lastPos.x + dx * t,
                  y: lastPos.y + dy * t,
                });
              }
            } else {
              eraserPointsRef.current.push({ x: worldX, y: worldY });
            }
          } else {
            eraserPointsRef.current.push({ x: worldX, y: worldY });
          }
          lastEraserPosRef.current = { x: worldX, y: worldY };

          // Compute temporary erased result for visual feedback without persisting
          const drawingClipId = currentDrawingClipId.current;
          if (drawingClipId) {
            const drawingClip = useClipStore
              .getState()
              .getClipById(drawingClipId) as DrawingClipProps | undefined;
            if (drawingClip && drawingClip.lines) {
              const baseLines: DrawingLine[] =
                tempErased && tempErased.clipId === drawingClip.clipId
                  ? tempErased.lines
                  : drawingClip.lines;
              const eraserBaseRadius = getCurrentSize() / 2;
              const worldEraserPts = eraserPointsRef.current;
              const updatedLines: DrawingLine[] = [];
              for (const line of baseLines) {
                if (line.tool !== "brush" && line.tool !== "highlighter") {
                  updatedLines.push(line);
                  continue;
                }
                const effectiveEraserRadius =
                  eraserBaseRadius + line.strokeWidth / 2;
                const { localPts, localRadius } = transformEraserToLocal(
                  worldEraserPts,
                  line.transform,
                  effectiveEraserRadius,
                );
                const splitPolylines = erasePolylineByEraser(
                  line.points,
                  localPts,
                  localRadius,
                );
                for (const polyline of splitPolylines) {
                  if (polyline.length >= 4) {
                    updatedLines.push({
                      ...line,
                      lineId: uuidv4(),
                      points: polyline,
                    });
                  }
                }
              }
              const mergedLines = mergeConnectedLines(updatedLines, false);
              setTempErased({ clipId: drawingClip.clipId, lines: mergedLines });
            }
          }
          return;
        }

        // Buffer brush/highlighter: just extend temp points
        setTempDrawingPoints((prev) =>
          prev ? [...prev, worldX, worldY] : [worldX, worldY],
        );
      }
    },
    [
      panBy,
      tool,
      isDrawingShape,
      shapeStart,
      position,
      scale,
      isDrawingText,
      textStart,
      drawingTool,
      getCurrentSize,
      cursorPreview,
      maskTool,
      isDrawingLasso,
      isDrawingMaskRect,
      maskRectStart,
      isDrawingMaskBrush,
      touchDrawMode,
      isDrawingTouchLasso,
    ],
  );

  const onMouseUp = useCallback(
    (e: any) => {
      if (tool === "hand") {
        isPanning.current = false;
        lastPointer.current = null;
        return;
      }

      if (
        tool === "mask" &&
        maskTool === "lasso" &&
        isDrawingLasso &&
        lassoPoints.length >= 6 &&
        !isMaskDragging
      ) {
        // Need at least 3 points (6 values) to create a valid lasso
        setIsDrawingLasso(false);

        // Find the topmost clip at the current frame
        const targetClip = getClipForMask() as PreprocessorClipType;

        if (
          !targetClip ||
          (targetClip.type !== "video" && targetClip.type !== "image")
        )
          return;

        const localFrame =
          targetClip.type === "image"
            ? 0
            : getLocalFrame(focusFrame, targetClip);
        const clipTransform =
          getClipTransform(targetClip.clipId) ?? targetClip.transform;
        const normalizedLasso = normalizeFlatPointsForRotation(
          lassoPoints,
          clipTransform,
        );
        // Create the mask
        const newMask = {
          id: uuidv4(),
          clipId: targetClip.clipId,
          tool: "lasso" as const,
          featherAmount: featherAmount,
          keyframes: {
            [localFrame]: {
              lassoPoints: normalizedLasso,
            },
          },
          isTracked: false,
          createdAt: Date.now(),
          lastModified: Date.now(),
          maskColor: "#FFFFFF",
          maskOpacity: 100,
          maskColorEnabled: true,
          backgroundColor: "#000000",
          backgroundOpacity: 100,
          backgroundColorEnabled: true,
          transform: clipTransform ? { ...clipTransform } : undefined,
        };

        // Get existing masks and check for containment
        const currentMasks = targetClip.masks || [];
        let masksToKeep = [...currentMasks];

        const newMaskArea = calculateArea(normalizedLasso);

        // Track masks that intersect with the new mask for merging
        const intersectingMasks: any[] = [];
        let mergedPoints = [...normalizedLasso];

        // Check each existing mask at this frame for containment and intersection
        masksToKeep = currentMasks.filter((existingMask) => {
          if (existingMask.tool !== "lasso") {
            return true; // Keep non-lasso masks for now
          }

          // Get the lasso points for this mask at this frame
          const keyframes =
            existingMask.keyframes instanceof Map
              ? existingMask.keyframes
              : (existingMask.keyframes as Record<number, any>);

          const keyframeNumbers =
            keyframes instanceof Map
              ? Array.from(keyframes.keys()).sort((a, b) => a - b)
              : Object.keys(keyframes)
                  .map(Number)
                  .sort((a, b) => a - b);

          const activeKeyframe = keyframeNumbers
            .filter((k) => k <= localFrame)
            .pop();

          if (activeKeyframe === undefined) {
            return true; // Keep if no keyframe at this frame
          }

          const maskData =
            keyframes instanceof Map
              ? keyframes.get(activeKeyframe)
              : keyframes[activeKeyframe];

          if (!maskData?.lassoPoints) {
            return true; // Keep if no lasso points
          }

          const existingPoints = maskData.lassoPoints;
          const existingArea = calculateArea(existingPoints);

          // Check containment
          const newInsideExisting = isPolygonInsidePolygon(
            mergedPoints,
            existingPoints,
          );
          const existingInsideNew = isPolygonInsidePolygon(
            existingPoints,
            mergedPoints,
          );

          if (newInsideExisting || existingInsideNew) {
            // One mask is inside the other - keep the larger one
            if (existingArea > newMaskArea) {
              return true; // Keep existing, discard new
            } else {
              return false; // Discard existing, keep new
            }
          }

          // Check intersection
          if (doPolygonsIntersect(mergedPoints, existingPoints)) {
            // Masks intersect - merge them
            intersectingMasks.push(existingMask);
            mergedPoints = mergePolygons(mergedPoints, existingPoints);
            return false; // Remove this mask, it will be merged into new mask
          }

          return true; // Keep if no containment or intersection
        });

        // Update the new mask with merged points if any masks were merged
        if (intersectingMasks.length > 0) {
          newMask.keyframes[localFrame].lassoPoints = mergedPoints;
        }

        // Add the new mask (which may now be a merged mask)
        masksToKeep.push(newMask);

        updateClip(targetClip.clipId, {
          masks: masksToKeep,
        });

        if (masksToKeep.length > 0) {
          setSelectedMaskId(newMask.id);
        }

        // Clear lasso points immediately since mask rendering will handle display
        setLassoPoints([]);

        return;
      }

      // If lasso was being drawn but not enough points, just clear
      if (tool === "mask" && maskTool === "lasso" && isDrawingLasso) {
        setIsDrawingLasso(false);
        setLassoPoints([]);
        return;
      }

      // Handle shape mask creation
      if (
        tool === "mask" &&
        maskTool === "shape" &&
        isDrawingMaskRect &&
        maskRectStart &&
        maskRectCurrent
      ) {
        setIsDrawingMaskRect(false);

        const storedClip = maskShapeTargetClipRef.current;
        const candidateClip =
          storedClip &&
          (storedClip.type === "video" || storedClip.type === "image")
            ? storedClip
            : (getClipForMask() as PreprocessorClipType);
        const targetClip =
          candidateClip &&
          (candidateClip.type === "video" || candidateClip.type === "image")
            ? candidateClip
            : null;

        if (!targetClip) {
          resetMaskShapeDrawingRefs();
          setMaskRectStart(null);
          setMaskRectCurrent(null);
          return;
        }

        const storedTransform = maskShapeTransformRef.current;
        const clipTransform =
          storedClip &&
          storedTransform &&
          storedClip.clipId === targetClip.clipId
            ? storedTransform
            : (getClipTransform(targetClip.clipId) ?? targetClip.transform);
        const worldMinX = Math.min(maskRectStart.x, maskRectCurrent.x);
        const worldMaxX = Math.max(maskRectStart.x, maskRectCurrent.x);
        const worldMinY = Math.min(maskRectStart.y, maskRectCurrent.y);
        const worldMaxY = Math.max(maskRectStart.y, maskRectCurrent.y);
        const normalizedCorners = [
          normalizeMaskPointForRotation(worldMinX, worldMinY, clipTransform),
          normalizeMaskPointForRotation(worldMaxX, worldMinY, clipTransform),
          normalizeMaskPointForRotation(worldMinX, worldMaxY, clipTransform),
          normalizeMaskPointForRotation(worldMaxX, worldMaxY, clipTransform),
        ];
        const xs = normalizedCorners.map((corner) => corner.x);
        const ys = normalizedCorners.map((corner) => corner.y);
        let x = Math.min(...xs);
        let y = Math.min(...ys);
        const maxX = Math.max(...xs);
        const maxY = Math.max(...ys);
        let width = maxX - x;
        let height = maxY - y;

        if (width <= 5 || height <= 5) {
          resetMaskShapeDrawingRefs();
          setMaskRectStart(null);
          setMaskRectCurrent(null);
          return;
        }

        if (maskShape === "star") {
          const size = Math.max(width, height);
          width = size;
          height = size;
        } else if (maskShape === "polygon") {
          const ratio = 1.1543665517482078;
          if (width / (height || 1) >= ratio) {
            height = width / ratio;
          } else {
            width = height * ratio;
          }
        }

        const localFrame =
          targetClip.type === "image"
            ? 0
            : getLocalFrame(focusFrame, targetClip);

        // Create the mask
        const newMask = {
          id: uuidv4(),
          clipId: targetClip.clipId,
          tool: "shape" as const,
          featherAmount: featherAmount,
          keyframes: {
            [localFrame]: {
              shapeBounds: {
                x,
                y,
                width,
                height,
                rotation: 0,
                shapeType: maskShape,
                renderOnce: maskShape === "polygon" || maskShape === "star",
              },
            },
          },
          isTracked: false,
          createdAt: Date.now(),
          lastModified: Date.now(),
          maskColor: "#FFFFFF",
          maskOpacity: 100,
          maskColorEnabled: true,
          backgroundColor: "#000000",
          backgroundOpacity: 100,
          backgroundColorEnabled: true,
          transform: clipTransform ? { ...clipTransform } : undefined,
        };

        const currentMasks = targetClip.masks || [];

        const isRectInsideRect = (
          inner: { x: number; y: number; width: number; height: number },
          outer: { x: number; y: number; width: number; height: number },
        ) => {
          const innerRight = inner.x + inner.width;
          const innerBottom = inner.y + inner.height;
          const outerRight = outer.x + outer.width;
          const outerBottom = outer.y + outer.height;

          return (
            inner.x >= outer.x &&
            inner.y >= outer.y &&
            innerRight <= outerRight &&
            innerBottom <= outerBottom
          );
        };

        const newMaskArea = width * height;
        const newRectBounds = { x, y, width, height };

        const masksToKeep = currentMasks.filter((existingMask) => {
          if (existingMask.tool !== "shape") {
            return true;
          }

          const keyframes =
            existingMask.keyframes instanceof Map
              ? existingMask.keyframes
              : (existingMask.keyframes as Record<number, any>);

          const keyframeNumbers =
            keyframes instanceof Map
              ? Array.from(keyframes.keys()).sort((a, b) => a - b)
              : Object.keys(keyframes)
                  .map(Number)
                  .sort((a, b) => a - b);

          const activeKeyframe = keyframeNumbers
            .filter((k) => k <= localFrame)
            .pop();

          if (activeKeyframe === undefined) {
            return true;
          }

          const maskData =
            keyframes instanceof Map
              ? keyframes.get(activeKeyframe)
              : keyframes[activeKeyframe];

          const existingBounds =
            maskData?.shapeBounds || maskData?.rectangleBounds;
          if (!existingBounds) {
            return true;
          }

          const existingRectBounds = existingBounds;
          const existingArea =
            existingRectBounds.width * existingRectBounds.height;

          const newInsideExisting = isRectInsideRect(
            newRectBounds,
            existingRectBounds,
          );
          const existingInsideNew = isRectInsideRect(
            existingRectBounds,
            newRectBounds,
          );

          if (newInsideExisting || existingInsideNew) {
            return existingArea > newMaskArea;
          }

          return true;
        });

        const shouldAddNewMask = !currentMasks.some((existingMask) => {
          if (existingMask.tool !== "shape") {
            return false;
          }

          const keyframes =
            existingMask.keyframes instanceof Map
              ? existingMask.keyframes
              : (existingMask.keyframes as Record<number, any>);

          const keyframeNumbers =
            keyframes instanceof Map
              ? Array.from(keyframes.keys()).sort((a, b) => a - b)
              : Object.keys(keyframes)
                  .map(Number)
                  .sort((a, b) => a - b);

          const activeKeyframe = keyframeNumbers
            .filter((k) => k <= localFrame)
            .pop();

          if (activeKeyframe === undefined) return false;

          const maskData =
            keyframes instanceof Map
              ? keyframes.get(activeKeyframe)
              : keyframes[activeKeyframe];

          const existingBounds =
            maskData?.shapeBounds || maskData?.rectangleBounds;
          if (!existingBounds) return false;

          const existingRectBounds = existingBounds;
          const existingArea =
            existingRectBounds.width * existingRectBounds.height;

          const newInsideExisting = isRectInsideRect(
            newRectBounds,
            existingRectBounds,
          );

          return newInsideExisting && existingArea > newMaskArea;
        });

        if (shouldAddNewMask) {
          masksToKeep.push(newMask);
        }

        updateClip(targetClip.clipId, {
          masks: masksToKeep,
        });

        if (shouldAddNewMask) {
          setSelectedMaskId(newMask.id);
        }

        // Reset rectangle drawing state
        setMaskRectStart(null);
        setMaskRectCurrent(null);
        resetMaskShapeDrawingRefs();
        return;
      }

      if (tool === "draw" && isDrawingRef.current) {
        const clipId = currentDrawingClipId.current;
        isDrawingRef.current = false;
        currentDrawingClipId.current = null;

        // Apply buffered erasing once on mouse up
        if (activeDrawingToolRef.current === "eraser" && clipId) {
          const worldEraserPts = eraserPointsRef.current.slice();
          eraserPointsRef.current = [];
          lastEraserPosRef.current = null;
          const drawingClip = useClipStore.getState().getClipById(clipId) as
            | DrawingClipProps
            | undefined;
          if (drawingClip && drawingClip.lines) {
            if (tempErased && tempErased.clipId === clipId) {
              const finalLines = mergeConnectedLines(tempErased.lines, true);
              useClipStore.getState().updateClip(clipId, { lines: finalLines });
            } else if (worldEraserPts.length > 0) {
              const eraserBaseRadius = getCurrentSize() / 2;
              const updatedLines: DrawingLine[] = [];
              for (const line of drawingClip.lines) {
                if (line.tool !== "brush" && line.tool !== "highlighter") {
                  updatedLines.push(line);
                  continue;
                }
                const effectiveEraserRadius =
                  eraserBaseRadius + line.strokeWidth / 2;
                const { localPts, localRadius } = transformEraserToLocal(
                  worldEraserPts,
                  line.transform,
                  effectiveEraserRadius,
                );
                const splitPolylines = erasePolylineByEraser(
                  line.points,
                  localPts,
                  localRadius,
                );
                for (const polyline of splitPolylines) {
                  if (polyline.length >= 4) {
                    updatedLines.push({
                      ...line,
                      lineId: uuidv4(),
                      points: polyline,
                    });
                  }
                }
              }
              const fullyMergedLines = mergeConnectedLines(updatedLines, true);
              useClipStore
                .getState()
                .updateClip(clipId, { lines: fullyMergedLines });
            }
          }

          setTempErased(null);
          activeDrawingToolRef.current = null;
          return;
        }

        // Commit buffered brush/highlighter line
        const completedPoints = tempDrawingPoints;
        const style = tempDrawingStyleRef.current;
        setTempDrawingPoints(null);
        tempDrawingStyleRef.current = null;
        setCurrentLineId(null);
        if (!completedPoints || completedPoints.length < 2 || !style) {
          activeDrawingToolRef.current = null;
          return;
        }

        // Try to append to an existing draw clip at this frame
        const existingDrawingClip = clips.find(
          (c) => c.type === "draw" && clipWithinFrame(c, focusFrame),
        ) as DrawingClipProps | undefined;

        const newLine: DrawingLine = {
          lineId: uuidv4(),
          tool: style.tool,
          points: completedPoints,
          stroke: style.stroke,
          strokeWidth: style.strokeWidth,
          opacity: style.opacity,
          smoothing: style.smoothing,
          transform: {
            x: 0,
            y: 0,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
            opacity: 100,
          },
        };

        if (existingDrawingClip) {
          const updatedLines = [...(existingDrawingClip.lines || []), newLine];
          useClipStore
            .getState()
            .updateClip(existingDrawingClip.clipId, { lines: updatedLines });
          activeDrawingToolRef.current = null;
          return;
        }

        // Otherwise create a new drawing clip and timeline if needed
        let drawingTimeline = timelines.find((t) => t.type === "draw");
        if (!drawingTimeline) {
          drawingTimeline = {
            timelineId: uuidv4(),
            type: "draw" as const,
            timelineY: 0,
            timelineHeight: 40,
            timelineWidth: 0,
            timelinePadding: 24,
            muted: false,
            hidden: false,
          };
          addTimeline(drawingTimeline, -1);
        }
        const clipDuration = 3 * DEFAULT_FPS;
        let proposedEndFrame = Math.min(
          focusFrame + clipDuration,
          totalTimelineFrames - 1,
        );
        const clipsOnDrawingTimeline = clips.filter(
          (c) => c.timelineId === drawingTimeline!.timelineId,
        );
        for (const existingClip of clipsOnDrawingTimeline) {
          const existingStart = existingClip.startFrame ?? 0;
          if (existingStart > focusFrame && existingStart < proposedEndFrame) {
            proposedEndFrame = existingStart;
          }
        }
        const newClipId = uuidv4();
        const newClip: DrawingClipProps = {
          clipId: newClipId,
          type: "draw" as const,
          timelineId: drawingTimeline.timelineId,
          startFrame: focusFrame,
          endFrame: Math.max(focusFrame + 1, proposedEndFrame),
          trimEnd: -Infinity,
          trimStart: Infinity,
          lines: [newLine],
        };
        addClip(newClip);
        activeDrawingToolRef.current = null;
        return;
      }

      if (tool === "text" && isDrawingText && textStart && textCurrent) {
        const stage = e.target.getStage();
        if (!stage) {
          setIsDrawingText(false);
          setTextStart(null);
          setTextCurrent(null);
          return;
        }

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) {
          setIsDrawingText(false);
          setTextStart(null);
          setTextCurrent(null);
          return;
        }

        setIsDrawingText(false);
        setTextStart(null);
        setTextCurrent(null);

        const worldEndX = (pointerPos.x - position.x) / scale;
        const worldEndY = (pointerPos.y - position.y) / scale;

        // Calculate shape dimensions using start and current end position
        const x = Math.min(textStart.x, worldEndX);
        const y = Math.min(textStart.y, worldEndY);
        const width = Math.abs(worldEndX - textStart.x);
        const height = Math.abs(worldEndY - textStart.y);

        if (width < 5 || height < 5) {
          setIsDrawingText(false);
          setTextStart(null);
          setTextCurrent(null);
          return;
        }

        let textTimeline = {
          timelineId: uuidv4(),
          type: "text" as const,
          timelineY: 0,
          timelineHeight: 40,
          timelineWidth: 0,
          muted: false,
          hidden: false,
        };
        addTimeline(textTimeline, -1);

        const clipDuration = 3 * DEFAULT_FPS;
        const newClip: TextClipProps = {
          clipId: uuidv4(),
          type: "text" as const,
          timelineId: textTimeline.timelineId,
          startFrame: focusFrame,
          endFrame: Math.min(
            focusFrame + clipDuration,
            totalTimelineFrames - 1,
          ),
          trimEnd: -Infinity,
          trimStart: Infinity,
          text: "Default Text",
          fontSize: 32,
          fontWeight: 400,
          fontStyle: "normal",
          fontFamily: "Arial",
          color: "#ffffff",
          colorOpacity: 100,
          textAlign: "left",
          verticalAlign: "top",
          textTransform: "none",
          textDecoration: "none",
          strokeEnabled: false,
          stroke: "#000000",
          strokeWidth: 2,
          strokeOpacity: 100,
          shadowEnabled: false,
          shadowColor: "#000000",
          shadowOpacity: 75,
          shadowBlur: 4,
          shadowOffsetX: 2,
          shadowOffsetY: 2,
          shadowOffsetLocked: true,
          transform: {
            x: x,
            y: y,
            width: width,
            height: height,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
            cornerRadius: 0,
            opacity: 100,
          },
        };

        addClip(newClip);
      }

      if (tool === "shape" && isDrawingShape && shapeStart && shapeCurrent) {
        // Get current pointer position to ensure accuracy
        const stage = e.target.getStage();
        if (!stage) {
          setIsDrawingShape(false);
          setShapeStart(null);
          setShapeCurrent(null);
          return;
        }

        const pointerPos = stage.getPointerPosition();
        if (!pointerPos) {
          setIsDrawingShape(false);
          setShapeStart(null);
          setShapeCurrent(null);
          return;
        }

        // Convert screen coordinates to world coordinates
        const worldEndX = (pointerPos.x - position.x) / scale;
        const worldEndY = (pointerPos.y - position.y) / scale;

        // Calculate shape dimensions using start and current end position
        const x = Math.min(shapeStart.x, worldEndX);
        const y = Math.min(shapeStart.y, worldEndY);
        let width = Math.abs(worldEndX - shapeStart.x);
        let height = Math.abs(worldEndY - shapeStart.y);
        if (shape === "star") {
          const size = Math.max(width, height);
          width = size;
          height = size;
        } else if (shape === "polygon") {
          const ratio = 1.1543665517482078;
          if (width / (height || 1) >= ratio) {
            height = width / ratio;
          } else {
            width = height * ratio;
          }
        }

        // Only create shape if it has meaningful size
        if (width > 5 && height > 5) {
          // Find or create shape timeline
          let shapeTimeline = {
            timelineId: uuidv4(),
            type: "shape" as const,
            timelineY: 0,
            timelineHeight: 40,
            timelineWidth: 0,
            muted: false,
            hidden: false,
          };

          addTimeline(shapeTimeline, -1);

          // Create shape clip with 3-second duration (72 frames at 24fps)
          const clipDuration = 3 * DEFAULT_FPS;
          const newClip: ShapeClipProps = {
            clipId: uuidv4(),
            type: "shape" as const,
            timelineId: shapeTimeline.timelineId,
            startFrame: focusFrame,
            endFrame: Math.min(
              focusFrame + clipDuration,
              totalTimelineFrames - 1,
            ),
            trimEnd: -Infinity,
            trimStart: Infinity,
            shapeType: shape,
            fill: "#E3E3E3",
            stroke: "#E3E3E3",
            strokeWidth: 1,
            transform: {
              x,
              y,
              width,
              height,
              scaleX: 1,
              scaleY: 1,
              rotation: 0,
              cornerRadius: 0,
              opacity: 100,
            },
          };

          if (shape === "polygon") {
            (newClip as PolygonClipProps).sides = 3;
          }

          addClip(newClip);
        }

        // Reset shape creation state
        setIsDrawingShape(false);
        setShapeStart(null);
        setShapeCurrent(null);
      }
    },
    [
      tool,
      isDrawingShape,
      shapeStart,
      shapeCurrent,
      shape,
      position,
      scale,
      addClip,
      addTimeline,
      isDrawingText,
      textStart,
      textCurrent,
      setCurrentLineId,
      maskTool,
      isDrawingLasso,
      lassoPoints,
      featherAmount,
      focusFrame,
      isMaskDragging,
      getClipForMask,
      isDrawingMaskRect,
      maskRectStart,
      maskRectCurrent,
      isDrawingMaskBrush,
      maskBrushPoints,
      maskBrushSize,
      maskShape,
      touchDrawMode,
      isDrawingTouchLasso,
      touchLassoPoints,
      resetMaskShapeDrawingRefs,
      tempDrawingPoints,
    ],
  );

  const onMouseLeave = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      // set pointer to default
      const container = e.target?.getStage()?.container();
      if (container) {
        container.style.cursor = "default";
      }
      // Hide cursor preview when leaving canvas
      setCursorPreview(null);

      // Clean up any in-progress drawing
      if (isDrawingTouchLasso) {
        setIsDrawingTouchLasso(false);
        setTouchLassoPoints([]);
      }
      if (isDrawingLasso) {
        setIsDrawingLasso(false);
        setLassoPoints([]);
      }
      if (isDrawingMaskBrush) {
        setIsDrawingMaskBrush(false);
        setMaskBrushPoints([]);
      }
    },
    [isDrawingTouchLasso, isDrawingLasso, isDrawingMaskBrush],
  );

  const onMouseEnter = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      const container = e.target?.getStage()?.container();
      if (container) {
        if (tool === "hand") {
          container.style.cursor = "grab";
        } else if (tool === "shape" || tool === "text") {
          container.style.cursor = "crosshair";
        } else if (tool === "draw") {
          // Hide cursor and use preview circle instead for accurate positioning
          container.style.cursor = "none";
        } else if (tool === "mask" && maskTool === "draw") {
          // Hide cursor and use preview circle for mask brush tool
          container.style.cursor = "none";
        } else if (
          tool === "mask" &&
          (maskTool === "lasso" || maskTool === "shape" || maskTool === "touch")
        ) {
          container.style.cursor = "crosshair";
        } else {
          container.style.cursor = "default";
        }
      }
    },
    [tool, drawingTool, maskTool],
  );

  useEffect(() => {
    const container = stageRef.current?.container();
    if (container) {
      if (tool === "hand") {
        container.style.cursor = "grab";
      } else if (tool === "shape" || tool === "text") {
        container.style.cursor = "crosshair";
      } else if (tool === "draw") {
        // Hide cursor and use preview circle instead for accurate positioning
        container.style.cursor = "none";
      } else if (tool === "mask" && maskTool === "draw") {
        // Hide cursor and use preview circle for mask brush tool
        container.style.cursor = "none";
      } else if (
        tool === "mask" &&
        (maskTool === "lasso" || maskTool === "shape" || maskTool === "touch")
      ) {
        // In mask mode, use crosshair when not over a mask (individual mask components set pointer/move cursor)
        if (!isOverMask) {
          container.style.cursor = "crosshair";
        }
      } else {
        container.style.cursor = "default";
      }
    }
  }, [tool, drawingTool, maskTool, isOverMask]);

  // Clear temporary eraser preview when leaving eraser/draw mode and not actively drawing
  useEffect(() => {
    if (!isDrawingRef.current) {
      if (tool !== "draw" || drawingTool !== "eraser") {
        setTempErased(null);
      }
    }
  }, [tool, drawingTool]);

  // Render preview shape while drawing
  const renderDrawingShape = useCallback(() => {
    if (!isDrawingShape || !shapeStart || !shapeCurrent) return null;

    const x = Math.min(shapeStart.x, shapeCurrent.x);
    const y = Math.min(shapeStart.y, shapeCurrent.y);
    let width = Math.abs(shapeCurrent.x - shapeStart.x);
    let height = Math.abs(shapeCurrent.y - shapeStart.y);
    if (shape === "star") {
      const size = Math.max(width, height);
      width = size;
      height = size;
    } else if (shape === "polygon") {
      const ratio = 1.1543665517482078;
      if (width / (height || 1) >= ratio) {
        height = width / ratio;
      } else {
        width = height * ratio;
      }
    }

    const sharedProps = {
      stroke: "#3b82f6",
      strokeOpacity: 100,
      strokeWidth: 2,
      fill: "#3b82f644",
      fillOpacity: 100,
      dash: [5, 5],
    };

    switch (shape) {
      case "rectangle":
        return (
          <Rect {...sharedProps} x={x} y={y} width={width} height={height} />
        );
      case "ellipse":
        return (
          <KonvaEllipse
            {...sharedProps}
            x={x + width / 2}
            y={y + height / 2}
            radiusX={width / 2}
            radiusY={height / 2}
          />
        );
      case "polygon": {
        const radius = Math.min(width / Math.sqrt(3), height / 1.5);
        return (
          <RegularPolygon
            {...sharedProps}
            x={x + width / 2}
            y={y + height / 2}
            sides={3}
            radius={radius}
          />
        );
      }
      case "line":
        return (
          <KonvaLine
            {...sharedProps}
            points={[
              shapeStart.x,
              shapeStart.y,
              shapeCurrent.x,
              shapeCurrent.y,
            ]}
          />
        );
      case "star":
        return (
          <KonvaStar
            {...sharedProps}
            x={x + width / 2}
            y={y + height / 2}
            numPoints={5}
            innerRadius={width / 4}
            outerRadius={width / 2}
          />
        );
      default:
        return (
          <Rect {...sharedProps} x={x} y={y} width={width} height={height} />
        );
    }
  }, [isDrawingShape, shapeStart, shapeCurrent, shape]);

  const renderDrawingText = useCallback(() => {
    if (!isDrawingText || !textStart || !textCurrent) return null;

    const x = Math.min(textStart.x, textCurrent.x);
    const y = Math.min(textStart.y, textCurrent.y);
    const width = Math.abs(textCurrent.x - textStart.x);
    const height = Math.abs(textCurrent.y - textStart.y);

    const sharedProps = {
      stroke: "#3b82f6",
      strokeOpacity: 100,
      strokeWidth: 2,
      fill: "#3b82f644",
      fillOpacity: 100,
      dash: [5, 5],
    };

    return <Rect {...sharedProps} x={x} y={y} width={width} height={height} />;
  }, [isDrawingText, textStart, textCurrent]);

  const renderCursorPreview = useCallback(() => {
    if (!cursorPreview || tool !== "draw") return null;

    const brushSize = getCurrentSize();
    const radius = brushSize / 2;

    // Get the color and opacity based on the tool
    let previewColor = drawingColor;
    let previewOpacity = drawingOpacity / 100;

    if (drawingTool === "highlighter") {
      previewColor = "#FFFF00";
      // Fixed 50% opacity for highlighter
      previewOpacity = 0.5;
    } else if (drawingTool === "eraser") {
      previewColor = "#FFFFFF";
      previewOpacity = 0.5; // Fixed opacity for eraser preview
    }

    return (
      <>
        {/* Outer circle showing brush size */}
        <Circle
          x={cursorPreview.x}
          y={cursorPreview.y}
          radius={radius}
          stroke={previewColor}
          strokeWidth={1}
          fill={previewColor}
          opacity={previewOpacity}
          listening={false}
        />
        {/* Center dot showing exact drawing point */}
        <Circle
          x={cursorPreview.x}
          y={cursorPreview.y}
          radius={1}
          fill={drawingTool === "eraser" ? "#000000" : previewColor}
          opacity={1}
          listening={false}
        />
      </>
    );
  }, [
    cursorPreview,
    tool,
    drawingTool,
    drawingColor,
    drawingOpacity,
    getCurrentSize,
  ]);

  const renderMaskCursorPreview = useCallback(() => {
    // Hide cursor preview while actively drawing to avoid overlap
    if (
      !maskCursorPreview ||
      tool !== "mask" ||
      maskTool !== "draw" ||
      isDrawingMaskBrush
    )
      return null;

    const radius = maskBrushSize / 2;

    return (
      <>
        {/* Outer circle showing brush size */}
        <Circle
          x={maskCursorPreview.x}
          y={maskCursorPreview.y}
          radius={radius}
          stroke="rgb(0, 127, 245)"
          strokeWidth={1}
          fill="rgb(0, 127, 245)"
          opacity={0.4}
          listening={false}
        />
        {/* Center dot showing exact drawing point */}
        <Circle
          x={maskCursorPreview.x}
          y={maskCursorPreview.y}
          radius={1}
          fill="rgb(0, 127, 245)"
          opacity={1}
          listening={false}
        />
      </>
    );
  }, [maskCursorPreview, maskBrushSize, tool, maskTool, isDrawingMaskBrush]);

  const renderMaskBrushDrawing = useCallback(() => {
    // Need at least 2 points (4 values) to render a line
    if (!isDrawingMaskBrush || maskBrushPoints.length < 4) return null;

    return (
      <KonvaLine
        points={maskBrushPoints}
        stroke="rgb(0, 127, 245)"
        strokeWidth={maskBrushSize}
        opacity={0.4}
        lineCap="round"
        lineJoin="round"
        tension={0.5}
        listening={false}
      />
    );
  }, [isDrawingMaskBrush, maskBrushPoints, maskBrushSize]);

  const renderTempDrawingLine = useCallback(() => {
    if (!tempDrawingPoints || tempDrawingPoints.length < 4) return null;
    const style = tempDrawingStyleRef.current;
    if (!style) return null;
    const opacity = (style.opacity ?? 100) / 100;
    const tension =
      typeof style.smoothing === "number"
        ? style.smoothing
        : style.smoothing
          ? 0.5
          : 0;
    return (
      <KonvaLine
        points={tempDrawingPoints}
        stroke={style.stroke}
        strokeWidth={style.strokeWidth}
        opacity={opacity}
        lineCap="round"
        lineJoin="round"
        tension={tension}
        listening={false}
      />
    );
  }, [tempDrawingPoints]);

  const renderLassoPath = useCallback(() => {
    if (lassoPoints.length < 2) return null;

    // While drawing, show a glowing line
    if (isDrawingLasso) {
      return (
        <>
          {/* Outer glow - largest, most transparent */}
          <KonvaLine
            points={lassoPoints}
            stroke="rgba(0, 127, 245, 0.2)"
            strokeWidth={6}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Middle glow */}
          <KonvaLine
            points={lassoPoints}
            stroke="rgba(0, 127, 245, 0.4)"
            strokeWidth={4}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Inner glow */}
          <KonvaLine
            points={lassoPoints}
            stroke="rgba(0, 127, 245, 0.6)"
            strokeWidth={2}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Core line - bright and solid */}
          <KonvaLine
            points={lassoPoints}
            stroke="rgb(0, 127, 245)"
            strokeWidth={2}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
        </>
      );
    }

    return null;
  }, [lassoPoints, isDrawingLasso]);

  const renderTouchLassoPath = useCallback(() => {
    if (touchLassoPoints.length < 2) return null;

    // While drawing, show a glowing line
    if (isDrawingTouchLasso) {
      return (
        <>
          {/* Outer glow - largest, most transparent */}
          <KonvaLine
            points={touchLassoPoints}
            stroke="rgba(0, 127, 245, 0.2)"
            strokeWidth={6}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Middle glow */}
          <KonvaLine
            points={touchLassoPoints}
            stroke="rgba(0, 127, 245, 0.4)"
            strokeWidth={4}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Inner glow */}
          <KonvaLine
            points={touchLassoPoints}
            stroke="rgba(0, 127, 245, 0.6)"
            strokeWidth={2}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Core line - bright and solid */}
          <KonvaLine
            points={touchLassoPoints}
            stroke="rgb(0, 127, 245)"
            strokeWidth={2}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
        </>
      );
    }

    return null;
  }, [touchLassoPoints, isDrawingTouchLasso]);

  const renderMaskRectPreview = useCallback(() => {
    if (!isDrawingMaskRect || !maskRectStart || !maskRectCurrent) return null;

    const sharedProps = {
      stroke: "rgb(0, 127, 245)",
      strokeWidth: 1.5,
      fill: "rgba(0, 127, 245, 0.2)",
      dash: [5, 5],
      listening: false,
    };

    const clipTransform = maskShapeTransformRef.current;
    if (clipTransform) {
      const worldMinX = Math.min(maskRectStart.x, maskRectCurrent.x);
      const worldMaxX = Math.max(maskRectStart.x, maskRectCurrent.x);
      const worldMinY = Math.min(maskRectStart.y, maskRectCurrent.y);
      const worldMaxY = Math.max(maskRectStart.y, maskRectCurrent.y);
      const normalizedCorners = [
        normalizeMaskPointForRotation(worldMinX, worldMinY, clipTransform),
        normalizeMaskPointForRotation(worldMaxX, worldMinY, clipTransform),
        normalizeMaskPointForRotation(worldMinX, worldMaxY, clipTransform),
        normalizeMaskPointForRotation(worldMaxX, worldMaxY, clipTransform),
      ];
      const xs = normalizedCorners.map((corner) => corner.x);
      const ys = normalizedCorners.map((corner) => corner.y);
      let x = Math.min(...xs);
      let y = Math.min(...ys);
      const maxX = Math.max(...xs);
      const maxY = Math.max(...ys);
      let width = maxX - x;
      let height = maxY - y;

      if (width <= 0 || height <= 0) {
        return null;
      }

      if (maskShape === "star") {
        const size = Math.max(width, height);
        width = size;
        height = size;
      } else if (maskShape === "polygon") {
        const ratio = 1.1543665517482078;
        if (Math.abs(height) < 1e-6) {
          height = width / ratio;
        } else if (Math.abs(width) < 1e-6) {
          width = height * ratio;
        } else if (width / height >= ratio) {
          height = width / ratio;
        } else {
          width = height * ratio;
        }
      }

      const centerLocal = { x: x + width / 2, y: y + height / 2 };
      const centerWorld = denormalizeMaskPointForRotation(
        centerLocal.x,
        centerLocal.y,
        clipTransform,
      );
      const rotation = getFiniteNumber(clipTransform.rotation);

      switch (maskShape) {
        case "rectangle":
          return (
            <Rect
              {...sharedProps}
              x={centerWorld.x}
              y={centerWorld.y}
              width={width}
              height={height}
              offsetX={width / 2}
              offsetY={height / 2}
              rotation={rotation}
            />
          );
        case "ellipse":
          return (
            <KonvaEllipse
              {...sharedProps}
              x={centerWorld.x}
              y={centerWorld.y}
              radiusX={width / 2}
              radiusY={height / 2}
              rotation={rotation}
            />
          );
        case "polygon": {
          const radius = Math.min(width / Math.sqrt(3), height / 1.5);
          return (
            <RegularPolygon
              {...sharedProps}
              x={centerWorld.x}
              y={centerWorld.y}
              sides={3}
              radius={radius}
              rotation={rotation}
            />
          );
        }
        case "star":
          return (
            <KonvaStar
              {...sharedProps}
              x={centerWorld.x}
              y={centerWorld.y}
              numPoints={5}
              innerRadius={width / 4}
              outerRadius={width / 2}
              rotation={rotation}
            />
          );
        default:
          return (
            <Rect
              {...sharedProps}
              x={centerWorld.x}
              y={centerWorld.y}
              width={width}
              height={height}
              offsetX={width / 2}
              offsetY={height / 2}
              rotation={rotation}
            />
          );
      }
    }

    const x = Math.min(maskRectStart.x, maskRectCurrent.x);
    const y = Math.min(maskRectStart.y, maskRectCurrent.y);
    let width = Math.abs(maskRectCurrent.x - maskRectStart.x);
    let height = Math.abs(maskRectCurrent.y - maskRectStart.y);
    if (maskShape === "star") {
      const size = Math.max(width, height);
      width = size;
      height = size;
    }

    switch (maskShape) {
      case "rectangle":
        return (
          <Rect {...sharedProps} x={x} y={y} width={width} height={height} />
        );
      case "ellipse":
        return (
          <KonvaEllipse
            {...sharedProps}
            x={x + width / 2}
            y={y + height / 2}
            radiusX={width / 2}
            radiusY={height / 2}
          />
        );
      case "polygon": {
        const radius = Math.min(width / Math.sqrt(3), height / 1.5);
        return (
          <RegularPolygon
            {...sharedProps}
            x={x + width / 2}
            y={y + height / 2}
            sides={3}
            radius={radius}
          />
        );
      }
      case "star":
        return (
          <KonvaStar
            {...sharedProps}
            x={x + width / 2}
            y={y + height / 2}
            numPoints={5}
            innerRadius={width / 4}
            outerRadius={width / 2}
          />
        );
      default:
        return (
          <Rect {...sharedProps} x={x} y={y} width={width} height={height} />
        );
    }
  }, [isDrawingMaskRect, maskRectStart, maskRectCurrent, maskShape]);

  const redrawCount = useRef(0);
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    const handler = () => {
      // runs any time this layer does a scene draw
      redrawCount.current++;
      // dev log, metrics, etc.
    };
    layer.on("draw", handler);
    return () => layer.off("draw", handler);
  }, []);

  
  return (
    <>
      {isFullscreen ? (
        <FullscreenPreview onExit={() => setIsFullscreen(false)} />
      ) : (
        <div className="w-full h-full relative" ref={containerRef}>
          <Stage
            ref={stageRef}
            width={size.width}
            height={size.height}
            className="bg-brand-background"
            onWheel={handleWheel}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseLeave}
            onMouseEnter={onMouseEnter}
          >
            <Layer ref={layerRef}>
              <Group
                x={position.x}
                y={position.y}
                scaleX={scale}
                scaleY={scale}
                width={rectWidth}
                height={rectHeight}
              >
                <Rect
                  ref={aspectRectRef}
                  x={0}
                  y={0}
                  width={rectWidth}
                  height={rectHeight}
                  // Keep the stage background visible so clips with transparency
                  // don't appear to have a "black screen" behind them.
                  fill={"rgb(0,0,0)"}
                  onTransform={() => {
                    const node = aspectRectRef.current;
                    if (!node) return;
                    // bake scaleX into width immediately to avoid jitter
                    const scaleX = node.scaleX() || 1;
                    const newW = Math.max(1, node.width() * scaleX);
                    node.width(newW);
                    node.scaleX(1);
                    const newH = Math.max(1, node.height()); // vertical locked by Transformer
                    // keep rect pinned to x=0 during live transform
                    if (node.x() !== 0) node.x(0);
                    // avoid accidental vertical scale jitter
                    if (node.scaleY() !== 1) node.scaleY(1);
                    // rAF-throttle live updates to avoid setState storms
                    if (liveAspectUpdateRafRef.current) {
                      cancelAnimationFrame(liveAspectUpdateRafRef.current);
                      liveAspectUpdateRafRef.current = null;
                    }
                    const ratio = newW / newH;
                    liveAspectUpdateRafRef.current = requestAnimationFrame(
                      () => {
                        const base = 1000;
                        let w = Math.max(1, Math.round(ratio * base));
                        let h = base;
                        const gcd = (a: number, b: number): number =>
                          b === 0 ? a : gcd(b, a % b);
                        const g = gcd(w, h) || 1;
                        w = Math.round(w / g);
                        h = Math.round(h / g);
                        setAspectRatio({ width: w, height: h, id: "custom" });
                      },
                    );
                  }}
                  onTransformEnd={() => {
                    const node = aspectRectRef.current;
                    if (!node) return;
                    const newW = Math.max(1, node.width() * node.scaleX());
                    const newH = Math.max(1, node.height() * node.scaleY());
                    node.scaleX(1);
                    node.scaleY(1);
                    const ratio = newW / newH;
                    const base = 1000;
                    let w = Math.max(1, Math.round(ratio * base));
                    let h = base;
                    const gcd = (a: number, b: number): number =>
                      b === 0 ? a : gcd(b, a % b);
                    const g = gcd(w, h) || 1;
                    w = Math.round(w / g);
                    h = Math.round(h / g);
                    setAspectRatio({ width: w, height: h, id: "custom" });
                    // Recenter all clips based on the committed aspect ratio
                    const baseShortSide = BASE_LONG_SIDE;
                    const committedRectWidth = baseShortSide * (w / h);
                    const committedRectHeight = baseShortSide;
                    recenterAllClips(committedRectWidth, committedRectHeight);
                  }}
                />
                {sortClips(filterClips(clips)).map((clip: AnyClipProps) => {
                  const startFrame = clip.startFrame || 0;
                  const hasOverlap =
                    (clip.type === "video" || clip.type === "image") &&
                    startFrame > 0
                      ? true
                      : false;

                  const clipAtFrame = clipWithinFrame(
                    clip,
                    focusFrame,
                    hasOverlap,
                    1,
                  );
                  const clipAtFrameNoOverlap = clipWithinFrame(
                    clip,
                    focusFrame,
                  );
                  if (!clipAtFrame) return null;

                  // Get applicators for clips that support effects (video, image, etc.)

                  const applicators = getClipApplicators(clip.clipId);
          
                  if (clipAtFrame) {
                    switch (clip.type) {
                      case "video":
                        return (
                          <VideoPreview
                            key={clip.clipId}
                            {...clip}
                            rectWidth={rectWidth}
                            rectHeight={rectHeight}
                            applicators={applicators}
                            overlap={clipAtFrameNoOverlap}
                          />
                        );
                      case "image":
                        return (
                          <ImagePreview
                            key={clip.clipId}
                            {...clip}
                            rectWidth={rectWidth}
                            rectHeight={rectHeight}
                            applicators={applicators}
                            overlap={clipAtFrameNoOverlap}
                          />
                        );
                      case "model":
                        return (
                          <DynamicModelPreview
                            key={clip.clipId}
                            clip={clip as any}
                            rectWidth={rectWidth}
                            rectHeight={rectHeight}
                            applicators={applicators}
                            overlap={clipAtFrameNoOverlap}
                          />
                        );
                      case "shape":
                        return (
                          <ShapePreview
                            key={clip.clipId}
                            {...clip}
                            rectWidth={rectWidth}
                            rectHeight={rectHeight}
                            applicators={applicators}
                          />
                        );
                      case "text":
                        return (
                          <TextPreview
                            key={clip.clipId}
                            {...clip}
                            rectWidth={rectWidth}
                            rectHeight={rectHeight}
                            applicators={applicators}
                          />
                        );
                      case "draw":
                        return (
                          <DrawingPreview
                            key={clip.clipId}
                            {...clip}
                            rectWidth={rectWidth}
                            rectHeight={rectHeight}
                            tempLinesOverride={
                              tempErased && tempErased.clipId === clip.clipId
                                ? tempErased.lines
                                : undefined
                            }
                            applicators={applicators}
                          />
                        );
                      default:
                        return null;
                    }
                  } else {
                    return null;
                  }
                })}
                {renderDrawingShape()}
                {renderDrawingText()}
                <MaskPreview
                  clips={clips}
                  sortClips={sortClips}
                  filterClips={filterClips}
                  rectWidth={rectWidth}
                  rectHeight={rectHeight}
                />
                {renderLassoPath()}
                {renderTouchLassoPath()}
                {renderMaskRectPreview()}
                {renderMaskBrushDrawing()}
                {renderTempDrawingLine()}
                {renderCursorPreview()}
                {renderMaskCursorPreview()}
                {/* Aspect editing edge bars */}
                {isAspectEditing && (
                  <>
                    <Rect
                      x={0}
                      y={0}
                      width={rectWidth}
                      height={2}
                      fill={"#FFFFFF"}
                      opacity={0.9}
                      listening={false}
                    />
                    <Rect
                      x={0}
                      y={rectHeight - 2}
                      width={rectWidth}
                      height={2}
                      fill={"#FFFFFF"}
                      opacity={0.9}
                      listening={false}
                    />
                    <Rect
                      x={0}
                      y={0}
                      width={2}
                      height={rectHeight}
                      fill={"#FFFFFF"}
                      opacity={0.9}
                      listening={false}
                    />
                    <Rect
                      x={rectWidth - 2}
                      y={0}
                      width={2}
                      height={rectHeight}
                      fill={"#FFFFFF"}
                      opacity={0.9}
                      listening={false}
                    />
                  </>
                )}
              </Group>
            </Layer>
            {/* Transformer overlay for aspect editing */}
            {isAspectEditing && (
              <Layer>
                <Transformer
                  ref={transformerRef}
                  rotateEnabled={false}
                  boundBoxFunc={(oldBox, newBox) => {
                    // lock Y and height; only allow horizontal resizing
                    const minWidth = 50;
                    const tr = transformerRef.current as any;
                    const active = tr?.getActiveAnchor?.();
                    let width = newBox.width;
                    if (active === "middle-left") {
                      // compute width based on left edge movement, keep x pinned to 0
                      const deltaLeft = newBox.x - oldBox.x;
                      width = oldBox.width - deltaLeft;
                    } else {
                      // right edge or unknown: take proposed width
                      width = newBox.width;
                    }
                    width = Math.max(minWidth, width);
                    return {
                      x: 0,
                      y: oldBox.y,
                      width,
                      height: oldBox.height,
                      rotation: oldBox.rotation,
                    };
                  }}
                  enabledAnchors={["middle-right"]}
                  anchorStroke={"#1D4ED8"}
                  anchorStrokeWidth={2}
                  anchorFill={"#3B82F6"}
                  anchorCornerRadius={12}
                  anchorSize={28}
                  borderStroke={"rgba(59,130,246,0.35)"}
                  borderDash={[6, 6]}
                  ignoreStroke={true}
                  anchorStyleFunc={(anchor: any) => {
                    anchor.cornerRadius(10);
                    if (anchor.hasName("middle-right")) {
                      const h = Math.max(80, Math.min(180, rectHeight * 0.4));
                      const w = 14;
                      anchor.width(w);
                      anchor.height(h);
                      anchor.offsetX(w / 2);
                      anchor.offsetY(h / 2);
                      // gradient for visibility
                      if (anchor.fillLinearGradientColorStops) {
                        anchor.fillLinearGradientStartPoint({ x: 0, y: 0 });
                        anchor.fillLinearGradientEndPoint({ x: w, y: 0 });
                        anchor.fillLinearGradientColorStops([
                          0,
                          "#60A5FA",
                          1,
                          "#1D4ED8",
                        ]);
                      }
                      // glow/shadow
                      anchor.shadowColor("#1D4ED8");
                      anchor.shadowBlur(14);
                      anchor.shadowOpacity(0.7);
                      // enlarge hit area without changing visual size
                      anchor.hitStrokeWidth(36);
                    }
                  }}
                />
              </Layer>
            )}
          </Stage>
          {/* Fullscreen button */}
          <button
            onClick={() => setIsFullscreen(true)}
            className="absolute bottom-4 right-4 p-2 bg-brand cursor-pointer hover:bg-brand-background-dark hover:text-blue-400 text-white rounded-md transition-colors"
          >
            <SlSizeFullscreen className="h-3 w-3" />
          </button>
        </div>
      )}
      {
        <>
          {sortClips(filterClips(clips, true)).map((clip: AnyClipProps) => {
            const hasOverlap =
              (clip.type === "video" || clip.type === "model" && clip.assetId) && (clip.startFrame || 0) > 0
                ? true
                : false;
            const clipAtFrame = clipWithinFrame(
              clip,
              focusFrame,
              hasOverlap,
              1,
            );
            if (!clipAtFrame) return null;
            if (clip.type === "audio" || clip.type === "video" || (clip.type === "model" && clip.assetId)) {
              return (
                <AudioPreview key={`audio-${clip.clipId}`} {...(clip as any)} />
              );
            }
            return null;
          })}
        </>
      }
    </>
  );
};

export default Preview;
