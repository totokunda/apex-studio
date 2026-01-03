import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Rect, Ellipse, Line, Star, Transformer, Group } from "react-konva";
import type Konva from "konva";
import {
  PolygonClipProps,
  ShapeClipProps,
  ShapeTool,
  StarClipProps,
} from "@/lib/types";
import { useClipStore } from "@/lib/clip";
import { useViewportStore } from "@/lib/viewport";
import { useControlsStore } from "@/lib/control";
import RoundedRegularPolygon from "./custom/RoundedRegularPolygon";
import { BaseClipApplicator } from "./apply/base";
import ApplicatorFilter from "./custom/ApplicatorFilter";
import { sanitizeCornerRadius } from "@/lib/konva/sanitizeCornerRadius";

//@ts-ignore
Konva.Filters.Applicator = ApplicatorFilter;

interface ShapePreviewProps extends ShapeClipProps {
  rectWidth: number;
  rectHeight: number;
  applicators: BaseClipApplicator[];
  assetMode?: boolean;
}

const ShapePreview: React.FC<ShapePreviewProps> = ({
  clipId,
  transform,
  rectWidth,
  rectHeight,
  applicators,
}) => {
  // This ref is attached to multiple concrete Konva node types depending on `shapeType`
  // (Rect/Ellipse/Line/Star/RegularPolygon). Use a permissive type to satisfy react-konva.
  const shapeRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  const groupRef = useRef<any>(null);
  const suppressUntilRef = useRef<number>(0);

  const tool = useViewportStore((s) => s.tool);
  const scale = useViewportStore((s) => s.scale);
  const position = useViewportStore((s) => s.position);
  const setClipTransform = useClipStore((s) => s.setClipTransform);
  const clipTransform = useClipStore((s) => s.getClipTransform(clipId));
  const clip = useClipStore((s) => s.getClipById(clipId));
  const focusFrame = useControlsStore((s) => s.focusFrame);

  const isInFrame = useMemo(() => {
    const f = Number(focusFrame);
    if (!Number.isFinite(f)) return true;
    const start = Number((clip as any)?.startFrame ?? 0);
    const endRaw = (clip as any)?.endFrame;
    const end =
      typeof endRaw === "number" && Number.isFinite(endRaw) ? endRaw : Infinity;
    return f >= start && f <= end;
  }, [focusFrame, (clip as any)?.startFrame, (clip as any)?.endFrame]);

  const isSelected = useControlsStore((s) =>
    s.selectedClipIds.includes(clipId),
  );
  const addClipSelection = useControlsStore((s) => s.addClipSelection);
  const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
  const clearSelection = useControlsStore((s) => s.clearSelection);
  const isFullscreen = useControlsStore((s) => s.isFullscreen);
  const getClipById = useClipStore((s) => s.getClipById);

  const applicatorClips = useMemo(() => {
    return applicators.map((a) => getClipById(a.getClip().clipId));
  }, [applicators, getClipById]);

  const applicatorClipsSignature = useMemo(() => {
    // make it a combination of clipId, startFrame, endFrame
    if (!applicatorClips || applicatorClips.length === 0) return "none";
    return applicatorClips
      .map(
        (c) => `${c?.clipId ?? "u"}:${c?.startFrame ?? 0}-${c?.endFrame ?? 0}`,
      )
      .join("|");
  }, [applicatorClips]);

  const SNAP_THRESHOLD_PX = 4;
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

  const currentTransform = clipTransform ||
    transform || {
      x: 0,
      y: 0,
      width: 100,
      height: 100,
      scaleX: 1,
      scaleY: 1,
      rotation: 0,
    };

  const {
    x = 0,
    y = 0,
    width = 100,
    height = 100,
    scaleX = 1,
    scaleY = 1,
    rotation = 0,
  } = currentTransform;

  const {
    shapeType = "rectangle" as ShapeTool,
    fill = "#3b82f6",
    stroke = "#1e40af",
    strokeWidth = 2,
  } = clip as ShapeClipProps;

  // Opacity values used in caching dependencies
  const fillOpacity = (clip as ShapeClipProps)?.fillOpacity ?? 100;
  const strokeOpacity = (clip as ShapeClipProps)?.strokeOpacity ?? 100;

  // Stable filters array so react-konva doesn't set a new filters prop each render
  const filtersArray = useMemo(
    () => [
      //@ts-ignore
      Konva.Filters.Applicator,
    ],
    [],
  );

  const updateGuidesAndMaybeSnap = useCallback(
    (opts: { snap: boolean; commit?: boolean }) => {
      if (isRotating) return;
      const node = shapeRef.current;
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
      setGuides((prev) => {
        if (
          prev.vCenter === nextGuides.vCenter &&
          prev.hCenter === nextGuides.hCenter &&
          prev.v25 === nextGuides.v25 &&
          prev.v75 === nextGuides.v75 &&
          prev.h25 === nextGuides.h25 &&
          prev.h75 === nextGuides.h75 &&
          prev.left === nextGuides.left &&
          prev.right === nextGuides.right &&
          prev.top === nextGuides.top &&
          prev.bottom === nextGuides.bottom
        ) {
          return prev;
        }
        return nextGuides;
      });

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
          // For centered shapes, convert from center position to top-left
          // We must use the transform state dimensions, not node dimensions, because
          // polygon/star shapes don't have width/height, they have radius
          const isCentered =
            shapeType === "ellipse" ||
            shapeType === "polygon" ||
            shapeType === "star";
          const nodeX = node.x();
          const nodeY = node.y();
          const nodeScaleX = node.scaleX();
          const nodeScaleY = node.scaleY();
          // Use the dimensions from clipTransform state
          const currentWidth = clipTransform?.width ?? width;
          const currentHeight = clipTransform?.height ?? height;
          const actualWidth = currentWidth * nodeScaleX;
          const actualHeight = currentHeight * nodeScaleY;
          const saveX = isCentered ? nodeX - actualWidth / 2 : nodeX;
          const saveY = isCentered ? nodeY - actualHeight / 2 : nodeY;
          if (opts.commit) {
            setClipTransform(clipId, { x: saveX, y: saveY });
          }
        }
      }
    },
    [
      rectWidth,
      rectHeight,
      scale,
      setClipTransform,
      clipId,
      isRotating,
      SNAP_THRESHOLD_PX,
      shapeType,
      clipTransform,
      width,
      height,
    ],
  );

  const transformerBoundBoxFunc = useCallback(
    (_oldBox: any, newBox: any) => {
      if (isRotating) return newBox;
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
      if (Math.abs(rectWidth - right) <= thresholdLocal) {
        local.width = rectWidth - local.x;
      } else if (Math.abs(v75 - right) <= thresholdLocal) {
        local.width = v75 - local.x;
      } else if (Math.abs(v25 - right) <= thresholdLocal) {
        local.width = v25 - local.x;
      }
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
      if (Math.abs(rectHeight - bottom) <= thresholdLocal) {
        local.height = rectHeight - local.y;
      } else if (Math.abs(h75 - bottom) <= thresholdLocal) {
        local.height = h75 - local.y;
      } else if (Math.abs(h25 - bottom) <= thresholdLocal) {
        local.height = h25 - local.y;
      }

      let adjusted = {
        ...newBox,
        x: position.x + local.x * scale,
        y: position.y + local.y * scale,
        width: local.width * scale,
        height: local.height * scale,
      };

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
      SNAP_THRESHOLD_PX,
    ],
  );

  useEffect(() => {
    if (!isInFrame) return;
    if (!isSelected) return;
    const tr = transformerRef.current;
    const shape = shapeRef.current;
    if (!tr || !shape) return;
    const raf = requestAnimationFrame(() => {
      tr.nodes([shape]);
      if (typeof (tr as any).forceUpdate === "function") {
        (tr as any).forceUpdate();
      }
      tr.getLayer()?.batchDraw?.();
    });
    return () => cancelAnimationFrame(raf);
  }, [isSelected]);

  // Cache node bitmap when visual attributes change and we are not interacting.
  // This allows Konva to reuse a bitmap for static frames and only recache
  // when something actually changes (e.g., style, geometry, filters).
  useEffect(() => {
    if (!isInFrame) return;
    const node = shapeRef.current;
    if (!node) return;
    if (isInteracting) return;
    // Refresh cache to reflect latest visual attributes
    node.clearCache();
    node.cache({ pixelRatio: 2 });
    node.getLayer()?.batchDraw?.();
  }, [
    // Avoid recache on pure transforms (x, y, rotation, scale) — cached bitmap will be transformed.
    // Recache on changes that affect raster content
    isInteracting,
    shapeType,
    fill,
    stroke,
    strokeWidth,
    fillOpacity,
    strokeOpacity,
    clipTransform?.cornerRadius,
    width,
    height,
    applicatorClipsSignature,
    isInFrame,
  ]);

  const handleClick = useCallback(() => {
    if (isFullscreen) return;
    clearSelection();
    addClipSelection(clipId);
  }, [addClipSelection, clipId, isFullscreen]);

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
      updateGuidesAndMaybeSnap({ snap: true, commit: false });
    },
    [clipId, addClipSelection, updateGuidesAndMaybeSnap],
  );

  const handleDragMove = useCallback(
    (_e: Konva.KonvaEventObject<MouseEvent>) => {
      updateGuidesAndMaybeSnap({ snap: true, commit: false });
    },
    [updateGuidesAndMaybeSnap],
  );

  const handleDragEnd = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      e.target.getStage()!.container().style.cursor = "default";
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
      updateGuidesAndMaybeSnap({ snap: true, commit: false });
      const node = shapeRef.current;
      if (node) {
        const isCentered =
          shapeType === "ellipse" ||
          shapeType === "polygon" ||
          shapeType === "star";
        const nodeX = node.x();
        const nodeY = node.y();
        const nodeScaleX = node.scaleX();
        const nodeScaleY = node.scaleY();
        const currentWidth = clipTransform?.width ?? width;
        const currentHeight = clipTransform?.height ?? height;
        const actualWidth = currentWidth * nodeScaleX;
        const actualHeight = currentHeight * nodeScaleY;
        const saveX = isCentered ? nodeX - actualWidth / 2 : nodeX;
        const saveY = isCentered ? nodeY - actualHeight / 2 : nodeY;
        setClipTransform(clipId, { x: saveX, y: saveY });
      } else {
        setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
      }
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
    [setClipTransform, clipId, shapeType, width, height, clipTransform],
  );

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
      const node = shapeRef.current;
      if (!node) return;
      // For centered shapes, node.x() is the center position, so we need to subtract the offset
      const isCentered =
        shapeType === "ellipse" ||
        shapeType === "polygon" ||
        shapeType === "star";
      const nodeX = node.x();
      const nodeY = node.y();
      const nodeScaleX = node.scaleX();
      const nodeScaleY = node.scaleY();
      // Use the stored width/height from clipTransform for consistency
      // This is critical for polygon/star which don't have explicit width/height
      const currentWidth = clipTransform?.width ?? width;
      const currentHeight = clipTransform?.height ?? height;
      const actualWidth = currentWidth * nodeScaleX;
      const actualHeight = currentHeight * nodeScaleY;
      const saveX = isCentered ? nodeX - actualWidth / 2 : nodeX;
      const saveY = isCentered ? nodeY - actualHeight / 2 : nodeY;

      // For rectangle, ellipse, and line: update width/height, reset scale to 1
      // For polygon and star: keep scaleX/scaleY behavior
      const useWidthHeight =
        shapeType === "rectangle" ||
        shapeType === "ellipse" ||
        shapeType === "line";

      if (useWidthHeight) {
        setClipTransform(clipId, {
          x: saveX,
          y: saveY,
          width: actualWidth,
          height: actualHeight,
          scaleX: 1,
          scaleY: 1,
          rotation: node.rotation(),
        });
        // Only update node dimensions for shapes that have width/height properties
        if (shapeType === "rectangle" || shapeType === "line") {
          node.width(actualWidth);
          node.height(actualHeight);
        } else if (shapeType === "ellipse") {
          //@ts-ignore
          node.radiusX(actualWidth / 2);
          //@ts-ignore
          node.radiusY(actualHeight / 2);
        }
        node.scaleX(1);
        node.scaleY(1);
        node.getLayer()?.batchDraw?.();
      } else {
        setClipTransform(clipId, {
          x: saveX,
          y: saveY,
          width: currentWidth,
          height: currentHeight,
          scaleX: nodeScaleX,
          scaleY: nodeScaleY,
          rotation: node.rotation(),
        });
      }
    };

    const onTransform = () => {
      bumpSuppress();
      if (!isRotating) {
        updateGuidesAndMaybeSnap({ snap: false, commit: false });
      }
      // Avoid committing transform during interaction for performance
    };
    const onTransformEnd = () => {
      bumpSuppress();
      // Persist final geometry first to eliminate end-of-resize snap
      persistTransform();
      // Defer UI state updates to next frame so render uses updated store values
      requestAnimationFrame(() => {
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
      });
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
    shapeType,
    clipTransform,
    width,
    height,
  ]);

  useEffect(() => {
    if (!isInFrame) return;
    const handleWindowClick = (e: MouseEvent) => {
      if (!isSelected) return;
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      if (now < suppressUntilRef.current) return;
      const stage = shapeRef.current?.getStage();
      const container = stage?.container();
      const node = e.target;
      if (!container?.contains(node as Node)) return;
      if (!stage || !container || !shapeRef.current) return;
      const containerRect = container.getBoundingClientRect();
      const pointerX = e.clientX - containerRect.left;
      const pointerY = e.clientY - containerRect.top;
      const shapeRect = shapeRef.current.getClientRect({
        skipShadow: true,
        skipStroke: true,
      });
      const insideShape =
        pointerX >= shapeRect.x &&
        pointerX <= shapeRect.x + shapeRect.width &&
        pointerY >= shapeRect.y &&
        pointerY <= shapeRect.y + shapeRect.height;

      if (!insideShape) {
        removeClipSelection(clipId);
      }
    };
    window.addEventListener("click", handleWindowClick);
    return () => {
      window.removeEventListener("click", handleWindowClick);
    };
  }, [clipId, isSelected, removeClipSelection, isInFrame]);

  const hexToRgba = (hex: string, opacity: number) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${opacity / 100})`;
  };

  const renderShape = () => {
    const fillWithOpacity = hexToRgba(
      fill,
      (clip as ShapeClipProps)?.fillOpacity ?? 100,
    );
    const strokeWithOpacity = hexToRgba(
      stroke,
      (clip as ShapeClipProps)?.strokeOpacity ?? 100,
    );

    const sides = (clip as PolygonClipProps)?.sides ?? 3;

    const safeCornerRadius = sanitizeCornerRadius(
      clipTransform?.cornerRadius,
      width,
      height,
    );

    const baseProps = {
      ref: shapeRef,
      scaleX,
      scaleY,
      rotation,
      fill: fillWithOpacity,
      stroke: strokeWithOpacity,
      strokeWidth,
      draggable: tool === "pointer" && !isTransforming,
      opacity: (clipTransform?.opacity ?? 100) / 100,
      onDragStart: handleDragStart,
      onDragMove: handleDragMove,
      onDragEnd: handleDragEnd,
      onClick: handleClick,
      perfectDrawEnabled: false,
      shadowForStrokeEnabled: false,
    };

    // Props for shapes positioned from top-left corner
    const cornerProps = {
      ...baseProps,
      x,
      y,
    };

    // Props for shapes positioned from center (account for scale in offset)
    const actualWidth = width * scaleX;
    const actualHeight = height * scaleY;
    const centerProps = {
      ...baseProps,
      x: x + actualWidth / 2,
      y: y + actualHeight / 2,
    };

    // Use stable references to avoid invalidating cached filter output

    switch (shapeType) {
      case "rectangle":
        //@ts-ignore
        return (
          <Rect
            {...cornerProps}
            width={width}
            height={height}
            cornerRadius={safeCornerRadius}
            applicators={applicators}
            filters={filtersArray}
          />
        );

      case "ellipse":
        //@ts-ignore
        return (
          <Ellipse
            {...centerProps}
            radiusX={width / 2}
            radiusY={height / 2}
            applicators={applicators}
            filters={filtersArray}
          />
        );

      case "polygon": {
        // Match dashed preview geometry: fit equilateral triangle inside width/height
        // For RegularPolygon/triangle: width = R*sqrt(3), height = 1.5*R
        const radius = Math.min(width / Math.sqrt(3), height / 1.5);
        return (
          <RoundedRegularPolygon
            {...centerProps}
            sides={sides}
            radius={radius}
            cornerRadius={safeCornerRadius as number}
            applicators={applicators}
            filters={filtersArray}
          />
        );
      }

      case "line":
        //@ts-ignore
        return (
          <Line
            {...cornerProps}
            points={[0, 0, width, 0]}
            applicators={applicators}
            filters={filtersArray}
          />
        );

      case "star":
        //@ts-ignore
        return (
          <Star
            {...centerProps}
            numPoints={(clip as StarClipProps)?.points ?? 5}
            innerRadius={Math.min(width, height) / 4}
            outerRadius={Math.min(width, height) / 2}
            applicators={applicators}
            filters={filtersArray}
          />
        );

      default:
        //@ts-ignore
        return (
          <Rect
            {...cornerProps}
            width={width}
            height={height}
            cornerRadius={safeCornerRadius}
            applicators={applicators}
            filters={filtersArray}
          />
        );
    }
  };

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
        {renderShape()}
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
        visible={tool === "pointer" && isSelected && !isFullscreen}
        borderStrokeWidth={2}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]}
        boundBoxFunc={transformerBoundBoxFunc as any}
        ref={(node) => {
          transformerRef.current = node;
          if (node && shapeRef.current) {
            node.nodes([shapeRef.current]);
            if (typeof (node as any).forceUpdate === "function") {
              (node as any).forceUpdate();
            }
            node.getLayer()?.batchDraw?.();
          }
        }}
        enabledAnchors={[
          "top-left",
          "top-center",
          "top-right",
          "bottom-left",
          "bottom-right",
          "middle-left",
          "middle-right",
          "bottom-center",
        ]}
      />
    </React.Fragment>
  );
};

export default ShapePreview;
