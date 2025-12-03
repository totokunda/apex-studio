import React, { useCallback, useRef, useEffect } from "react";
import {
  Group,
  Rect as KonvaRect,
  Ellipse as KonvaEllipse,
  Star as KonvaStar,
  Transformer,
  RegularPolygon,
} from "react-konva";
import { useClipStore } from "@/lib/clip";
import { useViewportStore } from "@/lib/viewport";
import { useControlsStore } from "@/lib/control";
import { useMaskStore } from "@/lib/mask";
import { upsertMaskKeyframe } from "@/lib/mask/keyframeUtils";
import { MaskClipProps, MaskShapeTool } from "@/lib/types";
import Konva from "konva";
import { KonvaEventObject } from "konva/lib/Node";

interface ShapeMaskPreviewProps {
  mask: MaskClipProps;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  animationOffset: number;
  rectWidth: number;
  rectHeight: number;
  shapeType: MaskShapeTool;
  scaleX?: number;
  scaleY?: number;
}

const ShapeMaskPreview: React.FC<ShapeMaskPreviewProps> = ({
  mask,
  x,
  y,
  width,
  height,
  rotation,
  animationOffset,
  rectWidth,
  rectHeight,
  shapeType,
}) => {
  const tool = useViewportStore((s) => s.tool);
  const isFullscreen = useControlsStore((s) => s.isFullscreen);
  const selectedMaskId = useControlsStore((s) => s.selectedMaskId);
  const setSelectedMaskId = useControlsStore((s) => s.setSelectedMaskId);
  const focusFrame = useControlsStore((s) => s.focusFrame);

  const selectedClipIds = useControlsStore((s) => s.selectedClipIds);
  const setSelectedClipIds = useControlsStore((s) => s.setSelectedClipIds);
  const getClipTransform = useClipStore((s) => s.getClipTransform);

  const shapeWhiteRef = useRef<any>(null);
  const shapeBlackRef = useRef<any>(null);
  const groupRef = useRef<Konva.Group>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const suppressUntilRef = useRef(0);

  const isSelected = selectedMaskId === mask.id;

  // Allow interaction only when in mask mode with matching shape tool
  const canInteract = tool === "mask";
  const setIsOverMask = useMaskStore((s) => s.setIsOverMask);

  const handleClick = useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!canInteract) return;
      e.cancelBubble = true;
      e.evt?.preventDefault?.();
      e.evt?.stopPropagation?.();
      try {
        // Selecting a mask should clear any selected touch points globally
        window.dispatchEvent(
          new CustomEvent("apex-mask-clear-touch-selection"),
        );
      } catch {}
      setSelectedMaskId(mask.id);

      // Also select the corresponding clip
      if (mask.clipId && !selectedClipIds.includes(mask.clipId)) {
        setSelectedClipIds([mask.clipId]);
      }
    },
    [
      canInteract,
      mask.id,
      mask.clipId,
      selectedClipIds,
      setSelectedMaskId,
      setSelectedClipIds,
    ],
  );

  const updateMaskBounds = useCallback(
    (isResize: boolean = false) => {
      const shape = shapeWhiteRef.current;
      if (!shape) return;

      const keyframes =
        mask.keyframes instanceof Map
          ? mask.keyframes
          : (mask.keyframes as Record<number, any>);

      const keyframeNumbers =
        keyframes instanceof Map
          ? Array.from(keyframes.keys())
              .map(Number)
              .sort((a, b) => a - b)
          : Object.keys(keyframes)
              .map(Number)
              .sort((a, b) => a - b);

      if (keyframeNumbers.length === 0) {
        return;
      }

      const activeKeyframe = keyframeNumbers
        .filter((k) => k <= focusFrame)
        .pop();
      const referenceKeyframe = activeKeyframe ?? keyframeNumbers[0];
      if (referenceKeyframe === undefined) {
        return;
      }

      const maskData =
        keyframes instanceof Map
          ? keyframes.get(referenceKeyframe)
          : keyframes[referenceKeyframe];

      const existingBounds = maskData?.shapeBounds || maskData?.rectangleBounds;
      if (!existingBounds) {
        return;
      }

      // For centered shapes (ellipse, polygon, star), convert from center position to top-left
      const isCentered =
        shapeType === "ellipse" ||
        shapeType === "polygon" ||
        shapeType === "star";
      const shapeX = shape.x();
      const shapeY = shape.y();
      const shapeScaleX = shape.scaleX();
      const shapeScaleY = shape.scaleY();

      let actualWidth: number;
      let actualHeight: number;

      if (shapeType === "ellipse") {
        actualWidth = shape.radiusX() * 2 * shapeScaleX;
        actualHeight = shape.radiusY() * 2 * shapeScaleY;
      } else if (shapeType === "rectangle") {
        actualWidth = shape.width() * shapeScaleX;
        actualHeight = shape.height() * shapeScaleY;
      } else if (shapeType === "polygon" || shapeType === "star") {
        // Compute actual size from effective local radii (rotation-independent)
        if (
          shapeType === "polygon" &&
          typeof (shape as any).radius === "function"
        ) {
          const effectiveR = (shape as any).radius() * (shape.scaleX() || 1);
          actualWidth = Math.sqrt(3) * effectiveR; // width = √3 * R
          actualHeight = 1.5 * effectiveR; // height = 1.5 * R
        } else if (
          shapeType === "star" &&
          typeof (shape as any).outerRadius === "function" &&
          typeof (shape as any).innerRadius === "function"
        ) {
          const effectiveOuter =
            (shape as any).outerRadius() * (shape.scaleX() || 1);
          actualWidth = effectiveOuter * 2;
          actualHeight = effectiveOuter * 2;
        } else {
          // Fallback to clientRect if radius APIs not available
          const rect = shape.getClientRect();
          actualWidth = rect.width;
          actualHeight = rect.height;
        }
      } else {
        const rect = shape.getClientRect();
        actualWidth = rect.width;
        actualHeight = rect.height;
      }

      const prevBounds = existingBounds as
        | { width?: number; height?: number }
        | undefined;
      const widthForPosition = isResize
        ? actualWidth
        : (prevBounds?.width ?? actualWidth);
      const heightForPosition = isResize
        ? actualHeight
        : (prevBounds?.height ?? actualHeight);
      const saveX = isCentered ? shapeX - widthForPosition / 2 : shapeX;
      const saveY = isCentered ? shapeY - heightForPosition / 2 : shapeY;

      const newBounds = {
        x: saveX,
        y: saveY,
        width: isResize ? actualWidth : (prevBounds?.width ?? actualWidth),
        height: isResize ? actualHeight : (prevBounds?.height ?? actualHeight),
        rotation: shape.rotation(),
        shapeType,
      };

      const clipStoreState = useClipStore.getState();
      const targetClip = clipStoreState.clips.find(
        (c) => c.clipId === mask.clipId,
      );

      if (targetClip && mask.clipId) {
        const result = upsertMaskKeyframe({
          mask,
          clip: targetClip,
          focusFrame,
          updater: (previous) => ({
            ...previous,
            shapeBounds: newBounds,
          }),
        });

        if (result) {
          const currentMasks = (targetClip as any).masks || [];
          const updatedMasks = currentMasks.map((m: MaskClipProps) =>
            m.id === mask.id ? { ...m, keyframes: result.keyframes } : m,
          );

          clipStoreState.updateClip(mask.clipId, { masks: updatedMasks });
        }
      }

      // Reset scale to 1 for all shapes; we only persist width/height
      if (shapeType === "rectangle") {
        shape.scaleX(1);
        shape.scaleY(1);
      } else if (shapeType === "ellipse") {
        shape.scaleX(1);
        shape.scaleY(1);
      } else if (shapeType === "polygon" || shapeType === "star") {
        shape.scaleX(1);
        shape.scaleY(1);
      }

      // Sync black shape with white shape
      syncBlackShape();
    },
    [mask, focusFrame, width, height, shapeType],
  );

  const syncBlackShape = useCallback(() => {
    const whiteShape = shapeWhiteRef.current;
    const blackShape = shapeBlackRef.current;

    if (whiteShape && blackShape) {
      blackShape.x(whiteShape.x());
      blackShape.y(whiteShape.y());
      blackShape.rotation(whiteShape.rotation());

      if (shapeType === "rectangle") {
        blackShape.width(whiteShape.width() * whiteShape.scaleX());
        blackShape.height(whiteShape.height() * whiteShape.scaleY());
        blackShape.scaleX(1);
        blackShape.scaleY(1);
      } else if (shapeType === "ellipse") {
        blackShape.radiusX(whiteShape.radiusX() * whiteShape.scaleX());
        blackShape.radiusY(whiteShape.radiusY() * whiteShape.scaleY());
        blackShape.scaleX(1);
        blackShape.scaleY(1);
      } else if (shapeType === "polygon") {
        // Mirror white radii and scale for live alignment during resize
        if (
          typeof whiteShape.radius === "function" &&
          typeof blackShape.radius === "function"
        ) {
          blackShape.radius(whiteShape.radius());
        }
        blackShape.scaleX(whiteShape.scaleX());
        blackShape.scaleY(whiteShape.scaleY());
      } else if (shapeType === "star") {
        // Mirror white radii and scale for live alignment during resize
        if (
          typeof whiteShape.outerRadius === "function" &&
          typeof blackShape.outerRadius === "function"
        ) {
          blackShape.outerRadius(whiteShape.outerRadius());
        }
        if (
          typeof whiteShape.innerRadius === "function" &&
          typeof blackShape.innerRadius === "function"
        ) {
          blackShape.innerRadius(whiteShape.innerRadius());
        }
        blackShape.scaleX(whiteShape.scaleX());
        blackShape.scaleY(whiteShape.scaleY());
      }
    }
  }, [shapeType]);

  const handleTransform = useCallback(() => {
    const whiteShape = shapeWhiteRef.current;
    const blackShape = shapeBlackRef.current;
    if (
      whiteShape &&
      blackShape &&
      (shapeType === "polygon" || shapeType === "star")
    ) {
      // Recompute radii only during resize, not rotation
      const activeAnchor = transformerRef.current?.getActiveAnchor?.();
      const isRotate = activeAnchor === "rotater";
      if (!isRotate) {
        const rect = whiteShape.getClientRect();
        let stageW = rect.width;
        let stageH = rect.height;
        // Enforce polygon ratio in stage space first
        if (shapeType === "polygon") {
          const ratio = 1.1543665517482078;
          if (stageW / (stageH || 1) >= ratio) {
            stageH = stageW / ratio;
          } else {
            stageW = stageH * ratio;
          }
        } else {
          // star square
          const size = Math.max(stageW, stageH);
          stageW = size;
          stageH = size;
        }
        const absScale = whiteShape.getAbsoluteScale();
        const localScaleX = whiteShape.scaleX() || 1;
        const localScaleY = whiteShape.scaleY() || 1;
        const parentScaleX = absScale.x !== 0 ? absScale.x / localScaleX : 1;
        const parentScaleY = absScale.y !== 0 ? absScale.y / localScaleY : 1;
        const parentScale =
          (Math.abs(parentScaleX) + Math.abs(parentScaleY)) / 2;
        const localW = stageW / (parentScale || 1);
        const localH = stageH / (parentScale || 1);
        if (shapeType === "polygon") {
          const radius = Math.min(localW / Math.sqrt(3), localH / 1.5);
          blackShape.radius(radius);
        } else {
          blackShape.outerRadius(Math.min(localW, localH) / 2);
          blackShape.innerRadius(Math.min(localW, localH) / 4);
        }
        blackShape.scaleX(1);
        blackShape.scaleY(1);
      } else {
        // Rotation: keep radii equal and just sync transform
        if (
          shapeType === "polygon" &&
          typeof whiteShape.radius === "function" &&
          typeof blackShape.radius === "function"
        ) {
          blackShape.radius(whiteShape.radius());
        } else if (shapeType === "star") {
          if (
            typeof whiteShape.outerRadius === "function" &&
            typeof blackShape.outerRadius === "function"
          ) {
            blackShape.outerRadius(whiteShape.outerRadius());
          }
          if (
            typeof whiteShape.innerRadius === "function" &&
            typeof blackShape.innerRadius === "function"
          ) {
            blackShape.innerRadius(whiteShape.innerRadius());
          }
        }
        blackShape.scaleX(1);
        blackShape.scaleY(1);
      }
    }
    syncBlackShape();
  }, [syncBlackShape, shapeType]);

  const handleDragMove = useCallback(() => {
    syncBlackShape();
  }, [syncBlackShape]);

  const handleTransformEnd = useCallback(() => {
    updateMaskBounds(true);

    // Add suppression period after transform
    const now =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    suppressUntilRef.current = now + 250;
  }, [updateMaskBounds]);

  const handleDragEnd = useCallback(() => {
    updateMaskBounds();

    // Add suppression period after drag
    const now =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    suppressUntilRef.current = now + 250;
  }, [updateMaskBounds]);

  React.useEffect(() => {
    if (!isSelected || !canInteract) return;
    const tr = transformerRef.current;
    const whiteShape = shapeWhiteRef.current;
    if (!tr || !whiteShape) return;

    // Sync black shape with white shape initially
    syncBlackShape();

    tr.nodes([whiteShape]);
    tr.getLayer()?.batchDraw();
  }, [isSelected, canInteract, syncBlackShape]);

  // Deselect when clicking outside
  React.useEffect(() => {
    if (!isSelected) return;

    const handleClickOutside = (e: MouseEvent) => {
      // Check suppression period
      const now =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      if (now < suppressUntilRef.current) return;

      const shape = shapeWhiteRef.current;
      if (!shape) return;

      const stage = shape.getStage();
      if (!stage) return;

      const container = stage.container();
      if (!container.contains(e.target as Node)) return;

      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;

      const shapePos = shape.getClientRect();
      const isInside =
        pointerPos.x >= shapePos.x &&
        pointerPos.x <= shapePos.x + shapePos.width &&
        pointerPos.y >= shapePos.y &&
        pointerPos.y <= shapePos.y + shapePos.height;

      if (!isInside) {
        setSelectedMaskId(null);
      }
    };

    window.addEventListener("click", handleClickOutside);
    return () => window.removeEventListener("click", handleClickOutside);
  }, [isSelected, setSelectedMaskId]);

  useEffect(() => {
    const stopDraggingIfNeeded = () => {
      const shape = shapeWhiteRef.current;
      if (!shape) return;
      if (typeof shape.isDragging === "function" && shape.isDragging()) {
        shape.stopDrag();
      }
    };

    window.addEventListener("mouseup", stopDraggingIfNeeded);
    window.addEventListener("pointerup", stopDraggingIfNeeded);
    window.addEventListener("touchend", stopDraggingIfNeeded);
    window.addEventListener("touchcancel", stopDraggingIfNeeded);

    return () => {
      window.removeEventListener("mouseup", stopDraggingIfNeeded);
      window.removeEventListener("pointerup", stopDraggingIfNeeded);
      window.removeEventListener("touchend", stopDraggingIfNeeded);
      window.removeEventListener("touchcancel", stopDraggingIfNeeded);
    };
  }, []);

  const renderShape = useCallback(
    (isWhite: boolean) => {
      const baseProps = {
        ref: isWhite ? shapeWhiteRef : shapeBlackRef,
        stroke: isWhite ? "#ffffff" : "#000000",
        strokeWidth: 1,
        fill: isWhite ? "rgba(0, 127, 245, 0.4)" : undefined,
        draggable: canInteract && isSelected && isWhite,
        onClick: isWhite ? handleClick : undefined,
        onTap: isWhite ? handleClick : undefined,
        onMouseDown: isWhite
          ? (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
              if (!canInteract) return;
              // Only stop propagation if not selected (to prevent creating new mask)
              // If selected, allow event to bubble for dragging/transforming
              if (!isSelected) {
                e.cancelBubble = true;
                e.evt?.preventDefault?.();
                e.evt?.stopPropagation?.();
              }
            }
          : undefined,
        onMouseEnter: isWhite
          ? () => {
              if (!canInteract) return;
              const stage = groupRef.current?.getStage();
              if (stage) {
                const container = stage.container();
                container.style.cursor = isSelected ? "move" : "pointer";
              }
            }
          : undefined,
        onMouseLeave: isWhite
          ? () => {
              const stage = groupRef.current?.getStage();
              if (stage) {
                const container = stage.container();
                container.style.cursor = "crosshair";
              }
            }
          : undefined,
        onTransform: isWhite ? handleTransform : undefined,
        onTransformEnd: isWhite ? handleTransformEnd : undefined,
        onDragMove: isWhite ? handleDragMove : undefined,
        onDragEnd: isWhite ? handleDragEnd : undefined,
        listening: isWhite,
        rotation: rotation,
        dash: !isWhite ? [4.5, 4.5] : undefined,
        dashOffset: !isWhite ? -animationOffset : undefined,
      };

      // Props for shapes positioned from top-left corner
      const cornerProps = {
        ...baseProps,
        x,
        y,
      };

      // Props for shapes positioned from center
      // No render-time ratio adjustment; bounds already enforce it during updates
      const actualWidth = width;
      const actualHeight = height;

      const centerProps = {
        ...baseProps,
        x: x + actualWidth / 2,
        y: y + actualHeight / 2,
        scaleX: 1,
        scaleY: 1,
      };

      switch (shapeType) {
        case "rectangle":
          return <KonvaRect {...cornerProps} width={width} height={height} />;

        case "ellipse":
          return (
            <KonvaEllipse
              {...centerProps}
              radiusX={width / 2}
              radiusY={height / 2}
            />
          );

        case "polygon": {
          const radius = Math.min(
            actualWidth / Math.sqrt(3),
            actualHeight / 1.5,
          );
          return <RegularPolygon {...centerProps} sides={3} radius={radius} />;
        }

        case "star":
          return (
            <KonvaStar
              {...centerProps}
              numPoints={5}
              innerRadius={Math.min(width, height) / 4}
              outerRadius={Math.min(width, height) / 2}
            />
          );

        default:
          return <KonvaRect {...cornerProps} width={width} height={height} />;
      }
    },
    [
      x,
      y,
      width,
      height,
      rotation,
      animationOffset,
      shapeType,
      canInteract,
      isSelected,
      handleClick,
      handleTransform,
      handleTransformEnd,
      handleDragMove,
      handleDragEnd,
      groupRef,
    ],
  );

  return (
    <Group
      visible={tool === "mask"}
      ref={groupRef}
      clipX={0}
      clipY={0}
      onMouseOver={() => setIsOverMask(true)}
      onMouseOut={() => setIsOverMask(false)}
    >
      {/* White shape background */}
      {renderShape(true)}

      {/* Black shape foreground with animation */}
      {renderShape(false)}

      <Transformer
        visible={canInteract && isSelected && !isFullscreen}
        ref={transformerRef}
        borderStroke="#AE81CE"
        anchorCornerRadius={8}
        anchorStroke="#E3E3E3"
        anchorStrokeWidth={1}
        borderStrokeWidth={2}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]}
        enabledAnchors={
          shapeType === "star"
            ? ["top-left", "top-right", "bottom-left", "bottom-right"]
            : shapeType === "polygon"
              ? ["top-left", "top-right", "bottom-left", "bottom-right"]
              : [
                  "top-left",
                  "top-center",
                  "top-right",
                  "bottom-left",
                  "bottom-right",
                  "middle-left",
                  "middle-right",
                  "bottom-center",
                ]
        }
      />
    </Group>
  );
};

export default ShapeMaskPreview;
