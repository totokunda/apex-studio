import React, { useCallback, useEffect, useRef } from "react";
import {
  Group,
  Line as KonvaLine,
  Rect as KonvaRect,
  Transformer,
} from "react-konva";
import { useClipStore, getLocalFrame } from "@/lib/clip";
import { useViewportStore } from "@/lib/viewport";
import { useControlsStore } from "@/lib/control";
import { useMaskStore } from "@/lib/mask";
import { MaskClipProps, MaskData } from "@/lib/types";
import Konva from "konva";
import { KonvaEventObject } from "konva/lib/Node";

interface LassoMaskPreviewProps {
  mask: MaskClipProps;
  points: number[];
  animationOffset: number;
  rectWidth: number;
  rectHeight: number;
}

const LassoMaskPreview: React.FC<LassoMaskPreviewProps> = ({
  mask,
  points,
  animationOffset,
}) => {
  const tool = useViewportStore((s) => s.tool);
  const isFullscreen = useControlsStore((s) => s.isFullscreen);
  const selectedMaskId = useControlsStore((s) => s.selectedMaskId);
  const setSelectedMaskId = useControlsStore((s) => s.setSelectedMaskId);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const selectedClipIds = useControlsStore((s) => s.selectedClipIds);
  const setSelectedClipIds = useControlsStore((s) => s.setSelectedClipIds);
  const maskTool = useMaskStore((s) => s.tool);
  const setIsMaskDragging = useMaskStore((s) => s.setIsMaskDragging);
  const groupRef = useRef<Konva.Group>(null);
  const rectRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const suppressUntilRef = useRef(0);
  const setIsOverMask = useMaskStore((s) => s.setIsOverMask);
  const isSelected = selectedMaskId === mask.id;

  // Allow interaction only when in mask mode with lasso tool
  const canInteract = tool === "mask" && maskTool === "lasso";

  // Calculate bounding box for the lasso
  const bounds = React.useMemo(() => {
    if (points.length < 2) return { x: 0, y: 0, width: 100, height: 100 };

    let minX = points[0];
    let maxX = points[0];
    let minY = points[1];
    let maxY = points[1];

    for (let i = 0; i < points.length; i += 2) {
      minX = Math.min(minX, points[i]);
      maxX = Math.max(maxX, points[i]);
      minY = Math.min(minY, points[i + 1]);
      maxY = Math.max(maxY, points[i + 1]);
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }, [points]);

  const handleClick = useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!canInteract) return;
      e.cancelBubble = true;
      e.evt?.preventDefault?.();
      e.evt?.stopPropagation?.();
      try {
        // Selecting a lasso mask should clear any selected touch points globally
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

  const handleDragStart = useCallback(() => {
    if (!canInteract || !isSelected) return;
    setIsMaskDragging(true);
    try {
      // When starting to drag a mask, also clear touch point selections
      window.dispatchEvent(new CustomEvent("apex-mask-clear-touch-selection"));
    } catch {}
  }, [canInteract, isSelected, setIsMaskDragging]);

  const handleDragEnd = useCallback(() => {
    setIsMaskDragging(false);
    const group = groupRef.current;
    if (!group) return;

    const dx = group.x();
    const dy = group.y();

    // Only update if there was actual movement
    if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) {
      group.position({ x: 0, y: 0 });
      return;
    }

    // Update mask position by shifting all points
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

    const activeKeyframe = keyframeNumbers.filter((k) => k <= focusFrame).pop();
    const referenceKeyframe = activeKeyframe ?? keyframeNumbers[0];

    if (referenceKeyframe !== undefined) {
      const maskData =
        keyframes instanceof Map
          ? keyframes.get(referenceKeyframe)
          : keyframes[referenceKeyframe];

      const basePoints = maskData?.lassoPoints;

      if (basePoints && basePoints.length > 0) {
        const clipStoreState = useClipStore.getState();
        const targetClip = mask.clipId
          ? clipStoreState.getClipById(mask.clipId)
          : undefined;

        if (!targetClip || !mask.clipId) {
          group.position({ x: 0, y: 0 });
          return;
        }

        let targetFrame = focusFrame;
        if (targetClip.type === "video") {
          targetFrame = Math.max(
            0,
            Math.round(getLocalFrame(focusFrame, targetClip)),
          );
        } else if (targetClip.type === "image") {
          targetFrame = 0;
        }

        const frameExists = keyframeNumbers.includes(targetFrame);
        const sourceFrame = frameExists ? targetFrame : referenceKeyframe;
        const sourceData =
          sourceFrame !== undefined
            ? keyframes instanceof Map
              ? keyframes.get(sourceFrame)
              : keyframes[sourceFrame]
            : undefined;

        const pointsToShift =
          sourceData?.lassoPoints && sourceData.lassoPoints.length > 0
            ? sourceData.lassoPoints
            : basePoints;

        if (!pointsToShift || pointsToShift.length === 0) {
          group.position({ x: 0, y: 0 });
          return;
        }

        const shiftedPoints: number[] = [];
        for (let i = 0; i < pointsToShift.length; i += 2) {
          shiftedPoints.push(pointsToShift[i] + dx);
          shiftedPoints.push(pointsToShift[i + 1] + dy);
        }

        const updatedKeyframes =
          keyframes instanceof Map ? new Map(keyframes) : { ...keyframes };

        const baseClone: MaskData = sourceData ? { ...sourceData } : {};
        baseClone.lassoPoints = [...shiftedPoints];

        if (updatedKeyframes instanceof Map) {
          updatedKeyframes.set(targetFrame, baseClone);
        } else {
          (updatedKeyframes as Record<number, any>)[targetFrame] = baseClone;
        }

        const currentMasks = (targetClip as any).masks || [];
        const updatedMasks = currentMasks.map((m: MaskClipProps) =>
          m.id === mask.id ? { ...m, keyframes: updatedKeyframes } : m,
        );

        clipStoreState.updateClip(mask.clipId, { masks: updatedMasks });
      }
    }

    group.position({ x: 0, y: 0 });

    // Add suppression period after drag
    const now =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    suppressUntilRef.current = now + 250;
  }, [mask, focusFrame]);

  React.useEffect(() => {
    if (!isSelected || !canInteract) return;
    const tr = transformerRef.current;
    const rect = rectRef.current;
    if (!tr || !rect) return;

    tr.nodes([rect]);
    tr.getLayer()?.batchDraw();
  }, [isSelected, canInteract]);

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

      const stage = groupRef.current?.getStage();
      if (!stage) return;

      const container = stage.container();
      if (!container.contains(e.target as Node)) return;

      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;

      const rect = rectRef.current;
      if (!rect) return;

      const rectPos = rect.getClientRect();
      const isInside =
        pointerPos.x >= rectPos.x &&
        pointerPos.x <= rectPos.x + rectPos.width &&
        pointerPos.y >= rectPos.y &&
        pointerPos.y <= rectPos.y + rectPos.height;

      if (!isInside) {
        setSelectedMaskId(null);
      }
    };

    window.addEventListener("click", handleClickOutside);
    return () => window.removeEventListener("click", handleClickOutside);
  }, [isSelected, setSelectedMaskId]);

  useEffect(() => {
    const stopDraggingIfNeeded = () => {
      const group = groupRef.current;
      if (!group) return;
      if (typeof group.isDragging === "function" && group.isDragging()) {
        group.stopDrag();
        setIsMaskDragging(false);
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
  }, [setIsMaskDragging]);

  return (
    <Group
      ref={groupRef}
      draggable={canInteract && isSelected}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      clipX={0}
      clipY={0}
      onMouseOver={() => setIsOverMask(true)}
      onMouseOut={() => setIsOverMask(false)}
    >
      {/* Invisible rect for selection and dragging */}
      <KonvaRect
        ref={rectRef}
        x={bounds.x}
        y={bounds.y}
        width={bounds.width}
        height={bounds.height}
        fill="transparent"
        onClick={handleClick}
        onTap={handleClick}
        onMouseDown={(e) => {
          if (!canInteract) return;
          // Only stop propagation if not selected (to prevent creating new mask)
          // If selected, allow event to bubble for dragging
          if (!isSelected) {
            e.cancelBubble = true;
            e.evt?.preventDefault?.();
            e.evt?.stopPropagation?.();
          }
        }}
        onMouseEnter={() => {
          if (!canInteract) return;
          const stage = groupRef.current?.getStage();
          if (stage) {
            const container = stage.container();
            container.style.cursor = isSelected ? "move" : "pointer";
          }
        }}
        onMouseLeave={() => {
          const stage = groupRef.current?.getStage();
          if (stage) {
            const container = stage.container();
            container.style.cursor = "crosshair";
          }
        }}
      />

      {/* White stripe background */}
      <KonvaLine
        points={points}
        stroke="#ffffff"
        strokeWidth={1}
        lineCap="round"
        lineJoin="round"
        closed={true}
        listening={false}
        fill="rgba(0, 127, 245, 0.4)"
      />

      {/* Black stripe foreground with animation */}
      <KonvaLine
        points={points}
        stroke="#000000"
        strokeWidth={1}
        dash={[4.5, 4.5]}
        dashOffset={-animationOffset}
        lineCap="round"
        lineJoin="round"
        closed={true}
        listening={false}
      />

      {/* Transformer for visual feedback (drag only, no anchors) */}
      {canInteract && isSelected && !isFullscreen && (
        <Transformer
          ref={transformerRef}
          borderStroke="#AE81CE"
          borderStrokeWidth={2}
          enabledAnchors={[]}
          rotateEnabled={false}
          resizeEnabled={false}
        />
      )}
    </Group>
  );
};

export default LassoMaskPreview;
