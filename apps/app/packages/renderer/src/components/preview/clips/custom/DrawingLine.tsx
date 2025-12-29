import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Line as KonvaLine, Rect, Line, Group } from "react-konva";
import { DrawingLine } from "@/lib/types";
import Konva from "konva";
import { Transformer } from "react-konva";
import { useViewportStore } from "@/lib/viewport";
import { useDrawingStore } from "@/lib/drawing";
import { useControlsStore } from "@/lib/control";
import { useClipStore } from "@/lib/clip";
import ApplicatorFilter from "../custom/ApplicatorFilter";
import { BaseClipApplicator } from "../apply/base";

//@ts-ignore
Konva.Filters.Applicator = ApplicatorFilter;

const SNAP_THRESHOLD_PX = 5;

interface DrawingLineProps {
  applicators: BaseClipApplicator[];
  line: DrawingLine;
  lineOpacity: number;
  isLineSelected: boolean;
  handleLineClick: (e: any, lineId: string) => void;
  handleDragEnd: (lineId: string) => void;
  handleTransformEnd: (lineId: string) => void;
  selectedLineId: string;
  setLineRef: (lineId: string, ref: Konva.Line | null) => void;
  rectWidth: number;
  rectHeight: number;
}

interface Guides {
  vCenter: boolean;
  hCenter: boolean;
  v25: boolean;
  v75: boolean;
  h25: boolean;
  h75: boolean;
  left: boolean;
  right: boolean;
  top: boolean;
  bottom: boolean;
}

const getToolConfig = (tool: string, smoothing: number = 0.5) => {
  switch (tool) {
    case "brush":
      return {
        lineCap: "round" as const,
        lineJoin: "round" as const,
        tension: smoothing,
        globalCompositeOperation: "source-over" as const,
      };
    case "highlighter":
      return {
        lineCap: "square" as const,
        lineJoin: "bevel" as const,
        tension: 0,
        globalCompositeOperation: "multiply" as const,
      };
    case "eraser":
      return {
        lineCap: "round" as const,
        lineJoin: "round" as const,
        tension: 0.5,
        globalCompositeOperation: "destination-out" as const,
      };
    default:
      return {
        lineCap: "round" as const,
        lineJoin: "round" as const,
        tension: smoothing,
        globalCompositeOperation: "source-over" as const,
      };
  }
};

const DrawingLineComponent: React.FC<DrawingLineProps> = ({
  applicators,
  line,
  handleLineClick,
  handleDragEnd,
  handleTransformEnd,
  isLineSelected,
  selectedLineId,
  setLineRef,
  rectWidth,
  rectHeight,
}) => {
  const toolConfig = getToolConfig(line.tool, line.smoothing);
  const lineOpacity = line.tool === "eraser" ? 1 : line.opacity / 100;
  const ref = useRef<Konva.Line>(null);
  const rectRef = useRef<Konva.Rect>(null);
  const groupRef = useRef<Konva.Group>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const isTransformingRef = useRef(false);
  const tool = useViewportStore((s) => s.tool);
  const [guides, setGuides] = useState<Guides>({
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

  useEffect(() => {
    if (ref.current) {
      setLineRef(line.lineId, ref.current);
    }
  }, [ref.current, line.lineId, setLineRef]);

  useEffect(() => {
    if (rectRef.current) {
      rectRef.current.moveToTop();
      rectRef.current.getLayer()?.batchDraw();
    }
  }, [tool]);

  const isDrawing = useDrawingStore((s) => s.isDrawing);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const clipsState = useClipStore((s) => s.clips);

  // Stable filters array
  const filtersArray = useMemo(
    () => [
      //@ts-ignore
      Konva.Filters.Applicator,
    ],
    [],
  );

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

  // Store-driven active flag for current focusFrame
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

  // Cache line bitmap only when appearance or active-range changes and not during drawing/transforming
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (isDrawing || isTransformingRef.current) return;
    node.clearCache();
    node.cache({ pixelRatio: 2 });
    node.getLayer()?.batchDraw?.();
  }, [
    isDrawing,
    applicatorsSignature,
    applicatorsActiveStore,
    line.stroke,
    line.strokeWidth,
    line.opacity,
    line.tool,
    line.smoothing,
    // Points define geometry of the stroke; stringify length and a hash-y join to avoid huge deps
    line.points.length,
  ]);

  useEffect(() => {
    if (
      transformerRef.current &&
      rectRef.current &&
      isLineSelected &&
      tool === "pointer"
    ) {
      transformerRef.current.nodes([rectRef.current]);
      transformerRef.current.moveToTop();
      transformerRef.current.getLayer()?.batchDraw();
    }
  }, [isLineSelected, tool]);

  const getBoundingBox = useCallback(() => {
    if (!line.points.length)
      return { x: 0, y: 0, width: 0, height: 0, minX: 0, minY: 0 };

    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (let i = 0; i < line.points.length; i += 2) {
      minX = Math.min(minX, line.points[i]);
      maxX = Math.max(maxX, line.points[i]);
      minY = Math.min(minY, line.points[i + 1]);
      maxY = Math.max(maxY, line.points[i + 1]);
    }

    // Account for stroke width by expanding the bounding box
    const halfStroke = line.strokeWidth / 2;
    minX -= halfStroke;
    minY -= halfStroke;
    maxX += halfStroke;
    maxY += halfStroke;

    // Calculate rect position accounting for rotation and scale
    const offsetX = minX;
    const offsetY = minY;
    const rad = (line.transform.rotation * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    const rotatedOffsetX =
      offsetX * line.transform.scaleX * cos -
      offsetY * line.transform.scaleY * sin;
    const rotatedOffsetY =
      offsetX * line.transform.scaleX * sin +
      offsetY * line.transform.scaleY * cos;

    return {
      x: line.transform.x + rotatedOffsetX,
      y: line.transform.y + rotatedOffsetY,
      width: maxX - minX,
      height: maxY - minY,
      minX,
      minY,
    };
  }, [line.points, line.transform, line.strokeWidth]);

  const bbox = getBoundingBox();

  const updateGuidesAndMaybeSnap = useCallback(
    (opts: { snap: boolean }) => {
      const rect = rectRef.current;
      const group = groupRef.current;
      if (!rect || !group) return;

      const thresholdLocal = SNAP_THRESHOLD_PX;
      const client = rect.getClientRect({
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
          rect.x(rect.x() + deltaX);
          rect.y(rect.y() + deltaY);
          // Update line position accordingly
          if (ref.current) {
            const rectX = rect.x();
            const rectY = rect.y();
            const rectScaleX = rect.scaleX();
            const rectScaleY = rect.scaleY();
            const rectRotation = rect.rotation();

            const offsetX = bbox.minX;
            const offsetY = bbox.minY;
            const rad = (rectRotation * Math.PI) / 180;
            const cos = Math.cos(rad);
            const sin = Math.sin(rad);
            const rotatedOffsetX =
              offsetX * rectScaleX * cos - offsetY * rectScaleY * sin;
            const rotatedOffsetY =
              offsetX * rectScaleX * sin + offsetY * rectScaleY * cos;

            ref.current.x(rectX - rotatedOffsetX);
            ref.current.y(rectY - rotatedOffsetY);
          }
        }
      }
    },
    [rectWidth, rectHeight, bbox.minX, bbox.minY],
  );

  useEffect(() => {
    // Sync rect position with line transform, but skip during active transformation
    if (rectRef.current && !isTransformingRef.current) {
      rectRef.current.x(bbox.x);
      rectRef.current.y(bbox.y);
      rectRef.current.scaleX(line.transform.scaleX);
      rectRef.current.scaleY(line.transform.scaleY);
      rectRef.current.rotation(line.transform.rotation);
      rectRef.current.getLayer()?.batchDraw();
    }
  }, [
    line.transform.x,
    line.transform.y,
    line.transform.scaleX,
    line.transform.scaleY,
    line.transform.rotation,
    bbox.x,
    bbox.y,
  ]);

  const handleRectDragStart = useCallback(() => {
    isTransformingRef.current = true;
  }, []);

  const handleRectDrag = useCallback(
    (e: Konva.KonvaEventObject<DragEvent>) => {
      const rect = e.target as Konva.Rect;
      const rectX = rect.x();
      const rectY = rect.y();
      const rectScaleX = rect.scaleX();
      const rectScaleY = rect.scaleY();
      const rectRotation = rect.rotation();

      // Calculate the offset from rect's top-left to the line's origin
      const offsetX = bbox.minX;
      const offsetY = bbox.minY;

      // Apply rotation to the offset vector
      const rad = (rectRotation * Math.PI) / 180;
      const cos = Math.cos(rad);
      const sin = Math.sin(rad);

      const rotatedOffsetX =
        offsetX * rectScaleX * cos - offsetY * rectScaleY * sin;
      const rotatedOffsetY =
        offsetX * rectScaleX * sin + offsetY * rectScaleY * cos;

      if (ref.current) {
        ref.current.x(rectX - rotatedOffsetX);
        ref.current.y(rectY - rotatedOffsetY);
        ref.current.getLayer()?.batchDraw();
      }

      updateGuidesAndMaybeSnap({ snap: true });
    },
    [bbox.minX, bbox.minY, updateGuidesAndMaybeSnap],
  );

  const handleRectDragEnd = useCallback(() => {
    isTransformingRef.current = false;
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
    handleDragEnd(line.lineId);
  }, [handleDragEnd, line.lineId]);

  const handleRectTransformStart = useCallback(() => {
    isTransformingRef.current = true;
  }, []);

  const handleRectTransform = useCallback(() => {
    const rect = rectRef.current;
    const lineNode = ref.current;

    if (rect && lineNode) {
      const rectX = rect.x();
      const rectY = rect.y();
      const rectScaleX = rect.scaleX();
      const rectScaleY = rect.scaleY();
      const rectRotation = rect.rotation();

      // Calculate the offset from rect's top-left to the line's origin
      const offsetX = bbox.minX;
      const offsetY = bbox.minY;

      // Apply rotation to the offset vector
      const rad = (rectRotation * Math.PI) / 180;
      const cos = Math.cos(rad);
      const sin = Math.sin(rad);

      const rotatedOffsetX =
        offsetX * rectScaleX * cos - offsetY * rectScaleY * sin;
      const rotatedOffsetY =
        offsetX * rectScaleX * sin + offsetY * rectScaleY * cos;

      lineNode.x(rectX - rotatedOffsetX);
      lineNode.y(rectY - rotatedOffsetY);
      lineNode.scaleX(rectScaleX);
      lineNode.scaleY(rectScaleY);
      lineNode.rotation(rectRotation);
      lineNode.getLayer()?.batchDraw();
    }

    updateGuidesAndMaybeSnap({ snap: false });
  }, [bbox.minX, bbox.minY, updateGuidesAndMaybeSnap]);

  const handleRectTransformEnd = useCallback(() => {
    isTransformingRef.current = false;
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
    handleTransformEnd(line.lineId);
  }, [handleTransformEnd, line.lineId]);

  const transformerBoundBoxFunc = useCallback(
    (_oldBox: any, newBox: any) => {
      const thresholdLocal = SNAP_THRESHOLD_PX;

      const left = newBox.x;
      const right = newBox.x + newBox.width;
      const top = newBox.y;
      const bottom = newBox.y + newBox.height;
      const centerX = newBox.x + newBox.width / 2;
      const centerY = newBox.y + newBox.height / 2;
      const v25 = rectWidth * 0.25;
      const v75 = rectWidth * 0.75;
      const h25 = rectHeight * 0.25;
      const h75 = rectHeight * 0.75;
      const vCenter = rectWidth / 2;
      const hCenter = rectHeight / 2;

      let adjustedBox = { ...newBox };

      // Snap left edge
      if (Math.abs(left - 0) <= thresholdLocal) {
        adjustedBox.x = 0;
        adjustedBox.width = right - adjustedBox.x;
      } else if (Math.abs(left - v25) <= thresholdLocal) {
        adjustedBox.x = v25;
        adjustedBox.width = right - adjustedBox.x;
      } else if (Math.abs(left - v75) <= thresholdLocal) {
        adjustedBox.x = v75;
        adjustedBox.width = right - adjustedBox.x;
      } else if (Math.abs(centerX - vCenter) <= thresholdLocal) {
        const halfWidth = newBox.width / 2;
        adjustedBox.x = vCenter - halfWidth;
      }

      // Snap right edge
      if (Math.abs(right - rectWidth) <= thresholdLocal) {
        adjustedBox.width = rectWidth - adjustedBox.x;
      } else if (Math.abs(right - v75) <= thresholdLocal) {
        adjustedBox.width = v75 - adjustedBox.x;
      } else if (Math.abs(right - v25) <= thresholdLocal) {
        adjustedBox.width = v25 - adjustedBox.x;
      }

      // Snap top edge
      if (Math.abs(top - 0) <= thresholdLocal) {
        adjustedBox.y = 0;
        adjustedBox.height = bottom - adjustedBox.y;
      } else if (Math.abs(top - h25) <= thresholdLocal) {
        adjustedBox.y = h25;
        adjustedBox.height = bottom - adjustedBox.y;
      } else if (Math.abs(top - h75) <= thresholdLocal) {
        adjustedBox.y = h75;
        adjustedBox.height = bottom - adjustedBox.y;
      } else if (Math.abs(centerY - hCenter) <= thresholdLocal) {
        const halfHeight = newBox.height / 2;
        adjustedBox.y = hCenter - halfHeight;
      }

      // Snap bottom edge
      if (Math.abs(bottom - rectHeight) <= thresholdLocal) {
        adjustedBox.height = rectHeight - adjustedBox.y;
      } else if (Math.abs(bottom - h75) <= thresholdLocal) {
        adjustedBox.height = h75 - adjustedBox.y;
      } else if (Math.abs(bottom - h25) <= thresholdLocal) {
        adjustedBox.height = h25 - adjustedBox.y;
      }

      const MIN_SIZE_ABS = 5;
      if (adjustedBox.width < MIN_SIZE_ABS) adjustedBox.width = MIN_SIZE_ABS;
      if (adjustedBox.height < MIN_SIZE_ABS) adjustedBox.height = MIN_SIZE_ABS;

      return adjustedBox;
    },
    [rectWidth, rectHeight],
  );

  const applicatorProps = useMemo(
    () => ({
      applicators,
      //@ts-ignore
      filters: filtersArray,
    }),
    [applicators, filtersArray],
  );

  return (
    <>
      <Group
        ref={groupRef}
        clipX={0}
        clipY={0}
        clipWidth={rectWidth}
        clipHeight={rectHeight}
      >
        <KonvaLine
          key={line.lineId}
          points={line.points}
          stroke={line.stroke}
          strokeWidth={line.strokeWidth}
          opacity={lineOpacity}
          x={line.transform.x}
          y={line.transform.y}
          scaleX={line.transform.scaleX}
          scaleY={line.transform.scaleY}
          rotation={line.transform.rotation}
          onClick={(e) => handleLineClick(e, line.lineId)}
          onTap={(e) => handleLineClick(e, line.lineId)}
          onDragEnd={() => handleDragEnd(line.lineId)}
          onTransformEnd={() => handleTransformEnd(line.lineId)}
          lineCap={toolConfig.lineCap}
          lineJoin={toolConfig.lineJoin}
          tension={toolConfig.tension}
          ref={ref}
          {...applicatorProps}
        />
        <Rect
          ref={rectRef}
          stroke={"#AE81CE"}
          strokeWidth={0}
          x={bbox.x}
          y={bbox.y}
          width={bbox.width}
          height={bbox.height}
          scaleX={line.transform.scaleX}
          scaleY={line.transform.scaleY}
          rotation={line.transform.rotation}
          offsetX={0}
          offsetY={0}
          visible={tool === "pointer"}
          draggable={
            tool === "pointer" &&
            selectedLineId === line.lineId &&
            isLineSelected
          }
          onDragStart={handleRectDragStart}
          onDragMove={handleRectDrag}
          onDragEnd={handleRectDragEnd}
          onTransformStart={handleRectTransformStart}
          onTransform={handleRectTransform}
          onTransformEnd={handleRectTransformEnd}
          onClick={(e) => {
            handleLineClick(e, line.lineId);
          }}
          onTap={(e) => handleLineClick(e, line.lineId)}
        />

        {/* Guide Lines */}
        {guides.vCenter && (
          <Line
            points={[rectWidth / 2, 0, rectWidth / 2, rectHeight]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.hCenter && (
          <Line
            points={[0, rectHeight / 2, rectWidth, rectHeight / 2]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.v25 && (
          <Line
            points={[rectWidth * 0.25, 0, rectWidth * 0.25, rectHeight]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.v75 && (
          <Line
            points={[rectWidth * 0.75, 0, rectWidth * 0.75, rectHeight]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.h25 && (
          <Line
            points={[0, rectHeight * 0.25, rectWidth, rectHeight * 0.25]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.h75 && (
          <Line
            points={[0, rectHeight * 0.75, rectWidth, rectHeight * 0.75]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.left && (
          <Line
            points={[0, 0, 0, rectHeight]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.right && (
          <Line
            points={[rectWidth, 0, rectWidth, rectHeight]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.top && (
          <Line
            points={[0, 0, rectWidth, 0]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
        {guides.bottom && (
          <Line
            points={[0, rectHeight, rectWidth, rectHeight]}
            stroke="#AE81CE"
            strokeWidth={1}
            dash={[6, 4]}
            listening={false}
          />
        )}
      </Group>
      <Transformer
        borderStroke="#AE81CE"
        anchorCornerRadius={8}
        anchorStroke="#E3E3E3"
        anchorStrokeWidth={1}
        visible={
          tool === "pointer" && selectedLineId === line.lineId && isLineSelected
        }
        borderStrokeWidth={2}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]}
        ref={transformerRef}
        rotateEnabled={true}
        enabledAnchors={[
          "top-left",
          "top-right",
          "bottom-left",
          "bottom-right",
          "top-center",
          "middle-left",
          "middle-right",
          "bottom-center",
        ]}
        boundBoxFunc={transformerBoundBoxFunc}
      />
    </>
  );
};

export default DrawingLineComponent;
