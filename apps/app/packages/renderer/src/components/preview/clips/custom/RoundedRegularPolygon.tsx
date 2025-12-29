import React from "react";
import { RegularPolygon } from "react-konva";
import Konva from "konva";
import { BaseClipApplicator } from "../apply/base";

interface RoundedRegularPolygonProps {
  x: number;
  y: number;
  sides: number;
  radius: number;
  cornerRadius?: number;
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  rotation?: number;
  scaleX?: number;
  scaleY?: number;
  draggable?: boolean;
  onDragStart?: (e: Konva.KonvaEventObject<MouseEvent>) => void;
  onDragMove?: (e: Konva.KonvaEventObject<MouseEvent>) => void;
  onDragEnd?: (e: Konva.KonvaEventObject<MouseEvent>) => void;
  onClick?: () => void;
  filters?: any[];
  applicators?: BaseClipApplicator[];
}

const RoundedRegularPolygon = React.forwardRef<
  Konva.Shape,
  RoundedRegularPolygonProps
>((props, ref) => {
  const {
    x,
    y,
    sides,
    radius,
    cornerRadius = 0,
    fill,
    stroke,
    strokeWidth,
    rotation,
    scaleX,
    scaleY,
    draggable,
    onDragStart,
    onDragMove,
    onDragEnd,
    onClick,
    filters,
    applicators,
  } = props;

  const getPoints = (r: number, s: number) => {
    const points = [];
    for (let n = 0; n < s; n++) {
      points.push({
        x: r * Math.sin((n * 2 * Math.PI) / s),
        y: -1 * r * Math.cos((n * 2 * Math.PI) / s),
      });
    }
    return points;
  };

  const sceneFunc = (context: Konva.Context, shape: Konva.Shape) => {
    const points = getPoints(radius, sides);

    if (cornerRadius === 0 || points.length < 3) {
      // No corner radius, draw regular polygon
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      for (let n = 1; n < points.length; n++) {
        context.lineTo(points[n].x, points[n].y);
      }
      context.closePath();
      context.fillStrokeShape(shape);
      return;
    }

    // Draw polygon with rounded corners
    context.beginPath();

    for (let i = 0; i < points.length; i++) {
      const current = points[i];
      const next = points[(i + 1) % points.length];
      const prev = points[(i - 1 + points.length) % points.length];

      // Calculate vectors
      const v1x = current.x - prev.x;
      const v1y = current.y - prev.y;
      const v2x = next.x - current.x;
      const v2y = next.y - current.y;

      // Normalize vectors
      const len1 = Math.sqrt(v1x * v1x + v1y * v1y);
      const len2 = Math.sqrt(v2x * v2x + v2y * v2y);
      const n1x = v1x / len1;
      const n1y = v1y / len1;
      const n2x = v2x / len2;
      const n2y = v2y / len2;

      // Calculate the effective corner radius (can't be larger than half the edge length)
      const maxRadius = Math.min(len1, len2) / 2;
      const effectiveRadius = Math.min(cornerRadius, maxRadius);

      // Calculate start and end points for the arc
      const startX = current.x - n1x * effectiveRadius;
      const startY = current.y - n1y * effectiveRadius;
      const endX = current.x + n2x * effectiveRadius;
      const endY = current.y + n2y * effectiveRadius;

      if (i === 0) {
        context.moveTo(startX, startY);
      } else {
        context.lineTo(startX, startY);
      }

      // Draw the rounded corner using quadratic curve
      context.quadraticCurveTo(current.x, current.y, endX, endY);
    }

    context.closePath();
    context.fillStrokeShape(shape);
  };

  const handleRef = React.useCallback(
    (node: Konva.RegularPolygon) => {
      if (node) {
        if (typeof ref === "function") {
          ref(node);
        } else if (ref) {
          ref.current = node;
        }
      }
    },
    [sides, radius],
  );

  return (
    <RegularPolygon
      ref={handleRef}
      x={x}
      y={y}
      sceneFunc={sceneFunc}
      sides={sides}
      radius={radius}
      fill={fill}
      stroke={stroke}
      strokeWidth={strokeWidth}
      rotation={rotation}
      scaleX={scaleX}
      scaleY={scaleY}
      draggable={draggable}
      onDragStart={onDragStart}
      onDragMove={onDragMove}
      onDragEnd={onDragEnd}
      onClick={onClick}
      //@ts-ignore
      filters={filters}
      //@ts-ignore
      applicators={applicators}
    />
  );
});

RoundedRegularPolygon.displayName = "RoundedRegularPolygon";

export default RoundedRegularPolygon;
