import React from "react";
import { Group, Line as KonvaLine, Rect } from "react-konva";
import { MaskClipProps } from "@/lib/types";

interface DrawMaskPreviewProps {
  mask: MaskClipProps;
  drawStrokes: Array<{ points: number[]; strokeWidth: number }>;
  animationOffset: number;
  rectWidth: number;
  rectHeight: number;
}

const DrawMaskPreview: React.FC<DrawMaskPreviewProps> = ({
  mask,
  drawStrokes,
  animationOffset,
  rectWidth,
  rectHeight,
}) => {
  return (
    <Group
      clipX={0}
      clipY={0}
      clipWidth={rectWidth}
      clipHeight={rectHeight}
    ></Group>
  );
};

export default DrawMaskPreview;
