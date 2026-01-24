import React from "react";
import { Group } from "react-konva";
import { MaskClipProps } from "@/lib/types";

interface DrawMaskPreviewProps {
  mask: MaskClipProps;
  drawStrokes: Array<{ points: number[]; strokeWidth: number }>;
  animationOffset: number;
  rectWidth: number;
  rectHeight: number;
}

const DrawMaskPreview: React.FC<DrawMaskPreviewProps> = ({
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
