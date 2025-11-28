import { useClipStore } from "@/lib/clip";
import { AnyClipProps } from "@/lib/types";
import { useState } from "react";
import React from "react";
import { IoRefreshOutline } from "react-icons/io5";
import Input from "./Input";
import {
  PiAlignLeft,
  PiAlignRight,
  PiAlignCenterVertical,
  PiAlignTop,
  PiAlignBottom,
  PiAlignCenterHorizontal,
} from "react-icons/pi";
import { useViewportStore } from "@/lib/viewport";

interface PositionPropertiesProps {
  clipId: string;
}

const PositionProperties: React.FC<PositionPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as AnyClipProps;
  const { updateClip } = useClipStore();
  const contentBounds = useViewportStore((s) => s.contentBounds);
  const setClipTransform = useClipStore((s) => s.setClipTransform);

  const [spinning, setSpinning] = useState(false);

  const handleRotationChange = (newRotation: number) => {
    if (!clip?.transform) return;
    if (!isFinite(newRotation)) return;

    const {
      x = 0,
      y = 0,
      width = 0,
      height = 0,
      scaleX = 1,
      scaleY = 1,
      rotation = 0,
    } = clip.transform;

    const scaledWidth = (width || 0) * (scaleX || 1);
    const scaledHeight = (height || 0) * (scaleY || 1);

    // Vector from top-left to visual center
    const vx = scaledWidth / 2;
    const vy = scaledHeight / 2;

    const r0 = (rotation * Math.PI) / 180;
    const r1 = (newRotation * Math.PI) / 180;

    // Rotate v by r: R(r) * v
    const r0vx = vx * Math.cos(r0) - vy * Math.sin(r0);
    const r0vy = vx * Math.sin(r0) + vy * Math.cos(r0);
    const r1vx = vx * Math.cos(r1) - vy * Math.sin(r1);
    const r1vy = vx * Math.sin(r1) + vy * Math.cos(r1);

    // Keep center fixed: P' = P + R(r0)v - R(r1)v
    const nx = x + (r0vx - r1vx);
    const ny = y + (r0vy - r1vy);

    setClipTransform(clipId, { x: nx, y: ny, rotation: newRotation });
  };

  const handleAlign = (
    type: "left" | "center-h" | "right" | "top" | "center-v" | "bottom",
  ) => {
    if (!clip?.transform || !contentBounds) return;

    const { width, height, x, y, scaleX, scaleY } = clip.transform;
    const containerWidth = contentBounds.width;
    const containerHeight = contentBounds.height;

    const scaledWidth = width * scaleX;
    const scaledHeight = height * scaleY;

    let newX = x;
    let newY = y;

    switch (type) {
      case "left":
        newX = 0;
        break;
      case "center-h":
        newX = (containerWidth - scaledWidth) / 2;
        break;
      case "right":
        newX = containerWidth - scaledWidth;
        break;
      case "top":
        newY = 0;
        break;
      case "center-v":
        newY = (containerHeight - scaledHeight) / 2;
        break;
      case "bottom":
        newY = containerHeight - scaledHeight;
        break;
    }

    setClipTransform(clipId, { x: newX, y: newY });
  };

  const handleReset = () => {
    if (!clip?.transform || !contentBounds) return;

    const { width, height, scaleX, scaleY } = clip.transform;
    const containerWidth = contentBounds.width;
    const containerHeight = contentBounds.height;

    const scaledWidth = width * scaleX;
    const scaledHeight = height * scaleY;

    const centerX = (containerWidth - scaledWidth) / 2;
    const centerY = (containerHeight - scaledHeight) / 2;

    setClipTransform(clipId, { x: centerX, y: centerY, rotation: 0 });

    setSpinning(true);
    setTimeout(() => {
      setSpinning(false);
    }, 500);
  };

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 flex flex-col gap-y-4 px-5 min-w-0">
        <div className="flex flex-row items-center justify-between ">
          <h4 className="text-brand-light text-[12px] font-medium text-start">
            Position
          </h4>
          <span
            onClick={handleReset}
            className="text-brand-light text-sm cursor-pointer"
          >
            <IoRefreshOutline
              className={spinning ? "animate-spin duration-500" : ""}
              onAnimationEnd={() => setSpinning(false)}
            />
          </span>
        </div>
        <div className="flex flex-col gap-y-2">
          <span className="text-brand-light text-[10.5px] font-medium text-start">
            Alignment
          </span>
          <div className="flex flex-row gap-x-2 w-full ">
            <div className="flex flex-row divide-x divide-brand-light/10 h-6 w-1/2 cursor-pointer rounded bg-brand">
              <button
                onClick={() => handleAlign("left")}
                className="flex flex-row items-center justify-center w-full hover:bg-brand-light/10 transition-all duration-200 cursor-pointer rounded-l"
              >
                <PiAlignLeft className="text-brand-light h-4 w-4" />
              </button>

              <button
                onClick={() => handleAlign("center-h")}
                className="flex flex-row items-center justify-center w-full hover:bg-brand-light/10 transition-all duration-200 cursor-pointer "
              >
                <PiAlignCenterHorizontal className="text-brand-light h-4 w-4" />
              </button>
              <button
                onClick={() => handleAlign("right")}
                className="flex flex-row items-center justify-center w-full hover:bg-brand-light/10 transition-all duration-200 cursor-pointer rounded-r"
              >
                <PiAlignRight className="text-brand-light h-4 w-4" />
              </button>
            </div>
            <div className="flex flex-row divide-x divide-brand-light/10 h-6 w-1/2 cursor-pointer rounded bg-brand">
              <button
                onClick={() => handleAlign("top")}
                className="flex flex-row items-center justify-center w-full hover:bg-brand-light/10 transition-all duration-200 cursor-pointer rounded-l"
              >
                <PiAlignTop className="text-brand-light h-4 w-4" />
              </button>
              <button
                onClick={() => handleAlign("center-v")}
                className="flex flex-row items-center justify-center w-full hover:bg-brand-light/10 transition-all duration-200 cursor-pointer"
              >
                <PiAlignCenterVertical className="text-brand-light h-4 w-4" />
              </button>
              <button
                onClick={() => handleAlign("bottom")}
                className="flex flex-row items-center justify-center w-full hover:bg-brand-light/10 transition-all duration-200 cursor-pointer rounded-r"
              >
                <PiAlignBottom className="text-brand-light h-4 w-4" />
              </button>
            </div>
          </div>
        </div>

        <div className="flex flex-row gap-x-2">
          <Input
            label="Position"
            value={clip?.transform?.x.toFixed(0).toString() ?? "0"}
            onChange={(value) =>
              updateClip(clipId, {
                transform: { ...clip?.transform!, x: Number(value) },
              })
            }
            startLogo="X"
          />
          <Input
            emptyLabel
            value={clip?.transform?.y.toFixed(0).toString() ?? "0"}
            onChange={(value) =>
              updateClip(clipId, {
                transform: { ...clip?.transform!, y: Number(value) },
              })
            }
            startLogo="Y"
          />
        </div>
        <Input
          label="Rotation"
          value={clip?.transform?.rotation.toFixed(0).toString() ?? "0"}
          onChange={(value) => handleRotationChange(Number(value))}
          startLogo="R"
        />
      </div>
    </div>
  );
};

export default PositionProperties;
