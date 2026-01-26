import { useClipStore } from "@/lib/clip";
import { DrawingClipProps } from "@/lib/types";
import { useState } from "react";
import React from "react";
import { IoRefreshOutline } from "react-icons/io5";
import Input from "./Input";
import { useDrawingStore } from "@/lib/drawing";

interface LinePropertiesProps {
  clipId: string;
}

const LineProperties: React.FC<LinePropertiesProps> = ({ clipId }) => {
  const updateClip = useClipStore((s) => s.updateClip);
  const selectedLineId = useDrawingStore((s) => s.selectedLineId);

  const selectedLine = useClipStore((s) => {
    const clip = s.getClipById(clipId) as DrawingClipProps;
    return clip?.lines?.find((line) => line.lineId === selectedLineId);
  });

  const clip = useClipStore((s) => s.getClipById(clipId)) as DrawingClipProps;

  const [spinning, setSpinning] = useState(false);

  if (!selectedLine || !selectedLineId) {
    return null;
  }

  const handleReset = () => {
    if (!selectedLine?.transform) return;

    const updatedLines = clip.lines.map((l) => {
      if (l.lineId === selectedLineId) {
        return {
          ...l,
          transform: {
            ...l.transform,
            x: 0,
            y: 0,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
          },
        };
      }
      return l;
    });

    updateClip(clipId, { lines: updatedLines });

    setSpinning(true);
    setTimeout(() => {
      setSpinning(false);
    }, 500);
  };

  const updateLineTransform = (
    updates: Partial<typeof selectedLine.transform>,
  ) => {
    const updatedLines = clip.lines.map((l) => {
      if (l.lineId === selectedLineId) {
        return {
          ...l,
          transform: {
            ...l.transform,
            ...updates,
          },
        };
      }
      return l;
    });

    updateClip(clipId, { lines: updatedLines });
  };

  const handleXChange = (value: string) => {
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    updateLineTransform({ x: numValue });
  };

  const handleYChange = (value: string) => {
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    updateLineTransform({ y: numValue });
  };

  const handleRotationChange = (value: string) => {
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    updateLineTransform({ rotation: numValue });
  };

  const handleScaleXChange = (value: string) => {
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    updateLineTransform({ scaleX: numValue });
  };

  const handleScaleYChange = (value: string) => {
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    updateLineTransform({ scaleY: numValue });
  };

  const handleOpacityChange = (value: string) => {
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    if (numValue < 0 || numValue > 100) return;
    updateLineTransform({ opacity: numValue });
  };

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 flex flex-col gap-y-4 px-5 min-w-0">
        <div className="flex flex-row items-center justify-between">
          <h4 className="text-brand-light text-[12px] font-medium text-start">
            Line
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

        <div className="flex flex-col gap-y-3">
          <div className="flex flex-row gap-x-2">
            <Input
              label="Position"
              value={selectedLine.transform.x.toFixed(0).toString()}
              onChange={handleXChange}
              startLogo="X"
            />
            <Input
              emptyLabel
              value={selectedLine.transform.y.toFixed(0).toString()}
              onChange={handleYChange}
              startLogo="Y"
            />
          </div>

          <Input
            label="Rotation"
            value={selectedLine.transform.rotation.toFixed(0).toString()}
            onChange={handleRotationChange}
            startLogo="R"
          />

          <div className="flex flex-col gap-y-1">
            <span className="text-brand-light text-[10px] text-start">
              Scale
            </span>
            <div className="flex flex-row gap-x-2">
              <Input
                value={selectedLine.transform.scaleX.toFixed(2).toString()}
                onChange={handleScaleXChange}
                startLogo="X"
                canStep
                step={0.1}
                min={0.1}
              />
              <Input
                value={selectedLine.transform.scaleY.toFixed(2).toString()}
                onChange={handleScaleYChange}
                startLogo="Y"
                canStep
                step={0.1}
                min={0.1}
              />
            </div>
          </div>

          <Input
            label="Opacity"
            value={selectedLine.transform.opacity.toFixed(0).toString()}
            onChange={handleOpacityChange}
            startLogo="O"
            canStep
            step={1}
            min={0}
            max={100}
          />
        </div>
      </div>
    </div>
  );
};

export default LineProperties;
