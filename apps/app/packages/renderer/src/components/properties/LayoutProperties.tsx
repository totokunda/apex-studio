import { useClipStore } from "@/lib/clip";
import { AnyClipProps, PolygonClipProps, StarClipProps } from "@/lib/types";
import { useState } from "react";
import React from "react";
import { IoRefreshOutline, IoLockClosed, IoLockOpen } from "react-icons/io5";
import Input from "./Input";

interface LayoutPropertiesProps {
  clipId: string;
}

const LayoutProperties: React.FC<LayoutPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as AnyClipProps;
  const setClipTransform = useClipStore((s) => s.setClipTransform);
  const updateClip = useClipStore((s) => s.updateClip);

  const hasSides = clip?.type === "shape" && clip?.shapeType === "polygon";
  const hasPoints = clip?.type === "shape" && clip?.shapeType === "star";
  const hasScale = clip?.type === "image" || clip?.type === "video" || clip?.type === "model";
  const hasWidthHeight = clip?.type === "shape";

  const [spinning, setSpinning] = useState(false);
  const [scaleLocked, setScaleLocked] = useState(true);

  const getEffectiveScaleX = () => {
    if (!clip?.transform) return 1;
    const base = clip.originalTransform || clip.transform;
    if (base.width && clip.transform.width) {
      return clip.transform.width / base.width;
    }
    return clip.transform.scaleX ?? 1;
  };

  const getEffectiveScaleY = () => {
    if (!clip?.transform) return 1;
    const base = clip.originalTransform || clip.transform;
    if (base.height && clip.transform.height) {
      return clip.transform.height / base.height;
    }
    return clip.transform.scaleY ?? 1;
  };

  const updateSides = (value: number) => {
    if (isNaN(value) || !isFinite(value)) return;
    if (value < 3 || value > 12) return;
    updateClip(clipId, { sides: value });
  };

  const updatePoints = (value: number) => {
    if (isNaN(value) || !isFinite(value)) return;
    if (value < 4 || value > 100) return;
    const updatedClip = { ...clip, points: value };
    updateClip(clipId, updatedClip);
  };

  const handleReset = () => {
    if (!clip?.transform) return;

    setClipTransform(clipId, {
      width: clip.originalTransform?.width,
      height: clip.originalTransform?.height,
      scaleX: 1,
      scaleY: 1,
    });

    setSpinning(true);
    setTimeout(() => {
      setSpinning(false);
    }, 500);
  };

  const handleWidthChange = (value: string) => {
    if (!clip?.transform) return;
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue) || numValue <= 0) return;

    setClipTransform(clipId, { width: numValue });
  };

  const handleHeightChange = (value: string) => {
    if (!clip?.transform) return;
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue) || numValue <= 0) return;

    setClipTransform(clipId, { height: numValue });
  };
  

  const handleScaleXChange = (value: string) => {
    if (!clip?.transform) return;
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;

    // For image/video clips, interpret scale as changing width/height,
    // and keep transform scale at 1 so we don't double‑scale.
    if (hasScale) {
      const base = clip.originalTransform || clip.transform;
      if (!base) return;

      if (scaleLocked) {
        setClipTransform(clipId, {
          width: base.width * numValue,
          height: base.height * numValue,
          scaleX: 1,
          scaleY: 1,
        });
      } else {
        const currentHeight = clip.transform.height;
        const baseHeight = base.height;
        const currentScaleY =
          baseHeight && currentHeight ? currentHeight / baseHeight : 1;

        setClipTransform(clipId, {
          width: base.width * numValue,
          height: base.height * currentScaleY,
          scaleX: 1,
          scaleY: 1,
        });
      }
      return;
    }

    // Fallback behavior (if we ever show scale for other clip types)
    if (scaleLocked) {
      setClipTransform(clipId, { scaleX: numValue, scaleY: numValue });
    } else {
      setClipTransform(clipId, { scaleX: numValue });
    }
  };

  const handleScaleYChange = (value: string) => {
    if (!clip?.transform) return;
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;

    // For image/video clips, interpret scale as changing width/height,
    // and keep transform scale at 1 so we don't double‑scale.
    if (hasScale) {
      const base = clip.originalTransform || clip.transform;
      if (!base) return;

      if (scaleLocked) {
        setClipTransform(clipId, {
          width: base.width * numValue,
          height: base.height * numValue,
          scaleX: 1,
          scaleY: 1,
        });
      } else {
        const currentWidth = clip.transform.width;
        const baseWidth = base.width;
        const currentScaleX =
          baseWidth && currentWidth ? currentWidth / baseWidth : 1;

        setClipTransform(clipId, {
          width: base.width * currentScaleX,
          height: base.height * numValue,
          scaleX: 1,
          scaleY: 1,
        });
      }
      return;
    }

    // Fallback behavior (if we ever show scale for other clip types)
    if (scaleLocked) {
      setClipTransform(clipId, { scaleX: numValue, scaleY: numValue });
    } else {
      setClipTransform(clipId, { scaleY: numValue });
    }
  };

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 flex flex-col gap-y-4 px-5 min-w-0">
        <div className="flex flex-row items-center justify-between">
          <h4 className="text-brand-light text-[12px] font-medium text-start">
            Layout
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
          {hasWidthHeight && (
            <div className="flex flex-row gap-x-2">
            <Input
              label="Size"
              value={clip?.transform?.width.toFixed(0).toString() ?? "0"}
              onChange={handleWidthChange}
              startLogo="W"
            />
            <Input
              emptyLabel
              value={clip?.transform?.height.toFixed(0).toString() ?? "0"}
              onChange={handleHeightChange}
              startLogo="H"
            />
          </div>)}

          {hasScale && (
            <div className="flex flex-col ">
              <div className="flex flex-row items-center justify-between">
                <span className="text-brand-light text-[10px] text-start">
                  Scale
                </span>
                <button
                  onClick={() => setScaleLocked(!scaleLocked)}
                  className="text-brand-light/70 hover:text-brand-light transition-colors cursor-pointer"
                >
                  {scaleLocked ? (
                    <IoLockClosed className="h-3 w-3" />
                  ) : (
                    <IoLockOpen className="h-3 w-3" />
                  )}
                </button>
              </div>
              <div className="flex flex-row gap-x-2">
                <Input
                  value={
                    getEffectiveScaleX().toFixed(2).toString() ?? "1.00"
                  }
                  onChange={handleScaleXChange}
                  startLogo="X"
                  canStep
                  step={0.1}
                  min={0.1}
                />
                <Input
                  value={
                    getEffectiveScaleY().toFixed(2).toString() ?? "1.00"
                  }
                  onChange={handleScaleYChange}
                  startLogo="Y"
                  canStep
                  step={0.1}
                  min={0.1}
                />
              </div>
            </div>
          )}
          {hasSides && (
            <div className="flex flex-row gap-x-2">
              <Input
                label="Sides"
                value={(clip as PolygonClipProps)?.sides?.toString() ?? "3"}
                onChange={(value) => updateSides(Number(value))}
                startLogo="S"
                canStep
                step={1}
                min={3}
                max={12}
              />
            </div>
          )}
          {hasPoints && (
            <div className="flex flex-row gap-x-2">
              <Input
                label="Points"
                value={(clip as StarClipProps)?.points?.toString() ?? "5"}
                onChange={(value) => updatePoints(Number(value))}
                startLogo="P"
                canStep
                step={1}
                min={3}
                max={12}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LayoutProperties;
