import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { MaskClipProps } from "@/lib/types";
import { useMemo } from "react";
import React from "react";
import Input from "../Input";
import { upsertMaskKeyframe } from "@/lib/mask/keyframeUtils";

interface LassoMaskTransformPropertiesProps {
  mask: MaskClipProps;
  clipId: string;
}

const LassoMaskTransformProperties: React.FC<
  LassoMaskTransformPropertiesProps
> = ({ mask, clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId));
  const updateClip = useClipStore((s) => s.updateClip);
  const clipWithinFrame = useClipStore((s) => s.clipWithinFrame);
  const focusFrame = useControlsStore((s) => s.focusFrame);

  const activeKeyframeData = useMemo(() => {
    if (!mask || !clip) return null;
    if (!clipWithinFrame(clip as any, focusFrame)) return null;

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

    if (keyframeNumbers.length === 0) return null;

    // Compute local frame relative to the clip
    const startFrame = (clip as any).startFrame ?? 0;
    const trimStart = isFinite((clip as any).trimStart ?? 0)
      ? ((clip as any).trimStart ?? 0)
      : 0;
    const realStartFrame = startFrame + trimStart;
    const localFrame = focusFrame - realStartFrame;

    const nearestKeyframe = (frame: number) => {
      if (frame < keyframeNumbers[0]) return keyframeNumbers[0];
      const atOrBefore = keyframeNumbers.filter((k) => k <= frame).pop();
      return atOrBefore ?? keyframeNumbers[keyframeNumbers.length - 1];
    };

    const activeKeyframe =
      nearestKeyframe(localFrame) ?? nearestKeyframe(focusFrame);
    const maskData =
      keyframes instanceof Map
        ? keyframes.get(activeKeyframe)
        : keyframes[activeKeyframe];

    if (!maskData) return null;
    return { keyframe: activeKeyframe, data: maskData };
  }, [mask, clip, focusFrame, clipWithinFrame]);

  const lassoPoints = useMemo(() => {
    return activeKeyframeData?.data?.lassoPoints || [];
  }, [activeKeyframeData]);

  // Calculate bounding box from lasso points
  const bounds = useMemo(() => {
    if (lassoPoints.length < 2) return { x: 0, y: 0, centerX: 0, centerY: 0 };

    let minX = lassoPoints[0];
    let maxX = lassoPoints[0];
    let minY = lassoPoints[1];
    let maxY = lassoPoints[1];

    for (let i = 0; i < lassoPoints.length; i += 2) {
      minX = Math.min(minX, lassoPoints[i]);
      maxX = Math.max(maxX, lassoPoints[i]);
      minY = Math.min(minY, lassoPoints[i + 1]);
      maxY = Math.max(maxY, lassoPoints[i + 1]);
    }

    return {
      x: minX,
      y: minY,
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2,
    };
  }, [lassoPoints]);

  const updateLassoPoints = (newPoints: number[]) => {
    if (!mask || !activeKeyframeData) return;

    if (!clip) return;

    const result = upsertMaskKeyframe({
      mask,
      clip,
      focusFrame,
      updater: (previous) => ({
        ...previous,
        lassoPoints: newPoints,
      }),
    });

    if (!result) return;

    const masks = (clip as any).masks || [];
    const now = Date.now();
    const updatedMasks = masks.map((m: MaskClipProps) =>
      m.id === mask.id
        ? { ...m, keyframes: result.keyframes, lastModified: now }
        : m,
    );

    updateClip(clipId, { masks: updatedMasks });
  };

  const handlePositionChange = (axis: "x" | "y", value: string) => {
    const numValue = Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;

    const currentValue = axis === "x" ? bounds.centerX : bounds.centerY;
    const delta = numValue - currentValue;

    // Translate all points by the delta
    const newPoints = [...lassoPoints];
    const offset = axis === "x" ? 0 : 1;

    for (let i = offset; i < newPoints.length; i += 2) {
      newPoints[i] += delta;
    }

    updateLassoPoints(newPoints);
  };

  if (!mask || lassoPoints.length === 0) {
    return (
      <div className="p-4 px-5">
        <p className="text-brand-light/50 text-[11px]">
          No lasso mask selected
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-y-2 min-w-0 pb-2">
      <div className="p-4 flex flex-col gap-y-4 px-5 min-w-0">
        <div className="flex flex-row items-center justify-between">
          <h4 className="text-brand-light text-[12px] font-medium text-start">
            Transform
          </h4>
        </div>

        <div className="flex flex-col gap-y-3">
          {/* Position */}
          <div className="flex flex-row gap-x-2">
            <Input
              label="Position"
              value={bounds.centerX.toFixed(0).toString()}
              onChange={(value) => handlePositionChange("x", value)}
              startLogo="X"
            />
            <Input
              emptyLabel
              value={bounds.centerY.toFixed(0).toString()}
              onChange={(value) => handlePositionChange("y", value)}
              startLogo="Y"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LassoMaskTransformProperties;
