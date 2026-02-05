import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { MaskClipProps, MaskShapeTool } from "@/lib/types";
import { useState, useMemo } from "react";
import React from "react";
import { IoRefreshOutline } from "react-icons/io5";
import Input from "../Input";
import { upsertMaskKeyframe } from "@/lib/mask/keyframeUtils";

interface ShapeMaskTransformPropertiesProps {
  mask: MaskClipProps;
  clipId: string;
}

const ShapeMaskTransformProperties: React.FC<
  ShapeMaskTransformPropertiesProps
> = ({ mask, clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId));
  const updateClip = useClipStore((s) => s.updateClip);
  const clipWithinFrame = useClipStore((s) => s.clipWithinFrame);
  const focusFrame = useControlsStore((s) => s.focusFrame);

  const [spinning, setSpinning] = useState(false);

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

  const shapeBounds = useMemo(() => {
    return (
      activeKeyframeData?.data?.shapeBounds ||
      activeKeyframeData?.data?.rectangleBounds
    );
  }, [activeKeyframeData]);

  const shapeType: MaskShapeTool = shapeBounds?.shapeType || "rectangle";
  const supportsSize = shapeType === "ellipse" || shapeType === "rectangle";
  const supportsSizeHeightWidth =
    shapeType === "polygon" || shapeType === "star";

  const updateShapeBounds = (updates: Partial<typeof shapeBounds>) => {
    if (!mask || !activeKeyframeData || !shapeBounds) return;

    if (!clip) return;

    const result = upsertMaskKeyframe({
      mask,
      clip,
      focusFrame,
      updater: (previous) => {
        const baseBounds = previous.shapeBounds || shapeBounds;
        const mergedBounds = { ...(baseBounds || {}), ...updates };

        return {
          ...previous,
          shapeBounds: mergedBounds,
        };
      },
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

  const handleReset = () => {
    if (!shapeBounds) return;

    updateShapeBounds({
      rotation: 0,
      scaleX: 1,
      scaleY: 1,
    });

    setSpinning(true);
    setTimeout(() => {
      setSpinning(false);
    }, 500);
  };

  if (!mask || !shapeBounds) {
    return (
      <div className="p-4 px-5">
        <p className="text-brand-light/50 text-[11px]">
          No shape mask selected
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 flex flex-col gap-y-4 px-5 min-w-0">
        <div className="flex flex-row items-center justify-between">
          <h4 className="text-brand-light text-[12px] font-medium text-start">
            Transform
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
          {/* Position */}
          <div className="flex flex-row gap-x-2">
            <Input
              label="Position"
              value={shapeBounds.x?.toFixed(0).toString() ?? "0"}
              onChange={(value) => updateShapeBounds({ x: Number(value) })}
              startLogo="X"
            />
            <Input
              emptyLabel
              value={shapeBounds.y?.toFixed(0).toString() ?? "0"}
              onChange={(value) => updateShapeBounds({ y: Number(value) })}
              startLogo="Y"
            />
          </div>

          {/* Rotation */}
          <Input
            label="Rotation"
            value={(shapeBounds.rotation ?? 0).toFixed(0).toString()}
            onChange={(value) => updateShapeBounds({ rotation: Number(value) })}
            startLogo="R"
          />

          {/* Size */}
          {supportsSize && (
            <div className="flex flex-row gap-x-2">
              <Input
                label="Size"
                value={shapeBounds.width?.toFixed(0).toString() ?? "0"}
                onChange={(value) =>
                  updateShapeBounds({ width: Number(value) })
                }
                startLogo="W"
              />
              <Input
                emptyLabel
                value={shapeBounds.height?.toFixed(0).toString() ?? "0"}
                onChange={(value) =>
                  updateShapeBounds({ height: Number(value) })
                }
                startLogo="H"
              />
            </div>
          )}
          {supportsSizeHeightWidth && (
            <div className="flex flex-row gap-x-2">
              <Input
                label="Size"
                value={shapeBounds.width?.toFixed(0).toString() ?? "0"}
                onChange={(value) =>
                  updateShapeBounds({
                    width: Number(value),
                    height: Number(value),
                  })
                }
                startLogo="S"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ShapeMaskTransformProperties;
