import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { MaskClipProps, PreprocessorClipType } from "@/lib/types";
import React, { useMemo } from "react";
import ShapeMaskTransformProperties from "./ShapeMaskTransformProperties";
import LassoMaskTransformProperties from "./LassoMaskTransformProperties";
import MaskTrackingProperties from "./MaskTrackingProperties";
import MaskOperationsProperties from "./MaskOperationsProperties";
import { cn } from "@/lib/utils";
import { LuX } from "react-icons/lu";

interface MaskPropertiesPanelProps {
  clipId: string;
}

const MaskPropertiesPanel: React.FC<MaskPropertiesPanelProps> = ({
  clipId,
}) => {
  const selectedMaskId = useControlsStore((s) => s.selectedMaskId);
  const setSelectedMaskId = useControlsStore((s) => s.setSelectedMaskId);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | PreprocessorClipType
    | undefined;
  const updateClip = useClipStore((s) => s.updateClip);
  const clipWithinFrame = useClipStore((s) => s.clipWithinFrame);

  const activeMasks = useMemo(() => {
    if (!clip || (clip.type !== "video" && clip.type !== "image")) return [];
    // Only show masks when the clip is active at the current frame
    const isActive = clipWithinFrame(clip, focusFrame);
    if (!isActive) return [];
    const masks = clip.masks || [];
    return masks.filter((m) => {
      const keyframes =
        m.keyframes instanceof Map
          ? m.keyframes
          : (m.keyframes as Record<number, any>);
      const keyframeNumbers =
        keyframes instanceof Map
          ? Array.from(keyframes.keys())
              .map(Number)
              .sort((a, b) => a - b)
          : Object.keys(keyframes)
              .map(Number)
              .sort((a, b) => a - b);
      if (keyframeNumbers.length === 0) return false;

      // Compute local frame relative to clip start (aligns with how mask keyframes are stored)
      const startFrame = clip.startFrame ?? 0;
      const trimStart = isFinite(clip.trimStart ?? 0)
        ? (clip.trimStart ?? 0)
        : 0;
      const realStartFrame = startFrame + trimStart;
      const localFrame = focusFrame - realStartFrame;

      const nearestKeyframe = (frame: number) => {
        if (frame < keyframeNumbers[0]) return keyframeNumbers[0];
        const atOrBefore = keyframeNumbers.filter((k) => k <= frame).pop();
        return atOrBefore ?? keyframeNumbers[keyframeNumbers.length - 1];
      };

      const candidateLocal = nearestKeyframe(localFrame);
      const candidateGlobal = nearestKeyframe(focusFrame);
      const activeKeyframe = candidateLocal ?? candidateGlobal;

      if (activeKeyframe === undefined) return false;
      const maskData =
        keyframes instanceof Map
          ? keyframes.get(activeKeyframe)
          : keyframes[activeKeyframe];
      return !!maskData;
    });
  }, [clip, focusFrame, clipWithinFrame]);

  const handleRemoveMask = (maskId: string) => {
    if (!clip) return;
    const masks = clip.masks || [];
    const updatedMasks = masks.filter((m: MaskClipProps) => m.id !== maskId);
    updateClip(clipId, { masks: updatedMasks });
  };

  const mask = useMemo(() => {
    if (!clip || (clip.type !== "video" && clip.type !== "image")) return null;
    const masks = clip.masks || [];
    let mask = masks.find(
      (m: MaskClipProps) => m.id === selectedMaskId,
    ) as MaskClipProps | null;
    if (!mask) {
      mask = masks[0] as MaskClipProps | null;
      setSelectedMaskId(mask?.id || null);
    }
    return mask;
  }, [clip, selectedMaskId]);

  if (!mask || clip?.masks.length === 0 || activeMasks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-9 gap-y-1">
        <p className="text-brand-light/90 font-medium text-[13px] text-center">
          No masks currently active in this clip.
        </p>
        <p className="text-brand-light/60 text-[12px] text-center">
          Use the mask tool to create one.
        </p>
      </div>
    );
  }

  const isShapeMask = mask?.tool === "shape";
  const isLassoMask = mask?.tool === "lasso";
  const showTransform = isShapeMask || isLassoMask;

  return (
    <div className="flex flex-col gap-y-2">
      {activeMasks.length > 0 && (
        <div className="flex flex-row gap-x-2 items-center justify-start px-4 pt-4 flex-wrap gap-y-2">
          {activeMasks.map((m: MaskClipProps, idx: number) => (
            <div
              key={m.id}
              onClick={() => setSelectedMaskId(m.id)}
              className={cn(
                "flex flex-row group relative items-center justify-between px-3 py-1 cursor-pointer rounded-full w-fit border border-brand-light/10 bg-brand",
                m.id === selectedMaskId &&
                  "bg-brand-light/10 border-brand-light/20",
              )}
            >
              <p className="text-brand-light text-[11px] font-medium">
                {m.tool.charAt(0).toUpperCase() + m.tool.slice(1)} Mask{" "}
                {idx + 1}
              </p>
              <div
                onClick={() => handleRemoveMask(m.id)}
                className="text-brand-light/50 hover:text-red-400 transition-colors absolute -right-1 -top-1 group-hover:block hidden p-0.5 border bg-brand border-brand-light/10 rounded-full cursor-pointer"
              >
                <LuX className="w-2.5 h-2.5" />
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="flex flex-col divide-y divide-brand-light/10 min-w-0">
        {/* Transform Properties (conditional based on mask type) */}
        {showTransform && mask && (
          <>
            {isShapeMask && (
              <ShapeMaskTransformProperties mask={mask} clipId={clipId} />
            )}
            {isLassoMask && (
              <LassoMaskTransformProperties mask={mask} clipId={clipId} />
            )}
          </>
        )}

        {clip?.type === "video" && mask && (
          <MaskTrackingProperties mask={mask} clipId={clipId} />
        )}

        {/* Operations Properties */}
        {mask && <MaskOperationsProperties mask={mask} clipId={clipId} />}
      </div>
    </div>
  );
};

export default MaskPropertiesPanel;
