import { useClipStore } from "@/lib/clip";
import { MaskClipProps } from "@/lib/types";
import React, { useCallback } from "react";
import { cn } from "@/lib/utils";
import ColorInput from "../ColorInput";

interface MaskOperationsPropertiesProps {
  mask: MaskClipProps;
  clipId: string;
}

const MaskOperationsProperties: React.FC<MaskOperationsPropertiesProps> = ({
  mask,
  clipId,
}) => {
  const getClipById = useClipStore((s) => s.getClipById);
  const updateClip = useClipStore((s) => s.updateClip);

  const updateMask = useCallback(
    (updates: Partial<MaskClipProps>) => {
      const currentClip = getClipById(clipId);
      if (
        !currentClip ||
        (currentClip.type !== "video" && currentClip.type !== "image")
      )
        return;

      const masks = (currentClip as any).masks || [];
      const updatedMasks = masks.map((m: MaskClipProps) =>
        m.id === mask.id ? { ...m, ...updates } : m,
      );

      updateClip(clipId, { masks: updatedMasks });
    },
    [mask.id, clipId, getClipById, updateClip],
  );

  const handleBinaryMode = () => {
    updateMask({
      maskColor: "#ffffff",
      maskOpacity: 100,
      maskColorEnabled: true,
      backgroundColor: "#000000",
      backgroundOpacity: 100,
      backgroundColorEnabled: true,
    });
  };

  const handleTransparentForeground = () => {
    updateMask({
      maskColorEnabled: false,
      backgroundColorEnabled: true,
    });
  };

  const handleTransparentBackground = () => {
    updateMask({
      backgroundColorEnabled: false,
      maskColorEnabled: true,
    });
  };

  const handleGrayMode = () => {
    updateMask({
      maskColor: "#808080",
      maskOpacity: 100,
      maskColorEnabled: true,
      backgroundColor: "#000000",
      backgroundOpacity: 100,
      backgroundColorEnabled: true,
    });
  };

  const handleInvertedMode = () => {
    updateMask({
      maskColor: "#000000",
      maskOpacity: 100,
      maskColorEnabled: true,
      backgroundColor: "#ffffff",
      backgroundOpacity: 100,
      backgroundColorEnabled: true,
    });
  };

  const handleSemiTransparentMode = () => {
    updateMask({
      maskColor: "#ffffff",
      maskOpacity: 50,
      maskColorEnabled: true,
      backgroundColor: "#000000",
      backgroundOpacity: 100,
      backgroundColorEnabled: true,
    });
  };

  if (!mask) {
    return (
      <div className="p-4 px-5">
        <p className="text-brand-light/50 text-[11px]">No mask selected</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 flex flex-col gap-y-5 px-5 min-w-0">
        <h4 className="text-brand-light text-[12px] font-medium text-start">
          Rendering
        </h4>
        <div className="flex flex-col gap-y-2">
          <span className="text-brand-light text-[11px] font-medium text-start">
            Display{" "}
          </span>
          <p className="text-brand-light/40 text-[10px] text-start">
            Select how your masked region should be displayed.
          </p>
          <div className="flex flex-row flex-wrap gap-x-2 gap-y-2 mb-1.5">
            <button onClick={handleBinaryMode}>
              <span className="text-brand-light text-[10px] font-medium px-3 py-1 rounded-md border border-brand-light/10 bg-brand hover:bg-brand-light/10 transition-all duration-200">
                Binary
              </span>
            </button>
            <button onClick={handleTransparentForeground}>
              <span className="text-brand-light text-[10px] font-medium px-3 py-1 rounded-md border border-brand-light/10 bg-brand hover:bg-brand-light/10 transition-all duration-200">
                Transparent Foreground
              </span>
            </button>
            <button onClick={handleTransparentBackground}>
              <span className="text-brand-light text-[10px] font-medium px-3 py-1 rounded-md border border-brand-light/10 bg-brand hover:bg-brand-light/10 transition-all duration-200">
                Transparent Background
              </span>
            </button>
            <button onClick={handleGrayMode}>
              <span className="text-brand-light text-[10px] font-medium px-3 py-1 rounded-md border border-brand-light/10 bg-brand hover:bg-brand-light/10 transition-all duration-200">
                Gray
              </span>
            </button>
            <button onClick={handleInvertedMode}>
              <span className="text-brand-light text-[10px] font-medium px-3 py-1 rounded-md border border-brand-light/10 bg-brand hover:bg-brand-light/10 transition-all duration-200">
                Binary Inverted
              </span>
            </button>
            <button onClick={handleSemiTransparentMode}>
              <span className="text-brand-light text-[10px] font-medium px-3 py-1 rounded-md border border-brand-light/10 bg-brand hover:bg-brand-light/10 transition-all duration-200">
                Semi-Transparent
              </span>
            </button>
          </div>

          <div className="flex flex-col gap-y-3">
            {/* Mask Color */}
            <div className="flex flex-col gap-y-2">
              <div className="flex flex-row items-center justify-between">
                <label className="text-brand-light text-[11px] font-medium">
                  Masked Region
                </label>
              </div>
              <div className="flex flex-row items-center justify-between gap-x-3">
                {(mask.maskColorEnabled ?? true) && (
                  <ColorInput
                    labelClass="text-brand-light text-[11px] font-medium"
                    size="medium"
                    percentValue={mask.maskOpacity ?? 100}
                    setPercentValue={(value) =>
                      updateMask({ maskOpacity: value })
                    }
                    value={mask.maskColor ?? "#000000"}
                    onChange={(value) => updateMask({ maskColor: value })}
                  />
                )}
                <button
                  onClick={() =>
                    updateMask({
                      maskColorEnabled: !(mask.maskColorEnabled ?? true),
                    })
                  }
                  className={cn(
                    "px-6 py-1.5 rounded-full text-[9px] font-medium transition-all duration-200 w-fit",
                    (mask.maskColorEnabled ?? true)
                      ? "bg-brand-light/10 text-brand-light/80 border border-brand-light/20 hover:bg-brand-light/20"
                      : "bg-brand border border-brand-light/10 text-brand-light/70 hover:bg-brand-light/10",
                  )}
                >
                  {(mask.maskColorEnabled ?? true) ? "Disable" : "Enable"}
                </button>
              </div>
            </div>

            {/* Background Color */}
            <div className="flex flex-col gap-y-2">
              <div className="flex flex-row items-center justify-between">
                <label className="text-brand-light text-[11px] font-medium">
                  Background
                </label>
              </div>
              <div className="flex flex-row items-center justify-between gap-x-3">
                {(mask.backgroundColorEnabled ?? true) && (
                  <ColorInput
                    labelClass="text-brand-light text-[11px] font-medium"
                    size="medium"
                    percentValue={mask.backgroundOpacity ?? 100}
                    setPercentValue={(value) =>
                      updateMask({ backgroundOpacity: value })
                    }
                    value={mask.backgroundColor ?? "#ffffff"}
                    onChange={(value) => updateMask({ backgroundColor: value })}
                  />
                )}
                <button
                  onClick={() =>
                    updateMask({
                      backgroundColorEnabled: !(
                        mask.backgroundColorEnabled ?? true
                      ),
                    })
                  }
                  className={cn(
                    "px-6 py-1.5 rounded-full text-[9px] font-medium transition-all duration-200 w-fit",
                    (mask.backgroundColorEnabled ?? true)
                      ? "bg-brand-light/10 text-brand-light/80 border border-brand-light/20 hover:bg-brand-light/20"
                      : "bg-brand border border-brand-light/10 text-brand-light/70 hover:bg-brand-light/10",
                  )}
                >
                  {(mask.backgroundColorEnabled ?? true) ? "Disable" : "Enable"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MaskOperationsProperties;
