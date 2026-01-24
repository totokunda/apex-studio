import React from "react";
import { TextClipProps } from "@/lib/types";
import { useClipStore } from "@/lib/clip";
import { Checkbox } from "@/components/ui/checkbox";
import ColorInput from "./ColorInput";
import PropertiesSlider from "./PropertiesSlider";
import { IoLockClosed, IoLockOpen } from "react-icons/io5";
import { cn } from "@/lib/utils";
import Input from "./Input";

interface TextShadowSectionProps {
  clipId: string;
}

const TextShadowSection: React.FC<TextShadowSectionProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as TextClipProps;
  const updateClip = useClipStore((s) => s.updateClip);

  const shadowEnabled = clip?.shadowEnabled ?? false;
  const shadowColor = clip?.shadowColor ?? "#000000";
  const shadowOpacity = clip?.shadowOpacity ?? 75;
  const shadowBlur = clip?.shadowBlur ?? 4;
  const shadowOffsetX = clip?.shadowOffsetX ?? 2;
  const shadowOffsetY = clip?.shadowOffsetY ?? 2;
  const shadowOffsetLocked = clip?.shadowOffsetLocked ?? true;

  const handleOffsetXChange = (value: number) => {
    if (shadowOffsetLocked) {
      updateClip(clipId, { shadowOffsetX: value, shadowOffsetY: value });
    } else {
      updateClip(clipId, { shadowOffsetX: value });
    }
  };

  const handleOffsetYChange = (value: number) => {
    if (shadowOffsetLocked) {
      updateClip(clipId, { shadowOffsetX: value, shadowOffsetY: value });
    } else {
      updateClip(clipId, { shadowOffsetY: value });
    }
  };

  return (
    <div className="flex flex-col gap-y-4 p-4 px-5 border-t border-brand-light/5">
      <div className="flex flex-row items-center gap-x-3">
        <Checkbox
          checked={shadowEnabled}
          onCheckedChange={(checked) =>
            updateClip(clipId, { shadowEnabled: !!checked })
          }
        />
        <h3 className="text-brand-light text-xs font-medium">Shadow</h3>
      </div>

      {shadowEnabled && (
        <div className="flex flex-col gap-y-4 pl-7">
          <ColorInput
            percentValue={shadowOpacity}
            value={shadowColor}
            setPercentValue={(value) =>
              updateClip(clipId, { shadowOpacity: value })
            }
            onChange={(value) => updateClip(clipId, { shadowColor: value })}
            label="Color"
          />
          <PropertiesSlider
            label="Blur"
            value={shadowBlur}
            onChange={(value) => updateClip(clipId, { shadowBlur: value })}
            suffix="px"
            min={0}
            max={50}
            step={1}
            toFixed={0}
          />
          <div className="flex flex-col gap-y-2 ">
            <div className="flex flex-row items-center gap-x-2 w-full -mb-1.5">
              <h4 className="text-brand-light text-[10.5px] font-medium flex-1 text-start ">
                Offset
              </h4>
              <button
                onClick={() =>
                  updateClip(clipId, {
                    shadowOffsetLocked: !shadowOffsetLocked,
                  })
                }
                className={cn(
                  "p-1 rounded transition-colors",
                  "text-brand-light/40 hover:text-brand-light/60",
                )}
              >
                {shadowOffsetLocked ? (
                  <IoLockClosed className="h-3.5 w-3.5" />
                ) : (
                  <IoLockOpen className="h-3.5 w-3.5" />
                )}
              </button>
            </div>
            <div className="flex flex-row gap-x-2">
              <Input
                value={shadowOffsetX.toFixed(2).toString() ?? "1.00"}
                onChange={(value) => handleOffsetXChange(Number(value))}
                startLogo="X"
                canStep
                step={0.1}
                min={0.1}
              />
              <Input
                value={shadowOffsetY.toFixed(2).toString() ?? "1.00"}
                onChange={(value) => handleOffsetYChange(Number(value))}
                startLogo="Y"
                canStep
                step={0.1}
                min={0.1}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TextShadowSection;
