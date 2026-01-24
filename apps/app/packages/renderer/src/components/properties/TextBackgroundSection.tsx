import React from "react";
import { TextClipProps } from "@/lib/types";
import { useClipStore } from "@/lib/clip";
import { Checkbox } from "@/components/ui/checkbox";
import ColorInput from "./ColorInput";
import PropertiesSlider from "./PropertiesSlider";

interface TextBackgroundSectionProps {
  clipId: string;
}

const TextBackgroundSection: React.FC<TextBackgroundSectionProps> = ({
  clipId,
}) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as TextClipProps;
  const updateClip = useClipStore((s) => s.updateClip);

  const backgroundEnabled = clip?.backgroundEnabled ?? false;
  const backgroundColor = clip?.backgroundColor ?? "#000000";
  const backgroundOpacity = clip?.backgroundOpacity ?? 100;
  const backgroundCornerRadius = clip?.backgroundCornerRadius ?? 0;

  return (
    <div className="flex flex-col gap-y-4 p-4 px-5 border-t border-brand-light/5">
      <div className="flex flex-row items-center gap-x-3">
        <Checkbox
          checked={backgroundEnabled}
          onCheckedChange={(checked) =>
            updateClip(clipId, { backgroundEnabled: !!checked })
          }
        />
        <h3 className="text-brand-light text-[12px] font-medium">Background</h3>
      </div>

      {backgroundEnabled && (
        <div className="flex flex-col gap-y-4 pl-7 pb-3">
          <ColorInput
            percentValue={backgroundOpacity}
            value={backgroundColor}
            setPercentValue={(value) =>
              updateClip(clipId, { backgroundOpacity: value })
            }
            onChange={(value) => updateClip(clipId, { backgroundColor: value })}
            label="Color"
          />
          <PropertiesSlider
            label="Corner Radius"
            value={backgroundCornerRadius}
            onChange={(value) =>
              updateClip(clipId, { backgroundCornerRadius: value })
            }
            suffix="px"
            min={0}
            max={50}
            step={1}
            toFixed={0}
          />
        </div>
      )}
    </div>
  );
};

export default TextBackgroundSection;
