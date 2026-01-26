import React from "react";
import { TextClipProps } from "@/lib/types";
import { useClipStore } from "@/lib/clip";
import { Checkbox } from "@/components/ui/checkbox";
import ColorInput from "./ColorInput";
import PropertiesSlider from "./PropertiesSlider";

interface TextStrokeSectionProps {
  clipId: string;
}

const TextStrokeSection: React.FC<TextStrokeSectionProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as TextClipProps;
  const updateClip = useClipStore((s) => s.updateClip);

  const strokeEnabled = clip?.strokeEnabled ?? false;
  const stroke = clip?.stroke ?? "#000000";
  const strokeWidth = clip?.strokeWidth ?? 2;
  const strokeOpacity = clip?.strokeOpacity ?? 100;

  return (
    <div className="flex flex-col gap-y-4 p-4 px-5 border-t border-brand-light/5">
      <div className="flex flex-row items-center gap-x-3">
        <Checkbox
          checked={strokeEnabled}
          onCheckedChange={(checked) =>
            updateClip(clipId, { strokeEnabled: !!checked })
          }
        />
        <h3 className="text-brand-light text-[12px] font-medium">Stroke</h3>
      </div>

      {strokeEnabled && (
        <div className="flex flex-col gap-y-4 pl-7">
          <ColorInput
            percentValue={strokeOpacity}
            value={stroke}
            setPercentValue={(value) =>
              updateClip(clipId, { strokeOpacity: value })
            }
            onChange={(value) => updateClip(clipId, { stroke: value })}
            label="Color"
          />
          <PropertiesSlider
            label="Width"
            value={strokeWidth}
            onChange={(value) => updateClip(clipId, { strokeWidth: value })}
            suffix="px"
            min={1}
            max={20}
            step={0.5}
            toFixed={1}
          />
        </div>
      )}
    </div>
  );
};

export default TextStrokeSection;
