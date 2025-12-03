import React, { useMemo } from "react";
import { useClipStore } from "@/lib/clip";
import { FilterClipProps } from "@/lib/types";
import { useControlsStore } from "@/lib/control";
import PropertiesSlider from "./PropertiesSlider";

interface FilterPropertiesProps {
  clipId: string;
}

const FilterProperties: React.FC<FilterPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | FilterClipProps
    | undefined;
  const { updateClip } = useClipStore();
  const { isPlaying, pause } = useControlsStore();

  const intensity = useMemo(() => {
    const raw = clip?.intensity ?? 100;
    if (!Number.isFinite(raw)) return 100;
    return Math.max(0, Math.min(100, Number(raw)));
  }, [clip?.intensity]);

  const setIntensity = (value: number) => {
    if (isPlaying) pause();
    const numValue = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { intensity: clamped });
  };

  return (
    <div className="flex flex-col gap-y-2 min-w-0 p-4 px-5 border-b border-brand-light/5">
      <h4 className="text-brand-light text-[12px] font-medium text-start mb-4">
        Filter
      </h4>
      <PropertiesSlider
        label="Intensity"
        value={Math.round(intensity)}
        onChange={setIntensity}
        min={0}
        max={100}
        step={1}
        suffix="%"
        toFixed={0}
      />
    </div>
  );
};

export default FilterProperties;
