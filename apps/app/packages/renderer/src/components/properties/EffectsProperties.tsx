import { useClipStore } from "@/lib/clip";
import { VideoClipProps, ImageClipProps } from "@/lib/types";
import { useMemo } from "react";
import React from "react";
import { IoRefreshOutline } from "react-icons/io5";
import PropertiesSlider from "./PropertiesSlider";
import { useControlsStore } from "@/lib/control";

interface EffectsPropertiesProps {
  clipId: string;
}

const EffectsProperties: React.FC<EffectsPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | VideoClipProps
    | ImageClipProps;
  const { updateClip } = useClipStore();
  const { pause, isPlaying } = useControlsStore();
  const pausePlayback = () => {
    if (isPlaying) {
      pause();
    }
  };

  const sharpness = useMemo(() => clip?.sharpness ?? 0, [clip?.sharpness]);
  const noise = useMemo(() => clip?.noise ?? 0, [clip?.noise]);
  const blur = useMemo(() => clip?.blur ?? 0, [clip?.blur]);
  const vignette = useMemo(() => clip?.vignette ?? 0, [clip?.vignette]);

  const setSharpness = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { sharpness: clamped });
  };

  const setNoise = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { noise: clamped });
  };

  const setBlur = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { blur: clamped });
  };

  const setVignette = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { vignette: clamped });
  };

  const resetAll = () => {
    pausePlayback();
    updateClip(clipId, {
      sharpness: 0,
      noise: 0,
      blur: 0,
      vignette: 0,
    });
  };

  return (
    <div className="p-4 px-5 min-w-0 pb-6">
      <div className="flex flex-row items-center justify-between mb-4">
        <h4 className="text-brand-light text-[12px] font-medium text-start">
          Effects
        </h4>
        <span
          onClick={resetAll}
          className="text-brand-light text-sm cursor-pointer"
        >
          <IoRefreshOutline />
        </span>
      </div>
      <div className="flex flex-col gap-y-2">
        <PropertiesSlider
          label="Sharpness"
          value={sharpness}
          onChange={setSharpness}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Noise"
          value={noise}
          onChange={setNoise}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Blur"
          value={blur}
          onChange={setBlur}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Vignette"
          value={vignette}
          onChange={setVignette}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
      </div>
    </div>
  );
};

export default EffectsProperties;
