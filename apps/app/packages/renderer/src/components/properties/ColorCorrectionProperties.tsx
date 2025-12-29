import { useClipStore } from "@/lib/clip";
import { VideoClipProps, ImageClipProps } from "@/lib/types";
import { useMemo } from "react";
import React from "react";
import { IoRefreshOutline } from "react-icons/io5";
import PropertiesSlider from "./PropertiesSlider";
import { useControlsStore } from "@/lib/control";

interface ColorCorrectionPropertiesProps {
  clipId: string;
}

const ColorCorrectionProperties: React.FC<ColorCorrectionPropertiesProps> = ({
  clipId,
}) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | VideoClipProps
    | ImageClipProps;
  const { updateClip } = useClipStore();
  const { pause, isPlaying } = useControlsStore();

  const brightness = useMemo(() => clip?.brightness ?? 0, [clip?.brightness]);
  const contrast = useMemo(() => clip?.contrast ?? 0, [clip?.contrast]);
  const hue = useMemo(() => clip?.hue ?? 0, [clip?.hue]);
  const saturation = useMemo(() => clip?.saturation ?? 0, [clip?.saturation]);

  const pausePlayback = () => {
    if (isPlaying) {
      pause();
    }
  };

  const setBrightness = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(-100, Math.min(100, numValue));
    updateClip(clipId, { brightness: clamped });
  };

  const setContrast = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(-100, Math.min(100, numValue));
    updateClip(clipId, { contrast: clamped });
  };

  const setHue = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(-100, Math.min(100, numValue));
    updateClip(clipId, { hue: clamped });
  };

  const setSaturation = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(-100, Math.min(100, numValue));
    updateClip(clipId, { saturation: clamped });
  };

  const resetAll = () => {
    pausePlayback();
    updateClip(clipId, {
      brightness: 0,
      contrast: 0,
      hue: 0,
      saturation: 0,
    });
  };

  return (
    <div className="p-4 px-5 min-w-0">
      <div className="flex flex-row items-center justify-between mb-4">
        <h4 className="text-brand-light text-[12px] font-medium text-start">
          Color Correction
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
          label="Brightness"
          value={brightness}
          onChange={setBrightness}
          min={-100}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Contrast"
          value={contrast}
          onChange={setContrast}
          min={-100}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Hue"
          value={hue}
          onChange={setHue}
          min={-100}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Saturation"
          value={saturation}
          onChange={setSaturation}
          min={-100}
          max={100}
          step={1}
          toFixed={0}
        />
      </div>
    </div>
  );
};

export default ColorCorrectionProperties;
