import { useClipStore } from "@/lib/clip";
import { VideoClipProps, ImageClipProps } from "@/lib/types";
import { useMemo } from "react";
import React from "react";
import { IoRefreshOutline } from "react-icons/io5";
import PropertiesSlider from "./PropertiesSlider";
import ColorInput from "./ColorInput";
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
  const colorTintColor = useMemo(
    () => clip?.colorTintColor ?? "#00ff4c",
    [clip?.colorTintColor],
  );
  const colorTintIntensity = useMemo(
    () => clip?.colorTintIntensity ?? 0,
    [clip?.colorTintIntensity],
  );
  const scanLines = useMemo(() => clip?.scanLines ?? 0, [clip?.scanLines]);
  const chromaticAberration = useMemo(
    () => clip?.chromaticAberration ?? 0,
    [clip?.chromaticAberration],
  );
  const interlace = useMemo(() => clip?.interlace ?? 0, [clip?.interlace]);
  const pixelate = useMemo(() => clip?.pixelate ?? 0, [clip?.pixelate]);
  const jitter = useMemo(() => clip?.jitter ?? 0, [clip?.jitter]);

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

  const setColorTintColor = (value: string) => {
    pausePlayback();
    updateClip(clipId, { colorTintColor: value });
  };

  const setColorTintIntensity = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { colorTintIntensity: clamped });
  };

  const setScanLines = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { scanLines: clamped });
  };

  const setChromaticAberration = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { chromaticAberration: clamped });
  };

  const setInterlace = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { interlace: clamped });
  };

  const setPixelate = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { pixelate: clamped });
  };

  const setJitter = (value: number) => {
    pausePlayback();
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, { jitter: clamped });
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

  const resetStylize = () => {
    pausePlayback();
    updateClip(clipId, {
      colorTintColor: "#00ff4c",
      colorTintIntensity: 0,
      scanLines: 0,
      chromaticAberration: 0,
      interlace: 0,
      pixelate: 0,
      jitter: 0,
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
          tooltip="Enhances edge detail using an unsharp mask. Higher values make fine details and edges more defined."
          value={sharpness}
          onChange={setSharpness}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Noise"
          tooltip="Adds random film grain across the image. Useful for creating a raw, textured look or matching footage shot on analog media."
          value={noise}
          onChange={setNoise}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Blur"
          tooltip="Applies a Gaussian blur that softens the image. Useful for defocusing backgrounds or simulating a low-quality lens."
          value={blur}
          onChange={setBlur}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Vignette"
          tooltip="Darkens the edges of the frame toward the corners. Mimics the natural falloff of older camera lenses and draws focus to the center."
          value={vignette}
          onChange={setVignette}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
      </div>

      {/* Stylize / Found Footage Effects */}
      <div className="flex flex-row items-center justify-between mb-4 mt-6">
        <h4 className="text-brand-light text-[12px] font-medium text-start">
          Stylize
        </h4>
        <span
          onClick={resetStylize}
          className="text-brand-light text-sm cursor-pointer"
        >
          <IoRefreshOutline />
        </span>
      </div>
      <div className="flex flex-col gap-y-2">
        <ColorInput
          label="Color Tint"
          tooltip="Maps the image luminance to a single color. Pick a color and set the intensity percentage. Use green for night vision, blue for security cam, or warm tones for vintage looks."
          value={colorTintColor}
          onChange={setColorTintColor}
          percentValue={colorTintIntensity}
          setPercentValue={setColorTintIntensity}
          size="small"
        />
        <PropertiesSlider
          label="Scan Lines"
          tooltip="Adds horizontal scan lines to simulate CRT monitors, VHS tapes, or old camcorder displays. Creates alternating bright and dark bands across the frame."
          labelClass="mt-2"
          value={scanLines}
          onChange={setScanLines}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Chromatic Aberration"
          tooltip="Separates the red, green, and blue color channels outward from the center. Simulates the color fringing caused by cheap or damaged camera lenses."
          value={chromaticAberration}
          onChange={setChromaticAberration}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Interlace"
          tooltip="Simulates interlaced video artifacts from old camcorders and CRT displays. Adds alternating field dimming, horizontal combing, and subtle line blending."
          value={interlace}
          onChange={setInterlace}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Pixelate"
          tooltip="Reduces apparent resolution by grouping pixels into larger blocks. Simulates the look of low-resolution cameras or retro digital footage."
          value={pixelate}
          onChange={setPixelate}
          min={0}
          max={100}
          step={1}
          toFixed={0}
        />
        <PropertiesSlider
          label="Jitter"
          tooltip="Applies random per-frame position shifts to simulate camera shake or handheld instability. Creates an organic, unsteady found-footage feel."
          value={jitter}
          onChange={setJitter}
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
