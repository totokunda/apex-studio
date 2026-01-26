import React, { useMemo, useState } from "react";
import PropertiesSlider from "./PropertiesSlider";
import { useClipStore } from "@/lib/clip";
import { AudioClipProps } from "@/lib/types";
import { getMediaInfoCached } from "@/lib/media/utils";
import { IoRefreshOutline } from "react-icons/io5";

interface AudioPropertiesProps {
  clipId: string;
}

const AudioProperties: React.FC<AudioPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as AudioClipProps;
  const { updateClip } = useClipStore();
  const volume = useMemo(() => clip?.volume ?? 0, [clip?.volume]);
  const fadeIn = useMemo(() => clip?.fadeIn ?? 0, [clip?.fadeIn]);
  const fadeOut = useMemo(() => clip?.fadeOut ?? 0, [clip?.fadeOut]);
  const [spinning, setSpinning] = useState(false);

  const setVolume = (value: number) => {
    updateClip(clipId, { volume: value });
  };
  const setFadeIn = (value: number) => {
    updateClip(clipId, { fadeIn: value });
  };
  const setFadeOut = (value: number) => {
    updateClip(clipId, { fadeOut: value });
  };

  const clipDuration = useMemo(() => {
    if (!clip?.assetId) return 0;
    const duration = getMediaInfoCached(clip?.assetId)?.duration;
    return duration ?? 0;
  }, [clip?.assetId]);

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 px-5 min-w-0">
        <div className="flex flex-row items-center justify-between mb-4">
          <h4 className="text-brand-light text-[12px] font-medium text-start ">
            Audio
          </h4>
          <span
            onClick={() => {
              updateClip(clipId, { volume: 0, fadeIn: 0, fadeOut: 0 });
              setSpinning(true);
              setTimeout(() => {
                setSpinning(false);
              }, 500);
            }}
            className="text-brand-light text-sm cursor-pointer"
          >
            <IoRefreshOutline
              className={spinning ? "animate-spin duration-500" : ""}
              onAnimationEnd={() => setSpinning(false)}
            />
          </span>
        </div>
        <div className="flex flex-col gap-y-2">
          <PropertiesSlider
            label="Volume"
            value={volume}
            onChange={setVolume}
            suffix="dB"
            min={-60}
            max={20}
            step={1}
          />
          <PropertiesSlider
            label="Fade In"
            value={fadeIn}
            onChange={setFadeIn}
            suffix="s"
            max={clipDuration}
            min={0}
            step={0.1}
          />
          <PropertiesSlider
            label="Fade Out"
            value={fadeOut}
            onChange={setFadeOut}
            suffix="s"
            max={clipDuration}
            min={0}
            step={0.1}
          />
        </div>
      </div>
    </div>
  );
};

export default AudioProperties;
