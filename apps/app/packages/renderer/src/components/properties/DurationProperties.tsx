import { useClipStore } from "@/lib/clip";
import { AudioClipProps, VideoClipProps, ModelClipProps } from "@/lib/types";
import { useMemo } from "react";
import { useState } from "react";
import React from "react";
import { IoRefreshOutline } from "react-icons/io5";
import PropertiesSlider from "./PropertiesSlider";
import Input from "./Input";
import { getMediaInfoCached } from "@/lib/media/utils";
import { useControlsStore } from "@/lib/control";

interface DurationPropertiesProps {
  clipId: string;
}

const DurationProperties: React.FC<DurationPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | VideoClipProps
    | AudioClipProps
    | ModelClipProps;
    
  const { updateClip } = useClipStore();
  const speed = useMemo(() => clip?.speed ?? 1.0, [clip?.speed]);
  const [spinning, setSpinning] = useState(false);
  const startFrame = useMemo(() => clip?.startFrame ?? 0, [clip?.startFrame]);
  const endFrame = useMemo(() => clip?.endFrame ?? 0, [clip?.endFrame]);
  const { fps } = useControlsStore();
  const resizeClip = useClipStore((s) => s.resizeClip);
  const isValidResize = useClipStore((s) => s.isValidResize);
  const isPlaying = useControlsStore((s) => s.isPlaying);
  const pause = useControlsStore((s) => s.pause);

  const hasSpeed = useMemo(() => {
    return clip?.type === "video" || clip?.type === "audio";
  }, [clip?.type]);

  const setSpeed = (value: number) => {
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0.1, Math.min(5, numValue));
    if (isPlaying) {
      // pause the playback
      pause();
    }
    updateClip(clipId, { speed: clamped });
  };

  const setStartFrame = (value: number) => {
    if (isNaN(value) || !isFinite(value)) return;
    if (!isValidResize(clipId, "left", value)) return;
    resizeClip(clipId, "left", value);
  };

  const setEndFrame = (value: number) => {
    if (isNaN(value) || !isFinite(value)) return;
    if (!isValidResize(clipId, "right", value)) return;
    resizeClip(clipId, "right", value);
  };

  // Model max duration (in frames) if available
  const modelMaxFrames = useMemo(() => {
    if (clip?.type !== "model") return Infinity;
    const secs = Number(
      (clip as ModelClipProps)?.manifest?.spec?.max_duration_secs,
    );
    if (!Number.isFinite(secs) || secs <= 0) return Infinity;
    return Math.max(1, Math.floor(secs * fps));
  }, [clip?.type, (clip as any)?.manifest, fps]);

  const startFrameMin = Math.max(
    0,
    startFrame - Math.abs(clip?.trimStart ?? 0),
    Number.isFinite(modelMaxFrames)
      ? Math.max(0, endFrame - (modelMaxFrames as number))
      : 0,
  );
  const startFrameMax = endFrame - 1;
  const endFrameMin = startFrame + 1;
  const rawTotalFrames = Math.max(
    0,
    Math.floor((getMediaInfoCached(clip?.assetId as string)?.duration ?? 0) * fps),
  );
  const effectiveSpeed = Math.max(0.1, Math.min(5, Number(clip?.speed ?? 1)));
  const maxByMedia = Math.max(0, Math.floor(rawTotalFrames / effectiveSpeed));
  const modelEndLimit =
    clip?.type === "model" && Number.isFinite(modelMaxFrames)
      ? startFrame + (modelMaxFrames as number)
      : Infinity;
  const endFrameMax = Math.min(
    maxByMedia,
    endFrame + Math.abs(clip?.trimEnd ?? 0),
    modelEndLimit,
  );

  const canRefresh = useMemo(() => {
    return isFinite(clip?.trimEnd ?? 0) && isFinite(clip?.trimStart ?? 0);
  }, [clip?.type]);

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 px-5 min-w-0">
        <div className="flex flex-row items-center justify-between mb-4">
          <h4 className="text-brand-light text-[12px] font-medium text-start">
            Duration
          </h4>
          {canRefresh && (
            <span
              onClick={() => {
                const startFrame =
                  (clip?.startFrame ?? 0) - (clip?.trimStart ?? 0);
                const endFrame = (clip?.endFrame ?? 0) - (clip?.trimEnd ?? 0);
                resizeClip(clipId, "left", startFrame);
                resizeClip(clipId, "right", endFrame);
                setTimeout(() => {
                  updateClip(clipId, { speed: 1 });
                }, 10);
                updateClip(clipId, { speed: 1 });
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
          )}
        </div>
        <div className="flex flex-col gap-y-2">
          {hasSpeed && (
            <PropertiesSlider
              label="Speed"
              value={Math.round(speed * 10) / 10}
              onChange={setSpeed}
              suffix="x"
              min={0.1}
              max={5}
              step={0.1}
            />
          )}
          <div className="flex flex-row gap-x-2">
            <Input
              label="Start Frame"
              value={startFrame.toString()}
              onChange={(value) => setStartFrame(Number(value))}
              startLogo="F"
              canStep
              step={1}
              min={startFrameMin}
              max={startFrameMax}
            />
            <Input
              label="End Frame"
              value={endFrame.toString()}
              onChange={(value) => setEndFrame(Number(value))}
              startLogo="F"
              canStep
              step={1}
              min={endFrameMin}
              max={endFrameMax}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default DurationProperties;
