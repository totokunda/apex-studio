import { useClipStore } from '@/lib/clip';
import { AudioClipProps, VideoClipProps } from '@/lib/types';
import { useMemo } from 'react';
import { useState } from 'react';
import React from 'react'
import { IoRefreshOutline } from 'react-icons/io5';
import PropertiesSlider from './PropertiesSlider';
import Input from './Input';
import { getMediaInfoCached } from '@/lib/media/utils';
import { useControlsStore } from '@/lib/control';

interface DurationPropertiesProps {
  clipId: string
}

const DurationProperties: React.FC<DurationPropertiesProps> = ({ clipId }) => {
    const clip = useClipStore((s) => s.getClipById(clipId)) as VideoClipProps | AudioClipProps;
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

    const setSpeed = (value: number) => {
        const numValue = typeof value === 'number' ? value : Number(value);
        if (isNaN(numValue) || !isFinite(numValue)) return;
        const clamped = Math.max(0.1, Math.min(5, numValue));
        if (isPlaying) {
            // pause the playback
            pause();
        }
        updateClip(clipId, { speed: clamped });
    }

    const setStartFrame = (value: number) => {
        if (isNaN(value) || !isFinite(value)) return;
        if (!isValidResize(clipId, 'left', value)) return;
        resizeClip(clipId, 'left', value);
    }

    const setEndFrame = (value: number) => {
        if (isNaN(value) || !isFinite(value)) return;
        if (!isValidResize(clipId, 'right', value)) return;
        
        resizeClip(clipId, 'right', value);
    }

    const startFrameMin = Math.max(0, startFrame - Math.abs(clip?.framesToGiveStart ?? 0));
    const startFrameMax = endFrame - 1;
    const endFrameMin = startFrame + 1;
    const rawTotalFrames = Math.max(0, Math.floor(((getMediaInfoCached(clip?.src)?.duration ?? 0) * fps)));
    const effectiveSpeed = Math.max(0.1, Math.min(5, Number(clip?.speed ?? 1)));
    const maxByMedia = Math.max(0, Math.floor(rawTotalFrames / effectiveSpeed));
    const endFrameMax = Math.min(maxByMedia, endFrame + Math.abs(clip?.framesToGiveEnd ?? 0));

  return (
    <div className="flex flex-col gap-y-2">
      <div className="p-5">
      <div className="flex flex-row items-center justify-between">
        <h4 className="text-brand-light text-sm font-medium text-start mb-4">Duration</h4>
        <span
          onClick={() => {
            const startFrame = (clip?.startFrame ?? 0) - (clip?.framesToGiveStart ?? 0);
            const endFrame = (clip?.endFrame ?? 0) - (clip?.framesToGiveEnd ?? 0);
            resizeClip(clipId, 'left', startFrame);
            resizeClip(clipId, 'right', endFrame);
            setTimeout(() => {
              updateClip(clipId, { speed: 1});
            }, 10);
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
        <PropertiesSlider label="Speed" value={Math.round(speed * 10) / 10} onChange={setSpeed} suffix="x" min={0.1} max={5} step={0.1}  />
            <div className="flex flex-row gap-x-2">
        <Input label="Start Frame" value={startFrame.toString()} onChange={
            (value) => setStartFrame(Number(value))
        } startLogo="F" canStep step={1} min={startFrameMin} max={startFrameMax} />
        <Input label="End Frame" value={endFrame.toString()} onChange={(value) => setEndFrame(Number(value))} startLogo="F" canStep step={1} min={endFrameMin} max={endFrameMax} />    
        </div>
        </div>
        </div>
    </div>
  )
}

export default DurationProperties