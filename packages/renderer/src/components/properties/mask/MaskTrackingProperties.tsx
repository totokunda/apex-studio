import { getLocalFrame, useClipStore } from '@/lib/clip';
import { useControlsStore } from '@/lib/control';
import { MaskClipProps } from '@/lib/types';
import React, { useCallback, useMemo } from 'react';
import PropertiesSlider from '../PropertiesSlider';
import { LuPlay, LuPlus, LuTrash2 } from 'react-icons/lu';
import { cn } from '@/lib/utils';
import { getGlobalFrame } from '@/lib/clip';

interface MaskTrackingPropertiesProps {
  mask: MaskClipProps;
  clipId: string;
}

const MaskTrackingProperties: React.FC<MaskTrackingPropertiesProps> = ({ mask, clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId));
  const updateClip = useClipStore((s) => s.updateClip);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const setFocusFrame = useControlsStore((s) => s.setFocusFrame);

  const keyframeNumbers = useMemo(() => {
    if (!mask) return [];
    
    const keyframes = mask.keyframes instanceof Map 
      ? mask.keyframes 
      : (mask.keyframes as Record<number, any>);
    
    return keyframes instanceof Map
      ? Array.from(keyframes.keys()).sort((a, b) => a - b)
      : Object.keys(keyframes).map(Number).sort((a, b) => a - b);
  }, [mask]);

  const activeKeyframe = useMemo(() => {
    return keyframeNumbers.filter(k => k <= focusFrame).pop();
  }, [keyframeNumbers, focusFrame]);

  const updateMask = (updates: Partial<MaskClipProps>) => {
    if (!clip || !mask) return;
    
    const masks = (clip as any).masks || [];
    const updatedMasks = masks.map((m: MaskClipProps) =>
      m.id === mask.id ? { ...m, ...updates } : m
    );
    
    updateClip(clipId, { masks: updatedMasks });
  };


  const handleRemoveKeyframe = (frame: number) => {
    if (!mask || !clip) return;
    
    // Don't allow removing if it's the only keyframe
    if (keyframeNumbers.length <= 1) return;
    
    const keyframes = mask.keyframes instanceof Map 
      ? mask.keyframes 
      : (mask.keyframes as Record<number, any>);
    
    const updatedKeyframes = keyframes instanceof Map
      ? new Map(keyframes)
      : { ...keyframes };
    
    if (updatedKeyframes instanceof Map) {
      updatedKeyframes.delete(frame);
    } else {
      delete updatedKeyframes[frame];
    }
    
    

    // set the focus frame to the nearest keyframe in any direction
    const nearestKeyframe = [...keyframeNumbers]
      .sort((a, b) => Math.abs(a - frame) - Math.abs(b - frame)).filter(k => k !== frame)[0]
    if (nearestKeyframe !== undefined) {
      const globalFrame = getGlobalFrame(nearestKeyframe, clip);
      setFocusFrame(globalFrame, true);
    }

    updateMask({ keyframes: updatedKeyframes });
  };

  const handleSelectKeyframe = useCallback((frame: number) => {
    if (!clip) return;
    setFocusFrame(getGlobalFrame(frame, clip));
  }, [clip, setFocusFrame]);


  if (!mask) {
    return (
      <div className="p-4 px-5">
        <p className="text-brand-light/50 text-[11px]">No mask selected</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 flex flex-col gap-y-4 px-5 min-w-0">
        <h4 className="text-brand-light text-[12px] font-medium text-start">Tracking</h4>
        {/* Confidence Threshold */}
        <PropertiesSlider
          label="Confidence Threshold"
          value={mask.confidenceThreshold ?? 0.5}
          onChange={(value) => updateMask({ confidenceThreshold: value })}
          min={0}
          max={1}
          step={0.01}
          toFixed={2}
        />
        {/* Keyframe Management */}
        <div className="flex flex-col gap-y-2">
          <div className="flex flex-row items-center justify-between">
            <span className="text-brand-light text-[11px] font-medium">Keyframes</span>
          </div>

          {/* Keyframe List */}
          <div className="flex flex-col gap-y-1 max-h-32 overflow-y-auto">
            {keyframeNumbers.length === 0 ? (
              <p className="text-brand-light/50 text-[10px] italic">No keyframes</p>
            ) : (
              keyframeNumbers.map((frame) => (
                <div
                  key={frame}
                  onClick={() => handleSelectKeyframe(frame)}
                  className={cn(
                    "flex flex-row items-center justify-between px-2 py-1.5 rounded border cursor-pointer hover:bg-brand-light/5 transition-all duration-200",
                    frame === activeKeyframe
                      ? "bg-brand-light/10 border-brand-light/20"
                      : "bg-brand border-brand-light/10"
                  )}
                >
                  <span className="text-brand-light text-[11px] flex flex-row items-center">
                    <span className="mr-0.5">Frame {frame}</span>
                    {frame === activeKeyframe && (
                      <span className="text-brand-light/60 text-[8px] ml-1">active</span>
                    )}
                  </span>
                  {keyframeNumbers.length > 1 && (
                    <button
                      onClick={(event) => {
                        event.stopPropagation();
                        handleRemoveKeyframe(frame);
                      }}
                      className="text-brand-light/60 hover:text-red-400 transition-colors"
                    >
                      <LuTrash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
        <button className="w-full flex flex-row items-center gap-x-2 px-2 py-2.5 cursor-pointer rounded-md justify-center bg-brand border border-brand-light/20 hover:bg-brand-light/10 transition-all duration-200">
          <LuPlay className="w-3 h-3 text-brand-light" />
          <span className="text-brand-light text-[11px] font-medium">Track Mask</span>
        </button>

      </div>
    </div>
  );
};

export default MaskTrackingProperties;
