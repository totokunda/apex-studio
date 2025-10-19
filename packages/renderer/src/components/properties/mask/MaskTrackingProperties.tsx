import { useClipStore } from '@/lib/clip';
import { useControlsStore } from '@/lib/control';
import { MaskClipProps, MaskTrackingDirection } from '@/lib/types';
import React, { useCallback, useMemo } from 'react';
import { LuChevronDown, LuPlay, LuTrash2 } from 'react-icons/lu';
import { cn } from '@/lib/utils';
import { getGlobalFrame } from '@/lib/clip';
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuRadioItem, DropdownMenuRadioGroup } from '@/components/ui/dropdown-menu';

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
        <h4 className="text-brand-light text-[12px] font-medium text-start">Mask Tracking</h4>
        
        <div>
          <h3 className="text-brand-light text-[11px] font-medium text-start">
            Direction
          </h3>
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="w-full mt-2 flex flex-row items-center justify-between gap-x-2 px-3 py-2 cursor-pointer rounded-md bg-brand border border-brand-light/20 hover:bg-brand-light/10 transition-all duration-200">
                <span className="text-brand-light text-[11px] font-medium">
                  {mask.trackingDirection === 'forward' ? 'Forward' : mask.trackingDirection === 'backward' ? 'Backward' : 'Both'}
                </span>
                <LuChevronDown className="w-3 h-3 text-brand-light" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className='w-[var(--radix-dropdown-menu-trigger-width)] dark font-poppins bg-brand-background'>
              <DropdownMenuRadioGroup value={mask.trackingDirection ?? 'both'} onValueChange={(value: string) => updateMask({ trackingDirection: value as MaskTrackingDirection })}>
                <DropdownMenuRadioItem value='forward' className='w-full'>
                  <span className="text-brand-light text-[11px] font-medium">Forward</span>
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value='backward' className='w-full'>
                  <span className="text-brand-light text-[11px] font-medium">Backward</span>
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value='both' className='w-full'>
                  <span className="text-brand-light text-[11px] font-medium">Both</span>
                </DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
        </DropdownMenu>
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
