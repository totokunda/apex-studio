import { FiMinusCircle, FiPlusCircle } from "react-icons/fi";
import React, { useState, useRef, useEffect, useMemo } from "react";
import { cn } from '@/lib/utils';
import { BackButton, RewindBackward, RewindForward, ScissorsButton, TrashButton, PlayPauseButton, ExtendTimelineButton, ReduceTimelineButton } from "./Buttons";
import { useControlsStore } from "@/lib/control";
import { useClipStore } from "@/lib/clip";
import { ZoomLevel } from '@/lib/types';
import { MIN_DURATION } from '@/lib/settings';

const TimelineZoom = () => {
    const { zoomLevel, setZoomLevel, setTimelineDuration, resetTimelineDuration,  focusFrame, setFocusFrame, focusAnchorRatio, totalTimelineFrames, setFocusAnchorRatio, minZoomLevel, maxZoomLevel } = useControlsStore();
    const [isDragging, setIsDragging] = useState(false);
    const [isHovering, setIsHovering] = useState(false);
    const barRef = useRef<HTMLDivElement>(null);
    const clips = useClipStore((state) => state.clips);
    const hasClips = useMemo(() => clips.length > 0, [clips]);
    useEffect(() => {
        if (!hasClips) {
            setZoomLevel(1);
            setFocusFrame(0);
            resetTimelineDuration();
        }
    }, [hasClips]);

    

    
    const setZoom = (level:number) => { 
        // Clamp to valid integer zoom step
        const clampedLevel = Math.max(minZoomLevel, Math.min(maxZoomLevel, Math.round(level)));
        
        // Use dynamic baseline where zoomLevel 1 spans the entire timeline baseline
        // and max zoom spans exactly MIN_DURATION (5 frames)
        const maxDuration = Math.max(1, totalTimelineFrames);
        const minDuration = Math.max(1, Math.min(MIN_DURATION, maxDuration));
        const steps = Math.max(1, maxZoomLevel - minZoomLevel);
        const ratio = minDuration / maxDuration;
        const levelIndex = clampedLevel - minZoomLevel; // 0..steps

        // Build deterministic duration table once per call (cheap) to avoid rounding drift
        const durations: number[] = new Array(steps + 1).fill(0).map((_, i) => {
            const ti = i / steps; // 0..1
            const d = Math.round(maxDuration * Math.pow(ratio, ti));
            return Math.max(minDuration, Math.min(maxDuration, d));
        });

        const targetDuration = durations[levelIndex];

        // Anchor focusFrame; clamp within timeline bounds while keeping exact width
        let newStart = Math.round(focusFrame - (focusAnchorRatio * targetDuration));
        newStart = Math.max(0, Math.min(newStart, Math.max(0, totalTimelineFrames - targetDuration)));
        const newEnd = newStart + targetDuration;

        // Keep anchor ratio consistent with the final clamped window
        const newAnchor = targetDuration > 0 ? (focusFrame - newStart) / targetDuration : 0.5;
        setFocusAnchorRatio(Math.max(0, Math.min(1, newAnchor)));

        
        setTimelineDuration(newStart, newEnd);
        setZoomLevel(clampedLevel as ZoomLevel);
    }

    const handleMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsDragging(true);
        updateZoomFromMouse(e);
    };

    const handleMouseMove = (e: MouseEvent) => {
        if (!isDragging) return;
        updateZoomFromMouse(e);
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const updateZoomFromMouse = (e: React.MouseEvent | MouseEvent) => {
        if (!barRef.current) return;
        
        const rect = barRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));
        const newZoom = minZoomLevel + Math.round(percentage * (maxZoomLevel - minZoomLevel));
        setZoom(newZoom);
    };

    // Calculate circle position based on zoom level
    const getCirclePosition = () => {
        const progress = (zoomLevel - minZoomLevel) / (maxZoomLevel - minZoomLevel);
        return progress * 100; // percentage
    };

    // Add global event listeners for mouse move and up
    React.useEffect(() => {
        if (isDragging) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            return () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
            };
        }
    }, [isDragging]);

    return (
        <div className="flex items-center  gap-x-2 justify-center transition-opacity duration-300"> 
            
            <FiMinusCircle onClick={() => {
                if (zoomLevel === 1 || !hasClips) return;
                setZoom(zoomLevel - 1);
            }} className={cn('text-brand-light/70 h-4 w-4 duration-300', {
                "opacity-60": zoomLevel === 1 || !hasClips,
                "opacity-100": zoomLevel > 1 && hasClips,
                "cursor-not-allowed": zoomLevel === 1 || !hasClips,
                "cursor-pointer": zoomLevel > 1 && hasClips,
            })} />
            <div 
                ref={barRef}
                onMouseDown={(e) => hasClips && handleMouseDown(e)}
                onMouseEnter={() => hasClips && setIsHovering(true)}
                onMouseLeave={() => hasClips && setIsHovering(false)}
                className={cn("h-1 w-24 rounded-full bg-brand-light/10 transform-gpu cursor-pointer relative", {
                    "cursor-grabbing": isDragging,
                })}
            >
                <div className={cn("h-1 rounded-full bg-brand-light/70 transition-all duration-300 transform-gpu pointer-events-none", {
                    "w-0": zoomLevel === 1 || !hasClips,
                    "w-1/9": zoomLevel === 2,    
                    "w-2/9": zoomLevel === 3, 
                    "w-3/9": zoomLevel === 4,
                    "w-4/9": zoomLevel === 5,
                    "w-5/9": zoomLevel === 6,
                    "w-6/9": zoomLevel === 7,
                    "w-7/9": zoomLevel === 8,
                    "w-8/9": zoomLevel === 9,
                    "w-9/9": zoomLevel === 10,
                    "transition-none": isDragging
                })}></div>
                 
                {/* Draggable Circle */}
                <div 
                    className={cn("absolute top-1/2 w-3 h-3 bg-brand-light/90 rounded-full transform -translate-y-1/2 -translate-x-1/2 transition-all duration-200 pointer-events-none", {
                        "opacity-100 scale-100": isHovering || isDragging,
                        "opacity-0 scale-75": !isHovering && !isDragging,
                        "bg-brand-light": isDragging,
                        "transition-none": isDragging,
                       
                    })}
                    style={{
                        left: `${getCirclePosition()}%`
                    }}
                />
            </div>
            <FiPlusCircle onClick={() => {
                if (zoomLevel >= 10 || !hasClips) return;
                setZoom(zoomLevel + 1);
            }} className={cn('text-brand-light/70 h-4 w-4 duration-300', {
                "opacity-100": zoomLevel < 10 && hasClips,
                "opacity-60": zoomLevel >= 10 || !hasClips,
                "cursor-not-allowed": zoomLevel >= 10 || !hasClips,
                "cursor-pointer": zoomLevel < 10 && hasClips,
            })} />
            
        </div>
    )
}

interface TimeControlProps {

}

const TimeControl:React.FC<TimeControlProps> = () => {
    const { clipDuration} = useClipStore();
    const { focusFrame, fps } = useControlsStore();

    const formatTime = (frames: number) => {
        if (frames === 0 || frames === undefined || frames === null || isNaN(frames) || frames === Infinity || frames === -Infinity) return '00:00.00';
        // convert frames to seconds
        const seconds = frames / fps;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toFixed(2).padStart(5, '0')}`;
    };

    return (
        <div className="flex items-center gap-x-2 w-full">
            <span className="text-brand-light/60 text-xs">
            <span className="w-15 inline-block">{formatTime(focusFrame)}</span>/<span className="w-15 inline-block">{formatTime(clipDuration)}</span>
            </span>
        </div>
    )
}

const Controls = () => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || (target as HTMLElement).isContentEditable)) {
        return;
      }

      const isMod = e.metaKey || e.ctrlKey;
      const controls = useControlsStore.getState();
      const clipsStore = useClipStore.getState();
      const selectedIds = controls.selectedClipIds || [];

      // Delete selected clips
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedIds.length > 0) {
          e.preventDefault();
          selectedIds.forEach((id) => clipsStore.removeClip(id));
          controls.clearSelection();
        }
        return;
      }

      // Copy
      if (isMod && e.key.toLowerCase() === 'c') {
        if (selectedIds.length > 0) {
          e.preventDefault();
          clipsStore.copyClips(selectedIds);
        }
        return;
      }

      // Cut
      if (isMod && e.key.toLowerCase() === 'x') {
        if (selectedIds.length > 0) {
          e.preventDefault();
          clipsStore.cutClips(selectedIds);
          controls.clearSelection();
        }
        return;
      }

      // Paste at current focus frame
      if (isMod && e.key.toLowerCase() === 'v') {
        e.preventDefault();
        const { focusFrame } = controls;
        clipsStore.pasteClips(focusFrame);
        return;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);
  return (
    <div className='relative  flex items-center '>
        <div className="flex items-center w-full bg-brand-background/30 justify-between border-b border-brand-light/5 px-5 py-1">
    <div className="flex items-center gap-x-2"> 
        <ScissorsButton />
        <TrashButton />
        </div>
    <div className=' flex items-center justify-center gap-x-4 absolute left-1/2 -translate-x-1/2'>
        <BackButton />
        <div className="flex items-center gap-x-2">
            <RewindBackward />
            <RewindForward />
        </div>
        <PlayPauseButton />
        <TimeControl />
    </div>
    <div className="flex items-center gap-x-2">

    <div className="ml-3">
        <TimelineZoom />
    </div>
    </div>
    </div>
    </div>
  )
}

export default Controls; 