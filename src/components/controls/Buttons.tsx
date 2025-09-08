import { TbPlayerSkipBack,  TbRewindBackward5,  TbLadder, TbPlus, TbMinus} from "react-icons/tb";

import { TbRewindForward5 } from "react-icons/tb";
import { FaCirclePause, FaCirclePlay } from "react-icons/fa6";
import { FiScissors, FiTrash } from "react-icons/fi";
import { CgMergeVertical } from "react-icons/cg";
import { useControlsStore } from "@/lib/control";
import { useClipStore } from "@/lib/clip";
import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { MAX_DURATION } from "@/lib/settings";

const BackButton = () => {
    const {setFocusFrame,  focusFrame} = useControlsStore();
    const {clips} = useClipStore();
    const hasClips = clips.length > 0;
    const disabled = focusFrame === 0 || !hasClips;
    const handleBack = useCallback(() => {
        if (disabled) return;
        setFocusFrame(0);
    }, [focusFrame, hasClips, setFocusFrame]);
    return (
        <div className={cn("flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300", disabled && "opacity-30 cursor-not-allowed hover:opacity-30")} onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleBack();
        }}> 
            <TbPlayerSkipBack className='text-brand-light/60 h-4 w-4' />
        </div>
    )
}

const RewindBackward = () => {
    const {setFocusFrame, fps, focusFrame} = useControlsStore();
    const {clips} = useClipStore();
    const hasClips = clips.length > 0;
    const disabled = focusFrame === 0 || !hasClips;
    const handleRewindBackward = useCallback(() => {    
        if (disabled) return;
        const framesBack = Math.max(0, focusFrame - fps * 5);
        setFocusFrame(framesBack);
    }, [disabled, focusFrame, fps, setFocusFrame]);
    return (
        <div className={cn("flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300", disabled && "opacity-30 cursor-not-allowed hover:opacity-30")}   onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleRewindBackward();
        }}> 
            <TbRewindBackward5 className='text-brand-light/60 h-5 w-5' />
        </div>
    )
}

const RewindForward = () => {
    const {setFocusFrame, fps, focusFrame} = useControlsStore();
    const {clipDuration, clips} = useClipStore();
    const hasClips = clips.length > 0;
    const disabled = focusFrame === clipDuration || !hasClips;
    const handleRewindForward = useCallback(() => {
        if (disabled) return;
        const framesForward = Math.min(focusFrame + fps * 5, clipDuration);
        setFocusFrame(framesForward);
    }, [disabled, focusFrame, fps, clipDuration, setFocusFrame]);
    return (
        <div className={cn("flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300", disabled && "opacity-30 cursor-not-allowed hover:opacity-30")} onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleRewindForward();
        }}> 
            <TbRewindForward5 className='text-brand-light/60 h-5 w-5' />
        </div>
    )
}

const PauseButton = () => {
    return (
        <div onClick={(e) => {
        }} className="flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300"> 
            <FaCirclePause className='text-brand-light/70 h-7 w-7' />
        </div>
    )
}   

const PlayButton = () => {
    return (
        <div className="flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300"> 
            <FaCirclePlay className='text-brand-light/70 h-7 w-7' />
        </div>
    )
}

const ExtendTimelineButton:React.FC<{numSeconds?: number}> = ({numSeconds = 15}) => {
    const { zoomLevel, fps, incrementTotalTimelineFrames, shiftTimelineDuration, setFocusFrame, focusFrame} = useControlsStore();
    const {clips} = useClipStore();
    const hasClips = clips.length > 0;
    const disabled = !hasClips || zoomLevel !== 1;
  const handleExtendTimeline = useCallback(() => {
        const increment = fps * numSeconds; // 15 second at 24fps default
        if (zoomLevel === 1) {
            incrementTotalTimelineFrames(increment);
            setFocusFrame(focusFrame + increment);
            shiftTimelineDuration(increment);
        }
    }, [fps, numSeconds, incrementTotalTimelineFrames, shiftTimelineDuration, focusFrame]);
    return (
        <div className={cn("flex items-center cursor-pointer  relative justify-center opacity-60 hover:opacity-100 transition-opacity duration-300", disabled && "opacity-30 cursor-not-allowed hover:opacity-30")} onClick={handleExtendTimeline}> 
            <TbLadder className='h-6 w-6 rotate-90 text-brand-light' /> 
            <TbPlus className='absolute top-1.5 -right-2 h-1.5 w-1.5 text-brand-light stroke-3' /> 
        </div>
    )
}

const ReduceTimelineButton:React.FC<{numSeconds?: number}> = ({numSeconds = 15}) => {
    const { timelineDuration, zoomLevel, fps, decrementTotalTimelineFrames, totalTimelineFrames, shiftTimelineDuration, setFocusFrame, focusFrame } = useControlsStore();
    const {clipDuration, clips} = useClipStore();
    const hasClips = clips.length > 0;
    const disabled = (totalTimelineFrames <= MAX_DURATION || totalTimelineFrames - (fps * numSeconds) < clipDuration) || zoomLevel !== 1 || !hasClips;

  const handleReduceTimeline = useCallback(() => {
        if (disabled) return;
        const decrement = fps * numSeconds; // 15 second at 24fps default
        if (zoomLevel === 1) {
            const timelineEndFrame = timelineDuration[1]
            decrementTotalTimelineFrames(decrement);
            setFocusFrame(focusFrame - decrement);
            if (timelineEndFrame >= totalTimelineFrames - decrement) { 
                shiftTimelineDuration(totalTimelineFrames - timelineEndFrame);
                // ensure focus frame is within the new timeline duration
            }
            
        }
        
    }, [zoomLevel, fps, numSeconds, timelineDuration, decrementTotalTimelineFrames, clipDuration, hasClips, focusFrame]);
    return (
        <div className={cn("flex items-center cursor-pointer  relative justify-center opacity-60 hover:opacity-100 transition-opacity duration-300", {
            "opacity-30 cursor-not-allowed hover:opacity-30": disabled,
            "opacity-60 hover:opacity-100": !disabled
        })} onClick={handleReduceTimeline}> 
            <TbLadder className='h-6 w-6 rotate-90 text-brand-light' />
            <TbMinus className='absolute top-1.5 -left-2 h-1.5 w-1.5 text-brand-light stroke-3' /> 
        </div>
    )
}

const ScissorsButton = () => {
    const { splitClip } = useClipStore();
    const {focusFrame} = useControlsStore();
    const {clips} = useClipStore();
    const hasClips = clips.length > 0;
    const disabled = !hasClips || focusFrame === 0;
    const handleSplit = useCallback(() => {
        if (disabled) return;
        splitClip(focusFrame);
    }, [disabled, splitClip, focusFrame]);
    return (
        <div className={cn("flex shrink-0 transform-gpu items-center cursor-pointer gap-x-2 py-4 justify-center opacity-60 hover:opacity-100 transition-opacity duration-300", disabled && "opacity-30 cursor-not-allowed hover:opacity-30")} onClick={handleSplit}> 
            <FiScissors className=" text-brand-light h-4 w-4" />
        </div>
    )
}

const TrashButton = () => {
    const { selectedClipIds, clearSelection } = useControlsStore();
    const { removeClip, clips } = useClipStore();
    const hasClips = clips.length > 0;
    const disabled = selectedClipIds.length === 0 || !hasClips;
    const handleDelete = useCallback(() => {
        if (disabled) return;
        if (selectedClipIds.length === 0) return;
        selectedClipIds.forEach((clipId) => {
            removeClip(clipId);
        });
        clearSelection();
    }, [disabled, selectedClipIds, removeClip, clearSelection]);

    return (
        <div  className={cn("flex shrink-0 transform-gpu items-center cursor-pointer  gap-x-2 py-4 justify-center opacity-60 hover:opacity-100 transition-opacity duration-300", disabled && "opacity-30 cursor-not-allowed hover:opacity-30")} onClick={handleDelete}> 
            <FiTrash className=" text-brand-light h-4 w-4" />
        </div>
    )
}

const MergeButton = () => {
    const { selectedClipIds, clearSelection } = useControlsStore();
    const { mergeClips, clips } = useClipStore();
    const hasClips = clips.length > 0;
    const disabled = selectedClipIds.length < 2 || !hasClips;

    const handleMerge = useCallback(() => {
        if (disabled) return;
        if (selectedClipIds.length < 2) return;
        mergeClips(selectedClipIds);
        clearSelection();
    }, [selectedClipIds, mergeClips, clearSelection]);

    return (
        <div className={cn("flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300", disabled && "opacity-30 cursor-not-allowed hover:opacity-30")}  onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleMerge();
        }}> 
            <CgMergeVertical className='text-brand-light h-5 w-5' />
        </div>
    )
}

const PlayPauseButton = () => {
    const { play, pause, isPlaying } = useControlsStore();
    return (
        <div onClick={() => {
            if (isPlaying) {
                pause();
            } else {
                play();
            }
        }} className="flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300"> 
            {isPlaying ? <PauseButton /> : <PlayButton />}
        </div>
    )
}


export { BackButton, RewindBackward, RewindForward, PauseButton, PlayButton, ScissorsButton, TrashButton, MergeButton, PlayPauseButton, ExtendTimelineButton, ReduceTimelineButton };