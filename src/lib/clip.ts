import { create } from "zustand";
import { AnyClipProps, TimelineProps } from "./types";
import { v4 as uuidv4 } from 'uuid';
import { useControlsStore } from "./control";

interface ClipStore {  
    // Clips
    clipDuration: number;
    _setClipDuration: (duration: number) => void;
    clips: AnyClipProps[];
    getClipById: (clipId: string, timelineId?: string) => AnyClipProps | undefined;
    setClips: (clips: AnyClipProps[]) => void;
    addClip: (clip: AnyClipProps) => void;
    removeClip: (clipId: string) => void;
    updateClip: (clipId: string, clipToUpdate: Partial<AnyClipProps>) => void;
    resizeClip: (clipId: string, side: 'left' | 'right', newFrame: number) => void;
    splitClip: (cutFrame: number) => void;
    mergeClips: (clipIds: string[]) => void;
    moveClipToEnd: (clipId: string) => void;
    clipboard: AnyClipProps[];
    copyClips: (clipIds: string[]) => void;
    cutClips: (clipIds: string[]) => void;
    pasteClips: (atFrame?: number) => void;
    getClipAtFrame: (frame: number) => [AnyClipProps, number] | null;

    // Timelines
    timelines: TimelineProps[];
    getClipsForTimeline: (timelineId: string) => AnyClipProps[];
    getTimelineById: (timelineId: string) => TimelineProps | undefined;
    setTimelines: (timelines: TimelineProps[]) => void;
    addTimeline: (timeline: TimelineProps) => void;
    removeTimeline: (timelineId: string) => void;
    updateTimeline: (timelineId: string, timelineToUpdate: Partial<TimelineProps>) => void;
}

// Helper function to calculate total duration of all clips
const calculateTotalClipDuration = (clips: AnyClipProps[]): number => {
    const maxEndFrame = Math.max(...clips.map(clip => clip.endFrame || 0));
    return maxEndFrame;
};


// Helper function to resolve overlaps by shifting clips to maintain frame gaps
const resolveOverlaps = (clips: AnyClipProps[]): AnyClipProps[] => {
    if (clips.length === 0) return clips;
    
    // Sort clips by start frame
    const sortedClips = [...clips].sort((a, b) => (a.startFrame || 0) - (b.startFrame || 0));
    const resolvedClips: AnyClipProps[] = [];
    
    for (let i = 0; i < sortedClips.length; i++) {
        const currentClip = { ...sortedClips[i] };
        const currentStart = currentClip.startFrame || 0;
        const currentEnd = currentClip.endFrame || 0;
        
        // Check for overlap with previous clip
        if (resolvedClips.length > 0) {
            const previousClip = resolvedClips[resolvedClips.length - 1];
            const previousEnd = previousClip.endFrame || 0;
            
            // If current clip overlaps with previous clip, shift it to start after previous clip ends
            if (currentStart < previousEnd) {
                const clipDuration = currentEnd - currentStart;
                currentClip.startFrame = previousEnd;
                currentClip.endFrame = previousEnd + clipDuration;
            }
        }
        
        resolvedClips.push(currentClip);
    }
    
    return resolvedClips;
};

export const getCorrectedClip = (clipId: string, clips: AnyClipProps[]): AnyClipProps | null => {
    // find the clip in the clips array
    const resolvedClips = resolveOverlaps(clips);
    const clip = resolvedClips.find((clip) => clip.clipId === clipId);
    if (!clip) return null;
    return clip;
}

export const useClipStore = create<ClipStore>((set, get) => ({       
    clipDuration: 0,
    _setClipDuration: (duration) => set({ clipDuration: duration }),
    clips: [],  
    timelines: [],
    getTimelineById: (timelineId: string) => get().timelines.find((timeline) => timeline.timelineId === timelineId),
    setTimelines: (timelines: TimelineProps[]) => set({ timelines }),
    addTimeline: (timeline: TimelineProps) => set((state) => {
        const newTimelines = [...state.timelines, timeline];
        return { timelines: newTimelines };
    }),
    removeTimeline: (timelineId: string) => set((state) => {
        const newTimelines = state.timelines.filter((timeline) => timeline.timelineId !== timelineId);
        return { timelines: newTimelines };
    }),
    updateTimeline: (timelineId: string, timelineToUpdate: Partial<TimelineProps>) => set((state) => {
        const newTimelines = state.timelines.map((timeline) => timeline.timelineId === timelineId ? { ...timeline, ...timelineToUpdate } : timeline);
        return { timelines: newTimelines };
    }),
    clipboard: [],
    getClipsForTimeline: (timelineId: string) => get().clips.filter((clip) => clip.timelineId === timelineId),
    getClipById: (clipId: string, timelineId?: string) => get().clips.find((clip) => clip.clipId === clipId && (timelineId ? clip.timelineId === timelineId : true)),
    setClips: (clips: AnyClipProps[]) => {
        const resolvedClips = resolveOverlaps(clips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        set({ clips: resolvedClips, clipDuration });
    },
    addClip: (clip: AnyClipProps) => set((state) => {
        const newClips = [...state.clips, clip];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    removeClip: (clipId: string) => set((state) => {
        const newClips = state.clips.filter((clip) => clip.clipId !== clipId);
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    updateClip: (clipId: string, clipToUpdate: Partial<AnyClipProps>) => set((state) => {
        const newClips = state.clips.map((clip) => clip.clipId === clipId ? { ...clip, ...clipToUpdate } : clip);
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    resizeClip: (clipId: string, side: 'left' | 'right', newFrame: number) => set((state) => {
        const sortedClips = [...state.clips].sort((a, b) => (a.startFrame || 0) - (b.startFrame || 0));
        const currentIndex = sortedClips.findIndex(c => c.clipId === clipId);
        if (currentIndex === -1) return { clips: state.clips };
        
        const currentClip = sortedClips[currentIndex];
        const newClips = [...state.clips];
        
        if (side === 'right') {
            // Resize right edge - adjust current clip's end and shift all clips after it
            const oldEndFrame = currentClip.endFrame || 0;
            const newEndFrame = Math.max((currentClip.startFrame || 0) + 1, newFrame);
            const frameDelta = newEndFrame - oldEndFrame;
            
            if (frameDelta + (currentClip.framesToGiveEnd || 0) > 0) {
                return { clips: state.clips };
            }

            const currentClipIndex = newClips.findIndex(c => c.clipId === clipId);
            newClips[currentClipIndex] = { ...currentClip, endFrame: newEndFrame, framesToGiveEnd: frameDelta + (currentClip.framesToGiveEnd || 0) };

        } else if (side === 'left') {
            // Resize left edge - adjust current clip's start and shift all clips before it
            const oldStartFrame = currentClip.startFrame || 0;
            const newStartFrame = Math.min((currentClip.endFrame || 0) - 1, newFrame);
            let frameDelta = newStartFrame - oldStartFrame;

            if (frameDelta + (currentClip.framesToGiveStart || 0) < 0) {
                return { clips: state.clips };
            }

            if (frameDelta == 0 && (currentClip.framesToGiveStart || 0) >  0) {
                frameDelta = Math.max(0, Math.min(1, (currentClip.framesToGiveStart || 0) - 1)); 

            } else {
                const currentClipIndex = newClips.findIndex(c => c.clipId === clipId);
                newClips[currentClipIndex] = { ...currentClip, startFrame: newStartFrame, framesToGiveStart: frameDelta + (currentClip.framesToGiveStart || 0) };

            }
        }

        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    // Create two new clips from the original clip at the cut frame
    splitClip: (cutFrame: number) => set((state) => {
        // Find the clip that contains the cut frame
        const clip = state.clips.find((clip) => {
            const startFrame = clip.startFrame || 0;
            const endFrame = clip.endFrame || 0;
            return cutFrame > startFrame && cutFrame < endFrame;
        });
        
        if (!clip) return { clips: state.clips };
        
        // remove the clip from the array 
        const filteredClips = state.clips.filter((c) => c.clipId !== clip.clipId);
        
        // create new clip ids
        const newClipId1 = uuidv4();
        const newClipId2 = uuidv4();
        const newClip1 = { ...clip, endFrame: cutFrame, clipId: newClipId1 };
        const newClip2 = { ...clip, startFrame: cutFrame, clipId: newClipId2};
        const newClips = [...filteredClips, newClip1, newClip2];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    mergeClips: (clipIds: string[]) => set((state) => {
        if (clipIds.length < 2) return { clips: state.clips };
        
        // Find all clips to merge
        const clipsToMerge = clipIds.map(id => state.clips.find(clip => clip.clipId === id)).filter(Boolean) as AnyClipProps[];
        
        if (clipsToMerge.length < 2) return { clips: state.clips };
        
        // Sort clips by start frame to check adjacency
        const sortedClips = clipsToMerge.sort((a, b) => (a.startFrame || 0) - (b.startFrame || 0));
        
        // Check if all clips are frame-adjacent (no gaps between them)
        for (let i = 0; i < sortedClips.length - 1; i++) {
            const currentEnd = sortedClips[i].endFrame || 0;
            const nextStart = sortedClips[i + 1].startFrame || 0;
            
            // If there's a gap between clips, don't merge
            if (currentEnd !== nextStart) {
                return { clips: state.clips };
            }
        }
        
        // Remove the clips from the array
        const filteredClips = state.clips.filter((clip) => !clipIds.includes(clip.clipId));
        
        // Find the bounds of all clips to merge
        const clipStart = sortedClips[0].startFrame || 0;
        const clipEnd = sortedClips[sortedClips.length - 1].endFrame || 0;
        
        // Use the first clip as the base and merge all others into it
        const baseClip = sortedClips[0];
        const newClip = { ...baseClip, startFrame: clipStart, endFrame: clipEnd };
        const newClips = [...filteredClips, newClip];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    copyClips: (clipIds: string[]) => set((state) => {
        if (!clipIds || clipIds.length === 0) return { clipboard: state.clipboard };
        const toCopy = clipIds
            .map(id => state.clips.find(c => c.clipId === id))
            .filter(Boolean) as AnyClipProps[];
        return { clipboard: toCopy.map(c => ({ ...c })) };
    }),
    cutClips: (clipIds: string[]) => set((state) => {
        if (!clipIds || clipIds.length === 0) return { clips: state.clips, clipboard: state.clipboard };
        const toCut = clipIds
            .map(id => state.clips.find(c => c.clipId === id))
            .filter(Boolean) as AnyClipProps[];
        const remaining = state.clips.filter(c => !clipIds.includes(c.clipId));
        const resolvedClips = resolveOverlaps(remaining);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration, clipboard: toCut.map(c => ({ ...c })) };
    }),
    pasteClips: (atFrame?: number) => set((state) => {
        const clipboardItems = state.clipboard || [];
        if (clipboardItems.length === 0) return { clips: state.clips };
        const baseStart = Math.min(...clipboardItems.map(c => c.startFrame || 0));
        const insertionFrame = Math.max(0, Math.round(atFrame || 0));
        const newIds: string[] = [];
        const pasted = clipboardItems.map(template => {
            const templateStart = template.startFrame || 0;
            const templateEnd = template.endFrame || 0;
            const duration = Math.max(1, templateEnd - templateStart);
            const offset = templateStart - baseStart;
            const start = insertionFrame + offset;
            const end = start + duration;
            const newId = uuidv4();
            newIds.push(newId);
            return { ...template, clipId: newId, startFrame: start, endFrame: end, framesToGiveEnd: 0, framesToGiveStart: 0 } as AnyClipProps;
        });
        const newClips = [...state.clips, ...pasted];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        // Select newly pasted clips
        try {
            const controls = useControlsStore.getState();
            controls.setSelectedClipIds(newIds);
        } catch {}
        return { clips: resolvedClips, clipDuration };
    }),
    moveClipToEnd: (clipId: string) => set((state) => {
        const clip = state.clips.find((clip) => clip.clipId === clipId);
        if (!clip) return { clips: state.clips };
        const newClips = [...state.clips.filter((c) => c.clipId !== clipId), clip];
        const clipDuration = calculateTotalClipDuration(newClips);
        return { clips: newClips, clipDuration };
    }),
    getClipAtFrame: (frame: number) => {
        const clips = get().clips;
        const clip = clips.find((clip) => frame >= (clip.startFrame || 0) && frame <= (clip.endFrame || 0));
        if (!clip) return null;
        return [clip, frame - (clip.startFrame || 0)];
    }
}));

export const getClipWidth = (startFrame:number, endFrame:number, timelineWidth:number, timelineDuration:number[]) => {
    const [timelineStartFrame, timelineEndFrame] = timelineDuration;
    const percentage = (endFrame - startFrame) / (timelineEndFrame - timelineStartFrame);
    return timelineWidth * percentage;
}

export const getClipX = (startFrame:number | null, endFrame:number | null, timelineWidth:number | null, timelineDuration:number[]) => {
    if (startFrame === null || endFrame === null || timelineWidth === null) return 0;

    const [timelineStartFrame, timelineEndFrame] = timelineDuration;
    const relativePosition = (startFrame - timelineStartFrame) / (timelineEndFrame - timelineStartFrame); 
    return relativePosition * timelineWidth;
}   
