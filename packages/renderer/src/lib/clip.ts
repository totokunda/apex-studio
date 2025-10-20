import { create } from "zustand";
import { AnyClipProps, TimelineProps, ClipTransform, TimelineType, VideoClipProps, AudioClipProps, PreprocessorClipProps, ImageClipProps, PreprocessorClipType, MaskClipProps, MaskData } from "./types";
import { v4 as uuidv4 } from 'uuid';
import { useControlsStore } from "./control";
import { MediaItem } from "@/components/media/Item";
import { AUDIO_EXTS, MIN_DURATION, VIDEO_EXTS } from "./settings";
import { getMediaInfo } from "./media/utils";
import { getLowercaseExtension } from "@app/preload";
import { Preprocessor } from "./preprocessor";
import { remapMaskWithClipTransform } from "@/lib/mask/transformUtils";

export const PREPROCESSOR_BAR_HEIGHT = 24;

interface ClipStore {  
    // Clips
    clipDuration: number;
    _setClipDuration: (duration: number) => void;
    clips: AnyClipProps[];
    getClipById: (clipId: string, timelineId?: string) => AnyClipProps | undefined;
    getClipTransform: (clipId: string) => ClipTransform | undefined;
    setClipTransform: (clipId: string, transform: Partial<ClipTransform>) => void;
    setClips: (clips: AnyClipProps[]) => void;
    addClip: (clip: AnyClipProps) => void;
    removeClip: (clipId: string) => void;
    updateClip: (clipId: string, clipToUpdate: Partial<AnyClipProps>) => void;
    updatePreprocessor: (clipId: string, preprocessorId: string, preprocessorToUpdate: Partial<PreprocessorClipProps>) => void;
    resizeClip: (clipId: string, side: 'left' | 'right', newFrame: number) => void;
    separateClip: (clipId: string) => void;
    isValidResize: (clipId: string, side: 'left' | 'right', newFrame: number) => boolean;   
    splitClip: (cutFrame: number, clipId: string) => void;
    mergeClips: (clipIds: string[]) => void;
    moveClipToEnd: (clipId: string) => void;
    clipboard: AnyClipProps[];
    copyClips: (clipIds: string[]) => void;
    cutClips: (clipIds: string[]) => void;
    pasteClips: (atFrame?: number, targetTimelineId?: string) => void;
    getClipAtFrame: (frame: number) => [AnyClipProps, number] | null;
    activeMediaItem: MediaItem | null;
    setActiveMediaItem: (mediaItem: MediaItem | null) => void;
    ghostStartEndFrame: [number, number];
    setGhostStartEndFrame: (startFrame: number, endFrame: number) => void;
    ghostX: number;
    setGhostX: (x: number) => void;
    ghostGuideLines: [number, number] | null;
    setGhostGuideLines: (guideLines: [number, number] | null) => void;
    muteTimeline: (timelineId: string) => void;
    unmuteTimeline: (timelineId: string) => void;
    hideTimeline: (timelineId: string) => void;
    unhideTimeline: (timelineId: string) => void;
    isTimelineMuted: (timelineId: string) => boolean;
    isTimelineHidden: (timelineId: string) => boolean;
    ghostInStage: boolean;
    setGhostInStage: (inStage: boolean) => void;
    hoveredTimelineId: string | null;
    setHoveredTimelineId: (timelineId: string | null) => void;
    ghostTimelineId: string | null;
    setGhostTimelineId: (timelineId: string | null) => void;
    draggingClipId: string | null;
    setDraggingClipId: (clipId: string | null) => void;
    isDragging: boolean;
    setIsDragging: (isDragging: boolean) => void;
    selectedPreprocessorId: string | null;
    setSelectedPreprocessorId: (preprocessorId: string | null) => void;
    getPreprocessorById: (preprocessorId: string) => PreprocessorClipProps | null;
    addPreprocessorToClip: (clipId: string, preprocessor: PreprocessorClipProps) => void;
    removePreprocessorFromClip: (clipId: string, preprocessor: PreprocessorClipProps | string) => void;
    getPreprocessorsForClip: (clipId: string) => PreprocessorClipProps[];
    getClipFromPreprocessorId: (preprocessorId: string) => PreprocessorClipType | null;
    // Global snap guideline (absolute stage X in px). Null when inactive
    snapGuideX: number | null;
    setSnapGuideX: (x: number | null) => void;
    _updateZoomLevel: (clips:AnyClipProps[], clipDuration: number) => void;
    getTimelinePosition: (timelineId: string, scrollY?: number) => {top: number, bottom: number, left: number, right: number};
    getClipPosition: (clipId: string, scrollY?: number) => {top: number, bottom: number, left: number, right: number};
    // Timelines
    timelines: TimelineProps[];
    getClipsForTimeline: (timelineId: string) => AnyClipProps[];
    getTimelineById: (timelineId: string) => TimelineProps | undefined;
    setTimelines: (timelines: TimelineProps[]) => void;
    addTimeline: (timeline: Partial<TimelineProps>, index?: number) => void;
    removeTimeline: (timelineId: string) => void;
    updateTimeline: (timelineId: string, timelineToUpdate: Partial<TimelineProps>) => void;
    clipWithinFrame: (clip: AnyClipProps, frame: number, overlap?:boolean, overlapAmount?:number) => boolean;
    updateMaskKeyframes: (clipId: string, maskId: string, updater: (keyframes: Map<number, MaskData> | Record<number, MaskData>) => Map<number, MaskData> | Record<number, MaskData>) => void;
}

// Helper function to calculate total duration of all clips
const calculateTotalClipDuration = (clips: AnyClipProps[]): number => {
    const maxEndFrame = Math.max(...clips.map(clip => clip.endFrame || 0));
    return maxEndFrame;
};

export const isValidTimelineForClip = (timeline: TimelineProps, clip: AnyClipProps | MediaItem | string | Preprocessor) => {
    if (typeof clip === 'string') 
        clip = {type:clip} as AnyClipProps;
    if (timeline.type === 'media') {
        return clip.type === 'video' || clip.type === 'image';
    }
    return timeline.type === clip.type;
}

export const getTimelineTypeForClip = (clip: AnyClipProps | MediaItem | string):TimelineType => {
    if (typeof clip === 'string') 
        clip = {type:clip} as AnyClipProps;
    if (clip.type === 'video' || clip.type === 'image') {
        return 'media';
    }
    return clip.type;
}

export const getTimelineHeightForClip = (clip: AnyClipProps | MediaItem | string):number => {
    if (typeof clip === 'string') 
        clip = {type:clip} as AnyClipProps;
    if (clip.type === 'video' || clip.type === 'image'  ||(clip.type as any === 'media')) {
        return 72;
    } 
    if (clip.type === 'audio') {
        return 54;
    }
    return 36;
}



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
        const currentTimelineId = currentClip.timelineId || '';
        
        // Check for overlap with previous clip
        if (resolvedClips.length > 0) {
            const previousClip = resolvedClips[resolvedClips.length - 1];
            const previousEnd = previousClip.endFrame || 0;
            
            // If current clip overlaps with previous clip, shift it to start after previous clip ends
            if (currentStart < previousEnd && currentTimelineId === previousClip.timelineId) {
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

export const resolveOverlapsTimelines = (timelines: TimelineProps[]): TimelineProps[] => {
    if (timelines.length === 0) return timelines;
    
    const resolvedTimelines: TimelineProps[] = [];
    
    for (let i = 0; i < timelines.length; i++) {
        const timeline = { ...timelines[i] };

        // First timeline starts at 0
        if (i === 0) {
            timeline.timelineY = 0;
        } else {
            // Each subsequent timeline is positioned based on the previous timeline's position + height
            const previousTimeline = resolvedTimelines[i - 1];
            const previousY = previousTimeline.timelineY || 0;
            const previousHeight = previousTimeline.timelineHeight || 54;
            timeline.timelineY = previousY + previousHeight;
        }
        
        resolvedTimelines.push(timeline);
    }
    
    return resolvedTimelines;
}

export const useClipStore = create<ClipStore>((set, get) => ({       
    clipDuration: 0,
    _setClipDuration: (duration) => set({ clipDuration: duration }),
    clips: [],  
    timelines: [],
    isTimelineMuted: (timelineId) => get().timelines.find((timeline) => timeline.timelineId === timelineId)?.muted || false,
    isTimelineHidden: (timelineId) => get().timelines.find((timeline) => timeline.timelineId === timelineId)?.hidden || false,
    muteTimeline: (timelineId) => set((state) => {
        const newTimelines = state.timelines.map((timeline) => timeline.timelineId === timelineId ? { ...timeline, muted: true } : timeline);
        const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
        return { timelines: resolvedTimelines };   
    }),
    unmuteTimeline: (timelineId) => set((state) => {
        const newTimelines = state.timelines.map((timeline) => timeline.timelineId === timelineId ? { ...timeline, muted: false } : timeline);
        const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
        return { timelines: resolvedTimelines };
        return state;
    }),
    hideTimeline: (timelineId) => set((state) => {
        const newTimelines = state.timelines.map((timeline) => timeline.timelineId === timelineId ? { ...timeline, hidden: true } : timeline);
        const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
        return { timelines: resolvedTimelines };
        return state;
    }),
    unhideTimeline: (timelineId) => set((state) => {
        const newTimelines = state.timelines.map((timeline) => timeline.timelineId === timelineId ? { ...timeline, hidden: false } : timeline);
        const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
        return { timelines: resolvedTimelines };
        return state;
    }),
    ghostStartEndFrame: [0, 0],
    activeMediaItem: null,
    setActiveMediaItem: (mediaItem) => set({ activeMediaItem: mediaItem }),
    setGhostStartEndFrame: (startFrame, endFrame) => set({ ghostStartEndFrame: [startFrame, endFrame] }),
    ghostX: 0,
    _updateZoomLevel: (clips:AnyClipProps[], clipDuration: number) => {
        // Determine if this is the first clip being added (based on current store state before set)
        const hadNoClips = (get().clips || []).length === 0;

        // Longest clip length in frames (endFrame max)
        const longestClipFrames = Math.max(0, Math.round(clipDuration || 0));

        // Compute new baseline total timeline frames at zoom level 1 so that
        // the longest clip occupies 60% of the timeline (5/3 multiplier).
        // Clamp to at least MIN_DURATION to ensure we can always zoom to 5 frames.
        const newTotalFrames = Math.max(MIN_DURATION, Math.round((longestClipFrames * 5) / 3));

        const controls = useControlsStore.getState();
        const prevTotalFrames = Math.max(0, controls.totalTimelineFrames || 0);
        const minZoomLevel = controls.minZoomLevel;
        const maxZoomLevel = controls.maxZoomLevel;
        const [currentStart, currentEnd] = controls.timelineDuration || [0, newTotalFrames];
        const currentWidth = Math.max(1, currentEnd - currentStart);

        const needsBaselineUpdate = newTotalFrames !== prevTotalFrames && clips.length > 0;

        if (!needsBaselineUpdate) {
            return;
        }

        // Apply the new baseline total frames
        useControlsStore.setState({ totalTimelineFrames: newTotalFrames });

        if (hadNoClips) {
            // First clip: set zoom level 1 window to exactly the baseline width,
            // so the clip takes 60% of the timeline at zoom level 1.
            controls.setTimelineDuration(0, newTotalFrames);
            controls.setZoomLevel(1 as any);
            return;
        }

        // For subsequent changes (e.g., adding a longer clip), keep the current
        // window to avoid jitter, but recalibrate the zoom level so that the
        // current window width maps to the nearest zoom step under the new baseline.
        const steps = Math.max(1, maxZoomLevel - minZoomLevel);
        const ratio = MIN_DURATION / newTotalFrames;
        const durations: number[] = new Array(steps + 1).fill(0).map((_, i) => {
            const ti = i / steps;
            const d = Math.round(newTotalFrames * Math.pow(ratio, ti));
            return Math.max(MIN_DURATION, Math.min(newTotalFrames, d));
        });

        // Keep the same window; clamp within new total if needed
        let clampedStart = currentStart;
        let clampedEnd = currentEnd;
        const width = Math.max(1, clampedEnd - clampedStart);
        if (clampedEnd > newTotalFrames) {
            clampedStart = Math.max(0, newTotalFrames - width);
            clampedEnd = clampedStart + width;
            controls.setTimelineDuration(clampedStart, clampedEnd);
        }

        // Pick the zoom level whose target duration is closest to the current width
        let bestIndex = 0;
        let bestDiff = Number.POSITIVE_INFINITY;
        for (let i = 0; i <= steps; i++) {
            const diff = Math.abs(durations[i] - currentWidth);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIndex = i;
            }
        }
        const newZoomLevel = (minZoomLevel + bestIndex) as any;
        controls.setZoomLevel(newZoomLevel);
    },
    setGhostX: (x) => set({ ghostX: x }),
    selectedPreprocessorId: null,
    setSelectedPreprocessorId: (preprocessorId) => {
        // control store set clip id to empty
        if (preprocessorId) {
        useControlsStore.setState({ selectedClipIds: [] });
        }
        set({ selectedPreprocessorId: preprocessorId});
    },
    ghostGuideLines: null,
    setGhostGuideLines: (guideLines) => set({ ghostGuideLines: guideLines }),
    hoveredTimelineId: null,
    setHoveredTimelineId: (timelineId) => set({ hoveredTimelineId: timelineId }),
    ghostTimelineId: null,
    setGhostTimelineId: (timelineId) => set({ ghostTimelineId: timelineId }),
    ghostInStage: false,
    setGhostInStage: (inStage) => set({ ghostInStage: inStage }),
    draggingClipId: null,
    setDraggingClipId: (clipId) => set({ draggingClipId: clipId }),
    isDragging: false,
    setIsDragging: (isDragging) => set({ isDragging }),
    snapGuideX: null,
    setSnapGuideX: (x) => set({ snapGuideX: x }),
    getClipTransform: (clipId: string) => {
        const clip = get().clips.find((c) => c.clipId === clipId);
        return clip?.transform;
    },
    setClipTransform: (clipId: string, transform: Partial<ClipTransform>) => set((state) => {
        const index = state.clips.findIndex((c) => c.clipId === clipId);
        if (index === -1) return { clips: state.clips };

        const current = state.clips[index];
        const hadTransform = !!current.transform;
        const previous: ClipTransform = current.transform || { x: 0, y: 0, width: 0, height: 0, scaleX: 1, scaleY: 1, rotation: 0, cornerRadius: 0, opacity: 100 };
        const next: ClipTransform = { ...previous, ...transform };

        const updatedClip: AnyClipProps = { ...current, transform: next };

        if (!current.originalTransform) {
            updatedClip.originalTransform = { ...next };
        }

        if ((current.type === 'video' || current.type === 'image') && Array.isArray((current as VideoClipProps | ImageClipProps).masks)) {
            const currentMasks = (current as VideoClipProps | ImageClipProps).masks;
            if (currentMasks.length > 0) {
                const remappedMasks = currentMasks.map((mask) => {
                    const baseTransform = mask.transform ?? previous;
                    if (!hadTransform) {
                        return { ...mask, transform: { ...next } };
                    }
                    return remapMaskWithClipTransform(mask, baseTransform, next);
                });
                (updatedClip as VideoClipProps | ImageClipProps).masks = remappedMasks;
            }
        }

        const newClips = [...state.clips];
        newClips[index] = updatedClip;
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    getPreprocessorById: (preprocessorId: string) => {
        // get the clip that contains the preprocessor
        const clip = get().clips.find((c) => (c.type === 'video' || c.type === 'image') && c.preprocessors?.some((p) => p.id === preprocessorId));
        if (!clip) return null;
        const preprocessor = (clip as VideoClipProps | ImageClipProps).preprocessors?.find((p) => p.id === preprocessorId);
        return preprocessor || null;
    },
    getClipFromPreprocessorId: (preprocessorId: string) => {
        const clip = get().clips.find((c) => (c.type === 'video' || c.type === 'image') && c.preprocessors?.some((p) => p.id === preprocessorId)) as PreprocessorClipType | null;
        return clip || null;
    },
    updatePreprocessor: (clipId: string, preprocessorId: string, preprocessorToUpdate: Partial<PreprocessorClipProps>) => set((state) => {
        const newClips = [...state.clips];
        const currentClipIndex = newClips.findIndex((c) => c.clipId === clipId);

        if (currentClipIndex === -1) return { clips: state.clips };
        const currentClip = newClips[currentClipIndex];
        if (!currentClip) return { clips: state.clips };
        if (currentClip.type !== 'video' && currentClip.type !== 'image') return { clips: state.clips };
        const newPreprocessors = currentClip.preprocessors?.map((p) => p.id === preprocessorId ? { ...p, ...preprocessorToUpdate } : p) || [];
        const newClip = { ...currentClip, preprocessors: newPreprocessors } as AnyClipProps;
        newClips[currentClipIndex] = newClip;
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    addPreprocessorToClip: (clipId: string, preprocessor: PreprocessorClipProps) => set((state) => {
       const currentClipIndex = state.clips.findIndex((c) => c.clipId === clipId);
       const currentClip = state.clips[currentClipIndex];
       if (!currentClip) return { clips: state.clips };
       const newClips = [...state.clips];
       if (currentClip.type !== 'video' && currentClip.type !== 'image') return { clips: state.clips };
       const newPreprocessors = [...(currentClip.preprocessors || []), preprocessor];
       const newClip = { ...currentClip, preprocessors: newPreprocessors } as AnyClipProps;
       newClips[currentClipIndex] = newClip;
       const resolvedClips = resolveOverlaps(newClips);
       const clipDuration = calculateTotalClipDuration(resolvedClips);
       return { clips: resolvedClips, clipDuration };
    }),
    removePreprocessorFromClip: (clipId: string, preprocessor: PreprocessorClipProps | string) => set((state) => {
        const newClips = [...state.clips];
        const currentClipIndex = newClips.findIndex((c) => c.clipId === clipId);
        if (currentClipIndex === -1) return { clips: state.clips };
        const currentClip = newClips[currentClipIndex];
        if (!currentClip) return { clips: state.clips };
        if (currentClip.type !== 'video' && currentClip.type !== 'image') return { clips: state.clips };
        const newPreprocessors = currentClip.preprocessors?.filter((p) => p.id !== (typeof preprocessor === 'string' ? preprocessor : preprocessor.id)) || [];
        const newClip = { ...currentClip, preprocessors: newPreprocessors } as AnyClipProps;
        newClips[currentClipIndex] = newClip;
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        let selectedPreprocessorId = state.selectedPreprocessorId;
        if (selectedPreprocessorId === (typeof preprocessor === 'string' ? preprocessor : preprocessor.id)) {
            selectedPreprocessorId = null;
        }
        return { clips: resolvedClips, clipDuration, selectedPreprocessorId };
    }),
    getPreprocessorsForClip: (clipId: string) => {
        const clip = get().clips.find((c) => c.clipId === clipId);
        if (!clip || (clip.type !== 'video' && clip.type !== 'image')) return [];
        return clip.preprocessors || [];
    },
    getTimelineById: (timelineId: string) => get().timelines.find((timeline) => timeline.timelineId === timelineId),
    setTimelines: (timelines: TimelineProps[]) => set({ timelines }),
    addTimeline: (timeline: Partial<TimelineProps>, index?: number) => set((state) => {
        const newTimeline: TimelineProps = {
            timelineId: timeline.timelineId ?? uuidv4(),
            timelineHeight: timeline.timelineHeight ?? 54,
            timelineWidth: timeline.timelineWidth ?? 0,
            timelineY: timeline.timelineY ?? 0,
            timelinePadding: timeline.timelinePadding ?? 24,
            type: timeline.type ?? 'media',
            muted: timeline.muted ?? false,
            hidden: timeline.hidden ?? false,
        };
        const timelines = index !== undefined ? [...state.timelines.slice(0, index + 1), newTimeline, ...state.timelines.slice(index + 1)] : [...state.timelines, newTimeline];
        const resolvedTimelines = resolveOverlapsTimelines(timelines);
        return { timelines: resolvedTimelines };
    }),
    removeTimeline: (timelineId: string) => set((state) => {
        const newTimelines = state.timelines.filter((timeline) => timeline.timelineId !== timelineId);
        const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
        return { timelines: resolvedTimelines };
    }),
    updateTimeline: (timelineId: string, timelineToUpdate: Partial<TimelineProps>) => set((state) => {
        const newTimelines = state.timelines.map((timeline) => timeline.timelineId === timelineId ? { ...timeline, ...timelineToUpdate } : timeline);
        const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
        return { timelines: resolvedTimelines };
    }),
    clipboard: [],
    getClipsForTimeline: (timelineId: string) => get().clips.filter((clip) => clip.timelineId === timelineId) || [],
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
        get()._updateZoomLevel(resolvedClips, clipDuration);
        return { clips: resolvedClips, clipDuration };
    }),
    removeClip: (clipId: string) => set((state) => {
        const newClips = state.clips.filter((clip) => clip.clipId !== clipId);
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        get()._updateZoomLevel(resolvedClips, clipDuration);
        return { clips: resolvedClips, clipDuration };
    }),
    updateClip: (clipId: string, clipToUpdate: Partial<AnyClipProps>) => set((state) => {
        const index = state.clips.findIndex((c) => c.clipId === clipId);
        if (index === -1) {
            return { clips: state.clips };
        }
        const current = state.clips[index];

        // Prepare update payload and handle speed-induced duration rescale
        let nextUpdate: Partial<AnyClipProps> = { ...clipToUpdate };
        if (Object.prototype.hasOwnProperty.call(clipToUpdate, 'speed')) {
            const oldSpeed = Math.max(0.1, Number((current as VideoClipProps).speed || 1));
            const newSpeedRaw = Number((clipToUpdate as any).speed);
            const newSpeed = Math.max(0.1, Math.min(5, Number.isFinite(newSpeedRaw) ? newSpeedRaw : 1));

            // Anchor at left edge: keep startFrame, adjust endFrame to preserve source coverage
            const start = Math.max(0, Number(current.startFrame || 0));
            const end = Math.max(start + 1, Number(current.endFrame || (start + 1)));
            const oldDuration = Math.max(1, end - start);
            const newDuration = Math.max(1, Math.round(oldDuration * (oldSpeed / newSpeed)));
            nextUpdate.endFrame = start + newDuration;
            (nextUpdate as VideoClipProps).speed = newSpeed as any;

            // Update preprocessors timing to match the new speed
            if ((current.type === 'video' || current.type === 'image') && current.preprocessors && current.preprocessors.length > 0) {
                const speedRatio = oldSpeed / newSpeed;
                const clipStart = start;
                const updatedPreprocessors = current.preprocessors.map(preprocessor => {
                    const preprocessorStart = Math.max(0, Number(preprocessor.startFrame || 0));
                    const preprocessorEnd = Math.max(preprocessorStart + 1, Number(preprocessor.endFrame || (preprocessorStart + 1)));
                    
                    // Calculate relative positions from clip start
                    const relativeStart = preprocessorStart - clipStart;
                    const relativeEnd = preprocessorEnd - clipStart;
                    
                    // Scale the relative positions
                    const newRelativeStart = Math.round(relativeStart * speedRatio);
                    const newRelativeEnd = Math.round(relativeEnd * speedRatio);
                    
                    return {
                        ...preprocessor,
                        startFrame: clipStart + newRelativeStart,
                        endFrame: clipStart + newRelativeEnd
                    };
                });
                (nextUpdate as VideoClipProps | ImageClipProps).preprocessors = updatedPreprocessors;
            }
        }

        const updatedClip = { ...current, ...nextUpdate } as AnyClipProps;
        const newClips = [...state.clips];
        newClips[index] = updatedClip;

        const resolvedClips = resolveOverlaps(newClips as AnyClipProps[]);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        // update the zoom level
        get()._updateZoomLevel(resolvedClips, clipDuration);
        return { clips: resolvedClips, clipDuration };
    }),
    separateClip: (clipId) => set((state) => {
        const clip = state.clips.find((c) => c.clipId === clipId);
        if (!clip || clip.type !== 'video') return { clips: state.clips };
        const newClipId1 = uuidv4();
        const newClipId2 = uuidv4();
        const newAudioTimelineId = uuidv4();

        // Find the index of the current timeline
        const currentTimelineIndex = state.timelines.findIndex(t => t.timelineId === clip.timelineId);
        const clipTimeline = state.getTimelineById(clip.timelineId || '');
        
        // Create a new audio timeline for the separated audio clip
        const audioTimeline: TimelineProps = {
            timelineId: newAudioTimelineId,
            type: 'audio',
            timelineHeight: getTimelineHeightForClip('audio'),
            timelineWidth: clipTimeline?.timelineWidth ?? 0,
            timelineY: (clipTimeline?.timelineY ?? 0) + (clipTimeline?.timelineHeight ?? 54),
            timelinePadding: clipTimeline?.timelinePadding ?? 0,
            muted: false,
            hidden: false,
        };

        const url1 = new URL(clip.src);
        const url2 = new URL(clip.src);

        url1.hash = 'video';
        url2.hash = 'audio';

        const clipVideo: AnyClipProps = {
            ...clip,
            src: url1.toString(),
            clipId: newClipId1,
        };
        const clipAudio: AnyClipProps = {
            ...clip,
            type: 'audio',
            src: url2.toString(),
            clipId: newClipId2,
            timelineId: newAudioTimelineId,
        };
        // Async run these two commands
        getMediaInfo(clipVideo.src);
        getMediaInfo(clipAudio.src);
        
        // Remove the original clip and add both new clips
        const newClips = [...state.clips.filter((c) => c.clipId !== clipId), clipVideo, clipAudio];
        
        // Insert the new audio timeline directly below the current timeline
        const newTimelines = currentTimelineIndex !== -1 
            ? [...state.timelines.slice(0, currentTimelineIndex + 1), audioTimeline, ...state.timelines.slice(currentTimelineIndex + 1)]
            : [...state.timelines, audioTimeline];
        const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
        
        // Resolve overlaps and calculate duration
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        
        return { clips: resolvedClips, clipDuration, timelines: resolvedTimelines };
    }),
    isValidResize: (clipId: string, side: 'left' | 'right', newFrame: number) => {
        const state = get();
        const sortedClips = [...state.clips].sort((a, b) => (a.startFrame || 0) - (b.startFrame || 0));
        const currentIndex = sortedClips.findIndex(c => c.clipId === clipId);
        if (currentIndex === -1) return false;
        
        const currentClip = sortedClips[currentIndex];

        if (side === 'right') {
            // Resize right edge - adjust current clip's end and shift all clips after it
            const oldEndFrame = currentClip.endFrame || 0;
            const newEndFrame = Math.max((currentClip.startFrame || 0) + 1, newFrame);
            const frameDelta = newEndFrame - oldEndFrame;

            if (frameDelta + (currentClip.framesToGiveEnd || 0) > 0) {
                return false;
            }
        } else if (side === 'left') {
            // Resize left edge - adjust current clip's start and shift all clips before it
            const oldStartFrame = currentClip.startFrame || 0;
            const newStartFrame = Math.min((currentClip.endFrame || 0) - 1, newFrame);
            let frameDelta = newStartFrame - oldStartFrame;

            if (frameDelta + (currentClip.framesToGiveStart || 0) < 0) {
                return false;
            }
        }
        return true;
    },
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
        get()._updateZoomLevel(resolvedClips, clipDuration);
        return { clips: resolvedClips, clipDuration };
    }),
    updateMaskKeyframes: (clipId, maskId, updater) => set((state) => {
        const newClips = [...state.clips];
        const clipIndex = newClips.findIndex((c) => c.clipId === clipId && (c.type === 'video' || c.type === 'image'));
        if (clipIndex === -1) return { clips: state.clips };

        const clip = newClips[clipIndex] as VideoClipProps | ImageClipProps;
        const masks = [...(clip.masks || [])];
        const maskIndex = masks.findIndex((m) => m.id === maskId);
        if (maskIndex === -1) return { clips: state.clips };

        const targetMask = masks[maskIndex];
        const updatedKeyframes = updater(targetMask.keyframes as Map<number, MaskData> | Record<number, MaskData>);

        masks[maskIndex] = {
            ...targetMask,
            keyframes: updatedKeyframes,
            lastModified: Date.now(),
        } as MaskClipProps;

        newClips[clipIndex] = {
            ...clip,
            masks,
        } as AnyClipProps;

        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        return { clips: resolvedClips, clipDuration };
    }),
    // Create two new clips from the original clip at the cut frame
    splitClip: (cutFrame: number, clipId: string) => set((state) => {
        // Find the clip that contains the cut frame
        const clip = state.clips.find((clip) => {
            const startFrame = clip.startFrame || 0;
            const endFrame = clip.endFrame || 0;
            return cutFrame > startFrame && cutFrame < endFrame && clip.clipId === clipId;
        });
        
        if (!clip) return { clips: state.clips };
        
        // remove the clip from the array 
        const filteredClips = state.clips.filter((c) => c.clipId !== clip.clipId);
        
        // create new clip ids
        const newClipId1 = uuidv4();
        const newClipId2 = uuidv4();
        const infinityFramestoGiveEnd = !isFinite(clip.framesToGiveEnd || 0);
        const infinityFramestoGiveStart = !isFinite(clip.framesToGiveStart || 0);

        // First clip: from original start to cut frame
        // Keeps original framesToGiveStart, but can't extend past cut
        const newClip1: AnyClipProps = { 
            ...clip, 
            endFrame: cutFrame, 
            clipId: newClipId1,
            framesToGiveStart: infinityFramestoGiveStart ? Infinity : 0,
            framesToGiveEnd: infinityFramestoGiveEnd ? -Infinity : 0,
        };
        
        // Second clip: from cut frame to original end
        // Maintains proper media offset, but can't extend before cut
        const newClip2: AnyClipProps = { 
            ...clip, 
            startFrame: cutFrame, 
            clipId: newClipId2,
            framesToGiveStart: infinityFramestoGiveStart ? Infinity : 0,
            framesToGiveEnd: infinityFramestoGiveEnd ? -Infinity : 0,
        };
        
        if (clip.type === 'image' || clip.type === 'video') { 
            if (clip.preprocessors && clip.preprocessors.length > 0) {
                const clip1Preprocessors: PreprocessorClipProps[] = [];
                const clip2Preprocessors: PreprocessorClipProps[] = [];
                clip.preprocessors.forEach(preprocessor => {
                    // check if preprocessor is within the new clip
                    if (preprocessor.startFrame !== undefined && preprocessor.endFrame !== undefined) {
                        // Case 1: Preprocessor is entirely before the cut frame
                        if (preprocessor.endFrame <= cutFrame) {
                            clip1Preprocessors.push(preprocessor);
                        }
                        // Case 2: Preprocessor is entirely after the cut frame
                        else if (preprocessor.startFrame > cutFrame) {
                            clip2Preprocessors.push({
                                ...preprocessor,
                                startFrame: preprocessor.startFrame - cutFrame,
                                endFrame: preprocessor.endFrame - cutFrame,
                            });
                        }
                        // Case 3: Preprocessor spans across the cut frame
                        else if (preprocessor.startFrame < cutFrame && preprocessor.endFrame > cutFrame) {
                            // Calculate how much of the preprocessor media is used before/after cut
                            const preprocessorMediaDurationBeforeCut = cutFrame - preprocessor.startFrame;
                            const preprocessorMediaDurationAfterCut = preprocessor.endFrame - cutFrame;
                            
                            const preprocessor1 = {
                                ...preprocessor,
                                endFrame: cutFrame
                            };
                            
                            const preprocessor2 = {
                                ...preprocessor,
                                id: uuidv4(),
                                startFrame: 0,
                                endFrame: preprocessorMediaDurationAfterCut,
                            };
                            
                            // Update src URLs to reflect the split in preprocessor media
                            if (preprocessor.src) {
                                const url1 = new URL(preprocessor.src);
                                const url2 = new URL(preprocessor.src);
                                const currentStartFrame = url1.searchParams.get('startFrame') ? Number(url1.searchParams.get('startFrame')) : 0;
                                const currentEndFrame = url1.searchParams.get('endFrame') ? Number(url1.searchParams.get('endFrame')) : undefined;
                                
                                // First preprocessor: from start to cutFrame in preprocessor media
                                url1.searchParams.set('startFrame', String(currentStartFrame));
                                url1.searchParams.set('endFrame', String(currentStartFrame + preprocessorMediaDurationBeforeCut));
                                
                                // Second preprocessor: from cutFrame to end in preprocessor media
                                url2.searchParams.set('startFrame', String(currentStartFrame + preprocessorMediaDurationBeforeCut));
                                if (currentEndFrame !== undefined) {
                                    url2.searchParams.set('endFrame', String(currentEndFrame));
                                } else {
                                    url2.searchParams.set('endFrame', String(currentStartFrame + preprocessorMediaDurationBeforeCut + preprocessorMediaDurationAfterCut));
                                }

                                preprocessor1.src = url1.toString();
                                preprocessor2.src = url2.toString();
                                
                                void getMediaInfo(preprocessor1.src);
                                void getMediaInfo(preprocessor2.src);
                            }
                            
                            clip1Preprocessors.push(preprocessor1);
                            clip2Preprocessors.push(preprocessor2);

                        }
                    }
                });
                
                (newClip1 as VideoClipProps | ImageClipProps).preprocessors = clip1Preprocessors;
                (newClip2 as VideoClipProps | ImageClipProps).preprocessors = clip2Preprocessors;
            }
            // Split masks and rebase keyframes for each new clip
            if ((clip as VideoClipProps | ImageClipProps).masks && (clip as VideoClipProps | ImageClipProps).masks.length > 0) {
                const originalMasks = (clip as VideoClipProps | ImageClipProps).masks;

                const masksForClip1: MaskClipProps[] = [];
                const masksForClip2: MaskClipProps[] = [];

                // Compute local cut position relative to original clip start
                const startFrame = clip.startFrame ?? 0;
                const framesToGiveStart = isFinite(clip.framesToGiveStart ?? 0) ? (clip.framesToGiveStart ?? 0) : 0;
                const realStartFrame = startFrame + framesToGiveStart;
                const cutLocal = Math.max(0, cutFrame - realStartFrame);

                const makeEmptyKeyframesLike = (kf: Map<number, MaskData> | Record<number, MaskData>) =>
                    (kf instanceof Map ? new Map<number, MaskData>() : {} as Record<number, MaskData>);

                const setKeyframe = (
                    target: Map<number, MaskData> | Record<number, MaskData>,
                    frame: number,
                    data: MaskData
                ) => {
                    if (target instanceof Map) {
                        target.set(frame, data);
                    } else {
                        (target as Record<number, MaskData>)[frame] = data;
                    }
                };

                const getKeyframe = (
                    source: Map<number, MaskData> | Record<number, MaskData>,
                    frame: number
                ): MaskData | undefined => {
                    return source instanceof Map ? source.get(frame) : (source as Record<number, MaskData>)[frame];
                };

                const getSortedFrames = (kf: Map<number, MaskData> | Record<number, MaskData>): number[] =>
                    (kf instanceof Map ? Array.from(kf.keys()) : Object.keys(kf).map(Number)).sort((a, b) => a - b);

                for (const mask of originalMasks) {
                    const keyframes = mask.keyframes;
                    const frames = getSortedFrames(keyframes);

                    // Image clips only use frame 0; copy that to both sides
                    if (clip.type === 'image') {
                        const dataAt0 = getKeyframe(keyframes, 0) ?? (frames.length > 0 ? getKeyframe(keyframes, frames[0]) : undefined);
                        const now = Date.now();
                        if (dataAt0) {
                            const kf1 = makeEmptyKeyframesLike(keyframes);
                            const kf2 = makeEmptyKeyframesLike(keyframes);
                            setKeyframe(kf1, 0, { ...dataAt0 });
                            setKeyframe(kf2, 0, { ...dataAt0 });
                            masksForClip1.push({ ...mask, keyframes: kf1, clipId: newClip1.clipId, lastModified: now });
                            masksForClip2.push({ ...mask, keyframes: kf2, clipId: newClip2.clipId, lastModified: now });
                        }
                        continue;
                    }

                    // Video clips: split around cutLocal; left keeps < cutLocal, right keeps >= cutLocal rebased to start at 0
                    const kfLeft = makeEmptyKeyframesLike(keyframes);
                    const kfRight = makeEmptyKeyframesLike(keyframes);

                    for (const f of frames) {
                        const data = getKeyframe(keyframes, f);
                        if (!data) continue;
                        if (f < cutLocal) {
                            setKeyframe(kfLeft, f, { ...data });
                        } else {
                            const rebased = Math.max(0, f - cutLocal);
                            setKeyframe(kfRight, rebased, { ...data });
                        }
                    }

                    const now = Date.now();
                    masksForClip1.push({ ...mask, keyframes: kfLeft, clipId: newClip1.clipId, lastModified: now });
                    masksForClip2.push({ ...mask, keyframes: kfRight, clipId: newClip2.clipId, lastModified: now });
                }

                (newClip1 as VideoClipProps | ImageClipProps).masks = masksForClip1;
                (newClip2 as VideoClipProps | ImageClipProps).masks = masksForClip2;
            }
        }

        if (Object.prototype.hasOwnProperty.call(clip, 'src') && clip.src && (AUDIO_EXTS.includes(getLowercaseExtension(clip.src)) || VIDEO_EXTS.includes(getLowercaseExtension(clip.src)))) {
            // one of audio or video
            let speed = 1;
            if (Object.prototype.hasOwnProperty.call(clip, 'speed')) {
                speed = (clip as AudioClipProps).speed || 1;
            }
            const frameShift = (clip.startFrame || 0) - (clip.framesToGiveStart || 0);
            const url1 = new URL(clip.src);
            const url2 = new URL(clip.src);
            const startFrame1 = (clip.startFrame || 0) - frameShift;
            const endFrame1 = (cutFrame * speed) - frameShift;
            const startFrame2 = (cutFrame * speed) - frameShift;
            const endFrame2 = ((clip.endFrame || 0) * speed) - frameShift;
            const currentStartFrame = url1.searchParams.get('startFrame') ? Number(url1.searchParams.get('startFrame')) : 0;    
            url1.searchParams.set('startFrame', String(startFrame1 + currentStartFrame));
            url1.searchParams.set('endFrame', String(endFrame1 + currentStartFrame));
            url2.searchParams.set('startFrame', String(startFrame2 + currentStartFrame));
            url2.searchParams.set('endFrame', String(endFrame2 + currentStartFrame));
            newClip1.src = url1.toString();
            newClip2.src = url2.toString();
            void getMediaInfo(newClip1.src);
            void getMediaInfo(newClip2.src);
        }
        
        const newClips = [...filteredClips, newClip1, newClip2];
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        get()._updateZoomLevel(resolvedClips, clipDuration);
        const controlStore = useControlsStore.getState();
        controlStore.setSelectedClipIds([newClip1.clipId, newClip2.clipId]);
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
    pasteClips: (atFrame?: number, targetTimelineId?: string) => set((state) => {
        const clipboardItems = state.clipboard || [];
        if (clipboardItems.length === 0) return { clips: state.clips };
        const baseStart = Math.min(...clipboardItems.map(c => c.startFrame || 0));
        const insertionFrame = Math.max(0, Math.round(atFrame || 0));
        const newIds: string[] = [];
        // Determine the destination timeline per item: prefer provided target, otherwise a compatible existing timeline or create one
        const chooseTimelineFor = (template: AnyClipProps): string => {
            if (targetTimelineId) return targetTimelineId;
            const desiredType = getTimelineTypeForClip(template);
            const existing = state.timelines.find(t => t.type === desiredType);
            if (existing) return existing.timelineId;
            // If no matching timeline, create one beside last timeline
            const timelineId = uuidv4();
            const last = state.timelines[state.timelines.length - 1];
            const newTimeline: TimelineProps = {
                timelineId,
                type: desiredType,
                timelineHeight: getTimelineHeightForClip(desiredType),
                timelineWidth: last?.timelineWidth ?? 0,
                timelineY: (last?.timelineY ?? 0) + (last?.timelineHeight ?? 54),
                timelinePadding: last?.timelinePadding ?? 24,
                muted: false,
                hidden: false,
            };
            state.addTimeline(newTimeline);
            return timelineId;
        };
        const pasted = clipboardItems.map(template => {
            const templateStart = template.startFrame || 0;
            const templateEnd = template.endFrame || 0;
            const duration = Math.max(1, templateEnd - templateStart);
            const offset = templateStart - baseStart;
            const start = insertionFrame + offset;
            const end = start + duration;
            const newId = uuidv4();
            newIds.push(newId);
            const timelineId = chooseTimelineFor(template);
            return { ...template, clipId: newId, timelineId, startFrame: start, endFrame: end, framesToGiveEnd: 0, framesToGiveStart: 0 } as AnyClipProps;
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
    },
    clipWithinFrame: (clip: AnyClipProps, frame: number, overlap:boolean = false, overlapAmount:number = 0 ) => {
        if (overlap) {
            return frame >= (clip.startFrame || 0) - (overlapAmount || 0) && frame <= (clip.endFrame || 0);
        }
        return frame >= (clip.startFrame || 0) && frame <= (clip.endFrame || 0);
    },
    getTimelinePosition: (timelineId: string, scrollY?: number) => {
        const timeline = get().timelines.find((t) => t.timelineId === timelineId);
        if (!timeline) return {top: 0, bottom: 0, left: 0, right: 0};
        const top = (timeline.timelineY ?? 0) + 8 - (scrollY || 0);
        const bottom = top + (timeline.timelineHeight ?? 54);
        return {top, bottom, left: timeline.timelinePadding!, right: timeline.timelinePadding! + (timeline.timelineWidth ?? 0)};
    },
    getClipPosition: (clipId: string, scrollY?: number) => {
        const clip = get().clips.find((c) => c.clipId === clipId);
        if (!clip) return {top: 0, bottom: 0, left: 0, right: 0};
        
        const timeline = get().timelines.find((t) => t.timelineId === clip.timelineId);
        if (!timeline) return {top: 0, bottom: 0, left: 0, right: 0};
        
        const controls = useControlsStore.getState();
        const timelineDuration = controls.timelineDuration;
        
        const clipX = getClipX(clip.startFrame ?? 0, clip.endFrame ?? 0, timeline.timelineWidth ?? 0, timelineDuration);
        const clipWidth = getClipWidth(clip.startFrame ?? 0, clip.endFrame ?? 0, timeline.timelineWidth ?? 0, timelineDuration);
        
        const top = (timeline.timelineY ?? 0) + 8 - (scrollY || 0);
        const bottom = top + (timeline.timelineHeight ?? 54);
        const left = (timeline.timelinePadding ?? 0) + clipX;
        const right = left + clipWidth;
        
        return {top, bottom, left, right};
    },
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

export const getTimelineX = (timelineWidth:number, timelinePadding:number, timelineDuration:number[]) => {
    const [timelineStartFrame, timelineEndFrame] = timelineDuration;
    const timelineX = timelinePadding - (timelineWidth / (timelineEndFrame - timelineStartFrame)) * timelineStartFrame;
    return Math.max(0, timelineX);
}


export const getLocalFrame = (focusFrame: number, clip: AnyClipProps) => {
    const startFrame = clip.startFrame ?? 0;
    const framesToGiveStart = isFinite(clip.framesToGiveStart ?? 0) ? clip.framesToGiveStart ?? 0 : 0;
    const realStartFrame = startFrame + framesToGiveStart;
    return focusFrame - realStartFrame;
}

export const getGlobalFrame = (focusFrame: number, clip: AnyClipProps) => {
    const startFrame = clip.startFrame ?? 0;
    const framesToGiveStart = isFinite(clip.framesToGiveStart ?? 0) ? clip.framesToGiveStart ?? 0 : 0;
    const realStartFrame = startFrame + framesToGiveStart;
    return focusFrame + realStartFrame;
}
