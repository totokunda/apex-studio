import { create } from "zustand";
import {  ZoomLevel } from "./types";
import { TIMELINE_DURATION_SECONDS, DEFAULT_FPS, MAX_DURATION } from "./settings";
import {useClipStore} from "./clip";
import _ from "lodash";

// Playback internals (module-scoped to avoid causing unnecessary rerenders)
let __playbackRafId: number | null = null;
let __lastTickMs = 0;
let __frameAccumulator = 0;

const __playbackTick = (now: number) => {
    const controls = useControlsStore.getState();
    const clips = useClipStore.getState();
    if (!controls.isPlaying) return;

    const fps = Math.max(1, controls.fps || 1);
    const totalFrames = Math.max(0, clips.clipDuration || 0);

    if (__lastTickMs === 0) {
        __lastTickMs = now;
        __playbackRafId = requestAnimationFrame(__playbackTick);
        return;
    }

    const dt = (now - __lastTickMs) / 1000;
    __frameAccumulator += dt * fps;
    let steps = Math.floor(__frameAccumulator);
    if (steps > 0) {
        __frameAccumulator -= steps;
        const current = controls.focusFrame || 0;
        const next = Math.min(totalFrames, current + steps);
        controls.setFocusFrame(next);
        if (next >= totalFrames) {
            controls.pause();
            __lastTickMs = 0;
            __frameAccumulator = 0;
            return;
        }
    }

    __lastTickMs = now;
    __playbackRafId = requestAnimationFrame(__playbackTick);
};

interface ControlStore {
    zoomLevel: ZoomLevel;
    setZoomLevel: (level: ZoomLevel) => void;
    totalTimelineFrames: number; // total frames in the timeline
    incrementTotalTimelineFrames: (duration: number) => void;
    decrementTotalTimelineFrames: (duration: number) => void;
    timelineDuration: [number, number]; // [startFrame, endFrame]
    setTimelineDuration: (startFrame: number, endFrame: number) => void;
    shiftTimelineDuration: (duration: number) => void;
    canTimelineDurationBeShifted: (duration: number) => boolean;
    resetTimelineDuration: () => void;
    fps: number;
    setFps: (fps: number) => void;
    focusFrame: number;
    setFocusFrame: (frame: number) => void;
    focusAnchorRatio: number; // 0..1 position of scrubber within viewport width
    setFocusAnchorRatio: (ratio: number) => void;
    selectedClipIds: string[];
    setSelectedClipIds: (clipIds: string[]) => void;
    addClipSelection: (clipId: string) => void;
    removeClipSelection: (clipId: string) => void;
    toggleClipSelection: (clipId: string, isShiftClick?: boolean) => void;
    clearSelection: () => void;
    play:() => void;
    pause:() => void;
    isPlaying: boolean;
    setIsPlaying: (isPlaying: boolean) => void;
}

export const useControlsStore = create<ControlStore>((set, get) => ({
    // Zoom State
    zoomLevel: 1,
    setZoomLevel: (level) => set({ zoomLevel: level }),
    // Timeline State
    totalTimelineFrames: TIMELINE_DURATION_SECONDS * DEFAULT_FPS, // total frames in the timeline
    incrementTotalTimelineFrames: (duration: number) => set((state) => {
        return { totalTimelineFrames: state.totalTimelineFrames + duration };
    }),
    decrementTotalTimelineFrames: (duration: number) => set((state) => {
        if (state.totalTimelineFrames - duration < MAX_DURATION) return { totalTimelineFrames: MAX_DURATION };
        return { totalTimelineFrames: state.totalTimelineFrames - duration };
    }),
    canTimelineDurationBeShifted: (duration: number) => {
        const [startFrame, endFrame] = get().timelineDuration;
        const requestedShift = Math.trunc(duration);
        const minShift = -startFrame; // cannot move left beyond 0
        const maxShift = get().totalTimelineFrames - endFrame; // cannot move right beyond total
        const clampedShift = Math.max(minShift, Math.min(maxShift, requestedShift));
        return clampedShift !== 0;
    },
    timelineDuration: [0, TIMELINE_DURATION_SECONDS * DEFAULT_FPS], // [startFrame, endFrame]
    setTimelineDuration: (startFrame: number, endFrame: number) => set({ timelineDuration: [startFrame, endFrame] }),
    shiftTimelineDuration: (duration: number) => set((state) => {
        const [startFrame, endFrame] = state.timelineDuration;
        const requestedShift = Math.trunc(duration);
        const minShift = -startFrame; // cannot move left beyond 0
        const maxShift = state.totalTimelineFrames - endFrame; // cannot move right beyond total
        const clampedShift = Math.max(minShift, Math.min(maxShift, requestedShift));
        if (clampedShift === 0) {
            return { timelineDuration: [startFrame, endFrame] };
        }
        const newStartFrame = startFrame + clampedShift;
        const newEndFrame = endFrame + clampedShift;
        return { timelineDuration: [newStartFrame, newEndFrame] };
    }),
    resetTimelineDuration: () => set({ timelineDuration: [0, TIMELINE_DURATION_SECONDS * DEFAULT_FPS] }),
    // FPS State
    fps: DEFAULT_FPS,
    setFps: (fps) => set({ fps: fps }),
    // Focus State
    focusFrame: 0,
    setFocusFrame: (frame) => set({ focusFrame: Math.max(0, Math.round(frame)) }),
    focusAnchorRatio: 0.5,
    setFocusAnchorRatio: (ratio) => set({ focusAnchorRatio: Math.max(0, Math.min(1, ratio)) }),
    selectedClipIds: [],
    setSelectedClipIds: (clipIds) => set({ selectedClipIds: clipIds }),
    addClipSelection: (clipId) => set((state) => {
        return { selectedClipIds: _.uniq([...state.selectedClipIds, clipId]) };
    }),
    removeClipSelection: (clipId) => set((state) => {
        return { selectedClipIds: state.selectedClipIds.filter(id => id !== clipId) };
    }),
    toggleClipSelection: (clipId, isShiftClick = false) => set((state) => {
        if (isShiftClick) {
            // Add to selection if shift-clicking
            if (state.selectedClipIds.includes(clipId)) {
                // Remove from selection if already selected
                return { selectedClipIds: state.selectedClipIds.filter(id => id !== clipId) };
            } else {
                // Add to selection
                return { selectedClipIds: [...state.selectedClipIds, clipId] };
            }
        } else {
            // Regular click - select only this clip
            return { selectedClipIds: [clipId] };
        }
    }),
    clearSelection: () => set({ selectedClipIds: [] }),
    play:() => {
        const state = get();
        const clips = useClipStore.getState();
        if (state.isPlaying) return;
        if (!clips || (clips.clips || []).length === 0) return;
        // If we're at or past the end, restart from the beginning
        if (state.focusFrame >= clips.clipDuration) {
            set({ focusFrame: 0 });
        }
        __lastTickMs = 0;
        __frameAccumulator = 0;
        if (__playbackRafId != null) cancelAnimationFrame(__playbackRafId);
        set({ isPlaying: true });
        __playbackRafId = requestAnimationFrame(__playbackTick);
    },
    pause:() => {
        if (__playbackRafId != null) {
            cancelAnimationFrame(__playbackRafId);
            __playbackRafId = null;
        }
        __lastTickMs = 0;
        __frameAccumulator = 0;
        set({ isPlaying: false });
    },
    isPlaying: false,
    setIsPlaying: (isPlaying) => set({ isPlaying }),
    }));
