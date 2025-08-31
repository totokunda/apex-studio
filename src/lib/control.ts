import { create } from "zustand";
import {  ZoomLevel } from "./types";


interface ControlStore {
    zoomLevel: ZoomLevel;
    setZoomLevel: (level: ZoomLevel) => void;
    totalTimelineFrames: number; // total frames in the timeline
    incrementTotalTimelineFrames: (duration: number) => void;
    decrementTotalTimelineFrames: (duration: number) => void;
    timelineDuration: [number, number]; // [startFrame, endFrame]
    setTimelineDuration: (startFrame: number, endFrame: number) => void;
    shiftTimelineDuration: (duration: number) => void;
    resetTimelineDuration: () => void;
    fps: number;
    setFps: (fps: number) => void;
    focusFrame: number;
    setFocusFrame: (frame: number) => void;
    focusAnchorRatio: number; // 0..1 position of scrubber within viewport width
    setFocusAnchorRatio: (ratio: number) => void;
    selectedClipIds: string[];
    setSelectedClipIds: (clipIds: string[]) => void;
    toggleClipSelection: (clipId: string, isShiftClick?: boolean) => void;
    clearSelection: () => void;
}

export const useControlsStore = create<ControlStore>((set) => ({
    // Zoom State
    zoomLevel: 1,
    setZoomLevel: (level) => set({ zoomLevel: level }),
    // Timeline State
    totalTimelineFrames: 1440, // total frames in the timeline
    incrementTotalTimelineFrames: (duration: number) => set((state) => {
        return { totalTimelineFrames: state.totalTimelineFrames + duration };
    }),
    decrementTotalTimelineFrames: (duration: number) => set((state) => {
        if (state.totalTimelineFrames - duration < 1440) return { totalTimelineFrames: 1440 };
        return { totalTimelineFrames: state.totalTimelineFrames - duration };
    }),
    timelineDuration: [0, 1440], // [startFrame, endFrame]
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
    resetTimelineDuration: () => set({ timelineDuration: [0, 1440] }),
    // FPS State
    fps: 16,
    setFps: (fps) => set({ fps: fps }),
    // Focus State
    focusFrame: 0,
    setFocusFrame: (frame) => set({ focusFrame: Math.max(0, Math.round(frame)) }),
    focusAnchorRatio: 0.5,
    setFocusAnchorRatio: (ratio) => set({ focusAnchorRatio: Math.max(0, Math.min(1, ratio)) }),
    selectedClipIds: [],
    setSelectedClipIds: (clipIds) => set({ selectedClipIds: clipIds }),
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
}));
