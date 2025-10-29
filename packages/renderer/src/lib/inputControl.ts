import { create } from "zustand";
import { ZoomLevel, AnyClipProps } from "./types";
import { DEFAULT_FPS, TIMELINE_DURATION_SECONDS, MIN_DURATION } from "./settings";

interface InputControlStore {
    // Zoom state (independent from other controls)
    maxZoomLevel: ZoomLevel;
    setMaxZoomLevel: (level: ZoomLevel) => void;
    minZoomLevel: ZoomLevel;
    setMinZoomLevel: (level: ZoomLevel) => void;
    zoomLevel: ZoomLevel;
    setZoomLevel: (level: ZoomLevel) => void;

    // Timeline window (independent)
    totalTimelineFrames: number;
    setTotalTimelineFrames: (frames: number) => void;
    incrementTotalTimelineFrames: (frames: number) => void;
    decrementTotalTimelineFrames: (frames: number) => void;
    timelineDuration: [number, number];
    setTimelineDuration: (startFrame: number, endFrame: number) => void;
    shiftTimelineDuration: (delta: number) => void;

    // FPS
    fps: number;
    setFps: (fps: number) => void;

    // Focus state
    focusFrame: number;
    setFocusFrame: (frame: number) => void;
    focusAnchorRatio: number;
    setFocusAnchorRatio: (ratio: number) => void;

    // Optional selection for preview/input timelines
    selectedInputClipId: string | null;
    setSelectedInputClipId: (clipId: string | null) => void;
    clearSelectedInput: () => void;
    selectedInputChangeHandler: ((clipId: string | null) => void) | null;
    setSelectedInputChangeHandler: (handler: ((clipId: string | null) => void) | null) => void;
}

export const useInputControlsStore = create<InputControlStore>((set, get) => ({
    // Zoom
    maxZoomLevel: 10,
    setMaxZoomLevel: (level) => set({ maxZoomLevel: level }),
    minZoomLevel: 1,
    setMinZoomLevel: (level) => set({ minZoomLevel: level }),
    zoomLevel: 1,
    setZoomLevel: (level) => set({ zoomLevel: level }),

    // Timeline duration/window
    totalTimelineFrames: TIMELINE_DURATION_SECONDS * DEFAULT_FPS,
    setTotalTimelineFrames: (frames) => set({ totalTimelineFrames: Math.max(1, Math.round(frames)) }),
    incrementTotalTimelineFrames: (frames) => set((s) => ({ totalTimelineFrames: Math.max(1, s.totalTimelineFrames + Math.round(frames)) })),
    decrementTotalTimelineFrames: (frames) => set((s) => ({ totalTimelineFrames: Math.max(1, s.totalTimelineFrames - Math.round(frames)) })),
    timelineDuration: [0, TIMELINE_DURATION_SECONDS * DEFAULT_FPS],
    setTimelineDuration: (startFrame: number, endFrame: number) => {
        const s = Math.max(0, Math.round(startFrame));
        const e = Math.max(s + 1, Math.round(endFrame));
        set({ timelineDuration: [s, e] });
    },
    shiftTimelineDuration: (delta: number) => {
        const state = get();
        const [start, end] = state.timelineDuration;
        const total = Math.max(1, state.totalTimelineFrames);
        const win = Math.max(1, end - start);
        const desiredStart = Math.max(0, Math.min(total - win, start + Math.trunc(delta)));
        set({ timelineDuration: [desiredStart, desiredStart + win] });
    },

    // FPS
    fps: DEFAULT_FPS,
    setFps: (fps) => set({ fps: Math.max(1, Math.round(fps)) }),

    // Focus
    focusFrame: 0,
    setFocusFrame: (frame) => set({ focusFrame: Math.max(0, Math.round(frame)) }),
    focusAnchorRatio: 0.5,
    setFocusAnchorRatio: (ratio) => set({ focusAnchorRatio: Math.max(0, Math.min(1, ratio)) }),

    // Selection (optional)
    selectedInputClipId: null,
    setSelectedInputClipId: (clipId) => {
        set({ selectedInputClipId: clipId });
        try {
            const handler = get().selectedInputChangeHandler;
            if (typeof handler === 'function') handler(clipId);
        } catch {}
    },
    clearSelectedInput: () => set({ selectedInputClipId: null }),
    selectedInputChangeHandler: null,
    setSelectedInputChangeHandler: (handler) => set({ selectedInputChangeHandler: handler ?? null }),
}));

// Baseline zoom recalibration for input/preview timelines
export function updateInputZoomLevel(clips: AnyClipProps[], _clipDuration: number) {
    try {
        if (!Array.isArray(clips) || clips.length === 0) return;

        let longestClipFrames = 0;
        for (let i = 0; i < clips.length; i++) {
            const end = Math.max(0, clips[i].endFrame || 0);
            if (end > longestClipFrames) longestClipFrames = end;
        }

        const newTotalFrames = Math.max(MIN_DURATION, Math.round((longestClipFrames * 5) / 3));

        const controls = useInputControlsStore.getState();
        const prevTotalFrames = Math.max(0, controls.totalTimelineFrames || 0);
        if (newTotalFrames === prevTotalFrames) return;

        useInputControlsStore.setState({ totalTimelineFrames: newTotalFrames });
        controls.setTimelineDuration(0, newTotalFrames);
        controls.setZoomLevel(1 as any);
    } catch {}
}


