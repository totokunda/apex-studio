import { create } from "zustand";
import { ZoomLevel, AnyClipProps } from "./types";
import { DEFAULT_FPS, TIMELINE_DURATION_SECONDS, MIN_DURATION } from "./settings";
import { useClipStore } from "./clip";

interface InputControlStore {
    // Zoom state (independent from other controls)
    maxZoomLevel: ZoomLevel;
    setMaxZoomLevel: (level: ZoomLevel) => void;
    minZoomLevel: ZoomLevel;
    setMinZoomLevel: (level: ZoomLevel) => void;
    zoomLevel: ZoomLevel;
    setZoomLevel: (level: ZoomLevel, inputId?: string) => void;

    // Timeline window (independent)
    totalTimelineFrames: number;
    setTotalTimelineFrames: (frames: number, inputId?: string) => void;
    incrementTotalTimelineFrames: (frames: number, inputId?: string) => void;
    decrementTotalTimelineFrames: (frames: number, inputId?: string) => void;
    timelineDuration: [number, number];
    getTimelineDuration: (inputId?: string) => [number, number];
    setTimelineDuration: (startFrame: number, endFrame: number, inputId?: string) => void;
    shiftTimelineDuration: (delta: number, inputId?: string) => void;

    // FPS
    fps: number;
    setFps: (fps: number, inputId?: string) => void;
    getFps: (inputId?: string) => number;

    // Focus state
    focusFrame: number;
    setFocusFrame: (frame: number, inputId?: string) => void;
    getFocusFrame: (inputId?: string) => number;
    focusAnchorRatio: number;
    setFocusAnchorRatio: (ratio: number, inputId?: string) => void;
    getFocusAnchorRatio: (inputId?: string) => number;

    // Optional selection for preview/input timelines
    selectedInputClipId: string | null;
    setSelectedInputClipId: (clipId: string | null, inputId?: string) => void;
    clearSelectedInput: (inputId?: string) => void;
    getSelectedInputClipId: (inputId?: string) => string | null;
    selectedInputChangeHandler: ((clipId: string | null) => void) | null;
    setSelectedInputChangeHandler: (handler: ((clipId: string | null) => void) | null, inputId?: string) => void;

    // Range selection (for inputs that support ranges)
    selectedRange: [number, number];
    setSelectedRange: (startFrame: number, endFrame: number, inputId?: string) => void;
    getSelectedRange: (inputId?: string) => [number, number];

    // Per-input scoped state
    totalTimelineFramesByInputId: Record<string, number>;
    timelineDurationByInputId: Record<string, [number, number]>;
    zoomLevelByInputId: Record<string, ZoomLevel>;
    fpsByInputId: Record<string, number>;
    focusFrameByInputId: Record<string, number>;
    focusAnchorRatioByInputId: Record<string, number>;
    selectedInputClipIdByInputId: Record<string, string | null>;
    selectedInputChangeHandlersByInputId: Record<string, ((clipId: string | null) => void) | null>;
    selectedRangeByInputId: Record<string, [number, number]>;
    getTotalTimelineFrames: (inputId?: string) => number;
    getZoomLevel: (inputId?: string) => ZoomLevel;

    // Playback state (per-input)
    isPlayingByInputId: Record<string, boolean>;
    setIsPlaying: (isPlaying: boolean, inputId?: string) => void;
    getIsPlaying: (inputId?: string) => boolean;

    // Playback controls (per-input)
    play: (inputId: string) => void;
    pause: (inputId: string) => void;
}

const defaultTotalFrames = TIMELINE_DURATION_SECONDS * DEFAULT_FPS;
const defaultTimelineDuration: [number, number] = [0, defaultTotalFrames];

export const useInputControlsStore = create<InputControlStore>((set, get) => ({
    // Zoom
    maxZoomLevel: 10,
    setMaxZoomLevel: (level) => set({ maxZoomLevel: level }),
    minZoomLevel: 1,
    setMinZoomLevel: (level) => set({ minZoomLevel: level }),
    zoomLevel: 1,
    setZoomLevel: (level, inputId) => {
        if (!inputId) {
            set({ zoomLevel: level });
            return;
        }
        set((state) => {
            const prevLevel = state.zoomLevelByInputId[inputId] ?? 1;
            const next: any = {
                zoomLevelByInputId: {
                    ...state.zoomLevelByInputId,
                    [inputId]: level,
                },
            };
            // When zooming in, temporarily center focus within current selected range
            if (level > prevLevel) {
                const range = state.selectedRangeByInputId[inputId] ?? state.selectedRange ?? [0, 1];
                const start = Math.max(0, Math.round(range[0] ?? 0));
                const endExclusive = Math.max(start + 1, Math.round(range[1] ?? start + 1));
                const center = Math.floor((start + (endExclusive - 1)) / 2);
                next.focusFrameByInputId = {
                    ...state.focusFrameByInputId,
                    [inputId]: center,
                };
            }
            return next;
        });
    },
    getZoomLevel: (inputId) => {
        if (!inputId) return get().zoomLevel;
        return get().zoomLevelByInputId[inputId] ?? 1;
    },

    // Playback state (per-input)
    isPlayingByInputId: {},
    setIsPlaying: (isPlaying, inputId) => {
        if (!inputId) {
            // no-op for global; inputs should always specify inputId
            return;
        }
        set((state) => ({
            isPlayingByInputId: {
                ...state.isPlayingByInputId,
                [inputId]: !!isPlaying,
            },
        }));
    },
    getIsPlaying: (inputId) => {
        if (!inputId) return false;
        return !!get().isPlayingByInputId[inputId];
    },

    // playback controls filled in after store definition (see below)
    play: (_inputId: string) => {},
    pause: (_inputId: string) => {},

    // Timeline duration/window
    totalTimelineFrames: defaultTotalFrames,
    setTotalTimelineFrames: (frames, inputId) => {
        const clamped = Math.max(1, Math.round(frames));
        if (!inputId) {
            set({ totalTimelineFrames: clamped });
            return;
        }
        set((state) => ({
            totalTimelineFramesByInputId: {
                ...state.totalTimelineFramesByInputId,
                [inputId]: clamped,
            },
        }));
    },
    incrementTotalTimelineFrames: (frames, inputId) => {
        const delta = Math.round(frames);
        if (!inputId) {
            set((s) => ({ totalTimelineFrames: Math.max(1, s.totalTimelineFrames + delta) }));
            return;
        }
        set((state) => {
            const current = state.totalTimelineFramesByInputId[inputId] ?? defaultTotalFrames;
            return {
                totalTimelineFramesByInputId: {
                    ...state.totalTimelineFramesByInputId,
                    [inputId]: Math.max(1, current + delta),
                },
            };
        });
    },
    decrementTotalTimelineFrames: (frames, inputId) => {
        const delta = Math.round(frames);
        if (!inputId) {
            set((s) => ({ totalTimelineFrames: Math.max(1, s.totalTimelineFrames - delta) }));
            return;
        }
        set((state) => {
            const current = state.totalTimelineFramesByInputId[inputId] ?? defaultTotalFrames;
            return {
                totalTimelineFramesByInputId: {
                    ...state.totalTimelineFramesByInputId,
                    [inputId]: Math.max(1, current - delta),
                },
            };
        });
    },
    getTotalTimelineFrames: (inputId) => {
        if (!inputId) return get().totalTimelineFrames;
        return get().totalTimelineFramesByInputId[inputId] ?? defaultTotalFrames;
    },
    timelineDuration: defaultTimelineDuration,
    setTimelineDuration: (startFrame: number, endFrame: number, inputId) => {
        const s = Math.max(0, Math.round(startFrame));
        const e = Math.max(s + 1, Math.round(endFrame));
        const next: [number, number] = [s, e];
        if (!inputId) {
            set({ timelineDuration: next });
            return;
        }
        set((state) => ({
            timelineDurationByInputId: {
                ...state.timelineDurationByInputId,
                [inputId]: next,
            },
        }));
    },
    getTimelineDuration: (inputId) => {
        if (!inputId) return get().timelineDuration;
        return get().timelineDurationByInputId[inputId] ?? defaultTimelineDuration;
    },
    shiftTimelineDuration: (delta: number, inputId) => {
        const state = get();
        const [start, end] = state.getTimelineDuration(inputId);
        const total = Math.max(1, state.getTotalTimelineFrames(inputId));
        const win = Math.max(1, end - start);
        const desiredStart = Math.max(0, Math.min(total - win, start + Math.trunc(delta)));
        state.setTimelineDuration(desiredStart, desiredStart + win, inputId);
    },

    // FPS
    fps: DEFAULT_FPS,
    setFps: (fps, inputId) => {
        const clamped = Math.max(1, Math.round(fps));
        if (!inputId) {
            set({ fps: clamped });
            return;
        }
        set((state) => ({
            fpsByInputId: {
                ...state.fpsByInputId,
                [inputId]: clamped,
            },
        }));
    },
    getFps: (inputId) => {
        if (!inputId) return get().fps;
        return get().fpsByInputId[inputId] ?? DEFAULT_FPS;
    },

    // Focus
    focusFrame: 0,
    setFocusFrame: (frame, inputId) => {
        const clamped = Math.max(0, Math.round(frame));
        if (!inputId) {
            set({ focusFrame: clamped });
            return;
        }
        set((state) => ({
            focusFrameByInputId: {
                ...state.focusFrameByInputId,
                [inputId]: clamped,
            },
        }));
    },
    getFocusFrame: (inputId) => {
        if (!inputId) return get().focusFrame;
        return get().focusFrameByInputId[inputId] ?? 0;
    },
    focusAnchorRatio: 0.5,
    setFocusAnchorRatio: (ratio, inputId) => {
        const clamped = Math.max(0, Math.min(1, ratio));
        if (!inputId) {
            set({ focusAnchorRatio: clamped });
            return;
        }
        set((state) => ({
            focusAnchorRatioByInputId: {
                ...state.focusAnchorRatioByInputId,
                [inputId]: clamped,
            },
        }));
    },
    getFocusAnchorRatio: (inputId) => {
        if (!inputId) return get().focusAnchorRatio;
        return get().focusAnchorRatioByInputId[inputId] ?? 0.5;
    },

    // Selection (optional)
    selectedInputClipId: null,
    setSelectedInputClipId: (clipId, inputId) => {
        if (!inputId) {
            set({ selectedInputClipId: clipId });
            try {
                const handler = get().selectedInputChangeHandler;
                if (typeof handler === 'function') handler(clipId);
            } catch {}
            return;
        }
        set((state) => ({
            selectedInputClipIdByInputId: {
                ...state.selectedInputClipIdByInputId,
                [inputId]: clipId,
            },
        }));
        try {
            const handler = get().selectedInputChangeHandlersByInputId[inputId];
            if (typeof handler === 'function') handler(clipId);
        } catch {}
    },
    getSelectedInputClipId: (inputId) => {
        if (!inputId) return get().selectedInputClipId;
        return get().selectedInputClipIdByInputId[inputId] ?? null;
    },
    clearSelectedInput: (inputId) => {
        if (!inputId) {
            set({ selectedInputClipId: null });
            return;
        }
        set((state) => ({
            selectedInputClipIdByInputId: {
                ...state.selectedInputClipIdByInputId,
                [inputId]: null,
            },
        }));
    },
    selectedInputChangeHandler: null,
    setSelectedInputChangeHandler: (handler, inputId) => {
        if (!inputId) {
            set({ selectedInputChangeHandler: handler ?? null });
            return;
        }
        set((state) => ({
            selectedInputChangeHandlersByInputId: {
                ...state.selectedInputChangeHandlersByInputId,
                [inputId]: handler ?? null,
            },
        }));
    },

    // Range selection (defaults to a 1-frame span)
    selectedRange: [0, 1],
    setSelectedRange: (startFrame, endFrame, inputId) => {
        const start = Math.max(0, Math.round(startFrame));
        const end = Math.max(start + 1, Math.round(endFrame));
        const next: [number, number] = [start, end];
        if (!inputId) {
            set({ selectedRange: next, focusFrame: start });
            return;
        }
        set((state) => ({
            selectedRangeByInputId: {
                ...state.selectedRangeByInputId,
                [inputId]: next,
            },
            focusFrameByInputId: {
                ...state.focusFrameByInputId,
                [inputId]: start,
            },
        }));
    },
    getSelectedRange: (inputId) => {
        if (!inputId) return get().selectedRange;
        return get().selectedRangeByInputId[inputId] ?? [0, 1];
    },

    // Per-input scoped state containers
    totalTimelineFramesByInputId: {},
    timelineDurationByInputId: {},
    zoomLevelByInputId: {},
    fpsByInputId: {},
    focusFrameByInputId: {},
    focusAnchorRatioByInputId: {},
    selectedInputClipIdByInputId: {},
    selectedInputChangeHandlersByInputId: {},
    selectedRangeByInputId: {},
}));

// Playback internals (per-input, module-scoped to avoid causing unnecessary rerenders)
const __inputPlaybackRafIdByInput: Record<string, number | null> = {};
const __inputLastTickMsByInput: Record<string, number> = {};
const __inputFrameAccumulatorByInput: Record<string, number> = {};

function __requestNextInputTick(inputId: string) {
    __inputPlaybackRafIdByInput[inputId] = requestAnimationFrame((now) => __inputPlaybackTick(inputId, now));
}

function __inputPlaybackTick(inputId: string, now: number) {
    const controls = useInputControlsStore.getState();
    if (!controls.getIsPlaying(inputId)) return;

    const fps = Math.max(1, controls.getFps(inputId) || 1);
    const totalFrames = Math.max(0, controls.getTotalTimelineFrames(inputId) || 0);
    const [rangeStartRaw, rangeEndRaw] = controls.getSelectedRange(inputId) || [0, 1];
    const rangeStart = Math.max(0, Math.round(rangeStartRaw));
    const rangeEndExclusive = Math.max(rangeStart + 1, Math.round(rangeEndRaw));
    const stopFrameExclusive = Math.max(1, Math.min(totalFrames || rangeEndExclusive, rangeEndExclusive));

    if (!__inputLastTickMsByInput[inputId]) {
        __inputLastTickMsByInput[inputId] = now;
        __requestNextInputTick(inputId);
        return;
    }

    const dt = (now - __inputLastTickMsByInput[inputId]) / 1000;
    __inputFrameAccumulatorByInput[inputId] = (__inputFrameAccumulatorByInput[inputId] || 0) + dt * fps;
    let steps = Math.floor(__inputFrameAccumulatorByInput[inputId]);
    if (steps > 0) {
        __inputFrameAccumulatorByInput[inputId] -= steps;
        const current = controls.getFocusFrame(inputId) || 0;
        // Ensure playback occurs within the selected range
        const clampedCurrent = current < rangeStart ? rangeStart : current;
        const next = Math.min(stopFrameExclusive, clampedCurrent + steps);
        controls.setFocusFrame(next, inputId);
        if (next >= stopFrameExclusive) {
            controls.pause(inputId);
            __inputLastTickMsByInput[inputId] = 0;
            __inputFrameAccumulatorByInput[inputId] = 0;
            return;
        }
    }

    __inputLastTickMsByInput[inputId] = now;
    __requestNextInputTick(inputId);
}

// Wire up play/pause now that helpers exist
useInputControlsStore.setState((prev) => ({
    ...prev,
    play: (inputId: string) => {
        const state = useInputControlsStore.getState();
        if (state.getIsPlaying(inputId)) return;
        const total = Math.max(0, state.getTotalTimelineFrames(inputId) || 0);
        const [rangeStartRaw, rangeEndRaw] = state.getSelectedRange(inputId) || [0, 1];
        const rangeStart = Math.max(0, Math.round(rangeStartRaw));
        const rangeEndExclusive = Math.max(rangeStart + 1, Math.round(rangeEndRaw));
        const stopFrameExclusive = Math.max(1, Math.min(total || rangeEndExclusive, rangeEndExclusive));
        if (stopFrameExclusive <= rangeStart) return;
        // If focus is outside the range or at/past the stop frame, start from range start
        const currentFocus = state.getFocusFrame(inputId) || 0;
        if (currentFocus < rangeStart || currentFocus >= stopFrameExclusive) {
            state.setFocusFrame(rangeStart, inputId);
        }
        __inputLastTickMsByInput[inputId] = 0;
        __inputFrameAccumulatorByInput[inputId] = 0;
        const rafId = __inputPlaybackRafIdByInput[inputId];
        if (rafId != null) cancelAnimationFrame(rafId);
        useInputControlsStore.setState((s) => ({
            isPlayingByInputId: { ...s.isPlayingByInputId, [inputId]: true },
        }));
        __requestNextInputTick(inputId);
    },
    pause: (inputId: string) => {
        const rafId = __inputPlaybackRafIdByInput[inputId];
        if (rafId != null) {
            cancelAnimationFrame(rafId);
            __inputPlaybackRafIdByInput[inputId] = null;
        }
        __inputLastTickMsByInput[inputId] = 0;
        __inputFrameAccumulatorByInput[inputId] = 0;
        useInputControlsStore.setState((s) => ({
            isPlayingByInputId: { ...s.isPlayingByInputId, [inputId]: false },
        }));
    },
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
