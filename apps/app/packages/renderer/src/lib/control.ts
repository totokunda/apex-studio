import { create } from "zustand";
import { ZoomLevel } from "./types";
import {
  TIMELINE_DURATION_SECONDS,
  DEFAULT_FPS,
  MAX_DURATION,
} from "./settings";
import { useClipStore } from "./clip";
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
    controls.setFocusFrame(next, false);
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
  maxZoomLevel: ZoomLevel;
  setMaxZoomLevel: (level: ZoomLevel) => void;
  minZoomLevel: ZoomLevel;
  setMinZoomLevel: (level: ZoomLevel) => void;
  zoomLevel: ZoomLevel;
  setZoomLevel: (level: ZoomLevel) => void;
  totalTimelineFrames: number; // total frames in the timeline
  incrementTotalTimelineFrames: (duration: number) => void;
  decrementTotalTimelineFrames: (duration: number) => void;
  timelineDuration: [number, number]; // [startFrame, endFrame]
  setTimelineDuration: (startFrame: number, endFrame: number) => void;
  shiftTimelineDuration: (
    duration: number,
    shiftFocusFrame?: boolean,
    pause?: boolean,
  ) => void;
  canTimelineDurationBeShifted: (duration: number) => boolean;
  resetTimelineDuration: () => void;
  fps: number;
  setFps: (fps: number) => void;
  setFpsWithRescale: (fps: number) => void;
  defaultClipLength: number;
  setDefaultClipLength: (length: number) => void;
  focusFrame: number;
  setFocusFrame: (frame: number, pause?: boolean) => void;
  focusAnchorRatio: number; // 0..1 position of scrubber within viewport width
  setFocusAnchorRatio: (ratio: number) => void;
  selectedClipIds: string[];
  setSelectedClipIds: (clipIds: string[]) => void;
  addClipSelection: (clipId: string) => void;
  removeClipSelection: (clipId: string) => void;
  toggleClipSelection: (clipId: string, isShiftClick?: boolean) => void;
  clearSelection: () => void;
  play: () => void;
  pause: () => void;
  isPlaying: boolean;
  setIsPlaying: (isPlaying: boolean) => void;
  isFullscreen: boolean;
  setIsFullscreen: (isFullscreen: boolean) => void;
  selectedMaskId: string | null;
  setSelectedMaskId: (maskId: string | null) => void;
  possibleKeyFocusFrames: number[];
  setPossibleKeyFocusFrames: (frames: number[]) => void;
  isAccurateSeekNeeded: boolean;
  setIsAccurateSeekNeeded: (needed: boolean) => void;
}

export const useControlsStore = create<ControlStore>((set, get) => ({
  // Zoom State
  maxZoomLevel: 10,
  setMaxZoomLevel: (level) => set({ maxZoomLevel: level }),
  minZoomLevel: 1,
  setMinZoomLevel: (level) => set({ minZoomLevel: level }),
  zoomLevel: 1,
  setZoomLevel: (level) => set({ zoomLevel: level }),
  // Timeline State
  totalTimelineFrames: TIMELINE_DURATION_SECONDS * DEFAULT_FPS, // total frames in the timeline
  incrementTotalTimelineFrames: (duration: number) =>
    set((state) => {
      return { totalTimelineFrames: state.totalTimelineFrames + duration };
    }),
  decrementTotalTimelineFrames: (duration: number) =>
    set((state) => {
      if (state.totalTimelineFrames - duration < MAX_DURATION)
        return { totalTimelineFrames: MAX_DURATION };
      return { totalTimelineFrames: state.totalTimelineFrames - duration };
    }),
  canTimelineDurationBeShifted: (duration: number) => {
    const state = get();
    // Preserve existing rule
    const basicAllowed =
      state.totalTimelineFrames - duration > state.totalTimelineFrames;
    if (basicAllowed) return true;

    // Additional rule:
    // Allow shifting if any clip overlaps the current window and extends beyond it
    // in the direction of the requested shift. This enables panning to reveal
    // the beginning/end of partially-visible clips.
    const [startFrame, endFrame] = state.timelineDuration;
    const clips =
      (useClipStore.getState().clips as Array<{
        startFrame?: number;
        endFrame?: number;
      }>) || [];
    if (!clips.length) return false;

    // Add a small buffer so that the entire clip can be brought fully into view
    const fps = Math.max(1, state.fps || DEFAULT_FPS);
    const bufferFrames = Math.max(1, Math.round(fps / 2)); // ~0.5s buffer

    const overlapsWindow = (cs: number, ce: number) =>
      ce > startFrame && cs < endFrame;

    if (duration < 0) {
      // Request to shift left: allow if any overlapping clip starts before the window start
      return clips.some((c) => {
        const cs = Math.max(0, c.startFrame ?? 0);
        const ce = Math.max(cs + 1, c.endFrame ?? 0);
        return overlapsWindow(cs, ce) && cs < startFrame + bufferFrames;
      });
    } else if (duration > 0) {
      // Request to shift right: allow if any overlapping clip ends after the window end
      return clips.some((c) => {
        const cs = Math.max(0, c.startFrame ?? 0);
        const ce = Math.max(cs + 1, c.endFrame ?? 0);
        return overlapsWindow(cs, ce) && ce > endFrame - bufferFrames;
      });
    }

    return false;
  },
  timelineDuration: [0, TIMELINE_DURATION_SECONDS * DEFAULT_FPS], // [startFrame, endFrame]
  setTimelineDuration: (startFrame: number, endFrame: number) =>
    set({ timelineDuration: [startFrame, endFrame] }),
  shiftTimelineDuration: (
    duration: number,
    shiftFocusFrame?: boolean,
    pause: boolean = false,
  ) => {
    const state = get();
    const [startFrame, endFrame] = state.timelineDuration;
    const requestedShift = Math.trunc(duration);
    const minShift = -startFrame; // cannot move left beyond 0
    const maxShift = state.totalTimelineFrames - endFrame; // cannot move right beyond total
    const clampedShift = Math.max(minShift, Math.min(maxShift, requestedShift));
    if (clampedShift === 0) {
      return;
    }
    const wasPlaying = state.isPlaying;
    if (wasPlaying && pause) {
      state.pause();
    }
    const newStartFrame = startFrame + clampedShift;
    const newEndFrame = endFrame + clampedShift;
    const nextUpdate: Partial<ControlStore> = shiftFocusFrame
      ? {
          timelineDuration: [newStartFrame, newEndFrame],
          focusFrame: Math.max(
            0,
            Math.min(
              state.totalTimelineFrames - 1,
              state.focusFrame + clampedShift,
            ),
          ),
        }
      : { timelineDuration: [newStartFrame, newEndFrame] };
    set(nextUpdate as any);
    if (wasPlaying && pause) {
      setTimeout(() => {
        const s = get();
        if (!s.isPlaying) s.play();
      }, 0);
    }
  },
  resetTimelineDuration: () =>
    set({ timelineDuration: [0, TIMELINE_DURATION_SECONDS * DEFAULT_FPS] }),
  // FPS State
  fps: DEFAULT_FPS,
  setFps: (fps) => set({ fps }),
  setFpsWithRescale: (fps) => {
    const oldFps = get().fps || DEFAULT_FPS;
    const newFps = Math.max(1, fps || DEFAULT_FPS);
    if (oldFps === newFps) {
      set({ fps: newFps });
      return;
    }
    // First update clips/preprocessors to preserve timing
    useClipStore.getState().rescaleForFpsChange(oldFps, newFps);
    // Then update control fps and timeline duration bounds
    set((state) => {
      const scale = newFps / Math.max(1, oldFps);
      const [start, end] = state.timelineDuration;
      const newStart = Math.round(start * scale);
      const newEnd = Math.round(end * scale);
      const totalFrames = Math.max(
        0,
        Math.round(state.totalTimelineFrames * scale),
      );
      const newFocus = Math.max(
        0,
        Math.min(
          Math.max(0, totalFrames - 1),
          Math.round((state.focusFrame || 0) * scale),
        ),
      );
      return {
        fps: newFps,
        totalTimelineFrames: totalFrames,
        timelineDuration: [newStart, newEnd],
        focusFrame: newFocus,
      };
    });
  },
  defaultClipLength: 5,
  setDefaultClipLength: (length: number) =>
    set({ defaultClipLength: Math.max(1, Math.round(length || 1)) }),
  // Focus State
  focusFrame: 0,
  setFocusFrame: (frame, pause = true) => {
    const state = get();
    const clamped = Math.max(0, Math.round(frame));
    if (pause) {
      const wasPlaying = state.isPlaying;
      if (wasPlaying) state.pause();
      set({ focusFrame: clamped });
      if (wasPlaying) {
        setTimeout(() => {
          const s = get();
          if (!s.isPlaying) s.play();
        }, 0);
      }
    } else {
      set({ focusFrame: clamped });
    }
  },
  focusAnchorRatio: 0.5,
  setFocusAnchorRatio: (ratio) =>
    set({ focusAnchorRatio: Math.max(0, Math.min(1, ratio)) }),
  selectedClipIds: [],
  setSelectedClipIds: (clipIds) => set({ selectedClipIds: clipIds }),
  addClipSelection: (clipId) =>
    set((state) => {
      return { selectedClipIds: _.uniq([...state.selectedClipIds, clipId]) };
    }),
  removeClipSelection: (clipId) =>
    set((state) => {
      return {
        selectedClipIds: state.selectedClipIds.filter((id) => id !== clipId),
      };
    }),
  toggleClipSelection: (clipId, isShiftClick = false) =>
    set((state) => {
      if (isShiftClick) {
        // Add to selection if shift-clicking
        if (state.selectedClipIds.includes(clipId)) {
          // Remove from selection if already selected
          return {
            selectedClipIds: state.selectedClipIds.filter(
              (id) => id !== clipId,
            ),
          };
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
  play: () => {
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
  pause: () => {
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
  isFullscreen: false,
  setIsFullscreen: (isFullscreen) => set({ isFullscreen }),
  selectedMaskId: null,
  setSelectedMaskId: (maskId) => set({ selectedMaskId: maskId }),
  possibleKeyFocusFrames: [],
  setPossibleKeyFocusFrames: (frames) => set({ possibleKeyFocusFrames: frames }),
  isAccurateSeekNeeded: true,
  setIsAccurateSeekNeeded: (needed) => set({ isAccurateSeekNeeded: needed }),
}));
