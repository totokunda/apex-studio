import React, { createContext, useContext } from "react";
import { createStore, StoreApi } from "zustand/vanilla";
import { useStore } from "zustand";
import { ZoomLevel } from "./types";
import {
  DEFAULT_FPS,
  TIMELINE_DURATION_SECONDS,
  MIN_DURATION,
} from "./settings";
import _ from "lodash";

/**
 * Internal, clip-aware store shape.
 *
 * NOTE: this store is keyed by both inputId and clipId. Callers should
 * never use this directly – instead they should go through the
 * `useInputControlsStore` hook which exposes a per-clip façade that
 * only requires an inputId.
 */
interface InputControlStore {
  // Zoom state (independent from other controls)
  setZoomLevel: (level: ZoomLevel, inputId: string, clipId: string) => void;
  setTotalTimelineFrames: (frames: number, inputId: string, clipId: string) => void;
  incrementTotalTimelineFrames: (frames: number, inputId: string, clipId: string) => void;
  decrementTotalTimelineFrames: (frames: number, inputId: string, clipId: string) => void;
  getTimelineDuration: (inputId: string, clipId: string) => [number, number];
  setTimelineDuration: (startFrame: number, endFrame: number, inputId: string, clipId: string) => void;
  shiftTimelineDuration: (delta: number, inputId: string, clipId: string) => void;
  setFps: (fps: number, inputId: string, clipId: string) => void;
  getFps: (inputId: string, clipId: string) => number;
  // Optional selection for preview/input timelines
  clearSelectedInput: (inputId: string, clipId: string) => void;
  selectedInputChangeHandler: ((clipId: string | null) => void) | null;
  setSelectedInputChangeHandler: (handler: ((clipId: string | null) => void) | null, inputId: string, clipId: string) => void;
  setSelectedRange: (startFrame: number, endFrame: number, inputId: string, clipId: string) => void;
  getSelectedRange: (inputId: string, clipId: string) => [number, number];
  clearSelectedRange: (inputId: string, clipId: string) => void;
  setFocusFrame: (frame: number, inputId: string, clipId: string) => void;
  getFocusFrame: (inputId: string, clipId: string) => number;
  clearFocusFrame: (inputId: string, clipId: string) => void;
  setFocusAnchorRatio: (ratio: number, inputId: string, clipId: string) => void;
  getFocusAnchorRatio: (inputId: string, clipId: string) => number;

  // Per-input scoped state
  totalTimelineFramesByInputId: Record<string, Record<string, number>>;
  timelineDurationByInputId: Record<string, Record<string, [number, number]>>;
  zoomLevelByInputId: Record<string, Record<string, ZoomLevel>>;
  fpsByInputId: Record<string, Record<string, number>>;
  focusFrameByInputId: Record<string, Record<string, number>>;
  focusAnchorRatioByInputId: Record<string, Record<string, number>>;
  selectedInputClipIdByInputId: Record<string, Record<string, string | null>>;
  selectedInputChangeHandlersByInputId: Record<
    string,
    ((clipId: string | null) => void) | null
  >;
  selectedRangeByInputId: Record<string, Record<string, [number, number]>>;
  getTotalTimelineFrames: (inputId: string, clipId: string) => number;
  getZoomLevel: (inputId: string, clipId: string) => ZoomLevel;

  // Playback state (per-input)
  isPlayingByInputId: Record<string, Record<string, boolean>>;
  setIsPlaying: (isPlaying: boolean, inputId: string, clipId: string) => void;
  getIsPlaying: (inputId: string, clipId: string) => boolean;

  // Playback controls (per-input, per-clip)
  play: (inputId: string, clipId: string) => void;
  pause: (inputId: string, clipId: string) => void;
}

/**
 * Public, per-clip view of the input controls store.
 *
 * All APIs only require an inputId – the active clipId is injected
 * by the hook based on either:
 * - the surrounding `InputControlsProvider` (preferred), or
 * - a global fallback clip id when no provider is mounted.
 */
export interface PerClipInputControlsStore {
  // Zoom state (independent from other controls)
  setZoomLevel: (level: ZoomLevel, inputId: string) => void;
  setTotalTimelineFrames: (frames: number, inputId: string) => void;
  incrementTotalTimelineFrames: (frames: number, inputId: string) => void;
  decrementTotalTimelineFrames: (frames: number, inputId: string) => void;
  getTimelineDuration: (inputId: string) => [number, number];
  setTimelineDuration: (
    startFrame: number,
    endFrame: number,
    inputId: string,
  ) => void;
  shiftTimelineDuration: (delta: number, inputId: string) => void;
  setFps: (fps: number, inputId: string) => void;
  getFps: (inputId: string) => number;

  // Optional selection for preview/input timelines
  clearSelectedInput: (inputId: string) => void;
  selectedInputChangeHandler: ((clipId: string | null) => void) | null;
  setSelectedInputChangeHandler: (
    handler: ((clipId: string | null) => void) | null,
    inputId: string,
  ) => void;
  setSelectedRange: (
    startFrame: number,
    endFrame: number,
    inputId: string,
  ) => void;
  getSelectedRange: (inputId: string) => [number, number];
  clearSelectedRange: (inputId: string) => void;
  setFocusFrame: (frame: number, inputId: string) => void;
  getFocusFrame: (inputId: string) => number;
  clearFocusFrame: (inputId: string) => void;
  setFocusAnchorRatio: (ratio: number, inputId: string) => void;
  getFocusAnchorRatio: (inputId: string) => number;

  // Per-input scoped state, already scoped to the active clipId.
  totalTimelineFramesByInputId: Record<string, number>;
  timelineDurationByInputId: Record<string, [number, number]>;
  zoomLevelByInputId: Record<string, ZoomLevel>;
  fpsByInputId: Record<string, number>;
  focusFrameByInputId: Record<string, number>;
  focusAnchorRatioByInputId: Record<string, number>;
  selectedInputClipIdByInputId: Record<string, string | null>;
  selectedInputChangeHandlersByInputId: Record<
    string,
    ((clipId: string | null) => void) | null
  >;
  selectedRangeByInputId: Record<string, [number, number]>;
  getTotalTimelineFrames: (inputId: string) => number;
  getZoomLevel: (inputId: string) => ZoomLevel;

  // Playback state (per-input)
  isPlayingByInputId: Record<string, boolean>;
  setIsPlaying: (isPlaying: boolean, inputId: string) => void;
  getIsPlaying: (inputId: string) => boolean;

  // Playback controls (per-input)
  play: (inputId: string) => void;
  pause: (inputId: string) => void;
}

const defaultTotalFrames = TIMELINE_DURATION_SECONDS * DEFAULT_FPS;
const defaultTimelineDuration: [number, number] = [0, defaultTotalFrames];

// Playback internals (per-input, per-store, module-scoped to avoid causing
// unnecessary rerenders). We scope by store instance using a WeakMap to keep
// cleanup automatic once a store is no longer referenced.
type PlaybackStatePerStore = {
  rafIdByInput: Record<string, number | null>;
  lastTickMsByInput: Record<string, number>;
  frameAccumulatorByInput: Record<string, number>;
};

const playbackStateByStore = new WeakMap<
  StoreApi<InputControlStore>,
  PlaybackStatePerStore
>();

function getPlaybackState(
  store: StoreApi<InputControlStore>,
): PlaybackStatePerStore {
  let st = playbackStateByStore.get(store);
  if (!st) {
    st = {
      rafIdByInput: {},
      lastTickMsByInput: {},
      frameAccumulatorByInput: {},
    };
    playbackStateByStore.set(store, st);
  }
  return st;
}

const GLOBAL_FALLBACK_CLIP_ID = "__global__";

// Cache of projected per-clip stores so that method identities remain stable
// across renders and store updates (avoids infinite effect loops that depend
// on the projected methods).
const projectedStoreCache = new WeakMap<
  StoreApi<InputControlStore>,
  Map<string, PerClipInputControlsStore>
>();

function flattenNestedForClip<T>(
  nested: Record<string, Record<string, T>>,
  clipId: string,
): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [inputId, byClip] of Object.entries(nested || {})) {
    if (
      byClip &&
      Object.prototype.hasOwnProperty.call(byClip, clipId)
    ) {
      out[inputId] = byClip[clipId] as T;
    }
  }
  return out;
}

function getProjectedStoreForClip(
  store: StoreApi<InputControlStore>,
  clipId: string,
): PerClipInputControlsStore {
  let byClip = projectedStoreCache.get(store);
  if (!byClip) {
    byClip = new Map();
    projectedStoreCache.set(store, byClip);
  }

  const existing = byClip.get(clipId);
  if (existing) return existing;

  const getBase = () => store.getState();
  const projected: Partial<PerClipInputControlsStore> = {};

  // Methods – all delegate to the underlying store for this clipId
  projected.setZoomLevel = (level, inputId) =>
    getBase().setZoomLevel(level, inputId, clipId);
  projected.setTotalTimelineFrames = (frames, inputId) =>
    getBase().setTotalTimelineFrames(frames, inputId, clipId);
  projected.incrementTotalTimelineFrames = (frames, inputId) =>
    getBase().incrementTotalTimelineFrames(frames, inputId, clipId);
  projected.decrementTotalTimelineFrames = (frames, inputId) =>
    getBase().decrementTotalTimelineFrames(frames, inputId, clipId);
  projected.getTimelineDuration = (inputId) =>
    getBase().getTimelineDuration(inputId, clipId) ?? defaultTimelineDuration;
  projected.setTimelineDuration = (startFrame, endFrame, inputId) =>
    getBase().setTimelineDuration(startFrame, endFrame, inputId, clipId);
  projected.shiftTimelineDuration = (delta, inputId) =>
    getBase().shiftTimelineDuration(delta, inputId, clipId);
  projected.setFps = (fps, inputId) =>
    getBase().setFps(fps, inputId, clipId);
  projected.getFps = (inputId) =>
    getBase().getFps(inputId, clipId) ?? DEFAULT_FPS;
  projected.clearSelectedInput = (inputId) =>
    getBase().clearSelectedInput(inputId, clipId);
  projected.selectedInputChangeHandler = getBase().selectedInputChangeHandler;
  projected.setSelectedInputChangeHandler = (handler, inputId) =>
    getBase().setSelectedInputChangeHandler(handler, inputId, clipId);
  projected.setSelectedRange = (startFrame, endFrame, inputId) => {
    getBase().setSelectedRange(startFrame, endFrame, inputId, clipId);
  }
    
  projected.getSelectedRange = (inputId) =>
    getBase().getSelectedRange(inputId, clipId) ?? [0, 1];
  projected.clearSelectedRange = (inputId) =>
    getBase().clearSelectedRange(inputId, clipId);
  projected.setFocusFrame = (frame, inputId) =>
    getBase().setFocusFrame(frame, inputId, clipId);
  projected.getFocusFrame = (inputId) =>
    getBase().getFocusFrame(inputId, clipId) ?? 0;
  projected.clearFocusFrame = (inputId) =>
    getBase().clearFocusFrame(inputId, clipId);
  projected.setFocusAnchorRatio = (ratio, inputId) =>
    getBase().setFocusAnchorRatio(ratio, inputId, clipId);
  projected.getFocusAnchorRatio = (inputId) =>
    getBase().getFocusAnchorRatio(inputId, clipId) ?? 0.5;

  projected.getTotalTimelineFrames = (inputId) =>
    getBase().getTotalTimelineFrames(inputId, clipId) ?? defaultTotalFrames;
  projected.getZoomLevel = (inputId) =>
    getBase().getZoomLevel(inputId, clipId);

  projected.setIsPlaying = (isPlaying, inputId) =>
    getBase().setIsPlaying(isPlaying, inputId, clipId);
  projected.getIsPlaying = (inputId) =>
    getBase().getIsPlaying(inputId, clipId) ?? false;
  projected.play = (inputId) => getBase().play(inputId, clipId);
  projected.pause = (inputId) => getBase().pause(inputId, clipId);


  // Data maps – exposed as getters so callers always see latest values,
  // while the projected object and method identities remain stable.
  Object.defineProperty(projected, "totalTimelineFramesByInputId", {
    get() {
      return flattenNestedForClip(
        getBase().totalTimelineFramesByInputId,
        clipId,
      );
    },
  });

  Object.defineProperty(projected, "timelineDurationByInputId", {
    get() {
      return flattenNestedForClip(
        getBase().timelineDurationByInputId,
        clipId,
      );
    },
  });

  Object.defineProperty(projected, "zoomLevelByInputId", {
    get() {
      return flattenNestedForClip(getBase().zoomLevelByInputId, clipId);
    },
  });

  Object.defineProperty(projected, "fpsByInputId", {
    get() {
      return flattenNestedForClip(getBase().fpsByInputId, clipId);
    },
  });

  Object.defineProperty(projected, "focusFrameByInputId", {
    get() {
      return flattenNestedForClip(getBase().focusFrameByInputId, clipId);
    },
  });

  Object.defineProperty(projected, "focusAnchorRatioByInputId", {
    get() {
      return flattenNestedForClip(
        getBase().focusAnchorRatioByInputId,
        clipId,
      );
    },
  });

  Object.defineProperty(projected, "selectedInputClipIdByInputId", {
    get() {
      return flattenNestedForClip(
        getBase().selectedInputClipIdByInputId,
        clipId,
      );
    },
  });

  Object.defineProperty(projected, "selectedInputChangeHandlersByInputId", {
    get() {
      const nested = getBase().selectedInputChangeHandlersByInputId;
      const out: Record<
        string,
        ((clipId: string | null) => void) | null
      > = {};
      for (const [inputId, rawByClip] of Object.entries(nested || {})) {
        const byClip = rawByClip as Record<string, unknown> | null;
        if (
          byClip &&
          Object.prototype.hasOwnProperty.call(byClip, clipId)
        ) {
          const handler = byClip[clipId as unknown as string];
          if (handler == null || typeof handler === "function") {
            out[inputId] = handler as
              | ((clipId: string | null) => void)
              | null;
          }
        }
      }
      return out;
    },
  });

  Object.defineProperty(projected, "selectedRangeByInputId", {
    get() {
      return flattenNestedForClip(getBase().selectedRangeByInputId, clipId);
    },
  });

  Object.defineProperty(projected, "isPlayingByInputId", {
    get() {
      return flattenNestedForClip(getBase().isPlayingByInputId, clipId);
    },
  });

  const finalized = projected as PerClipInputControlsStore;
  byClip.set(clipId, finalized);
  return finalized;
}

function requestNextInputTick(
  store: StoreApi<InputControlStore>,
  inputId: string,
  clipId: string,
) {
  const playback = getPlaybackState(store);
  playback.rafIdByInput[inputId] = requestAnimationFrame((now) =>
    inputPlaybackTick(store, inputId, now, clipId),
  );
}

function inputPlaybackTick(
  store: StoreApi<InputControlStore>,
  inputId: string,
  now: number,
  clipId: string,
) {
  const controls = store.getState();
  if (!controls.getIsPlaying(inputId, clipId)) return;

  const playback = getPlaybackState(store);

  const fps = Math.max(1, controls.getFps(inputId, clipId) || 1);
  const totalFrames = Math.max(
    0,
    controls.getTotalTimelineFrames(inputId, clipId) || 0,
  );
  const [rangeStartRaw, rangeEndRaw] = controls.getSelectedRange(inputId, clipId) || [
    0, 1,
  ];
  const rangeStart = Math.max(0, Math.round(rangeStartRaw));
  const rangeEndExclusive = Math.max(rangeStart + 1, Math.round(rangeEndRaw));

  const stopFrameExclusive = Math.max(
    1,
    Math.min(totalFrames || rangeEndExclusive, rangeEndExclusive),
  );
  if (!playback.lastTickMsByInput[inputId]) {
    playback.lastTickMsByInput[inputId] = now;
    requestNextInputTick(store, inputId, clipId);
    return;
  }

  const dt = (now - playback.lastTickMsByInput[inputId]) / 1000;
  playback.frameAccumulatorByInput[inputId] =
    (playback.frameAccumulatorByInput[inputId] || 0) + dt * fps;
  let steps = Math.floor(playback.frameAccumulatorByInput[inputId]);
  if (steps > 0) {
    playback.frameAccumulatorByInput[inputId] -= steps;
    const current = controls.getFocusFrame(inputId, clipId) || 0;
    // Ensure playback occurs within the selected range
    const clampedCurrent = current < rangeStart ? rangeStart : current;
    const unclampedNext = clampedCurrent + steps;
    // If we've reached or passed the exclusive stop frame, clamp to the last valid frame and stop.
    if (unclampedNext >= stopFrameExclusive) {
      const finalFrame = Math.max(rangeStart, stopFrameExclusive);
      controls.setFocusFrame(finalFrame, inputId, clipId);
      controls.pause(inputId, clipId);
      playback.lastTickMsByInput[inputId] = 0;
      playback.frameAccumulatorByInput[inputId] = 0;
      return;
    }
    controls.setFocusFrame(unclampedNext, inputId, clipId);
  }

  playback.lastTickMsByInput[inputId] = now;
  requestNextInputTick(store, inputId, clipId);
}

function wirePlaybackControls(store: StoreApi<InputControlStore>) {
  const playback = getPlaybackState(store);
  store.setState((prev) => ({
    ...prev,
    play: (inputId: string, clipId: string) => {
      const state = store.getState();
      if (state.getIsPlaying(inputId, clipId)) return;
      const total = Math.max(0, state.getTotalTimelineFrames(inputId, clipId) || 0);
      const [rangeStartRaw, rangeEndRaw] = state.getSelectedRange(inputId, clipId) || [
        0, 1,
      ];
      const rangeStart = Math.max(0, Math.round(rangeStartRaw));
      const rangeEndExclusive = Math.max(
        rangeStart + 1,
        Math.round(rangeEndRaw),
      );
      const stopFrameExclusive = Math.max(
        1,
        Math.min(total || rangeEndExclusive, rangeEndExclusive),
      );
      if (stopFrameExclusive <= rangeStart) return;
      // If focus is outside the range or at/past the stop frame, start from range start
      const currentFocus = state.getFocusFrame(inputId, clipId) || 0;
      if (currentFocus < rangeStart || currentFocus >= stopFrameExclusive) {
        state.setFocusFrame(rangeStart, inputId, clipId);
      }
      playback.lastTickMsByInput[inputId] = 0;
      playback.frameAccumulatorByInput[inputId] = 0;
      const rafId = playback.rafIdByInput[inputId];
      if (rafId != null) cancelAnimationFrame(rafId);
      store.setState((s) => {
        const currentInputPlaying = s.isPlayingByInputId[inputId] || {};
        return {
          isPlayingByInputId: {
            ...s.isPlayingByInputId,
            [inputId]: {
              ...currentInputPlaying,
              [clipId]: true,
            },
          },
        };
      });
      requestNextInputTick(store, inputId, clipId);
    },
    pause: (inputId: string, clipId: string) => {
      const rafId = playback.rafIdByInput[inputId];
      if (rafId != null) {
        cancelAnimationFrame(rafId);
        playback.rafIdByInput[inputId] = null;
      }
      playback.lastTickMsByInput[inputId] = 0;
      playback.frameAccumulatorByInput[inputId] = 0;
      store.setState((s) => {
        const currentInputPlaying = s.isPlayingByInputId[inputId] || {};
        return {
          isPlayingByInputId: {
            ...s.isPlayingByInputId,
            [inputId]: {
              ...currentInputPlaying,
              [clipId]: false,
            },
          },
        };
      });
    },
  }));
}

/**
 * Core backing store factory. This creates an internal clip-aware store
 * and wires up playback controls.
 * */
export const globalInputControlsStore = createStore<InputControlStore>((set, get) => ({
    // Per-input scoped state (keyed by inputId, then clipId)
    totalTimelineFramesByInputId: {},
    timelineDurationByInputId: {},
    zoomLevelByInputId: {},
    fpsByInputId: {},
    focusFrameByInputId: {},
    focusAnchorRatioByInputId: {},
    selectedInputClipIdByInputId: {},
    selectedInputChangeHandlersByInputId: {},
    selectedRangeByInputId: {},
    isPlayingByInputId: {},

    // Legacy, currently unused, kept for compatibility
    selectedInputChangeHandler: null,

    // Zoom / timeline helpers
    setZoomLevel: (level, inputId, clipId) =>
      set((state) => {
        const existing = state.zoomLevelByInputId[inputId] || {};
        return {
          zoomLevelByInputId: {
            ...state.zoomLevelByInputId,
            [inputId]: {
              ...existing,
              [clipId]: level,
            },
          },
        };
      }),
    setTotalTimelineFrames: (frames, inputId, clipId) =>
      set((state) => {
        const existing = state.totalTimelineFramesByInputId[inputId] || {};
        const safeFrames = Math.max(MIN_DURATION, Math.round(frames || 0));
        return {
          totalTimelineFramesByInputId: {
            ...state.totalTimelineFramesByInputId,
            [inputId]: {
              ...existing,
              [clipId]: safeFrames,
            },
          },
        };
      }),
    incrementTotalTimelineFrames: (frames, inputId, clipId) => {
      const current = get().getTotalTimelineFrames(inputId, clipId);
      get().setTotalTimelineFrames(current + frames, inputId, clipId);
    },
    decrementTotalTimelineFrames: (frames, inputId, clipId) => {
      const current = get().getTotalTimelineFrames(inputId, clipId);
      get().setTotalTimelineFrames(current - frames, inputId, clipId);
    },
    getTotalTimelineFrames: (inputId, clipId) => {
      const byInput = get().totalTimelineFramesByInputId[inputId];
      const val = byInput && byInput[clipId];
      return val != null ? val : defaultTotalFrames;
    },

    getTimelineDuration: (inputId, clipId) => {
      const byInput = get().timelineDurationByInputId[inputId];
      const val = byInput && byInput[clipId];
      if (val && Array.isArray(val) && val.length === 2) return val;
      const total = get().getTotalTimelineFrames(inputId, clipId);
      return [0, total];
    },
    setTimelineDuration: (startFrame, endFrame, inputId, clipId) =>
      set((state) => {
        const byInput = state.timelineDurationByInputId[inputId] || {};
        const s = Math.max(0, Math.round(startFrame || 0));
        const e = Math.max(s + 1, Math.round(endFrame || 0));
        return {
          timelineDurationByInputId: {
            ...state.timelineDurationByInputId,
            [inputId]: {
              ...byInput,
              [clipId]: [s, e],
            },
          },
        };
      }),
    shiftTimelineDuration: (delta, inputId, clipId) => {
      const [start, end] = get().getTimelineDuration(inputId, clipId);
      const total = get().getTotalTimelineFrames(inputId, clipId);
      const width = Math.max(1, end - start);
      let newStart = Math.round(start + delta);
      newStart = Math.max(0, Math.min(newStart, Math.max(0, total - width)));
      const newEnd = newStart + width;
      get().setTimelineDuration(newStart, newEnd, inputId, clipId);
    },

    setFps: (fps, inputId, clipId) =>
      set((state) => {
        const byInput = state.fpsByInputId[inputId] || {};
        const safe = Math.max(1, Math.round(fps || 0));
        return {
          fpsByInputId: {
            ...state.fpsByInputId,
            [inputId]: {
              ...byInput,
              [clipId]: safe,
            },
          },
        };
      }),
    getFps: (inputId, clipId) => {
      const byInput = get().fpsByInputId[inputId];
      const val = byInput && byInput[clipId];
      return val != null ? val : DEFAULT_FPS;
    },

    clearSelectedInput: (inputId, clipId) =>
      set((state) => {
        const byInput = state.selectedInputClipIdByInputId[inputId] || {};
        const { [clipId]: _removed, ...rest } = byInput;
        return {
          selectedInputClipIdByInputId: {
            ...state.selectedInputClipIdByInputId,
            [inputId]: rest,
          },
        };
      }),
    setSelectedInputChangeHandler: (handler, inputId, _clipId) =>
      set((state) => {
        return {
          selectedInputChangeHandlersByInputId: {
            ...state.selectedInputChangeHandlersByInputId,
            [inputId]: handler,
          },
        } as Partial<InputControlStore>;
      }),
    setSelectedRange: (startFrame, endFrame, inputId, clipId) =>
      set((state) => {
        const byInput = state.selectedRangeByInputId[inputId] || {};
        const s = Math.max(0, Math.round(startFrame || 0));
        const e = Math.max(s + 1, Math.round(endFrame || 0));
        return {
          selectedRangeByInputId: {
            ...state.selectedRangeByInputId,
            [inputId]: {
              ...byInput,
              [clipId]: [s, e],
            },
          },
        };
      }),
    clearSelectedRange: (inputId, clipId) =>
      set((state) => {
        const byInput = state.selectedRangeByInputId[inputId] || {};
        const { [clipId]: _removed, ...rest } = byInput;
        const next = { ...state.selectedRangeByInputId };
        if (Object.keys(rest).length === 0) {
          delete next[inputId];
        } else {
          next[inputId] = rest;
        }
        return { selectedRangeByInputId: next };
      }),
    getSelectedRange: (inputId, clipId) => {
      const byInput = get().selectedRangeByInputId[inputId];
      const val = byInput && byInput[clipId];
      if (val && Array.isArray(val) && val.length === 2) return val;
      const total = get().getTotalTimelineFrames(inputId, clipId);
      return [0, total];
    },

    setFocusFrame: (frame, inputId, clipId) =>
      set((state) => {
        const byInput = state.focusFrameByInputId[inputId] || {};
        const safe = Math.max(0, Math.round(frame || 0));
        return {
          focusFrameByInputId: {
            ...state.focusFrameByInputId,
            [inputId]: {
              ...byInput,
              [clipId]: safe,
            },
          },
        };
      }),
    clearFocusFrame: (inputId, clipId) =>
      set((state) => {
        const byInput = state.focusFrameByInputId[inputId] || {};
        const { [clipId]: _removed, ...rest } = byInput;
        const next = { ...state.focusFrameByInputId };
        if (Object.keys(rest).length === 0) {
          delete next[inputId];
        } else {
          next[inputId] = rest;
        }
        return { focusFrameByInputId: next };
      }),
    getFocusFrame: (inputId, clipId) => {
      const byInput = get().focusFrameByInputId[inputId];
      const val = byInput && byInput[clipId];
      if (val != null) return val;
      const [start] = get().getTimelineDuration(inputId, clipId);
      return start;
    },

    setFocusAnchorRatio: (ratio, inputId, clipId) =>
      set((state) => {
        const byInput = state.focusAnchorRatioByInputId[inputId] || {};
        const safe = _.clamp(Number(ratio) || 0.5, 0, 1);
        return {
          focusAnchorRatioByInputId: {
            ...state.focusAnchorRatioByInputId,
            [inputId]: {
              ...byInput,
              [clipId]: safe,
            },
          },
        };
      }),
    getFocusAnchorRatio: (inputId, clipId) => {
      const byInput = get().focusAnchorRatioByInputId[inputId];
      const val = byInput && byInput[clipId];
      return val != null ? val : 0.5;
    },

    getZoomLevel: (inputId, clipId) => {
      const byInput = get().zoomLevelByInputId[inputId];
      const val = byInput && byInput[clipId];
      // Default zoom level is 1.
      return (val != null ? val : (1 as ZoomLevel)) as ZoomLevel;
    },

    // Playback state helpers
    setIsPlaying: (isPlaying, inputId, clipId) =>
      set((state) => {
        const byInput = state.isPlayingByInputId[inputId] || {};
        return {
          isPlayingByInputId: {
            ...state.isPlayingByInputId,
            [inputId]: {
              ...byInput,
              [clipId]: !!isPlaying,
            },
          },
        };
      }),
    getIsPlaying: (inputId, clipId) => {
      const byInput = get().isPlayingByInputId[inputId];
      const val = byInput && byInput[clipId];
      return !!val;
    },

    // Playback controls are wired by `wirePlaybackControls` below.
    play: (_inputId: string, _clipId: string) => {
      // no-op placeholder; replaced by wirePlaybackControls
    },
    pause: (_inputId: string, _clipId: string) => {
      // no-op placeholder; replaced by wirePlaybackControls
    },
  }));

wirePlaybackControls(globalInputControlsStore);


type InputControlsContextValue = {
  store: StoreApi<InputControlStore>;
  clipId: string;
};

// React context carrying the per-clip InputControlStore instance + clipId.
const InputControlsContext =
  createContext<InputControlsContextValue | null>(null);

/**
 * Provider that scopes an InputControlStore instance to a logical owner
 * (e.g. a Model clip). The store is created once per provider mount.
 */
export const InputControlsProvider: React.FC<{
  clipId: string;
  children: React.ReactNode;
}> = ({ clipId, children }) => {

  return React.createElement(
    InputControlsContext.Provider,
    { value: { store: globalInputControlsStore, clipId } },
    children,
  );
};

/**
 * Internal hook implementation that:
 * - resolves the backing store + clipId from context (or global fallback)
 * - projects the internal clip-aware state into a per-clip façade
 */
function useInputControlsStoreImpl(): PerClipInputControlsStore;
function useInputControlsStoreImpl<TSelected>(
  selector: (state: PerClipInputControlsStore) => TSelected,
): TSelected;
function useInputControlsStoreImpl<TSelected = PerClipInputControlsStore>(
  selector?: (state: PerClipInputControlsStore) => TSelected,
): TSelected | PerClipInputControlsStore {
  const ctx = useContext(InputControlsContext);
  const store = ctx?.store ?? globalInputControlsStore;
  const clipId = ctx?.clipId ?? GLOBAL_FALLBACK_CLIP_ID;

  // Subscribe to the underlying store so this component re-renders on updates.
  // We ignore the returned base state and instead expose a stable, projected
  // per-clip view with getters that read from the latest store state.
  useStore(store);
  const projected = getProjectedStoreForClip(store, clipId);

  if (!selector) {
    return projected;
  }

  return selector(projected);
}

// Attach static helpers that operate on the global fallback store.
useInputControlsStoreImpl.getState = (): PerClipInputControlsStore =>
  getProjectedStoreForClip(globalInputControlsStore, GLOBAL_FALLBACK_CLIP_ID);

useInputControlsStoreImpl.setState = (
  _partial:
    | PerClipInputControlsStore
    | Partial<PerClipInputControlsStore>
    | ((
        state: PerClipInputControlsStore,
      ) => PerClipInputControlsStore | Partial<PerClipInputControlsStore>),
  _replace?: boolean,
): void => {
  // For now we do not attempt to project arbitrary partials back into the
  // internal store. This static is kept for API compatibility but is a no-op.
};

/**
 * Hook-compatible facade that mirrors the original useInputControlsStore
 * API. Under the hood, it resolves the current store from context, falling
 * back to the legacy global store when no provider is present.
 *
 * - useInputControlsStore(selector) → selector result
 * - useInputControlsStore() → full store state
 *
 * Note: direct static access (useInputControlsStore.getState / setState)
 * continues to operate on the global fallback store. Callers that need
 * per-clip semantics should prefer the hook form within a provider.
 */
type UseInputControlsStoreHook = {
  (): PerClipInputControlsStore;
  <TSelected>(
    selector: (state: PerClipInputControlsStore) => TSelected,
  ): TSelected;
  getState: () => PerClipInputControlsStore;
  setState: (
    partial:
      | PerClipInputControlsStore
      | Partial<PerClipInputControlsStore>
      | ((
          state: PerClipInputControlsStore,
        ) => PerClipInputControlsStore | Partial<PerClipInputControlsStore>),
    replace?: boolean,
  ) => void;
};

// Re-export under the original name used throughout the app.
// eslint-disable-next-line @typescript-eslint/naming-convention
export const useInputControlsStore: UseInputControlsStoreHook = ((
  selector?: (state: PerClipInputControlsStore) => unknown,
) => useInputControlsStoreImpl(selector as any)) as UseInputControlsStoreHook;

useInputControlsStore.getState = useInputControlsStoreImpl.getState;
useInputControlsStore.setState = useInputControlsStoreImpl.setState;
