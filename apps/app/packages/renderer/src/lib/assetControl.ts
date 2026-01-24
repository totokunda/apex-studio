import { create } from "zustand";
import { ZoomLevel, AnyClipProps } from "./types";
import {
  DEFAULT_FPS,
  TIMELINE_DURATION_SECONDS,
  MIN_DURATION,
} from "./settings";

interface AssetControlStore {
  // Zoom state (independent from main controls)
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

  // FPS (kept separate in case assets want their own cadence)
  fps: number;
  setFps: (fps: number) => void;

  // Focus state (kept separate to avoid interfering with main focus)
  focusFrame: number;
  setFocusFrame: (frame: number) => void;
  focusAnchorRatio: number;
  setFocusAnchorRatio: (ratio: number) => void;

  // Asset selection is distinct from main selectedClipIds
  selectedAssetClipId: string | null;
  setSelectedAssetClipId: (clipId: string | null) => void;
  clearSelectedAsset: () => void;
  // Optional handler invoked whenever setSelectedAssetClipId is called (including deselect with null)
  selectedAssetChangeHandler: ((clipId: string | null) => void) | null;
  setSelectedAssetChangeHandler: (
    handler: ((clipId: string | null) => void) | null,
  ) => void;
}

export const useAssetControlsStore = create<AssetControlStore>((set, get) => ({
  // Zoom
  maxZoomLevel: 10,
  setMaxZoomLevel: (level) => set({ maxZoomLevel: level }),
  minZoomLevel: 1,
  setMinZoomLevel: (level) => set({ minZoomLevel: level }),
  zoomLevel: 1,
  setZoomLevel: (level) => set({ zoomLevel: level }),

  // Timeline duration/window
  totalTimelineFrames: TIMELINE_DURATION_SECONDS * DEFAULT_FPS,
  setTotalTimelineFrames: (frames) =>
    set({ totalTimelineFrames: Math.max(1, Math.round(frames)) }),
  incrementTotalTimelineFrames: (frames) =>
    set((s) => ({
      totalTimelineFrames: Math.max(
        1,
        s.totalTimelineFrames + Math.round(frames),
      ),
    })),
  decrementTotalTimelineFrames: (frames) =>
    set((s) => ({
      totalTimelineFrames: Math.max(
        1,
        s.totalTimelineFrames - Math.round(frames),
      ),
    })),
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
    const desiredStart = Math.max(
      0,
      Math.min(total - win, start + Math.trunc(delta)),
    );
    set({ timelineDuration: [desiredStart, desiredStart + win] });
  },

  // FPS
  fps: DEFAULT_FPS,
  setFps: (fps) => set({ fps: Math.max(1, Math.round(fps)) }),

  // Focus
  focusFrame: 0,
  setFocusFrame: (frame) => set({ focusFrame: Math.max(0, Math.round(frame)) }),
  focusAnchorRatio: 0.5,
  setFocusAnchorRatio: (ratio) =>
    set({ focusAnchorRatio: Math.max(0, Math.min(1, ratio)) }),

  // Asset selection
  selectedAssetClipId: null,
  setSelectedAssetClipId: (clipId) => {
    set({ selectedAssetClipId: clipId });
    try {
      const handler = get().selectedAssetChangeHandler;
      if (typeof handler === "function") handler(clipId);
    } catch {}
  },
  clearSelectedAsset: () => set({ selectedAssetClipId: null }),
  selectedAssetChangeHandler: null,
  setSelectedAssetChangeHandler: (handler) =>
    set({ selectedAssetChangeHandler: handler ?? null }),
}));

// Mirrors the main controls' zoom baseline update, but scoped to the asset controls store.
// Sets a baseline totalTimelineFrames so that the longest clip occupies ~60% of the window at zoom level 1.
// Then preserves the current window where possible by recalibrating the zoom level to the closest step.
export function updateAssetZoomLevel(
  clips: AnyClipProps[],
  _clipDuration: number,
) {
  try {
    // Require clips; do not rely on external duration state
    if (!Array.isArray(clips) || clips.length === 0) return;

    // Compute longest clip end solely from provided clips
    let longestClipFrames = 0;
    for (let i = 0; i < clips.length; i++) {
      const end = Math.max(0, clips[i].endFrame || 0);
      if (end > longestClipFrames) longestClipFrames = end;
    }

    // Baseline so the longest clip occupies ~60% of the window at zoom level 1
    const newTotalFrames = Math.max(
      MIN_DURATION,
      Math.round((longestClipFrames * 5) / 3),
    );

    const controls = useAssetControlsStore.getState();
    const prevTotalFrames = Math.max(0, controls.totalTimelineFrames || 0);

    if (newTotalFrames === prevTotalFrames) return;

    // Apply baseline and explicitly reset the asset window without referencing the current one
    useAssetControlsStore.setState({ totalTimelineFrames: newTotalFrames });
    controls.setTimelineDuration(0, newTotalFrames);
    controls.setZoomLevel(1 as any);
  } catch {}
}
