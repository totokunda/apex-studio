import { MediaInfo } from "../types";
import { create, useStore } from "zustand";
import { VideoSample, WrappedCanvas, WrappedAudioBuffer } from "mediabunny";
import { FRAMES_CACHE_MAX_BYTES } from "../settings";

export interface MediaCache {
  media: Record<string, MediaInfo>;
  setMedia: (path: string, metadata: MediaInfo) => void;
  getMedia: (path: string) => MediaInfo | undefined;
  isMediaCached: (path: string) => boolean;
  thumbnailCache: Record<string, HTMLCanvasElement>;
  setThumbnailCache: (path: string, thumbnail: HTMLCanvasElement) => void;
  getThumbnailCache: (path: string) => HTMLCanvasElement | undefined;
  isThumbnailCached: (path: string) => boolean;
}

export const MediaCache = create<MediaCache>((set, get) => ({
  media: {},
  setMedia: (path: string, metadata: MediaInfo) =>
    set({ media: { ...get().media, [path]: metadata } }),
  getMedia: (path: string) => get().media[path],
  isMediaCached: (path: string) => !!get().getMedia(path),
  thumbnailCache: {},
  setThumbnailCache: (path: string, thumbnail: HTMLCanvasElement) =>
    set({ thumbnailCache: { ...get().thumbnailCache, [path]: thumbnail } }),
  getThumbnailCache: (path: string) => get().thumbnailCache[path],
  isThumbnailCached: (path: string) => !!get().getThumbnailCache(path),
}));

export const useMediaCache = () => {
  return useStore(MediaCache);
};

// ========================= Frames LRU Cache =========================

export type FrameKey = string; // `${path}@${width}x${height}#${frameIndex}`

export type WrappedAmplitudes = {
  amplitudes: Float32Array;
};

export type FrameCacheEntry =
  | VideoSample
  | WrappedCanvas
  | WrappedAudioBuffer
  | WrappedAmplitudes;

export interface FramesCacheStore {
  capacityBytes: number;
  totalBytes: number;
  frames: Map<FrameKey, FrameCacheEntry>;
  version: number;
  setCapacityBytes: (bytes: number) => void;
  has: (key: FrameKey) => boolean;
  get: (key: FrameKey) => FrameCacheEntry | undefined;
  put: (key: FrameKey, entry: FrameCacheEntry) => void;
  ensureCapacity: () => void;
  clear: () => void;
}

export const buildFrameKey = (
  path: string,
  width: number,
  height: number,
  frameIndex: number,
  useCanvas?: boolean,
): FrameKey =>
  `${path}@${width}x${height}#${frameIndex}#${useCanvas ? "canvas" : "video"}`;
export const buildImageKey = (
  path: string,
  width: number,
  height: number,
): FrameKey => `${path}@${width}x${height}#image`;

export const FramesCache = create<FramesCacheStore>((set, get) => ({
  capacityBytes: FRAMES_CACHE_MAX_BYTES,
  totalBytes: 0,
  frames: new Map(),
  version: 0,
  setCapacityBytes: (bytes: number) =>
    set({ capacityBytes: Math.max(16 * 1024 * 1024, bytes) }),
  has: (key: FrameKey) => get().frames.has(key),
  get: (key: FrameKey) => {
    const frames = get().frames;
    const found = frames.get(key);
    if (found) {
      // touch as MRU: re-insert to move to end
      frames.delete(key);
      frames.set(key, found);
      set({ frames, version: get().version + 1 });
    }
    return found;
  },
  put: (key: FrameKey, entry: FrameCacheEntry) => {
    const state = get();
    const frames = state.frames;
    if (frames.has(key)) {
      const prev = frames.get(key)!;
      // Remove previous entry accounting and close underlying resources if no other references remain
      if (prev instanceof VideoSample) {
        state.totalBytes -= prev.allocationSize();
      } else if (isWrappedAudioBuffer(prev)) {
        state.totalBytes -= estimateAudioBufferSize(prev);
      } else if (isWrappedAmplitudes(prev)) {
        state.totalBytes -= estimateAmplitudesSize(prev);
      } else {
        state.totalBytes -= prev.canvas.width * prev.canvas.height * 4;
      }
      frames.delete(key);
      // Only close the underlying sample if no other cache entry references the same object
      if (prev instanceof VideoSample) {
        let stillReferenced = false;
        for (const value of frames.values()) {
          if (value === prev) {
            stillReferenced = true;
            break;
          }
        }
        if (!stillReferenced) {
          try {
            (prev as VideoSample).close();
          } catch {}
        }
      }
    }
    frames.set(key, entry);
    const totalBytes =
      state.totalBytes +
      (entry instanceof VideoSample
        ? entry.allocationSize()
        : isWrappedAudioBuffer(entry)
          ? estimateAudioBufferSize(entry)
          : isWrappedAmplitudes(entry)
            ? estimateAmplitudesSize(entry)
            : entry.canvas.width * entry.canvas.height * 4);
    set({ frames, totalBytes, version: state.version + 1 });
    get().ensureCapacity();
  },
  ensureCapacity: () => {
    const state = get();
    const frames = state.frames;
    while (state.totalBytes > state.capacityBytes && frames.size > 0) {
      const oldestKey = frames.keys().next().value as FrameKey;
      const removed = frames.get(oldestKey);
      frames.delete(oldestKey);
      if (removed) {
        state.totalBytes -=
          removed instanceof VideoSample
            ? removed.allocationSize()
            : isWrappedAudioBuffer(removed)
              ? estimateAudioBufferSize(removed)
              : isWrappedAmplitudes(removed)
                ? estimateAmplitudesSize(removed)
                : removed.canvas.width * removed.canvas.height * 4;
        // Only close the sample if no other cache entry references the same object
        if (removed instanceof VideoSample) {
          let stillReferenced = false;
          for (const value of frames.values()) {
            if (value === removed) {
              stillReferenced = true;
              break;
            }
          }
          if (!stillReferenced) {
            try {
              (removed as VideoSample).close();
            } catch {}
          }
        }
      }
    }
    set({ frames, totalBytes: state.totalBytes, version: state.version + 1 });
  },
  clear: () => {
    const state = get();
    const frames = state.frames;
    for (const entry of frames.values()) {
      if (entry instanceof VideoSample) {
        try {
          (entry as VideoSample).close();
        } catch {}
      }
    }
    set({ frames: new Map(), totalBytes: 0, version: state.version + 1 });
  },
}));

export const useFramesCache = () => useStore(FramesCache);

export const getCachedSample = (
  path: string,
  frameIndex: number,
  width?: number,
  height?: number,
  useCanvas?: boolean,
): VideoSample | WrappedCanvas | null => {
  const frameCache = FramesCache.getState();
  const mediaCache = MediaCache.getState();

  height = height || mediaCache.getMedia(path)?.video?.displayHeight || 0;
  width = width || mediaCache.getMedia(path)?.video?.displayWidth || 0;

  const key = buildFrameKey(path, width, height, frameIndex, useCanvas);

  if (frameCache.has(key)) {
    const found = frameCache.get(key);
    if (found instanceof VideoSample) return found;
    // Narrow type: treat any object with a canvas property as WrappedCanvas
    if (found && (found as any).canvas) return found as WrappedCanvas;
  }

  return null;
};

export const getCachedSamples = (
  path: string,
  frameIndices: number[],
  width?: number,
  height?: number,
  useCanvas?: boolean,
): (VideoSample | WrappedCanvas | null)[] => {
  const samples: (VideoSample | WrappedCanvas | null)[] = new Array(
    frameIndices.length,
  ).fill(null);
  for (let i = 0; i < frameIndices.length; i++) {
    samples[i] = getCachedSample(
      path,
      frameIndices[i],
      width,
      height,
      useCanvas,
    ) as VideoSample | WrappedCanvas;
  }
  return samples;
};

export const getCachedImage = (path: string, width: number, height: number) => {
  const imageKey = buildImageKey(path, width, height);

  const frameCache = FramesCache.getState();

  return frameCache.get(imageKey);
};

// ========================= Audio keys and helpers =========================

export const buildAudioKey = (
  path: string,
  sampleFrameIndex: number,
): FrameKey => `${path}@audio#${sampleFrameIndex}`;

export const getCachedAudioBuffer = (
  path: string,
  sampleFrameIndex: number,
): WrappedAudioBuffer | null => {
  const key = buildAudioKey(path, sampleFrameIndex);
  const frameCache = FramesCache.getState();
  const found = frameCache.get(key);
  return isWrappedAudioBuffer(found) ? (found as WrappedAudioBuffer) : null;
};

export const getCachedAudioBuffers = (
  path: string,
  sampleFrameIndices: number[],
): (WrappedAudioBuffer | null)[] => {
  const frameCache = FramesCache.getState();
  const results: (WrappedAudioBuffer | null)[] = new Array(
    sampleFrameIndices.length,
  ).fill(null);
  for (let i = 0; i < sampleFrameIndices.length; i++) {
    const key = buildAudioKey(path, sampleFrameIndices[i]!);
    const found = frameCache.get(key);
    results[i] = isWrappedAudioBuffer(found)
      ? (found as WrappedAudioBuffer)
      : null;
  }
  return results;
};

// ========================= Size estimation =========================

function isWrappedAudioBuffer(entry: unknown): entry is WrappedAudioBuffer {
  const e: any = entry as any;
  return (
    !!e &&
    typeof e === "object" &&
    "buffer" in e &&
    e.buffer &&
    typeof (e.buffer as any).getChannelData === "function"
  );
}

function estimateAudioBufferSize(sample: WrappedAudioBuffer): number {
  try {
    const buffer = sample.buffer as AudioBuffer;
    const channels = buffer.numberOfChannels || 1;
    const frames = buffer.length || 0;
    // Float32 per sample by default
    return Math.max(0, frames * channels * 4);
  } catch {
    return 0;
  }
}

function isWrappedAmplitudes(entry: unknown): entry is WrappedAmplitudes {
  const e: any = entry as any;
  return (
    !!e &&
    typeof e === "object" &&
    "amplitudes" in e &&
    e.amplitudes instanceof Float32Array
  );
}

function estimateAmplitudesSize(sample: WrappedAmplitudes): number {
  try {
    // Float32 per sample
    return Math.max(0, sample.amplitudes.length * 4);
  } catch {
    return 0;
  }
}
