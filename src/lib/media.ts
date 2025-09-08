import { MediaInfo } from "./types";
import { create, useStore } from "zustand";
import { FRAMES_CACHE_MAX_BYTES } from "./settings";
import { Input, ALL_FORMATS, UrlSource, VideoSampleSink, VideoSample, CanvasSink, WrappedCanvas } from 'mediabunny';

interface MediaCache {
    media: Record<string, MediaInfo>;
    setMedia: (path: string, metadata: MediaInfo) => void;
    getMedia: (path: string) => MediaInfo | undefined;
    isMediaCached: (path: string) => boolean;
    thumbnailCache: Record<string, HTMLCanvasElement>;
    setThumbnailCache: (path: string, thumbnail: HTMLCanvasElement) => void;
    getThumbnailCache: (path: string) => HTMLCanvasElement | undefined;
    isThumbnailCached: (path: string) => boolean;
}

const MediaCache = create<MediaCache>((set, get) => ({
    media: {},
    setMedia: (path: string, metadata: MediaInfo) => set({ media: { ...get().media, [path]: metadata } }),
    getMedia: (path: string) => get().media[path],
    isMediaCached: (path: string) => !!get().getMedia(path),
    thumbnailCache: {},
    setThumbnailCache: (path: string, thumbnail: HTMLCanvasElement) => set({ thumbnailCache: { ...get().thumbnailCache, [path]: thumbnail } }),
    getThumbnailCache: (path: string) => get().thumbnailCache[path],
    isThumbnailCached: (path: string) => !!get().getThumbnailCache(path),
}));

export const useMediaCache = () => {
    return useStore(MediaCache);
}

export const getMediaInfo = async (path: string): Promise<MediaInfo> => {

    // check if the metadata is cached
    const mediaCache = MediaCache.getState();
    if (mediaCache.isMediaCached(path)) {
        return mediaCache.getMedia(path)!;
    }

    const input = new Input({
        formats: ALL_FORMATS,
        source: new UrlSource(path),
    });

    const videoTrack = await input.getPrimaryVideoTrack();
    const audioTrack = await input.getPrimaryAudioTrack();
    
    const packetStats = await videoTrack?.computePacketStats(100);
    const audioPacketStats = await audioTrack?.computePacketStats(100);
    const duration = await input.computeDuration();
    const metadata = await input.getMetadataTags();
    const mimeType = await input.getMimeType();
    const format = await input.getFormat();

    const mediaInfo: MediaInfo = {
        path,
        video: videoTrack,
        audio: audioTrack,
        stats: {
            video: packetStats,
            audio: audioPacketStats,
        },
        duration,
        metadata,
        mimeType,
        format,
    }

    mediaCache.setMedia(path, mediaInfo);
    return mediaInfo;
}


// ========================= Frames LRU Cache =========================

type FrameKey = string; // `${path}@${width}x${height}#${frameIndex}`

type FrameCacheEntry = VideoSample | WrappedCanvas;

interface FramesCacheStore {
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

const buildFrameKey = (path: string, width: number, height: number, frameIndex: number, useCanvas?: boolean): FrameKey => `${path}@${width}x${height}#${frameIndex}#${useCanvas ? 'canvas' : 'video'}`;

const FramesCache = create<FramesCacheStore>((set, get) => ({
    capacityBytes: FRAMES_CACHE_MAX_BYTES,
    totalBytes: 0,
    frames: new Map(),
    version: 0,
    setCapacityBytes: (bytes: number) => set({ capacityBytes: Math.max(16 * 1024 * 1024, bytes) }),
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
            if (prev instanceof VideoSample) {
                state.totalBytes -= prev.allocationSize();
            } else {
                state.totalBytes -= prev.canvas.width * prev.canvas.height * 4;
            }
            frames.delete(key);
        }
        frames.set(key, entry);
        const totalBytes = state.totalBytes + (entry instanceof VideoSample ? entry.allocationSize() : entry.canvas.width * entry.canvas.height * 4);
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
                state.totalBytes -= removed instanceof VideoSample ? removed.allocationSize() : removed.canvas.width * removed.canvas.height * 4;
            }
        }
        set({ frames, totalBytes: state.totalBytes, version: state.version + 1 });
    },
    clear: () => set({ frames: new Map(), totalBytes: 0, version: get().version + 1 }),
}));

export const useFramesCache = () => useStore(FramesCache);

// Reserved for future use when generating placeholder frames
// const createBlackFrame = (width: number, height: number): Uint8Array => {
//     const bytes = new Uint8Array(width * height * 4);
//     for (let i = 3; i < bytes.length; i += 4) bytes[i] = 255;
//     return bytes;
// };


export const getCachedSample = (path: string, frameIndex: number, width?: number, height?: number, useCanvas?: boolean): VideoSample | WrappedCanvas | null => {
    const frameCache = FramesCache.getState();
    const mediaCache = MediaCache.getState();

    height = height || mediaCache.getMedia(path)?.video?.codedHeight || 0;
    width = width || mediaCache.getMedia(path)?.video?.codedWidth || 0;

    const key = buildFrameKey(path, width, height, frameIndex, useCanvas);

    if (frameCache.has(key)) {
        return frameCache.get(key)!;
    }

    return null
}

export const getCachedSamples = (path: string, frameIndices: number[], width?: number, height?: number, useCanvas?: boolean): (VideoSample | WrappedCanvas | null)[] => {
    const samples: (VideoSample | WrappedCanvas | null)[] = new Array(frameIndices.length).fill(null);
    for (let i = 0; i < frameIndices.length; i++) {
        samples[i] = getCachedSample(path, frameIndices[i], width, height, useCanvas) as VideoSample | WrappedCanvas;
    }
    return samples;
}

export const fetchSample = async (path: string, frameIndex: number, width?: number, height?: number, options?: {mediaInfo?: MediaInfo}): Promise<VideoSample | null> => {
    const frame = getCachedSample(path, frameIndex, width, height);
    if (frame) return frame as VideoSample;

    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    
    if (!mediaInfo) return null;

    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;

    const key = buildFrameKey(path, width, height, frameIndex);
    const sink = new VideoSampleSink(mediaInfo.video!);

    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    
    const timestamp = frameIndex / frameRate;

    const sample = await sink.getSample(timestamp);

    // cache the sample
    if (sample) {
        const frameCache = FramesCache.getState();
        frameCache.put(key, sample);
    }

    return sample;
}

export const fetchSamples = async (path: string, frameIndices: number[], width?: number, height?: number, options?: {mediaInfo?: MediaInfo}): Promise<(VideoSample | null)[]> => {
    // make samples a list of nulls to start with
    const samples = getCachedSamples(path, frameIndices, width, height);
   
    if (samples.every(sample => sample !== null)) {
        return samples as (VideoSample | null)[];
    }
  
    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo) return samples as (VideoSample | null)[];

    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;

    const sink = new VideoSampleSink(mediaInfo.video!);

    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    
    // only get frame indices that are not cached
    const frameIndexToIndex = new Map<number, number>();
    for (let i = 0; i < frameIndices.length; i++) {
        if (!samples[i]) {
            frameIndexToIndex.set(frameIndices[i], i);
        }
    }

    if (frameIndexToIndex.size === 0) {
        return samples as (VideoSample | null)[];
    }

    const sparseTimestamps = Array.from(frameIndexToIndex.keys()).map(idx => idx / frameRate);
    const sparseSamples = await sink.samplesAtTimestamps(sparseTimestamps);

    const frameCache = FramesCache.getState();

    for await (const sample of sparseSamples) {
        if (sample) {
            const estimatedFrameIndex = Math.round(sample.timestamp * frameRate);
            let index = frameIndexToIndex.get(estimatedFrameIndex);
            if (index === undefined) {
                index = frameIndexToIndex.get(estimatedFrameIndex - 1) ?? frameIndexToIndex.get(estimatedFrameIndex + 1);
            }
            if (index !== undefined) {
                samples[index] = sample;
                const frameKey = buildFrameKey(path, width, height, estimatedFrameIndex);
                frameCache.put(frameKey, sample);
            }
        }
    }

    return samples as (VideoSample | null)[];
   
}

export const prefetchSamples = async (path: string, frameIndices: number[], width?: number, height?: number, options?: {mediaInfo?: MediaInfo}): Promise<void> => {
    await fetchSamples(path, frameIndices, width, height, options);
}




export const fetchCanvasSample = async (path: string, frameIndex: number, width?: number, height?: number, options?: {mediaInfo?: MediaInfo}): Promise<WrappedCanvas | null> => {
    const frame = getCachedSample(path, frameIndex, width, height, true);
    if (frame) return frame as WrappedCanvas;

    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    
    if (!mediaInfo) return null;

    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;

    const key = buildFrameKey(path, width, height, frameIndex, true);
    const sink = new CanvasSink(mediaInfo.video!, {
        width,
        height,
        fit: 'fill',
    });

    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    
    const timestamp = frameIndex / frameRate;

    const sample = await sink.getCanvas(timestamp);

    // cache the sample
    if (sample) {
        const frameCache = FramesCache.getState();
        frameCache.put(key, sample);
    }

    return sample;
}

export const fetchCanvasSamples = async (path: string, frameIndices: number[], width?: number, height?: number, options?: {mediaInfo?: MediaInfo}): Promise<(WrappedCanvas | null)[]> => {
    // make samples a list of nulls to start with
    const samples = getCachedSamples(path, frameIndices, width, height, true);
   
    if (samples.every(sample => sample !== null)) {
        return samples as (WrappedCanvas | null)[];
    }
  
    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo) return samples as (WrappedCanvas | null)[];

    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;

    const sink = new CanvasSink(mediaInfo.video!, {
        width,
        height,
        fit: 'fill',
    });

    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    
    // only get frame indices that are not cached
    const frameIndexToIndex = new Map<number, number>();
    for (let i = 0; i < frameIndices.length; i++) {
        if (!samples[i]) {
            frameIndexToIndex.set(frameIndices[i], i);
        }
    }

    if (frameIndexToIndex.size === 0) {
        return samples as (WrappedCanvas | null)[];
    }

    const sparseTimestamps = Array.from(frameIndexToIndex.keys()).map(idx => idx / frameRate);
    const sparseSamples = await sink.canvasesAtTimestamps(sparseTimestamps);

    const frameCache = FramesCache.getState();

    for await (const sample of sparseSamples) {
        if (sample) {
            const estimatedFrameIndex = Math.round(sample.timestamp * frameRate);
            let index = frameIndexToIndex.get(estimatedFrameIndex);
            if (index === undefined) {
                index = frameIndexToIndex.get(estimatedFrameIndex - 1) ?? frameIndexToIndex.get(estimatedFrameIndex + 1);
            }
            if (index !== undefined) {
                samples[index] = sample;
                const frameKey = buildFrameKey(path, width, height, estimatedFrameIndex, true);
                frameCache.put(frameKey, sample);
            }
        }
    }

    return samples as (WrappedCanvas | null)[];
   
}


const createThumbnailKey = (
    path: string,
    width: number,
    height: number,
    totalCanvasWidth: number,
    frameIndices: number[]
): string => `${path}@${width}x${height}#w${totalCanvasWidth}#${frameIndices.join(',')}`;

export const generateTimelineThumbnail = async (path: string, frameIndices: number[], width: number, height: number, totalCanvasWidth: number, options?: {canvas?: HTMLCanvasElement, mediaInfo?: MediaInfo}): Promise<CanvasImageSource | null> => {
    const thumbnailKey = createThumbnailKey(path, width, height, totalCanvasWidth, frameIndices);

    const mediaCache = MediaCache.getState();

    // If caller provided a canvas, bypass cache lookup to ensure immediate redraw on same canvas
    if (!options?.canvas && mediaCache.isThumbnailCached(thumbnailKey)) {
        return mediaCache.getThumbnailCache(thumbnailKey)!;
    }

    const fetchedSamples = await fetchCanvasSamples(path, frameIndices, width, height, options);
    // Filter out invalid or zero-dimension samples to avoid non-progressing loops
    const samples = fetchedSamples.filter((s) => s && s.canvas && s.canvas.width > 0 && s.canvas.height > 0) as WrappedCanvas[];
    if (samples.length === 0) return null;
    // concatenate the samples into a single canvas
    const patternWidth = samples.reduce((acc, sample) => acc + sample.canvas.width, 0);
    if (!Number.isFinite(patternWidth) || patternWidth <= 0) return null;
    const canvas = options?.canvas || document.createElement('canvas');
    canvas.width = Math.max(1, Math.floor(totalCanvasWidth));
    canvas.height = Math.max(1, Math.floor(height));
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let x = 0;
    // Draw a single pass of samples
    for (const sample of samples) {
        if (x >= canvas.width) break;
        ctx.drawImage(sample.canvas, x, 0, sample.canvas.width, sample.canvas.height);
        x += sample.canvas.width;
    }
    // Fill the remaining width safely by cycling through samples
    if (x < canvas.width) {
        const minTileWidth = Math.min(...samples.map(s => s.canvas.width));
        if (minTileWidth > 0) {
            const maxTiles = Math.ceil((canvas.width - x) / minTileWidth) + samples.length + 1;
            let tiles = 0;
            let idx = 0;
            while (x < canvas.width && tiles < maxTiles) {
                const sample = samples[idx % samples.length];
                const w = sample.canvas.width;
                if (w > 0) {
                    ctx.drawImage(sample.canvas, x, 0, w, sample.canvas.height);
                    x += w;
                }
                idx++;
                tiles++;
            }
        }
    }

    // Only cache when we created a new canvas internally. If a canvas is supplied, skip caching
    if (!options?.canvas) {
        mediaCache.setThumbnailCache(thumbnailKey, canvas);
    }

    return canvas;
}