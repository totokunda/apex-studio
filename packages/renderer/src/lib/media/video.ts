import { MediaInfo } from "../types";
import { VideoDecoderContext, VideoDecoderKey } from "./types";
import { VideoSampleSink, VideoSample } from "mediabunny";
import { nowMs } from "./utils";
import { FramesCache, MediaCache } from "./cache";
import { PREFETCH_BACK, PREFETCH_AHEAD } from "../settings";
import { buildFrameKey, getCachedSample, getCachedSamples } from "./cache";
import {videoDecoders, pruneStaleDecoders} from "./utils";

// ========================= Idle decoding scheduler =========================
type IdleTarget = {
    path: string;
    mediaInfo: MediaInfo;
    width: number;
    height: number;
    centerFrame: number;
    nextRadius: number;
    lastTouchedTs: number;
};

const idleTargets = new Map<string, IdleTarget>();
let idleScheduled = false;
let lastPruneTs = 0;

function getFrameRate(mi: MediaInfo): number {
    return mi.stats.video?.averagePacketRate || 0;
}

function timeRemaining(deadline?: any): number {
    try {
        if (deadline && typeof deadline.timeRemaining === 'function') return deadline.timeRemaining();
    } catch {}
    return 10; // ms fallback
}

function scheduleNextIdle() {
    if (idleScheduled) return;
    idleScheduled = true;
    const ric: any = (globalThis as any).requestIdleCallback;
    if (typeof ric === 'function') {
        ric(runIdle, { timeout: 50 });
    } else {
        setTimeout(() => runIdle({ timeRemaining: () => 8 }), 16);
    }
}

async function scheduleIdlePrefetchStep(target: IdleTarget, maxFrames: number): Promise<void> {
    const ctx = getOrCreateVideoDecoder(target.path, target.mediaInfo);
    if (!ctx) return;
    const frameRate = ctx.frameRate || getFrameRate(target.mediaInfo) || 0;
    if (!Number.isFinite(frameRate) || frameRate <= 0) return;
    const totalFrames = Math.floor((target.mediaInfo.duration || 0) * frameRate);

    const frameCache = FramesCache.getState();
    const candidates: number[] = [];
    // Expand symmetrically around center using nextRadius cursor
    while (candidates.length < maxFrames && target.nextRadius < totalFrames) {
        const r = ++target.nextRadius;
        const a = target.centerFrame + r;
        const b = target.centerFrame - r;
        if (a >= 0 && a < totalFrames && !ctx.inFlight.has(a)) {
            const keyA = buildFrameKey(target.path, target.width, target.height, a);
            if (!frameCache.has(keyA)) candidates.push(a);
        }
        if (candidates.length >= maxFrames) break;
        if (b >= 0 && b < totalFrames && !ctx.inFlight.has(b)) {
            const keyB = buildFrameKey(target.path, target.width, target.height, b);
            if (!frameCache.has(keyB)) candidates.push(b);
        }
    }

    if (candidates.length === 0) return;

    candidates.forEach(f => ctx.inFlight.add(f));
    try {
        const timestamps = candidates.map(f => f / frameRate);
        const iterable = await ctx.sink.samplesAtTimestamps(timestamps) as AsyncIterable<VideoSample | null>;
        for await (const sample of iterable) {
            if (!sample) continue;
            const estIndex = Math.round(sample.timestamp * frameRate);
            const key = buildFrameKey(target.path, target.width, target.height, estIndex);
            frameCache.put(key, sample);
            ctx.inFlight.delete(estIndex);
        }
    } catch (e) {
        // swallow errors, ensure we clear inFlight below
    } finally {
        // Clear any remaining in-flight frames so future attempts can retry
        candidates.forEach(f => ctx.inFlight.delete(f));
    }
}

function runIdle(deadline?: any) {
    idleScheduled = false;
    if (idleTargets.size === 0) return;

    const budgetMs = Math.max(4, Math.min(12, timeRemaining(deadline)));
    const endBy = (typeof performance !== 'undefined' && performance.now) ? performance.now() + budgetMs : Date.now() + budgetMs;

    const targets = Array.from(idleTargets.values()).sort((a, b) => a.lastTouchedTs - b.lastTouchedTs);

    const stepPromises: Promise<void>[] = [];
    for (const t of targets) {
        // Limit per-loop work; 2-3 frames per target keeps UI responsive
        stepPromises.push(scheduleIdlePrefetchStep(t, 3));
        if (((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now()) >= endBy) break;
    }

    void Promise.allSettled(stepPromises).then(() => {
        const now = nowMs();
        if (now - lastPruneTs > 2000) {
            lastPruneTs = now;
            pruneStaleDecoders();
        }
        scheduleNextIdle();
    });
}

export function setIdleDecodeTarget(path: string, centerFrame: number, width?: number, height?: number, options?: { mediaInfo?: MediaInfo }): void {
    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo || !mediaInfo.video) return;
    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    const existing = idleTargets.get(path);
    if (existing) {
        const previousCenter = existing.centerFrame;
        existing.centerFrame = centerFrame;
        existing.lastTouchedTs = nowMs();
        existing.width = width;
        existing.height = height;
        // reset expansion when center changes
        if (centerFrame !== previousCenter) {
            existing.nextRadius = 0;
        }
    } else {
        idleTargets.set(path, {
            path,
            mediaInfo,
            width,
            height,
            centerFrame,
            nextRadius: 0,
            lastTouchedTs: nowMs(),
        });
    }
    scheduleNextIdle();
}

export function removeIdleDecodeTarget(path: string): void {
    idleTargets.delete(path);
}

function getOrCreateVideoDecoder(path: string, mediaInfo: MediaInfo): VideoDecoderContext | null {
    if (!mediaInfo?.video) return null;
    const key: VideoDecoderKey = `${path}#video`;
    const existing = videoDecoders.get(key);
    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    if (existing) {
        existing.lastAccessTs = nowMs();
        existing.frameRate = frameRate || existing.frameRate;
        return existing;
    }
    try {
        const sink = new VideoSampleSink(mediaInfo.video);
        const ctx: VideoDecoderContext = {
            sink,
            inFlight: new Set<number>(),
            lastAccessTs: nowMs(),
            frameRate: frameRate || 0,
        };
        videoDecoders.set(key, ctx);
        // Prime decoder lightly (do not await)
        void (async () => {
            try {
                const ts = 0;
                await sink.getSample(ts);
            } catch {}
        })();
        return ctx;
    } catch (e) {
        console.warn('[media] Failed to create VideoSampleSink', { path, error: e });
        return null;
    }
}


function schedulePrefetchVideo(path: string, mediaInfo: MediaInfo, centerFrame: number, width?: number, height?: number) {
    const ctx = getOrCreateVideoDecoder(path, mediaInfo);
    if (!ctx) return;
    const frameRate = ctx.frameRate || mediaInfo.stats.video?.averagePacketRate || 0;
    if (!Number.isFinite(frameRate) || frameRate <= 0) return;
    const start = Math.max(0, Math.floor(centerFrame - PREFETCH_BACK));
    const end = Math.max(start, Math.floor(centerFrame + PREFETCH_AHEAD));
    const frameCache = FramesCache.getState();
    const tasks: number[] = [];
    for (let i = start; i <= end; i++) {
        // Skip the center (likely decoded synchronously) and anything cached or already in-flight
        if (i === centerFrame) continue;
        if (ctx.inFlight.has(i)) continue;
        const w = width || mediaInfo.video?.codedWidth || 0;
        const h = height || mediaInfo.video?.codedHeight || 0;
        if (w > 0 && h > 0) {
            const key = buildFrameKey(path, w, h, i);
            if (frameCache.has(key)) continue;
        }
        tasks.push(i);
    }
    if (tasks.length === 0) return;
    // Mark in-flight
    tasks.forEach(f => ctx.inFlight.add(f));
    void (async () => {
        try {
            const timestamps = tasks.map(f => f / frameRate);
            const iterable = await ctx.sink.samplesAtTimestamps(timestamps) as AsyncIterable<VideoSample | null>;
            const frameCache = FramesCache.getState();
            for await (const sample of iterable) {
                if (!sample) continue;
                const estIndex = Math.round(sample.timestamp * frameRate);
                const w = width || mediaInfo.video?.codedWidth || 0;
                const h = height || mediaInfo.video?.codedHeight || 0;
                const key = buildFrameKey(path, w, h, estIndex);
                frameCache.put(key, sample);
                ctx.inFlight.delete(estIndex);
            }
        } catch (e) {
            // swallow; cleanup happens in finally
        } finally {
            // Always clear remaining in-flight marks to allow future retries
            tasks.forEach(f => ctx.inFlight.delete(f));
            pruneStaleDecoders();
        }
    })();
}

export const fetchVideoSample = async (path: string, frameIndex: number, width?: number, height?: number, options?: {mediaInfo?: MediaInfo}): Promise<VideoSample | null> => {
    const frame = getCachedSample(path, frameIndex, width, height);
    if (frame) return frame as VideoSample;

    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo || !mediaInfo.video) return null;

    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        return null;
    }

    const key = buildFrameKey(path, width, height, frameIndex);

    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    if (!Number.isFinite(frameRate) || frameRate <= 0) {
        return null;
    }
    const timestamp = frameIndex / frameRate;

    let sample: VideoSample | null = null;
    try {
        const decoder = getOrCreateVideoDecoder(path, mediaInfo);
        if (!decoder) return null;
        decoder.lastAccessTs = nowMs();
        try {
            sample = await decoder.sink.getSample(timestamp);
        } catch (e) {
            sample = null;
        }
        // Fallback to nearby timestamps if exact seek fails
        if (!sample) {
            for (const df of [1, -1, 2, -2]) {
                try {
                    sample = await decoder.sink.getSample(timestamp + df / frameRate);
                    if (sample) break;
                } catch {}
            }
        }
    } catch (error) {
        console.warn('[media] Failed to decode video sample', { path, frameIndex, error });
        sample = null;
    }

    // cache the sample
    if (sample) {
        const frameCache = FramesCache.getState();
        frameCache.put(key, sample);
        // Also store under estimated index if different to avoid cache misses
        const estIndex = Math.round(sample.timestamp * frameRate);
        if (estIndex !== frameIndex) {
            const altKey = buildFrameKey(path, width, height, estIndex);
            frameCache.put(altKey, sample);
        }
        // Schedule background prefetch around this frame to keep decoder warm
        schedulePrefetchVideo(path, mediaInfo, frameIndex, width, height);
    }

    return sample;
}


export const fetchVideoSamples = async (path: string, frameIndices: number[], width?: number, height?: number, options?: {mediaInfo?: MediaInfo}): Promise<(VideoSample | null)[]> => {
    // make samples a list of nulls to start with
    const samples = getCachedSamples(path, frameIndices, width, height);
   
    if (samples.every(sample => sample !== null)) {
        return samples as (VideoSample | null)[];
    }
  
    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo || !mediaInfo.video) return samples as (VideoSample | null)[];

    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        return samples as (VideoSample | null)[];
    }

    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    if (!Number.isFinite(frameRate) || frameRate <= 0) {
        return samples as (VideoSample | null)[];
    }
    const decoder = getOrCreateVideoDecoder(path, mediaInfo);
    if (!decoder) return samples as (VideoSample | null)[];
    decoder.lastAccessTs = nowMs();

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
    let sparseSamples: AsyncIterable<VideoSample | null> | any;
    try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        sparseSamples = await decoder.sink.samplesAtTimestamps(sparseTimestamps);
    } catch (error) {
        console.warn('[media] Failed to request video samples at timestamps', { path, error });
        return samples as (VideoSample | null)[];
    }

    const frameCache = FramesCache.getState();

    try {
        for await (const sample of sparseSamples as AsyncIterable<VideoSample | null>) {
            if (!sample) continue;
            const estimatedFrameIndex = Math.round(sample.timestamp * frameRate);
            let index = frameIndexToIndex.get(estimatedFrameIndex);
            if (index === undefined) {
                index = frameIndexToIndex.get(estimatedFrameIndex - 1) ?? frameIndexToIndex.get(estimatedFrameIndex + 1);
            }
            if (index !== undefined) {
                samples[index] = sample;
                const frameKey = buildFrameKey(path, width, height, estimatedFrameIndex);
                frameCache.put(frameKey, sample);
                const requestedFrameIndex = frameIndices[index];
                if (Number.isFinite(requestedFrameIndex) && estimatedFrameIndex !== requestedFrameIndex) {
                    const altKey = buildFrameKey(path, width, height, requestedFrameIndex);
                    frameCache.put(altKey, sample);
                }
            }
        }
    } catch (error) {
        console.warn('[media] Error while iterating decoded video samples', { path, error });
    }

    return samples as (VideoSample | null)[];
   
}

export const prefetchVideoSamples = async (path: string, frameIndices: number[], width?: number, height?: number, options?: {mediaInfo?: MediaInfo}): Promise<void> => {
    await fetchVideoSamples(path, frameIndices, width, height, options);
}