import { MediaInfo } from "../types";
import { CanvasDecoderContext, CanvasDecoderKey } from "./types";
import { CanvasSink, WrappedCanvas } from "mediabunny";
import { nowMs } from "./utils";
import { FramesCache, MediaCache } from "./cache";
import { PREFETCH_BACK, PREFETCH_AHEAD } from "../settings";
import { buildFrameKey, getCachedSample, getCachedSamples } from "./cache";
import {canvasDecoders, pruneStaleDecoders} from "./utils";

function getOrCreateCanvasDecoder(path: string, mediaInfo: MediaInfo, width: number, height: number, poolSize?: number): CanvasDecoderContext | null {
    if (!mediaInfo?.video) return null;
    const targetW = Math.max(1, Math.floor(width || mediaInfo.video.codedWidth || 0));
    const targetH = Math.max(1, Math.floor(height || mediaInfo.video.codedHeight || 0));
    const key: CanvasDecoderKey = `${path}#canvas@${targetW}x${targetH}`;
    const existing = canvasDecoders.get(key);
    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    if (existing) {
        existing.lastAccessTs = nowMs();
        existing.frameRate = frameRate || existing.frameRate;
        return existing;
    }
    try {
        const sink = new CanvasSink(mediaInfo.video, {
            width: targetW,
            height: targetH,
            fit: 'fill',
            poolSize: poolSize || 3,
        });
        const ctx: CanvasDecoderContext = {
            sink,
            inFlight: new Set<number>(),
            lastAccessTs: nowMs(),
            frameRate: frameRate || 0,
            width: targetW,
            height: targetH,
        };
        canvasDecoders.set(key, ctx);
        // Prime decoder lightly (do not await)
        void (async () => {
            try {
                const ts = 0;
                await sink.getCanvas(ts);
            } catch {}
        })();
        return ctx;
    } catch (e) {
        console.warn('[media] Failed to create CanvasSink', { path, error: e });
        return null;
    }
}


function schedulePrefetchCanvas(path: string, mediaInfo: MediaInfo, centerFrame: number, width: number, height: number, poolSize?: number) {
    const ctx = getOrCreateCanvasDecoder(path, mediaInfo, width, height, poolSize);
    if (!ctx) return;
    const frameRate = ctx.frameRate || mediaInfo.stats.video?.averagePacketRate || 0;
    if (!Number.isFinite(frameRate) || frameRate <= 0) return;
    const start = Math.max(0, Math.floor(centerFrame - PREFETCH_BACK));
    const end = Math.max(start, Math.floor(centerFrame + PREFETCH_AHEAD));
    const frameCache = FramesCache.getState();
    const tasks: number[] = [];
    for (let i = start; i <= end; i++) {
        if (i === centerFrame) continue;
        if (ctx.inFlight.has(i)) continue;
        const key = buildFrameKey(path, ctx.width, ctx.height, i, true);
        if (frameCache.has(key)) continue;
        tasks.push(i);
    }
    if (tasks.length === 0) return;
    tasks.forEach(f => ctx.inFlight.add(f));
    void (async () => {
        try {
            const timestamps = tasks.map(f => f / frameRate);
            const iterable = await ctx.sink.canvasesAtTimestamps(timestamps) as AsyncIterable<WrappedCanvas | null>;
            const frameCache = FramesCache.getState();
            for await (const wrapped of iterable) {
                if (!wrapped) continue;
                const estIndex = Math.round(wrapped.timestamp * frameRate);
                const key = buildFrameKey(path, ctx.width, ctx.height, estIndex, true);
                frameCache.put(key, wrapped);
                ctx.inFlight.delete(estIndex);
            }
        } catch (e) {
            tasks.forEach(f => ctx.inFlight.delete(f));
        } finally {
            pruneStaleDecoders();
        }
    })();
}



export const fetchCanvasSample = async (path: string, frameIndex: number, width?: number, height?: number, options?: {mediaInfo?: MediaInfo, poolSize?:number}): Promise<WrappedCanvas | null> => {
    const frame = getCachedSample(path, frameIndex, width, height, true);
    if (frame) return frame as WrappedCanvas;

    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo || !mediaInfo.video) return null;

    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        return null;
    }

    const key = buildFrameKey(path, width, height, frameIndex, true);
    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    if (!Number.isFinite(frameRate) || frameRate <= 0) {
        return null;
    }
    const timestamp = frameIndex / frameRate;

    let sample: WrappedCanvas | null = null;
    try {
        const decoder = getOrCreateCanvasDecoder(path, mediaInfo, width, height, options?.poolSize || 3);
        if (!decoder) return null;
        decoder.lastAccessTs = nowMs();
        sample = await decoder.sink.getCanvas(timestamp);
        if (!sample) {
            for (const df of [1, -1, 2, -2]) {
                try {
                    sample = await decoder.sink.getCanvas(timestamp + df / frameRate);
                    if (sample) break;
                } catch {}
            }
        }
    } catch (error) {
        console.warn('[media] Failed to decode canvas sample', { path, frameIndex, error });
        sample = null;
    }

    // cache the sample
    if (sample) {
        const frameCache = FramesCache.getState();
        frameCache.put(key, sample);
        const estIndex = Math.round(sample.timestamp * frameRate);
        if (estIndex !== frameIndex) {
            const altKey = buildFrameKey(path, width, height, estIndex, true);
            frameCache.put(altKey, sample);
        }
        schedulePrefetchCanvas(path, mediaInfo, frameIndex, width, height, options?.poolSize || 3);
    }

    return sample;
}

export const fetchCanvasSamples = async (path: string, frameIndices: number[], width?: number, height?: number, options?: {mediaInfo?: MediaInfo, poolSize?:number}): Promise<(WrappedCanvas | null)[]> => {
    // make samples a list of nulls to start with
    const samples = getCachedSamples(path, frameIndices, width, height, true);
   
    if (samples.every(sample => sample !== null)) {
        return samples as (WrappedCanvas | null)[];
    }
  
    const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo || !mediaInfo.video) return samples as (WrappedCanvas | null)[];

    width = width || mediaInfo.video?.codedWidth || 0;
    height = height || mediaInfo.video?.codedHeight || 0;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        return samples as (WrappedCanvas | null)[];
    }

    const decoder = getOrCreateCanvasDecoder(path, mediaInfo, width, height, options?.poolSize || 3);
    if (!decoder) return samples as (WrappedCanvas | null)[];
    decoder.lastAccessTs = nowMs();

    const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
    if (!Number.isFinite(frameRate) || frameRate <= 0) {
        return samples as (WrappedCanvas | null)[];
    }
    
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
    let sparseSamples: AsyncIterable<WrappedCanvas | null> | any;
    try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        sparseSamples = await decoder.sink.canvasesAtTimestamps(sparseTimestamps);
    } catch (error) {
        console.warn('[media] Failed to request canvas samples at timestamps', { path, error });
        return samples as (WrappedCanvas | null)[];
    }

    const frameCache = FramesCache.getState();

    try {
        for await (const sample of sparseSamples as AsyncIterable<WrappedCanvas | null>) {
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
    } catch (error) {
        console.warn('[media] Error while iterating decoded canvas samples', { path, error });
    }

    return samples as (WrappedCanvas | null)[];
   
}
