import { MediaInfo } from "../types";
import { VideoDecoderContext, VideoDecoderKey } from "./types";
import { CanvasSink } from "mediabunny";
import { nowMs } from "./utils";
import { MediaCache } from "./cache";
import {videoDecoders, pruneStaleDecoders} from "./utils";



function getOrCreateVideoDecoder(path: string, mediaInfo: MediaInfo, canBeTransparent: boolean): VideoDecoderContext | null {
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
    
        const sink = new CanvasSink(mediaInfo.video, {
            poolSize: 2,
			fit: 'contain', // In case the video changes dimensions over time
			alpha: canBeTransparent,
        });
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
                await sink.getCanvas(ts);
            } catch {}
        })();
        return ctx;
    } catch (e) {
        console.warn('[media] Failed to create VideoSampleSink', { path, error: e });
        return null;
    }
}

export const getVideoIterator = async (path: string, options?: { mediaInfo?: MediaInfo, fps?: number, startIndex: number, endIndex?: number }) => {
    try {
        const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
        if (!mediaInfo || !mediaInfo.video) throw new Error('Media info not found');
        const fps = options?.fps || mediaInfo.stats.video?.averagePacketRate || 0;      

        const startTimestamp = (options?.startIndex || 0) / fps;
        const endTimestamp = options?.endIndex ? (options.endIndex) / fps : undefined;
        const videoCanBeTransparent = await mediaInfo.video.canBeTransparent();
        const decoder = getOrCreateVideoDecoder(path, mediaInfo, videoCanBeTransparent);
        if (!decoder) throw new Error('Decoder not found');
        decoder.lastAccessTs = nowMs();
        return decoder.sink.canvases(startTimestamp, endTimestamp);
    } 
    finally {
        pruneStaleDecoders();
    }
}
