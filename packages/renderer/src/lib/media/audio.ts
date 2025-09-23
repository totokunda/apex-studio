import { MediaInfo } from "../types";
import { AudioDecoderContext, AudioDecoderKey } from "./types";
import {  AudioBufferSink} from "mediabunny";
import { nowMs, audioDecoders, pruneStaleDecoders} from "./utils";
import { MediaCache} from "./cache";

export const BLOCK_SIZE = 2048;

function getOrCreateAudioDecoder(path: string, mediaInfo: MediaInfo): AudioDecoderContext | null {
    if (!mediaInfo?.audio) return null;
    const key: AudioDecoderKey = `${path}#audio`;
    const existing = audioDecoders.get(key);
    const sampleRate = mediaInfo.audio?.sampleRate || 0;
    if (existing) {
        existing.lastAccessTs = nowMs();
        existing.sampleRate = sampleRate || existing.sampleRate;
        return existing;
    }
    try {
        const sink = new AudioBufferSink(mediaInfo.audio);
    
        const ctx: AudioDecoderContext = {
            sink,
            inFlight: new Set<number>(),
            lastAccessTs: nowMs(),
            sampleRate: sampleRate || 0,
        };
        audioDecoders.set(key, ctx);
        // Prime decoder lightly (do not await)
        void (async () => {
            try {
                const ts = 0;
                await sink.getBuffer(ts);
            } catch {}
        })();
        return ctx;
    } catch (e) {
        console.warn('[media] Failed to create AudioSampleSink', { path, error: e });
        return null;
    }
}

export const getAudioIterator = async (path: string, options?: { mediaInfo?: MediaInfo, sampleRate?: number, fps?: number, sampleSize?: number, index?: number }) => {
    try {
        const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
        if (!mediaInfo || !mediaInfo.audio) throw new Error('Media info not found');
        const sampleRate = options?.sampleRate || mediaInfo.audio?.sampleRate || 0;
        const sampleSize = options?.sampleSize || 2048.0;
        const startTimestamp = (sampleSize * (options?.index || 0)) / sampleRate;
        const decoder = getOrCreateAudioDecoder(path, mediaInfo);
        if (!decoder) throw new Error('Decoder not found');
        decoder.lastAccessTs = nowMs();
        return decoder.sink.buffers(startTimestamp);
    } 
    finally {
        pruneStaleDecoders();
    }
}
