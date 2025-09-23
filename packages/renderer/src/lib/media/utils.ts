import { VideoDecoderKey } from "./types";
import { VideoDecoderContext } from "./types";
import { AudioDecoderKey } from "./types";
import { AudioDecoderContext } from "./types";
import { CanvasDecoderKey } from "./types";
import { CanvasDecoderContext} from "./types";
import { DECODER_STALE_MS } from "../settings";
import { MediaInfo } from "../types";
import { MediaCache } from "./cache";
import { IMAGE_EXTS } from "../settings";
import { readImageMetadataFast } from "./image";
import { Input, ALL_FORMATS,  BlobSource } from "mediabunny";
import { getLowercaseExtension, readFileBuffer } from "@app/preload";



export function nowMs(): number {
    return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
}

export const videoDecoders = new Map<VideoDecoderKey, VideoDecoderContext>();
export const audioDecoders = new Map<AudioDecoderKey, AudioDecoderContext>();
export const canvasDecoders = new Map<CanvasDecoderKey, CanvasDecoderContext>();

export const getMediaInfo = async (path: string): Promise<MediaInfo> => {

    // check if the metadata is cached
    const mediaCache = MediaCache.getState();
    if (mediaCache.isMediaCached(path)) {
        return mediaCache.getMedia(path)!;
    }
    const ext = getLowercaseExtension(path);
    // check if file is an image. then modify 

    if (IMAGE_EXTS.includes(ext)) {

        // get height and width from the file    
        const metadata = await readImageMetadataFast(path);

        const mediaInfo = {
            path,
            video: null,
            audio: null,
            image: metadata,
            stats: {
                video: undefined,
                audio: undefined,
            },
            duration: undefined,
            metadata: undefined,
            mimeType: metadata.mime,
            format: undefined,
        };
        mediaCache.setMedia(path, mediaInfo);
        return mediaInfo;
    }

    // Prefer streaming via UrlSource, but fall back to BlobSource if the server truncates streams or range reads fail
    let input: Input | null = null;
    try {
        const buffer = await readFileBuffer(path);
        const blob = new Blob([buffer as unknown as ArrayBuffer]);
        input = new Input({ formats: ALL_FORMATS, source: new BlobSource(blob) });
    } catch (e) {
        console.error('Error reading file', e);
        input = null;
    }

    // If UrlSource creation failed for some reason, or if later reads fail, we'll fallback below
    async function gatherInfo(inp: Input) {
        const videoTrack = await inp.getPrimaryVideoTrack();
        const audioTrack = await inp.getPrimaryAudioTrack();
        const packetStats = await videoTrack?.computePacketStats();
        const audioPacketStats = await audioTrack?.computePacketStats();
        const duration = await inp.computeDuration();
        const metadata = await inp.getMetadataTags();
        const mimeType = await inp.getMimeType();
        const format = await inp.getFormat();
        return { videoTrack, audioTrack, packetStats, audioPacketStats, duration, metadata, mimeType, format };
    }

    let infoBundle: any | null = null;
    try {
        if (!input) throw new Error('UrlSource init failed');
        infoBundle = await gatherInfo(input);
    } catch (e) {
        // Fallback: fetch the entire resource as a Blob and use BlobSource to avoid streaming issues
        try {
            const resp = await fetch(path, { cache: 'no-store', credentials: 'omit', mode: 'cors' });
            if (!resp.ok) throw new Error(`Failed to fetch media: ${resp.status} ${resp.statusText}`);
            const blob = await resp.blob();
            const blobInput = new Input({ formats: ALL_FORMATS, source: new BlobSource(blob) });
            infoBundle = await gatherInfo(blobInput);
        } catch (fallbackErr) {
            throw fallbackErr;
        }
    }

    let { videoTrack, audioTrack, packetStats, audioPacketStats, duration, metadata, mimeType, format } = infoBundle;

    if (audioTrack && !(await audioTrack.canDecode())) {
        audioTrack = null;
        console.warn('Audio track cannot be decoded', path);
    } else if (videoTrack && !(await videoTrack.canDecode())) {
        videoTrack = null;
        console.warn('Video track cannot be decoded', path);
    }

    const mediaInfo: MediaInfo = {
        path,
        video: videoTrack,
        audio: audioTrack,
        image: null,
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


export function pruneStaleDecoders() {
    const threshold = nowMs() - DECODER_STALE_MS;
    for (const [key, ctx] of videoDecoders) {
        if (ctx.lastAccessTs < threshold) {
            try { (ctx.sink as any)?.close?.(); } catch {}
            videoDecoders.delete(key);
        }
    }
    for (const [key, ctx] of audioDecoders) {
        if (ctx.lastAccessTs < threshold) {
            try { (ctx.sink as any)?.close?.(); } catch {}
            audioDecoders.delete(key);
        }
    }
    for (const [key, ctx] of canvasDecoders) {
        if (ctx.lastAccessTs < threshold) {
            try { (ctx.sink as any)?.close?.(); } catch {}
            canvasDecoders.delete(key);
        }
    }
}