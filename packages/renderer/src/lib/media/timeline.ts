import { MediaInfo } from "../types";
import { MediaCache } from "./cache";
import { WrappedCanvas } from "mediabunny";
import { fetchImage } from "./image";
import { fetchCanvasSamples, fetchCanvasSample } from "./canvas";
import { IMAGE_EXTS, AUDIO_EXTS } from "../settings";
import { getLowercaseExtension, readFileBuffer } from "@app/preload";
import { getMediaInfo } from "./utils";
import { useControlsStore } from "../control";

export const createThumbnailKey = (
    id: string,
    path: string,
    width: number,
    height: number,
    totalCanvasWidth: number,
    frameIndices: number[]
): string => `${id}@${path}@${width}x${height}#w${totalCanvasWidth}#${frameIndices.join(',')}`;

export const generateTimelineThumbnail = async (id: string, path: string, frameIndices: number[], width: number, height: number, totalCanvasWidth: number, options?: {canvas?: HTMLCanvasElement, mediaInfo?: MediaInfo, poolSize?:number, startFrame?: number, endFrame?: number}): Promise<CanvasImageSource | null> => {
    const thumbnailKey = createThumbnailKey(id, path, width, height, totalCanvasWidth, frameIndices);

    const mediaCache = MediaCache.getState();

    // If caller provided a canvas, bypass cache lookup to ensure immediate redraw on same canvas
    if (!options?.canvas && mediaCache.isThumbnailCached(thumbnailKey)) {
        return mediaCache.getThumbnailCache(thumbnailKey)!;
    }

    const ext = getLowercaseExtension(path);
    let fetchedSamples:(WrappedCanvas | null)[];

    if (IMAGE_EXTS.includes(ext)) {
        const image = await fetchImage(path, width, height) as WrappedCanvas;
        fetchedSamples = [image];
    } else if (AUDIO_EXTS.includes(ext)) {
        const mediaInfo = options?.mediaInfo || await getMediaInfo(path);
        const canvas = await generateAudioWaveformCanvas(path, totalCanvasWidth, height, {
            mediaInfo,
            color: '#7791C4',
            startFrame: options?.startFrame,
            endFrame: options?.endFrame,
        });
        
        fetchedSamples = [{
            canvas: canvas as HTMLCanvasElement,
            duration: mediaInfo?.duration ?? 1,
            timestamp: frameIndices[0] ?? 0
        }];
    } else {
        fetchedSamples = await fetchCanvasSamples(path, frameIndices, width, height, options);
    }
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

// Legacy helper retained for reference. Not used by the new full-waveform pipeline.
// const createWaveformKey = (
//     path: string,
//     width: number,
//     height: number,
//     samples?: number,
//     color?: string,
//     startFrame?: number,
//     endFrame?: number
// ): string => `waveform:${path}@${width}x${height}#s${samples ?? 0}#c${color ?? ''}#${startFrame ?? ''}:${endFrame ?? ''}`;

/**
 * Generate a static waveform canvas for an audio URL. Cached by path/size.
 */
export const generateAudioWaveformCanvas = async (
    path: string,
    width: number,
    height: number,
    options?: { samples?: number; color?: string; mediaInfo?: MediaInfo; startFrame?: number; endFrame?: number }
): Promise<HTMLCanvasElement | null> => {
    let ac: AudioContext | null = null;
    try {
        const targetW = Math.max(1, Math.floor(width));
        const targetH = Math.max(1, Math.floor(height));
        const color = options?.color || '#A477C4';
        // We render the full waveform at 1px per frame; 'samples' is unused here.
        const rawStart = options?.startFrame;
        const rawEnd = options?.endFrame;
        const startDefined = Number.isFinite(rawStart as number) && (rawStart as number) > 0;
        const endDefined = Number.isFinite(rawEnd as number) && (rawEnd as number) > 0;
        const mediaCache = MediaCache.getState();
        const fpsFromStore = Math.max(1, Math.round(useControlsStore.getState().fps || 24));

        const fullKey = `waveform_full:${path}#h${targetH}#fps${fpsFromStore}#c${color}`;
        const tryFull = mediaCache.getThumbnailCache(fullKey);
        if (tryFull) {
            // Crop from the full cached waveform without decoding again
            const fullCanvas = tryFull as HTMLCanvasElement;
            const totalFrames = fullCanvas.width; // 1px per frame
            const startFrame = startDefined ? Math.max(0, Math.floor(rawStart as number)) : 0;
            const endFrame = endDefined ? Math.max(1, Math.floor(rawEnd as number)) : totalFrames;
            const segStart = Math.min(Math.max(0, startFrame), totalFrames - 1);
            const segEnd = Math.max(segStart + 1, Math.min(totalFrames, endFrame));
            const sx = segStart;
            const sw = Math.max(1, segEnd - segStart);

            const canvas = document.createElement('canvas');
            canvas.width = targetW;
            canvas.height = targetH;
            const ctx = canvas.getContext('2d');
            if (!ctx) return null;
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(0, 0, targetW, targetH);
            ctx.drawImage(fullCanvas, sx, 0, sw, fullCanvas.height, 0, 0, targetW, targetH);
            return canvas;
        }

        const resp = await readFileBuffer(path);
        const arr = resp.buffer as ArrayBuffer;

        // Decode audio
        const AudioContextCtor = (window as any).AudioContext || (window as any).webkitAudioContext;
        if (!AudioContextCtor) return null;
        ac = new AudioContextCtor();
        const audioBuffer: AudioBuffer = await new Promise((resolve, reject) => {
            // Some browsers require callback-style decode
            try {
                // Safari requires a copy of the ArrayBuffer sometimes
                const copy = arr.slice(0);
                (ac as AudioContext).decodeAudioData(copy, resolve, reject);
            } catch (e) {
                reject(e);
            }
        });

        const channels = audioBuffer.numberOfChannels;
        const totalLength = audioBuffer.length; // PCM samples per channel
        const sr = Math.max(1, audioBuffer.sampleRate || 1);
        const fpsLocal = fpsFromStore;
        const perFrameDecoded = sr / fpsLocal;
        const totalFrames = Math.max(1, Math.floor((totalLength * fpsLocal) / sr));

        // Build full-frame peaks (1 px per frame)
        const sOf = (f: number) => Math.round(f * perFrameDecoded);
        const framePeaks = new Float32Array(totalFrames);
        for (let f = 0; f < totalFrames; f++) {
            const s0 = Math.max(0, Math.min(totalLength - 1, sOf(f)));
            const s1 = Math.max(s0 + 1, Math.min(totalLength, sOf(f + 1)));
            let min = 1.0;
            let max = -1.0;
            const windowLen = s1 - s0;
            const innerStride = Math.max(1, Math.floor(windowLen / 64));
            for (let j = s0; j < s1; j += innerStride) {
                let sum = 0;
                for (let c = 0; c < channels; c++) sum += audioBuffer.getChannelData(c)[j] || 0;
                const v = sum / channels;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            framePeaks[f] = Math.max(Math.abs(min), Math.abs(max));
        }

        // Light symmetric smoothing
        const smoothed = new Float32Array(totalFrames);
        const halfK = Math.max(1, Math.min(3, Math.floor(totalFrames / 200)));
        for (let i = 0; i < totalFrames; i++) {
            let acc = 0;
            let cnt = 0;
            for (let k = -halfK; k <= halfK; k++) {
                const idx = i + k;
                if (idx >= 0 && idx < totalFrames) { acc += framePeaks[idx]; cnt++; }
            }
            smoothed[i] = cnt > 0 ? acc / cnt : framePeaks[i];
        }

        // Render full waveform once (1 px per frame)
        const fullCanvas = document.createElement('canvas');
        fullCanvas.width = totalFrames;
        fullCanvas.height = targetH;
        const fullCtx = fullCanvas.getContext('2d');
        if (!fullCtx) return null;
        fullCtx.fillStyle = '#4564a0';
        fullCtx.fillRect(0, 0, fullCanvas.width, fullCanvas.height);
        const centerY = Math.floor(targetH / 2);
        fullCtx.fillStyle = '#cad4e8';
        for (let x = 0; x < totalFrames; x++) {
            const amp = smoothed[x] || 0;
            const h = Math.max(1, Math.floor(amp * (targetH - 2)));
            const y = centerY - Math.floor(h / 2);
            fullCtx.fillRect(x, y, 1, h);
        }

        // Cache full waveform only
        mediaCache.setThumbnailCache(fullKey, fullCanvas);

        // Crop requested segment from the full waveform
        const startFrame = startDefined ? Math.max(0, Math.floor(rawStart as number)) : 0;
        const endFrame = endDefined ? Math.max(1, Math.floor(rawEnd as number)) : totalFrames;
        const segStart = Math.min(Math.max(0, startFrame), totalFrames - 1);
        const segEnd = Math.max(segStart + 1, Math.min(totalFrames, endFrame));
        const sx = segStart;
        const sw = Math.max(1, segEnd - segStart);

        const canvas = document.createElement('canvas');
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, targetW, targetH);
        ctx.drawImage(fullCanvas, sx, 0, sw, fullCanvas.height, 0, 0, targetW, targetH);
        return canvas;
    } catch (e) {
        console.warn('[media] Failed to generate audio waveform canvas', { path, error: e });
        return null;
    } finally {
        if (ac) {
            try { await (ac as AudioContext).close(); } catch {}
        }
    }
}

/**
 * Generate a poster canvas (first frame by default) for a given media URL using mediabunny.
 * Returns an HTMLCanvasElement sized approximately to the requested width/height.
 */
export const generatePosterCanvas = async (
    path: string,
    width?: number,
    height?: number,
    options?: { mediaInfo?: MediaInfo; frameIndex?: number }
): Promise<CanvasImageSource | null> => {
    try {
        // Detect images by extension and render directly without mediabunny
        const lower = (path || "").split('?')[0].toLowerCase();
        const dot = lower.lastIndexOf('.');
        const ext = dot >= 0 ? lower.slice(dot + 1) : "";
        if (IMAGE_EXTS.includes(ext)) {
            const img = await new Promise<HTMLImageElement>(async (resolve, reject) => {
                const el = new Image();
                el.onload = () => resolve(el);
                el.onerror = (err) => reject(err);
                el.crossOrigin = 'anonymous';
                // convert path to inline data
                const res = await readFileBuffer(path);
                const blob = new Blob([res as unknown as ArrayBuffer]);
                const url = URL.createObjectURL(blob);
                el.src = url;
            });

            const sourceW = Math.max(1, img.naturalWidth || img.width || 1);
            const sourceH = Math.max(1, img.naturalHeight || img.height || 1);
            const targetW = Math.max(1, Math.floor(width || sourceW));
            const targetH = Math.max(1, Math.floor(height || sourceH));

            const canvas = document.createElement('canvas');
            canvas.width = targetW;
            canvas.height = targetH;
            const ctx = canvas.getContext('2d');
            if (!ctx) return null;
            ctx.imageSmoothingEnabled = true;
            // @ts-ignore
            ctx.imageSmoothingQuality = 'high';

            // Draw with object-fit: cover to fill target while preserving aspect ratio
            const scale = Math.max(targetW / sourceW, targetH / sourceH);
            const drawW = sourceW * scale;
            const drawH = sourceH * scale;
            const sx = Math.max(0, (drawW - targetW) / 2) / scale;
            const sy = Math.max(0, (drawH - targetH) / 2) / scale;
            const sw = Math.min(sourceW, targetW / scale);
            const sh = Math.min(sourceH, targetH / scale);

            ctx.clearRect(0, 0, targetW, targetH);
            ctx.drawImage(img, sx, sy, sw, sh, 0, 0, targetW, targetH);
            return canvas;
        }

        // Fallback: treat as video-like source and use mediabunny
        const mediaInfo = options?.mediaInfo || await getMediaInfo(path);
        if (!mediaInfo?.video) return null;

        const targetWidth = Math.max(1, Math.floor(width || mediaInfo.video.codedWidth || 320));
        const targetHeight = Math.max(1, Math.floor(height || mediaInfo.video.codedHeight || 180));
        const frameIndex = options?.frameIndex ?? 0;
        const wrapped = await fetchCanvasSample(path, frameIndex, targetWidth, targetHeight, { mediaInfo });
        return wrapped?.canvas ?? null;
    } catch (e) {
        console.error(e);
        return null;
    }
}