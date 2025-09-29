import { MediaInfo } from "../types";
import { FramesCache, WrappedAmplitudes } from "./cache";
import { WrappedCanvas } from "mediabunny";
import { fetchImage } from "./image";
import { fetchCanvasSamples, fetchCanvasSample } from "./canvas";
import { IMAGE_EXTS, AUDIO_EXTS } from "../settings";
import { getLowercaseExtension, readFileBuffer } from "@app/preload";
import { getMediaInfo } from "./utils";

export const createThumbnailKey = (
    id: string,
    path: string,
    width: number,
    height: number,
    totalCanvasWidth: number,
    frameIndices: number[]
): string => `${id}@${path}@${width}x${height}#w${totalCanvasWidth}#${frameIndices.join(',')}`;

export const createWaveformKey = (
    path: string,
    width: number,
    height: number,
    color?: string,
    startFrame?: number,
    endFrame?: number
): string => `waveform:${path}@${width}x${height}#c${color ?? ''}#${startFrame ?? ''}:${endFrame ?? ''}`;

export const createAmplitudeKey = (path: string): string => `amplitude:${path}`;

export const generateTimelineSamples = async (_id: string, path: string, frameIndices: number[], width: number, height: number, totalCanvasWidth: number, options?: {mediaInfo?: MediaInfo, startFrame?: number, endFrame?: number}): Promise<WrappedCanvas[] | null> => {

    const ext = getLowercaseExtension(path);
    let fetchedSamples:(WrappedCanvas | null)[];

    if (IMAGE_EXTS.includes(ext)) {
        const image = await fetchImage(path, width, height, { mediaInfo: options?.mediaInfo });
        fetchedSamples = [image];
    } else if (AUDIO_EXTS.includes(ext)) {
        const mediaInfo = options?.mediaInfo || await getMediaInfo(path);
        const canvas = await generateAudioWaveformCanvas(path, totalCanvasWidth, height, {
            mediaInfo,
            color: '#7791C4',
            startFrame: options?.startFrame,
            endFrame: options?.endFrame
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

    return samples
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

function mapRange(value: number, inMin: number, inMax: number, outMin: number, outMax: number): number {
    const clampedValue = Math.max(inMin, Math.min(value, inMax));
    return ((clampedValue - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

function amplitudeToDBFS(amplitude: number, minimumDb = -100): number {
    const absAmp = Math.abs(amplitude);
    if (absAmp === 0) return minimumDb;
    const db = 20 * Math.log10(absAmp);
    return Math.max(db, minimumDb);
}

function calculateRMS(samples: Float32Array): number {
    if (samples.length === 0) return 0;
    let sumOfSquares = 0;
    for (let i = 0; i < samples.length; i++) {
        sumOfSquares += samples[i] * samples[i];
    }
    const meanSquare = sumOfSquares / samples.length;
    return Math.sqrt(meanSquare);
}

async function decodeAudioAmplitudes(path: string): Promise<Float32Array> {
    const resp = await readFileBuffer(path);
    const arr = resp.buffer as ArrayBuffer;

    // Decode audio
    const AudioContextCtor = (window as any).AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextCtor) throw new Error('AudioContext not available');
    
    const ac = new AudioContextCtor();
    const audioBuffer: AudioBuffer = await new Promise((resolve, reject) => {
        // Some browsers require callback-style decode
        try {
            // Safari requires a copy of the ArrayBuffer sometimes
            const copy = arr.slice(0);
            ac.decodeAudioData(copy, resolve, reject);
        } catch (e) {
            reject(e);
        }
    });

    // get audio buffer as a float32 array (mean of all channels)
    const channelCount = audioBuffer.numberOfChannels;
    const sampleCount = audioBuffer.length;
    const amplitudes = new Float32Array(sampleCount);
    
    for (let i = 0; i < sampleCount; i++) {
        let sum = 0;
        for (let channel = 0; channel < channelCount; channel++) {
            sum += audioBuffer.getChannelData(channel)[i];
        }
        amplitudes[i] = sum / channelCount;
    }
    
    return amplitudes;
}


function drawRoundedTopBar(
    context: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number
): void {
    if (height < radius || width < radius * 2) {
        // If the bar is too small for the radius, draw a simple rectangle.
        context.fillRect(x, y, width, height);
        return;
    }
    
    context.beginPath();
    context.moveTo(x, y + height); // Start bottom-left
    context.lineTo(x, y + radius); // Left side
    context.arcTo(x, y, x + radius, y, radius); // Top-left corner
    context.lineTo(x + width - radius, y); // Top side
    context.arcTo(x + width, y, x + width, y + radius, radius); // Top-right corner
    context.lineTo(x + width, y + height); // Right side
    context.closePath(); // Bottom side
    context.fill();
}

/**
     * Reduces a large dataset to a smaller one by finding the peak value in chunks.
     * @param values The full array of visualizer values (0-100).
     * @param chunkSize The number of data points to group into one chunk.
     * @returns A smaller array of {x, y} coordinates for drawing the line.
     */
function simplifyDataForLine(
    barWidth: number,
    gap: number,
    canvas: HTMLCanvasElement,
    values: number[],
    chunkSize: number
): { x: number; y: number }[] {
    const simplifiedPoints: { x: number; y: number }[] = [];
        const totalBarSpace = barWidth + gap; // barWidth + gap

        for (let i = 0; i < values.length; i += chunkSize) {
            const chunk = values.slice(i, i + chunkSize);
            if (chunk.length === 0) continue;

            // --- THIS IS THE KEY CHANGE ---
            // Instead of finding the peak, we calculate the average.
            const sum = chunk.reduce((acc, val) => acc + val, 0);
            const averageValue = sum / chunk.length;
            // -----------------------------

            const barHeight = (averageValue / 100) * canvas.height;
            const chunkCenterIndex = i + chunk.length / 2;
            const x = chunkCenterIndex * totalBarSpace;
            
            // The y position is now calculated directly from the average, with no gap.
            const y = canvas.height - barHeight;

            simplifiedPoints.push({ x, y });
        }
        return simplifiedPoints;
}

/**
 * Extracts a segment from a Float32Array of audio samples that corresponds to a
 * specific range of video frames.
 *
 * @param {Float32Array} fullAudioAmplitudes The complete array of audio samples.
 * @param {number} startFrame The starting video frame number (0-indexed).
 * @param {number} endFrame The ending video frame number (exclusive).
 * @param {number} audioSampleRate The sample rate of the audio (e.g., 48000).
 * @param {number} videoFrameRate The frame rate of the video (e.g., 24).
 * @returns {Float32Array} A new Float32Array containing the audio for the specified frame range.
 */
function getAudioForVideoFrames(
    fullAudioAmplitudes: Float32Array,
    startFrame?: number,
    endFrame?: number,
    audioSampleRate: number = 48000,
    videoFrameRate: number = 24
): Float32Array {
    if (!startFrame) startFrame = 0;
    
    // 1. Calculate how many audio samples correspond to one video frame.
    const samplesPerFrame = audioSampleRate / videoFrameRate;
    // 2. Calculate the starting sample index.
    // We use Math.floor() to ensure we get an integer index.
    const startSampleIndex = Math.floor(startFrame * samplesPerFrame);
    const endSampleIndex = endFrame ? Math.floor(endFrame * samplesPerFrame) : fullAudioAmplitudes.length;

    // 4. Slice the original audio array to get the desired segment.
    // The .slice() method is perfect for this, as it doesn't modify the original array.
    // It extracts from the start index up to (but not including) the end index.
    const audioClip = fullAudioAmplitudes.slice(startSampleIndex, endSampleIndex);

    return audioClip;
}

/**
 * Generate a static waveform canvas for an audio URL. Cached by path/size.
 */
export const generateAudioWaveformCanvas = async (
    path: string,
    width: number,
    height: number,
    options?: { samples?: number; color?: string; mediaInfo?: MediaInfo; startFrame?: number; endFrame?: number }
): Promise<HTMLCanvasElement | null> => {
    
    const color = options?.color || '#A477C4';
    // Check canvas cache first
    const frameCache = FramesCache.getState();

    // Check amplitude cache second
    const amplitudeKey = createAmplitudeKey(path);
    let amplitudes: Float32Array;
    
    if (frameCache.has(amplitudeKey)) {
        const cached = frameCache.get(amplitudeKey);
        if (cached && (cached as any).amplitudes) {
            amplitudes = (cached as any).amplitudes as Float32Array;
        } else {
            // Cache miss - decode audio
            amplitudes = await decodeAudioAmplitudes(path);
            // Cache the amplitude data
            frameCache.put(amplitudeKey, {
                amplitudes: amplitudes
            } as WrappedAmplitudes);
        }
    } else {
        // Cache miss - decode audio
        amplitudes = await decodeAudioAmplitudes(path);
        // Cache the amplitude data
        frameCache.put(amplitudeKey, {
            amplitudes: amplitudes
        } as WrappedAmplitudes);
    }
    

    const audioClip = getAudioForVideoFrames(amplitudes, options?.startFrame, options?.endFrame);


    // 1. Define your source and target ranges
    const SOURCE_DB_MIN = -60.0; // What we consider "silence" in the original signal
    const SOURCE_DB_MAX = 0.0;    // The loudest possible signal

    // **** THIS IS THE ONLY PART THAT CHANGES ****
    const TARGET_VISUAL_MIN = 0.0;   // Your desired minimum visual value
    const TARGET_VISUAL_MAX = 100.0; // Your desired maximum visual value
    const barWidth = width > 240? 2 : 1;
    const gap = 1;
    // determine the number of bars we want to have
    const barCount = Math.max(1, Math.floor((width + gap) / (barWidth + gap)));
    const chunkSize = Math.max(1, Math.floor(audioClip.length / barCount));

    const visualizerValues: number[] = [];

    for (let i = 0; i < audioClip.length; i += chunkSize) {
        const chunk = audioClip.slice(i, i + chunkSize);

        // Calculate the loudness of this chunk in dBFS
        const rmsValue = calculateRMS(chunk);
        const dbfsValue = amplitudeToDBFS(rmsValue, SOURCE_DB_MIN);

        // Map the dBFS value to your target visual range (0 to 100)
        const visualValue = mapRange(
            dbfsValue,
            SOURCE_DB_MIN,
            SOURCE_DB_MAX,
            TARGET_VISUAL_MIN,
            TARGET_VISUAL_MAX
        );

        visualizerValues.push(visualValue);
    }

    // 'visualizerValues' is now an array of numbers, where each number is 
    // between 0 and 100, perfect for a progress bar or percentage-based visual.
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    const backgroundColor = '#1A2138';
    ctx.fillStyle = backgroundColor; // The same dark navy blue
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const barColor = color;
    const cornerRadius = barWidth / 2;
    const totalBarSpace = barWidth + gap;
    ctx.fillStyle = barColor;

    visualizerValues.forEach((value, index) => {
        const barHeight: number = (value / 100) * canvas.height;
        const x: number = index * totalBarSpace;
        const y: number = canvas.height - barHeight;
        if (value > 0) {
            drawRoundedTopBar(ctx, x, y, barWidth, barHeight, cornerRadius);
        }
    });

    const lineChunkSize = width > 240? 8 : 4; // <-- CONTROL THE SMOOTHNESS HERE
    const linePoints = simplifyDataForLine(barWidth, gap, canvas, visualizerValues, lineChunkSize);

    // --- SECOND PASS: Draw the simplified, smooth line ---
    if (linePoints.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = '#E8E8E8';
        ctx.lineWidth = 1;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        
        // (The smooth curve drawing logic is the same as before)
        ctx.moveTo(linePoints[0].x, linePoints[0].y);
        for (let i = 1; i < linePoints.length - 1; i++) {
            const xc = (linePoints[i].x + linePoints[i + 1].x) / 2;
            const yc = (linePoints[i].y + linePoints[i + 1].y) / 2;
            ctx.quadraticCurveTo(linePoints[i].x, linePoints[i].y, xc, yc);
        }
        const last = linePoints.length - 1;
        ctx.lineTo(linePoints[last].x, linePoints[last].y);
        
        ctx.stroke();
    }

    return canvas;
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
        const wrapped = await fetchCanvasSample(path, frameIndex, targetWidth, targetHeight, { mediaInfo, prefetch: false });
        return wrapped?.canvas ?? null;
    } catch (e) {
        console.error(e);
        return null;
    }
}