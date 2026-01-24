import { MediaInfo, MaskClipProps, PreprocessorClipProps } from "../types";
import { FramesCache, WrappedAmplitudes } from "./cache";
import { WrappedCanvas } from "mediabunny";
import { fetchImage } from "./image";
import { fetchCanvasSamples, fetchCanvasSample } from "./canvas";
import { IMAGE_EXTS, AUDIO_EXTS } from "../settings";
import { getLowercaseExtension, readFileBuffer } from "@app/preload";
import { getMediaInfo } from "./utils";
import { ShapeMask } from "@/components/preview/mask/shape";
import { LassoMask } from "@/components/preview/mask/lasso";
import { TouchMask } from "@/components/preview/mask/touch";
import { useClipStore } from "../clip";

export const createThumbnailKey = (
  id: string,
  path: string,
  width: number,
  height: number,
  totalCanvasWidth: number,
  frameIndices: number[],
): string =>
  `${id}@${path}@${width}x${height}#w${totalCanvasWidth}#${frameIndices.join(",")}`;

export const createWaveformKey = (
  path: string,
  width: number,
  height: number,
  color?: string,
  startFrame?: number,
  endFrame?: number,
  volume?: number,
  fadeIn?: number,
  fadeOut?: number,
): string =>
  `waveform:${path}@${width}x${height}#c${color ?? ""}#${startFrame ?? ""}:${endFrame ?? ""}#v${volume ?? 0}#fi${fadeIn ?? 0}#fo${fadeOut ?? 0}`;

export const createAmplitudeKey = (path: string): string => `amplitude:${path}`;

export const generateTimelineSamples = async (
  _id: string,
  path: string,
  frameIndices: number[],
  width: number,
  height: number,
  totalCanvasWidth: number,
  options?: {
    mediaInfo?: MediaInfo;
    startFrame?: number;
    endFrame?: number;
    volume?: number;
    fadeIn?: number;
    fadeOut?: number;
  },
): Promise<WrappedCanvas[] | null> => {
  const ext = getLowercaseExtension(path);
  let fetchedSamples: (WrappedCanvas | null)[];

  if (IMAGE_EXTS.includes(ext)) {
    const image = await fetchImage(path, width, height, {
      mediaInfo: options?.mediaInfo,
    });
    fetchedSamples = [image];
  } else if (AUDIO_EXTS.includes(ext) || path.includes("#audio")) {
    const mediaInfo = options?.mediaInfo || (await getMediaInfo(path));
    const canvas = await generateAudioWaveformCanvas(
      path,
      totalCanvasWidth,
      height,
      {
        mediaInfo,
        color: "#7791C4",
        startFrame: options?.startFrame,
        endFrame: options?.endFrame,
        volume: options?.volume,
        fadeIn: options?.fadeIn,
        fadeOut: options?.fadeOut,
      },
    );

    fetchedSamples = [
      {
        canvas: canvas as HTMLCanvasElement,
        duration: mediaInfo?.duration ?? 1,
        timestamp: frameIndices[0] ?? 0,
      },
    ];
  } else {
    fetchedSamples = await fetchCanvasSamples(
      path,
      frameIndices,
      width,
      height,
      options,
    );
  }
  // Filter out invalid or zero-dimension samples to avoid non-progressing loops
  const samples = fetchedSamples.filter(
    (s) => s && s.canvas && s.canvas.width > 0 && s.canvas.height > 0,
  ) as WrappedCanvas[];

  return samples;
};

function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number {
  const clampedValue = Math.max(inMin, Math.min(value, inMax));
  return (
    ((clampedValue - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin
  );
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
  const AudioContextCtor =
    (window as any).AudioContext || (window as any).webkitAudioContext;
  if (!AudioContextCtor) throw new Error("AudioContext not available");

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
  radius: number,
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
  chunkSize: number,
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
  videoFrameRate: number = 24,
  sampleSize: number | undefined = undefined,
): Float32Array {
  if (!startFrame) startFrame = 0;

  // 1. Calculate how many audio samples correspond to one video frame.
  const samplesPerFrame = sampleSize
    ? sampleSize
    : audioSampleRate / videoFrameRate;

  // 2. Calculate the starting sample index.
  // We use Math.floor() to ensure we get an integer index.
  const startSampleIndex = Math.floor(startFrame * samplesPerFrame);
  let endSampleIndex = endFrame
    ? Math.floor(endFrame * samplesPerFrame)
    : fullAudioAmplitudes.length;
  // ensure endSampleIndex is less than fullAudioAmplitudes.length
  endSampleIndex = Math.min(endSampleIndex, fullAudioAmplitudes.length);

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
  options?: {
    samples?: number;
    color?: string;
    mediaInfo?: MediaInfo;
    startFrame?: number;
    endFrame?: number;
    volume?: number;
    fadeIn?: number;
    fadeOut?: number;
    style?: "modern" | "legacy";
    legacy?: boolean;
  },
): Promise<HTMLCanvasElement | null> => {
  // Allow selecting the legacy renderer while keeping the same API
  if (options?.style === "legacy" || options?.legacy) {
    return generateAudioWaveformCanvasLegacy(path, width, height, options);
  }

  const color = options?.color || "#A477C4";
  const volume = options?.volume ?? 0;
  const fadeIn = options?.fadeIn ?? 0;
  const fadeOut = options?.fadeOut ?? 0;

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
        amplitudes: amplitudes,
      } as WrappedAmplitudes);
    }
  } else {
    // Cache miss - decode audio
    amplitudes = await decodeAudioAmplitudes(path);
    // Cache the amplitude data
    frameCache.put(amplitudeKey, {
      amplitudes: amplitudes,
    } as WrappedAmplitudes);
  }

  let startFrame = options?.startFrame;
  let endFrame = options?.endFrame;

  // If mediaInfo has startFrame/endFrame, it means the media is already trimmed
  // We need to offset our requested frames by the mediaInfo's startFrame
  const mediaStartOffset = options?.mediaInfo?.startFrame ?? 0;

  // Adjust the requested frames to account for media trimming
  if (startFrame !== undefined) {
    startFrame = startFrame + mediaStartOffset;
  } else if (mediaStartOffset > 0) {
    startFrame = mediaStartOffset;
  }

  if (endFrame !== undefined) {
    endFrame = endFrame + mediaStartOffset;
  } else if (options?.mediaInfo?.endFrame !== undefined) {
    endFrame = options?.mediaInfo?.endFrame;
  }

  const audioSampleRate = options?.mediaInfo?.audio?.sampleRate ?? 48000;
  const videoFrameRate =
    options?.mediaInfo?.stats.video?.averagePacketRate ?? 24;

  const audioClip = getAudioForVideoFrames(
    amplitudes,
    startFrame,
    endFrame,
    audioSampleRate,
    videoFrameRate,
    options?.mediaInfo?.audio?.sampleSize,
  );

  // 1. Define your source and target ranges
  const SOURCE_DB_MIN = -60.0; // What we consider "silence" in the original signal
  const SOURCE_DB_MAX = 0.0; // The loudest possible signal

  // **** THIS IS THE ONLY PART THAT CHANGES ****
  const TARGET_VISUAL_MIN = 0.0; // Your desired minimum visual value
  const TARGET_VISUAL_MAX = 100.0; // Your desired maximum visual value
  const barWidth = 2;
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
      TARGET_VISUAL_MAX,
    );

    visualizerValues.push(visualValue);
  }

  // 'visualizerValues' is now an array of numbers, where each number is
  // between 0 and 100, perfect for a progress bar or percentage-based visual.
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  // --- Background: rich gradient with subtle vignette ---
  const backgroundGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  backgroundGradient.addColorStop(0, "#121832");
  backgroundGradient.addColorStop(1, "#0B0F1F");
  ctx.fillStyle = backgroundGradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Subtle vertical vignette
  const vignette = ctx.createLinearGradient(0, 0, canvas.width, 0);
  vignette.addColorStop(0, "rgba(0,0,0,0.35)");
  vignette.addColorStop(0.5, "rgba(0,0,0,0)");
  vignette.addColorStop(1, "rgba(0,0,0,0.35)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // --- Grid and ticks (behind bars) ---
  ctx.save();
  // Horizontal grid lines at 25%, 50%, 75%
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 6]);
  const gridRows = [0.25, 0.5, 0.75];
  gridRows.forEach((fraction) => {
    const y = Math.round(canvas.height * fraction) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Vertical ticks: minor every 10px, major every 50px
  for (let x = 0; x <= canvas.width; x += 10) {
    const isMajor = x % 50 === 0;
    const tickHeight = isMajor ? 8 : 4;
    const alpha = isMajor ? 0.12 : 0.06;
    ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    ctx.beginPath();
    const xAligned = Math.round(x) + 0.5;
    ctx.moveTo(xAligned, 0);
    ctx.lineTo(xAligned, tickHeight);
    ctx.moveTo(xAligned, canvas.height);
    ctx.lineTo(xAligned, canvas.height - tickHeight);
    ctx.stroke();
  }
  ctx.restore();
  const barColor = color;
  const cornerRadius = barWidth / 2;
  const totalBarSpace = barWidth + gap;

  // Calculate volume and fade parameters
  const totalDuration = options?.mediaInfo?.duration || 1;
  const dbToGain = (db: number) => Math.pow(10, db / 20);
  const volumeGain = dbToGain(volume);
  const fadeInWidth = (fadeIn / totalDuration) * canvas.width;
  const fadeOutWidth = (fadeOut / totalDuration) * canvas.width;
  const fadeOutStart = canvas.width - fadeOutWidth;

  // --- Color utilities (local, no effect on data) ---
  const clamp255 = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  const hexToRgb = (
    hex: string,
  ): { r: number; g: number; b: number } | null => {
    const m = hex.trim().replace("#", "");
    if (m.length === 3) {
      const r = parseInt(m[0] + m[0], 16);
      const g = parseInt(m[1] + m[1], 16);
      const b = parseInt(m[2] + m[2], 16);
      return { r, g, b };
    }
    if (m.length === 6) {
      const r = parseInt(m.slice(0, 2), 16);
      const g = parseInt(m.slice(2, 4), 16);
      const b = parseInt(m.slice(4, 6), 16);
      return { r, g, b };
    }
    return null;
  };
  const rgba = (hex: string, alpha: number) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
  };
  const lighten = (hex: string, pct: number) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    const r = clamp255(rgb.r + (255 - rgb.r) * (pct / 100));
    const g = clamp255(rgb.g + (255 - rgb.g) * (pct / 100));
    const b = clamp255(rgb.b + (255 - rgb.b) * (pct / 100));
    return `rgb(${r},${g},${b})`;
  };
  const darken = (hex: string, pct: number) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    const r = clamp255(rgb.r * (1 - pct / 100));
    const g = clamp255(rgb.g * (1 - pct / 100));
    const b = clamp255(rgb.b * (1 - pct / 100));
    return `rgb(${r},${g},${b})`;
  };

  // Offscreen canvas for bars (for fast compositing and single-pass glow)
  const barsCanvas = document.createElement("canvas");
  barsCanvas.width = canvas.width;
  barsCanvas.height = canvas.height;
  const bctx = barsCanvas.getContext("2d");
  if (!bctx) return null;
  const lightBar = lighten(barColor, 28);
  const darkBar = darken(barColor, 18);
  const barFillGradient = bctx.createLinearGradient(0, 0, 0, canvas.height);
  barFillGradient.addColorStop(0, lightBar);
  barFillGradient.addColorStop(0.6, barColor);
  barFillGradient.addColorStop(1, darkBar);
  bctx.fillStyle = barFillGradient;

  visualizerValues.forEach((value, index) => {
    const x: number = index * totalBarSpace;
    let heightMultiplier = 1.0;
    heightMultiplier *= volumeGain;
    if (fadeIn > 0 && x < fadeInWidth) {
      const fadeInProgress = x / fadeInWidth;
      heightMultiplier *= fadeInProgress;
    }
    if (fadeOut > 0 && x >= fadeOutStart) {
      const fadeOutProgress = (canvas.width - x) / fadeOutWidth;
      heightMultiplier *= fadeOutProgress;
    }
    const barHeight: number = Math.min(
      (value / 100) * canvas.height * heightMultiplier,
      canvas.height,
    );
    const y: number = canvas.height - barHeight;
    if (value > 0 && barHeight > 0) {
      drawRoundedTopBar(
        bctx as CanvasRenderingContext2D,
        x,
        y,
        barWidth,
        barHeight,
        cornerRadius,
      );
    }
  });

  // Draw a single blurred glow of the entire bar field, then the crisp bars
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.filter = "blur(4px)";
  ctx.globalAlpha = 0.45;
  ctx.drawImage(barsCanvas, 0, 0);
  ctx.restore();

  ctx.drawImage(barsCanvas, 0, 0);

  const lineChunkSize = width > 240 ? 8 : 4; // <-- CONTROL THE SMOOTHNESS HERE

  // Apply volume and fade effects to visualizer values for line drawing
  const adjustedVisualizerValues = visualizerValues.map((value, index) => {
    const x = index * totalBarSpace;
    let heightMultiplier = 1.0;

    heightMultiplier *= volumeGain;

    if (fadeIn > 0 && x < fadeInWidth) {
      heightMultiplier *= x / fadeInWidth;
    }

    if (fadeOut > 0 && x >= fadeOutStart) {
      heightMultiplier *= (canvas.width - x) / fadeOutWidth;
    }

    return Math.min(value * heightMultiplier, 100);
  });

  const linePoints = simplifyDataForLine(
    barWidth,
    gap,
    canvas,
    adjustedVisualizerValues,
    lineChunkSize,
  );

  // --- SECOND PASS: Draw the simplified, smooth line with glow and gradient ---
  if (linePoints.length > 1) {
    // Glow pass
    ctx.save();
    ctx.beginPath();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = rgba(lighten(barColor, 25), 0.5);
    ctx.shadowBlur = 10;
    ctx.strokeStyle = rgba(lighten(barColor, 15), 0.85);
    ctx.moveTo(linePoints[0].x, linePoints[0].y);
    for (let i = 1; i < linePoints.length - 1; i++) {
      const xc = (linePoints[i].x + linePoints[i + 1].x) / 2;
      const yc = (linePoints[i].y + linePoints[i + 1].y) / 2;
      ctx.quadraticCurveTo(linePoints[i].x, linePoints[i].y, xc, yc);
    }
    const lastGlow = linePoints.length - 1;
    ctx.lineTo(linePoints[lastGlow].x, linePoints[lastGlow].y);
    ctx.stroke();
    ctx.restore();

    // Main line pass
    ctx.beginPath();
    const lineGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    lineGradient.addColorStop(0, "#F2F2F2");
    lineGradient.addColorStop(1, rgba("#E8E8E8", 0.85));
    ctx.strokeStyle = lineGradient;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
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

  // Subtle top highlight overlay
  const topHighlight = ctx.createLinearGradient(0, 0, 0, canvas.height);
  topHighlight.addColorStop(0, "rgba(255,255,255,0.06)");
  topHighlight.addColorStop(0.15, "rgba(255,255,255,0.02)");
  topHighlight.addColorStop(0.5, "rgba(255,255,255,0)");
  ctx.fillStyle = topHighlight;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Bottom edge subtle separator
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, Math.floor(canvas.height - 0.5) + 0.5);
  ctx.lineTo(canvas.width, Math.floor(canvas.height - 0.5) + 0.5);
  ctx.stroke();
  ctx.restore();

  return canvas;
};

/**
 * Legacy renderer that preserves the original simple look and rendering path.
 * Same signature so callers can opt-in via options or import directly.
 */
export const generateAudioWaveformCanvasLegacy = async (
  path: string,
  width: number,
  height: number,
  options?: {
    samples?: number;
    color?: string;
    mediaInfo?: MediaInfo;
    startFrame?: number;
    endFrame?: number;
    volume?: number;
    fadeIn?: number;
    fadeOut?: number;
    style?: "modern" | "legacy";
    legacy?: boolean;
  },
): Promise<HTMLCanvasElement | null> => {
  const color = options?.color || "#A477C4";
  const volume = options?.volume ?? 0;
  const fadeIn = options?.fadeIn ?? 0;
  const fadeOut = options?.fadeOut ?? 0;

  const frameCache = FramesCache.getState();
  const amplitudeKey = createAmplitudeKey(path);
  let amplitudes: Float32Array;
  if (frameCache.has(amplitudeKey)) {
    const cached = frameCache.get(amplitudeKey);
    if (cached && (cached as any).amplitudes) {
      amplitudes = (cached as any).amplitudes as Float32Array;
    } else {
      amplitudes = await decodeAudioAmplitudes(path);
      frameCache.put(amplitudeKey, { amplitudes } as WrappedAmplitudes);
    }
  } else {
    amplitudes = await decodeAudioAmplitudes(path);
    frameCache.put(amplitudeKey, { amplitudes } as WrappedAmplitudes);
  }

  let startFrame = options?.startFrame;
  let endFrame = options?.endFrame;
  const mediaStartOffset = options?.mediaInfo?.startFrame ?? 0;
  if (startFrame !== undefined) {
    startFrame = startFrame + mediaStartOffset;
  } else if (mediaStartOffset > 0) {
    startFrame = mediaStartOffset;
  }
  if (endFrame !== undefined) {
    endFrame = endFrame + mediaStartOffset;
  } else if (options?.mediaInfo?.endFrame !== undefined) {
    endFrame = options?.mediaInfo?.endFrame;
  }

  const audioSampleRate = options?.mediaInfo?.audio?.sampleRate ?? 48000;
  const videoFrameRate =
    options?.mediaInfo?.stats.video?.averagePacketRate ?? 24;
  const audioClip = getAudioForVideoFrames(
    amplitudes,
    startFrame,
    endFrame,
    audioSampleRate,
    videoFrameRate,
    options?.mediaInfo?.audio?.sampleSize,
  );

  const SOURCE_DB_MIN = -60.0;
  const SOURCE_DB_MAX = 0.0;
  const TARGET_VISUAL_MIN = 0.0;
  const TARGET_VISUAL_MAX = 100.0;
  const barWidth = 2;
  const gap = 1;
  const barCount = Math.max(1, Math.floor((width + gap) / (barWidth + gap)));
  const chunkSize = Math.max(1, Math.floor(audioClip.length / barCount));

  const visualizerValues: number[] = [];
  for (let i = 0; i < audioClip.length; i += chunkSize) {
    const chunk = audioClip.slice(i, i + chunkSize);
    const rmsValue = calculateRMS(chunk);
    const dbfsValue = amplitudeToDBFS(rmsValue, SOURCE_DB_MIN);
    const visualValue = mapRange(
      dbfsValue,
      SOURCE_DB_MIN,
      SOURCE_DB_MAX,
      TARGET_VISUAL_MIN,
      TARGET_VISUAL_MAX,
    );
    visualizerValues.push(visualValue);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  // Original flat background
  ctx.fillStyle = "#1A2138";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const cornerRadius = barWidth / 2;
  const totalBarSpace = barWidth + gap;
  ctx.fillStyle = color;

  const totalDuration = options?.mediaInfo?.duration || 1;
  const dbToGain = (db: number) => Math.pow(10, db / 20);
  const volumeGain = dbToGain(volume);
  const fadeInWidth = (fadeIn / totalDuration) * canvas.width;
  const fadeOutWidth = (fadeOut / totalDuration) * canvas.width;
  const fadeOutStart = canvas.width - fadeOutWidth;

  // Simple per-bar draw (no gradients, no shadows)
  visualizerValues.forEach((value, index) => {
    const x: number = index * totalBarSpace;
    let heightMultiplier = 1.0;
    heightMultiplier *= volumeGain;
    if (fadeIn > 0 && x < fadeInWidth) {
      const fadeInProgress = x / fadeInWidth;
      heightMultiplier *= fadeInProgress;
    }
    if (fadeOut > 0 && x >= fadeOutStart) {
      const fadeOutProgress = (canvas.width - x) / fadeOutWidth;
      heightMultiplier *= fadeOutProgress;
    }
    const barHeight: number = Math.min(
      (value / 100) * canvas.height * heightMultiplier,
      canvas.height,
    );
    const y: number = canvas.height - barHeight;
    if (value > 0 && barHeight > 0) {
      drawRoundedTopBar(ctx, x, y, barWidth, barHeight, cornerRadius);
    }
  });

  const lineChunkSize = width > 240 ? 8 : 4;
  const adjustedVisualizerValues = visualizerValues.map((value, index) => {
    const x = index * totalBarSpace;
    let heightMultiplier = 1.0;
    heightMultiplier *= volumeGain;
    if (fadeIn > 0 && x < fadeInWidth) {
      heightMultiplier *= x / fadeInWidth;
    }
    if (fadeOut > 0 && x >= fadeOutStart) {
      heightMultiplier *= (canvas.width - x) / fadeOutWidth;
    }
    return Math.min(value * heightMultiplier, 100);
  });
  const linePoints = simplifyDataForLine(
    barWidth,
    gap,
    canvas,
    adjustedVisualizerValues,
    lineChunkSize,
  );

  if (linePoints.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = "#E8E8E8";
    ctx.lineWidth = 1;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
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
};

/**
 * Generate a poster canvas (first frame by default) for a given media URL using mediabunny.
 * Returns an HTMLCanvasElement sized approximately to the requested width/height.
 */
export const generatePosterCanvas = async (
  path: string,
  width?: number,
  height?: number,
  options?: {
    mediaInfo?: MediaInfo;
    frameIndex?: number;
    masks?: MaskClipProps[];
    preprocessors?: PreprocessorClipProps[];
  },
): Promise<CanvasImageSource | null> => {
  try {
    // Resolve effective path based on preprocessors and frameIndex
    const baseLower = (path || "").split("?")[0].toLowerCase();
    const baseDot = baseLower.lastIndexOf(".");
    const baseExt = baseDot >= 0 ? baseLower.slice(baseDot + 1) : "";
    let effectivePath = path;
    let frameIndex: number | undefined = options?.frameIndex;

    // If preprocessors exist, compute a frame index and pick matching src
    if (options?.preprocessors && options.preprocessors.length > 0) {
      if (frameIndex === undefined) {
        if (IMAGE_EXTS.includes(baseExt)) {
          frameIndex = 0;
        } else {
          const mi = options?.mediaInfo || (await getMediaInfo(path));
          if (mi?.video) {
            const duration = mi.duration ?? 1;
            const numFrames =
              duration * (mi.stats.video?.averagePacketRate ?? 24);
            frameIndex = Math.floor(numFrames * 0.1);
          } else {
            frameIndex = 0;
          }
        }
      }
      const fidx = frameIndex ?? 0;
      const matched = options.preprocessors.find((p) => {
        if (!p?.assetId) return false;
        const s = typeof p.startFrame === "number" ? p.startFrame : undefined;
        const e = typeof p.endFrame === "number" ? p.endFrame : s;
        if (s === undefined && e === undefined) return false;
        if (s !== undefined && e !== undefined) return fidx >= s && fidx <= e;
        if (s !== undefined) return fidx === s;
        return false;
      });
      if (matched?.assetId) {
        const asset = useClipStore.getState().getAssetById(matched.assetId);
        if (asset) {
          effectivePath = asset.path;
        }
      }
    }

    // Helper to apply masks onto a canvas if provided
    const applyMasksIfAny = (
      source: HTMLCanvasElement,
      fidx?: number,
    ): HTMLCanvasElement => {
      const masks = options?.masks || [];
      if (!masks || masks.length === 0) return source;
      const shape = new ShapeMask();
      const lasso = new LassoMask();
      const touch = new TouchMask();
      try {
        const working = source.cloneNode(false) as HTMLCanvasElement;
        working.width = source.width;
        working.height = source.height;
        const wctx = working.getContext("2d");
        if (!wctx) return source;
        wctx.drawImage(source, 0, 0);
        return masks.reduce((acc, mask, index) => {
          const effectiveMask =
            index === 0
              ? mask
              : ({ ...mask, backgroundColorEnabled: false } as MaskClipProps);
          let masked: HTMLCanvasElement = acc;
          if ((mask as any).tool === "shape") {
            masked = shape.apply(
              acc,
              effectiveMask,
              fidx ?? 0,
              undefined,
              effectiveMask.transform,
            );
          } else if ((mask as any).tool === "lasso") {
            masked = lasso.apply(
              acc,
              effectiveMask,
              fidx ?? 0,
              undefined,
              effectiveMask.transform,
            );
          } else if ((mask as any).tool === "touch") {
            masked = touch.apply(
              acc,
              effectiveMask,
              fidx ?? 0,
              undefined,
              effectiveMask.transform,
            );
          }
          if (masked !== acc) {
            const c = acc.getContext("2d");
            if (c) {
              c.clearRect(0, 0, acc.width, acc.height);
              c.drawImage(masked, 0, 0);
            }
          }
          return acc;
        }, working);
      } finally {
        shape.dispose();
        lasso.dispose();
        touch.dispose();
      }
    };

    // Detect by effective path extension
    const lower = (effectivePath || "").split("?")[0].toLowerCase();
    const dot = lower.lastIndexOf(".");
    const ext = dot >= 0 ? lower.slice(dot + 1) : "";
    if (IMAGE_EXTS.includes(ext)) {
      const img = await new Promise<HTMLImageElement>(
        async (resolve, reject) => {
          const el = new Image();
          el.onload = () => resolve(el);
          el.onerror = (err) => reject(err);
          el.crossOrigin = "anonymous";
          const res = await readFileBuffer(effectivePath);
          const blob = new Blob([res as unknown as ArrayBuffer]);
          const url = URL.createObjectURL(blob);
          el.src = url;
        },
      );

      const sourceW = Math.max(1, img.naturalWidth || img.width || 1);
      const sourceH = Math.max(1, img.naturalHeight || img.height || 1);
      const targetW = Math.max(1, Math.floor(width || sourceW));
      const targetH = Math.max(1, Math.floor(height || sourceH));

      const canvas = document.createElement("canvas");
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.imageSmoothingEnabled = true;
      // @ts-ignore
      ctx.imageSmoothingQuality = "high";

      const scale = Math.max(targetW / sourceW, targetH / sourceH);
      const drawW = sourceW * scale;
      const drawH = sourceH * scale;
      const sx = Math.max(0, (drawW - targetW) / 2) / scale;
      const sy = Math.max(0, (drawH - targetH) / 2) / scale;
      const sw = Math.min(sourceW, targetW / scale);
      const sh = Math.min(sourceH, targetH / scale);

      ctx.clearRect(0, 0, targetW, targetH);
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, targetW, targetH);

      // Apply masks if provided
      return applyMasksIfAny(canvas, frameIndex);
    }

    // Video-like: use mediabunny and apply masks
    const mediaInfo =
      options?.mediaInfo && effectivePath === path
        ? options.mediaInfo
        : await getMediaInfo(effectivePath);
    if (!mediaInfo?.video) return null;

    const targetWidth = Math.max(
      1,
      Math.floor(width || mediaInfo.video.displayWidth || 320),
    );
    const targetHeight = Math.max(
      1,
      Math.floor(height || mediaInfo.video.displayHeight || 180),
    );
    if (frameIndex === undefined) {
      const duration = mediaInfo.duration ?? 1;
      const numFrames =
        duration * (mediaInfo.stats.video?.averagePacketRate ?? 24);
      frameIndex = Math.floor(numFrames * 0.1);
    }
    const wrapped = await fetchCanvasSample(
      effectivePath,
      frameIndex,
      targetWidth,
      targetHeight,
      { mediaInfo, prefetch: false },
    );
    const baseCanvas = wrapped?.canvas as HTMLCanvasElement | undefined;
    if (!baseCanvas) return null;
    return applyMasksIfAny(baseCanvas, frameIndex);
  } catch (e) {
    console.error(e);
    return null;
  }
};
