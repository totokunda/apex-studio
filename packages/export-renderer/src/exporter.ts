import { blitDrawing, blitImage, blitShape, blitText, blitVideo, cleanupVideoDecoders, type ImageClipProps as BlitImageClipProps, type VideoClipProps as BlitVideoClipProps, type TextClipProps as BlitTextClipProps, type ShapeClipProps as BlitShapeClipProps, type DrawingClipProps as BlitDrawingClipProps } from './blit';
import { getVideoFrameIterator } from '../../../packages/renderer/src/lib/media/video';
import { renderAudioMixWithFfmpeg, deleteFile } from '@app/preload';
import type { WrappedCanvas } from 'mediabunny';
import { acquireHaldClut, releaseHaldClut } from './webgl-filters/hald-clut-singleton';
import { createApplicatorFromClip, FfmpegEncoderOptionsNoFilename, FfmpegFrameEncoder} from './index';

type ClipType = 'image' | 'text' | 'shape' | 'draw' | 'video' | 'filter' | 'audio';

type TransformLike = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  scaleX?: number;
  scaleY?: number;
  rotation?: number;
  cornerRadius?: number;
  opacity?: number;
};

export interface ExportClipBase {
  clipId: string;
  type: ClipType;
  timelineId?: string;
  startFrame?: number;
  endFrame?: number;
  transform?: TransformLike;
  originalTransform?: TransformLike;
  // Applicators bound directly to this clip
  applicators?: ExportApplicatorClip[];
}

export interface ExportImageClip extends ExportClipBase {
  type: 'image';
  src: string;
  brightness?: number;
  contrast?: number;
  hue?: number;
  saturation?: number;
  blur?: number;
  noise?: number;
  sharpness?: number;
  vignette?: number;
  masks?: unknown[];
}

export interface ExportVideoClip extends ExportClipBase {
  type: 'video';
  src: string;
  // Optional external/attached audio source to include during export
  audioSrc?: string;
  speed?: number;
  trimStart?: number;
  brightness?: number;
  contrast?: number;
  hue?: number;
  saturation?: number;
  blur?: number;
  noise?: number;
  sharpness?: number;
  vignette?: number;
  masks?: unknown[];
  preprocessors?: Array<{ src?: string; startFrame?: number; endFrame?: number; status?: 'running' | 'complete' | 'failed' }>;
}

export interface ExportTextClip extends ExportClipBase {
  type: 'text';
  text?: string;
}

export interface ExportShapeClip extends ExportClipBase {
  type: 'shape';
  shapeType?: 'rectangle' | 'ellipse' | 'polygon' | 'line' | 'star';
  fill?: string;
  fillOpacity?: number;
  stroke?: string;
  strokeOpacity?: number;
  strokeWidth?: number;
  sides?: number; // for polygon
  points?: number; // for star
}

export interface ExportDrawClip extends ExportClipBase {
  type: 'draw';
  lines: unknown[];
}

export interface ExportApplicatorClip extends ExportClipBase {
  type: 'filter';
}

export type ExportClip = ExportImageClip | ExportTextClip | ExportShapeClip | ExportDrawClip | ExportVideoClip | ExportAudioClip | ExportApplicatorClip | (ExportClipBase & { type: 'filter' });

export interface FrameEncoder {
  start: (opts: { width: number; height: number; fps: number; audioPath?: string }) => Promise<void>;
  addFrame: (canvas: HTMLCanvasElement) => Promise<void>;
  finalize: () => Promise<Blob | Uint8Array | string | void>;
}

export interface ExportOptions {
  canvas?: HTMLCanvasElement;
  clips: ExportClip[];
  fps: number;
  mode: 'video' | 'image' | 'audio';
  includeAudio?: boolean; // defaults to true
  imageFrame?: number; // for image mode
  range?: { start: number; end: number }; // inclusive start, exclusive end
  encoderOptions?: FfmpegEncoderOptionsNoFilename;
  filename?: string; // for image download
  width?: number; // when canvas not supplied
  height?: number; // when canvas not supplied
}

export interface ExportClipOptions {
  canvas?: HTMLCanvasElement;
  clip: ExportClip;
  fps: number;
  mode: 'video' | 'image' | 'audio';
  includeAudio?: boolean; // defaults to true
  imageFrame?: number; // for image mode
  range?: { start: number; end: number }; // inclusive start, exclusive end (LOCAL to clip)
  encoderOptions?: FfmpegEncoderOptionsNoFilename;
  filename?: string; // output name hint
  width?: number; // when canvas not supplied
  height?: number; // when canvas not supplied
}

export interface ExportAudioClip extends ExportClipBase {
  type: 'audio';
  src: string;
  volume?: number;
  fadeIn?: number;
  fadeOut?: number;
  speed?: number;
  trimStart?: number;
  trimEnd?: number;
}

const resolveVideoSourceForFrame = (c: ExportVideoClip, projectFrame: number): { selectedSrc: string; frameOffset: number } => {
  const preprocessors = Array.isArray(c?.preprocessors) ? c.preprocessors : [];
  if (!preprocessors.length) return { selectedSrc: c.src, frameOffset: 0 };
  const localFrame = Math.max(0, projectFrame - (Number(c?.startFrame) || 0));
  for (const p of preprocessors) {
    if (p?.status !== 'complete' || !p?.src) continue;
    const s = (Number(p.startFrame) || 0);
    const e = (Number(p.endFrame) ?? Number(p.startFrame) ?? 0);
    if (localFrame >= s && localFrame <= e) {
      return { selectedSrc: p.src, frameOffset: 0 };
    }
  }
  return { selectedSrc: c.src, frameOffset: 0 };
};

export async function exportSequence(opts: ExportOptions): Promise<Blob | Uint8Array | string | void> {
  const { clips, fps, mode, imageFrame, range, encoderOptions, filename } = opts;
  const includeAudio = (opts as any)?.includeAudio !== undefined ? !!(opts as any).includeAudio : true;

  // Resolve output canvas (create if not provided)
  let canvas = opts.canvas;
  if (!canvas) {
    // Infer size from clips' transforms or fallback
    const inferredWidth = Math.max(
      opts.width || 0,
      ...clips.map((c) => Number(((c.transform as { width?: number } | undefined)?.width)) || 0),
    );
    const inferredHeight = Math.max(
      opts.height || 0,
      ...clips.map((c) => Number(((c.transform as { height?: number } | undefined)?.height)) || 0),
    );
    const w = Math.max(1, inferredWidth || 1920);
    const h = Math.max(1, inferredHeight || 1080);
    canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // precompute duration if not provided
  let startFrame = 0;
  let endFrame = 0;
  const globalStart = range?.start ?? Math.min(...clips.map(c => c.startFrame ?? 0), 0);
  const globalEnd = range?.end ?? Math.max(...clips.map(c => c.endFrame ?? 0), 0);
  startFrame = Math.max(0, globalStart);
  endFrame = Math.max(startFrame + 1, globalEnd);
  
  


  // Prepare applicators per clip using shared hald clut (only if needed by filters)
  const hald = acquireHaldClut();

  // Temp canvas for per-clip drawing to preserve prior layers (since some blits clear target)
  const temp = document.createElement('canvas');
  temp.width = canvas.width;
  temp.height = canvas.height;
  const tctx = temp.getContext('2d');

  // Maintain per-clip video iterators across frames (for this export run)
  type IterCtx = { key: string; iter: AsyncIterator<WrappedCanvas | null>; currentProjectIndex: number };
  const videoIters: Map<string, IterCtx> = new Map<string, IterCtx>();


  const drawFrame = async (frame: number) => {

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Active clips at this frame
    const active = clips.filter(c => {
      const s = c.startFrame ?? 0 - (range?.start ? c.startFrame ?? 0 : 0);
      const e = c.endFrame ?? 0 - (range?.start ? c.startFrame ?? 0 : 0);
      return frame >= s && frame <= e;
    });


    // Preserve the incoming order of clips; draw in that sequence
    const ordered = active;

    for (const clip of ordered) {
      // Clear temp
      if (tctx) {
        tctx.clearRect(0, 0, temp.width, temp.height);
      }

      // Gather applicators bound to this clip, active at this frame
      const bound = clip.applicators && Array.isArray(clip.applicators) ? clip.applicators as ExportApplicatorClip[] : [];
      const activeApps = bound.filter((a) => {
        const s = a.startFrame ?? Number.NEGATIVE_INFINITY;
        const e = a.endFrame ?? Number.POSITIVE_INFINITY;
        return frame >= s && frame <= e;
      });

      const applicatorsRaw = activeApps
        .map((a) => createApplicatorFromClip(a as any, { haldClutInstance: hald, focusFrameOverride: frame }))
        .filter((x): x is NonNullable<ReturnType<typeof createApplicatorFromClip>> => x !== null);
      const applicators = applicatorsRaw as unknown as Array<{ apply: (c: HTMLCanvasElement) => HTMLCanvasElement; ensureResources?: () => Promise<void> }>;

      if (clip.type === 'image') {
        await blitImage(temp, clip as unknown as BlitImageClipProps, applicators, frame);
      } else if (clip.type === 'video') {
        const c = clip as ExportVideoClip;
        const { selectedSrc, frameOffset } = resolveVideoSourceForFrame(c, frame);
        const speed = (() => { const s = Number(c?.speed ?? 1); return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1; })();
        const baseOffsetFrames = (selectedSrc === c.src) ? Math.max(0, Number(c?.trimStart) || 0) : -Math.max(0, frameOffset || 0);
        const clipStart = Number(c?.startFrame) || 0;
        const clipEnd = Number(c?.endFrame);
        const durationFrames = Number.isFinite(clipEnd) ? Math.max(0, Math.floor(clipEnd - clipStart)) : undefined;
        const providedRangeStart = typeof range?.start === 'number' ? Math.max(0, Math.floor(range.start)) : undefined;
        const providedRangeEnd = typeof range?.end === 'number' ? Math.max(0, Math.floor(range.end)) : undefined;
        let rangeStartLocal = providedRangeStart !== undefined ? providedRangeStart : Math.max(0, baseOffsetFrames);
        if (providedRangeStart !== undefined) {
          rangeStartLocal += baseOffsetFrames;
        }
        let rangeEndLocal = providedRangeEnd !== undefined ? providedRangeEnd : (typeof durationFrames === 'number' ? rangeStartLocal + durationFrames : undefined);
        if (providedRangeEnd !== undefined && rangeEndLocal !== undefined) {
          rangeEndLocal += baseOffsetFrames;
        }
        
        const iterKey = `${selectedSrc}|s:${speed}|pfps:${fps}|rs:${rangeStartLocal}|re:${rangeEndLocal ?? 'inf'}`;

        let ctxIter = videoIters.get(c.clipId) as IterCtx | undefined;
        const reusable = !!(ctxIter && ctxIter.key === iterKey && ctxIter.currentProjectIndex <= (rangeStartLocal + Math.max(0, Math.floor(frame - startFrame))));
        if (!reusable || (ctxIter && ctxIter.currentProjectIndex > (rangeStartLocal + Math.max(0, Math.floor(frame - startFrame))))) {
          const asyncIterable = await getVideoFrameIterator(selectedSrc, { projectFps: Math.max(1, fps), startIndex: rangeStartLocal, endIndex: rangeEndLocal, speed });
          ctxIter = { key: iterKey, iter: asyncIterable[Symbol.asyncIterator](), currentProjectIndex: rangeStartLocal };
          videoIters.set(c.clipId, ctxIter);
        }

        if (mode === 'image') {
          // go to the target frame
          while ((ctxIter as IterCtx).currentProjectIndex < frame) {
            await (ctxIter as IterCtx).iter.next();
            (ctxIter as IterCtx).currentProjectIndex++;
          }
        }

        await blitVideo(temp, c as unknown as BlitVideoClipProps, applicators, (ctxIter as IterCtx).iter, frame);
        (ctxIter as IterCtx).currentProjectIndex++;
      } else if (clip.type === 'text') {
        blitText(temp, clip as unknown as BlitTextClipProps, applicators);
      } else if (clip.type === 'shape') {
        blitShape(temp, clip as unknown as BlitShapeClipProps, applicators);
      } else if (clip.type === 'draw') {
        await blitDrawing(temp, clip as unknown as BlitDrawingClipProps, applicators);
      } else {
        continue;
      }

      // Composite temp onto destination
      ctx.drawImage(temp, 0, 0);
    }
  };

  // Helper: render audio mix for provided audio clips using ffmpeg (pitch-preserving)
  const renderAudioMixToFile = async (
    mixSpecs: Array<{
      src: string;
      startFrame: number;
      endFrame: number;
      trimStart: number;
      volumeDb: number;
      fadeInSec: number;
      fadeOutSec: number;
      speed: number;
    }>,
    exportStartFrame: number,
    exportEndFrame: number,
    exportFps: number,
    outFormat: 'wav' | 'mp3',
    nameHint?: string,
    filename?: string
  ): Promise<string | null> => {
    try {
      if (!mixSpecs || mixSpecs.length === 0) return null;
      const out = await renderAudioMixWithFfmpeg(mixSpecs as any, { fps: exportFps, exportStartFrame, exportEndFrame, outFormat, fileNameHint: nameHint, filename: filename });
      return out as string | null;
    } catch {
      return null;
    }
  };

  let audioPath: string | undefined = undefined;
  try {
    if (mode === 'image') {
      const frame = typeof imageFrame === 'number' ? imageFrame : startFrame;
      await drawFrame(frame);
      // Download image
      const blob: Blob = await new Promise((resolve) => canvas.toBlob(b => resolve(b || new Blob()), 'image/png'));
      return blob;
    }

    // Determine audio usage
    const audioClips = (clips.filter(c => (c as any).type === 'audio') as unknown[]) as ExportAudioClip[];
    const videoAudioClips = includeAudio
      ? (clips.filter(c => (c as any).type === 'video') as unknown[]) as ExportVideoClip[]
      : ([] as ExportVideoClip[]);
    const allAudio = clips.length > 0 && clips.every(c => (c as any).type === 'audio');

    if (allAudio) {
      const specsAll = [
        ...audioClips.map((c) => ({
          src: (c as any).src as string,
          startFrame: Number(c.startFrame) || 0,
          endFrame: typeof c.endFrame === 'number' ? Number(c.endFrame) : endFrame,
          trimStart: Math.max(0, Number((c as any)?.trimStart) || 0),
          volumeDb: Number((c as any)?.volume || 0),
          fadeInSec: Math.max(0, Number((c as any)?.fadeIn) || 0),
          fadeOutSec: Math.max(0, Number((c as any)?.fadeOut) || 0),
          speed: (() => { const s = Number((c as any)?.speed ?? 1); return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1; })(),
        })),
      ];
      const outPath = await renderAudioMixToFile(specsAll, startFrame, endFrame, fps, 'mp3', filename ?? 'output.mp3', filename);
      return outPath || undefined;
    }

    if (includeAudio) {
      const specs = [
        ...audioClips.map((c) => ({
          src: (c as any).src as string,
          startFrame: Number(c.startFrame) || 0,
          endFrame: typeof c.endFrame === 'number' ? Number(c.endFrame) : endFrame,
          trimStart: Math.max(0, Number((c as any)?.trimStart) || 0),
          volumeDb: Number((c as any)?.volume || 0),
          fadeInSec: Math.max(0, Number((c as any)?.fadeIn) || 0),
          fadeOutSec: Math.max(0, Number((c as any)?.fadeOut) || 0),
          speed: (() => { const s = Number((c as any)?.speed ?? 1); return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1; })(),
        })),
        ...videoAudioClips.map((c) => ({
          src: String((c as any).src || ''),
          startFrame: Number(c.startFrame) || 0,
          endFrame: typeof c.endFrame === 'number' ? Number(c.endFrame) : endFrame,
          trimStart: Math.max(0, Number((c as any)?.trimStart) || 0),
          volumeDb: 0,
          fadeInSec: 0,
          fadeOutSec: 0,
          speed: (() => { const s = Number((c as any)?.speed ?? 1); return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1; })(),
        })),
      ];
      if (specs.length > 0) {
        const outPath = await renderAudioMixToFile(specs, startFrame, endFrame, fps, 'wav', (filename ?? 'temp_audio') + '.wav');
        if (outPath) audioPath = outPath;
      }
    }

    const encoder = new FfmpegFrameEncoder({ filename: filename ?? 'output.mp4', ...(encoderOptions ?? {}) });
    if (!encoder) throw new Error('No encoder provided. Please provide a FrameEncoder implemented in preload or main.');
    await encoder.start({ width: canvas.width, height: canvas.height, fps, audioPath });


    for (let f = startFrame; f < endFrame; f++) {
      await drawFrame(f);
      await encoder.addFrame(canvas);
    }

    const result = await encoder.finalize();
    return result;
  } finally {
    try {
      if (audioPath) await deleteFile(audioPath);
    } catch {}
    try { cleanupVideoDecoders(); } catch {}
    releaseHaldClut();
  }
}


export async function exportClip(opts: ExportClipOptions): Promise<Blob | Uint8Array | string | void> {
  const { clip, fps, mode, imageFrame, range, encoderOptions, filename } = opts;
  const includeAudio = (opts as any)?.includeAudio !== undefined ? !!(opts as any).includeAudio : true;

  // Resolve output canvas (create if not provided)
  let canvas = opts.canvas;
  if (!canvas) {
    // Infer size from clip transform or fallback
    const inferredWidth = Math.max(
      opts.width || 0,
      Number(((clip.transform as { width?: number } | undefined)?.width)) || 0,
    );
    const inferredHeight = Math.max(
      opts.height || 0,
      Number(((clip.transform as { height?: number } | undefined)?.height)) || 0,
    );
    const w = Math.max(1, inferredWidth || 1920);
    const h = Math.max(1, inferredHeight || 1080);
    canvas = document.createElement('canvas');
    (canvas as HTMLCanvasElement).width = w;
    (canvas as HTMLCanvasElement).height = h;
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Local duration relative to this clip
  const clipStartGlobal = Number(clip.startFrame ?? 0);
  const clipEndGlobal = Number(clip.endFrame ?? clipStartGlobal + 1);
  const localDuration = Math.max(1, Math.max(0, clipEndGlobal - clipStartGlobal));
  const startFrame = range?.start ?? 0;
  const endFrame = range?.end ?? localDuration;

  // Prepare applicators using shared hald clut
  const hald = acquireHaldClut();

  // Temp canvas for per-clip drawing
  const temp = document.createElement('canvas');
  temp.width = canvas.width;
  temp.height = canvas.height;
  const tctx = temp.getContext('2d');

  // Maintain per-clip video iterators across frames (for this export run)
  type IterCtx = { key: string; iter: AsyncIterator<WrappedCanvas | null>; currentProjectIndex: number };
  const videoIters: Map<string, IterCtx> = new Map<string, IterCtx>();

  const drawFrame = async (frame: number) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Clear temp
    if (tctx) {
      tctx.clearRect(0, 0, temp.width, temp.height);
    }

    // Applicators bound to this clip, active at this frame (LOCAL frame indexing)
    const bound = clip.applicators && Array.isArray(clip.applicators) ? (clip.applicators as ExportApplicatorClip[]) : [];
    const activeApps = bound.filter((a) => {
      const s = a.startFrame ?? Number.NEGATIVE_INFINITY;
      const e = a.endFrame ?? Number.POSITIVE_INFINITY;
      return frame >= s && frame <= e;
    });

    const applicatorsRaw = activeApps
      .map((a) => createApplicatorFromClip(a as any, { haldClutInstance: hald, focusFrameOverride: frame }))
      .filter((x): x is NonNullable<ReturnType<typeof createApplicatorFromClip>> => x !== null);
    const applicators = applicatorsRaw as unknown as Array<{ apply: (c: HTMLCanvasElement) => HTMLCanvasElement; ensureResources?: () => Promise<void> }>;


    if (clip.type === 'image') {
      await blitImage(temp, clip as unknown as BlitImageClipProps, applicators, frame);
    } else if (clip.type === 'video') {
      const { selectedSrc, frameOffset } = resolveVideoSourceForFrame(clip, frame);
      const speed = (() => { const s = Number(clip?.speed ?? 1); return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1; })();
      const baseOffsetFrames = (selectedSrc === clip.src) ? Math.max(0, Number(clip?.trimStart) || 0) : -Math.max(0, frameOffset || 0);
      const clipStart = 0; // local
      const clipEnd = localDuration; // local
      const durationFrames = Math.max(0, Math.floor(clipEnd - clipStart));
      const providedRangeStart = typeof range?.start === 'number' ? Math.max(0, Math.floor(range.start)) : undefined;
      const providedRangeEnd = typeof range?.end === 'number' ? Math.max(0, Math.floor(range.end)) : undefined;
      let rangeStartLocal = providedRangeStart !== undefined ? providedRangeStart : Math.max(0, baseOffsetFrames);
      if (providedRangeStart !== undefined) {
        rangeStartLocal += baseOffsetFrames;
      }
      let rangeEndLocal = providedRangeEnd !== undefined ? providedRangeEnd : (typeof durationFrames === 'number' ? rangeStartLocal + durationFrames : undefined);
      if (providedRangeEnd !== undefined && rangeEndLocal !== undefined) {
        rangeEndLocal += baseOffsetFrames;
      }
      const iterKey = `${selectedSrc}|s:${speed}|pfps:${fps}|rs:${rangeStartLocal}|re:${rangeEndLocal ?? 'inf'}`;

      let ctxIter = videoIters.get(clip.clipId) as IterCtx | undefined;
      const reusable = !!(ctxIter && ctxIter.key === iterKey && ctxIter.currentProjectIndex <= (rangeStartLocal + Math.max(0, Math.floor(frame - startFrame))));
      if (!reusable || (ctxIter && ctxIter.currentProjectIndex > (rangeStartLocal + Math.max(0, Math.floor(frame - startFrame))))) {
        const asyncIterable = await getVideoFrameIterator(selectedSrc, { projectFps: Math.max(1, fps), startIndex: rangeStartLocal, endIndex: rangeEndLocal, speed });
        ctxIter = { key: iterKey, iter: asyncIterable[Symbol.asyncIterator](), currentProjectIndex: rangeStartLocal };
        videoIters.set(clip.clipId, ctxIter);
      }
      if (mode === 'image') {
        // go to the target frame
        while ((ctxIter as IterCtx).currentProjectIndex < frame) {
          await (ctxIter as IterCtx).iter.next();
          (ctxIter as IterCtx).currentProjectIndex++;
        }
      }
      await blitVideo(temp, clip as unknown as BlitVideoClipProps, applicators, (ctxIter as IterCtx).iter, frame);
      (ctxIter as IterCtx).currentProjectIndex++;
    } else if (clip.type === 'text') {
      blitText(temp, clip as unknown as BlitTextClipProps, applicators);
    } else if (clip.type === 'shape') {
      blitShape(temp, clip as unknown as BlitShapeClipProps, applicators);
    } else if (clip.type === 'draw') {
      await blitDrawing(temp, clip as unknown as BlitDrawingClipProps, applicators);
    } else {
      // audio/filter-only clip does not draw to canvas
    }

    // Composite temp onto destination
    ctx.drawImage(temp, 0, 0);
  };

  // Helper: render audio mix for provided audio clip using ffmpeg (pitch-preserving)
  const renderAudioMixToFile = async (audioClip: ExportAudioClip, exportStartFrame: number, exportEndFrame: number, exportFps: number, outFormat: 'wav' | 'mp3', nameHint?: string, filename?: string): Promise<string | null> => {
    try {
      if (!audioClip) return null;
      const specs = {
        src: (audioClip as any).src as string,
        startFrame: Number(audioClip.startFrame) || 0,
        endFrame: typeof audioClip.endFrame === 'number' ? Number(audioClip.endFrame) : exportEndFrame,
        trimStart: Math.max(0, Number((audioClip as any)?.trimStart) || 0),
        volumeDb: Number((audioClip as any)?.volume || 0),
        fadeInSec: Math.max(0, Number((audioClip as any)?.fadeIn) || 0),
        fadeOutSec: Math.max(0, Number((audioClip as any)?.fadeOut) || 0),
        speed: (() => { const s = Number((audioClip as any)?.speed ?? 1); return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1; })(),
      };
      const out = await renderAudioMixWithFfmpeg([specs], { fps: exportFps, exportStartFrame:0, exportEndFrame:exportEndFrame, outFormat, fileNameHint: nameHint, filename: filename });
      return out as string | null;
    } catch {
      return null;
    }
  };

  let audioPath: string | undefined = undefined;
  try {
    if (mode === 'image') {
      const frame = typeof imageFrame === 'number' ? imageFrame : startFrame;
      await drawFrame(frame);
      const blob: Blob = await new Promise((resolve) => canvas.toBlob(b => resolve(b || new Blob()), 'image/png'));
      return blob;
    }

    if (clip.type === 'audio') {
      const outPath = await renderAudioMixToFile(clip as unknown as ExportAudioClip, startFrame, endFrame, fps, 'mp3', filename ?? 'output.mp3', filename);
      return outPath || undefined;
    } 
    
    // Optional: include audio when exporting a single video clip if an audio source is attached
    if (includeAudio && clip.type === 'video') {
      const spec = [{
        src: String((clip as any).src),
        startFrame: Number(clip.startFrame) || 0,
        endFrame: typeof clip.endFrame === 'number' ? Number(clip.endFrame) : endFrame,
        trimStart: Math.max(0, Number((clip as any)?.trimStart) || 0),
        volumeDb: 0,
        fadeInSec: 0,
        fadeOutSec: 0,
        speed: (() => { const s = Number((clip as any)?.speed ?? 1); return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1; })(),
      }];
      try {
        const out = await renderAudioMixWithFfmpeg(spec as any, { fps, exportStartFrame: startFrame, exportEndFrame: endFrame, outFormat: 'wav', fileNameHint: (filename ?? 'temp_audio') + '.wav' });
        if (out) audioPath = String(out);
      } catch {}
    }

    const encoder = new FfmpegFrameEncoder({ filename: filename ?? 'output.mp4', ...(encoderOptions ?? {}) });
    if (!encoder) throw new Error('No encoder provided. Please provide a FrameEncoder implemented in preload or main.');
    await encoder.start({ width: canvas.width, height: canvas.height, fps, ...(audioPath ? { audioPath } : {}) });

    for (let f = startFrame; f < endFrame; f++) {
      await drawFrame(f);
      await encoder.addFrame(canvas);
    }

    const result = await encoder.finalize();
    return result;
  } finally {
    try {
      if (audioPath) await deleteFile(audioPath);
    } catch {}
    try { cleanupVideoDecoders(); } catch {}
    releaseHaldClut();
  }
}

