import {
  cleanupVideoDecoders,
  type ImageClipProps as BlitImageClipProps,
  type VideoClipProps as BlitVideoClipProps,
  type TextClipProps as BlitTextClipProps,
  type ShapeClipProps as BlitShapeClipProps,
  type DrawingClipProps as BlitDrawingClipProps,
} from "./blit";
import { KonvaExportRenderer } from "./export";
import { Rect } from "konva/lib/shapes/Rect";
import { getVideoFrameIterator } from "../../../packages/renderer/src/lib/media/video";
import { renderAudioMixWithFfmpeg, deleteFile } from "@app/preload";
import type { WrappedCanvas } from "mediabunny";
import {
  acquireHaldClut,
  releaseHaldClut,
} from "./webgl-filters/hald-clut-singleton";
import {
  createApplicatorFromClip,
  FfmpegEncoderOptionsNoFilename,
  FfmpegFrameEncoder,
} from "./index";

type ClipType =
  | "image"
  | "text"
  | "shape"
  | "draw"
  | "video"
  | "filter"
  | "audio";

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
  crop?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

type NormalizedTransformLike = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

export interface ExportClipBase {
  clipId: string;
  type: ClipType;
  timelineId?: string;
  startFrame?: number;
  endFrame?: number;
  transform?: TransformLike;
  originalTransform?: TransformLike;
  /**
   * Optional normalized transform in [0, 1] space describing how this clip
   * should occupy the output canvas. When present, the renderer will map
   * these values to pixel coordinates using the export canvas size while
   * preserving `transform` for backwards compatibility.
   */
  normalizedTransform?: NormalizedTransformLike;
  // Applicators bound directly to this clip
  applicators?: ExportApplicatorClip[];
}

export interface ExportImageClip extends ExportClipBase {
  type: "image";
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
  type: "video";
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
  preprocessors?: Array<{
    src?: string;
    startFrame?: number;
    endFrame?: number;
    status?: "running" | "complete" | "failed";
  }>;
}

export interface ExportTextClip extends ExportClipBase {
  type: "text";
  text?: string;
}

export interface ExportShapeClip extends ExportClipBase {
  type: "shape";
  shapeType?: "rectangle" | "ellipse" | "polygon" | "line" | "star";
  fill?: string;
  fillOpacity?: number;
  stroke?: string;
  strokeOpacity?: number;
  strokeWidth?: number;
  sides?: number; // for polygon
  points?: number; // for star
}

export interface ExportDrawClip extends ExportClipBase {
  type: "draw";
  lines: unknown[];
}

export interface ExportApplicatorClip extends ExportClipBase {
  type: "filter";
}

export type ExportClip =
  | ExportImageClip
  | ExportTextClip
  | ExportShapeClip
  | ExportDrawClip
  | ExportVideoClip
  | ExportAudioClip
  | ExportApplicatorClip
  | (ExportClipBase & { type: "filter" });

export interface ExportCancelToken {
  cancelled: boolean;
}

export class ExportCancelledError extends Error {
  constructor(message = "Export cancelled") {
    super(message);
    this.name = "ExportCancelledError";
  }
}

export interface CancellableExportResult<
  T = Blob | Uint8Array | string | void,
> {
  promise: Promise<T>;
  cancel: () => void;
}

export interface FrameEncoder {
  start: (opts: {
    width: number;
    height: number;
    fps: number;
    audioPath?: string;
  }) => Promise<void>;
  addFrame: (buffer: Uint8Array) => Promise<void>;
  finalize: () => Promise<Blob | Uint8Array | string | void>;
}

export interface ExportOptions {
  canvas?: HTMLCanvasElement;
  clips: ExportClip[];
  fps: number;
  mode: "video" | "image" | "audio";
  // Optional progress callback for long-running exports (sequence rendering).
  // currentFrame is 1-based, totalFrames is the total number of frames that
  // will be rendered for this export, and ratio is currentFrame / totalFrames
  // clamped to [0, 1].
  onProgress?: (info: {
    currentFrame: number;
    totalFrames: number;
    ratio: number;
  }) => void;
  includeAudio?: boolean; // defaults to true
  imageFrame?: number; // for image mode
  range?: { start: number; end: number }; // inclusive start, exclusive end
  encoderOptions?: FfmpegEncoderOptionsNoFilename;
  filename?: string; // for image download
  width?: number; // when canvas not supplied
  height?: number; // when canvas not supplied
  backgroundColor?: string; // when canvas not supplied
  audioOptions?: {
    format?: "wav" | "mp3";
  };
  /**
   * Optional cooperative cancellation token. When `cancelled` is set to true,
   * the export loop will throw an `ExportCancelledError` at the next safe
   * checkpoint, allowing callers to abort long-running exports.
   */
  cancelToken?: ExportCancelToken;
  /**
   * Optional completion callback invoked after a successful export (image,
   * video, or audio) and before the result is returned. This is NOT called
   * when the export is cancelled or throws.
   */
  onDone?: () => void;
}

export interface ExportClipOptions {
  canvas?: HTMLCanvasElement;
  clip: ExportClip;
  fps: number;
  mode: "video" | "image" | "audio";
  includeAudio?: boolean; // defaults to true
  imageFrame?: number; // for image mode
  range?: { start: number; end: number }; // inclusive start, exclusive end (LOCAL to clip)
  encoderOptions?: FfmpegEncoderOptionsNoFilename;
  filename?: string; // output name hint
  width?: number; // when canvas not supplied
  height?: number; // when canvas not supplied
  backgroundColor?: string; // when canvas not supplied
  audioOptions?: {
    format?: "wav" | "mp3";
  };
  /**
   * Optional cooperative cancellation token for single-clip exports.
   */
  cancelToken?: ExportCancelToken;
  /**
   * Optional completion callback invoked after a successful export.
   */
  onDone?: () => void;
  /**
   * Optional progress callback for long-running single-clip exports.
   * Mirrors `ExportOptions.onProgress` but uses LOCAL frames:
   * - currentFrame is 1-based, totalFrames is end-start for this clip.
   */
  onProgress?: (info: {
    currentFrame: number;
    totalFrames: number;
    ratio: number;
  }) => void;
}

export interface ExportAudioClip extends ExportClipBase {
  type: "audio";
  src: string;
  volume?: number;
  fadeIn?: number;
  fadeOut?: number;
  speed?: number;
  trimStart?: number;
  trimEnd?: number;
}

/**
 * Extracts any `startFrame` / `endFrame` window encoded on a media URL.
 * These are used by the editor when splitting clips (see `clip.ts:1513-1535`)
 * to encode the effective media in/out points directly on `src`. During
 * export we treat `startFrame` as an additional trim offset applied on top
 * of the clip's own `trimStart`, and `endFrame` as an upper bound when
 * computing decode ranges.
 */
const getSrcFrameWindow = (
  src?: string | null,
): { srcStartFrame: number; srcEndFrame?: number } => {
  if (!src) return { srcStartFrame: 0, srcEndFrame: undefined };
  try {
    const url = new URL(src);
    const rawStart = url.searchParams.get("startFrame");
    const rawEnd = url.searchParams.get("endFrame");
    const hasStart = rawStart !== null && rawStart !== "";
    const hasEnd = rawEnd !== null && rawEnd !== "";

    const start = hasStart ? Number(rawStart) : 0;
    const end = hasEnd ? Number(rawEnd) : undefined;

    const srcStartFrame = Number.isFinite(start)
      ? Math.max(0, Math.floor(start))
      : 0;
    const srcEndFrame = Number.isFinite(end as number)
      ? Math.max(srcStartFrame, Math.floor(end as number))
      : undefined;

    return { srcStartFrame, srcEndFrame };
  } catch {
    // If URL parsing fails (e.g., non-standard string), fall back to no window.
    return { srcStartFrame: 0, srcEndFrame: undefined };
  }
};

const resolveVideoSourceForFrame = (
  c: ExportVideoClip,
  projectFrame: number,
): { selectedSrc: string; frameOffset: number } => {
  const preprocessors = Array.isArray(c?.preprocessors) ? c.preprocessors : [];
  if (!preprocessors.length) return { selectedSrc: c.src, frameOffset: 0 };
  const localFrame = Math.max(0, projectFrame - (Number(c?.startFrame) || 0));
  for (const p of preprocessors) {
    if (p?.status !== "complete" || !p?.src) continue;
    const s = Number(p.startFrame) || 0;
    const e = Number(p.endFrame) ?? Number(p.startFrame) ?? 0;
    if (localFrame >= s && localFrame <= e) {
      return { selectedSrc: p.src, frameOffset: 0 };
    }
  }
  return { selectedSrc: c.src, frameOffset: 0 };
};

export function exportSequenceCancellable(
  opts: ExportOptions,
): CancellableExportResult {
  const token: ExportCancelToken = { cancelled: false };
  const promise = exportSequence({ ...opts, cancelToken: token });
  const cancel = () => {
    token.cancelled = true;
  };
  return { promise, cancel };
}

export async function exportSequence(
  opts: ExportOptions,
): Promise<Blob | Uint8Array | string | void> {
  const {
    clips,
    fps,
    mode,
    imageFrame,
    range,
    encoderOptions,
    filename,
    audioOptions = { format: "mp3" },
    onProgress,
  } = opts;
  const includeAudio =
    (opts as any)?.includeAudio !== undefined
      ? !!(opts as any).includeAudio
      : true;
  const cancelToken = (opts as any)?.cancelToken as
    | ExportCancelToken
    | undefined;

  const checkCancelled = () => {
    if (cancelToken?.cancelled) {
      throw new ExportCancelledError();
    }
  };

  // Resolve output dimensions
  const inferredWidth = opts.width
    ? Math.max(opts.width, 0)
    : Math.max(
        ...clips.map(
          (c) =>
            Number((c.transform as { width?: number } | undefined)?.width) || 0,
        ),
      );
  const inferredHeight = opts.height
    ? Math.max(opts.height, 0)
    : Math.max(
        ...clips.map(
          (c) =>
            Number((c.transform as { height?: number } | undefined)?.height) ||
            0,
        ),
      );
  const w = Math.max(1, inferredWidth || 1920);
  const h = Math.max(1, inferredHeight || 1080);

  const renderer = new KonvaExportRenderer({
    width: w,
    height: h,
  });

  // precompute duration if not provided
  let startFrame = 0;
  let endFrame = 0;
  const globalStart =
    range?.start ?? Math.min(...clips.map((c) => c.startFrame ?? 0), 0);
  const globalEnd =
    range?.end ?? Math.max(...clips.map((c) => c.endFrame ?? 0), 0);
  startFrame = Math.max(0, globalStart);
  endFrame = Math.max(startFrame + 1, globalEnd);
  const totalFrames = Math.max(1, endFrame - startFrame);

  // Prepare applicators per clip using shared hald clut (only if needed by filters)
  const hald = acquireHaldClut();

  // Maintain per-clip video iterators across frames (for this export run)
  type IterCtx = {
    key: string;
    iter: AsyncIterator<WrappedCanvas | null>;
    currentProjectIndex: number;
  };
  const videoIters: Map<string, IterCtx> = new Map<string, IterCtx>();

  const drawFrame = async (frame: number) => {
    checkCancelled();

    renderer.clearStage();

    if (opts.backgroundColor) {
      const layer = renderer.getLayer();
      const bg = new Rect({
        x: 0,
        y: 0,
        width: w,
        height: h,
        fill: opts.backgroundColor,
        listening: false,
      });
      layer.add(bg);
    }

    // Active clips at this frame
    const active = clips.filter((c) => {
      const s = c.startFrame ?? 0 - (range?.start ? (c.startFrame ?? 0) : 0);
      const e = c.endFrame ?? 0 - (range?.start ? (c.startFrame ?? 0) : 0);
      return frame >= s && frame <= e;
    });

    // Preserve the incoming order of clips; draw in that sequence
    const ordered = active;

    for (const clip of ordered) {
      // Gather applicators bound to this clip, active at this frame
      const bound =
        clip.applicators && Array.isArray(clip.applicators)
          ? (clip.applicators as ExportApplicatorClip[])
          : [];
      const activeApps = bound.filter((a) => {
        const s = a.startFrame ?? Number.NEGATIVE_INFINITY;
        const e = a.endFrame ?? Number.POSITIVE_INFINITY;
        return frame >= s && frame <= e;
      });

      const applicatorsRaw = activeApps
        .map((a) =>
          createApplicatorFromClip(a as any, {
            haldClutInstance: hald,
            focusFrameOverride: frame,
          }),
        )
        .filter(
          (x): x is NonNullable<ReturnType<typeof createApplicatorFromClip>> =>
            x !== null,
        );
      const applicators = applicatorsRaw as unknown as Array<{
        apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
        ensureResources?: () => Promise<void>;
      }>;

      if (clip.type === "image") {
        await renderer.blitImage(
          clip as unknown as BlitImageClipProps,
          applicators,
          frame,
        );
      } else if (clip.type === "video") {
        const c = clip as ExportVideoClip;
        const { selectedSrc, frameOffset } = resolveVideoSourceForFrame(
          c,
          frame,
        );
        const speed = (() => {
          const s = Number(c?.speed ?? 1);
          return Number.isFinite(s) && s > 0
            ? Math.min(5, Math.max(0.1, s))
            : 1;
        })();

        // When clips are split, the editor encodes the effective media window on
        // the `src` as `?startFrame=...&endFrame=...` (see `clip.ts:1513-1535`).
        // Treat this `startFrame` as an additional trim offset on top of the
        // clip's own `trimStart` so exports honor cut clips exactly as seen in
        // the timeline/preview.

        const { srcStartFrame } = getSrcFrameWindow(selectedSrc);
        const trimStartFrames = Math.max(0, Number(c?.trimStart) || 0);

        const clipStartFrame = Number(c?.startFrame) || 0;
        const clipEndFrame = Number(c?.endFrame);
        const localFrame = Math.max(0, frame - clipStartFrame);

        let spanStartLocal = 0;
        let spanEndLocal = Number.isFinite(clipEndFrame)
          ? clipEndFrame - clipStartFrame
          : Number.MAX_SAFE_INTEGER;

        if (selectedSrc === c.src) {
          const preprocessors = c.preprocessors || [];
          // Find last preprocessor end before current frame
          const prevP = preprocessors
            .filter(
              (p) =>
                p.status === "complete" &&
                p.src &&
                (Number(p.endFrame) ?? Number(p.startFrame) ?? 0) < localFrame,
            )
            .sort(
              (a, b) => (Number(b.endFrame) ?? 0) - (Number(a.endFrame) ?? 0),
            )[0];
          spanStartLocal = prevP ? (Number(prevP.endFrame) ?? 0) + 1 : 0;

          // Find next preprocessor start after current frame
          const nextP = preprocessors
            .filter(
              (p) =>
                p.status === "complete" &&
                p.src &&
                (Number(p.startFrame) || 0) > localFrame,
            )
            .sort(
              (a, b) => (Number(a.startFrame) || 0) - (Number(b.startFrame) || 0),
            )[0];

          if (nextP) {
            spanEndLocal = Number(nextP.startFrame) || 0;
          }
        } else {
          // Preprocessor
          const p = (c.preprocessors || []).find((p) => p.src === selectedSrc);
          spanStartLocal = p ? Number(p.startFrame) || 0 : 0;
          const pEndLocal = p
            ? Number(p.endFrame) ?? Number(p.startFrame) ?? 0
            : 0;
          spanEndLocal = pEndLocal + 1;
        }

        const spanStartSourceIndex =
          selectedSrc === c.src
            ? trimStartFrames + srcStartFrame + Math.floor(spanStartLocal * speed)
            : srcStartFrame;

        const spanDuration = Math.max(0, spanEndLocal - spanStartLocal);
        const spanEndSourceIndex =
          spanStartSourceIndex + Math.floor(spanDuration * speed);

        const currentSourceIndex =
          selectedSrc === c.src
            ? trimStartFrames + srcStartFrame + Math.floor(localFrame * speed)
            : srcStartFrame +
              Math.floor(Math.max(0, localFrame - spanStartLocal) * speed);

        const iterKey = `${selectedSrc}|s:${speed}|pfps:${fps}|spanStart:${spanStartSourceIndex}|spanEnd:${spanEndSourceIndex}`;

        let ctxIter = videoIters.get(c.clipId) as IterCtx | undefined;
        const reusable = !!(
          ctxIter &&
          ctxIter.key === iterKey &&
          ctxIter.currentProjectIndex <= currentSourceIndex
        );

        if (!reusable) {
          const asyncIterable = await getVideoFrameIterator(selectedSrc, {
            projectFps: Math.max(1, fps),
            startIndex: currentSourceIndex,
            endIndex: spanEndSourceIndex,
            speed,
            useOriginal: true,
          });
          ctxIter = {
            key: iterKey,
            iter: asyncIterable[Symbol.asyncIterator](),
            currentProjectIndex: currentSourceIndex,
          };
          videoIters.set(c.clipId, ctxIter);
        }

        // Always advance if we are behind the target frame
        while ((ctxIter as IterCtx).currentProjectIndex < currentSourceIndex) {
          await (ctxIter as IterCtx).iter.next();
          (ctxIter as IterCtx).currentProjectIndex++;
        }


        await renderer.blitVideo(
          c as unknown as BlitVideoClipProps,
          applicators,
          (ctxIter as IterCtx).iter,
          frame,
        );
        (ctxIter as IterCtx).currentProjectIndex++;
      } else if (clip.type === "text") {
        await renderer.blitText(
          clip as unknown as BlitTextClipProps,
          applicators,
        );
      } else if (clip.type === "shape") {
        await renderer.blitShape(
          clip as unknown as BlitShapeClipProps,
          applicators,
        );
      } else if (clip.type === "draw") {
        await renderer.blitDrawing(
          clip as unknown as BlitDrawingClipProps,
          applicators,
        );
      } else {
        continue;
      }
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
    outFormat: "wav" | "mp3",
    nameHint?: string,
    filename?: string,
  ): Promise<string | null> => {
    try {
      if (!mixSpecs || mixSpecs.length === 0) return null;
      const out = await renderAudioMixWithFfmpeg(mixSpecs as any, {
        fps: exportFps,
        exportStartFrame,
        exportEndFrame,
        outFormat,
        fileNameHint: nameHint,
        filename: filename,
      });
      return out as string | null;
    } catch (e) {
      console.error(e, "error rendering audio mix");
      return null;
    }
  };

  let audioPath: string | undefined = undefined;
  try {
    if (mode === "image") {
      checkCancelled();
      const frame = typeof imageFrame === "number" ? imageFrame : startFrame;
      await drawFrame(frame);
      // Single-frame export; report completion.
      if (onProgress) {
        onProgress({ currentFrame: 1, totalFrames: 1, ratio: 1 });
      }
      // Download image
      const blob = await renderer.toBlob({ mimeType: "image/png" });
      if (!cancelToken?.cancelled) {
        try {
          opts.onDone?.();
        } catch {}
      }
      return blob || new Blob();
    }

    // Determine audio usage
    const audioClips = clips.filter(
      (c) => (c as any).type === "audio",
    ) as unknown[] as ExportAudioClip[];
    // Only include videos that have an explicit external/attached audioSrc.
    const videoAudioClips = includeAudio
      ? (clips.filter(
          (c) =>
            (c as any).type === "video" &&
            typeof (c as any).audioSrc === "string" &&
            (c as any).audioSrc,
        ) as unknown[] as ExportVideoClip[])
      : ([] as ExportVideoClip[]);
    const allAudio =
      clips.length > 0 && clips.every((c) => (c as any).type === "audio");

    if (allAudio) {
      checkCancelled();
      const specsAll = [
        ...audioClips.map((c) => {
          const { srcStartFrame } = getSrcFrameWindow(
            (c as any).src as string | undefined,
          );
          const trimStart =
            Math.max(0, Number((c as any)?.trimStart) || 0) + srcStartFrame;
          return {
            src: (c as any).src as string,
            startFrame: Number(c.startFrame) || 0,
            endFrame:
              typeof c.endFrame === "number" ? Number(c.endFrame) : endFrame,
            trimStart,
            volumeDb: Number((c as any)?.volume || 0),
            fadeInSec: Math.max(0, Number((c as any)?.fadeIn) || 0),
            fadeOutSec: Math.max(0, Number((c as any)?.fadeOut) || 0),
            speed: (() => {
              const s = Number((c as any)?.speed ?? 1);
              return Number.isFinite(s) && s > 0
                ? Math.min(5, Math.max(0.1, s))
                : 1;
            })(),
          };
        }),
      ];
      const outPath = await renderAudioMixToFile(
        specsAll,
        startFrame,
        endFrame,
        fps,
        audioOptions?.format ?? "mp3",
        filename ?? "output.mp3",
        filename,
      );
      checkCancelled();
      if (onProgress) {
        onProgress({ currentFrame: totalFrames, totalFrames, ratio: 1 });
      }
      if (!cancelToken?.cancelled) {
        try {
          opts.onDone?.();
        } catch {}
      }
      return outPath || undefined;
    }

    if (includeAudio) {
      checkCancelled();
      const specs = [
        ...audioClips.map((c) => {
          const { srcStartFrame } = getSrcFrameWindow(
            (c as any).src as string | undefined,
          );
          const trimStart =
            Math.max(0, Number((c as any)?.trimStart) || 0) + srcStartFrame;
          return {
            src: (c as any).src as string,
            startFrame: Number(c.startFrame) || 0,
            endFrame:
              typeof c.endFrame === "number" ? Number(c.endFrame) : endFrame,
            trimStart,
            volumeDb: Number((c as any)?.volume || 0),
            fadeInSec: Math.max(0, Number((c as any)?.fadeIn) || 0),
            fadeOutSec: Math.max(0, Number((c as any)?.fadeOut) || 0),
            speed: (() => {
              const s = Number((c as any)?.speed ?? 1);
              return Number.isFinite(s) && s > 0
                ? Math.min(5, Math.max(0.1, s))
                : 1;
            })(),
          };
        }),
        // For video clips, only use the explicit audioSrc and ignore clips without one.
        ...videoAudioClips.map((c) => {
          const { srcStartFrame } = getSrcFrameWindow(
            String((c as any).audioSrc || "") || undefined,
          );
          const trimStart =
            Math.max(0, Number((c as any)?.trimStart) || 0) + srcStartFrame;
          return {
            src: String((c as any).audioSrc || ""),
            startFrame: Number(c.startFrame) || 0,
            endFrame:
              typeof c.endFrame === "number" ? Number(c.endFrame) : endFrame,
            trimStart,
            volumeDb: 0,
            fadeInSec: 0,
            fadeOutSec: 0,
            speed: (() => {
              const s = Number((c as any)?.speed ?? 1);
              return Number.isFinite(s) && s > 0
                ? Math.min(5, Math.max(0.1, s))
                : 1;
            })(),
          };
        }),
      ];
      if (specs.length > 0) {
        const outPath = await renderAudioMixToFile(
          specs,
          startFrame,
          endFrame,
          fps,
          audioOptions?.format ?? "mp3",
          (filename ?? "temp_audio") + "." + (audioOptions?.format ?? "mp3"),
        );
        if (outPath) audioPath = outPath;
      }
    }

    const encoder = new FfmpegFrameEncoder({
      filename: filename ?? "output.mp4",
      ...(encoderOptions ?? {}),
    });
    if (!encoder)
      throw new Error(
        "No encoder provided. Please provide a FrameEncoder implemented in preload or main.",
      );
    await encoder.start({ width: w, height: h, fps, audioPath });

    for (let f = startFrame; f < endFrame; f++) {
      checkCancelled();
      await drawFrame(f);
      const blob = await renderer.toBlob({ mimeType: "image/png" });
      if (blob) {
        const buffer = new Uint8Array(await blob.arrayBuffer());
        await encoder.addFrame(buffer);
      }
      if (onProgress) {
        const currentFrame = Math.max(1, f - startFrame + 1);
        const clampedRatio = Math.min(
          1,
          Math.max(0, currentFrame / totalFrames),
        );
        onProgress({ currentFrame, totalFrames, ratio: clampedRatio });
      }
    }

    const result = await encoder.finalize();
    if (!cancelToken?.cancelled) {
      try {
        opts.onDone?.();
      } catch {}
    }
    return result;
  } finally {
    try {
      if (audioPath) await deleteFile(audioPath);
    } catch {}
    try {
      cleanupVideoDecoders();
    } catch {}
    renderer.destroy();
    releaseHaldClut();
  }
}

export async function exportClip(
  opts: ExportClipOptions,
): Promise<Blob | Uint8Array | string | void> {
  const {
    clip,
    fps,
    mode,
    imageFrame,
    range,
    encoderOptions,
    filename,
    audioOptions = { format: "mp3" },
    onProgress,
  } = opts;
  const includeAudio =
    (opts as any)?.includeAudio !== undefined
      ? !!(opts as any).includeAudio
      : true;
  const cancelToken = (opts as any)?.cancelToken as
    | ExportCancelToken
    | undefined;

  const checkCancelled = () => {
    if (cancelToken?.cancelled) {
      throw new ExportCancelledError();
    }
  };

  // For image/video clips, normalize the transform and output canvas so that:
  // - The clip.transform width/height correspond to the requested export size
  //   (width/height coming in via opts or existing transform/originalTransform)
  // - The canvas dimensions are tightly aligned to that transform and scaled
  //   to any normalized crop present on the transform.
  let workingClip = clip as ExportClip;

  // Resolve output dimensions
  const isImageOrVideo =
    workingClip.type === "image" || workingClip.type === "video";
  const fromTransformWidth =
    Number((workingClip.transform as { width?: number } | undefined)?.width) ||
    0;
  const fromTransformHeight =
    Number(
      (workingClip.transform as { height?: number } | undefined)?.height,
    ) || 0;

  // Base canvas size before any crop is applied.
  const inferredWidth = isImageOrVideo
    ? opts.width || fromTransformWidth || 0
    : Math.max(opts.width || 0, fromTransformWidth);
  const inferredHeight = isImageOrVideo
    ? opts.height || fromTransformHeight || 0
    : Math.max(opts.height || 0, fromTransformHeight);

  // If the clip has a normalized crop, shrink the export canvas so the
  // output resolution matches the cropped region instead of the full rect.
  const crop = (workingClip.transform as any)?.crop as
    | { width?: number; height?: number }
    | undefined;
  const cropWidthRatio =
    crop &&
    typeof crop.width === "number" &&
    isFinite(crop.width) &&
    crop.width > 0
      ? crop.width
      : 1;
  const cropHeightRatio =
    crop &&
    typeof crop.height === "number" &&
    isFinite(crop.height) &&
    crop.height > 0
      ? crop.height
      : 1;

  const w = Math.max(1, (inferredWidth || 1920) * cropWidthRatio);
  const h = Math.max(1, (inferredHeight || 1080) * cropHeightRatio);


  const renderer = new KonvaExportRenderer({
    width: w,
    height: h,
  });

  // Local duration relative to this clip
  const clipStartGlobal = Number(workingClip.startFrame ?? 0);
  const clipEndGlobal = Number(workingClip.endFrame ?? clipStartGlobal + 1);
  const localDuration = Math.max(
    1,
    Math.max(0, clipEndGlobal - clipStartGlobal),
  );
  const startFrame = range?.start ?? 0;
  const endFrame = range?.end ?? localDuration;
  const totalFrames = Math.max(1, Math.max(0, endFrame - startFrame));

  // Prepare applicators using shared hald clut
  const hald = acquireHaldClut();

  // Maintain per-clip video iterators across frames (for this export run)
  type IterCtx = {
    key: string;
    iter: AsyncIterator<WrappedCanvas | null>;
    currentProjectIndex: number;
  };

  const videoIters: Map<string, IterCtx> = new Map<string, IterCtx>();

  const drawFrame = async (frame: number) => {
    checkCancelled();
    renderer.clearStage();

    if (opts.backgroundColor) {
      const layer = renderer.getLayer();
      const bg = new Rect({
        x: 0,
        y: 0,
        width: w,
        height: h,
        fill: opts.backgroundColor,
        listening: false,
      });
      layer.add(bg);
    }

    // Applicators bound to this clip, active at this frame (LOCAL frame indexing)
    const bound =
      workingClip.applicators && Array.isArray(workingClip.applicators)
        ? (workingClip.applicators as ExportApplicatorClip[])
        : [];
    const activeApps = bound.filter((a) => {
      const s = a.startFrame ?? Number.NEGATIVE_INFINITY;
      const e = a.endFrame ?? Number.POSITIVE_INFINITY;
      return frame >= s && frame <= e;
    });

    const applicatorsRaw = activeApps
      .map((a) =>
        createApplicatorFromClip(a as any, {
          haldClutInstance: hald,
          focusFrameOverride: frame,
        }),
      )
      .filter(
        (x): x is NonNullable<ReturnType<typeof createApplicatorFromClip>> =>
          x !== null,
      );
    
    const applicators = applicatorsRaw as unknown as Array<{
      apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
      ensureResources?: () => Promise<void>;
    }>;

    if (workingClip.type === "image") {
      await renderer.blitImage(
        workingClip as unknown as BlitImageClipProps,
        applicators,
        frame,
      );
    } else if (workingClip.type === "video") {
      const { selectedSrc, frameOffset } = resolveVideoSourceForFrame(
        workingClip as ExportVideoClip,
        frame,
      );
      const speed = (() => {
        const s = Number((workingClip as any)?.speed ?? 1);
        return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
      })();
      const { srcStartFrame } = getSrcFrameWindow(selectedSrc);
      const trimStartFrames = Math.max(
        0,
        Number((workingClip as any)?.trimStart) || 0,
      );

      // In exportClip, 'frame' is already local to the clip (0-based)
      const localFrame = frame;
      const clipDuration = localDuration;

      let spanStartLocal = 0;
      let spanEndLocal = clipDuration;

      if (selectedSrc === (workingClip as ExportVideoClip).src) {
        const preprocessors =
          (workingClip as ExportVideoClip).preprocessors || [];
        // Find last preprocessor end before current frame
        const prevP = preprocessors
          .filter(
            (p) =>
              p.status === "complete" &&
              p.src &&
              (Number(p.endFrame) ?? Number(p.startFrame) ?? 0) < localFrame,
          )
          .sort(
            (a, b) => (Number(b.endFrame) ?? 0) - (Number(a.endFrame) ?? 0),
          )[0];
        spanStartLocal = prevP ? (Number(prevP.endFrame) ?? 0) + 1 : 0;

        // Find next preprocessor start after current frame
        const nextP = preprocessors
          .filter(
            (p) =>
              p.status === "complete" &&
              p.src &&
              (Number(p.startFrame) || 0) > localFrame,
          )
          .sort(
            (a, b) => (Number(a.startFrame) || 0) - (Number(b.startFrame) || 0),
          )[0];

        if (nextP) {
          spanEndLocal = Number(nextP.startFrame) || 0;
        }
      } else {
        // Preprocessor
        const p = (
          (workingClip as ExportVideoClip).preprocessors || []
        ).find((p) => p.src === selectedSrc);
        spanStartLocal = p ? Number(p.startFrame) || 0 : 0;
        const pEndLocal = p
          ? Number(p.endFrame) ?? Number(p.startFrame) ?? 0
          : 0;
        spanEndLocal = pEndLocal + 1;
      }

      const spanStartSourceIndex =
        selectedSrc === (workingClip as ExportVideoClip).src
          ? trimStartFrames + srcStartFrame + Math.floor(spanStartLocal * speed)
          : srcStartFrame;

      const spanDuration = Math.max(0, spanEndLocal - spanStartLocal);
      const spanEndSourceIndex =
        spanStartSourceIndex + Math.floor(spanDuration * speed);

      const currentSourceIndex =
        selectedSrc === (workingClip as ExportVideoClip).src
          ? trimStartFrames + srcStartFrame + Math.floor(localFrame * speed)
          : srcStartFrame +
            Math.floor(Math.max(0, localFrame - spanStartLocal) * speed);

      const iterKey = `${selectedSrc}|s:${speed}|pfps:${fps}|spanStart:${spanStartSourceIndex}|spanEnd:${spanEndSourceIndex}`;

      let ctxIter = videoIters.get(workingClip.clipId) as IterCtx | undefined;
      const reusable = !!(
        ctxIter &&
        ctxIter.key === iterKey &&
        ctxIter.currentProjectIndex <= currentSourceIndex
      );

      if (!reusable) {
        const asyncIterable = await getVideoFrameIterator(selectedSrc, {
          projectFps: Math.max(1, fps),
          startIndex: currentSourceIndex,
          endIndex: spanEndSourceIndex,
          speed,
          useOriginal: true,
        });
        ctxIter = {
          key: iterKey,
          iter: asyncIterable[Symbol.asyncIterator](),
          currentProjectIndex: currentSourceIndex,
        };
        videoIters.set(workingClip.clipId, ctxIter);
      }

      while ((ctxIter as IterCtx).currentProjectIndex < currentSourceIndex) {
        await (ctxIter as IterCtx).iter.next();
        (ctxIter as IterCtx).currentProjectIndex++;
      }
      
      await renderer.blitVideo(
        workingClip as unknown as BlitVideoClipProps,
        applicators,
        (ctxIter as IterCtx).iter,
        frame,
      );
      (ctxIter as IterCtx).currentProjectIndex++;
    } else if (workingClip.type === "text") {
      await renderer.blitText(
        workingClip as unknown as BlitTextClipProps,
        applicators,
      );
    } else if (workingClip.type === "shape") {
      await renderer.blitShape(
        workingClip as unknown as BlitShapeClipProps,
        applicators,
      );
    } else if (workingClip.type === "draw") {
      await renderer.blitDrawing(
        workingClip as unknown as BlitDrawingClipProps,
        applicators,
      );
    } else {
      // audio/filter-only clip does not draw to canvas
    }
  };

  // Helper: render audio mix for provided audio clip using ffmpeg (pitch-preserving)
  const renderAudioMixToFile = async (
    audioClip: ExportAudioClip,
    exportStartFrame: number,
    exportEndFrame: number,
    exportFps: number,
    outFormat: "wav" | "mp3",
    nameHint?: string,
    filename?: string,
  ): Promise<string | null> => {
    try {
      if (!audioClip) return null;
      const { srcStartFrame } = getSrcFrameWindow(
        (audioClip as any).src as string | undefined,
      );
      const baseTrimStart = Math.max(
        0,
        Number((audioClip as any)?.trimStart) || 0,
      );
      const effectiveTrimStart = baseTrimStart + srcStartFrame;
      const specs = {
        src: (audioClip as any).src as string,
        startFrame: Number(audioClip.startFrame) || 0,
        endFrame:
          typeof audioClip.endFrame === "number"
            ? Number(audioClip.endFrame)
            : exportEndFrame,
        trimStart: effectiveTrimStart,
        volumeDb: Number((audioClip as any)?.volume || 0),
        fadeInSec: Math.max(0, Number((audioClip as any)?.fadeIn) || 0),
        fadeOutSec: Math.max(0, Number((audioClip as any)?.fadeOut) || 0),
        speed: (() => {
          const s = Number((audioClip as any)?.speed ?? 1);
          return Number.isFinite(s) && s > 0
            ? Math.min(5, Math.max(0.1, s))
            : 1;
        })(),
      };
      const out = await renderAudioMixWithFfmpeg([specs], {
        fps: exportFps,
        exportStartFrame,
        exportEndFrame,
        outFormat,
        fileNameHint: nameHint,
        filename,
      });
      return out as string | null;
    } catch {
      return null;
    }
  };

  let audioPath: string | undefined = undefined;
  try {
    if (mode === "image") {
      checkCancelled();
      const frame = typeof imageFrame === "number" ? imageFrame : startFrame;
      await drawFrame(frame);
      if (onProgress) {
        onProgress({ currentFrame: 1, totalFrames: 1, ratio: 1 });
      }
      const blob = await renderer.toBlob({ mimeType: "image/png" });
      if (!cancelToken?.cancelled) {
        try {
          opts.onDone?.();
        } catch {}
      }
      return blob || new Blob();
    }

    if (workingClip.type === "audio") {
      checkCancelled();
      const outPath = await renderAudioMixToFile(
        workingClip as unknown as ExportAudioClip,
        startFrame,
        endFrame,
        fps,
        audioOptions?.format ?? "mp3",
        filename ?? "output.mp3",
        filename,
      );
      if (onProgress) {
        onProgress({ currentFrame: totalFrames, totalFrames, ratio: 1 });
      }
      if (!cancelToken?.cancelled) {
        try {
          opts.onDone?.();
        } catch {}
      }
      return outPath || undefined;
    }

    // Optional: include audio when exporting a single video clip if an audio source is attached
    if (includeAudio && workingClip.type === "video") {
      checkCancelled();
      const spec = [
        {
          src: String((workingClip as any).src),
          startFrame: Number(workingClip.startFrame) || 0,
          endFrame:
            typeof workingClip.endFrame === "number"
              ? Number(workingClip.endFrame)
              : endFrame,
          trimStart: (() => {
            const { srcStartFrame } = getSrcFrameWindow(
              String((workingClip as any).src),
            );
            const baseTrimStart = Math.max(
              0,
              Number((workingClip as any)?.trimStart) || 0,
            );
            return baseTrimStart + srcStartFrame;
          })(),
          volumeDb: 0,
          fadeInSec: 0,
          fadeOutSec: 0,
          speed: (() => {
            const s = Number((workingClip as any)?.speed ?? 1);
            return Number.isFinite(s) && s > 0
              ? Math.min(5, Math.max(0.1, s))
              : 1;
          })(),
        },
      ];
      try {
        const out = await renderAudioMixWithFfmpeg(spec as any, {
          fps,
          exportStartFrame: startFrame,
          exportEndFrame: endFrame,
          outFormat: audioOptions?.format ?? "mp3",
          fileNameHint:
            (filename ?? "temp_audio") + "." + (audioOptions?.format ?? "mp3"),
        });
        if (out) audioPath = String(out);
      } catch {}
    }

    const encoder = new FfmpegFrameEncoder({
      filename: filename ?? "output.mp4",
      ...({
        ...encoderOptions ? encoderOptions : {},
        resolution: { width: w, height: h },
      }),
    });
    if (!encoder)
      throw new Error(
        "No encoder provided. Please provide a FrameEncoder implemented in preload or main.",
      );
    await encoder.start({
      width: w,
      height: h,
      fps,
      ...(audioPath ? { audioPath } : {}),
    });

    for (let f = startFrame; f < endFrame; f++) {
      checkCancelled();
      await drawFrame(f);
      const blob = await renderer.toBlob({ mimeType: "image/png" });
      if (blob) {
        const buffer = new Uint8Array(await blob.arrayBuffer());
        await encoder.addFrame(buffer);
      }
      if (onProgress) {
        const currentFrame = Math.max(1, f - startFrame + 1);
        const ratio = Math.min(1, Math.max(0, currentFrame / totalFrames));
        onProgress({ currentFrame, totalFrames, ratio });
      }
    }

    const result = await encoder.finalize();
    if (!cancelToken?.cancelled) {
      try {
        opts.onDone?.();
      } catch {}
    }
    return result;
  } finally {
    try {
      if (audioPath) await deleteFile(audioPath);
    } catch {}
    try {
      cleanupVideoDecoders();
    } catch {}
    renderer.destroy();
    releaseHaldClut();
  }
}
