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
import { getMediaInfo, getMediaInfoCached } from "../../renderer/src/lib/media/utils";
import {
  renderAudioMixWithFfmpeg,
  deleteFile,
  sha256sum,
  exportCacheGet,
  exportCachePut,
  exportCacheMaterialize,
  exportVideoTranscodeWithFfmpeg,
} from "@app/preload";
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

const EXPORT_CACHE_VERSION = 1;

const getExtFromFilename = (p?: string | null): string | null => {
  if (!p) return null;
  const base = String(p).split(/[\\/]/).pop() || "";
  const idx = base.lastIndexOf(".");
  if (idx <= 0 || idx >= base.length - 1) return null;
  return base.slice(idx + 1).toLowerCase();
};

const stableStringify = (value: any): string => {
  const seen = new WeakSet<object>();
  const walk = (v: any): any => {
    if (v === null) return null;
    const t = typeof v;
    if (t === "string" || t === "number" || t === "boolean") return v;
    if (t === "bigint") return v.toString();
    if (t === "undefined") return undefined;
    if (t === "function") return undefined;
    if (t !== "object") return String(v);
    if (v instanceof Date) return v.toISOString();
    if (Array.isArray(v)) return v.map(walk);
    if (seen.has(v)) return "[Circular]";
    seen.add(v);
    const out: Record<string, any> = {};
    const keys = Object.keys(v).sort();
    for (const k of keys) {
      const next = walk(v[k]);
      if (typeof next !== "undefined") out[k] = next;
    }
    return out;
  };
  return JSON.stringify(walk(value));
};

const buildExportRequestHash = (payload: any): string => {
  return sha256sum(stableStringify({ v: EXPORT_CACHE_VERSION, ...payload }));
};


const getAudioSrc = async (src:string): Promise<URL | null> => {
  if (!src) return null;
  let mediaInfo = await getMediaInfoCached(src);
  if (!mediaInfo) mediaInfo = await getMediaInfo(src);
  // @ts-ignore
  return mediaInfo.audio?.input?.source?._url as URL;
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

const hasAnyVisualEffects = (c: ExportVideoClip): boolean => {
  const any =
    Number(c.brightness || 0) !== 0 ||
    Number(c.contrast || 0) !== 0 ||
    Number(c.hue || 0) !== 0 ||
    Number(c.saturation || 0) !== 0 ||
    Number(c.blur || 0) !== 0 ||
    Number(c.noise || 0) !== 0 ||
    Number(c.sharpness || 0) !== 0 ||
    Number(c.vignette || 0) !== 0;
  return !!any;
};

const canUseFfmpegFastPathForVideoClip = (
  clip: ExportVideoClip,
): boolean => {
  // Conservative: only allow pure video transforms that do not require canvas compositing.
  if (!clip || clip.type !== "video") return false;
  if (clip.normalizedTransform) return false;
  if (hasAnyVisualEffects(clip)) return false;
  if (Array.isArray(clip.masks) && clip.masks.length > 0) return false;
  if (Array.isArray(clip.applicators) && clip.applicators.length > 0) return false;
  if (Array.isArray(clip.preprocessors) && clip.preprocessors.length > 0) return false;

  const t = (clip.transform || {}) as TransformLike;
  // Only support simple geometry (crop + implicit resize). Anything needing pad/overlay/rotate is deferred.
  const rot = Number(t.rotation || 0);
  const sx = t.scaleX === undefined ? 1 : Number(t.scaleX);
  const sy = t.scaleY === undefined ? 1 : Number(t.scaleY);
  const x = Number(t.x || 0);
  const y = Number(t.y || 0);
  const opacity = t.opacity === undefined ? 100 : Number(t.opacity);

  if (!Number.isFinite(rot) || rot !== 0) return false;
  if (!Number.isFinite(sx) || sx !== 1) return false;
  if (!Number.isFinite(sy) || sy !== 1) return false;
  if (!Number.isFinite(x) || x !== 0) return false;
  if (!Number.isFinite(y) || y !== 0) return false;
  if (!Number.isFinite(opacity) || opacity !== 100) return false;
  if (t.cornerRadius !== undefined && Number(t.cornerRadius) !== 0) return false;

  // Crop (normalized) is supported; validate ranges lightly.
  if ((t as any).crop) {
    const c = (t as any).crop as any;
    const ok =
      c &&
      Number.isFinite(Number(c.x)) &&
      Number.isFinite(Number(c.y)) &&
      Number.isFinite(Number(c.width)) &&
      Number.isFinite(Number(c.height));
    if (!ok) return false;
  }

  return true;
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

  // Compute export time range early (used both for cache key and for progress when cache hits).
  const globalStart =
    range?.start ?? Math.min(...clips.map((c) => c.startFrame ?? 0), 0);
  const globalEnd =
    range?.end ?? Math.max(...clips.map((c) => c.endFrame ?? 0), 0);
  const startFrame = Math.max(0, globalStart);
  const endFrame = Math.max(startFrame + 1, globalEnd);
  const totalFrames = Math.max(1, endFrame - startFrame);

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

  // Persistent export cache: for audio/video exports with an explicit output filename,
  // short-circuit if we've already exported the exact same content/options before.
  const canUsePersistentCache = (mode === "video" || mode === "audio") &&
    typeof filename === "string" &&
    filename.length > 0;

  const inferredExt =
    mode === "video"
      ? String((encoderOptions as any)?.format || getExtFromFilename(filename) || "mp4")
      : mode === "audio"
        ? String(audioOptions?.format || getExtFromFilename(filename) || "mp3")
        : null;

  const exportHash = canUsePersistentCache
    ? buildExportRequestHash({
        fn: "exportSequence",
        mode,
        fps,
        includeAudio,
        range: { start: startFrame, end: endFrame },
        imageFrame,
        width: w,
        height: h,
        backgroundColor: opts.backgroundColor,
        encoderOptions: encoderOptions ?? null,
        audioOptions: audioOptions ?? null,
        clips,
      })
    : null;

  if (canUsePersistentCache && exportHash) {
    checkCancelled();
    const cachedPath = await exportCacheGet(exportHash);
    if (cachedPath) {
      // Ensure the requested output is present at the caller-specified path.
      const out = await exportCacheMaterialize(exportHash, filename as string);
      if (out) {
        if (onProgress) {
          onProgress({ currentFrame: totalFrames, totalFrames, ratio: 1 });
        }
        try {
          opts.onDone?.();
        } catch {}
        return out;
      }
    }
  }

  // Fast path: if this export is a single, effect-free video clip, do it directly in ffmpeg.
  if (
    mode === "video" &&
    clips.length === 1 &&
    clips[0]?.type === "video" &&
    canUseFfmpegFastPathForVideoClip(clips[0] as ExportVideoClip) &&
    !opts.backgroundColor
  ) {
    checkCancelled();
    const c = clips[0] as ExportVideoClip;
    const speed = (() => {
      const s = Number(c?.speed ?? 1);
      return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
    })();
    const { srcStartFrame } = getSrcFrameWindow(c.src);
    const trimStartFrames = Math.max(0, Number(c?.trimStart) || 0);
    const clipStartFrame = Number(c?.startFrame) || 0;

    const localStart = Math.max(0, startFrame - clipStartFrame);
    const localDurationFrames = Math.max(1, endFrame - startFrame);

    const srcStartFrames =
      trimStartFrames + srcStartFrame + Math.floor(localStart * speed);
    const srcDurationFrames = Math.floor(localDurationFrames * speed);
    const srcStartSec = srcStartFrames / Math.max(1, fps);
    const srcDurationSec = Math.max(1, srcDurationFrames) / Math.max(1, fps);

    const crop = (c.transform as any)?.crop as
      | { x?: number; y?: number; width?: number; height?: number }
      | undefined;

    const audioSrc =
      typeof (c as any).audioSrc === "string" && (c as any).audioSrc
        ? String((c as any).audioSrc)
        : undefined;

    const outPath = await exportVideoTranscodeWithFfmpeg({
      videoSrc: c.src,
      audioSrc: includeAudio ? audioSrc : undefined,
      includeAudio,
      // Preserve previous exportSequence behavior: only include video audio when an explicit audioSrc exists.
      allowEmbeddedAudio: !!audioSrc,
      outAbs: filename ?? "output.mp4",
      fps,
      srcStartSec,
      srcDurationSec,
      speed,
      crop: crop
        ? {
            x: Number(crop.x || 0),
            y: Number(crop.y || 0),
            width: Number(crop.width || 1),
            height: Number(crop.height || 1),
          }
        : undefined,
      width: (encoderOptions as any)?.resolution?.width ?? w,
      height: (encoderOptions as any)?.resolution?.height ?? h,
      format: (encoderOptions as any)?.format,
      codec: (encoderOptions as any)?.codec,
      preset: (encoderOptions as any)?.preset,
      crf: (encoderOptions as any)?.crf,
      bitrate: (encoderOptions as any)?.bitrate,
      alpha: (encoderOptions as any)?.alpha,
    });

    checkCancelled();
    if (onProgress) {
      onProgress({ currentFrame: totalFrames, totalFrames, ratio: 1 });
    }
    if (canUsePersistentCache && exportHash && typeof outPath === "string") {
      try {
        await exportCachePut(exportHash, outPath, {
          ext: inferredExt || undefined,
        });
      } catch {}
    }
    if (!cancelToken?.cancelled) {
      try {
        opts.onDone?.();
      } catch {}
    }
    return outPath;
  }

  const renderer = new KonvaExportRenderer({
    width: w,
    height: h,
  });

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
        const { selectedSrc } = resolveVideoSourceForFrame(
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
            speed: 1,
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
      const audioSpecs = await Promise.all(
        mixSpecs.map(async (spec) => {
          return {
            ...spec,
            src: (await getAudioSrc(spec.src))?.toString(),
          };
        }),
      );

      let safeFilename = filename;
      if (typeof safeFilename === "string" && safeFilename.length > 0) {
        const desiredExt = `.${outFormat}`;
        const lower = safeFilename.toLowerCase();
        if (!lower.endsWith(desiredExt)) {
          if (safeFilename.includes(".")) {
            safeFilename = safeFilename.replace(/\.[^.]+$/, desiredExt);
          } else {
            safeFilename = `${safeFilename}${desiredExt}`;
          }
        }
      }

      const out = await renderAudioMixWithFfmpeg(audioSpecs as any, {
        fps: exportFps,
        exportStartFrame,
        exportEndFrame,
        outFormat,
        fileNameHint: nameHint,
        filename: safeFilename,
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
      if (canUsePersistentCache && exportHash && outPath) {
        try {
          await exportCachePut(exportHash, outPath, { ext: inferredExt || undefined });
        } catch {}
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
    if (canUsePersistentCache && exportHash && typeof result === "string" && result) {
      try {
        await exportCachePut(exportHash, result, { ext: inferredExt || undefined });
      } catch {}
    }
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


  const canUsePersistentCache =
    (mode === "video" || mode === "audio") &&
    typeof filename === "string" &&
    filename.length > 0;
  const inferredExt =
    mode === "video"
      ? String((encoderOptions as any)?.format || getExtFromFilename(filename) || "mp4")
      : mode === "audio"
        ? String(audioOptions?.format || getExtFromFilename(filename) || "mp3")
        : null;


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

  const exportHash = canUsePersistentCache
    ? buildExportRequestHash({
        fn: "exportClip",
        mode,
        fps,
        includeAudio,
        range: { start: startFrame, end: endFrame },
        imageFrame,
        width: w,
        height: h,
        backgroundColor: opts.backgroundColor,
        encoderOptions: encoderOptions ?? null,
        audioOptions: audioOptions ?? null,
        clip: workingClip,
      })
    : null;

  if (canUsePersistentCache && exportHash) {
    checkCancelled();
    const cachedPath = await exportCacheGet(exportHash);
    if (cachedPath) {
      const out = await exportCacheMaterialize(exportHash, filename as string);
      if (out) {
        if (onProgress) {
          onProgress({ currentFrame: totalFrames, totalFrames, ratio: 1 });
        }
        try {
          opts.onDone?.();
        } catch {}
        return out;
      }
    }
  }

  // Fast path: single video clip export that requires no canvas compositing.

  if (
    mode === "video" &&
    workingClip.type === "video" &&
    canUseFfmpegFastPathForVideoClip(workingClip as ExportVideoClip) &&
    (!opts.backgroundColor || opts.backgroundColor === "#000000")
  ) {
    checkCancelled();
    const c = workingClip as ExportVideoClip;
    const speed = (() => {
      const s = Number((c as any)?.speed ?? 1);
      return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
    })();
    const { srcStartFrame } = getSrcFrameWindow(c.src);
    const trimStartFrames = Math.max(0, Number((c as any)?.trimStart) || 0);

    const localStart = Math.max(0, startFrame);
    const localDurationFrames = Math.max(1, endFrame - startFrame);

    const srcStartFrames =
      trimStartFrames + srcStartFrame + Math.floor(localStart * speed);
    const srcDurationFrames = Math.floor(localDurationFrames * speed);
    const srcStartSec = srcStartFrames / Math.max(1, fps);
    const srcDurationSec = Math.max(1, srcDurationFrames) / Math.max(1, fps);

    const crop = (c.transform as any)?.crop as
      | { x?: number; y?: number; width?: number; height?: number }
      | undefined;

    const audioSrc =
      typeof (c as any).audioSrc === "string" && (c as any).audioSrc
        ? String((c as any).audioSrc)
        : undefined;

    const outPath = await exportVideoTranscodeWithFfmpeg({
      videoSrc: c.src,
      audioSrc: includeAudio ? (audioSrc || undefined) : undefined,
      includeAudio,
      // For exportClip, allow embedded audio by default (matches previous behavior where we derived audio from the same src).
      allowEmbeddedAudio: true,
      outAbs: filename ?? "output.mp4",
      fps,
      srcStartSec,
      srcDurationSec,
      speed,
      crop: crop
        ? {
            x: Number(crop.x || 0),
            y: Number(crop.y || 0),
            width: Number(crop.width || 1),
            height: Number(crop.height || 1),
          }
        : undefined,
      width: (encoderOptions as any)?.resolution?.width ?? w,
      height: (encoderOptions as any)?.resolution?.height ?? h,
      format: (encoderOptions as any)?.format,
      codec: (encoderOptions as any)?.codec,
      preset: (encoderOptions as any)?.preset,
      crf: (encoderOptions as any)?.crf,
      bitrate: (encoderOptions as any)?.bitrate,
      alpha: (encoderOptions as any)?.alpha,
    });

    checkCancelled();
    if (onProgress) {
      onProgress({ currentFrame: totalFrames, totalFrames, ratio: 1 });
    }
    if (canUsePersistentCache && exportHash && typeof outPath === "string") {
      try {
        await exportCachePut(exportHash, outPath, {
          ext: inferredExt || undefined,
        });
      } catch {}
    }
    if (!cancelToken?.cancelled) {
      try {
        opts.onDone?.();
      } catch {}
    }
    return outPath;
  }

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
      const { selectedSrc } = resolveVideoSourceForFrame(
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
          speed:1,
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

      const audioSrc = await getAudioSrc(audioClip.src);

      
      if (!audioSrc) return null;

      const effectiveTrimStart = baseTrimStart + srcStartFrame;
      const specs = {
        src: audioSrc.toString(),
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

      let safeFilename = filename;
      if (typeof safeFilename === "string" && safeFilename.length > 0) {
        const desiredExt = `.${outFormat}`;
        const lower = safeFilename.toLowerCase();
        if (!lower.endsWith(desiredExt)) {
          if (safeFilename.includes(".")) {
            safeFilename = safeFilename.replace(/\.[^.]+$/, desiredExt);
          } else {
            safeFilename = `${safeFilename}${desiredExt}`;
          }
        }
      }

      const out = await renderAudioMixWithFfmpeg([specs], {
        fps: exportFps,
        exportStartFrame,
        exportEndFrame,
        outFormat,
        fileNameHint: nameHint,
        filename: safeFilename,
      });
      return out as string | null;
    } catch (e) {
      console.error("Error rendering audio mix to file", e);
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
      if (canUsePersistentCache && exportHash && outPath) {
        try {
          await exportCachePut(exportHash, outPath, { ext: inferredExt || undefined });
        } catch {}
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
      try {
        const out = await renderAudioMixToFile(workingClip as unknown as ExportAudioClip, startFrame, endFrame, fps, audioOptions?.format ?? "mp3", filename ?? "output.mp3", filename);
        if (out) audioPath = String(out);
      } catch (e) {
        console.error("Error rendering audio mix to file", e);
      }
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
    if (canUsePersistentCache && exportHash && typeof result === "string" && result) {
      try {
        await exportCachePut(exportHash, result, { ext: inferredExt || undefined });
      } catch {}
    }
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
