import {
  PacketStats,
  InputVideoTrack,
  InputAudioTrack,
  MetadataTags,
  InputFormat,
  Input,
} from "mediabunny";
import { Preprocessor } from "./preprocessor";
import { ManifestDocument } from "./manifest/api";

export type ClipType =
  | "video"
  | "image"
  | "audio"
  | "model"
  | "text"
  | "shape"
  | "draw"
  | "filter"
  | "group";
export type TimelineType =
  | "media"
  | "audio"
  | "model"
  | "text"
  | "shape"
  | "draw"
  | "filter";
export type ViewTool = "pointer" | "hand" | "mask" | "draw" | "shape" | "text";
export type ShapeTool = "rectangle" | "ellipse" | "polygon" | "line" | "star";

export interface MediaAdjustments {
  // Color Correction
  brightness?: number; // isFilter
  contrast?: number; // isFilter
  exposure?: number;
  hue?: number; // isFilter
  saturation?: number; // isFilter
  // Effects
  sharpness?: number;
  noise?: number; // isFilter
  blur?: number; // isFilter
  vignette?: number;
}

export interface ClipTransform {
  x: number;
  y: number;
  width: number;
  height: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  cornerRadius: number;
  opacity: number;
  crop?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface ClipProps {
  // For grouping
  groupId?: string;
  hidden?: boolean;
  //
  timelineId?: string;
  startFrame: number;
  endFrame: number;
  trimEnd?: number;
  trimStart?: number;
  clipPadding?: number;
  clipId: string;
  // Persisted transform for preview canvas (position/size/scale/rotation)
  transform?: ClipTransform;
  originalTransform?: ClipTransform;
  type: ClipType;
}

export interface TimelineProps {
  type: TimelineType;
  timelineId: string;
  timelineWidth: number;
  timelineY: number;
  timelineHeight: number;
  timelinePadding: number;
  muted: boolean;
  hidden: boolean;
}

export type PreprocessorClipType = VideoClipProps | ImageClipProps;

export type Asset = {
  id: string;
  type: 'video' | 'image' | 'audio';
  path: string;
  width?: number;
  height?: number;
  duration: number;
  thumbnail?: string;
  modelInputAsset?: boolean;
}

export type VideoClipProps = ClipProps &
  MediaAdjustments & {
    
    assetId: string;
    assetIdHistory: string[];
    type: "video";
    // Intrinsic media dimensions for consistent aspect ratio
    mediaWidth?: number;
    mediaHeight?: number;
    mediaAspectRatio?: number; // width / height
    volume?: number;
    fadeIn?: number;
    fadeOut?: number;
    speed?: number;
    preprocessors: PreprocessorClipProps[];
    masks: MaskClipProps[];
  };

export type ImageClipProps = ClipProps &
  MediaAdjustments & {
    assetId: string;
    assetIdHistory: string[];
    type: "image";
    // Intrinsic media dimensions for consistent aspect ratio
    mediaWidth?: number;
    mediaHeight?: number;
    mediaAspectRatio?: number; // width / height
    preprocessors: PreprocessorClipProps[];
    masks: MaskClipProps[];
  };

export type AudioClipProps = ClipProps & {
  assetId: string;
  type: "audio";
  volume?: number;
  fadeIn?: number;
  fadeOut?: number;
  speed?: number;
};

export type ShapeClipProps = ClipProps & {
  type: "shape";
  shapeType?: ShapeTool;
  fill?: string;
  fillOpacity?: number;
  stroke?: string;
  strokeOpacity?: number;
  strokeWidth?: number;
};

export type PolygonClipProps = ShapeClipProps & {
  shapeType: "polygon";
  sides?: number;
};

export type StarClipProps = ShapeClipProps & {
  shapeType: "star";
  points?: number;
};

export type TextClipProps = ClipProps & {
  type: "text";
  text?: string;
  fontSize?: number;
  fontWeight?: number;
  fontStyle?: "normal" | "italic";
  fontFamily?: string;
  color?: string;
  colorOpacity?: number;
  textAlign?: "left" | "center" | "right";
  verticalAlign?: "top" | "middle" | "bottom";
  textTransform?: "none" | "uppercase" | "lowercase" | "capitalize";
  textDecoration?: "none" | "underline" | "overline" | "line-through";
  // Stroke properties
  strokeEnabled?: boolean;
  stroke?: string;
  strokeWidth?: number;
  strokeOpacity?: number;
  // Shadow properties
  shadowEnabled?: boolean;
  shadowColor?: string;
  shadowOpacity?: number;
  shadowBlur?: number;
  shadowOffsetX?: number;
  shadowOffsetY?: number;
  shadowOffsetLocked?: boolean;
  // Background properties
  backgroundEnabled?: boolean;
  backgroundColor?: string;
  backgroundOpacity?: number;
  backgroundCornerRadius?: number;
};

export type FilterClipProps = ClipProps & {
  name?: string;
  type: "filter";
  smallPath?: string;
  fullPath?: string;
  category?: string;
  examplePath?: string;
  exampleAssetUrl?: string;
  intensity?: number;
};

export interface DrawingLineTransform {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  opacity: number;
}

export interface DrawingLine {
  lineId: string;
  tool: "brush" | "highlighter" | "eraser";
  points: number[]; // [x1, y1, x2, y2, ...]
  stroke: string;
  strokeWidth: number;
  opacity: number;
  smoothing: number; // 0-1 tension value
  transform: DrawingLineTransform;
}

export type DrawingClipProps = ClipProps & {
  type: "draw";
  lines: DrawingLine[];
};

export type MaskTransform = ClipTransform;

export type MaskTool = "lasso" | "shape" | "draw" | "touch";
export type MaskShapeTool = "rectangle" | "ellipse" | "polygon" | "star";
export type MaskTrackingDirection = "forward" | "backward" | "both";

export interface MaskData {
  // For lasso tool - closed path points
  lassoPoints?: number[]; // [x1, y1, x2, y2, ...]
  /**
   * Optional path to an external Float32Array binary file containing the
   * lasso points for this mask data. When present, the main process will
   * have serialized the lassoPoints array into this .bin file to keep the
   * JSON snapshot compact.
   */
  lassoPointsBinPath?: string;
  // For shape tool
  shapeBounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation?: number;
    shapeType?: MaskShapeTool;
    scaleX?: number;
    scaleY?: number;
    renderOnce?: boolean;
  };

  // For touch/SAM2 tool - AI generated mask or selection points
  touchPoints?: Array<{ x: number; y: number; label: 1 | 0 }>; // positive/negative
  touchBox?: { x1: number; y1: number; x2: number; y2: number };
  contours?: number[][];
  /**
   * Optional path to an external Float32Array binary file containing the
   * contours for this mask data. When present, the main process will have
   * serialized the contours 2D number[][] into this .bin file to keep the
   * JSON snapshot compact.
   */
  contoursBinPath?: string;
  // Generated mask data (binary mask as base64 encoded image or URL)
  maskImageData?: string;
}

export type MaskClipProps = {
  id: string;
  clipId?: string;
  tool: MaskTool;
  featherAmount: number;
  brushSize?: number;
  // Mask data for the initial frame/keyframes
  keyframes: Map<number, MaskData> | Record<number, MaskData>;
  // Tracking settings
  isTracked: boolean;
  trackingDirection?: MaskTrackingDirection;
  confidenceThreshold?: number;
  // Transform applied to mask
  transform?: MaskTransform;
  // Metadata
  createdAt: number;
  lastModified: number;
  // Operation settings
  inverted?: boolean; // Invert the mask
  backgroundColor?: string;
  backgroundOpacity?: number;
  backgroundColorEnabled?: boolean;
  maskColor?: string;
  maskOpacity?: number;
  maskColorEnabled?: boolean;
  maxTrackingFrames?: number;
};

export type PreprocessorClipProps = {
  assetId?: string;
  clipId?: string;
  id: string;
  preprocessor: Preprocessor;
  startFrame?: number;
  endFrame?: number;
  values: Record<string, any>;
  /**
   * When enabled, once this preprocessor has a completed result it will be
   * converted into a new clip on the timeline above its parent clip, and this
   * preprocessor will be removed from the parent.
   */
  createNewClip?: boolean;
  status?: "running" | "complete" | "failed";
  activeJobId?: string;
  jobIds?: string[];
  /**
   * Ephemeral progress percent (0..100) for the active run.
   * Not persisted; used only for UI rendering.
   */
  progress?: number;
};

export type GroupClipProps = ClipProps & {
  type: "group";
  children: string[][]; // clipIds for the children of the group
};

export type GenerationModelClipProps = {
  jobId: string;
  modelStatus: "pending" | "running" | "complete" | "failed";
  assetId: string;
  createdAt: number;
  selectedComponents?: Record<string, any>;
  values?: Record<string, any>;
  src?: string;
  // Persist the clip transform used when this generation was previewed/applied
  transform?: ClipTransform;
};

export type ModelClipProps = ClipProps & {
  assetId?: string;
  assetIdHistory?: string[];
  mediaWidth?: number;
  mediaHeight?: number;
  mediaAspectRatio?: number;
  previewPath?: string;
  type: "model";
  manifest: ManifestDocument;
  // Persist only per-clip UI input values here; the manifest JSON
  // saved to disk remains free of user-specific `value` fields.
  modelInputValues?: Record<string, any>;
  speed?: number;
  modelStatus?: "pending" | "running" | "complete" | "failed";
  // Persist user selections for model components (e.g., scheduler, transformer, vae, text_encoder)
  // Keyed by component type (e.g., 'scheduler', 'transformer'), value is a small descriptor object
  selectedComponents?: Record<string, any>;
  /**
   * Optional per-component offloading configuration.
   * Keyed by component key (usually manifest component `name` if present, else `type`).
   *
   * NOTE: This is UI-driven config and is forwarded to the engine as
   * `selected_components.offload` when present.
   */
  offload?: Record<
    string,
    {
      enabled?: boolean;
      level?: "leaf" | "block";
      num_blocks?: number;
      use_stream?: boolean;
      record_stream?: boolean;
    }
  >;
  generations?: GenerationModelClipProps[];
  activeJobId?: string;
};

export type AnyClipProps =
  | VideoClipProps
  | ImageClipProps
  | AudioClipProps
  | ShapeClipProps
  | PolygonClipProps
  | TextClipProps
  | FilterClipProps
  | DrawingClipProps
  | GroupClipProps
  | ModelClipProps;

// Persisted, type-specific properties for a clip, excluding the shared base ClipProps
export type ClipSpecificProps = Omit<AnyClipProps, keyof ClipProps>;

export type ZoomLevel = number;

export type InputImageTrack = {
  width: number;
  height: number;
  mime?: string;
  size?: number; // bytes (if Blob/File provided or fetched)
  animated?: boolean; // GIF/WebP/AVIF (if ImageDecoder path)
  orientation?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8; // EXIF tag 0x0112 (JPEG only here)
  input?: Blob | File | ArrayBuffer | Uint8Array | string,
};

export type MediaInfo = {
  path: string;
  video: InputVideoTrack | null;
  audio: (InputAudioTrack & { sampleSize?: number }) | null;
  image: InputImageTrack | null;
  stats: {
    video: PacketStats | undefined;
    audio: PacketStats | undefined;
  };
  duration: number | undefined;
  metadata: MetadataTags | undefined;
  mimeType: string | undefined;
  format: InputFormat | undefined;
  startFrame?: number;
  endFrame?: number;
  originalInput?: Input; // Only used for converting media to 24 fps
  videoDecoderConfig?: VideoDecoderConfig;
};

export type FrameBatch = {
  path: string;
  start_frame: number;
  end_frame: number;
  width: number;
  height: number;
  pixel_format: "rgba8";
  frames: number[][] | Uint8Array[];
};

export interface Point {
  x: number;
  y: number;
}

export type SidebarSection =
  | "media"
  | "models"
  | "filters"
  | "templates"
  | "generations"
  | "transitions"
  | "effects";

export interface Filter {
  id: string;
  name: string;
  smallPath: string;
  fullPath: string;
  category: string;
  examplePath: string;
  exampleAssetUrl: string;
}

export type FilterWithType = Filter & {
  type: "filter";
};

// Fast, compact signature generator for AnyClipProps to use in React deps
// Focuses on stable, lightweight fields and summarizes large arrays
export function clipSignature(clip: AnyClipProps): string {
  try {
    const parts: string[] = [];

    // Common/base fields
    parts.push(
      clip.type,
      clip.clipId,
      clip.timelineId ?? "",
      n(clip.startFrame),
      n(clip.endFrame),
      n(clip.trimStart),
      n(clip.trimEnd),
      n(clip.clipPadding),
      clip.hidden ? "1" : "0",
      clip.groupId ?? "",
      encodeTransform(clip.transform),
      encodeTransform(clip.originalTransform),
    );

    // Per-type specific fields (keep minimal for speed)
    switch (clip.type) {
      case "video": {
        parts.push(
          (clip as any)?.assetId ?? "",
          n(clip.mediaWidth),
          n(clip.mediaHeight),
          n(clip.mediaAspectRatio),
          n(clip.volume),
          n(clip.fadeIn),
          n(clip.fadeOut),
          n(clip.speed),
          preprocessorsSignature((clip as any)?.preprocessors),
          masksSignature((clip as any)?.masks),
        );
        break;
      }
      case "image": {
        parts.push(
          (clip as any)?.assetId ?? "",
          n(clip.mediaWidth),
          n(clip.mediaHeight),
          n(clip.mediaAspectRatio),
          preprocessorsSignature((clip as any)?.preprocessors),
          masksSignature((clip as any)?.masks),
        );
        break;
      }
      case "audio": {
        const a = clip as any;
        parts.push(
          a?.assetId ?? "",
          n(a?.volume),
          n(a?.fadeIn),
          n(a?.fadeOut),
          n(a?.speed),
        );
        break;
      }
      case "shape": {
        const s = clip as any;
        parts.push(
          s.shapeType ?? "",
          s.fill ?? "",
          n(s.fillOpacity),
          s.stroke ?? "",
          n(s.strokeOpacity),
          n(s.strokeWidth),
        );
        break;
      }
      case "text": {
        const t = clip as any;
        const text = t.text ?? "";
        const textPrefix =
          text.length > 32 ? text.slice(0, 32) + "\u2026" : text;
        parts.push(
          String(text.length),
          textPrefix,
          n(t.fontSize),
          n(t.fontWeight),
          t.fontStyle ?? "",
          t.fontFamily ?? "",
          t.color ?? "",
          n(t.colorOpacity),
          t.textAlign ?? "",
          t.verticalAlign ?? "",
          t.textTransform ?? "",
          t.textDecoration ?? "",
          t.strokeEnabled ? "1" : "0",
          t.stroke ?? "",
          n(t.strokeWidth),
          n(t.strokeOpacity),
          t.shadowEnabled ? "1" : "0",
          t.shadowColor ?? "",
          n(t.shadowOpacity),
          n(t.shadowBlur),
          n(t.shadowOffsetX),
          n(t.shadowOffsetY),
          t.shadowOffsetLocked ? "1" : "0",
          t.backgroundEnabled ? "1" : "0",
          t.backgroundColor ?? "",
          n(t.backgroundOpacity),
          n(t.backgroundCornerRadius),
        );
        break;
      }
      case "filter": {
        const f = clip as any;
        parts.push(
          f.name ?? "",
          f.smallPath ?? "",
          f.fullPath ?? "",
          f.category ?? "",
          f.examplePath ?? "",
          f.exampleAssetUrl ?? "",
          n(f.intensity),
        );
        break;
      }
      case "draw": {
        parts.push(linesSignature((clip as any)?.lines));
        break;
      }
      case "group": {
        parts.push(groupChildrenSignature((clip as any)?.children));
        break;
      }
      case "model": {
        const m = clip as any;
        parts.push(
          m.assetId ?? "",
          m.category ?? "",
          manifestIdSignature(m.manifest),
        );
        break;
      }
      default: {
        break;
      }
    }

    return parts.join("|");
  } catch (_err) {
    return fallbackSignature(clip);
  }
}

function fallbackSignature(clip: AnyClipProps): string {
  try {
    const t = (clip as any)?.type ?? "";
    const id = (clip as any)?.clipId ?? "";
    const tl = (clip as any)?.timelineId ?? "";
    const src = typeof (clip as any)?.assetId === "string" ? (clip as any).assetId : "";
    return [t, id, tl, src, "ERR"].join("|");
  } catch {
    return "CLIP|ERR";
  }
}

function n(value: number | undefined | null): string {
  if (value === null || value === undefined) return "";
  if (typeof value !== "number") return "";
  if (!Number.isFinite(value)) return "";
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function encodeTransform(t: ClipTransform | undefined): string {
  if (!t) return "";
  return [
    n(t.x),
    n(t.y),
    n(t.width),
    n(t.height),
    n(t.scaleX),
    n(t.scaleY),
    n(t.rotation),
    n(t.cornerRadius),
    n(t.opacity),
    encodeCrop(t.crop),
  ].join(",");
}

function encodeCrop(c: { x: number; y: number; width: number; height: number } | undefined): string {
  if (!c) return "";
  return [
    n(c.x),
    n(c.y),
    n(c.width),
    n(c.height),
  ].join(",");
}



function encodeLineTransform(t: DrawingLineTransform | undefined): string {
  if (!t) return "";
  return [
    n(t.x),
    n(t.y),
    n(t.scaleX),
    n(t.scaleY),
    n(t.rotation),
    n(t.opacity),
  ].join(",");
}

function preprocessorsSignature(
  arr: PreprocessorClipProps[] | undefined,
): string {
  if (!arr || arr.length === 0) return "";
  const parts: string[] = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const p = arr[i];
    const values = p.values ? fastValuesSignature(p.values) : "";
    parts[i] = [
      p.id,
      String(p.preprocessor),
      n(p.startFrame),
      n(p.endFrame),
      p.status ?? "",
      values,
    ].join("^");
  }
  return parts.join("~");
}

function fastValuesSignature(values: Record<string, any>): string {
  const keys = Object.keys(values).sort();
  if (keys.length === 0) return "";
  const out: string[] = new Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const v = (values as any)[k];
    const t = typeof v;
    let s: string;
    if (t === "number") s = n(v as number);
    else if (t === "string" || t === "boolean") s = String(v);
    else if (v == null) s = "";
    else s = "o"; // object/array - avoid heavy stringify
    out[i] = k + ":" + s;
  }
  return out.join(",");
}

function masksSignature(arr: MaskClipProps[] | undefined): string {
  if (!arr || arr.length === 0) return "";
  const parts: string[] = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const m = arr[i];
    const kf: any = m.keyframes as any;
    let kfCount = 0;
    if (kf) {
      if (typeof Map !== "undefined" && kf instanceof Map)
        kfCount = (kf as Map<any, any>).size;
      else if (typeof kf === "object") kfCount = Object.keys(kf).length;
    }
    parts[i] = [
      m.id,
      m.tool,
      n(m.featherAmount),
      n(m.brushSize),
      encodeTransform(m.transform),
      m.backgroundColor ?? "",
      n(m.backgroundOpacity),
      m.backgroundColorEnabled ? "1" : "0",
      m.maskColor ?? "",
      n(m.maskOpacity),
      m.maskColorEnabled ? "1" : "0",
      m.isTracked ? "1" : "0",
      m.inverted ? "1" : "0",
      kfCount.toString(),
    ].join(",");
  }
  return parts.join("|");
}

function linesSignature(lines: DrawingLine[] | undefined): string {
  if (!lines || lines.length === 0) return "";
  const count = lines.length;
  let totalPoints = 0;
  let rolling = 0;
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    const len = ln.points ? ln.points.length : 0;
    totalPoints += len;
    // fast rolling hash from lengths and widths
    rolling =
      ((rolling << 5) - rolling) ^
      (len & 0xffff) ^
      ((ln.strokeWidth | 0) & 0xffff);
  }
  const first = lines[0];
  const last = lines[count - 1];
  return [
    count.toString(),
    totalPoints.toString(),
    (rolling >>> 0).toString(36), // compact
    lineKey(first),
    lineKey(last),
  ].join("|");
}

function lineKey(l: DrawingLine | undefined): string {
  if (!l) return "";
  return [
    l.lineId,
    l.tool,
    (l.points ? l.points.length : 0).toString(),
    l.stroke,
    n(l.strokeWidth),
    n(l.opacity),
    n(l.smoothing),
    encodeLineTransform(l.transform),
  ].join(",");
}

function groupChildrenSignature(children: string[][]): string {
  if (!children || children.length === 0) return "";
  const groups = children.length;
  let total = 0;
  for (let i = 0; i < children.length; i++) total += children[i]?.length ?? 0;
  const first = children[0]?.[0] ?? "";
  const lastGroup = children[children.length - 1];
  const last =
    lastGroup && lastGroup.length > 0 ? lastGroup[lastGroup.length - 1] : "";
  return [groups.toString(), total.toString(), first, last].join(",");
}

function manifestIdSignature(manifest: ManifestDocument): string {
  const m: any = manifest as any;
  const id = m?.id ?? m?.name ?? m?.slug ?? "";
  return typeof id === "string" || typeof id === "number" ? String(id) : "";
}
