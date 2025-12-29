import _ from "lodash";

import type {
  AnyClipProps,
  Asset,
  ClipTransform,
  MaskClipProps,
  TimelineProps,
  VideoClipProps,
} from "@/lib/types";
import { BASE_LONG_SIDE } from "@/lib/settings";
import {
  getMediaInfoCached,
  convertApexCachePath,
} from "@/lib/media/utils";
import type { Canvas, ExportClip } from "@app/export-renderer";
import { sortClipsForStacking } from "@/lib/clipOrdering";
import { remapMaskForMediaDialog } from "@/lib/mask/transformUtils";

export interface PrepareExportClipsContext {
  aspectRatio: { width: number; height: number };
  getClipsForGroup: (children: any) => AnyClipProps[];
  getClipsByType: (type: any) => AnyClipProps[];
  getAssetById: (assetId: string) => Asset | undefined;
  getClipPositionScore: (clipId: any) => number;
  timelines: TimelineProps[];
}
type NormalizedTransform = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

function ensureTransform(
  transform?: ClipTransform,
  canvas?: Canvas,
): Required<Omit<ClipTransform, "crop">> & { crop?: ClipTransform["crop"] } {
  const defaultWidth = canvas?.width ?? 100;
  const defaultHeight = canvas?.height ?? 100;
  return {
    x: transform?.x ?? 0,
    y: transform?.y ?? 0,
    width: transform?.width ?? defaultWidth,
    height: transform?.height ?? defaultHeight,
    scaleX: transform?.scaleX ?? 1,
    scaleY: transform?.scaleY ?? 1,
    rotation: transform?.rotation ?? 0,
    cornerRadius: transform?.cornerRadius ?? 0,
    opacity: transform?.opacity ?? 100,
    crop: transform?.crop,
  };
}

function resolveTransformFromClip(
  transform: ClipTransform | undefined,
  normalizedTransform: NormalizedTransform | undefined,
  canvas?: Canvas,
): Required<Omit<ClipTransform, "crop">> & { crop?: ClipTransform["crop"] } {
  const base = ensureTransform(transform, canvas);
  if (!canvas || !normalizedTransform) return base;

  const cw = Math.max(1, canvas.width || 1);
  const ch = Math.max(1, canvas.height || 1);

  const toPx = (normVal: any, dim: number, fallback: number): number => {
    const n = Number(normVal);

    if (!Number.isFinite(n)) return fallback;
    // Allow values outside [0, 1] so off‑canvas normalized positions/sizes
    // are preserved when mapped back into pixel space.
    return n * dim;
  };

  const x = toPx(normalizedTransform.x, cw, base.x);
  const y = toPx(normalizedTransform.y, ch, base.y);
  const width = toPx(normalizedTransform.width, cw, base.width);
  const height = toPx(normalizedTransform.height, ch, base.height);

  return {
    ...base,
    x,
    y,
    width,
    height,
  };
}

export interface PrepareExportClipsOptions {
  /**
   * When true, any `masks` array present on the source clip or its children
   * will be cleared. This mirrors the behavior used for map target inputs in
   * `runModelGeneration`.
   */
  clearMasks?: boolean;
  /**
   * When true, image/video clips will have their `transform` adjusted so that
   * they are centered within the inferred width/height, taking scale and
   * rotation into account. This matches the centering logic currently used in
   * `runModelGeneration`. For timeline exports (Topbar) this should be false
   * so we respect the stage transform as–is.
   */
  applyCentering?: boolean;
  useOriginalTransform?: boolean;
  useMediaDimensionsForExport?: boolean;
  /**
   * Controls how the base width/height are inferred for centering and
   * normalization:
   * - 'clip' (default): use the clip's own transform/media dimensions when
   *   available (image/video), otherwise fall back to the aspect ratio.
   * - 'aspect': always derive width/height from the provided `aspectRatio`
   *   (canvas space), regardless of clip type.
   */
  dimensionsFrom?: "clip" | "aspect";
  target?: {
    width: number;
    height: number;
  };
}

export interface PreparedExportClipsResult {
  exportClips: ExportClip[];
  width: number;
  height: number;
  offsetStart: number;
}

/**
 * Create a normalized view of a clip transform relative to a canvas.
 *
 * Notes:
 * - All properties are normalized against the canvas size with **no clamping**.
 *   Values may be < 0 (off–canvas) or > 1 (larger than the canvas) and are
 *   preserved as–is in normalized space.
 * - This does NOT mutate the original transform; callers should assign the
 *   returned object where needed.
 */
export const normalizeTransformForCanvas = (
  transform: any,
  canvasWidth: number,
  canvasHeight: number,
) => {
  if (!transform || typeof transform !== "object") return transform;

  const w = canvasWidth > 0 ? canvasWidth : 1;
  const h = canvasHeight > 0 ? canvasHeight : 1;

  const toNorm = (value: any, denom: number) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return 0;
    // Preserve sign and allow values outside [0, 1] for off‑canvas or
    // over‑canvas positions/sizes.
    return num / denom;
  };

  const normalized = { ...transform };
  normalized.x = toNorm(normalized.x, w);
  normalized.y = toNorm(normalized.y, h);
  normalized.width = toNorm(normalized.width, w);
  normalized.height = toNorm(normalized.height, h);

  return normalized;
};

const hasAudio = (value: AnyClipProps): boolean => {
  if (value.type !== "video") return false;
  const mediaInfo = getMediaInfoCached(value.assetId);
  return mediaInfo?.audio !== null;
};

const convertCoerceModelToMedia = (clip: AnyClipProps, getAssetById: (assetId: string) => Asset | undefined): AnyClipProps => {
  if (clip.type == "model" && clip.assetId) {
    // check if model has assetId and status is complete
    const asset = getAssetById(clip.assetId);
    // check if asset is a video or image
    return {
      ...clip,
      type: asset?.type === "video" ? "video" : "image",
      assetId: clip.assetId,
      assetIdHistory: clip.assetIdHistory,
      modelStatus: "complete",
      preprocessors: [],
      masks: [],
    } as VideoClipProps;
  }
  return clip;
}

/**
 * Convert a single clip (which may be a group) into one or more `ExportClip`s,
 * applying:
 * - media path normalization (user-data paths → filesystem paths)
 * - group flattening with local start/end frames
 * - preprocessor src normalization (cache paths)
 * - filter/applicator attachment based on clip position score
 * - optional mask clearing for map targets
 * - optional transform centering based on inferred width/height
 *
 * This function intentionally mirrors the export preparation logic that was
 * previously inlined inside `runModelGeneration`, so that both model
 * generation previews and timeline exports (Topbar) can share the same
 * behavior.
 */
export function prepareExportClipsForValue(
  rawValue: AnyClipProps,
  ctx: PrepareExportClipsContext,
  options: PrepareExportClipsOptions = {},
): PreparedExportClipsResult {
  const {
    aspectRatio,
    getClipsForGroup,
    getClipsByType,
    getClipPositionScore,
    timelines,
    getAssetById,
  } = ctx;
  const {
    clearMasks = false,
    applyCentering = true,
    useOriginalTransform = false,
    dimensionsFrom = "clip",
    useMediaDimensionsForExport = false,
    target,
  } = options;

  // Clone so we don't mutate the caller's state
  const value: AnyClipProps & {
    src?: string;
    children?: any;
    originalTransform?: any;
    transform?: any;
    preprocessors?: any[];
    masks?: any[];
    startFrame?: number;
    audioSrc?: string | null;
    endFrame?: number;
  } = { ...(convertCoerceModelToMedia(rawValue, getAssetById)) };

  // For map targets we clear masks at the root before any processing
  if (clearMasks && Object.prototype.hasOwnProperty.call(value, "masks")) {
    value.masks = [];
  }

  // --- Infer width/height for this value (used for centering & normalization) ---
  let width = 0;
  let height = 0;
  const isImage = value.type === "image";
  const isVideo = value.type === "video";
  const isAudio = value.type === "audio";

  if (isImage) {
    const asset = getAssetById(value.assetId);
    if (!asset) return {
      exportClips: [],
      width: 0,
      height: 0,
      offsetStart: value.startFrame ?? 0,
    };
    const mediaInfo = getMediaInfoCached(asset.path);
    value.src = asset.path
    let transform = useOriginalTransform
      ? (value as any).originalTransform
      : (value as any).transform;

    if (transform && !transform?.width && useOriginalTransform) {
      transform.width = value.transform?.width
    }
    if (transform && !transform?.height && useOriginalTransform) {
      transform.height = value.transform?.height;
    }

    if (!mediaInfo) {
      // Mirror previous behavior: if media info is missing, skip this value
      return {
        exportClips: [],
        width: 0,
        height: 0,
        offsetStart: value.startFrame ?? 0,
      };
    }


    if (dimensionsFrom === "clip") {
      width = (transform?.width && transform.width !== 0) ? transform.width : (mediaInfo?.image?.width ?? 0);
      height = (transform?.height && transform.height !== 0) ? transform.height : (mediaInfo?.image?.height ?? 0);
    }
  } else if (isVideo) {
    const asset = getAssetById(value.assetId);
    const mediaInfo = getMediaInfoCached(value.assetId);
    const transform = useOriginalTransform
      ? (value as any).originalTransform
      : (value as any).transform;
    
    if (!asset) {
      return {
        exportClips: [],
        width: 0,
        height: 0,
        offsetStart: value.startFrame ?? 0,
      };
    }
    
    value.src = asset.path;

    if (!mediaInfo) {
      return {
        exportClips: [],
        width: 0,
        height: 0,
        offsetStart: value.startFrame ?? 0,
      };
    }

    if (dimensionsFrom === "clip") {
      width = transform?.width ?? mediaInfo?.video?.displayWidth ?? 0;
      height = transform?.height ?? mediaInfo?.video?.displayHeight ?? 0;
    }
  } else if (isAudio) {
    const asset = getAssetById(value.assetId);
    if (!asset) return {
      exportClips: [],
      width: 0,
      height: 0,
      offsetStart: value.startFrame ?? 0,
    };
    value.src = asset.path;
  }

  // If requested (or for non-image/video types), derive width/height from the
  // canvas aspect ratio instead of the clip's intrinsic dimensions.
  if (dimensionsFrom === "aspect" || (!isImage && !isVideo)) {
    const ratio = aspectRatio.width / aspectRatio.height;
    const baseShortSide = BASE_LONG_SIDE;
    if (Number.isFinite(ratio) && ratio > 0) {
      width = baseShortSide * ratio;
      height = baseShortSide;
    } else {
      width = 0;
      height = 0;
    }
  }

  // --- Expand groups into local child clips & compute offsetStart ---
  let clips: AnyClipProps[] = [value as AnyClipProps];
  let offsetStart = 0;

  if (value.type === "group") {
    const groupedClips = getClipsForGroup((value as any).children);
    const groupStart = value.startFrame ?? 0;
    offsetStart = groupStart;
    clips = groupedClips.map((c) => {
      const newClip: AnyClipProps & {
        src?: string;
        preprocessors?: any[];
        masks?: any[];
        startFrame?: number;
        endFrame?: number;
        audioSrc?: string | null;
      } = { ...(c as any) };
      newClip.startFrame = (c.startFrame ?? 0) - groupStart;
      newClip.endFrame = (c.endFrame ?? 0) - groupStart;

      if (
        clearMasks &&
        Object.prototype.hasOwnProperty.call(newClip, "masks")
      ) {
        newClip.masks = [];
      }

      if (newClip.type === "image") {
        const asset = getAssetById(newClip.assetId);
        if (!asset) return newClip;
        newClip.src = asset.path;
        const mediaInfo = getMediaInfoCached(newClip.src as string);
        if (!mediaInfo) return newClip;
      }
      if (newClip.type === "video") {
        const asset = getAssetById(newClip.assetId);
        if (!asset) return newClip;
        newClip.src = asset.path;
      }

      if (newClip.type === "video") {
        newClip.audioSrc = hasAudio(newClip) ? newClip.src : null;
      }


      if (Object.prototype.hasOwnProperty.call(newClip, "preprocessors")) {
        (newClip as any).preprocessors =
          (c as any).preprocessors?.map((p: any) => {
            const asset = getAssetById(p.assetId);
            if (!asset) return p;
            p.src = asset.path;
            const convertedSrc =
              p?.status === "complete" && p?.src
                ? convertApexCachePath(p.src)
                : p?.src;
            return {
              ...p,
              src: convertedSrc,
              startFrame: (p.startFrame ?? 0) - groupStart,
              endFrame: (p.endFrame ?? 0) - groupStart,
            };
          }) ?? [];
      }

      return newClip;
    });
    console.log(clips);
  }  else {
    offsetStart = value.startFrame ?? 0;
    if (value.type === "video") {
      const asset = getAssetById(value.assetId);
      if (!asset) {
        return {
          exportClips: [],
          width: 0,
          height: 0,
          offsetStart: value.startFrame ?? 0,
        };
      }
      value.src = asset.path;
      value.audioSrc = hasAudio(value) ? asset.path : null;
    }
    clips = [value as AnyClipProps];
  }

  // --- Attach filters as applicators based on clip position score ---
  const filterClips = getClipsByType("filter");
  clips = sortClipsForStacking(
    clips.filter((c) => c.type !== "filter"),
    timelines,
  );

  filterClips.forEach((c) => {
    if (c.type === "filter") {
      (c as any).score = getClipPositionScore(c.clipId);
    }
  });

  const exportClips: ExportClip[] = [];

  for (const clipItem of clips as ExportClip[]) {
    const newClip: any = { ...(clipItem as any) };

    // Normalize preprocessor src paths (use cache path once complete)
    if (Array.isArray((clipItem as any).preprocessors)) {
      newClip.preprocessors = (clipItem as any).preprocessors.map((p: any) => {
        const asset = getAssetById(p.assetId);
        if (!asset) return p;
        p.src = asset.path;
        const convertedSrc =
          p?.status === "complete" && p?.src
            ? convertApexCachePath(p.src)
            : p?.src;
        return { ...p, src: convertedSrc };
      });
    }

    // Optionally clear masks for map targets
    if (clearMasks && Object.prototype.hasOwnProperty.call(newClip, "masks")) {
      newClip.masks = [];
    }

    // Attach filters that are visually "under" this clip
    for (const filter of [...filterClips]) {
      if (
        getClipPositionScore(filter.clipId) <
        getClipPositionScore(newClip.clipId)
      ) {
        if (!newClip.applicators) {
          newClip.applicators = [];
        }
        const newFilter: any = { ...(filter as any) };
        newFilter.startFrame = (newFilter.startFrame ?? 0) - offsetStart;
        newFilter.endFrame = (newFilter.endFrame ?? 0) - offsetStart;
        newClip.applicators.push(newFilter);
      }
    }

    newClip.applicators = _.uniqBy(newClip.applicators ?? [], "clipId");

    // Ensure a basic transform exists for renderable clips that may have been
    // added and exported before the preview had a chance to create one. This
    // mirrors the aspect-fit initialization used in `ImagePreview`/`VideoPreview`
    // and the centering behavior in `TextPreview`.
    if (!useOriginalTransform) {
      // Images & video: fit into the rect while preserving intrinsic aspect
      if (newClip.type === "image" || newClip.type === "video") {
        const currentTransform = (newClip as any).transform as any | undefined;
        const hasTransform = !!currentTransform;
        const tw = Number(currentTransform?.width || 0);
        const th = Number(currentTransform?.height || 0);
        const needsInit = !hasTransform || tw <= 0 || th <= 0;

        if (needsInit) {
          // When dimensions come from the aspect ratio, fit the media inside the
          // canvas rect while preserving its intrinsic aspect ratio, and center it.
          if (dimensionsFrom === "aspect" && width > 0 && height > 0) {
            const mediaInfo = getMediaInfoCached(newClip.src as string);
            const originalWidth =
              (mediaInfo as any)?.image?.width ??
              (mediaInfo as any)?.video?.displayWidth ??
              0;
            const originalHeight =
              (mediaInfo as any)?.image?.height ??
              (mediaInfo as any)?.video?.displayHeight ??
              0;

            if (originalWidth > 0 && originalHeight > 0) {
              const rectWidth = width;
              const rectHeight = height;
              const aspectRatio = originalWidth / originalHeight;
              let dw = rectWidth;
              let dh = rectHeight;
              if (rectWidth / rectHeight > aspectRatio) {
                dw = rectHeight * aspectRatio;
              } else {
                dh = rectWidth / aspectRatio;
              }
              const ox = (rectWidth - dw) / 2;
              const oy = (rectHeight - dh) / 2;

              (newClip as any).transform = {
                ...(currentTransform || {}),
                x: ox,
                y: oy,
                width: dw,
                height: dh,
                scaleX: 1,
                scaleY: 1,
                rotation: 0,
              };
            }
          } else {
            // Non-aspect mode: fall back to the media's own dimensions (or the
            // inferred width/height) with a simple origin-based placement.
            const mediaInfo = getMediaInfoCached(newClip.src as string);
            const intrinsicWidth =
              (mediaInfo as any)?.image?.width ??
              (mediaInfo as any)?.video?.displayWidth ??
              width ??
              0;
            const intrinsicHeight =
              (mediaInfo as any)?.image?.height ??
              (mediaInfo as any)?.video?.displayHeight ??
              height ??
              0;

            if (intrinsicWidth > 0 && intrinsicHeight > 0) {
              (newClip as any).transform = {
                ...(currentTransform || {}),
                x: currentTransform?.x ?? 0,
                y: currentTransform?.y ?? 0,
                width: intrinsicWidth,
                height: intrinsicHeight,
                scaleX: 1,
                scaleY: 1,
                rotation: 0,
              };
            }
          }
        }
      }

      // Text: center a default rect in the editor canvas if no transform exists,
      // matching the initialization used by `TextPreview`.
      if (newClip.type === "text") {
        const currentTransform = (newClip as any).transform as any | undefined;
        const tw = Number(currentTransform?.width || 0);
        const th = Number(currentTransform?.height || 0);
        const hasTransform = !!currentTransform && tw > 0 && th > 0;

        if (!hasTransform && width > 0 && height > 0) {
          const defaultWidth = 400;
          const defaultHeight = 100;
          const offsetX = (width - defaultWidth) / 2;
          const offsetY = (height - defaultHeight) / 2;

          (newClip as any).transform = {
            ...(currentTransform || {}),
            x: offsetX,
            y: offsetY,
            width: defaultWidth,
            height: defaultHeight,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
          };
        }
      }
    }

    // Optional centering logic (used for model generation previews)
    if (applyCentering) {
      try {
        const isRenderable =
          newClip.type === "image" || newClip.type === "video";
        const isGroupChild =
          newClip?.groupId !== undefined && newClip?.groupId !== null;
        const originalTransform = (newClip?.transform ?? {}) as any;
        const t = { ...originalTransform } as any;

        if (
          isRenderable &&
          !isGroupChild &&
          t &&
          typeof width === "number" &&
          typeof height === "number"
        ) {
          const rawW = Number(t.width) || 0 || width;
          const rawH = Number(t.height) || 0 || height;
          const sx = Number.isFinite(t.scaleX) ? Number(t.scaleX) : 1;
          const sy = Number.isFinite(t.scaleY) ? Number(t.scaleY) : 1;
          const w = Math.max(0, rawW * sx);
          const h = Math.max(0, rawH * sy);
          const deg = Number.isFinite(t.rotation) ? Number(t.rotation) : 0;
          const rad = (deg * Math.PI) / 180;
          const cCos = Math.cos(rad);
          const sSin = Math.sin(rad);
          const x1 = w * cCos;
          const y1 = w * sSin;
          const x2 = -h * sSin;
          const y2 = h * cCos;
          const x3 = w * cCos - h * sSin;
          const y3 = w * sSin + h * cCos;
          const minX = Math.min(0, x1, x2, x3);
          const maxX = Math.max(0, x1, x2, x3);
          const minY = Math.min(0, y1, y2, y3);
          const maxY = Math.max(0, y1, y2, y3);
          const aabbW = maxX - minX;
          const aabbH = maxY - minY;
          t.x = (width - aabbW) / 2 - minX;
          t.y = (height - aabbH) / 2 - minY;
          newClip.transform = t;
        }
      } catch {
        // best-effort centering; ignore failures
      }
    }


    // Attach a normalized (0–1) transform snapshot describing how this clip
    // occupies the output canvas. This leaves `transform` in pixel space for
    // existing exporters while providing a normalized view for new ones.
    if (
      width > 0 &&
      height > 0 &&
      (newClip as any).transform &&
      !useOriginalTransform
    ) {
      (newClip as any).normalizedTransform = normalizeTransformForCanvas(
        (newClip as any).transform,
        width,
        height,
      );
    } else if (
      width > 0 &&
      height > 0 &&
      (newClip as any).originalTransform &&
      useOriginalTransform
    ) {
      (newClip as any).normalizedTransform = normalizeTransformForCanvas(
        (newClip as any).originalTransform,
        width,
        height,
      );
    }

    if (useOriginalTransform) {
      newClip.transform = {
        ...newClip.originalTransform,
        crop: newClip.transform?.crop,
      };
    }

    // Preserve bottom-to-top stacking order for the renderer:

    exportClips.push(newClip);

    if (useMediaDimensionsForExport && exportClips.length === 1) {
      const mediaInfo = getMediaInfoCached(newClip.src as string);
      if (newClip.type === "video") {
        width = mediaInfo?.video?.displayWidth ?? 0;
        height = mediaInfo?.video?.displayHeight ?? 0;
      } else if (newClip.type === "image") {
        width = mediaInfo?.image?.width ?? 0;
        height = mediaInfo?.image?.height ?? 0;
      }
    }

    if (
      newClip.masks &&
      newClip.transform &&
      newClip.normalizedTransform &&
      useOriginalTransform &&
      target
    ) {
      // When exporting with an explicit target canvas size while using the
      // original (editor) transform, we need clip, mask and canvas to live in
      // the same coordinate space. Previously we only remapped the masks to a
      // resolved transform but left `newClip.transform` pointing at the
      // original editor-space transform. The mask pipeline (`applyMasksToCanvas`)
      // then received mismatched `clipTransform` (editor space) and
      // `maskTransform` (export space), which caused mask bounds to be scaled
      // incorrectly for single‑clip exports (notably via `exportClip`).
      // Fix: resolve a unified export‑space transform and:
      //   1) remap masks into that space
      //   2) update clip transforms to match
      //   3) attach a normalized snapshot relative to the *target* canvas.
      // ensure newClip.originalTransform x, y at 0,0
      // update the scale 
      newClip.originalTransform.scaleX = newClip.transform?.scaleX ?? 1;
      newClip.originalTransform.scaleY = newClip.transform?.scaleY ?? 1;
      
      const resolvedTransform = resolveTransformFromClip(
        newClip.originalTransform,
        newClip.normalizedTransform,
        target as Canvas,
      );

      // Remap all mask keyframes into the resolved/export transform space.
      if ((newClip as any).masks?.length > 0) {
        const masks = [...(newClip as any).masks] as MaskClipProps[];
        
        const mappedMasks = masks.map((mask: MaskClipProps) => {
          // Calculate native dimensions for intermediate transform
          // @ts-ignore
  
          // Native transform at origin
          const nativeTransform: ClipTransform = {
            x: 0,
            y: 0,
            width: target.width,
            height: target.height,
            scaleX: 1,
            scaleY: 1,
            rotation: newClip.originalTransform?.rotation ?? 0,
            opacity: 1,
            cornerRadius: 0,
          };

          const currentTransform =
            clipItem.transform ?? nativeTransform;
          let xOffset = 0;
          let yOffset = 0;
          if (clipItem.transform?.crop) {
            // determine how much to offsetX 
            const fullWidth = (clipItem.transform.width ?? 0) / clipItem.transform.crop.width;
            const fullHeight = (clipItem.transform.height ?? 0) / clipItem.transform.crop.height;
            const offsetX = fullWidth * clipItem.transform.crop.x;
            const offsetY = fullHeight * clipItem.transform.crop.y;
            xOffset = offsetX;
            yOffset = offsetY;
          }

          // Zeroed current transform (preserve scale/crop but move to origin)
          
          const zeroCurrentTransform: ClipTransform = {
            ...currentTransform as ClipTransform,
            x:xOffset,
            y: yOffset,
            crop: undefined
          };
  
          // Map: Current(0,0) -> Native(0,0)
          // We explicitly use the calculated zeroCurrentTransform as the source of truth
          // for the current mask coordinate space, ignoring any potentially stale transform
          // on the mask object itself.
          const maskForRemap = { ...mask, transform: undefined };

          const toOriginal = remapMaskForMediaDialog(
            maskForRemap,
            zeroCurrentTransform,
            nativeTransform,
          ); 

          return toOriginal;
        });

        newClip.masks = mappedMasks;

      }
      // Keep clip transforms in the same coordinate system as the masks so
      // WebGL masks (Shape/Lasso/Touch) receive consistent inputs.
      newClip.originalTransform = { ...(resolvedTransform as any) };
      newClip.transform = {
        ...(resolvedTransform as any),
        // Preserve any crop encoded on the (previous) transform.
        crop: newClip.transform?.crop,
      };

      // Recompute the normalized transform against the actual export canvas
      // dimensions so downstream exporters can safely map back into pixels.
      (newClip as any).normalizedTransform = normalizeTransformForCanvas(
        resolvedTransform,
        target.width,
        target.height,
      );
    }
  }

  return { exportClips, width, height, offsetStart };
}
