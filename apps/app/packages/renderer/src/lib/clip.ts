import { create } from "zustand";
import {
  AnyClipProps,
  TimelineProps,
  ClipTransform,
  TimelineType,
  VideoClipProps,
  PreprocessorClipProps,
  ImageClipProps,
  PreprocessorClipType,
  MaskClipProps,
  MaskData,
  GroupClipProps,
  ModelClipProps,
  ClipType,
  Asset,
} from "./types";
import { v4 as uuidv4 } from "uuid";
import { useControlsStore } from "./control";
import { MediaItem } from "@/components/media/Item";
import { AUDIO_EXTS, MIN_DURATION, VIDEO_EXTS, IMAGE_EXTS } from "./settings";
import { getMediaInfo, getMediaInfoCached, getMediaInfoFromUrl } from "./media/utils";
import { getLowercaseExtension } from "@app/preload";
import { Preprocessor } from "./preprocessor";
import { ManifestWithType, UIInput } from "./manifest/api";
import { globalInputControlsStore } from "./inputControl";
import { remapMaskWithClipTransformProportional } from "./mask/clipTransformUtils";

import _ from "lodash";
export const PREPROCESSOR_BAR_HEIGHT = 24;

interface ClipStore {
  // Clips
  clipDuration: number;
  _setClipDuration: (duration: number) => void;
  clips: AnyClipProps[];
  convertToMedia: (clipId: string) => void;
  getAssetById: (assetId: string) => Asset | undefined;
  getAssetByPath: (path: string) => Asset | undefined;
  assets: Record<string, Asset>;  
  setAssets: (assets: Record<string, Asset>) => void;
  addAssetAsync: (asset: Partial<Asset> | string, sourceDir?: "user-data" | "apex-cache") => Promise<Asset>;
  addAsset: (asset: Partial<Asset> | string, sourceDir?: "user-data" | "apex-cache" | "backend") => Asset;
  removeAsset: (assetId: string) => void;
  updateAsset: (assetId: string, assetToUpdate: Partial<Asset>) => void;
  getClipById: (
    clipId: string,
    timelineId?: string,
  ) => AnyClipProps | undefined;
  getClipsByType: (type: ClipType) => AnyClipProps[];
  getClipTransform: (clipId: string) => ClipTransform | undefined;
  getUncroppedClipTransform: (clipId: string) => ClipTransform | undefined;
  setClipTransform: (
    clipId: string,
    transform: Partial<ClipTransform>,
    applyToMasks?: boolean,
    remapMasks?: boolean,
  ) => void;
  setClips: (clips: AnyClipProps[]) => void;
  addClip: (clip: AnyClipProps) => void;
  removeClip: (clipId: string) => void;
  updateClip: (clipId: string, clipToUpdate: Partial<AnyClipProps>) => void;
  updatePreprocessor: (
    clipId: string,
    preprocessorId: string,
    preprocessorToUpdate: Partial<PreprocessorClipProps>,
  ) => void;
  resizeClip: (
    clipId: string,
    side: "left" | "right",
    newFrame: number,
  ) => void;
  separateClip: (clipId: string) => void;
  isValidResize: (
    clipId: string,
    side: "left" | "right",
    newFrame: number,
  ) => boolean;
  splitClip: (cutFrame: number, clipId: string) => void;
  mergeClips: (clipIds: string[]) => void;
  moveClipToEnd: (clipId: string) => void;
  clipboard: AnyClipProps[];
  copyClips: (clipIds: string[]) => void;
  cutClips: (clipIds: string[]) => void;
  pasteClips: (atFrame?: number, targetTimelineId?: string) => void;
  groupClips: (clipIds: string[]) => void;
  ungroupClips: (groupId: string) => void;
  getClipsForGroup: (children: string[][]) => AnyClipProps[];
  getClipAtFrame: (frame: number) => [AnyClipProps, number] | null;
  activeMediaItem: MediaItem | null;
  setActiveMediaItem: (mediaItem: MediaItem | null) => void;
  ghostStartEndFrame: [number, number];
  setGhostStartEndFrame: (startFrame: number, endFrame: number) => void;
  ghostX: number;
  setGhostX: (x: number) => void;
  ghostGuideLines: [number, number] | null;
  setGhostGuideLines: (guideLines: [number, number] | null) => void;
  muteTimeline: (timelineId: string) => void;
  unmuteTimeline: (timelineId: string) => void;
  getModelValues: (clipId: string) => Record<string, any> | null;
  getRawModelValues: (clipId: string) => Record<string, any> | null;
  hideTimeline: (timelineId: string) => void;
  unhideTimeline: (timelineId: string) => void;
  isTimelineMuted: (timelineId: string) => boolean;
  isTimelineHidden: (timelineId: string) => boolean;
  ghostInStage: boolean;
  setGhostInStage: (inStage: boolean) => void;
  hoveredTimelineId: string | null;
  setHoveredTimelineId: (timelineId: string | null) => void;
  ghostTimelineId: string | null;
  setGhostTimelineId: (timelineId: string | null) => void;
  draggingClipId: string | null;
  setDraggingClipId: (clipId: string | null) => void;
  isDragging: boolean;
  setIsDragging: (isDragging: boolean) => void;
  selectedPreprocessorId: string | null;
  setSelectedPreprocessorId: (preprocessorId: string | null) => void;
  getPreprocessorById: (preprocessorId: string) => PreprocessorClipProps | null;
  addPreprocessorToClip: (
    clipId: string,
    preprocessor: PreprocessorClipProps,
  ) => void;
  removePreprocessorFromClip: (
    clipId: string,
    preprocessor: PreprocessorClipProps | string,
  ) => void;
  getPreprocessorsForClip: (clipId: string) => PreprocessorClipProps[];
  getClipFromPreprocessorId: (
    preprocessorId: string,
  ) => PreprocessorClipType | null;
  // Global snap guideline (absolute stage X in px). Null when inactive
  snapGuideX: number | null;
  setSnapGuideX: (x: number | null) => void;
  _updateZoomLevel: (clips: AnyClipProps[], clipDuration: number) => void;
  getTimelinePosition: (
    timelineId: string,
    scrollY?: number,
  ) => { top: number; bottom: number; left: number; right: number };
  getClipPosition: (
    clipId: string,
    scrollY?: number,
  ) => { top: number; bottom: number; left: number; right: number };
  // Computes a numeric score for a clip combining timeline order, group layering, and endFrame.
  // Higher score means visually "above" in stacking: lower timelines outrank above ones; within a timeline,
  // later layers (grouped upwards) outrank earlier ones; endFrame acts as final tiebreaker.
  getClipPositionScore: (clipId: string) => number;
  rescaleForFpsChange: (oldFps: number, newFps: number) => void;
  // Timelines
  timelines: TimelineProps[];
  getClipsForTimeline: (timelineId: string) => AnyClipProps[];
  getTimelineById: (timelineId: string) => TimelineProps | undefined;
  setTimelines: (timelines: TimelineProps[]) => void;
  addTimeline: (timeline: Partial<TimelineProps>, index?: number) => void;
  removeTimeline: (timelineId: string) => void;
  updateTimeline: (
    timelineId: string,
    timelineToUpdate: Partial<TimelineProps>,
  ) => void;
  clipWithinFrame: (
    clip: AnyClipProps,
    frame: number,
    overlap?: boolean,
    overlapAmount?: number,
  ) => boolean;
  updateMaskKeyframes: (
    clipId: string,
    maskId: string,
    updater: (
      keyframes: Map<number, MaskData> | Record<number, MaskData>,
    ) => Map<number, MaskData> | Record<number, MaskData>,
  ) => void;
  updateModelInput: (
    clipId: string,
    inputId: string,
    inputToUpdate: Partial<UIInput>,
  ) => void;
}

// Helper function to calculate total duration of all clips
const calculateTotalClipDuration = (clips: AnyClipProps[]): number => {
  const maxEndFrame = Math.max(...clips.map((clip) => clip.endFrame || 0));
  return maxEndFrame;
};

const rescaleFrame = (frame: any, oldFps: number, newFps: number): number | undefined => {
  if (frame == null) return undefined;
  const f = Number(frame);
  if (!Number.isFinite(f) || f < 0) return undefined;
  const safeOld = Math.max(1, Number(oldFps) || 1);
  const safeNew = Math.max(1, Number(newFps) || 1);
  if (safeOld === safeNew) return f;
  const seconds = f / safeOld;
  return Math.max(0, Math.round(seconds * safeNew));
};

// Helper to compute which assetIds are actually in use by current clips
const collectUsedAssetIds = (clips: AnyClipProps[]): Set<string> => {
  const used = new Set<string>();

  for (const clip of clips) {
    // Direct clip asset references
    if (
      (clip.type === "video" ||
        clip.type === "image" ||
        clip.type === "audio" ||
        clip.type === "model") &&
      (clip as any).assetId
    ) {
      used.add((clip as any).assetId as string);
    }

    if (clip.type === "image" || clip.type === "video") {
      if (clip.preprocessors && Array.isArray(clip.preprocessors)) {
        for (const p of clip.preprocessors as any[]) {
          const id = p?.assetId;
          if (typeof id === "string" && id) used.add(id);
        }
      }
    }

    // Asset history on media/model clips
    if (
      (clip.type === "video" || clip.type === "image") &&
      Array.isArray((clip as any).assetIdHistory)
    ) {
      for (const id of (clip as any).assetIdHistory as string[]) {
        if (typeof id === "string" && id) used.add(id);
      }
    }
    if (
      clip.type === "model" &&
      Array.isArray((clip as any).assetIdHistory)
    ) {
      for (const id of (clip as any).assetIdHistory as string[]) {
        if (typeof id === "string" && id) used.add(id);
      }
    }

    // Model generations can reference assets as well
    if (clip.type === "model" && Array.isArray((clip as any).generations)) {
      for (const gen of (clip as any).generations as any[]) {
        const id = gen?.assetId;
        if (typeof id === "string" && id) used.add(id);
      }
    }

    // Preprocessor assets nested under video/image clips
    if (
      (clip.type === "video" || clip.type === "image") &&
      Array.isArray((clip as any).preprocessors)
    ) {
      for (const p of (clip as any).preprocessors as any[]) {
        const id = p?.assetId;
        if (typeof id === "string" && id) used.add(id);
      }
    }
  }

  return used;
};

// Given current clips, drop any assets that are no longer referenced by clips
const pruneAssetsForClips = (
  assets: Record<string, Asset>,
  clips: AnyClipProps[],
): Record<string, Asset> => {
  if (!assets) return {};
  const assetValues = Object.values(assets);
  if (assetValues.length === 0) return assets;

  const numModelClips = clips.filter((clip) => clip.type === "model").length;

  const used = collectUsedAssetIds(clips);
  const pruned: Record<string, Asset> = {};
  for (const asset of assetValues) {
    if (!asset || !asset.id) continue;
    if (used.has(asset.id)) {
      // Always key by asset id to keep lookups consistent with getAssetById
      pruned[asset.id] = asset;
    }
  }
  // ensure assets with persist are still in the pruned assets
  for (const asset of assetValues) {
    if (!asset || !asset.id) continue;
    if (asset.modelInputAsset && numModelClips > 0) {
      pruned[asset.id] = asset;
    }
  }
  return pruned;
};

export const isValidTimelineForClip = (
  timeline: TimelineProps,
  clip: AnyClipProps | MediaItem | string | Preprocessor | ManifestWithType,
) => {
  if (typeof clip === "string") clip = { type: clip } as AnyClipProps;
  if (timeline.type === "media") {
    return (
      clip.type === "video" ||
      clip.type === "image" ||
      (clip as AnyClipProps).type === "group" ||
      (clip as AnyClipProps).type === "model"
    );
  }
  return timeline.type === clip.type;
};

export const getTimelineTypeForClip = (
  clip: AnyClipProps | MediaItem | string | ManifestWithType,
): TimelineType => {
  if (typeof clip === "string") clip = { type: clip } as AnyClipProps;
  if (
    clip.type === "video" ||
    clip.type === "image" ||
    clip.type === "group" ||
    clip.type === "model"
  ) {
    return "media";
  }
  return clip.type;
};

export const getTimelineHeightForClip = (
  clip: AnyClipProps | MediaItem | string | ManifestWithType,
): number => {
  if (typeof clip === "string") clip = { type: clip } as AnyClipProps;

  if (
    clip.type === "video" ||
    clip.type === "image" ||
    (clip.type as any) === "media" ||
    clip.type === "group" ||
    clip.type === "model"
  ) {
    return 72;
  }
  if (clip.type === "audio") {
    return 54;
  }
  return 36;
};

// Helper function to resolve overlaps by shifting clips to maintain frame gaps
export const resolveOverlaps = (clips: AnyClipProps[]): AnyClipProps[] => {
  if (clips.length === 0) return clips;

  // Sort clips by start frame
  // filter clips that are hidden
  const hiddenClips = clips.filter((clip) => clip.hidden);
  const visibleClips = clips.filter((clip) => !clip.hidden);
  const sortedClips = [...visibleClips].sort(
    (a, b) => (a.startFrame || 0) - (b.startFrame || 0),
  );
  const resolvedClips: AnyClipProps[] = [];

  for (let i = 0; i < sortedClips.length; i++) {
    const currentClip = { ...sortedClips[i] };
    const currentStart = currentClip.startFrame || 0;
    const currentEnd = currentClip.endFrame || 0;
    const currentTimelineId = currentClip.timelineId || "";

    // Check for overlap with previous clip
    if (resolvedClips.length > 0) {
      const previousClip = resolvedClips[resolvedClips.length - 1];
      const previousEnd = previousClip.endFrame || 0;

      // If current clip overlaps with previous clip, shift it to start after previous clip ends
      if (
        currentStart < previousEnd &&
        currentTimelineId === previousClip.timelineId
      ) {
        const clipDuration = currentEnd - currentStart;
        currentClip.startFrame = previousEnd;
        currentClip.endFrame = previousEnd + clipDuration;
      }
    }

    resolvedClips.push(currentClip);
  }
  // add hidden clips to the resolved clips
  hiddenClips.forEach((clip) => {
    resolvedClips.push(clip);
  });

  return resolvedClips;
};

export const getCorrectedClip = (
  clipId: string,
  clips: AnyClipProps[],
): AnyClipProps | null => {
  // find the clip in the clips array
  const resolvedClips = resolveOverlaps(clips);
  const clip = resolvedClips.find((clip) => clip.clipId === clipId);
  if (!clip) return null;
  return clip;
};

export const resolveOverlapsTimelines = (
  timelines: TimelineProps[],
): TimelineProps[] => {
  if (timelines.length === 0) return timelines;

  const resolvedTimelines: TimelineProps[] = [];

  for (let i = 0; i < timelines.length; i++) {
    const timeline = { ...timelines[i] };

    // First timeline starts at 0
    if (i === 0) {
      timeline.timelineY = 0;
    } else {
      // Each subsequent timeline is positioned based on the previous timeline's position + height
      const previousTimeline = resolvedTimelines[i - 1];
      const previousY = previousTimeline.timelineY || 0;
      const previousHeight = previousTimeline.timelineHeight || 54;
      timeline.timelineY = previousY + previousHeight;
    }

    resolvedTimelines.push(timeline);
  }

  return resolvedTimelines;
};

export const useClipStore = create<ClipStore>(((set, get) => ({
  clipDuration: 0,
  _setClipDuration: (duration) => set({ clipDuration: duration }),
  clips: [],
  assets: {},
  timelines: [],
  setAssets: (assets) => set({ assets }),
  addAssetAsync: async (asset, sourceDir: "user-data" | "apex-cache" = "user-data"): Promise<Asset> => {
    if (typeof asset === "string") {
      asset = { path: asset };
    }

    const isModelInputAsset = !!(asset as any).modelInputAsset;

    const path =
      typeof asset === "object" && "path" in asset
        ? (asset.path as string)
        : (asset as string);

    // Reuse existing asset for this path to avoid duplicates,
    // but NEVER reuse for modelInputAsset so they always get a unique assetId.
    const existingAssets = get().assets;
    if (!isModelInputAsset) {
      const existing = Object.values(existingAssets).find(
        (a) => a.path === path && !a.modelInputAsset,
      );
      if (existing) return existing;
    }


    const mediaInfo = getMediaInfoCached(path as string);

    // Initial, best-effort values; will be refined asynchronously if needed
    const inferredTypeFromMedia =
      mediaInfo?.video
        ? "video"
        : mediaInfo?.audio
          ? "audio"
          : mediaInfo
            ? "image"
            : undefined;

    let type =
      asset.type ??
      inferredTypeFromMedia;

    if (!type) {
      const ext = getLowercaseExtension(path ?? "");
      if (VIDEO_EXTS.includes(ext)) type = "video";
      else if (AUDIO_EXTS.includes(ext)) type = "audio";
      else if (IMAGE_EXTS.includes(ext)) type = "image";
      else type = "image";
    }

    const height =
      asset.height ??
      mediaInfo?.video?.displayHeight ??
      mediaInfo?.image?.height;
    const width =
      asset.width ??
      mediaInfo?.video?.displayWidth ??
      mediaInfo?.image?.width;
    const duration = asset.duration ?? mediaInfo?.duration ?? 0;

    const newAsset: Asset = {
      id: uuidv4(),
      path: path,
      type: type,
      duration: duration,
      width: height === undefined ? undefined : width,
      height: height === undefined ? undefined : height,
      modelInputAsset: asset.modelInputAsset ?? false,
    };

    const newAssets: Record<string, Asset> = {
      ...get().assets,
      [newAsset.id]: newAsset,
    };

    // If media info wasn't cached, fetch it asynchronously and update this asset
    if (!mediaInfo) {
      const info = await getMediaInfo(path as string, { sourceDir });
        
        if (!info) return newAsset;

          const updatedHeight =
            info.video?.displayHeight ?? info.image?.height;
          const updatedWidth =
            info.video?.displayWidth ?? info.image?.width;
          const updatedDuration = info.duration ?? 0;

          let updatedType: Asset["type"] = newAsset.type;
          if (info.video) updatedType = "video";
          else if (info.audio) updatedType = "audio";
          else if (info.image) updatedType = "image";

          newAsset.height = updatedHeight;
          newAsset.width = updatedWidth;
          newAsset.duration = updatedDuration;
          newAsset.type = updatedType;
    }
    set({ assets: newAssets });
    return newAsset;
  },
  addAsset: (asset, sourceDir: "user-data" | "apex-cache" | "backend" = "user-data"): Asset => {
    if (typeof asset === "string") {
      asset = { path: asset };
    }

    const isModelInputAsset = !!(asset as any).modelInputAsset;

    const path =
      typeof asset === "object" && "path" in asset
        ? (asset.path as string)
        : (asset as string);

    // Reuse existing asset for this path to avoid duplicates,
    // but NEVER reuse for modelInputAsset so they always get a unique assetId.
    const existingAssets = get().assets;
    if (!isModelInputAsset) {
      const existing = Object.values(existingAssets).find(
        (a) => a.path === path && !a.modelInputAsset,
      );
      if (existing) return existing;
    }


    const mediaInfo = getMediaInfoCached(path as string);

    // Initial, best-effort values; will be refined asynchronously if needed
    const inferredTypeFromMedia =
      mediaInfo?.video
        ? "video"
        : mediaInfo?.audio
          ? "audio"
          : mediaInfo
            ? "image"
            : undefined;

    let type =
      asset.type ??
      inferredTypeFromMedia;

    if (!type) {
      const ext = getLowercaseExtension(path ?? "");
      if (VIDEO_EXTS.includes(ext)) type = "video";
      else if (AUDIO_EXTS.includes(ext)) type = "audio";
      else if (IMAGE_EXTS.includes(ext)) type = "image";
      else type = "image";
    }

    const height =
      asset.height ??
      mediaInfo?.video?.displayHeight ??
      mediaInfo?.image?.height;
    const width =
      asset.width ??
      mediaInfo?.video?.displayWidth ??
      mediaInfo?.image?.width;
    const duration = asset.duration ?? mediaInfo?.duration ?? 0;

    const newAsset: Asset = {
      id: uuidv4(),
      path: path,
      type: type,
      duration: duration,
      width: height === undefined ? undefined : width,
      height: height === undefined ? undefined : height,
      modelInputAsset: asset.modelInputAsset ?? false,
    };

    const newAssets: Record<string, Asset> = {
      ...get().assets,
      [newAsset.id]: newAsset,
    };

    set({ assets: newAssets });
    // If media info wasn't cached, fetch it asynchronously and update this asset
    if (!mediaInfo ) {
      if (sourceDir === "backend") {
        void getMediaInfoFromUrl(path as string).then((info) => {
          if (!info) return;
          get().updateAsset(newAsset.id, {
            height: newAsset.height ?? info.video?.displayHeight ?? info.image?.height,
            width: newAsset.width ?? info.video?.displayWidth ?? info.image?.width,
            duration: newAsset.duration ?? info.duration ?? 0,
            type: newAsset.type ?? (info.video ? "video" : info.audio ? "audio" : info.image ? "image" : "image"),
          });
        });
      }
      else {
      void getMediaInfo(path as string, { sourceDir })
        .then((info) => {
          if (!info) return;

          const updatedHeight =
            info.video?.displayHeight ?? info.image?.height;
          const updatedWidth =
            info.video?.displayWidth ?? info.image?.width;
          const updatedDuration = info.duration ?? 0;

          let updatedType: Asset["type"] = newAsset.type;
          if (info.video) updatedType = "video";
          else if (info.audio) updatedType = "audio";
          else if (info.image) updatedType = "image";

          get().updateAsset(newAsset.id, {
            height: newAsset.height ?? updatedHeight,
            width: newAsset.width ?? updatedWidth,
            duration:
              newAsset.duration && newAsset.duration > 0
                ? newAsset.duration
                : updatedDuration,
            type: newAsset.type ?? updatedType,
          });
        })
        .catch((err) => {
          console.error("Failed to load media info for asset", path, err);
        });
      }
    }

    return newAsset;
  },
  removeAsset: (assetId) => set((state) => ({ assets: _.omit(state.assets, assetId) })),
  updateAsset: (assetId, assetToUpdate) => set((state) => ({ assets: _.set(state.assets, assetId, { ...state.assets[assetId], ...assetToUpdate }) })),
  getAssetById: (assetId) => get().assets[assetId],
  getAssetByPath: (path) => Object.values(get().assets).find((a) => a.path === path),
  convertToMedia: (clipId: string) =>
    set((state) => {
      const idx = state.clips.findIndex((c) => c.clipId === clipId);
      if (idx === -1) return { clips: state.clips };
      const clip = state.clips[idx];
      if (!clip || clip.type !== "model") return { clips: state.clips };
      const modelClip = clip as ModelClipProps;

      // Choose source: prefer selected src; fallback to active job; else last completed/running generation
      let chosenAssetId: string | null = (modelClip.assetId as any) || null;
      if (
        !chosenAssetId &&
        Array.isArray(modelClip.generations) &&
        modelClip.generations.length > 0
      ) {
        const byActive = modelClip.activeJobId
          ? modelClip.generations.find(
              (g) => g.jobId === modelClip.activeJobId && g.assetId,
            )
          : undefined;
        const byStatus = [...modelClip.generations]
          .reverse()
          .find(
            (g) =>
              (g.modelStatus === "complete" || g.modelStatus === "running") &&
              g.assetId,
          );
        chosenAssetId = (byActive?.assetId || byStatus?.assetId || null) as any;
      }
      if (!chosenAssetId) return { clips: state.clips };

      // Infer media type from extension; if unknown, we'll asynchronously refine via getMediaInfo
      const asset = get().getAssetById(chosenAssetId);
      const ext = getLowercaseExtension(asset?.path ?? "");
      const isVid = VIDEO_EXTS.includes(ext);
      const isImg = IMAGE_EXTS.includes(ext);

      const common = {
        clipId: modelClip.clipId,
        timelineId: modelClip.timelineId,
        startFrame: modelClip.startFrame,
        endFrame: modelClip.endFrame,
        trimStart: modelClip.trimStart,
        trimEnd: modelClip.trimEnd,
        clipPadding: modelClip.clipPadding,
        transform: modelClip.transform,
        originalTransform: modelClip.originalTransform,
        groupId: modelClip.groupId,
        hidden: modelClip.hidden,
        mediaWidth: modelClip.mediaWidth,
        mediaHeight: modelClip.mediaHeight,
        mediaAspectRatio: modelClip.mediaAspectRatio,
      } as Partial<AnyClipProps>;

      let converted: AnyClipProps | null = null;
      if (isVid || (!isVid && !isImg)) {
        converted = {
          ...(common as any),
          type: "video",
          assetId: chosenAssetId,
          volume: 1,
          fadeIn: 0,
          fadeOut: 0,
          speed:
            typeof modelClip.speed === "number" &&
            Number.isFinite(modelClip.speed)
              ? (modelClip.speed as any)
              : 1,
          preprocessors: [],
          masks: [],
        } as VideoClipProps as AnyClipProps;
      } else if (isImg) {
        converted = {
          ...(common as any),
          type: "image",
          assetId: chosenAssetId,
          preprocessors: [],
          masks: [],
        } as ImageClipProps as AnyClipProps;
      }
      if (!converted) return { clips: state.clips };

      const newClips = [...state.clips];
      newClips[idx] = converted;
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);

      get()._updateZoomLevel(resolvedClips, clipDuration);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  isTimelineMuted: (timelineId) =>
    get().timelines.find((timeline) => timeline.timelineId === timelineId)
      ?.muted || false,
  isTimelineHidden: (timelineId) =>
    get().timelines.find((timeline) => timeline.timelineId === timelineId)
      ?.hidden || false,
  muteTimeline: (timelineId) =>
    set((state) => {
      const newTimelines = state.timelines.map((timeline) =>
        timeline.timelineId === timelineId
          ? { ...timeline, muted: true }
          : timeline,
      );
      const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
      return { timelines: resolvedTimelines };
    }),
  unmuteTimeline: (timelineId) =>
    set((state) => {
      const newTimelines = state.timelines.map((timeline) =>
        timeline.timelineId === timelineId
          ? { ...timeline, muted: false }
          : timeline,
      );
      const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
      return { timelines: resolvedTimelines };
      return state;
    }),
  hideTimeline: (timelineId) =>
    set((state) => {
      const newTimelines = state.timelines.map((timeline) =>
        timeline.timelineId === timelineId
          ? { ...timeline, hidden: true }
          : timeline,
      );
      const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
      return { timelines: resolvedTimelines };
      return state;
    }),
  unhideTimeline: (timelineId) =>
    set((state) => {
      const newTimelines = state.timelines.map((timeline) =>
        timeline.timelineId === timelineId
          ? { ...timeline, hidden: false }
          : timeline,
      );
      const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
      return { timelines: resolvedTimelines };
    }),
  ghostStartEndFrame: [0, 0],
  activeMediaItem: null,
  setActiveMediaItem: (mediaItem) => set({ activeMediaItem: mediaItem }),
  setGhostStartEndFrame: (startFrame, endFrame) =>
    set({ ghostStartEndFrame: [startFrame, endFrame] }),
  ghostX: 0,
  _updateZoomLevel: (clips: AnyClipProps[], clipDuration: number) => {
    // Determine if this is the first clip being added (based on current store state before set)
    const hadNoClips = (get().clips || []).length === 0;

    // Longest clip length in frames (endFrame max)
    const longestClipFrames = Math.max(0, Math.round(clipDuration || 0));

    // Compute new baseline total timeline frames at zoom level 1 so that
    // the longest clip occupies 60% of the timeline (5/3 multiplier).
    // Clamp to at least MIN_DURATION to ensure we can always zoom to 5 frames.
    const newTotalFrames = Math.max(
      MIN_DURATION,
      Math.round((longestClipFrames * 5) / 3),
    );

    const controls = useControlsStore.getState();
    const prevTotalFrames = Math.max(0, controls.totalTimelineFrames || 0);
    const minZoomLevel = controls.minZoomLevel;
    const maxZoomLevel = controls.maxZoomLevel;
    const [currentStart, currentEnd] = controls.timelineDuration || [
      0,
      newTotalFrames,
    ];
    const currentWidth = Math.max(1, currentEnd - currentStart);

    const needsBaselineUpdate =
      newTotalFrames !== prevTotalFrames && clips.length > 0;

    if (!needsBaselineUpdate) {
      return;
    }

    // Apply the new baseline total frames
    useControlsStore.setState({ totalTimelineFrames: newTotalFrames });

    if (hadNoClips) {
      // First clip: set zoom level 1 window to exactly the baseline width,
      // so the clip takes 60% of the timeline at zoom level 1.
      controls.setTimelineDuration(0, newTotalFrames);
      controls.setZoomLevel(1 as any);
      return;
    }

    // For subsequent changes (e.g., adding a longer clip), keep the current
    // window to avoid jitter, but recalibrate the zoom level so that the
    // current window width maps to the nearest zoom step under the new baseline.
    const steps = Math.max(1, maxZoomLevel - minZoomLevel);
    const ratio = MIN_DURATION / newTotalFrames;
    const durations: number[] = new Array(steps + 1).fill(0).map((_, i) => {
      const ti = i / steps;
      const d = Math.round(newTotalFrames * Math.pow(ratio, ti));
      return Math.max(MIN_DURATION, Math.min(newTotalFrames, d));
    });

    // Keep the same window; clamp within new total if needed
    let clampedStart = currentStart;
    let clampedEnd = currentEnd;
    const width = Math.max(1, clampedEnd - clampedStart);
    if (clampedEnd > newTotalFrames) {
      clampedStart = Math.max(0, newTotalFrames - width);
      clampedEnd = clampedStart + width;
      controls.setTimelineDuration(clampedStart, clampedEnd);
    }

    // Pick the zoom level whose target duration is closest to the current width
    let bestIndex = 0;
    let bestDiff = Number.POSITIVE_INFINITY;
    for (let i = 0; i <= steps; i++) {
      const diff = Math.abs(durations[i] - currentWidth);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestIndex = i;
      }
    }
    const newZoomLevel = (minZoomLevel + bestIndex) as any;
    controls.setZoomLevel(newZoomLevel);
  },
  setGhostX: (x) => set({ ghostX: x }),
  selectedPreprocessorId: null,
  setSelectedPreprocessorId: (preprocessorId) => {
    // control store set clip id to empty
    if (preprocessorId) {
      useControlsStore.setState({ selectedClipIds: [] });
    }
    set({ selectedPreprocessorId: preprocessorId });
  },
  ghostGuideLines: null,
  setGhostGuideLines: (guideLines) => set({ ghostGuideLines: guideLines }),
  hoveredTimelineId: null,
  setHoveredTimelineId: (timelineId) => set({ hoveredTimelineId: timelineId }),
  ghostTimelineId: null,
  setGhostTimelineId: (timelineId) => set({ ghostTimelineId: timelineId }),
  ghostInStage: false,
  setGhostInStage: (inStage) => set({ ghostInStage: inStage }),
  draggingClipId: null,
  setDraggingClipId: (clipId) => set({ draggingClipId: clipId }),
  isDragging: false,
  setIsDragging: (isDragging) => set({ isDragging }),
  snapGuideX: null,
  setSnapGuideX: (x) => set({ snapGuideX: x }),
  getClipTransform: (clipId: string) => {
    const clip = get().clips.find((c) => c.clipId === clipId);
    return clip?.transform;
  },
  getUncroppedClipTransform: (clipId: string) => {
    const clip = get().clips.find((c) => c.clipId === clipId);
    if (!clip) return undefined;
    const crop = clip.transform?.crop;
    if (!crop) return clip.transform;
    return {
      ...clip.transform,
      width: (clip.transform?.width ?? 0) / crop.width,
      height: (clip.transform?.height ?? 0) / crop.height,
      crop: undefined,
      x: clip.transform?.x ?? 0,
      y: clip.transform?.y ?? 0,
      scaleX: clip.transform?.scaleX ?? 1,
      scaleY: clip.transform?.scaleY ?? 1,
      rotation: clip.transform?.rotation ?? 0,
      cornerRadius: clip.transform?.cornerRadius ?? 0,
      opacity: clip.transform?.opacity ?? 100,
    };
  },
  setClipTransform: (clipId: string, transform: Partial<ClipTransform>, applyToMasks: boolean = false, remapMasks: boolean = false) =>
    set((state) => {
      const index = state.clips.findIndex((c) => c.clipId === clipId);
      if (index === -1) return { clips: state.clips };

      const current = state.clips[index];
      const previous: ClipTransform = current.transform || {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        scaleX: 1,
        scaleY: 1,
        rotation: 0,
        cornerRadius: 0,
        opacity: 100,
        crop: { x: 0, y: 0, width: 1, height: 1 },
      };
      const next: ClipTransform = { ...previous, ...transform };

      const updatedClip: AnyClipProps = { ...current, transform: next };

      // give the transform to all masks if applyToMasks is true
      if (applyToMasks && Object.keys(current).includes("masks")) {
        let  masks = [...(current as VideoClipProps | ImageClipProps).masks];
        masks = masks.map((mask: MaskClipProps) => {
          if (remapMasks) {
            mask = remapMaskWithClipTransformProportional(mask, mask.transform as ClipTransform, next);
          } else {
            mask.transform = { ...next };
          }
          return mask;
        });
        (updatedClip as VideoClipProps | ImageClipProps).masks = masks;
      }

      if (!current.originalTransform && next.width > 0 && next.height > 0) {
        updatedClip.originalTransform = { ...next, x: 0, y: 0 };
      }
      const newClips = [...state.clips];
      newClips[index] = updatedClip;
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  getPreprocessorById: (preprocessorId: string) => {
    // get the clip that contains the preprocessor
    const clip = get().clips.find(
      (c) =>
        (c.type === "video" || c.type === "image") &&
        c.preprocessors?.some((p) => p.id === preprocessorId),
    );
    if (!clip) return null;
    const preprocessor = (
      clip as VideoClipProps | ImageClipProps
    ).preprocessors?.find((p) => p.id === preprocessorId);
    return preprocessor || null;
  },
  getClipFromPreprocessorId: (preprocessorId: string) => {
    const clip = get().clips.find(
      (c) =>
        (c.type === "video" || c.type === "image") &&
        c.preprocessors?.some((p) => p.id === preprocessorId),
    ) as PreprocessorClipType | null;
    return clip || null;
  },
  updatePreprocessor: (
    clipId: string,
    preprocessorId: string,
    preprocessorToUpdate: Partial<PreprocessorClipProps>,
  ) =>
    set((state) => {
      const newClips = [...state.clips];
      const currentClipIndex = newClips.findIndex((c) => c.clipId === clipId);

      if (currentClipIndex === -1) return { clips: state.clips };
      const currentClip = newClips[currentClipIndex];
      if (!currentClip) return { clips: state.clips };
      if (currentClip.type !== "video" && currentClip.type !== "image")
        return { clips: state.clips };
      const newPreprocessors =
        currentClip.preprocessors?.map((p) =>
          p.id === preprocessorId ? { ...p, ...preprocessorToUpdate } : p,
        ) || [];
      const newClip = {
        ...currentClip,
        preprocessors: newPreprocessors,
      } as AnyClipProps;
      newClips[currentClipIndex] = newClip;
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  addPreprocessorToClip: (
    clipId: string,
    preprocessor: PreprocessorClipProps,
  ) =>
    set((state) => {
      const currentClipIndex = state.clips.findIndex(
        (c) => c.clipId === clipId,
      );
      const currentClip = state.clips[currentClipIndex];
      if (!currentClip) return { clips: state.clips };
      const newClips = [...state.clips];
      if (currentClip.type !== "video" && currentClip.type !== "image")
        return { clips: state.clips };
      const newPreprocessors = [
        ...(currentClip.preprocessors || []),
        preprocessor,
      ];
      const newClip = {
        ...currentClip,
        preprocessors: newPreprocessors,
      } as AnyClipProps;
      newClips[currentClipIndex] = newClip;
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  removePreprocessorFromClip: (
    clipId: string,
    preprocessor: PreprocessorClipProps | string,
  ) =>
    set((state) => {
      const newClips = [...state.clips];
      const currentClipIndex = newClips.findIndex((c) => c.clipId === clipId);
      if (currentClipIndex === -1) return { clips: state.clips };
      const currentClip = newClips[currentClipIndex];
      if (!currentClip) return { clips: state.clips };
      if (currentClip.type !== "video" && currentClip.type !== "image")
        return { clips: state.clips };
      const newPreprocessors =
        currentClip.preprocessors?.filter(
          (p) =>
            p.id !==
            (typeof preprocessor === "string" ? preprocessor : preprocessor.id),
        ) || [];
      const newClip = {
        ...currentClip,
        preprocessors: newPreprocessors,
      } as AnyClipProps;
      newClips[currentClipIndex] = newClip;
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      let selectedPreprocessorId = state.selectedPreprocessorId;
      if (
        selectedPreprocessorId ===
        (typeof preprocessor === "string" ? preprocessor : preprocessor.id)
      ) {
        selectedPreprocessorId = null;
      }
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return {
        clips: resolvedClips,
        clipDuration,
        selectedPreprocessorId,
        assets: prunedAssets,
      };
    }),
  getPreprocessorsForClip: (clipId: string) => {
    const clip = get().clips.find((c) => c.clipId === clipId);
    if (!clip || (clip.type !== "video" && clip.type !== "image")) return [];
    return clip.preprocessors || [];
  },
  getTimelineById: (timelineId: string) =>
    get().timelines.find((timeline) => timeline.timelineId === timelineId),
  setTimelines: (timelines: TimelineProps[]) => set({ timelines }),
  addTimeline: (timeline: Partial<TimelineProps>, index?: number) =>
    set((state) => {
      const newTimeline: TimelineProps = {
        timelineId: timeline.timelineId ?? uuidv4(),
        timelineHeight: timeline.timelineHeight ?? 54,
        timelineWidth: timeline.timelineWidth ?? 0,
        timelineY: timeline.timelineY ?? 0,
        timelinePadding: timeline.timelinePadding ?? 24,
        type: timeline.type ?? "media",
        muted: timeline.muted ?? false,
        hidden: timeline.hidden ?? false,
      };
      const timelines =
        index !== undefined
          ? [
              ...state.timelines.slice(0, index + 1),
              newTimeline,
              ...state.timelines.slice(index + 1),
            ]
          : [...state.timelines, newTimeline];
      const resolvedTimelines = resolveOverlapsTimelines(timelines);
      return { timelines: resolvedTimelines };
    }),
  removeTimeline: (timelineId: string) =>
    set((state) => {
      const newTimelines = state.timelines.filter(
        (timeline) => timeline.timelineId !== timelineId,
      );
      const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
      return { timelines: resolvedTimelines };
    }),
  updateTimeline: (
    timelineId: string,
    timelineToUpdate: Partial<TimelineProps>,
  ) =>
    set((state) => {
      const newTimelines = state.timelines.map((timeline) =>
        timeline.timelineId === timelineId
          ? { ...timeline, ...timelineToUpdate }
          : timeline,
      );
      const resolvedTimelines = resolveOverlapsTimelines(newTimelines);
      return { timelines: resolvedTimelines };
    }),
  clipboard: [],
  getClipsForTimeline: (timelineId: string) =>
    get().clips.filter(
      (clip) => clip.timelineId === timelineId && !clip.hidden,
    ) || [],
  getClipById: (clipId: string, timelineId?: string) =>
    get().clips.find(
      (clip) =>
        clip.clipId === clipId &&
        (timelineId ? clip.timelineId === timelineId : true),
    ),
  setClips: (clips: AnyClipProps[]) => {
    set((state) => {
      const resolvedClips = resolveOverlaps(clips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    });
  },
  addClip: (clip: AnyClipProps) =>
    set((state) => {
      const newClips = [...state.clips, clip];
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      get()._updateZoomLevel(resolvedClips, clipDuration);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  removeClip: (clipId: string) =>
    set((state) => {
      const clipToDelete = state.clips.find((c) => c.clipId === clipId);

      let newClips: AnyClipProps[] = state.clips;

      if (!clipToDelete) {
        const resolvedClips = resolveOverlaps(newClips);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        get()._updateZoomLevel(resolvedClips, clipDuration);
        return { clips: resolvedClips, clipDuration };
      }

      // Case 1: Deleting a group – remove the group and all its children in one transaction
      if (clipToDelete.type === "group") {
        const nested = (clipToDelete as GroupClipProps).children || ([] as any);
        const childrenSet = new Set((nested as string[][]).flat());
        newClips = newClips.filter(
          (c) => c.clipId !== clipId && !childrenSet.has(c.clipId),
        );
      } else {
        // Case 2: Deleting a non-group clip
        newClips = newClips.filter((c) => c.clipId !== clipId);

        // If the clip belongs to a group, update the group's children and ungroup if only one remains
        if (clipToDelete.groupId) {
          const groupIdx = newClips.findIndex(
            (c) => c.clipId === clipToDelete.groupId,
          );
          const groupClip = groupIdx !== -1 ? newClips[groupIdx] : undefined;
          if (groupClip && groupClip.type === "group") {
            const prevChildren =
              (groupClip as GroupClipProps).children || ([] as any);
            const updatedNested = (prevChildren as string[][])
              .map((sub) => sub.filter((childId) => childId !== clipId))
              .filter((sub) => sub.length > 0);

            const totalRemaining = updatedNested.reduce(
              (acc, sub) => acc + sub.length,
              0,
            );
            if (totalRemaining <= 1) {
              // Ungroup: remove the group clip; if one child remains, reveal it and clear its groupId
              // call ungroupClips as this can get complex
              state.ungroupClips(clipToDelete.groupId);
            } else {
              // Just update group's children nested array
              newClips[groupIdx] = {
                ...(groupClip as GroupClipProps),
                children: updatedNested,
              } as AnyClipProps;
            }
          }
        }
      }

      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      get()._updateZoomLevel(resolvedClips, clipDuration);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  updateClip: (clipId: string, clipToUpdate: Partial<AnyClipProps>) =>
    set((state) => {
      const index = state.clips.findIndex((c) => c.clipId === clipId);
      if (index === -1) {
        return { clips: state.clips };
      }
      const current = state.clips[index];

      // Prepare update payload and handle speed-induced duration rescale
      let nextUpdate: Partial<AnyClipProps> = { ...clipToUpdate };
      if (Object.prototype.hasOwnProperty.call(clipToUpdate, "speed")) {
        const oldSpeed = Math.max(
          0.1,
          Number((current as VideoClipProps).speed || 1),
        );
        const newSpeedRaw = Number((clipToUpdate as any).speed);
        const newSpeed = Math.max(
          0.1,
          Math.min(5, Number.isFinite(newSpeedRaw) ? newSpeedRaw : 1),
        );

        // Anchor at left edge: keep startFrame, adjust endFrame to preserve source coverage
        const start = Math.max(0, Number(current.startFrame || 0));
        const end = Math.max(start + 1, Number(current.endFrame || start + 1));
        const oldDuration = Math.max(1, end - start);
        const newDuration = Math.max(
          1,
          Math.round(oldDuration * (oldSpeed / newSpeed)),
        );
        nextUpdate.endFrame = start + newDuration;
        (nextUpdate as VideoClipProps).speed = newSpeed as any;

        // Update preprocessors timing to match the new speed
        if (
          (current.type === "video" || current.type === "image") &&
          current.preprocessors &&
          current.preprocessors.length > 0
        ) {
          const speedRatio = oldSpeed / newSpeed;
          const clipStart = start;
          const updatedPreprocessors = current.preprocessors.map(
            (preprocessor) => {
              const preprocessorStart = Math.max(
                0,
                Number(preprocessor.startFrame || 0),
              );
              const preprocessorEnd = Math.max(
                preprocessorStart + 1,
                Number(preprocessor.endFrame || preprocessorStart + 1),
              );

              // Calculate relative positions from clip start
              const relativeStart = preprocessorStart - clipStart;
              const relativeEnd = preprocessorEnd - clipStart;

              // Scale the relative positions
              const newRelativeStart = Math.round(relativeStart * speedRatio);
              const newRelativeEnd = Math.round(relativeEnd * speedRatio);

              return {
                ...preprocessor,
                startFrame: clipStart + newRelativeStart,
                endFrame: clipStart + newRelativeEnd,
              };
            },
          );
          (nextUpdate as VideoClipProps | ImageClipProps).preprocessors =
            updatedPreprocessors;
        }
      }

      const updatedClip = { ...current, ...nextUpdate } as AnyClipProps;
      let newClips = [...state.clips];
      newClips[index] = updatedClip;

      // If updating a group clip's location, offset its children accordingly
      if (current.type === "group") {
        const oldStart = current.startFrame ?? 0;
        const newStart = updatedClip.startFrame ?? oldStart;
        const deltaStart = newStart - oldStart;
        const timelineChanged =
          typeof updatedClip.timelineId === "string" &&
          updatedClip.timelineId !== current.timelineId;
        const newTimelineId = updatedClip.timelineId;

        const childrenNested =
          (updatedClip as GroupClipProps).children || ([] as any);
        const childIdsFlat: string[] = (childrenNested as string[][]).reduce(
          (acc: string[], sub) => acc.concat(sub),
          [],
        );
        if (deltaStart !== 0 || timelineChanged) {
          newClips = newClips.map((clipItem) => {
            if (!childIdsFlat.includes(clipItem.clipId)) return clipItem;
            const shiftedStart = (clipItem.startFrame ?? 0) + deltaStart;
            const shiftedEnd = (clipItem.endFrame ?? 0) + deltaStart;
            const nextChild: AnyClipProps = {
              ...clipItem,
              startFrame: shiftedStart,
              endFrame: shiftedEnd,
            };
            if (timelineChanged && newTimelineId) {
              (nextChild as AnyClipProps).timelineId = newTimelineId;
            }
            return nextChild;
          });
        }
      }

      const resolvedClips = resolveOverlaps(newClips as AnyClipProps[]);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      // update the zoom level
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      get()._updateZoomLevel(resolvedClips, clipDuration);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  separateClip: (clipId) =>
    set((state) => {
      const clip = state.clips.find((c) => c.clipId === clipId);
      if (!clip || clip.type !== "video") return { clips: state.clips };
      const newClipId1 = uuidv4();
      const newClipId2 = uuidv4();
      const newAudioTimelineId = uuidv4();

      // Find the index of the current timeline
      const currentTimelineIndex = state.timelines.findIndex(
        (t) => t.timelineId === clip.timelineId,
      );
      const clipTimeline = state.getTimelineById(clip.timelineId || "");

      // Create a new audio timeline for the separated audio clip
      const audioTimeline: TimelineProps = {
        timelineId: newAudioTimelineId,
        type: "audio",
        timelineHeight: getTimelineHeightForClip("audio"),
        timelineWidth: clipTimeline?.timelineWidth ?? 0,
        timelineY:
          (clipTimeline?.timelineY ?? 0) + (clipTimeline?.timelineHeight ?? 54),
        timelinePadding: clipTimeline?.timelinePadding ?? 0,
        muted: false,
        hidden: false,
      };
      const asset = get().getAssetById(clip.assetId);
      const url1 = new URL(asset?.path ?? "");
      const url2 = new URL(asset?.path ?? "");

      url1.hash = "video";
      url2.hash = "audio";
      const asset1 = get().addAsset({ 
        path: url1.toString(),
        type: "video",
        duration: asset?.duration ?? 0,
        width: asset?.width ?? 0,
        height: asset?.height ?? 0,
      });
      const asset2 = get().addAsset({ 
        path: url2.toString(),
        type: "audio",
        duration: asset?.duration ?? 0,
        width: asset?.width ?? 0,
        height: asset?.height ?? 0,
      });

      const clipVideo: AnyClipProps = {
        ...clip,
        assetId: asset1.id,
        clipId: newClipId1,
      };
      const clipAudio: AnyClipProps = {
        ...clip,
        type: "audio",
        assetId: asset2.id,
        clipId: newClipId2,
        timelineId: newAudioTimelineId,
      };

      // Remove the original clip and add both new clips
      const newClips = [
        ...state.clips.filter((c) => c.clipId !== clipId),
        clipVideo,
        clipAudio,
      ];

      // Insert the new audio timeline directly below the current timeline
      const newTimelines =
        currentTimelineIndex !== -1
          ? [
              ...state.timelines.slice(0, currentTimelineIndex + 1),
              audioTimeline,
              ...state.timelines.slice(currentTimelineIndex + 1),
            ]
          : [...state.timelines, audioTimeline];
      const resolvedTimelines = resolveOverlapsTimelines(newTimelines);

      // Resolve overlaps and calculate duration
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(get().assets, resolvedClips);
      return {
        clips: resolvedClips,
        clipDuration,
        assets: prunedAssets,
        timelines: resolvedTimelines,
      };
    }),
  isValidResize: (clipId: string, side: "left" | "right", newFrame: number) => {
    const state = get();
    const sortedClips = [...state.clips].sort(
      (a, b) => (a.startFrame || 0) - (b.startFrame || 0),
    );
    const currentIndex = sortedClips.findIndex((c) => c.clipId === clipId);
    if (currentIndex === -1) return false;

    const currentClip = sortedClips[currentIndex];

    // Enforce min/max duration for model clips if specified in manifest (in frames: fps * duration_secs)
    try {
      if (
        currentClip.type === "model" &&
        (currentClip as any)?.manifest?.spec &&
        !currentClip.assetId
      ) {
        const manifestSpec = (currentClip as any).manifest.spec as any;
        const projectFps = Math.max(1, useControlsStore.getState().fps || 1);
        
        const maxSecs = Number(manifestSpec.max_duration_secs);
        const minSecs = Number(manifestSpec.min_duration_secs);
        const hasMax = Number.isFinite(maxSecs) && maxSecs > 0;
        const hasMin = Number.isFinite(minSecs) && minSecs > 0;
        const maxFrames = hasMax
          ? Math.max(1, Math.floor(maxSecs * projectFps))
          : null;
        const minFrames = hasMin ? Math.max(1, Math.ceil(minSecs * projectFps)) : null;
        if (maxFrames || minFrames) {
          const start = Math.max(0, currentClip.startFrame || 0);
          const end = Math.max(start + 1, currentClip.endFrame || start + 1);
          if (side === "right") {
            const newEndFrame = Math.max(start + 1, newFrame);
            const newDuration = newEndFrame - start;
            if (maxFrames && newDuration > maxFrames) return false;
            if (minFrames && newDuration < minFrames) return false;
          } else {
            const newStartFrame = Math.min(end - 1, newFrame);
            const newDuration = end - newStartFrame;
            if (maxFrames && newDuration > maxFrames) return false;
            if (minFrames && newDuration < minFrames) return false;
          }
        }
      }
    } catch {}

    if (side === "right") {
      // Resize right edge - adjust current clip's end and shift all clips after it
      const oldEndFrame = currentClip.endFrame || 0;
      const newEndFrame = Math.max((currentClip.startFrame || 0) + 1, newFrame);
      const frameDelta = newEndFrame - oldEndFrame;

      if (frameDelta + (currentClip.trimEnd || 0) > 0) {
        return false;
      }
    } else if (side === "left") {
      // Resize left edge - adjust current clip's start and shift all clips before it
      const oldStartFrame = currentClip.startFrame || 0;
      const newStartFrame = Math.min((currentClip.endFrame || 0) - 1, newFrame);
      let frameDelta = newStartFrame - oldStartFrame;

      if (frameDelta + (currentClip.trimStart || 0) < 0) {
        return false;
      }
    }
    return true;
  },
  resizeClip: (clipId: string, side: "left" | "right", newFrame: number) =>
    set((state) => {
      const sortedClips = [...state.clips].sort(
        (a, b) => (a.startFrame || 0) - (b.startFrame || 0),
      );
      const currentIndex = sortedClips.findIndex((c) => c.clipId === clipId);
      if (currentIndex === -1) return { clips: state.clips };

      const currentClip = sortedClips[currentIndex];
      if (currentClip.type === "group") {
        return { clips: state.clips };
      }
      const newClips = [...state.clips];

      if (side === "right") {
        // Resize right edge - adjust current clip's end and shift all clips after it
        const oldEndFrame = currentClip.endFrame || 0;
        const start = Math.max(0, currentClip.startFrame || 0);
        let desiredEndFrame = Math.max(start + 1, newFrame);

        // Clamp by model min/max duration if applicable
        try {
          if (
            currentClip.type === "model" &&
            (currentClip as any)?.manifest?.spec &&
            !currentClip.assetId
          ) {
            const manifestSpec = (currentClip as any).manifest.spec as any;
            const projectFps = Math.max(
              1,
              useControlsStore.getState().fps || 1,
            );
            const maxSecs = Number(manifestSpec.max_duration_secs);
            const minSecs = Number(manifestSpec.min_duration_secs);
            const hasMax = Number.isFinite(maxSecs) && maxSecs > 0;
            const hasMin = Number.isFinite(minSecs) && minSecs > 0;
            const maxFrames = hasMax
                ? Math.max(1, Math.floor(maxSecs * projectFps))
              : null;
            const minFrames = hasMin
              ? Math.max(1, Math.ceil(minSecs * projectFps))
              : null;
            if (maxFrames) {
              const limitEnd = start + maxFrames;
              desiredEndFrame = Math.min(desiredEndFrame, limitEnd);
            }
            if (minFrames) {
              const minEnd = start + minFrames;
              desiredEndFrame = Math.max(desiredEndFrame, minEnd);
            }
          }
        } catch {}

        const frameDelta = desiredEndFrame - oldEndFrame;
 
        
        if (frameDelta + (currentClip.trimEnd || 0) > 0) {
          return { clips: state.clips };
        }

        const currentClipIndex = newClips.findIndex((c) => c.clipId === clipId);
        newClips[currentClipIndex] = {
          ...currentClip,
          endFrame: desiredEndFrame,
          trimEnd: frameDelta + (currentClip.trimEnd || 0),
        };
      
      } else if (side === "left") {
        // Resize left edge - adjust current clip's start and shift all clips before it
        const oldStartFrame = currentClip.startFrame || 0;
        const end = Math.max(currentClip.endFrame || 0, oldStartFrame + 1);
        let desiredStartFrame = Math.min(end - 1, newFrame);

        // Clamp by model min/max duration if applicable
        try {
          if (
            currentClip.type === "model" &&
            (currentClip as any)?.manifest?.spec &&
            !currentClip.assetId
          ) {
            const manifestSpec = (currentClip as any).manifest.spec as any;
            const projectFps = Math.max(
              1,
              useControlsStore.getState().fps || 1,
            );
            const maxSecs = Number(manifestSpec.max_duration_secs);
            const minSecs = Number(manifestSpec.min_duration_secs);
            const hasMax = Number.isFinite(maxSecs) && maxSecs > 0;
            const hasMin = Number.isFinite(minSecs) && minSecs > 0;
            const maxFrames = hasMax
              ? Math.max(1, Math.floor(maxSecs * projectFps))
              : null;
            const minFrames = hasMin
              ? Math.max(1, Math.ceil(minSecs * projectFps))
              : null;
            if (maxFrames) {
              const limitStartMax = Math.max(0, end - maxFrames);
              desiredStartFrame = Math.max(desiredStartFrame, limitStartMax);
            }
            if (minFrames) {
              const limitStartMin = Math.max(0, end - minFrames);
              desiredStartFrame = Math.min(desiredStartFrame, limitStartMin);
            }
          }
        } catch {}

        let frameDelta = desiredStartFrame - oldStartFrame;

        if (frameDelta + (currentClip.trimStart || 0) < 0) {
          return { clips: state.clips };
        }

        if (frameDelta == 0 && (currentClip.trimStart || 0) > 0) {
          frameDelta = Math.max(
            0,
            Math.min(1, (currentClip.trimStart || 0) - 1),
          );
        } else {
          const currentClipIndex = newClips.findIndex(
            (c) => c.clipId === clipId,
          );
          newClips[currentClipIndex] = {
            ...currentClip,
            startFrame: desiredStartFrame,
            trimStart: frameDelta + (currentClip.trimStart || 0),
          };
        }
      }
  
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      get()._updateZoomLevel(resolvedClips, clipDuration);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  updateMaskKeyframes: (clipId, maskId, updater) =>
    set((state) => {
      const newClips = [...state.clips];
      const clipIndex = newClips.findIndex(
        (c) =>
          c.clipId === clipId && (c.type === "video" || c.type === "image"),
      );
      if (clipIndex === -1) return { clips: state.clips };

      const clip = newClips[clipIndex] as VideoClipProps | ImageClipProps;
      const masks = [...(clip.masks || [])];
      const maskIndex = masks.findIndex((m) => m.id === maskId);
      if (maskIndex === -1) return { clips: state.clips };

      const targetMask = masks[maskIndex];
      const updatedKeyframes = updater(
        targetMask.keyframes as
          | Map<number, MaskData>
          | Record<number, MaskData>,
      );

      masks[maskIndex] = {
        ...targetMask,
        keyframes: updatedKeyframes,
        lastModified: Date.now(),
      } as MaskClipProps;

      newClips[clipIndex] = {
        ...clip,
        masks,
      } as AnyClipProps;

      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  updateModelInput: (
    clipId: string,
    inputId: string,
    inputToUpdate: Partial<UIInput>,
  ) =>
    set((state) => {
      const newClips = [...state.clips];
      const clipIndex = newClips.findIndex(
        (c) => c.clipId === clipId && c.type === "model",
      );
      if (clipIndex === -1) return { clips: state.clips };

      const clip = newClips[clipIndex] as ModelClipProps;
      const manifest = clip.manifest;
      if (!manifest) return { clips: state.clips };

      const ui = manifest.spec?.ui || manifest.ui;
      if (!ui || !Array.isArray(ui.inputs)) return { clips: state.clips };

      const inputs: UIInput[] = ui.inputs.map((inp: UIInput) => {
        if (inp.id !== inputId) return inp;
        // Merge shallowly; ensure value override if provided
        const merged: UIInput = { ...inp, ...inputToUpdate } as UIInput;
        return merged;
      });

      // Rebuild manifest immutably preserving structure (prefer spec.ui if present)
      let updatedManifest = { ...manifest } as any;
      if (manifest.spec && manifest.spec.ui) {
        updatedManifest = {
          ...manifest,
          spec: {
            ...manifest.spec,
            ui: {
              ...(manifest.spec.ui || {}),
              inputs,
            },
          },
        } as any;
      } else if (manifest.ui) {
        updatedManifest = {
          ...manifest,
          ui: {
            ...(manifest.ui || {}),
            inputs,
          },
        } as any;
      } else {
        // No UI present; nothing to update
        return { clips: state.clips };
      }

      newClips[clipIndex] = {
        ...clip,
        manifest: updatedManifest,
      } as AnyClipProps;

      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  getRawModelValues: (clipId: string) => {
    const clip = get().clips.find(
      (c) => c.clipId === clipId && c.type === "model",
    ) as ModelClipProps | undefined;
    if (!clip) return null;
    const manifest = clip.manifest;
    if (!manifest) return null;
    const ui = manifest.spec?.ui || manifest.ui;
    if (!ui || !Array.isArray(ui.inputs)) return null;
    const output: Record<string, any> = {};
    ui.inputs.forEach((inp) => {
      output[inp.id] = inp.value;
    });
    return output;
  },
  getModelValues: (clipId: string) => {
    const clip = get().clips.find(
      (c) => c.clipId === clipId && c.type === "model",
    ) as ModelClipProps | undefined;
    if (!clip) return null;
    const manifest = clip.manifest;
    if (!manifest) return null;
    const ui = manifest.spec?.ui || manifest.ui;
    if (!ui || !Array.isArray(ui.inputs)) return null;
    const output: Record<string, any> = {};
    ui.inputs.forEach((inp) => {
      const typeStr = String((inp as any)?.type || "");
      // Only JSON-parse values for input types that intentionally store JSON in `inp.value`.
      // Critical: do NOT parse for plain `text` inputs, otherwise entering JSON would be
      // treated as an object and not a literal string.
      const shouldJsonParseValue = (() => {
        const t = typeStr.toLowerCase();
        if (t === "text_list") return true;
        if (t === "image_list") return true;
        if (t.startsWith("image")) return true;
        if (t.startsWith("video")) return true;
        if (t.startsWith("audio")) return true;
        return false;
      })();

      let parsedVal: any = inp.value;
      if (typeof parsedVal === "string" && shouldJsonParseValue) {
        try {
          parsedVal = JSON.parse(parsedVal);
        } catch {
          // ignore
        }
      }

      let finalVal: any = parsedVal ?? inp.default;
      // For media-like inputs, attach either selected range (video/audio) or selected frame (image)
      const isVideoish = typeStr.startsWith("video");
      const isAudioish = typeStr.startsWith("audio");
      const isImageish = typeStr.startsWith("image");
      const isImageList = typeStr === "image_list";
      // image_list is a special case: keep it as an array of selections, do not coerce to a single clip
      if (isVideoish || isAudioish || (isImageish && !isImageList)) {
        // Resolve composite values (selection + preprocessor flags) and ensure we return AnyClipProps
        // Prefer: finalVal = <clip object> with selectedRange/selectedFrame and apply_preprocessor (if present)
        let selectedClip: any = null;
        let applyFlag: boolean | undefined = undefined;
        if (finalVal && typeof finalVal === "object") {
          // Composite form: { selection, apply_preprocessor?, apply?, preprocessor_ref?, ... }
          const maybeSelection: any = (finalVal as any).selection ?? finalVal;
          if (
            maybeSelection &&
            typeof maybeSelection === "object" &&
            "clipId" in maybeSelection
          ) {
            const lookedUp = get().getClipById((maybeSelection as any).clipId);
            selectedClip = lookedUp || maybeSelection;
          } else {
            selectedClip = maybeSelection;
          }
          if (
            Object.prototype.hasOwnProperty.call(
              finalVal as any,
              "apply_preprocessor",
            ) &&
            typeof (finalVal as any).apply_preprocessor === "boolean"
          ) {
            applyFlag = (finalVal as any).apply_preprocessor as boolean;
          } else if (
            Object.prototype.hasOwnProperty.call(finalVal as any, "apply") &&
            typeof (finalVal as any).apply === "boolean"
          ) {
            applyFlag = (finalVal as any).apply as boolean;
          }
        } else {
          selectedClip = finalVal;
        }
        const inputStore = globalInputControlsStore.getState();

        if (isVideoish || isAudioish) {
          const [start, end] = inputStore.getSelectedRange(inp.id, clipId);
          if (selectedClip && typeof selectedClip === "object") {
            finalVal = {
              ...(selectedClip as any),
              selectedRange: [start, end],
              ...(typeof applyFlag === "boolean"
                ? { apply_preprocessor: applyFlag }
                : {}),
            };
          } else {
            finalVal = {
              selection: selectedClip,
              selectedRange: [start, end],
              ...(typeof applyFlag === "boolean"
                ? { apply_preprocessor: applyFlag }
                : {}),
            };
          }
        } else if (isImageish) {
          const focus = inputStore.getFocusFrame(inp.id, clipId);
          if (selectedClip && typeof selectedClip === "object") {
            finalVal = {
              ...(selectedClip as any),
              selectedFrame: focus,
              ...(typeof applyFlag === "boolean"
                ? { apply_preprocessor: applyFlag }
                : {}),
            };
          } else {
            finalVal = {
              selection: selectedClip,
              selectedFrame: focus,
              ...(typeof applyFlag === "boolean"
                ? { apply_preprocessor: applyFlag }
                : {}),
            };
          }
        }
      }

      output[inp.id] = finalVal;
    });
    return output as Record<string, any>;
  },
  groupClips: (clipIds: string[]) =>
    set((state) => {
      const newClips = [...state.clips];
      const clips = newClips.filter((c) => clipIds.includes(c.clipId));
      // Sort clips by vertical timeline position (top to bottom), then by startFrame
      const sortedByTimeline = [...clips].sort((a, b) => {
        const ta = state.timelines.find((t) => t.timelineId === a.timelineId);
        const tb = state.timelines.find((t) => t.timelineId === b.timelineId);
        const yA = ta?.timelineY ?? 0;
        const yB = tb?.timelineY ?? 0;
        if (yA !== yB) return yA - yB; // earlier (smaller) timelineY first
        return (a.startFrame || 0) - (b.startFrame || 0);
      });
      // sort our clips based on location on the tim
      const startFrame = clips.reduce(
        (min, c) => Math.min(min, c.startFrame || 0),
        Infinity,
      );
      const endFrame = clips.reduce(
        (max, c) => Math.max(max, c.endFrame || 0),
        -Infinity,
      );
      const groupClipId = uuidv4();

      // Choose a valid media timeline with sufficient free space; if none, create one
      const desiredType: TimelineType = "media";
      const intervalOverlaps = (
        loA: number,
        hiA: number,
        loB: number,
        hiB: number,
      ) => loA < hiB && hiA > loB;

      // Check free space on a timeline between startFrame and endFrame
      // Exclude the selected clips being grouped since they'll be hidden/removed from timelines
      const idsToExclude = new Set(clips.map((c) => c.clipId));
      const hasFreeSpace = (timelineId: string) => {
        const existing = state
          .getClipsForTimeline(timelineId)
          .filter((c) => !idsToExclude.has(c.clipId))
          .map((c) => ({ lo: c.startFrame || 0, hi: c.endFrame || 0 }))
          .filter((iv) => iv.hi > iv.lo);
        return !existing.some((iv) =>
          intervalOverlaps(startFrame, endFrame, iv.lo, iv.hi),
        );
      };

      // Candidates: all existing media timelines with free space, pick lowest timelineY
      const mediaTimelines = state.timelines.filter(
        (t) => t.type === desiredType,
      );
      const candidateTimelines = mediaTimelines
        .filter((t) => hasFreeSpace(t.timelineId))
        .sort((a, b) => (a.timelineY || 0) - (b.timelineY || 0));

      let targetTimelineId: string | undefined =
        candidateTimelines[0]?.timelineId;

      const timelines = [...state.timelines];

      // If no suitable existing timeline, create a new media timeline
      if (!targetTimelineId) {
        const newTimelineId = uuidv4();
        const last = state.timelines[state.timelines.length - 1];
        const newTimeline: TimelineProps = {
          timelineId: newTimelineId,
          type: desiredType,
          timelineHeight: getTimelineHeightForClip("group"),
          timelineWidth: last?.timelineWidth ?? 0,
          timelineY: (last?.timelineY ?? 0) + (last?.timelineHeight ?? 54),
          timelinePadding: last?.timelinePadding ?? 24,
          muted: false,
          hidden: false,
        };
        timelines.push(newTimeline);
        targetTimelineId = newTimelineId;
      }

      // Build nested children arrays: per-timeline lists ordered by y then by startFrame
      const uniqueSortedTimelineIds: string[] = Array.from(
        new Set(sortedByTimeline.map((c) => c.timelineId || "")),
      );
      const timelinesByY = uniqueSortedTimelineIds
        .map((id) => ({
          id,
          y: state.timelines.find((t) => t.timelineId === id)?.timelineY ?? 0,
        }))
        .sort((a, b) => a.y - b.y)
        .map((x) => x.id);

      const childrenNested: string[][] = timelinesByY
        .map((tid) =>
          sortedByTimeline
            .filter((c) => c.timelineId === tid)
            .sort((a, b) => (a.startFrame || 0) - (b.startFrame || 0))
            .map((c) => c.clipId),
        )
        .filter((sub) => sub.length > 0);

      const groupClip: GroupClipProps = {
        type: "group",
        clipId: groupClipId,
        timelineId: targetTimelineId,
        startFrame: startFrame,
        endFrame: endFrame,
        children: childrenNested,
      };

      // we need to replace our sortedbytimeline clips and add the group clip
      const clipsToReplace = [...state.clips];
      sortedByTimeline.forEach((c) => {
        const index = clipsToReplace.findIndex((x) => x.clipId === c.clipId);
        if (index !== -1) {
          clipsToReplace[index] = { ...c, groupId: groupClipId, hidden: true };
        }
      });

      const finalClips = [...clipsToReplace, groupClip];
      const resolvedClips = resolveOverlaps(finalClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      get()._updateZoomLevel(resolvedClips, clipDuration);
      const resolvedTimelines = resolveOverlapsTimelines(timelines);

      // Select the newly created group
      try {
        const controls = useControlsStore.getState();
        controls.setSelectedClipIds([groupClipId]);
      } catch {}

      return {
        clips: resolvedClips,
        clipDuration,
        assets: prunedAssets,
        timelines: resolvedTimelines,
      };
    }),
  ungroupClips: (_groupId: string) =>
    set((state) => {
      const groupClip = state.clips.find((c) => c.clipId === _groupId);
      if (!groupClip || groupClip.type !== "group")
        return { clips: state.clips };

      const childrenNested = ((groupClip as GroupClipProps).children ||
        []) as string[][];
      if (!childrenNested || childrenNested.length === 0) {
        // No children, just remove the empty group
        const remaining = state.clips.filter((c) => c.clipId !== _groupId);
        const resolvedClips = resolveOverlaps(remaining);
        const clipDuration = calculateTotalClipDuration(resolvedClips);
        get()._updateZoomLevel(resolvedClips, clipDuration);
        const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
        return { clips: resolvedClips, clipDuration, assets: prunedAssets };
      }

      // Determine ordering: always ungroup upwards
      // childrenNested is ordered top->bottom; bottom is the last array
      const bottomIndex = childrenNested.length - 1;

      // Precompute child clips map for quick lookup
      const clipsById = new Map(state.clips.map((c) => [c.clipId, c] as const));

      // Helper: derive required timeline type for a subarray from its first child
      const requiredTypeForSub = (ids: string[]): TimelineType => {
        const first = ids.map((id) => clipsById.get(id)).find(Boolean) as
          | AnyClipProps
          | undefined;
        if (!first) return "media";
        return getTimelineTypeForClip(first);
      };

      // Helper: compute [start,end) for a subarray
      const boundsForSub = (ids: string[]): [number, number] => {
        const items = ids
          .map((id) => clipsById.get(id))
          .filter(Boolean) as AnyClipProps[];
        const s = items.reduce(
          (min, c) => Math.min(min, c.startFrame || 0),
          Infinity,
        );
        const e = items.reduce(
          (max, c) => Math.max(max, c.endFrame || 0),
          -Infinity,
        );
        return [Math.max(0, s), Math.max(Math.max(0, s) + 1, e)];
      };

      const intervalOverlaps = (
        loA: number,
        hiA: number,
        loB: number,
        hiB: number,
      ) => loA < hiB && hiA > loB;

      // Build quick timeline index lookup and locate the group's timeline position
      const timelines = [...state.timelines];
      const timelineIndexById = new Map(
        timelines.map((t, i) => [t.timelineId, i] as const),
      );
      const groupTimelineIndex = timelineIndexById.get(
        groupClip.timelineId || "",
      );

      // If group's timeline is missing, fallback to appending to the bottom
      const fallbackGroupIndex =
        typeof groupTimelineIndex === "number"
          ? groupTimelineIndex
          : timelines.length - 1;

      // Track planned intervals per timeline to avoid conflicts when placing multiple subarrays
      const plannedIntervals = new Map<string, Array<[number, number]>>();

      const hasFreeSpace = (
        timelineId: string,
        start: number,
        end: number,
        excludeClipIds: Set<string>,
      ) => {
        const existing = state
          .getClipsForTimeline(timelineId)
          .filter((c) => !excludeClipIds.has(c.clipId))
          .map((c) => ({ lo: c.startFrame || 0, hi: c.endFrame || 0 }))
          .filter((iv) => iv.hi > iv.lo);
        const planned = plannedIntervals.get(timelineId) || [];
        return ![...existing, ...planned.map(([lo, hi]) => ({ lo, hi }))].some(
          (iv) => intervalOverlaps(start, end, iv.lo, iv.hi),
        );
      };

      // Assign a target timeline for a subarray, searching upwards (indices decreasing)
      const assignTimelineUpwards = (
        subIds: string[],
        baseIndex: number,
        preferExactIndexId?: string,
      ): string => {
        const [s, e] = boundsForSub(subIds);
        const excludeIds = new Set<string>(subIds.concat(_groupId));
        const requiredType = requiredTypeForSub(subIds);

        // 1) Try the preferred timeline (group's timeline) if provided and compatible
        if (preferExactIndexId) {
          const tl = timelines.find((t) => t.timelineId === preferExactIndexId);
          if (
            tl &&
            tl.type === requiredType &&
            hasFreeSpace(tl.timelineId, s, e, excludeIds)
          ) {
            const arr = plannedIntervals.get(tl.timelineId) || [];
            arr.push([s, e]);
            plannedIntervals.set(tl.timelineId, arr);
            return tl.timelineId;
          }
        }

        // 2) Search existing timelines above for matching type and free space
        for (let i = Math.max(0, baseIndex - 1); i >= 0; i--) {
          const tl = timelines[i];
          if (!tl) continue;
          if (tl.type !== requiredType) continue;
          if (hasFreeSpace(tl.timelineId, s, e, excludeIds)) {
            const arr = plannedIntervals.get(tl.timelineId) || [];
            arr.push([s, e]);
            plannedIntervals.set(tl.timelineId, arr);
            return tl.timelineId;
          }
        }

        // 3) Create a new timeline of the required type inserted above the baseIndex
        const newTimelineId = uuidv4();
        const height = getTimelineHeightForClip(requiredType);
        const ref = timelines[Math.max(0, baseIndex)];
        const newTimeline: TimelineProps = {
          timelineId: newTimelineId,
          type: requiredType,
          timelineHeight: height,
          timelineWidth: ref?.timelineWidth ?? 0,
          timelineY: 0,
          timelinePadding: ref?.timelinePadding ?? 24,
          muted: false,
          hidden: false,
        };
        const insertAt = Math.max(0, baseIndex);
        timelines.splice(insertAt, 0, newTimeline);

        // Reserve the interval on this newly created timeline
        plannedIntervals.set(newTimelineId, [[s, e]]);
        return newTimelineId;
      };

      // Build assignments from bottom subarray at group's timeline, then upwards
      const assignments = new Map<number, string>(); // subIndex -> timelineId

      // Bottom subarray: prefer group's timeline
      const groupTlId = (timelines[fallbackGroupIndex] || {}).timelineId;
      const bottomTlId = assignTimelineUpwards(
        childrenNested[bottomIndex],
        fallbackGroupIndex,
        groupTlId,
      );
      assignments.set(bottomIndex, bottomTlId);

      // Remaining subarrays: place upwards (search above the group's index)
      for (let idx = bottomIndex - 1; idx >= 0; idx--) {
        const tlId = assignTimelineUpwards(
          childrenNested[idx],
          fallbackGroupIndex,
        );
        assignments.set(idx, tlId);
      }

      // Reveal children and remove the group
      const childIdsFlat = childrenNested.flat();
      const childIdSet = new Set(childIdsFlat);

      const newClips = state.clips
        .filter((c) => c.clipId !== _groupId)
        .map((c) => {
          if (!childIdSet.has(c.clipId)) return c;
          // Determine which subarray this child belongs to
          const subIndex = childrenNested.findIndex((sub) =>
            sub.includes(c.clipId),
          );
          const targetTimelineId = assignments.get(subIndex || 0);
          return {
            ...c,
            hidden: false,
            groupId: undefined,
            timelineId: targetTimelineId ?? c.timelineId,
          } as AnyClipProps;
        });

      // Resolve timelines positioning after inserts
      const resolvedTimelines = resolveOverlapsTimelines(timelines);

      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      get()._updateZoomLevel(resolvedClips, clipDuration);

      // Select all ungrouped children
      try {
        const controls = useControlsStore.getState();
        controls.setSelectedClipIds(childIdsFlat);
      } catch {}

      return {
        clips: resolvedClips,
        clipDuration,
        assets: pruneAssetsForClips(state.assets, resolvedClips),
        timelines: resolvedTimelines,
      };
    }),

  // Create two new clips from the original clip at the cut frame
  splitClip: (cutFrame: number, clipId: string) =>
    set((state) => {
      // Find the clip that contains the cut frame
      const clip = state.clips.find((clip) => {
        const startFrame = clip.startFrame || 0;
        const endFrame = clip.endFrame || 0;
        return (
          cutFrame > startFrame && cutFrame < endFrame && clip.clipId === clipId
        );
      });

      if (!clip) return { clips: state.clips };

      // remove the clip from the array
      const filteredClips = state.clips.filter((c) => c.clipId !== clip.clipId);

      // create new clip ids
      const newClipId1 = uuidv4();
      const newClipId2 = uuidv4();
      // IMPORTANT:
      // Splitting must not change the displayed frames at any given timeline focusFrame.
      // We achieve this by preserving the original clip's media mapping and only
      // adjusting trims for the new boundaries.
      //
      // - `trimStart` is the source-frame offset at the clip's start (in project frames).
      // - `trimEnd` is a <= 0 "give" value used by resize logic (negative when trimmed in).
      // - The right clip must advance `trimStart` by the cut offset so it starts on the
      //   same source frame the original clip had at `cutFrame`.
      const oldStart = Math.max(0, Number(clip.startFrame ?? 0));
      const oldEnd = Math.max(oldStart + 1, Number(clip.endFrame ?? oldStart + 1));
      const safeOldTrimStart = isFinite(clip.trimStart ?? 0)
        ? Number(clip.trimStart ?? 0)
        : 0;
      const safeOldTrimEnd = isFinite(clip.trimEnd ?? 0)
        ? Number(clip.trimEnd ?? 0)
        : 0;
      const cutRel = Math.max(0, Math.min(oldEnd - oldStart, cutFrame - oldStart));

      // Left clip: same start/trimStart; end at cut; adjust trimEnd exactly like a right-edge resize.
      const newClip1: AnyClipProps = {
        ...clip,
        clipId: newClipId1,
        endFrame: cutFrame,
        trimStart: safeOldTrimStart,
        trimEnd: safeOldTrimEnd + (cutFrame - oldEnd),
      };

      // Right clip: start at cut; advance trimStart so it begins at the cut point in the source.
      const newClip2: AnyClipProps = {
        ...clip,
        clipId: newClipId2,
        startFrame: cutFrame,
        trimStart: safeOldTrimStart + cutRel,
        trimEnd: safeOldTrimEnd,
      };

      // Split/rebase preprocessors (stored in clip-local frames: 0 = clip start).
      if (clip.type === "image" || clip.type === "video") {
        const origPre = (clip as VideoClipProps | ImageClipProps).preprocessors;
        if (Array.isArray(origPre) && origPre.length > 0) {
          const clip1Preprocessors: PreprocessorClipProps[] = [];
          const clip2Preprocessors: PreprocessorClipProps[] = [];
          for (const p of origPre) {
            const ps = Number(p.startFrame ?? 0);
            const pe = Number(p.endFrame ?? 0);
            if (!Number.isFinite(ps) || !Number.isFinite(pe) || pe <= ps) continue;

            // Entirely before cut (left)
            if (pe <= cutRel) {
              clip1Preprocessors.push({ ...p });
              continue;
            }
            // Entirely after cut (right) - rebase to new clip start
            if (ps >= cutRel) {
              clip2Preprocessors.push({
                ...p,
                startFrame: ps - cutRel,
                endFrame: pe - cutRel,
              });
              continue;
            }
            // Spans cut: split into two preprocessors
            const left: PreprocessorClipProps = {
              ...p,
              endFrame: cutRel,
            };
            const right: PreprocessorClipProps = {
              ...p,
              id: uuidv4(),
              startFrame: 0,
              endFrame: pe - cutRel,
            };

            // If this preprocessor has a completed in-place output asset, slice the
            // underlying asset for the right side so its frame 0 corresponds to `cutRel`.
            // This preserves preview frame correctness when the cut happens mid-preprocessor.
            try {
              if (p.assetId) {
                const srcAsset = get().getAssetById(p.assetId);
                if (srcAsset?.path) {
                  const url = new URL(srcAsset.path);
                  const baseStart = url.searchParams.get("startFrame")
                    ? Number(url.searchParams.get("startFrame"))
                    : 0;
                  const skipFrames = Math.max(0, Math.round(cutRel - ps));
                  const remaining = Math.max(1, Math.round(pe - cutRel));
                  const newStart = Math.max(0, Math.round(baseStart + skipFrames));
                  const newEnd = Math.max(newStart + 1, newStart + remaining);
                  url.searchParams.set("startFrame", String(newStart));
                  url.searchParams.set("endFrame", String(newEnd));

                  const sliced = get().addAsset({
                    path: url.toString(),
                    type: srcAsset.type,
                    width: srcAsset.width,
                    height: srcAsset.height,
                  });
                  right.assetId = sliced.id;
                  // Best-effort warm media info for quick preview
                  void getMediaInfo(sliced.path);
                }
              }
            } catch {}
            clip1Preprocessors.push(left);
            clip2Preprocessors.push(right);
          }
          (newClip1 as VideoClipProps | ImageClipProps).preprocessors =
            clip1Preprocessors;
          (newClip2 as VideoClipProps | ImageClipProps).preprocessors =
            clip2Preprocessors;
        }

        // Split masks by local frame index (mask frames use the same local frame space
        // as preview: focusFrame - startFrame + trimStart).
        const originalMasks = (clip as VideoClipProps | ImageClipProps).masks;
        if (Array.isArray(originalMasks) && originalMasks.length > 0) {
          const masksForClip1: MaskClipProps[] = [];
          const masksForClip2: MaskClipProps[] = [];

          const startFrame = Number(clip.startFrame ?? 0);
          const trimStart = isFinite(clip.trimStart ?? 0)
            ? Number(clip.trimStart ?? 0)
            : 0;
          const cutLocal = Math.max(0, cutFrame - startFrame + trimStart);

          const makeEmptyKeyframesLike = (
            kf: Map<number, MaskData> | Record<number, MaskData>,
          ) =>
            kf instanceof Map
              ? new Map<number, MaskData>()
              : ({} as Record<number, MaskData>);

          const setKeyframe = (
            target: Map<number, MaskData> | Record<number, MaskData>,
            frame: number,
            data: MaskData,
          ) => {
            if (target instanceof Map) target.set(frame, data);
            else (target as Record<number, MaskData>)[frame] = data;
          };

          const getKeyframe = (
            source: Map<number, MaskData> | Record<number, MaskData>,
            frame: number,
          ): MaskData | undefined => {
            return source instanceof Map
              ? source.get(frame)
              : (source as Record<number, MaskData>)[frame];
          };

          const getSortedFrames = (
            kf: Map<number, MaskData> | Record<number, MaskData>,
          ): number[] =>
            (kf instanceof Map
              ? Array.from(kf.keys())
              : Object.keys(kf).map(Number)
            ).sort((a, b) => a - b);

          for (const mask of originalMasks) {
            const keyframes = mask.keyframes;
            const frames = getSortedFrames(keyframes);
            const now = Date.now();

            // Image clips only use frame 0; copy that to both
            if (clip.type === "image") {
              const dataAt0 =
                getKeyframe(keyframes, 0) ??
                (frames.length > 0
                  ? getKeyframe(keyframes, frames[0])
                  : undefined);
              if (dataAt0) {
                const kf1 = makeEmptyKeyframesLike(keyframes);
                const kf2 = makeEmptyKeyframesLike(keyframes);
                setKeyframe(kf1, 0, { ...dataAt0 });
                setKeyframe(kf2, 0, { ...dataAt0 });
                masksForClip1.push({
                  ...mask,
                  keyframes: kf1,
                  clipId: newClip1.clipId,
                  lastModified: now,
                });
                masksForClip2.push({
                  ...mask,
                  keyframes: kf2,
                  clipId: newClip2.clipId,
                  lastModified: now,
                });
              }
              continue;
            }

            const kfLeft = makeEmptyKeyframesLike(keyframes);
            const kfRight = makeEmptyKeyframesLike(keyframes);
            for (const f of frames) {
              const data = getKeyframe(keyframes, f);
              if (!data) continue;
              if (f < cutLocal) setKeyframe(kfLeft, f, { ...data });
              else setKeyframe(kfRight, f, { ...data });
            }

            masksForClip1.push({
              ...mask,
              keyframes: kfLeft,
              clipId: newClip1.clipId,
              lastModified: now,
            });
            masksForClip2.push({
              ...mask,
              keyframes: kfRight,
              clipId: newClip2.clipId,
              lastModified: now,
            });
          }

          (newClip1 as VideoClipProps | ImageClipProps).masks = masksForClip1;
          (newClip2 as VideoClipProps | ImageClipProps).masks = masksForClip2;
        }
      }

      const newClips = [...filteredClips, newClip1, newClip2];
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(get().assets, resolvedClips);
      get()._updateZoomLevel(resolvedClips, clipDuration);
      const controlStore = useControlsStore.getState();
      controlStore.setSelectedClipIds([newClip1.clipId, newClip2.clipId]);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  mergeClips: (clipIds: string[]) =>
    set((state) => {
      if (clipIds.length < 2) return { clips: state.clips };

      // Find all clips to merge
      const clipsToMerge = clipIds
        .map((id) => state.clips.find((clip) => clip.clipId === id))
        .filter(Boolean) as AnyClipProps[];

      if (clipsToMerge.length < 2) return { clips: state.clips };

      // Sort clips by start frame to check adjacency
      const sortedClips = clipsToMerge.sort(
        (a, b) => (a.startFrame || 0) - (b.startFrame || 0),
      );

      // Check if all clips are frame-adjacent (no gaps between them)
      for (let i = 0; i < sortedClips.length - 1; i++) {
        const currentEnd = sortedClips[i].endFrame || 0;
        const nextStart = sortedClips[i + 1].startFrame || 0;

        // If there's a gap between clips, don't merge
        if (currentEnd !== nextStart) {
          return { clips: state.clips };
        }
      }

      // Remove the clips from the array
      const filteredClips = state.clips.filter(
        (clip) => !clipIds.includes(clip.clipId),
      );

      // Find the bounds of all clips to merge
      const clipStart = sortedClips[0].startFrame || 0;
      const clipEnd = sortedClips[sortedClips.length - 1].endFrame || 0;

      // Use the first clip as the base and merge all others into it
      const baseClip = sortedClips[0];
      const newClip = { ...baseClip, startFrame: clipStart, endFrame: clipEnd };
      const newClips = [...filteredClips, newClip];
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  copyClips: (clipIds: string[]) =>
    set((state) => {
      if (!clipIds || clipIds.length === 0)
        return { clipboard: state.clipboard };
      const toCopy = clipIds
        .map((id) => state.clips.find((c) => c.clipId === id))
        .filter(Boolean) as AnyClipProps[];
      return { clipboard: toCopy.map((c) => ({ ...c })) };
    }),
  getClipsByType: (type: ClipType) =>
    get().clips.filter((c) => c.type === type),
  getClipsForGroup: (children: string[][]) => {
    const state = get();
    const clipsById = new Map(state.clips.map((c) => [c.clipId, c] as const));
    const getY = (timelineId?: string) =>
      state.timelines.find((t) => t.timelineId === timelineId)?.timelineY ?? 0;

    // Map each child id to its clip, filter missing, then sort by render order:
    // bottom-to-top (higher timelineY first). Use startFrame as a stable tie-breaker.
    const flat: AnyClipProps[] = (children || [])
      .flat()
      .map((id) => clipsById.get(id))
      .filter((c): c is AnyClipProps => Boolean(c))
      .sort((a, b) => {
        const ya = getY(a.timelineId);
        const yb = getY(b.timelineId);
        if (ya !== yb) return yb - ya; // larger y first (drawn earlier)
        const sa = a.startFrame ?? 0;
        const sb = b.startFrame ?? 0;
        return sa - sb;
      })
      .reverse();

    return flat;
  },
  cutClips: (clipIds: string[]) =>
    set((state) => {
      if (!clipIds || clipIds.length === 0)
        return { clips: state.clips, clipboard: state.clipboard };
      const toCut = clipIds
        .map((id) => state.clips.find((c) => c.clipId === id))
        .filter(Boolean) as AnyClipProps[];
      const remaining = state.clips.filter((c) => !clipIds.includes(c.clipId));
      const resolvedClips = resolveOverlaps(remaining);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      return {
        clips: resolvedClips,
        clipDuration,
        clipboard: toCut.map((c) => ({ ...c })),
        assets: prunedAssets,
      };
    }),
  pasteClips: (atFrame?: number, targetTimelineId?: string) =>
    set((state) => {
      const clipboardItems = state.clipboard || [];
      if (clipboardItems.length === 0) return { clips: state.clips };
      const baseStart = Math.min(
        ...clipboardItems.map((c) => c.startFrame || 0),
      );
      const insertionFrame = Math.max(0, Math.round(atFrame || 0));
      const newIds: string[] = [];
      // Determine the destination timeline per item: prefer provided target, otherwise a compatible existing timeline or create one
      const chooseTimelineFor = (template: AnyClipProps): string => {
        if (targetTimelineId) return targetTimelineId;
        const desiredType = getTimelineTypeForClip(template);
        const existing = state.timelines.find((t) => t.type === desiredType);
        if (existing) return existing.timelineId;
        // If no matching timeline, create one beside last timeline
        const timelineId = uuidv4();
        const last = state.timelines[state.timelines.length - 1];
        const newTimeline: TimelineProps = {
          timelineId,
          type: desiredType,
          timelineHeight: getTimelineHeightForClip(desiredType),
          timelineWidth: last?.timelineWidth ?? 0,
          timelineY: (last?.timelineY ?? 0) + (last?.timelineHeight ?? 54),
          timelinePadding: last?.timelinePadding ?? 24,
          muted: false,
          hidden: false,
        };
        state.addTimeline(newTimeline);
        return timelineId;
      };
      const pasted = clipboardItems.map((template) => {
        const templateStart = template.startFrame || 0;
        const templateEnd = template.endFrame || 0;
        const duration = Math.max(1, templateEnd - templateStart);
        const offset = templateStart - baseStart;
        const start = insertionFrame + offset;
        const end = start + duration;
        const newId = uuidv4();
        newIds.push(newId);
        const timelineId = chooseTimelineFor(template);
        return {
          ...template,
          clipId: newId,
          timelineId,
          startFrame: start,
          endFrame: end,
          trimEnd: 0,
          trimStart: 0,
        } as AnyClipProps;
      });
      const newClips = [...state.clips, ...pasted];
      const resolvedClips = resolveOverlaps(newClips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);
      // Select newly pasted clips
      try {
        const controls = useControlsStore.getState();
        controls.setSelectedClipIds(newIds);
      } catch {}
      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    }),
  moveClipToEnd: (clipId: string) =>
    set((state) => {
      const clip = state.clips.find((clip) => clip.clipId === clipId);
      if (!clip) return { clips: state.clips };
      const newClips = [
        ...state.clips.filter((c) => c.clipId !== clipId),
        clip,
      ];
      const clipDuration = calculateTotalClipDuration(newClips);
      return { clips: newClips, clipDuration };
    }),
  getClipAtFrame: (frame: number) => {
    const clips = get().clips;
    const clip = clips.find(
      (clip) =>
        frame >= (clip.startFrame || 0) && frame <= (clip.endFrame || 0),
    );
    if (!clip) return null;
    return [clip, frame - (clip.startFrame || 0)];
  },
  clipWithinFrame: (
    clip: AnyClipProps,
    frame: number,
    overlap: boolean = false,
    overlapAmount: number = 0,
  ) => {
    if (overlap) {
      return (
        frame >= (clip.startFrame || 0) - (overlapAmount || 0) &&
        frame <= (clip.endFrame || 0)
      );
    }
    return frame >= (clip.startFrame || 0) && frame <= (clip.endFrame || 0);
  },
  getTimelinePosition: (timelineId: string, scrollY?: number) => {
    const timeline = get().timelines.find((t) => t.timelineId === timelineId);
    if (!timeline) return { top: 0, bottom: 0, left: 0, right: 0 };
    const top = (timeline.timelineY ?? 0) + 8 - (scrollY || 0);
    const bottom = top + (timeline.timelineHeight ?? 54);
    return {
      top,
      bottom,
      left: timeline.timelinePadding!,
      right: timeline.timelinePadding! + (timeline.timelineWidth ?? 0),
    };
  },
  getClipPosition: (clipId: string, scrollY?: number) => {
    const clip = get().clips.find((c) => c.clipId === clipId);
    if (!clip) return { top: 0, bottom: 0, left: 0, right: 0 };

    const timeline = get().timelines.find(
      (t) => t.timelineId === clip.timelineId,
    );
    if (!timeline) return { top: 0, bottom: 0, left: 0, right: 0 };

    const controls = useControlsStore.getState();
    const timelineDuration = controls.timelineDuration;

    const clipX = getClipX(
      clip.startFrame ?? 0,
      clip.endFrame ?? 0,
      timeline.timelineWidth ?? 0,
      timelineDuration,
    );
    const clipWidth = getClipWidth(
      clip.startFrame ?? 0,
      clip.endFrame ?? 0,
      timeline.timelineWidth ?? 0,
      timelineDuration,
    );

    const top = (timeline.timelineY ?? 0) + 8 - (scrollY || 0);
    const bottom = top + (timeline.timelineHeight ?? 54);
    const left = (timeline.timelinePadding ?? 0) + clipX;
    const right = left + clipWidth;

    return { top, bottom, left, right };
  },
  getClipPositionScore: (clipId: string) => {
    const state = get();
    const clip = state.clips.find((c) => c.clipId === clipId);
    if (!clip) return 0;

    // Timeline order: sort by Y (top->bottom). Lower (greater Y) timelines get higher base score.
    const timelinesSorted = [...state.timelines].sort(
      (a, b) => (a.timelineY ?? 0) - (b.timelineY ?? 0),
    );
    const timelineIndex = Math.max(
      0,
      timelinesSorted.findIndex((t) => t.timelineId === clip.timelineId),
    );

    // Weights: ensure timeline separation dominates layer, which dominates endFrame
    const TIMELINE_WEIGHT = 1e9; // enough to dominate any layer/endFrame additions
    const LAYER_WEIGHT = 1e6; // enough to dominate endFrame within same timeline

    // Compute layer rank within a timeline considering groups (we group upwards)
    let layerRank: number = 0;

    // If clip is part of a group, rank by the child's subarray index; last children array ranks highest
    const parentGroup: GroupClipProps | undefined =
      typeof (clip as any).groupId === "string"
        ? (state.clips.find(
            (c) => c.clipId === (clip as any).groupId && c.type === "group",
          ) as GroupClipProps | undefined)
        : undefined;

    if (parentGroup && Array.isArray(parentGroup.children)) {
      const childrenNested = parentGroup.children as string[][];
      const idx = childrenNested.findIndex((arr) => arr.includes(clip.clipId));
      layerRank = Math.max(0, idx);
    } else if (clip.type === "group") {
      // Place the group container just above its highest child layer
      const childrenNested = ((clip as GroupClipProps).children ||
        []) as string[][];
      const lastIdx = Math.max(0, childrenNested.length - 1);
      layerRank = lastIdx + 0.75;
    } else {
      // Non-group clip: compare against any groups on the same timeline
      const groupsOnTimeline = state.clips.filter(
        (c) => c.type === "group" && c.timelineId === clip.timelineId,
      ) as GroupClipProps[];
      if (groupsOnTimeline.length > 0) {
        const clipEnd = Math.max(0, clip.endFrame ?? 0);
        let best = 0;
        for (const g of groupsOnTimeline) {
          const lastIdx = Math.max(0, (g.children?.length ?? 1) - 1);
          const gEnd = Math.max(0, g.endFrame ?? 0);
          // If this clip ends before the group's end, it should be below only the last children array
          // Otherwise, it should be above all clips in the group
          const candidate = clipEnd <= gEnd ? lastIdx - 0.5 : lastIdx + 1;
          if (candidate > best) best = candidate;
        }
        layerRank = best;
      } else {
        layerRank = 0;
      }
    }

    const end = Math.max(0, clip.endFrame ?? 0);
    return timelineIndex * TIMELINE_WEIGHT + layerRank * LAYER_WEIGHT + end;
  },
  rescaleForFpsChange: (oldFps: number, newFps: number) => {
    set((state) => {
      const clips = (state.clips || []).map((clip) => {
        const updated: any = { ...clip };

        const sf = rescaleFrame(clip.startFrame, oldFps, newFps);
        const ef = rescaleFrame(clip.endFrame, oldFps, newFps);
        const ts = rescaleFrame((clip as any).trimStart, oldFps, newFps);
        const te = rescaleFrame((clip as any).trimEnd, oldFps, newFps);

        if (sf != null) updated.startFrame = sf;
        if (ef != null) updated.endFrame = ef;
        if (ts != null) (updated as any).trimStart = ts;
        if (te != null) (updated as any).trimEnd = te;

        if (
          (clip.type === "video" || clip.type === "image") &&
          Array.isArray((clip as any).preprocessors)
        ) {
          (updated as any).preprocessors = ((clip as any)
            .preprocessors as any[]).map((p) => {
            const up = { ...p };
            const ps = rescaleFrame(p.startFrame, oldFps, newFps);
            const pe = rescaleFrame(p.endFrame, oldFps, newFps);
            if (ps != null) (up as any).startFrame = ps;
            if (pe != null) (up as any).endFrame = pe;
            return up;
          });
        }

        return updated as AnyClipProps;
      });

      const resolvedClips = resolveOverlaps(clips);
      const clipDuration = calculateTotalClipDuration(resolvedClips);
      const prunedAssets = pruneAssetsForClips(state.assets, resolvedClips);

      return { clips: resolvedClips, clipDuration, assets: prunedAssets };
    });
  },
})));

export const getClipWidth = (
  startFrame: number,
  endFrame: number,
  timelineWidth: number,
  timelineDuration: number[],
) => {
  const [timelineStartFrame, timelineEndFrame] = timelineDuration;
  const percentage =
    (endFrame - startFrame) / (timelineEndFrame - timelineStartFrame);
  return timelineWidth * percentage;
};

export const getClipX = (
  startFrame: number | null,
  endFrame: number | null,
  timelineWidth: number | null,
  timelineDuration: number[],
) => {
  if (startFrame === null || endFrame === null || timelineWidth === null)
    return 0;

  const [timelineStartFrame, timelineEndFrame] = timelineDuration;
  const relativePosition =
    (startFrame - timelineStartFrame) / (timelineEndFrame - timelineStartFrame);
  return relativePosition * timelineWidth;
};

export const getTimelineX = (
  timelineWidth: number,
  timelinePadding: number,
  timelineDuration: number[],
) => {
  const [timelineStartFrame, timelineEndFrame] = timelineDuration;
  const timelineX =
    timelinePadding -
    (timelineWidth / (timelineEndFrame - timelineStartFrame)) *
      timelineStartFrame;
  return Math.max(0, timelineX);
};

export const getLocalFrame = (focusFrame: number, clip: AnyClipProps) => {
  const startFrame = clip.startFrame ?? 0;
  const trimStart = isFinite(clip.trimStart ?? 0) ? (clip.trimStart ?? 0) : 0;
  const realStartFrame = startFrame + trimStart;
  return focusFrame - realStartFrame;
};

export const getGlobalFrame = (focusFrame: number, clip: AnyClipProps) => {
  const startFrame = clip.startFrame ?? 0;
  const trimStart = isFinite(clip.trimStart ?? 0) ? (clip.trimStart ?? 0) : 0;
  const realStartFrame = startFrame + trimStart;
  return focusFrame + realStartFrame;
};

export const convertFrameToProjectSeconds = (frame: number) => {
  // get fps from controls store
  const fps = useControlsStore((s) => s.fps);
  return frame / fps;
};
