import type { StateCreator } from "zustand";
import { useClipStore } from "../clip";
import { useControlsStore } from "../control";
import {
  saveProjectJson,
  loadProjectJson,
  createProject,
  listProjects,
  getActiveProjectId,
  setActiveProjectId,
  listManifests,
  saveProjectCover,
  clearProjectCover,
} from "@app/preload";

import type {
  AnyClipProps,
  Asset,
  TimelineProps,
  MaskClipProps,
  MaskData,
  PreprocessorClipProps,
  ModelClipProps,
  ShapeTool,
  ViewTool,
  Point,
  ZoomLevel,
} from "../types";

import _ from "lodash";
import { getMediaInfo } from "../media/utils";
import { Preprocessor } from "../preprocessor/api";
import {
  getManifest,
  listModelTypes,
  type ManifestDocument,
  type ModelTypeInfo,
} from "../manifest/api";
import { useViewportStore } from "../viewport";
import { globalInputControlsStore } from "../inputControl";
import { fetchPreprocessorsList } from "../preprocessor/queries";
import { queryClient } from "../react-query/queryClient";
import { exportSequence, type ExportClip } from "@app/export-renderer";
import { prepareExportClipsForValue } from "../prepareExportClips";
import { sortClipsForStacking } from "../clipOrdering";
type JsonProjectSlice = {
  projects: Array<{
    id: number;
    name: string;
    fps: number;
    aspectRatio: { width: number; height: number; id: string };
    folderUuid: string;
    coverPath?: string;
    createdAt?: number;
    lastModified?: number;
  }>;
  addProject: (project: {
    id: number;
    name: string;
    fps: number;
    aspectRatio: { width: number; height: number; id: string };
    folderUuid: string;
  }) => void;
  updateProject?: (
    id: string | number,
    payload: Partial<{
      name: string;
      fps: number;
      aspectRatio: { width: number; height: number; id: string };
      folderUuid: string;
      coverPath?: string;
      createdAt?: number;
      lastModified?: number;
    }>,
  ) => void;
  activeProjectId: string | number | null;
  setProjects: (projects: {
    id: number;
    name: string;
    fps: number;
    aspectRatio: { width: number; height: number; id: string };
    folderUuid: string;
  }[]) => void;
  setActiveProjectId: (projectId: string | number) => void;
};

// Keep tick conversion aligned with main PersistenceModule/JSONPersistenceModule
const JSON_TICKS_PER_SECOND = 1_000_000n;
const JSON_POSITIVE_INFINITY_TICK = 1n;
const JSON_NEGATIVE_INFINITY_TICK = 2n;

const frameToTickForJson = (frame: unknown, fps: number): bigint => {
  if (frame == null) return 0n;
  if (typeof frame === "number" && !Number.isFinite(frame)) {
    return frame > 0 ? JSON_POSITIVE_INFINITY_TICK : JSON_NEGATIVE_INFINITY_TICK;
  }
  if (typeof frame === "string") {
    if (frame === "Infinity") return JSON_POSITIVE_INFINITY_TICK;
    if (frame === "-Infinity") return JSON_NEGATIVE_INFINITY_TICK;
  }
  try {
    const fpsInt = BigInt(Math.max(1, Math.round(Number(fps) || 0)));
    const f = BigInt(Math.round(Number(frame as number)));
    const numerator = f * JSON_TICKS_PER_SECOND;
    const tick = numerator / fpsInt;
    return tick;
  } catch {
    return 0n;
  }
};

const tickToFrameForJson = (tick: unknown, fps: number): number => {
  if (tick == null) return 0;
  try {
    const fpsInt = BigInt(Math.max(1, Math.round(Number(fps) || 0)));
    let t: bigint;
    if (typeof tick === "bigint") {
      t = tick;
    } else {
      const s = String(tick);
      t = BigInt(s);
    }
    if (t === JSON_POSITIVE_INFINITY_TICK) return Number.POSITIVE_INFINITY;
    if (t === JSON_NEGATIVE_INFINITY_TICK) return Number.NEGATIVE_INFINITY;
    const numerator = t * fpsInt;
    const frame = numerator / JSON_TICKS_PER_SECOND;
    return Number(frame);
  } catch {
    return 0;
  }
};

type ProjectJsonSnapshot = {
  version: string;
  meta: {
    id: string;
    name: string;
    createdAt: number;
    lastModified: number;
    projectId: number | null;
  };
  settings: {
    fps: number;
    defaultClipLength: number;
    aspectRatio?: { width: number; height: number; id: string };
  };
  editorState: {
    duration: number;
    zoomLevel: number;
    scrollX: number;
    scrollY: number;
    playheadPosition: number;
    snapGuideX: number | null;
    position: Point;
    timelineWindow: [number, number];
    scale: number;
    tool: ViewTool;
    shape: ShapeTool;
    viewportSize: { width: number; height: number };
    contentBounds: { x: number; y: number; width: number; height: number } | null;  
  };
  assets: Record<string, Asset>;
  timeline: {
    tracks: TimelineProps[];
    clips: TimelineClipJson[];
  };
  /**
   * Optional persisted per-clip inputControls state keyed by clipId,
   * then by inputId. This is populated by the renderer-side
   * InputControlsProvider and read here for JSON persistence.
   */
  inputControls?: {
    selectedRangeByInputId: Record<string, Record<string, [number, number]>>;
    selectedInputClipIdByInputId: Record<string, Record<string, string | null>>;
    totalTimelineFramesByInputId: Record<string, Record<string, number>>;
    timelineDurationByInputId: Record<string, Record<string, [number, number]>>;
    fpsByInputId: Record<string, Record<string, number>>;
    focusFrameByInputId: Record<string, Record<string, number>>;
    focusAnchorRatioByInputId: Record<string, Record<string, number>>;
    zoomLevelByInputId: Record<string, Record<string, ZoomLevel>>;
    isPlayingByInputId: Record<string, Record<string, boolean>>;
  }
  
  /**
   * Optional embedded manifests and preprocessors for this project.
   * These are persisted to disk by the main JSONPersistenceModule into
   * local/manifest and local/preprocessor, and re-hydrated on load so
   * we don't need to hit the API to resolve them.
   */
  manifests?: Record<string, ManifestDocument>;
  preprocessors?: Record<string, Preprocessor>;
};

type SerializedMaskForJson = Omit<MaskClipProps, "keyframes"> & {
  keyframes: Record<number, MaskData>;
};

type SerializedPreprocessorForJson = {
  id: string;
  assetId?: string;
  clipId?: string;
  startTick?: string;
  endTick?: string;
  values: Record<string, any>;
  status?: "running" | "complete" | "failed";
  activeJobId?: string;
  jobIds?: string[];
  preprocessorId: string;
};

type TimelineClipBaseForJson = Omit<
  AnyClipProps,
  "masks" | "preprocessors"
>;

type TimelineClipJson = TimelineClipBaseForJson & {
  startTick: string;
  endTick: string;
  trimStartTick: string;
  trimEndTick: string;
  masks?: SerializedMaskForJson[];
  preprocessors?: SerializedPreprocessorForJson[];
  manifestRef?: string;
  manifestVersion?: string;
};

const serializeMaskKeyframesForJson = (
  keyframes: MaskClipProps["keyframes"],
): Record<number, MaskData> => {
  if (!keyframes) return {};
  if (keyframes instanceof Map) {
    const out: Record<number, MaskData> = {};
    for (const [frame, data] of keyframes.entries()) {
      const frameNum = Number(frame);
      if (!Number.isFinite(frameNum)) continue;
      out[frameNum] = data;
    }
    return out;
  }
  const out: Record<number, MaskData> = {};
  for (const [k, v] of Object.entries(keyframes as Record<string, MaskData>)) {
    const frameNum = Number(k);
    if (!Number.isFinite(frameNum)) continue;
    out[frameNum] = v;
  }
  return out;
};

const serializeMasksForJson = (clip: AnyClipProps): SerializedMaskForJson[] => {
  if (clip.type !== "video" && clip.type !== "image") return [];
  const masks = (clip as any).masks as MaskClipProps[] | undefined;
  if (!Array.isArray(masks) || masks.length === 0) return [];
  return masks.map((m) => ({
    ...m,
    keyframes: serializeMaskKeyframesForJson(m.keyframes),
  }));
};

const serializePreprocessorsForJson = (
  clip: AnyClipProps,
  fps: number,
  localPreprocessors?: Record<string, Preprocessor>,
): SerializedPreprocessorForJson[] => {
  if (clip.type !== "video" && clip.type !== "image") return [];
  const preprocessors = (clip as any)
    .preprocessors as PreprocessorClipProps[] | undefined;
  if (!Array.isArray(preprocessors) || preprocessors.length === 0) return [];

  return preprocessors.map((p) => {
    const pre: any = p.preprocessor as any;
    const preId: string =
      (pre && typeof pre === "object" && (pre.id || pre.name)) || String(pre);
    if (
      localPreprocessors &&
      pre &&
      typeof pre === "object" &&
      (pre.id || pre.name)
    ) {
      const key = String(pre.id || pre.name);
      if (!localPreprocessors[key]) {
        localPreprocessors[key] = pre as Preprocessor;
      }
    }
    const startFrameNum =
      typeof p.startFrame === "number" && Number.isFinite(p.startFrame)
        ? p.startFrame
        : undefined;
    const endFrameNum =
      typeof p.endFrame === "number" && Number.isFinite(p.endFrame)
        ? p.endFrame
        : undefined;

    const startTick =
      startFrameNum != null
        ? frameToTickForJson(startFrameNum, fps).toString()
        : undefined;
    const endTick =
      endFrameNum != null
        ? frameToTickForJson(endFrameNum, fps).toString()
        : undefined;

    return {
      id: p.id,
      assetId: p.assetId,
      clipId: p.clipId,
      ...(startTick ? { startTick } : {}),
      ...(endTick ? { endTick } : {}),
      values: p.values ?? {},
      status: p.status,
      activeJobId: p.activeJobId,
      jobIds: p.jobIds,
      preprocessorId: preId,
    };
  });
};

const getManifestRefForModelClip = (clip: AnyClipProps): string | undefined => {
  if (clip.type !== "model") return undefined;
  const model = clip as ModelClipProps;
  const manifest = model.manifest;
  if (!manifest) return undefined;
  const id = manifest.id ?? manifest.name;
  if (id == null) return undefined;
  return String(id);
};

const getManifestVersionForModelClip = (
  clip: AnyClipProps,
): string | undefined => {
  if (clip.type !== "model") return undefined;
  const model = clip as ModelClipProps;
  const manifest = model.manifest;
  if (!manifest || typeof manifest !== "object") return undefined;
  const version =
    typeof (manifest as any).version === "string"
      ? ((manifest as any).version as string)
      : typeof (manifest as any)?.metadata?.version === "string"
        ? ((manifest as any).metadata.version as string)
        : undefined;
  return version && version.trim().length > 0 ? version.trim() : undefined;
};

/**
 * Versioned manifest cache key. This allows us to keep multiple manifest
 * revisions locally and ensures clips hydrate with the exact manifest they
 * were saved with (when available).
 */
const makeManifestCacheKey = (
  manifestId: string,
  manifestVersion?: string,
): string => {
  const id = String(manifestId ?? "").trim();
  const v = String(manifestVersion ?? "").trim();
  return id && v ? `${id}@@${v}` : id;
};

/**
 * Extract current UI input values from a manifest into a compact
 * map keyed by input id. This lets us persist per-clip values
 * without mutating the manifest JSON we save to disk.
 */
const extractModelInputValuesFromManifest = (
  manifest: ManifestDocument | undefined,
):
  | {
      values: Record<string, any>;
      typesById: Record<string, string>;
    }
  | undefined => {
  try {
    if (!manifest) return undefined;
    const anyManifest: any = manifest as any;
    const ui =
      (anyManifest.spec && anyManifest.spec.ui) || anyManifest.ui || undefined;
    if (!ui || !Array.isArray(ui.inputs)) return undefined;

    const values: Record<string, any> = {};
    const typesById: Record<string, string> = {};
    for (const inp of ui.inputs as any[]) {
      if (!inp || typeof inp !== "object") continue;
      const id = typeof inp.id === "string" ? inp.id : undefined;
      if (!id) continue;
      typesById[id] = String((inp as any)?.type ?? "");
      if (Object.prototype.hasOwnProperty.call(inp, "value")) {
        values[id] = (inp as any).value;
      }
    }

    return Object.keys(values).length > 0 ? { values, typesById } : undefined;
  } catch {
    return undefined;
  }
};

/**
 * Internal shape used when persisting model input values that reference
 * existing timeline clips. This keeps the JSON snapshot compact and
 * avoids duplicating full clip state inside manifest UI values.
 */
type ModelInputClipRef = {
  __kind: "timelineClipRef";
  clipId: string;
};

const isClipLikeModelInputValue = (value: any): value is { clipId: string } => {

  return (
    value &&
    typeof value === "object" &&
    typeof (value as any).clipId === "string"
  );
};

/**
 * Given a map of model input values and the current set of timeline clips,
 * replace any clip-shaped values whose clipId exists on the timeline with
 * lightweight references. This lets us rehydrate them later by looking
 * up the live clip instead of persisting a full copy here.
 */
const encodeModelInputClipRefsForJson = (
  modelInputValues: Record<string, any> | undefined,
  inputTypesById: Record<string, string> | undefined,
  allClips: AnyClipProps[],
): Record<string, any> | undefined => {
  if (!modelInputValues) return modelInputValues;
  if (!Array.isArray(allClips) || allClips.length === 0) {
    return modelInputValues;
  }

  const clipIdsOnTimeline = new Set(
    allClips
      .map((c) => {
        const id = (c as any).clipId;
        return typeof id === "string" ? id : id != null ? String(id) : "";
      })
      .filter((id) => id !== ""),
  );


  if (clipIdsOnTimeline.size === 0) return modelInputValues;

  let changed = false;
  const out: Record<string, any> = {};

  const shouldTryParseForRef = (inputType: unknown, raw: any): boolean => {
    if (typeof raw !== "string") return false;
    const s = raw.trim();
    if (!s || (!s.startsWith("{") && !s.startsWith("["))) return false;
    const t = String(inputType ?? "").toLowerCase();
    // Only media-like inputs intentionally store JSON payloads that might contain clipId refs.
    if (t.startsWith("image")) return true;
    if (t.startsWith("video")) return true;
    if (t.startsWith("audio")) return true;
    if (t === "image_list") return true;
    return false;
  };

  const encodeValue = (raw: any, inputType: unknown): any => {
    if (!shouldTryParseForRef(inputType, raw)) return raw;
    try {
      const parsed = JSON.parse(raw);
      if (isClipLikeModelInputValue(parsed)) {
        const clipId = String((parsed as any).clipId);
        if (clipIdsOnTimeline.has(clipId)) {
          changed = true;
          const ref: ModelInputClipRef = {
            __kind: "timelineClipRef",
            clipId,
          };
          return ref;
        }
      }
    } catch {
      // ignore json parse errors
    }
    // Preserve the original raw string unless we successfully encoded a ref.
    return raw;
  };

  for (const [key, raw] of Object.entries(modelInputValues)) {
    const inputType = inputTypesById ? inputTypesById[key] : undefined;
    if (Array.isArray(raw)) {
      out[key] = (raw as any[]).map((item) => encodeValue(item, inputType));
    } else {
      out[key] = encodeValue(raw, inputType);
    }
  }

  return changed ? out : modelInputValues;
};

/**
 * Create a manifest clone suitable for on-disk storage by stripping
 * any UI input `value` fields. This keeps the saved manifest JSON
 * free of user-specific selections.
 */
const stripModelInputValuesFromManifest = (
  manifest: ManifestDocument,
): ManifestDocument => {
  try {
    const clone: any = JSON.parse(JSON.stringify(manifest));
    const uiTargets: any[] = [];
    if (clone.spec && clone.spec.ui && Array.isArray(clone.spec.ui.inputs)) {
      uiTargets.push(clone.spec.ui);
    }
    if (clone.ui && Array.isArray(clone.ui.inputs)) {
      uiTargets.push(clone.ui);
    }

    for (const ui of uiTargets) {
      if (!ui || !Array.isArray(ui.inputs)) continue;
      ui.inputs = (ui.inputs as any[]).map((inp: any) => {
        if (!inp || typeof inp !== "object") return inp;
        const { value, ...rest } = inp;
        return rest;
      });
    }

    return clone as ManifestDocument;
  } catch {
    return manifest;
  }
};

const buildProjectJsonSnapshot = (
  projectState: JsonProjectSlice,
): {
  projectId: number | null;
  snapshot: ProjectJsonSnapshot;
} | null => {
  try {
    const { projects, activeProjectId } = projectState;

    // If the clip/controls stores have not been fully initialized yet (for example
    // due to module loading order or circular imports), bail out gracefully
    // instead of throwing. We'll simply skip this save attempt and try again
    // on the next scheduled change.
    const clipState = useClipStore.getState();
    const controlsState = useControlsStore.getState();
    const viewportState = useViewportStore.getState();
    const inputControlsState = globalInputControlsStore.getState();

    if (!clipState || !controlsState || !viewportState) {
      console.warn(
        "jsonPersistence: clip/controls/viewport stores not ready yet, skipping JSON snapshot",
      );
      return null;
    }

    const activeProject =
      projects.find((p) => p.id === activeProjectId) ?? null;

    if (!activeProject) {
      return null;
    }

    const projectId = activeProject.id;
    const fps = controlsState.fps || activeProject.fps || 24;

    const allClips = Array.isArray(clipState.clips)
      ? (clipState.clips as AnyClipProps[])
      : ([] as AnyClipProps[]);
    const clipsForJson: ProjectJsonSnapshot["timeline"]["clips"] = [];
    const localManifests: Record<string, ManifestDocument> = {};
    const localPreprocessors: Record<string, Preprocessor> = {};

    for (const orig of allClips) {
      if (!orig) continue;

      const clipId = String((orig as any).clipId);
      const timelineId = String((orig as any).timelineId ?? "");

      const startFrameRaw = (orig as any).startFrame;
      const endFrameRaw = (orig as any).endFrame;
      const trimStartRaw = (orig as any).trimStart;
      const trimEndRaw = (orig as any).trimEnd;
      const clipPaddingRaw = (orig as any).clipPadding;

      const startFrame =
        typeof startFrameRaw === "number" && Number.isFinite(startFrameRaw)
          ? startFrameRaw
          : 0;
      const endFrame =
        typeof endFrameRaw === "number" && Number.isFinite(endFrameRaw)
          ? endFrameRaw
          : 0;
      const trimStart =
        typeof trimStartRaw === "number" && Number.isFinite(trimStartRaw)
          ? trimStartRaw
          : 0;
      const trimEnd =
        typeof trimEndRaw === "number" && Number.isFinite(trimEndRaw)
          ? trimEndRaw
          : 0;
      const clipPadding =
        typeof clipPaddingRaw === "number" && Number.isFinite(clipPaddingRaw)
          ? clipPaddingRaw
          : 24;

      const startTick = frameToTickForJson(startFrame, fps);
      const endTick = frameToTickForJson(endFrame, fps);
      const trimStartTick = frameToTickForJson(trimStart, fps);
      const trimEndTick = frameToTickForJson(trimEnd, fps);

      let masks: SerializedMaskForJson[] | undefined;
      let preprocessors: SerializedPreprocessorForJson[] | undefined;
      let manifestRef: string | undefined;

      const origClip = orig as AnyClipProps;

      // Capture full manifest documents for model clips so the main
      // JSONPersistenceModule can persist them locally.
      if (origClip.type === "model") {
        const model = origClip as ModelClipProps;
        const manifest = model.manifest;
        if (manifest) {
          const id = manifest.id ?? manifest.name;
          if (id != null) {
            const key = String(id);
            const manifestVersion = getManifestVersionForModelClip(origClip);
            const versionedKey = makeManifestCacheKey(key, manifestVersion);
            if (!localManifests[versionedKey]) {
              // Store a sanitized manifest without UI input values so that
              // the on-disk manifest JSON remains generic and not
              // user-specific. Per-clip values are stored on the clip
              // itself (see modelInputValues below).
              localManifests[versionedKey] =
                stripModelInputValuesFromManifest(manifest as ManifestDocument);
            }
          }
        }
      }

      const serializedMasks = serializeMasksForJson(origClip);
      if (serializedMasks.length > 0) {
        masks = serializedMasks;
      }
      const serializedPres = serializePreprocessorsForJson(
        origClip,
        fps,
        localPreprocessors,
      );
      if (serializedPres.length > 0) {
        preprocessors = serializedPres;
      }
      manifestRef = getManifestRefForModelClip(origClip);
      const manifestVersion = getManifestVersionForModelClip(origClip);

      // Normalize the clip state we persist and strip large/derived fields
      let clipForJson: any = {
        ...(origClip as any),
        clipId,
        timelineId,
        clipPadding,
      };

      // Do not store frame-based fields in JSON; we derive them from ticks on load.
      delete clipForJson.startFrame;
      delete clipForJson.endFrame;
      delete clipForJson.trimStart;
      delete clipForJson.trimEnd;

      // These are stored as JSON-specific shapes below
      delete clipForJson.masks;
      delete clipForJson.preprocessors;

      // For model clips, persist a lightweight map of UI input values
      // on the clip itself instead of embedding them into the manifest
      // that is written to disk. This keeps manifest JSON clean while
      // still allowing full restoration on hydrate.
      if (origClip.type === "model") {
        const model = origClip as ModelClipProps;
        const extracted = extractModelInputValuesFromManifest(
          model.manifest,
        );
        const encodedValues = encodeModelInputClipRefsForJson(
          extracted?.values,
          extracted?.typesById,
          allClips,
        );
        if (encodedValues && Object.keys(encodedValues).length > 0) {
          clipForJson.modelInputValues = encodedValues;
        } else {
          // Ensure we do not persist stale values from previous loads
          delete clipForJson.modelInputValues;
        }
      }

      if (
        clipForJson &&
        clipForJson.type === "model" &&
        Object.prototype.hasOwnProperty.call(clipForJson, "manifest")
      ) {
        const { manifest, ...rest } = clipForJson;
        clipForJson = rest;
      }

      const jsonClip: TimelineClipJson = {
        ...(clipForJson as TimelineClipBaseForJson),
        startTick: startTick.toString(),
        endTick: endTick.toString(),
        trimStartTick: trimStartTick.toString(),
        trimEndTick: trimEndTick.toString(),
        ...(masks ? { masks } : {}),
        ...(preprocessors ? { preprocessors } : {}),
        ...(manifestRef ? { manifestRef } : {}),
        ...(manifestRef && manifestVersion ? { manifestVersion } : {}),
      };

      clipsForJson.push(jsonClip);
    }

    // Normalize assets from the clip store into a Record<string, Asset> keyed by id.
    const assetsFromStore = clipState.assets as
      | Record<string, Asset>
      | Asset[]
      | undefined;

    const assets: Record<string, Asset> = {};
    if (Array.isArray(assetsFromStore)) {
      for (const asset of assetsFromStore as Asset[]) {
        if (!asset || !asset.id || asset.modelInputAsset) continue;
        assets[asset.id] = asset;
      }
    } else if (assetsFromStore && typeof assetsFromStore === "object") {
      for (const [, asset] of Object.entries(assetsFromStore)) {
        if (!asset || !asset.id) continue;
        assets[asset.id] = asset as Asset;
      }
    }

    // Build a set of clipIds that are actually present on the timeline so that
    // we only persist per-input inputControls state for real timeline clips.
    const activeClipIds = new Set<string>();
    for (const clip of clipsForJson) {
      const id = String((clip as any).clipId ?? "");
      if (id) {
        activeClipIds.add(id);
      }
    }

    const filterNestedByClip = <T,>(
      nested: Record<string, Record<string, T>> | undefined | null,
    ): Record<string, Record<string, T>> => {
      const result: Record<string, Record<string, T>> = {};
      if (!nested) return result;

      for (const [inputId, byClip] of Object.entries(nested)) {
        if (!byClip || typeof byClip !== "object") continue;

        const filteredByClip: Record<string, T> = {};
        for (const [clipId, value] of Object.entries(
          byClip as Record<string, T>,
        )) {
          if (activeClipIds.has(clipId)) {
            filteredByClip[clipId] = value;
          }
        }

        if (Object.keys(filteredByClip).length > 0) {
          result[inputId] = filteredByClip;
        }
      }

      return result;
    };

    const filteredSelectedRangeByInputId = filterNestedByClip(
      inputControlsState.selectedRangeByInputId,
    );
    const filteredSelectedInputClipIdByInputId = filterNestedByClip(
      inputControlsState.selectedInputClipIdByInputId,
    );
    const filteredTotalTimelineFramesByInputId = filterNestedByClip(
      inputControlsState.totalTimelineFramesByInputId,
    );
    const filteredTimelineDurationByInputId = filterNestedByClip(
      inputControlsState.timelineDurationByInputId,
    );
    const filteredFpsByInputId = filterNestedByClip(
      inputControlsState.fpsByInputId,
    );
    const filteredFocusFrameByInputId = filterNestedByClip(
      inputControlsState.focusFrameByInputId,
    );
    const filteredFocusAnchorRatioByInputId = filterNestedByClip(
      inputControlsState.focusAnchorRatioByInputId,
    );
    const filteredZoomLevelByInputId = filterNestedByClip(
      inputControlsState.zoomLevelByInputId,
    );
    const filteredIsPlayingByInputId = filterNestedByClip(
      inputControlsState.isPlayingByInputId,
    );

    const now = Date.now();

    const snapshot: ProjectJsonSnapshot = {
      version: "1.0.0",
      meta: {
        id:
          (activeProject as any)?.folderUuid ||
          `project-${projectId}`,
        name: activeProject?.name ?? "Untitled Project",
        createdAt: now,
        lastModified: now,
        projectId,
      },
      settings: {
        fps,
        aspectRatio: viewportState?.aspectRatio,
        defaultClipLength: controlsState.defaultClipLength,
      },
      editorState: {
        duration: Number(clipState.clipDuration || 0),
        zoomLevel: Number(controlsState.zoomLevel || 1),
        scrollX: 0,
        scrollY: 0,
        playheadPosition: Number(controlsState.focusFrame || 0),
        tool: viewportState.tool,
        shape: viewportState.shape, 
        scale: viewportState.scale,
        position: viewportState.position,
        viewportSize: viewportState.viewportSize,
        contentBounds: viewportState.contentBounds,
        snapGuideX:
          typeof (clipState as any).snapGuideX === "number"
            ? (clipState as any).snapGuideX
            : null,
        timelineWindow: Array.isArray(controlsState.timelineDuration)
          ? (controlsState.timelineDuration as [number, number])
          : [0, controlsState.totalTimelineFrames || 0],
      },
      assets,
      timeline: {
        tracks: Array.isArray(clipState.timelines)
          ? (clipState.timelines as TimelineProps[])
          : [],
        clips: clipsForJson,
      },
      manifests: localManifests,
      preprocessors: localPreprocessors,
      inputControls: {
        selectedRangeByInputId: filteredSelectedRangeByInputId,
        selectedInputClipIdByInputId: filteredSelectedInputClipIdByInputId,
        totalTimelineFramesByInputId: filteredTotalTimelineFramesByInputId,
        timelineDurationByInputId: filteredTimelineDurationByInputId,
        fpsByInputId: filteredFpsByInputId,
        focusFrameByInputId: filteredFocusFrameByInputId,
        focusAnchorRatioByInputId: filteredFocusAnchorRatioByInputId,
        zoomLevelByInputId: filteredZoomLevelByInputId,
        isPlayingByInputId: filteredIsPlayingByInputId,
      },
    };

    
    // Attach any captured manifests/preprocessors so the main process
    // can persist them to local/{manifest,preprocessor} and we can later
    // hydrate from those files without hitting the API again.
    if (Object.keys(localManifests).length > 0) {
      snapshot.manifests = localManifests;
    }
    if (Object.keys(localPreprocessors).length > 0) {
      snapshot.preprocessors = localPreprocessors;
    }

    return { projectId, snapshot };
  } catch (err) {
    console.error("jsonPersistence: failed to build project snapshot", err);
    return null;
  }
};

let jsonSyncTimeout: ReturnType<typeof setTimeout> | null = null;
let jsonSyncGeneration = 0;
let subscriptionsInitialized = false;
let jsonProjectLoadGeneration = 0;

// Track JSON hydration lifecycle so that autosave does not run while a project
// is being switched / hydrated, which can otherwise cause one project's state
// to be written under another project's id.
let isHydratingFromJson = false;
let currentHydratedProjectId: string | number | null = null;

const normalizeProjectId = (
  raw: string | number | null | undefined,
): string | number | null => {
  if (raw == null) return null;
  if (typeof raw === "number") return Number.isFinite(raw) ? raw : null;
  const s = String(raw).trim();
  if (!s) return null;
  const n = Number(s);
  if (Number.isInteger(n) && n > 0) return n;
  return s;
};

const cancelPendingJsonProjectSave = (): void => {
  // Invalidate any pending debounced save so it cannot fire after a project
  // switch/hydration and write an "empty" snapshot over another project.
  jsonSyncGeneration += 1;
  if (jsonSyncTimeout) {
    clearTimeout(jsonSyncTimeout);
    jsonSyncTimeout = null;
  }
};

const isLauncherWindow = (): boolean => {
  try {
    return typeof window !== "undefined" && window.location?.hash === "#launcher";
  } catch {
    return false;
  }
};

// Robust change detection for Zustand subscriptions.
// We avoid relying on `prev` because some store updates may mutate data in-place,
// which can make prev/current comparisons unreliable.
let lastClipSignature: string | null = null;
let lastTimelineSignature: string | null = null;
let lastAssetSignature: string | null = null;

const safeStringify = (value: unknown): string => {
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    // Fallback: force "changed" on the next compare if serialization fails.
    return `__unserializable__:${Date.now()}`;
  }
};

const computeClipSignature = (clips: unknown): string => {
  if (!Array.isArray(clips)) return "[]";
  // Only include fields that impact the visual first-frame result.
  const reduced = (clips as any[]).map((c) => ({
    clipId: c?.clipId,
    type: c?.type,
    timelineId: c?.timelineId,
    groupId: c?.groupId,
    hidden: c?.hidden,
    assetId: c?.assetId,
    startFrame: c?.startFrame,
    endFrame: c?.endFrame,
    trimStart: c?.trimStart,
    trimEnd: c?.trimEnd,
    transform: c?.transform,
    originalTransform: c?.originalTransform,
    children: c?.type === "group" ? c?.children : undefined,
  }));
  return safeStringify(reduced);
};

const computeTimelineSignature = (timelines: unknown): string => {
  if (!Array.isArray(timelines)) return "[]";
  const reduced = (timelines as any[]).map((t) => ({
    timelineId: t?.timelineId,
    hidden: t?.hidden,
    timelineY: t?.timelineY,
    muted: t?.muted,
  }));
  return safeStringify(reduced);
};

const computeAssetSignature = (assets: unknown): string => {
  if (!assets || typeof assets !== "object") return "{}";
  const rec = assets as Record<string, any>;
  const reduced = Object.keys(rec)
    .sort()
    .map((id) => {
      const a = rec[id];
      return [id, a?.type, a?.path];
    });
  return safeStringify(reduced);
};

// Cover generation debounce (export first frame as project cover when only clips change).
let coverUpdateGeneration = 0;
const debouncedGenerateProjectCover = _.debounce(
  (gen: number, getProjectState: () => JsonProjectSlice) => {
    void (async () => {
      try {
        // Never generate or persist covers from the launcher window.
        // Launcher doesn't have authoritative clip state and must not write project artifacts.
        if (isLauncherWindow()) return;
        // If another request superseded this one, bail.
        if (gen !== coverUpdateGeneration) return;
        if (isHydratingFromJson) return;

        const projectState = getProjectState();
        const activeProjectId = projectState?.activeProjectId;
        if (activeProjectId == null) return;

        // Avoid writing covers for a project that isn't the one currently hydrated.
        if (
          currentHydratedProjectId != null &&
          activeProjectId !== currentHydratedProjectId
        ) {
          return;
        }

        const clipState = useClipStore.getState();
        const controlsState = useControlsStore.getState();
        const viewportState = useViewportStore.getState();

        const getAssetById = (clipState)?.getAssetById;
        const getClipsForGroup = (clipState)?.getClipsForGroup;
        const getClipsByType = (clipState)?.getClipsByType;
        const getClipPositionScore = (clipState)?.getClipPositionScore;
        if (
          typeof getAssetById !== "function" ||
          typeof getClipsForGroup !== "function" ||
          typeof getClipsByType !== "function" ||
          typeof getClipPositionScore !== "function"
        ) {
          return;
        }

        const fps = Number(controlsState?.fps || 0) || 24;
        const aspect =
          viewportState?.aspectRatio ||
          (projectState.projects?.find((p) => p.id === activeProjectId)
            ?.aspectRatio) ||
          ({ width: 16, height: 9 });

        const rawRatio = Number(aspect?.width) / Number(aspect?.height);
        const ratio = Number.isFinite(rawRatio) && rawRatio > 0 ? rawRatio : 16 / 9;

        // Keep cover reasonably sized.
        const heightPx = 720;
        const widthPx = Math.max(1, Math.round(heightPx * ratio));

        const clips = Array.isArray((clipState as any)?.clips)
          ? ((clipState as any).clips as AnyClipProps[])
          : ([] as AnyClipProps[]);
        const timelines = Array.isArray((clipState as any)?.timelines)
          ? ((clipState as any).timelines as TimelineProps[])
          : ([] as TimelineProps[]);

        if (clips.length === 0) {
          // If the timeline is empty, clear the project cover.
          try {
            await clearProjectCover(activeProjectId);
          } catch {
            // ignore; best-effort
          }
          projectState?.updateProject?.(activeProjectId, {
            coverPath: undefined,
            lastModified: Date.now(),
          });
          return;
        }

        // Mirror export prep used by Topbar: stacking + ignore hidden timelines + expand groups/filters.
        let contentClips = sortClipsForStacking(clips, timelines);
        contentClips = contentClips.filter((clip) => {
          const timeline = timelines.find((t) => t.timelineId === clip.timelineId);
          return !timeline?.hidden;
        });

        const preparedClips: ExportClip[] = [];
        for (const clip of contentClips) {
          if (!clip) continue;
          if ((clip as any).type === "filter") continue;
          const prepared = prepareExportClipsForValue(
            clip as AnyClipProps,
            {
              aspectRatio: { width: Number(aspect.width), height: Number(aspect.height) },
              getAssetById,
              getClipsForGroup,
              getClipsByType,
              getClipPositionScore,
              timelines,
            },
            {
              clearMasks: false,
              applyCentering: false,
              dimensionsFrom: "aspect",
              target: { width: widthPx, height: heightPx },
            },
          );
          preparedClips.push(...(prepared.exportClips || []));
        }

        if (preparedClips.length === 0) return;

        // Export first frame (frame 0) as PNG, then transcode to JPEG for saveProjectCover.
        const result = await exportSequence({
          mode: "image",
          clips: preparedClips,
          fps,
          width: widthPx,
          height: heightPx,
          imageFrame: 0,
          backgroundColor: "#000000",
        } as any);

        if (!(result instanceof Blob)) return;

        let jpegBlob: Blob | null = null;
        try {
          const bitmap = await createImageBitmap(result);
          const canvas = document.createElement("canvas");
          canvas.width = widthPx;
          canvas.height = heightPx;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(bitmap, 0, 0, widthPx, heightPx);
          jpegBlob = await new Promise<Blob | null>((resolve) =>
            canvas.toBlob(resolve, "image/jpeg", 0.85),
          );
        } catch {
          jpegBlob = null;
        }
        if (!jpegBlob) return;

        const buffer = await jpegBlob.arrayBuffer();
        const res = await saveProjectCover(activeProjectId, new Uint8Array(buffer));

        if (res?.success && res.data?.path) {
          // Best-effort: update coverPath in the projects store so Launcher/Topbar can render it.
          const anyState = projectState;
          if (typeof anyState.updateProject === "function") {
            anyState.updateProject(activeProjectId, {
              coverPath: res.data.path,
              lastModified: Date.now(),
            });
          }
        }
      } catch (err) {
        console.error("jsonPersistence: failed to generate project cover", err);
      }
    })();
  },
  1000,
);

const hydrateStoresFromProjectJson = async (
  _projectId: string | number,
  doc: ProjectJsonSnapshot,
) => {
  // While we're hydrating from disk, disable autosave and remember which
  // project id the current in-memory stores correspond to once complete.
  isHydratingFromJson = true;
  try {
    try {
      // Hydrate the global inputControls store from any persisted inputControls
      // snapshot in the project JSON. The snapshot is keyed by property name
      // (e.g. selectedRangeByInputId), then by inputId, then by clipId.
      if (doc.inputControls && typeof doc.inputControls === "object") {
        try {
          const snapshot = doc.inputControls;
          globalInputControlsStore.setState((prev: any) => {
            const next = { ...prev };

            // Helper to merge nested { [inputId]: { [clipId]: T } } structures
            const mergeNested = <T,>(
              existing: Record<string, Record<string, T>> | undefined,
              incoming: Record<string, Record<string, T>> | undefined,
            ): Record<string, Record<string, T>> => {
              const result: Record<string, Record<string, T>> = { ...(existing || {}) };
              if (!incoming) return result;
              for (const [inputId, byClip] of Object.entries(incoming)) {
                if (!byClip || typeof byClip !== "object") continue;
                result[inputId] = { ...(result[inputId] || {}), ...byClip };
              }
              return result;
            };

            next.totalTimelineFramesByInputId = mergeNested(
              prev.totalTimelineFramesByInputId,
              snapshot.totalTimelineFramesByInputId as Record<string, Record<string, number>> | undefined,
            );
            next.timelineDurationByInputId = mergeNested(
              prev.timelineDurationByInputId,
              snapshot.timelineDurationByInputId as Record<string, Record<string, [number, number]>> | undefined,
            );
            next.fpsByInputId = mergeNested(
              prev.fpsByInputId,
              snapshot.fpsByInputId as Record<string, Record<string, number>> | undefined,
            );
            next.focusFrameByInputId = mergeNested(
              prev.focusFrameByInputId,
              snapshot.focusFrameByInputId as Record<string, Record<string, number>> | undefined,
            );
            next.focusAnchorRatioByInputId = mergeNested(
              prev.focusAnchorRatioByInputId,
              snapshot.focusAnchorRatioByInputId as Record<string, Record<string, number>> | undefined,
            );
            next.selectedInputClipIdByInputId = mergeNested(
              prev.selectedInputClipIdByInputId,
              snapshot.selectedInputClipIdByInputId as Record<string, Record<string, string | null>> | undefined,
            );
            next.selectedRangeByInputId = mergeNested(
              prev.selectedRangeByInputId,
              snapshot.selectedRangeByInputId as Record<string, Record<string, [number, number]>> | undefined,
            );
            next.zoomLevelByInputId = mergeNested(
              prev.zoomLevelByInputId,
              snapshot.zoomLevelByInputId as Record<string, Record<string, ZoomLevel>> | undefined,
            );
            // Don't restore isPlayingByInputId - always start paused

            return next;
          });
        } catch {
          // best-effort; skip inputControls hydration if anything goes wrong
        }
      }

      const timelineTracks = Array.isArray(doc.timeline?.tracks)
        ? (doc.timeline.tracks as TimelineProps[])
        : [];

      const clipsJson = Array.isArray(doc.timeline?.clips)
        ? (doc.timeline.clips as TimelineClipJson[])
        : [];

      const clips: AnyClipProps[] = [];

      const fpsForJson =
        typeof doc.settings?.fps === "number" &&
        Number.isFinite(doc.settings.fps)
          ? doc.settings.fps
          : 24;

      const defaultClipLength =
        typeof doc.settings?.defaultClipLength === "number" &&
        Number.isFinite(doc.settings.defaultClipLength)
          ? doc.settings.defaultClipLength
          : 5;

      const manifestsMap =
        doc && typeof (doc as any).manifests === "object"
          ? ((doc as any).manifests as Record<string, ManifestDocument>)
          : undefined;
      const preprocessorsMap =
        doc && typeof (doc as any).preprocessors === "object"
          ? ((doc as any).preprocessors as Record<string, Preprocessor>)
          : undefined;

      for (const c of clipsJson) {
        if (!c) continue;

        let videoModelWithSrc = false; 
        if (c.type === "model" && (c as any).assetId && (c as any).assetId !== "") {
         videoModelWithSrc = true;
        }

        const baseStartFrame =
          (c as any).startTick != null
            ? tickToFrameForJson((c as any).startTick, fpsForJson)
            : Number((c as any).startFrame ?? 0);
        const baseEndFrame =
          (c as any).endTick != null
            ? tickToFrameForJson((c as any).endTick, fpsForJson)
            : Number((c as any).endFrame ?? 0);
        const baseTrimStart =
          c.type !== "audio" && c.type !== "video" && c.type !== "group" && !videoModelWithSrc
            ? Infinity
            : tickToFrameForJson((c as any).trimStartTick, fpsForJson);
        const baseTrimEnd =
          c.type !== "audio" && c.type !== "video" && c.type !== "group" && !videoModelWithSrc
            ? -Infinity
            : tickToFrameForJson((c as any).trimEndTick, fpsForJson);

        const base: Partial<AnyClipProps> = {
          clipId: String((c as any).clipId),
          timelineId: String((c as any).timelineId ?? ""),
          type: (c as any).type,
          groupId: ((c as any).groupId ?? undefined) as string | undefined,
          startFrame: baseStartFrame,
          endFrame: baseEndFrame,
          trimStart: baseTrimStart,
          trimEnd: baseTrimEnd,
          clipPadding: Number(
            typeof (c as any).clipPadding === "number" &&
              Number.isFinite((c as any).clipPadding)
              ? (c as any).clipPadding
              : 24,
          ),
          transform: (c as any).transform,
          originalTransform: (c as any).originalTransform,
        };

        const hasLegacyProps =
          typeof (c as any).props === "object" && (c as any).props !== null;

        let merged: AnyClipProps;

        if (hasLegacyProps) {
          // Backwards compatibility for older JSON snapshots that used a separate
          // props bag alongside base clip fields.
          const props = ((c as any).props ?? {}) as Record<string, any>;
          merged = { ...(props as any), ...(base as any) } as AnyClipProps;
        } else {
          // New format: the JSON clip already contains the full clip state
          // inline, so we take everything from the JSON entry, drop
          // JSON-specific metadata fields, and overlay normalized base fields.
          const clone: Record<string, any> = { ...(c as any) };
          delete clone.startTick;
          delete clone.endTick;
          delete clone.trimStartTick;
          delete clone.trimEndTick;
          merged = { ...clone, ...(base as any) } as AnyClipProps;
        }

        const mergedAny = merged as any;

        if (
          (mergedAny.type === "video" || mergedAny.type === "image") &&
          Array.isArray((c as any).masks)
        ) {
          mergedAny.masks = (c as any).masks.map((m: SerializedMaskForJson) => ({
            ...m,
            keyframes: m.keyframes ?? {},
          }));
        }

        if (mergedAny.type === "model") {
          const manifestRef =
            (mergedAny as any).manifestRef ?? (c as any).manifestRef;
          const manifestId =
            manifestRef != null && manifestRef !== ""
              ? String(manifestRef)
              : undefined;
          const manifestVersion =
            typeof (c as any).manifestVersion === "string" &&
            (c as any).manifestVersion.trim().length > 0
              ? String((c as any).manifestVersion).trim()
              : undefined;
          const manifestCacheKey =
            manifestId ? makeManifestCacheKey(manifestId, manifestVersion) : "";

          let manifestFromDoc =
            manifestId && manifestsMap
              ? (manifestsMap[manifestCacheKey] ?? manifestsMap[manifestId])
              : undefined;

          // If we don't have an exact local match, try to refresh from the API.
          // This prevents hydration from using a stale local manifest that is
          // missing UI inputs, which can otherwise cause persisted input values
          // to be dropped during re-application.
          if (!manifestFromDoc && manifestId) {
            try {
              const res = await getManifest(manifestId);
              if (res?.success && res.data) {
                manifestFromDoc = res.data as ManifestDocument;
                if (manifestsMap) {
                  manifestsMap[manifestCacheKey || manifestId] = manifestFromDoc;
                }
              }
            } catch {
              // ignore; best-effort
            }
          }

          if (manifestFromDoc) {
            // Clone so that each clip gets its own manifest instance; this
            // allows per-clip UI values without leaking between clips.
            let manifestForClip: ManifestDocument | any;
            try {
              manifestForClip = JSON.parse(JSON.stringify(manifestFromDoc));
            } catch {
              manifestForClip = manifestFromDoc as any;
            }

            // Re-apply any persisted UI input values back onto the manifest.
            const modelInputValues =
              (c as any).modelInputValues ??
              (mergedAny as any).modelInputValues ??
              undefined;
            if (
              modelInputValues &&
              typeof modelInputValues === "object" &&
              Object.keys(modelInputValues).length > 0
            ) {
              const mfAny: any = manifestForClip as any;
              const ui =
                (mfAny.spec && mfAny.spec.ui) || mfAny.ui || undefined;
              if (ui && Array.isArray(ui.inputs)) {
                ui.inputs = ui.inputs.map((inp: any) => {
                  if (!inp || typeof inp !== "object") return inp;
                  const id = typeof inp.id === "string" ? inp.id : undefined;
                  if (!id) return inp;
                  if (
                    Object.prototype.hasOwnProperty.call(modelInputValues, id)
                  ) {
                    return {
                      ...inp,
                      value: (modelInputValues as any)[id],
                    };
                  }
                  return inp;
                });
              }
            }

            mergedAny.manifest = manifestForClip as ManifestDocument;
          }

          delete mergedAny.manifestRef;
        }

        if (
          (mergedAny.type === "video" || mergedAny.type === "image") &&
          Array.isArray((c as any).preprocessors)
        ) {
          const preprocessorClips: PreprocessorClipProps[] = [];
          for (const p of (c as any)
            .preprocessors as SerializedPreprocessorForJson[]) {
            const name = String(p.preprocessorId ?? "");
            const startFrameFromTick =
              p.startTick != null
                ? tickToFrameForJson(p.startTick, fpsForJson)
                : undefined;
            const endFrameFromTick =
              p.endTick != null
                ? tickToFrameForJson(p.endTick, fpsForJson)
                : undefined;

            const preprocessorData =
              name && preprocessorsMap ? preprocessorsMap[name] : undefined;
            if (!preprocessorData) continue;

            const pre: PreprocessorClipProps = {
              id: String(p.id),
              assetId: p.assetId ?? undefined,
              clipId: String(p.clipId ?? (c as any).clipId),
              preprocessor: preprocessorData as Preprocessor,
              startFrame: startFrameFromTick,
              endFrame: endFrameFromTick,
              values: p.values ?? {},
              status: p.status ?? undefined,
              activeJobId: p.activeJobId ?? undefined,
              jobIds: p.jobIds ?? undefined,
            };
            preprocessorClips.push(pre);
          }
          if (preprocessorClips.length > 0) {
            mergedAny.preprocessors = preprocessorClips;
          }
        }

        clips.push(merged);
      }

      /**
       * After all clips have been constructed, resolve any model input values
       * that were persisted as timeline clip references back into full clip
       * objects. This lets model UI inputs refer to live timeline clips without
       * duplicating clip state inside the JSON snapshot.
       */
      const decodeModelInputClipRefsFromJson = (
        modelInputValues: Record<string, any> | undefined,
        clipsById: Record<string, AnyClipProps>,
      ): Record<string, any> | undefined => {
        if (!modelInputValues) return modelInputValues;

        let changed = false;
        const out: Record<string, any> = {};

        const decodeValue = (raw: any): any => {
          if (
            raw &&
            typeof raw === "object" &&
            (raw as any).__kind === "timelineClipRef" &&
            typeof (raw as any).clipId === "string"
          ) {
            const clipId = String((raw as any).clipId);
            const clip = clipsById[clipId];
            if (clip) {
              changed = true;
              // When hydrating model input values, convert any resolved
              // timeline clip references back into the JSON string shape
              // that model inputs expect instead of returning a live clip
              // object. This keeps `modelInputValues` consistent with the
              // original manifest UI values (which are strings).
              try {
                return JSON.stringify({ clipId });
              } catch {
                // If JSON serialization fails for some reason, fall back
                // to the original raw value rather than a clip object.
                return raw;
              }
            }
          }
          return raw;
        };

        for (const [key, raw] of Object.entries(modelInputValues)) {
          if (Array.isArray(raw)) {
            out[key] = (raw as any[]).map((item) => decodeValue(item));
          } else {
            out[key] = decodeValue(raw);
          }
        }

        return changed ? out : modelInputValues;
      };

      if (clips.length > 0) {
        const clipsById: Record<string, AnyClipProps> = {};
        for (const clip of clips) {
          const id = (clip as any).clipId;
          if (id == null) continue;
          const key = typeof id === "string" ? id : String(id);
          if (!key) continue;
          clipsById[key] = clip;
        }

        for (const clip of clips) {
          const anyClip = clip as any;
          if (anyClip.type !== "model") continue;

          const modelClip = anyClip as ModelClipProps;
          const rawValues = modelClip.modelInputValues;
          if (
            !rawValues ||
            typeof rawValues !== "object" ||
            Object.keys(rawValues).length === 0
          ) {
            continue;
          }

          const resolvedValues = decodeModelInputClipRefsFromJson(
            rawValues as Record<string, any>,
            clipsById,
          );
          if (!resolvedValues) continue;

          modelClip.modelInputValues = resolvedValues;

          const mfAny: any = modelClip.manifest as any;
          const ui =
            (mfAny.spec && mfAny.spec.ui) || mfAny.ui || undefined;
          if (ui && Array.isArray(ui.inputs)) {
            ui.inputs = ui.inputs.map((inp: any) => {
              if (!inp || typeof inp !== "object") return inp;
              const id =
                typeof inp.id === "string" ? inp.id : undefined;
              if (!id) return inp;
              if (
                Object.prototype.hasOwnProperty.call(
                  resolvedValues,
                  id,
                )
              ) {
                return {
                  ...inp,
                  value: (resolvedValues as any)[id],
                };
              }
              return inp;
            });
          }
        }
      }

      const assetsRecord: Record<string, Asset> =
        (doc.assets as Record<string, Asset>) ?? {};
      
        // get active project folder uuid
      
      const folderUuid = doc.meta.id

      const promises = Object.values(assetsRecord).map(async (asset) => {
        let useCache = asset.path.includes('cache/engine_results');
        //
        return await getMediaInfo(asset.path, {sourceDir: useCache ? 'apex-cache' : 'user-data', folderUuid});
      });
      try { 
         await Promise.all(promises);
      } catch (err) {
        console.error("jsonPersistence: failed to get media info", err);
      }

      let clipDuration = 0;
      for (const clip of clips) {
        const end = Number((clip as any).endFrame ?? 0) || 0;
        if (end > clipDuration) clipDuration = end;
      }
      if (
        typeof doc.editorState?.duration === "number" &&
        Number.isFinite(doc.editorState.duration)
      ) {
        clipDuration = Math.max(clipDuration, doc.editorState.duration);
      }

      if (useClipStore && typeof useClipStore.setState === "function") {
        useClipStore.setState({
          clips,
          timelines: timelineTracks,
          assets: assetsRecord,
          clipDuration,
          snapGuideX:
            typeof doc.editorState?.snapGuideX === "number"
              ? doc.editorState.snapGuideX
              : null,
        });
      }

      if (useViewportStore && typeof useViewportStore.setState === "function" && Object.keys(doc.editorState ?? {}).length > 0) {
        useViewportStore.setState({
          aspectRatio: doc.settings?.aspectRatio,
          tool: doc.editorState?.tool,
          shape: doc.editorState?.shape,
          scale: doc.editorState?.scale,
          position: doc.editorState?.position,
          viewportSize: doc.editorState?.viewportSize,
          contentBounds: doc.editorState?.contentBounds,
          shouldUpdateViewport: false
        });
      } else {
        console.warn(
          "jsonPersistence: viewport store not ready for JSON snapshot; JSON autosave will rely on project-store changes only until it is available",
        );
      }

      if (globalInputControlsStore && typeof globalInputControlsStore.setState === "function") {
        globalInputControlsStore.setState({
          selectedRangeByInputId: doc.inputControls?.selectedRangeByInputId ?? {},
          selectedInputClipIdByInputId: doc.inputControls?.selectedInputClipIdByInputId ?? {},
          totalTimelineFramesByInputId: doc.inputControls?.totalTimelineFramesByInputId ?? {},
          timelineDurationByInputId: doc.inputControls?.timelineDurationByInputId ?? {},
          fpsByInputId: doc.inputControls?.fpsByInputId ?? {},
          focusFrameByInputId: doc.inputControls?.focusFrameByInputId ?? {},
          focusAnchorRatioByInputId: doc.inputControls?.focusAnchorRatioByInputId ?? {},
          zoomLevelByInputId: doc.inputControls?.zoomLevelByInputId ?? {},
          isPlayingByInputId: doc.inputControls?.isPlayingByInputId ?? {},
        });
      }

      if (useControlsStore && typeof useControlsStore.setState === "function") {
        const currentControls = useControlsStore.getState() ?? {};
        const fps =
          typeof doc.settings?.fps === "number" &&
          Number.isFinite(doc.settings.fps)
            ? doc.settings.fps
            : currentControls.fps || 24;

        const timelineWindow = Array.isArray(doc.editorState?.timelineWindow)
          ? (doc.editorState.timelineWindow as [number, number])
          : [0, clipDuration];

        const totalTimelineFrames = Math.max(
          Number(timelineWindow[1] ?? 0) || 0,
          clipDuration || 0,
        );

        useControlsStore.setState({
          fps,
          defaultClipLength: defaultClipLength ?? 5,
          zoomLevel:
            typeof doc.editorState?.zoomLevel === "number" &&
            Number.isFinite(doc.editorState.zoomLevel)
              ? doc.editorState.zoomLevel
              : currentControls.zoomLevel ?? 1,
          focusFrame:
            typeof doc.editorState?.playheadPosition === "number" &&
            Number.isFinite(doc.editorState.playheadPosition)
              ? doc.editorState.playheadPosition
              : currentControls.focusFrame ?? 0,
          timelineDuration: timelineWindow as [number, number],
          totalTimelineFrames,
        });
      }

      // At this point, the in-memory stores reflect the JSON for this project id.
      currentHydratedProjectId = _projectId;
    } catch (err) {
      console.error("jsonPersistence: failed to hydrate stores from JSON", err);
    }
  } finally {
    isHydratingFromJson = false;
  }
};

const scheduleJsonProjectSave = (getProjectState: () => JsonProjectSlice) => {
  // The launcher window is not an editor surface; it should never autosave
  // timeline JSON because it may have empty/unhydrated stores and would overwrite
  // the real project state created in the main window.
  if (isLauncherWindow()) return;
  const state = getProjectState();
  if (!state || state.activeProjectId == null) return;

  // Avoid saving while a project is actively being hydrated from JSON,
  // or when the active project id does not yet match the last hydrated
  // project. This prevents writing one project's state under another id
  // during rapid project switching.
  if (isHydratingFromJson) return;
  if (
    currentHydratedProjectId != null &&
    state.activeProjectId !== currentHydratedProjectId
  ) {
    return;
  }
  if ((state as any).projectsLoaded === false) {
    return;
  }

  const targetProjectId = normalizeProjectId(state.activeProjectId);
  if (targetProjectId == null) return;

  const myGen = ++jsonSyncGeneration;
  if (jsonSyncTimeout) clearTimeout(jsonSyncTimeout);
  jsonSyncTimeout = setTimeout(async () => {
    if (myGen !== jsonSyncGeneration) return;
    const latestState = getProjectState();
    if (!latestState || latestState.activeProjectId == null) return;

    // If the active project changed since we scheduled this save, abort.
    if (normalizeProjectId(latestState.activeProjectId) !== targetProjectId) {
      return;
    }
    // If hydration is in progress, abort.
    if (isHydratingFromJson) return;
    // If we're not fully loaded, abort.
    if ((latestState as any).projectsLoaded === false) return;
    // If our in-memory stores are known to correspond to a different project, abort.
    if (
      currentHydratedProjectId != null &&
      normalizeProjectId(currentHydratedProjectId) !== targetProjectId
    ) {
      return;
    }

    const snapshotResult = buildProjectJsonSnapshot(latestState);
    if (!snapshotResult || snapshotResult.projectId == null) return;
    try {
      await saveProjectJson(snapshotResult.projectId, snapshotResult.snapshot);
    } catch (err) {
      console.error("jsonPersistence: failed to save project JSON", err);
    }
  }, 500);
};

const initExternalSubscriptions = (getProjectState: () => JsonProjectSlice) => {
  if (subscriptionsInitialized) return;
  subscriptionsInitialized = true;

  try {

    if (useClipStore && typeof useClipStore.subscribe === "function") {
      useClipStore.subscribe((state) => {
        const clipsSig = computeClipSignature((state as any)?.clips);
        const timelinesSig = computeTimelineSignature((state as any)?.timelines);
        const assetsSig = computeAssetSignature((state as any)?.assets);

        // Initialize baseline signatures on first emission.
        if (
          lastClipSignature == null ||
          lastTimelineSignature == null ||
          lastAssetSignature == null
        ) {
          lastClipSignature = clipsSig;
          lastTimelineSignature = timelinesSig;
          lastAssetSignature = assetsSig;
          return;
        }

        const clipsChanged = clipsSig !== lastClipSignature;
        const timelinesChanged = timelinesSig !== lastTimelineSignature;
        const assetsChanged = assetsSig !== lastAssetSignature;

        lastClipSignature = clipsSig;
        lastTimelineSignature = timelinesSig;
        lastAssetSignature = assetsSig;

        if (clipsChanged || timelinesChanged || assetsChanged) {
          scheduleJsonProjectSave(getProjectState);
        }

        // Only generate a new cover when *only* clips changed (no timeline/asset changes),
        // and only after things have settled for at least 1s.
        if (clipsChanged && !timelinesChanged && !assetsChanged) {
          const gen = ++coverUpdateGeneration;
          debouncedGenerateProjectCover(gen, getProjectState);
        }
      });
    } else {
      console.warn(
        "jsonPersistence: clip store not ready for JSON subscriptions; JSON autosave will rely on project-store changes only until it is available",
      );
    }

    if (useViewportStore && typeof useViewportStore.subscribe === "function") {
      useViewportStore.subscribe((state, prev) => {
        if (
          !_.isEqual(state?.aspectRatio, prev?.aspectRatio) || 
          !_.isEqual(state?.scale, prev?.scale) ||
          !_.isEqual(state?.position, prev?.position) ||
          !_.isEqual(state?.tool, prev?.tool) ||
          !_.isEqual(state?.shape, prev?.shape) ||
          !_.isEqual(state?.viewportSize, prev?.viewportSize) ||
          !_.isEqual(state?.contentBounds, prev?.contentBounds)
        ) {
          scheduleJsonProjectSave(getProjectState);
        }
      });
    }
    else {
      console.warn(
        "jsonPersistence: viewport store not ready for JSON subscriptions; JSON autosave will rely on project-store changes only until it is available",
      );
    }

    if (useControlsStore && typeof useControlsStore.subscribe === "function") {
      useControlsStore.subscribe((state, prev) => {
        if (
          state?.fps !== prev?.fps ||
          state?.timelineDuration !== prev?.timelineDuration ||
          state?.zoomLevel !== prev?.zoomLevel ||
          state?.focusFrame !== prev?.focusFrame
        ) {
          scheduleJsonProjectSave(getProjectState);
        }
      });
    } else {
      console.warn(
        "jsonPersistence: controls store not ready for JSON subscriptions; JSON autosave will rely on project-store changes only until it is available",
      );
    }

    // Subscribe to global inputControls store so that any per-input timeline
    // changes (fps, focus frame, ranges, etc.) trigger JSON autosave.

    try {
      if (
        globalInputControlsStore &&
        typeof globalInputControlsStore.subscribe === "function"
      ) {
        globalInputControlsStore.subscribe((state, prev) => {
          if (
            !_.isEqual(
              state?.totalTimelineFramesByInputId,
              prev?.totalTimelineFramesByInputId,
            ) ||
            !_.isEqual(
              state?.timelineDurationByInputId,
              prev?.timelineDurationByInputId,
            ) ||
            !_.isEqual(state?.fpsByInputId, prev?.fpsByInputId) ||
            !_.isEqual(
              state?.focusFrameByInputId,
              prev?.focusFrameByInputId,
            ) ||
            !_.isEqual(
              state?.focusAnchorRatioByInputId,
              prev?.focusAnchorRatioByInputId,
            ) ||
            !_.isEqual(
              state?.selectedInputClipIdByInputId,
              prev?.selectedInputClipIdByInputId,
            ) ||
            !_.isEqual(
              state?.selectedRangeByInputId,
              prev?.selectedRangeByInputId,
            )
          ) {
            scheduleJsonProjectSave(getProjectState);
          }
        });
      } else {
        console.warn(
          "jsonPersistence: inputControls store not ready for JSON subscriptions; JSON autosave will omit inputControls until it is available",
        );
      }
    } catch (err) {
      console.error(
        "jsonPersistence: failed to subscribe to inputControls store",
        err,
      );
    }
  } catch (err) {
    console.error("jsonPersistence: failed to initialize subscriptions", err);
  }
};

export const withJsonProjectPersistence =
  <T extends JsonProjectSlice>(
    config: StateCreator<T, [], []>,
  ): StateCreator<T, [], []> =>
  (set, get, api) => {
    const base = config(set, get, api);

    // Initialize cross-store subscriptions once
    initExternalSubscriptions(() => get() as unknown as JsonProjectSlice);

    const loadActiveProjectFromJson = async (projectId?: string | number) => {
      const myGen = ++jsonProjectLoadGeneration;
      // Cancel any pending cover generation during project switches/hydration.
      try {
        debouncedGenerateProjectCover.cancel();
      } catch {
        // ignore
      }
      // Cancel any pending autosave debounce from the previous project.
      cancelPendingJsonProjectSave();

      try {
        // While we are hydrating from JSON, mark projects as not yet fully loaded
        (api.setState as any)({ projectsLoaded: false });
      } catch {
        // If the slice doesn't have projectsLoaded yet, ignore
      }

      const activeProjectId = normalizeProjectId(await getActiveProjectId());
      projectId = normalizeProjectId(projectId ?? undefined) ?? undefined;

      if (projectId != activeProjectId && projectId) {
        await setActiveProjectId(projectId);
      } else if (!activeProjectId && projectId) {
        await setActiveProjectId(projectId);
      } else if (activeProjectId && !projectId) {
        projectId = activeProjectId;
      }

       let projects = get().projects; 
       let hasAnyProjects = projects.length > 0;
        if (projects.length === 0) {
            const res = await listProjects<{
              id: number;
              name: string;
              fps: number;
              aspectRatio: { width: number; height: number; id: string };
              folderUuid: string;
              coverPath?: string;
              createdAt?: number;
              lastModified?: number;
            }>();
            if (res?.success && Array.isArray(res.data)) {
              projects = res.data;
            }
            // add the projects to the store
            get().setProjects(projects);
        }
        // check if active project id is in the projects array if not hasAnyProjects is false
        if (activeProjectId && !projects.some((p) => p.id === activeProjectId)) {
            if (projects.length > 0) {
              hasAnyProjects = true;
              setActiveProjectId(projects[0].id);
            } else {
              hasAnyProjects = false;
              setActiveProjectId(null);
              projectId = undefined
            }
          }

      if (!projectId) {
        try {
          if (!hasAnyProjects) {
            const controlsAny = useControlsStore as any;
            const controlsState = controlsAny?.getState?.();
            const fpsFromControls =
              controlsState &&
              typeof controlsState.fps === "number" &&
              Number.isFinite(controlsState.fps)
                ? controlsState.fps
                : 24;

            const res = await createProject<{
                id: number;
                name: string;
                fps: number;
                aspectRatio: {
                 width: number; 
                 height: number; 
                 id: string };
                folderUuid: string;}>({
              name: "Project 1",
              fps: fpsFromControls,
            });
            if (res?.success && res.data) {
                await setActiveProjectId(res.data.id);
                get().addProject(res.data)
            } else {
                console.error("jsonPersistence: failed to create project", res?.error);
            }
          }

        } finally {
          try {
              (api.setState as any)({ projectsLoaded: true });
          } catch {
            // ignore
          }
        }
        return;
      }

      try {
        const res = await loadProjectJson(projectId);
        if (myGen !== jsonProjectLoadGeneration) return;
        if (!res?.success) {
          console.error("jsonPersistence: failed to load project JSON", res?.error);
          return;
        }
        const doc = res.data as ProjectJsonSnapshot | null;
        if (!doc) {
          // New project with no on-disk JSON yet: explicitly reset stores to a blank
          // state under this project id, but do not trigger an immediate autosave.
          const normalizedId = normalizeProjectId(projectId);
          if (normalizedId == null) return;

          isHydratingFromJson = true;
          try {
            const projectState = get() as unknown as JsonProjectSlice;
            const activeProject =
              projectState.projects.find((p) => p.id === normalizedId) ?? null;
            const now = Date.now();

            // Clear clip/timeline/assets state.
            try {
              useClipStore.setState({
                clips: [],
                timelines: [],
                assets: {},
                clipDuration: 0,
                snapGuideX: null,
              } as any);
            } catch {
              // ignore; best-effort
            }

            // Reset viewport to sensible defaults (keep aspect ratio from project if present).
            try {
              const existingViewport = useViewportStore.getState();
              const aspectRatio =
                activeProject?.aspectRatio ?? existingViewport?.aspectRatio ?? { width: 16, height: 9, id: "16:9" };
              useViewportStore.setState({
                aspectRatio,
                tool: "pointer",
                shape: "rectangle",
                scale: 0.75,
                position: { x: 0, y: 0 },
                contentBounds: null,
                shouldUpdateViewport: true,
              } as any);
            } catch {
              // ignore; best-effort
            }

            // Reset input controls.
            try {
              globalInputControlsStore.setState({
                selectedRangeByInputId: {},
                selectedInputClipIdByInputId: {},
                totalTimelineFramesByInputId: {},
                timelineDurationByInputId: {},
                fpsByInputId: {},
                focusFrameByInputId: {},
                focusAnchorRatioByInputId: {},
                zoomLevelByInputId: {},
                isPlayingByInputId: {},
              } as any);
            } catch {
              // ignore; best-effort
            }

            // Reset controls to match project fps and a blank timeline.
            try {
              const controlsAny = useControlsStore as any;
              const currentControls = controlsAny?.getState?.() ?? {};
              const fps =
                typeof (activeProject as any)?.fps === "number" &&
                Number.isFinite((activeProject as any).fps)
                  ? (activeProject as any).fps
                  : currentControls?.fps || 24;
              useControlsStore.setState({
                fps,
                defaultClipLength: currentControls?.defaultClipLength ?? 5,
                zoomLevel: currentControls?.zoomLevel ?? 1,
                focusFrame: 0,
                timelineDuration: [0, 0] as [number, number],
                totalTimelineFrames: 0,
              } as any);
            } catch {
              // ignore; best-effort
            }

            // Mark the new blank project as the one our in-memory stores correspond to.
            currentHydratedProjectId = normalizedId;

            // Best-effort: update project metadata timestamps in the projects list so UI stays consistent.
            if (activeProject && typeof (projectState as any).updateProject === "function") {
              (projectState as any).updateProject(normalizedId, {
                createdAt: (activeProject as any).createdAt ?? now,
                lastModified: (activeProject as any).lastModified ?? now,
              });
            }
          } finally {
            isHydratingFromJson = false;
          }
          return;
        }
        await hydrateStoresFromProjectJson(projectId, doc);
        // we need to set the active project id to the project id
        get().setActiveProjectId(projectId as string | number);
      } catch (err) {
        console.error("jsonPersistence: error loading project JSON", err);
      } finally {
        if (myGen === jsonProjectLoadGeneration) {
          try {
            setTimeout(() => {
              (api.setState as any)({ projectsLoaded: true });
            }, 1000);
          } catch {
            // ignore
          }
        }
      }
    };

    // Also react to project-store changes (project list and active project id)
    try {
      const anyApi = api;
      if (typeof anyApi.subscribe === "function") {
        anyApi.subscribe((state, prev) => {
          const activeChanged =
            state?.activeProjectId !== prev?.activeProjectId;

          if (activeChanged) {
            try {
              debouncedGenerateProjectCover.cancel();
            } catch {
              // ignore
            }
            // Best-effort: flush the previous project's state to disk before we
            // switch/hydrate. This prevents a newly-created blank project from
            // ever overwriting the previous project's JSON due to timing.
            try {
              const prevId = normalizeProjectId(prev?.activeProjectId);
              if (
                prevId != null &&
                !isHydratingFromJson &&
                !isLauncherWindow() &&
                normalizeProjectId(currentHydratedProjectId) === prevId
              ) {
                const prevState = {
                  ...(prev as any),
                  activeProjectId: prevId,
                } as JsonProjectSlice;
                const snapshotResult = buildProjectJsonSnapshot(prevState);
                if (snapshotResult?.projectId != null) {
                  void saveProjectJson(
                    snapshotResult.projectId,
                    snapshotResult.snapshot,
                  );
                }
              }
            } catch {
              // ignore; best-effort
            }

            // Cancel any pending autosave so it can't land during the switch.
            cancelPendingJsonProjectSave();
            void loadActiveProjectFromJson(state?.activeProjectId ?? undefined);
          }

          const projectsChanged = state?.projects !== prev?.projects;

          // Do not autosave on active project switches: during creation/switching
          // we can briefly have "blank" stores that must not be persisted.
          if (projectsChanged) {
            scheduleJsonProjectSave(() => state);
          }

          // Whenever the project list changes (create/delete/rename), also
          // trigger a lightweight listProjects call so that the JSON
          // persistence layer stays in sync with on-disk JSON metadata.
          if (projectsChanged) {
            void (async () => {
              try {
                await listProjects();
              } catch (err) {
                console.error(
                  "jsonPersistence: failed to list projects after change",
                  err,
                );
              }
            })();
          }
        });
      }
    } catch (err) {
      console.error(
        "jsonPersistence: failed to subscribe to project store changes",
        err,
      );
    }

    // Initial load for whatever the current active project is (if any)
    try {
      void (async () => {
        // Warm React Query cache early (shared QueryClient instance).
        await Promise.allSettled([
          queryClient.prefetchQuery({
            queryKey: ["manifest"],
            queryFn: async () => {
              const response = await listManifests();
              if (!response.success) {
                throw new Error(
                  response.error ||
                    "Backend is unavailable (failed to load manifests).",
                );
              }
              return response.data ?? [];
            },
          }),
          queryClient.prefetchQuery({
            queryKey: ["modelTypes"],
            queryFn: async () => {
              const response = await listModelTypes();
              if (!response.success) {
                throw new Error(
                  response.error ||
                    "Backend is unavailable (failed to load model types).",
                );
              }
              const data = response.data;
              return (Array.isArray(data) ? data : []) as ModelTypeInfo[];
            },
          }),
          queryClient.prefetchQuery({
            queryKey: ["preprocessor", "list"],
            queryFn: () => fetchPreprocessorsList(),
          }),
        ]);
      })();

      void loadActiveProjectFromJson();
    } catch (err) {
      console.error("jsonPersistence: failed initial JSON project load", err);
    }
    return base;
  };


