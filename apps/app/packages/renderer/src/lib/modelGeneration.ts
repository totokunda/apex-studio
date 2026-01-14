import { v4 as uuidv4 } from "uuid";
import _ from "lodash";
import type { AnyClipProps, ModelClipProps, TimelineProps } from "@/lib/types";
import { getPreviewPath, savePreviewImage } from "@app/preload";
import { getPathExists } from "@app/preload";
import { cleanupOldPreviews } from "@app/preload";
import { getMediaInfoCached } from "@/lib/media/utils";
import { exportClip, exportSequence } from "@app/export-renderer";
import type { ManifestComponent } from "@/lib/manifest/api";
import { prepareExportClipsForValue } from "@/lib/prepareExportClips";
import { getSchedulerComponentKey } from "@/lib/manifest/componentKey";
import { useProjectsStore } from "@/lib/projects";
type MediaItem = {
  type: "image" | "video" | "audio";
  src: string;
};

type ExportMode = "image" | "video" | "audio";

type ExportCacheEntry = {
  mode: ExportMode;
  src: string;
};

const EXPORT_CACHE_MAX_ENTRIES = 256;

declare global {
  interface Window {
    __apexModelExportCache__?: Map<string, ExportCacheEntry>;
    __apexPreviewCleanupTimer__?: number;
  }
}

let fallbackExportResultCache:
  | Map<string, ExportCacheEntry>
  | undefined;

const DEFAULT_PREVIEW_MAX_AGE_MS = 24 * 60 * 60 * 1000;

/**
 * Starts a best-effort background cleanup that deletes preview files older than `maxAgeMs`
 * from the app's preview folder. Safe to call multiple times (it only schedules once).
 */
export const startPreviewFolderAutoCleanup = (options?: {
  /**
   * Delete preview files older than this age. Defaults to 24 hours.
   */
  maxAgeMs?: number;
  /**
   * How often to run the cleanup. Defaults to 24 hours.
   */
  intervalMs?: number;
  /**
   * Delay before first cleanup run. Defaults to 24 hours.
   */
  initialDelayMs?: number;
  /**
   * If true, run a cleanup immediately in addition to scheduling.
   */
  runImmediately?: boolean;
}): void => {
  const maxAgeMs =
    typeof options?.maxAgeMs === "number" && Number.isFinite(options.maxAgeMs)
      ? Math.max(0, options.maxAgeMs)
      : DEFAULT_PREVIEW_MAX_AGE_MS;
  const intervalMs =
    typeof options?.intervalMs === "number" && Number.isFinite(options.intervalMs)
      ? Math.max(10_000, options.intervalMs)
      : DEFAULT_PREVIEW_MAX_AGE_MS;
  const initialDelayMs =
    typeof options?.initialDelayMs === "number" &&
    Number.isFinite(options.initialDelayMs)
      ? Math.max(0, options.initialDelayMs)
      : DEFAULT_PREVIEW_MAX_AGE_MS;

  const runOnce = async () => {
    try {
      await cleanupOldPreviews({ maxAgeMs });
    } catch {
      // ignore
    }
  };

  if (options?.runImmediately) {
    void runOnce();
  }

  if (typeof window === "undefined") return;
  if (window.__apexPreviewCleanupTimer__ != null) return;

  const timer = window.setTimeout(function tick() {
    void runOnce();
    window.__apexPreviewCleanupTimer__ = window.setTimeout(tick, intervalMs);
  }, initialDelayMs);

  window.__apexPreviewCleanupTimer__ = timer;
};

const getExportCacheMap = (): Map<string, ExportCacheEntry> => {
  if (typeof window !== "undefined") {
    if (!window.__apexModelExportCache__) {
      window.__apexModelExportCache__ = new Map<string, ExportCacheEntry>();
    }
    return window.__apexModelExportCache__;
  }
  if (!fallbackExportResultCache) {
    fallbackExportResultCache = new Map<string, ExportCacheEntry>();
  }
  return fallbackExportResultCache;
};

const buildExportCacheKey = (params: {
  mode: ExportMode;
  exportClips: any[];
  width?: number;
  height?: number;
  imageFrame?: number;
  range?: { start: number; end: number };
  fps?: number;
  backgroundColor?: string;
  encoderOptions?: any;
  audioOptions?: any;
}): string | null => {
  try {
    const { mode, exportClips, ...rest } = params;
    return JSON.stringify({
      mode,
      exportClips,
      opts: rest,
    });
  } catch {
    return null;
  }
};

const getCachedExportResult = (key: string): string | undefined => {
  const cache = getExportCacheMap();
  const hit = cache.get(key);
  return hit?.src;
};

const deleteCachedExportResult = (key: string): void => {
  const cache = getExportCacheMap();
  cache.delete(key);
};

const setCachedExportResult = (key: string, entry: ExportCacheEntry): void => {
  const cache = getExportCacheMap();
  if (!key) return;
  if (cache.size >= EXPORT_CACHE_MAX_ENTRIES) {
    const firstKey = cache.keys().next().value as string | undefined;
    if (firstKey) {
      cache.delete(firstKey);
    }
  }
  cache.set(key, entry);
};

export interface GenerateContext {
  clipId: any;
  clip: any;
  fps: number;
  aspectRatio: { width: number; height: number };
  getClipsForGroup: (children: any) => any[];
  getClipsByType: (type: any) => any[];
  getClipPositionScore: (clipId: any) => number;
  timelines: TimelineProps[];
  getModelValues: (clipId?: any) => any;
  getRawModelValues: (clipId?: any) => any;
  manifestData: any;
  runEngine: (args: any) => Promise<any>;
  getAssetById: (assetId: string) => any;
  clearEngineJob: (jobId: any) => any;
  startEngineTracking: (jobId: any) => any;
  updateClip: (clipId: any, patch: any) => any;
  toast: {
    info: (msg: string) => void;
    success: (msg: string) => void;
    error: (msg: string) => void;
  };
  setSelectedTab: (tab: string) => void;
}

const mapOffloadToEngineInputs = (
  offload:
    | Record<
        string,
        {
          enabled?: boolean;
          level?: string;
          num_blocks?: number;
          use_stream?: boolean;
          record_stream?: boolean;
          low_cpu_mem_usage?: boolean;
        }
      >
    | undefined,
): Record<string, any> => {
    if (!offload) return {};
    const mappedObject: Record<string, any> = {};
    Object.entries(offload).forEach(([key, value]) => {
      if (value.enabled) {
        mappedObject[key] = {
          group_offload_type: value.level === "leaf" ? "leaf_level" : "block_level",
          group_offload_num_blocks_per_group: value.num_blocks,
          // Defaults: treat unset as enabled (UI defaults)
          group_offload_use_stream: value.use_stream ?? true,
          group_offload_record_stream: value.record_stream ?? true,
          group_offload_low_cpu_mem_usage: value.low_cpu_mem_usage ?? true,
        }
      }
    });
    return mappedObject;
}

const buildSelectedComponentDefaults = (manifest: any): Record<string, any> => {
  const defaults: Record<string, any> = {};
  if (!manifest) return defaults;
  const components: ManifestComponent[] = (manifest?.spec?.components ||
    []) as ManifestComponent[];

  const normalizeModelPaths = (c: ManifestComponent): Array<any> => {
    const raw = Array.isArray(c.model_path)
      ? c.model_path
      : c.model_path
        ? [{ path: c.model_path }]
        : [];
    return (raw as any[])
      .map((it) => (typeof it === "string" ? { path: it } : it))
      .filter((it) => it && it.path);
  };
  const isItemDownloaded = (item: any): boolean =>
    !!(item && item.is_downloaded === true);

  let schedulerIdx = 0;
  components.forEach((comp) => {
    const key =
      comp.type === "scheduler"
        ? getSchedulerComponentKey(comp, schedulerIdx++)
        : String((comp as any).name || comp.type || "component");
    if (
      comp.type === "scheduler" &&
      Array.isArray(comp.scheduler_options) &&
      comp.scheduler_options.length > 0
    ) {
      const first = comp.scheduler_options[0];
      defaults[key] = {
        name: first.name,
        base: (first as any).base,
        config_path: (first as any).config_path,
      };
    } else if (comp.model_path) {
      const items = normalizeModelPaths(comp).filter((it) =>
        isItemDownloaded(it),
      );
      if (items.length > 0) {
        const nameKey = String((comp as any).name || "");
        const matched = nameKey
          ? items.find((it: any) => {
              const itemName = String(
                it?.name || it?.component_name || it?.id || "",
              );
              return itemName && itemName === nameKey;
            })
          : undefined;
        const chosen = matched || items[0];
        defaults[key] = {
          path: chosen.path,
          variant: chosen.variant,
          precision: chosen.precision,
          type: chosen.type,
        };
      }
    }
  });
  return defaults;
};

export const runModelGeneration = async (ctx: GenerateContext) => {
  // Start preview cleanup timer once per session (best-effort).
  startPreviewFolderAutoCleanup({ runImmediately: true });

  const {
    clipId,
    clip,
    fps,
    aspectRatio,
    getClipsForGroup,
    getClipsByType,
    getClipPositionScore,
    timelines,
    getModelValues,
    getRawModelValues,
    getAssetById,
    manifestData,
    runEngine,
    clearEngineJob,
    startEngineTracking,
    updateClip,
    toast,
    setSelectedTab,
  } = ctx;

  toast.info("Preparing inputs and starting generation...");
  const modelValues = getModelValues(clipId);
  if (!modelValues) return;
  const inputs = (clip as ModelClipProps)?.manifest?.spec.ui?.inputs || [];

  const mapToTargets = new Set<string>();
  try {
    for (const uiInput of inputs as any[]) {
      const t = String(uiInput?.type || "").toLowerCase();
      if ((t === "image+mask" || t === "video+mask") && uiInput?.map_to) {
        mapToTargets.add(String(uiInput.map_to));
      }
    }
  } catch {}

  const clipValues: Record<string, AnyClipProps> = {};

  for (const input of inputs) {
    const typeStr = String(input.type);
    if (typeStr === "image_list") {
      const rawList = modelValues[input.id];
      const arr = Array.isArray(rawList) ? rawList : rawList ? [rawList] : [];
      const exportedList: Array<{ type: "image"; src: string }> = [];

      for (const item of arr as any[]) {
        if (!item) continue;
        const value = { ...(item as any) } as AnyClipProps & {
          selectedFrame?: number;
          selectedRange?: [number, number];
          selection?: string;
        };
        if (!value) continue;
        if (
          Object.prototype.hasOwnProperty.call(value as any, "selection") &&
          ((value as any).selection === undefined ||
            (value as any).selection === null ||
            (value as any).selection === "")
        ) {
          continue;
        }

        const prepared = prepareExportClipsForValue(
          value as AnyClipProps,
          {
            aspectRatio,
            getClipsForGroup,
            getAssetById,
            getClipsByType,
            getClipPositionScore,
            timelines,
          },
          {
            useMediaDimensionsForExport: true,
            useOriginalTransform: true,
            dimensionsFrom: "clip",
            clearMasks: false,
            applyCentering: true,
          },
        );



        const { exportClips, width, height } = prepared;

        

        let absolutePath: string | null = null;
        const frame =
          value.type === "video" || value.type === "group"
            ? ((value as any).selectedFrame ?? 0)
            : 0;

        const cacheKey =
          exportClips.length > 0
            ? buildExportCacheKey({
                mode: "image",
                exportClips,
                width,
                height,
                imageFrame: frame,
                fps,
                backgroundColor: "#000000",
              })
            : null;

        const cached = cacheKey ? getCachedExportResult(cacheKey) : undefined;
        if (cached && cacheKey) {
          try {
            const existsResp = await getPathExists(cached);
            if (existsResp?.success && existsResp.data?.exists) {
              absolutePath = cached;
            } else {
              deleteCachedExportResult(cacheKey);
            }
          } catch {
            // On IPC error, fall back to re-export path
          }
        }

        if (!absolutePath && exportClips.length === 1) {
          const result = await exportClip({
            mode: "image",
            width,
            height,
            imageFrame: frame,
            clip: exportClips[0],
            fps,
            backgroundColor: "#000000",
          });
          if (result instanceof Blob) {
            const buf = new Uint8Array(await result.arrayBuffer());
            absolutePath = await savePreviewImage(buf, {
              fileNameHint: `${clipId}_${input.id}_${frame}`,
            });
          }
        } else if (!absolutePath && exportClips.length > 1) {
          const result = await exportSequence({
            mode: "image",
            width,
            height,
            imageFrame: frame,
            clips: exportClips,
            fps,
          });
          if (result instanceof Blob) {
            const buf = new Uint8Array(await result.arrayBuffer());
            absolutePath = await savePreviewImage(buf, {
              fileNameHint: `${clipId}_${input.id}_${frame}`,
            });
          }
        }

        if (absolutePath && cacheKey) {
          setCachedExportResult(cacheKey, {
            mode: "image",
            src: absolutePath,
          });
        }

        if (absolutePath) {
          exportedList.push({ type: "image", src: absolutePath });
        }
      }

      modelValues[input.id] = exportedList;
      continue;
    }

    if (typeStr.startsWith("image") || typeStr.startsWith("video")) {
      const isMapTarget = mapToTargets.has(String(input.id));
      const rawValue = { ...modelValues[input.id] } as AnyClipProps & {
        selectedFrame?: number;
        selectedRange?: [number, number];
        selection?: string;
        apply_preprocessor?: boolean;
      };
      const value = isMapTarget
        ? ({ ...rawValue, masks: [] } as typeof rawValue)
        : rawValue;
      clipValues[input.id] = value;
      if (!value) continue;
      if (
        Object.prototype.hasOwnProperty.call(value, "selection") &&
        (value.selection === undefined ||
          value.selection === null ||
          value.selection === "")
      )
        continue;

      const prepared = prepareExportClipsForValue(
        value as AnyClipProps,
        {
          aspectRatio,
          getClipsForGroup,
          getAssetById,
          getClipsByType,
          getClipPositionScore,
          timelines,
        },
        {
          clearMasks: isMapTarget,
          useMediaDimensionsForExport: true,
          useOriginalTransform: value.type !== "group",
          applyCentering: true,
          dimensionsFrom: "clip",
        },
      );

      const { exportClips, width, height } = prepared;

      let absolutePath: string | null = null;

      if (String(input.type).startsWith("video")) {
        const frameRange = value.selectedRange ? value.selectedRange : [0, 1];

        const cacheKey =
          exportClips.length > 0
            ? buildExportCacheKey({
                mode: "video",
                exportClips,
                width,
                height,
                range: { start: frameRange[0], end: frameRange[1] },
                fps,
                backgroundColor: "#000000",
                encoderOptions: {
                  format: "webm",
                  codec: "vp9",
                  preset: "ultrafast",
                  crf: 23,
                  bitrate: "1000k",
                  resolution: { width, height },
                  alpha: true,
                },
              })
            : null;

        const cached = cacheKey ? getCachedExportResult(cacheKey) : undefined;
        if (cached && cacheKey) {
          try {
            const existsResp = await getPathExists(cached);
            if (existsResp?.success && existsResp.data?.exists) {
              absolutePath = cached;
            } else {
              deleteCachedExportResult(cacheKey);
            }
          } catch {
            // On IPC error, fall back to re-export path
          }
        }

        if (!absolutePath) {
          const filePath = await getPreviewPath(
            `${clipId}_${input.id}_${frameRange[0]}_${frameRange[1]}`,
          );
          if (exportClips.length === 1) {
            const result = await exportClip({
              mode: "video",
              width: width,
              height: height,
              range: { start: frameRange[0], end: frameRange[1] },
              clip: exportClips[0],
              fps: fps,
              backgroundColor: "#000000",
              filename: filePath,
              encoderOptions: {
                format: "webm",
                codec: "vp9",
                preset: "ultrafast",
                crf: 23,
                bitrate: "1000k",
                resolution: { width: width, height: height },
                alpha: true,
              },
            });
            if (typeof result === "string") {
              absolutePath = result;
            }
          } else {

            const result = await exportSequence({
              mode: "video",
              width: width,
              height: height,
              range: { start: frameRange[0], end: frameRange[1] },
              clips: exportClips,
              fps: fps,
              backgroundColor: "#000000",
              filename: filePath,
              encoderOptions: {
                format: "webm",
                codec: "vp9",
                preset: "ultrafast",
                crf: 23,
                bitrate: "1000k",
                resolution: { width: width, height: height },
                alpha: true,
              },
            });
            if (typeof result === "string") {
              absolutePath = result;
            }
          }
        }

        if (absolutePath && cacheKey) {
          setCachedExportResult(cacheKey, {
            mode: "video",
            src: absolutePath,
          });
        }
      } else {
        const frame =
          value.type === "video" || value.type === "group"
            ? (value as any).selectedFrame
            : 0;

        const cacheKey =
          exportClips.length > 0
            ? buildExportCacheKey({
                mode: "image",
                exportClips,
                width,
                height,
                imageFrame: frame,
                fps,
                backgroundColor: "#000000",
              })
            : null;

        const cached = cacheKey ? getCachedExportResult(cacheKey) : undefined;
        if (cached && cacheKey) {
          try {
            const existsResp = await getPathExists(cached);
            if (existsResp?.success && existsResp.data?.exists) {
              absolutePath = cached;
            } else {
              deleteCachedExportResult(cacheKey);
            }
          } catch {
            // On IPC error, fall back to re-export path
          }
        }

        if (!absolutePath && exportClips.length === 1) {
          const result = await exportClip({
            mode: "image",
            width: width,
            height: height,
            imageFrame: frame,
            clip: exportClips[0],
            fps: fps,
            backgroundColor: "#000000",
          });
          if (result instanceof Blob) {
            const buf = new Uint8Array(await result.arrayBuffer());
            absolutePath = await savePreviewImage(buf, {
              fileNameHint: `${clipId}_${input.id}_${frame}`,
            });
          }
        } else if (!absolutePath && exportClips.length > 1) {
          const result = await exportSequence({
            mode: "image",
            width: width,
            height: height,
            imageFrame: frame,
            clips: exportClips,
            fps: fps,
          });
          if (result instanceof Blob) {
            const buf = new Uint8Array(await result.arrayBuffer());
            absolutePath = await savePreviewImage(buf, {
              fileNameHint: `${clipId}_${input.id}_${frame}`,
            });
          }
        }

        if (absolutePath && cacheKey) {
          setCachedExportResult(cacheKey, {
            mode: "image",
            src: absolutePath,
          });
        }
      }

      (modelValues as any)[input.id] = {
        type: "image",
        src: absolutePath,
      };
    } else if (String(input.type).startsWith("audio")) {
      const value = { ...modelValues[input.id] } as AnyClipProps & {
        selectedFrame?: number;
        selectedRange?: [number, number];
      };
      if (!value) continue;
      if (value.type === "audio") {
        const asset = getAssetById(value.assetId);
        const mediaInfo = getMediaInfoCached(asset?.path);
        if (!mediaInfo) continue;
        const audioClipExport = prepareExportClipsForValue(value as AnyClipProps, {
          aspectRatio,
          getClipsForGroup,
          getAssetById,
          getClipsByType,
          getClipPositionScore,
          timelines,
        });
        const { exportClips } = audioClipExport;

        const frameRange = value.selectedRange ? value.selectedRange : [0, 1];
        value.startFrame = frameRange[0];
        value.endFrame = frameRange[1];

        const cacheKey =
          exportClips.length > 0
            ? buildExportCacheKey({
                mode: "audio",
                exportClips,
                range: { start: frameRange[0], end: frameRange[1] },
                fps,
                audioOptions: { format: "mp3" },
              })
            : null;

        const cached = cacheKey ? getCachedExportResult(cacheKey) : undefined;
        let resultPath: string | undefined;

        if (cached && cacheKey) {
          try {
            const existsResp = await getPathExists(cached);
            if (existsResp?.success && existsResp.data?.exists) {
              resultPath = cached;
            } else {
              deleteCachedExportResult(cacheKey);
            }
          } catch {
            // On IPC error, fall back to re-export path
          }
        }

        if (!resultPath) {
          const filePath = await getPreviewPath(`${clipId}_${input.id}`, {
            ext: "mp3",
          });
          const result = await exportClip({
            mode: "audio",
            clip: exportClips[0],
            range: { start: frameRange[0], end: frameRange[1] },
            fps: fps,
            filename: filePath,
          });

          if (typeof result === "string") {
            resultPath = result;
          }
        }

        if (resultPath) {
          if (cacheKey) {
            setCachedExportResult(cacheKey, {
              mode: "audio",
              src: resultPath,
            });
          }
          (modelValues as any)[input.id] = {
            type: input.type,
            src: resultPath,
          };
        }
      }
    } else if (input.type === "random") {
      const value = (modelValues as any)[input.id];
      if (value === -1 || value === "-1") {
        const min = input.min ?? 0;
        const max = input.max ?? Number.MAX_SAFE_INTEGER;
        const randomValue = Math.floor(Math.random() * (max - min + 1)) + min;
        (modelValues as any)[input.id] = randomValue;
      }
    }
  }

  try {
    const engineInputs: Record<string, any> = {};
    for (const input of inputs) {
      const raw = (modelValues as any)[input.id];
      const t = String(input.type);
      if (raw == null) continue;
      if (t === "image_list") {
        const listVal = Array.isArray(raw) ? raw : [];
        const paths = listVal
          .map((item: MediaItem) => {
            if (!item) return null;
            if (typeof item === "string") return item;
            if (
              item &&
              typeof item === "object" &&
              typeof item.src === "string"
            )
              return item.src;
            if (
              item &&
              typeof item === "object" &&
              typeof (item as any).input_path === "string"
            )
              return (item as any).input_path;
            return null;
          })
          .filter((p: any) => typeof p === "string" && p.length > 0);
        if (paths.length > 0) {
          engineInputs[input.id] = paths;
        }
        continue;
      }
      if (t.startsWith("image") || t.startsWith("video")) {
        const hasPreprocessor = Boolean((input as any)?.preprocessor_ref);
        const clipSource = clipValues[input.id];
        const resolveApplyPreprocessor = () => {
          if (clipSource && typeof clipSource === "object") {
            if (typeof (clipSource as any).apply_preprocessor === "boolean") {
              return (clipSource as any).apply_preprocessor;
            }
            if (typeof (clipSource as any).apply === "boolean") {
              return (clipSource as any).apply;
            }
          }
          return undefined;
        };
        const applyPreprocessor = resolveApplyPreprocessor();
        let mediaPath: string | undefined;
        if (typeof raw === "string") {
          mediaPath = raw;
        } else if (raw && typeof raw === "object") {
          if (typeof (raw as any).input_path === "string") {
            mediaPath = (raw as any).input_path;
          } else if (typeof (raw as MediaItem).src === "string") {
            mediaPath = (raw as MediaItem).src;
          }
        }
        if (!mediaPath) continue;
        if (hasPreprocessor) {
          engineInputs[input.id] = {
            input_path: mediaPath,
            apply_preprocessor:
              typeof applyPreprocessor === "boolean" ? applyPreprocessor : true,
          };
        } else {
          engineInputs[input.id] = mediaPath;
        }
      } else if (t.startsWith("audio")) {
        if (typeof raw === "string") {
          engineInputs[input.id] = raw;
        } else if (raw && typeof raw === "object" && (raw as MediaItem).src) {
          engineInputs[input.id] = (raw as MediaItem).src;
        }
      } else if (t === "boolean") {
        const v = (raw as any)?.value ?? raw;
        engineInputs[input.id] = String(v).toLowerCase() === "true";
      } else if (t === "number" || t === "number+slider" || t === "random") {
        const v = (raw as any)?.value ?? raw;
        const parsed = Number(v);
        engineInputs[input.id] = Number.isFinite(parsed) ? parsed : v;
      } else if (t === "number_list") {
        const v = (raw as any)?.value ?? raw;
        const arr = String(v)
          .split(/[\s,]+/)
          .map((s) => Number(s))
          .filter((n) => Number.isFinite(n));
        engineInputs[input.id] = arr;
      } else {
        engineInputs[input.id] = (raw as any)?.value ?? raw;
      }
    }

    let duration = (clip?.endFrame ?? 0) - (clip?.startFrame ?? 0);
    duration = Math.max(1, duration);
    duration = Math.min(
      duration,
      ((clip as ModelClipProps)?.manifest?.spec?.max_duration_secs ??
        Infinity) * fps,
    );
    const durationSeconds = duration / fps;
    engineInputs["duration"] = `${durationSeconds}s`;
    const manifestId = (clip as ModelClipProps)?.manifest?.metadata?.id;
    const selectedExisting = (clip as ModelClipProps)?.selectedComponents || {};
    const manifestForDefaults =
      manifestData || (clip as ModelClipProps)?.manifest;
    const selectedDefaults =
      buildSelectedComponentDefaults(manifestForDefaults);
    const selectedComponents = { ...selectedDefaults, ...selectedExisting };

    if (!selectedComponents.attention) {
      selectedComponents.attention = { name: "sdpa" };
    }

    // Forward optional offload config to the engine using the existing `selected_components` payload.
    // We merge the offload fields into their respective component keys (e.g. "unet", "vae", etc.)
    // to match the engine's expected shape.
    const offload = (clip as ModelClipProps)?.offload;
    const offloadEngineInputs = mapOffloadToEngineInputs(offload);
    const selectedComponentsWithOffload: Record<string, any> = {
      ...selectedComponents,
    };
    for (const [key, value] of Object.entries(offloadEngineInputs || {})) {
      const existing = (selectedComponentsWithOffload as any)[key];
      const existingObj =
        existing && typeof existing === "object" ? existing : {};
      (selectedComponentsWithOffload as any)[key] = { ...existingObj, ...value };
    }

    // Remove local filesystem paths from selected components (server fills these in).
    const selectedComponentsForEngine: Record<string, any> = Object.fromEntries(
      Object.entries(selectedComponentsWithOffload).map(([key, value]) => {
        if (!value || typeof value !== "object") return [key, value];
        const { path: _path, ...rest } = value as any;
        return [key, rest];
      }),
    );

    const activeJobId = uuidv4();

    const activeProject = useProjectsStore.getState().getActiveProject();
    const folderUuid = activeProject?.folderUuid;

    const res = await runEngine({
      manifest_id: manifestId,
      inputs: engineInputs,
      selected_components: selectedComponentsForEngine,
      job_id: activeJobId,
      folder_uuid: folderUuid,
    });
    if (res.success) {
      toast.success(
        `Generation started for ${(clip as ModelClipProps)?.manifest?.metadata?.name}`,
      );
      const returnedJobId = (res.data as any)?.job_id;
      const effectiveJobId = returnedJobId || activeJobId || clipId;
      if (effectiveJobId) {
        try {
          await clearEngineJob(effectiveJobId);
        } catch {}
        try {
          await startEngineTracking(effectiveJobId);
        } catch {}
      }
      try {
        const persistedValues: Record<string, any> = {};
        for (const input of inputs) {
          const t = String(input.type);
          if (t.startsWith("image") || t.startsWith("video")) {
            const uiSelection = (clipValues as any)?.[input.id];
            persistedValues[input.id] =
              uiSelection !== undefined
                ? uiSelection
                : ((input as any)?.value ?? "");
          } else {
            const engineVal = (engineInputs as any)[input.id];
            persistedValues[input.id] =
              engineVal !== undefined
                ? engineVal
                : ((input as any)?.value ?? "");
          }
        }
        
        const existingGenerations = (clip as ModelClipProps)?.generations ?? [];
        const newGeneration = {
          jobId: effectiveJobId,
          modelStatus: "pending" as const,
          assetId: undefined,
          createdAt: Date.now(),
          src: undefined,
          selectedComponents: selectedComponentsForEngine,
          values: getRawModelValues(clipId),
        };
        if (clipId) {
          updateClip(clipId, {
            activeJobId: effectiveJobId,
            modelStatus: "pending",
            generations: [...existingGenerations, newGeneration],
          } as any);
        }
      } catch {}
      try {
        setSelectedTab("model-progress");
      } catch {}
    } else {
      toast.error(res.error || "Failed to start generation");
    }
  } catch (err: any) {
    toast.error(err?.message || "Failed to start generation");
  }
};
