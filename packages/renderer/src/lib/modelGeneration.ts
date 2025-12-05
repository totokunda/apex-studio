import { v4 as uuidv4 } from "uuid";
import _ from "lodash";
import type { AnyClipProps, ModelClipProps, TimelineProps } from "@/lib/types";
import { getPreviewPath, savePreviewImage } from "@app/preload";
import { getMediaInfoCached } from "@/lib/media/utils";
import { exportClip, exportSequence } from "@app/export-renderer";
import type { ManifestComponent } from "@/lib/manifest/api";
import { prepareExportClipsForValue } from "@/lib/prepareExportClips";
type MediaItem = {
  type: "image" | "video" | "audio";
  src: string;
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
  setEngineJobId: (id: any) => void;
  setSelectedTab: (tab: string) => void;
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

  components.forEach((comp) => {
    const key = String((comp as any).name || comp.type || "component");
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
    setEngineJobId,
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

        if (exportClips.length === 1) {
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
        } else if (exportClips.length > 1) {
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
          useOriginalTransform: true,
          applyCentering: true,
          dimensionsFrom: "clip",
        },
      );

      const { exportClips, width, height } = prepared;
      let absolutePath: string | null = null;

      if (String(input.type).startsWith("video")) {
        const frameRange = value.selectedRange ? value.selectedRange : [0, 1];
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
      } else {
        const frame =
          value.type === "video" || value.type === "group"
            ? (value as any).selectedFrame
            : 0;
        if (exportClips.length === 1) {
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

        } else {
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

        const filePath = await getPreviewPath(`${clipId}_${input.id}`, {
          ext: "mp3",
        });
        const frameRange = value.selectedRange ? value.selectedRange : [0, 1];
        value.startFrame = frameRange[0];
        value.endFrame = frameRange[1];
        const result = await exportClip({
          mode: "audio",
          clip: value as any,
          range: { start: frameRange[0], end: frameRange[1] },
          fps: fps,
          filename: filePath,
        });

        if (typeof result === "string") {
          (modelValues as any)[input.id] = {
            type: input.type,
            src: result,
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

    const activeJobId = uuidv4();


    const res = await runEngine({
      manifest_id: manifestId,
      inputs: engineInputs,
      selected_components: selectedComponents,
      job_id: activeJobId,
    });
    if (res.success) {
      toast.success(
        `Generation started for ${(clip as ModelClipProps)?.manifest?.metadata?.name}`,
      );
      const returnedJobId = (res.data as any)?.job_id || clipId;
      setEngineJobId(returnedJobId);
      if (returnedJobId) {
        try {
          await clearEngineJob(returnedJobId);
        } catch {}
        try {
          await startEngineTracking(returnedJobId);
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
          jobId: activeJobId,
          modelStatus: "pending" as const,
          assetId: undefined,
          createdAt: Date.now(),
          src: undefined,
          selectedComponents: selectedComponents,
          values: getRawModelValues(clipId),
        };
        if (clipId) {
          updateClip(clipId, {
            activeJobId: activeJobId,
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
