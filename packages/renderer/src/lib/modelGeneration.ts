import { v4 as uuidv4 } from 'uuid';
import _ from 'lodash';
import type { AnyClipProps, ModelClipProps } from '@/lib/types';
import { getPreviewPath, savePreviewImage } from '@app/preload';
import { getMediaInfoCached, convertUserDataPath, convertApexCachePath } from '@/lib/media/utils';
import { BASE_LONG_SIDE } from '@/lib/settings';
import type { ExportClip } from '@app/export-renderer';
import { exportClip, exportSequence } from '@app/export-renderer';
import type { ManifestComponent } from '@/lib/manifest/api';

export interface GenerateContext {
  clipId: any;
  clip: any;
  fps: number;
  aspectRatio: { width: number; height: number };
  getClipsForGroup: (children: any) => any[];
  getClipsByType: (type: any) => any[];
  getClipPositionScore: (clipId: any) => number;
  getModelValues: (clipId?: any) => any;
  getRawModelValues: (clipId?: any) => any;
  manifestData: any;
  runEngine: (args: any) => Promise<any>;
  clearEngineJob: (jobId: any) => any;
  startEngineTracking: (jobId: any) => any;
  updateClip: (clipId: any, patch: any) => any;
  toast: { info: (msg: string) => void; success: (msg: string) => void; error: (msg: string) => void };
  setEngineJobId: (id: any) => void;
  setSelectedTab: (tab: string) => void;
}

const buildSelectedComponentDefaults = (manifest: any): Record<string, any> => {
  const defaults: Record<string, any> = {};
  if (!manifest) return defaults;
  const components: ManifestComponent[] = (manifest?.spec?.components || []) as ManifestComponent[];

  const normalizeModelPaths = (c: ManifestComponent): Array<any> => {
    const raw = Array.isArray(c.model_path) ? c.model_path : (c.model_path ? [{ path: c.model_path }] : []);
    return (raw as any[]).map((it) => (typeof it === 'string' ? { path: it } : it)).filter((it) => it && it.path);
  };
  const isItemDownloaded = (item: any): boolean => !!(item && item.is_downloaded === true);

  components.forEach((comp) => {
    const key = String((comp as any).name || comp.type || 'component');
    if (comp.type === 'scheduler' && Array.isArray(comp.scheduler_options) && comp.scheduler_options.length > 0) {
      const first = comp.scheduler_options[0];
      defaults[key] = { name: first.name, base: (first as any).base, config_path: (first as any).config_path };
    } else if (comp.model_path) {
      const items = normalizeModelPaths(comp).filter((it) => isItemDownloaded(it));
      if (items.length > 0) {
        const nameKey = String((comp as any).name || '');
        const matched = nameKey
          ? items.find((it: any) => {
              const itemName = String(it?.name || it?.component_name || it?.id || '');
              return itemName && itemName === nameKey;
            })
          : undefined;
        const chosen = matched || items[0];
        defaults[key] = { path: chosen.path, variant: chosen.variant, precision: chosen.precision, type: chosen.type };
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
    getModelValues,
    getRawModelValues,
    manifestData,
    runEngine,
    clearEngineJob,
    startEngineTracking,
    updateClip,
    toast,
    setEngineJobId,
    setSelectedTab,
  } = ctx;

  toast.info('Preparing inputs and starting generation...');
  const modelValues = getModelValues(clipId);
  const rawModelValues = getRawModelValues(clipId);
  if (!modelValues) return;
  const inputs = (clip as ModelClipProps)?.manifest?.spec.ui?.inputs || [];

  const mapToTargets = new Set<string>();
  try {
    for (const uiInput of inputs as any[]) {
      const t = String(uiInput?.type || '').toLowerCase();
      if ((t === 'image+mask' || t === 'video+mask') && uiInput?.map_to) {
        mapToTargets.add(String(uiInput.map_to));
      }
    }
  } catch {}

  const clipValues: Record<string, AnyClipProps> = {};

  for (const input of inputs) {
    const typeStr = String(input.type);
    if (typeStr === 'image_list') {
      const rawList = modelValues[input.id];
      const arr = Array.isArray(rawList) ? rawList : (rawList ? [rawList] : []);
      const exportedList: Array<{ type: 'image'; src: string }> = [];

      for (const item of arr as any[]) {
        if (!item) continue;
        const value = { ...(item as any) } as AnyClipProps & { selectedFrame?: number; selectedRange?: [number, number]; selection?: string };
        if (!value) continue;
        if (Object.prototype.hasOwnProperty.call(value as any, 'selection') && ((value as any).selection === undefined || (value as any).selection === null || (value as any).selection === '')) {
          continue;
        }

        let width = 0;
        let height = 0;
        if (value.type === 'image') {
          const mediaInfo = getMediaInfoCached(value.src);
          const filePath = convertUserDataPath(value.src);
          value.src = filePath;
          const transform = (value as any).originalTransform;
          width = transform?.width ?? mediaInfo?.image?.width ?? 0;
          height = transform?.height ?? mediaInfo?.image?.height ?? 0;
          if (!mediaInfo) continue;
        } else if (value.type === 'video') {
          const mediaInfo = getMediaInfoCached(value.src);
          const transform = (value as any).originalTransform;
          width = transform?.width ?? mediaInfo?.video?.displayWidth ?? 0;
          height = transform?.height ?? mediaInfo?.video?.displayHeight ?? 0;
          if (!mediaInfo) continue;
        } else {
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

        let clips: AnyClipProps[] = [value as any];
        let offsetStart = 0;
        if (value.type === 'group') {
          const groupedClips = getClipsForGroup((value as any).children);
          const groupStart = value.startFrame ?? 0;
          offsetStart = groupStart;
          clips = groupedClips.map((c) => {
            const newClip = { ...c };
            newClip.startFrame = (c.startFrame ?? 0) - groupStart;
            newClip.endFrame = (c.endFrame ?? 0) - groupStart;
            if (newClip.type === 'image') {
              const mediaInfo = getMediaInfoCached((newClip as any).src);
              if (!mediaInfo) return newClip;
              const filePath = convertUserDataPath((newClip as any).src);
              (newClip as any).src = filePath;
            }
            if (Object.prototype.hasOwnProperty.call(newClip, 'preprocessors')) {
              (newClip as any).preprocessors = (c as any).preprocessors?.map((p: any) => ({
                ...p,
                startFrame: (p.startFrame ?? 0) - groupStart,
                endFrame: (p.endFrame ?? 0) - groupStart,
              })) ?? [];
            }
            return newClip;
          });
        } else {
          offsetStart = value.startFrame ?? 0;
          clips = [value];
        }

        const filterClips = getClipsByType('filter');
        clips = clips
          .filter((c) => c.type !== 'filter')
          .sort((a, b) => getClipPositionScore(a.clipId) - getClipPositionScore(b.clipId));

        filterClips.forEach((c) => {
          if (c.type === 'filter') {
            (c as any).score = getClipPositionScore(c.clipId);
          }
        });

        const exportClips: ExportClip[] = [];
        for (const c of clips as ExportClip[]) {
          const newClip = { ...c };
          if (Array.isArray((c as any).preprocessors)) {
            (newClip as any).preprocessors = (c as any).preprocessors.map((p: any) => {
              const convertedSrc = p?.status === 'complete' && p?.src ? convertApexCachePath(p.src) : p?.src;
              return { ...p, src: convertedSrc };
            });
          }
          for (const filter of [...filterClips]) {
            if (getClipPositionScore(filter.clipId) < getClipPositionScore(newClip.clipId)) {
              if (!newClip.applicators) {
                newClip.applicators = [];
              }
              const newFilter = { ...filter };
              newFilter.startFrame = (newFilter.startFrame ?? 0) - offsetStart;
              newFilter.endFrame = (newFilter.endFrame ?? 0) - offsetStart;
              newClip.applicators?.push(newFilter as any);
            }
          }
          newClip.applicators = _.uniqBy(newClip.applicators ?? [], 'clipId');
          try {
            const isRenderable = newClip.type === 'image' || newClip.type === 'video';
            const isGroupChild = (newClip as any)?.groupId !== undefined && (newClip as any)?.groupId !== null;
            const originalTransform = ((newClip as any)?.transform ?? {}) as any;
            const t = { ...originalTransform } as any;
            if (isRenderable && !isGroupChild && t && typeof width === 'number' && typeof height === 'number') {
              const rawW = (Number(t.width) || 0) || width;
              const rawH = (Number(t.height) || 0) || height;
              const sx = Number.isFinite(t.scaleX) ? Number(t.scaleX) : 1;
              const sy = Number.isFinite(t.scaleY) ? Number(t.scaleY) : 1;
              const w = Math.max(0, rawW * sx);
              const h = Math.max(0, rawH * sy);
              const deg = Number.isFinite(t.rotation) ? Number(t.rotation) : 0;
              const rad = (deg * Math.PI) / 180;
              const cCos = Math.cos(rad);
              const sSin = Math.sin(rad);
              const x1 = w * cCos; const y1 = w * sSin;
              const x2 = -h * sSin; const y2 = h * cCos;
              const x3 = w * cCos - h * sSin; const y3 = w * sSin + h * cCos;
              const minX = Math.min(0, x1, x2, x3);
              const maxX = Math.max(0, x1, x2, x3);
              const minY = Math.min(0, y1, y2, y3);
              const maxY = Math.max(0, y1, y2, y3);
              const aabbW = maxX - minX;
              const aabbH = maxY - minY;
              t.x = (width - aabbW) / 2 - minX;
              t.y = (height - aabbH) / 2 - minY;
              (newClip as any).transform = t;
            }
          } catch {}
          exportClips.unshift(newClip);
        }

        let absolutePath: string | null = null;
        const frame = value.type === 'video' || value.type === 'group' ? (value as any).selectedFrame ?? 0 : 0;

        if (exportClips.length === 1) {
          const result = await exportClip({
            mode: 'image',
            width,
            height,
            imageFrame: frame,
            clip: exportClips[0],
            fps,
            backgroundColor: '#000000',
          });
          if (result instanceof Blob) {
            const buf = new Uint8Array(await result.arrayBuffer());
            absolutePath = await savePreviewImage(buf, { fileNameHint: `${clipId}_${input.id}_${frame}` });
          }
        } else if (exportClips.length > 1) {
          const result = await exportSequence({
            mode: 'image',
            width,
            height,
            imageFrame: frame,
            clips: exportClips,
            fps,
          });
          if (result instanceof Blob) {
            const buf = new Uint8Array(await result.arrayBuffer());
            absolutePath = await savePreviewImage(buf, { fileNameHint: `${clipId}_${input.id}_${frame}` });
          }
        }

        if (absolutePath) {
          exportedList.push({ type: 'image', src: absolutePath });
        }
      }

      modelValues[input.id] = exportedList;
      continue;
    }

    if (typeStr.startsWith('image') || typeStr.startsWith('video')) {
      const isMapTarget = mapToTargets.has(String(input.id));
      const rawValue = { ...modelValues[input.id] } as AnyClipProps & { selectedFrame?: number; selectedRange?: [number, number]; selection?: string; apply_preprocessor?: boolean };
      const value = isMapTarget ? ({ ...rawValue, masks: [] } as typeof rawValue) : rawValue;
      clipValues[input.id] = value;
      if (!value) continue;
      if (Object.prototype.hasOwnProperty.call(value, 'selection') && (value.selection === undefined || value.selection === null || value.selection === '')) continue;

      let width = 0;
      let height = 0;
      if (value.type === 'image') {
        const mediaInfo = getMediaInfoCached(value.src);
        const filePath = convertUserDataPath(value.src);
        value.src = filePath;
        const transform = (value as any).originalTransform;
        width = transform?.width ?? mediaInfo?.image?.width ?? 0;
        height = transform?.height ?? mediaInfo?.image?.height ?? 0;
        if (!mediaInfo) continue;
      } else if (value.type === 'video') {
        const mediaInfo = getMediaInfoCached(value.src);
        const transform = (value as any).originalTransform;
        width = transform?.width ?? mediaInfo?.video?.displayWidth ?? 0;
        height = transform?.height ?? mediaInfo?.video?.displayHeight ?? 0;
        if (!mediaInfo) continue;
      } else {
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

      let clips: AnyClipProps[] = [value as any];
      let offsetStart = 0;
      if (value.type === 'group') {
        const groupedClips = getClipsForGroup(value.children);
        const groupStart = value.startFrame ?? 0;
        offsetStart = groupStart;
        clips = groupedClips.map((c) => {
          const newClip = { ...c };
          newClip.startFrame = (c.startFrame ?? 0) - groupStart;
          newClip.endFrame = (c.endFrame ?? 0) - groupStart;
          if (isMapTarget && Object.prototype.hasOwnProperty.call(newClip, 'masks')) {
            (newClip as any).masks = [];
          }
          if (newClip.type === 'image') {
            const mediaInfo = getMediaInfoCached(newClip.src);
            if (!mediaInfo) return newClip;
            const filePath = convertUserDataPath(newClip.src);
            newClip.src = filePath;
          }
          if (Object.prototype.hasOwnProperty.call(newClip, 'preprocessors')) {
            (newClip as any).preprocessors = (c as any).preprocessors?.map((p: any) => ({
              ...p,
              startFrame: (p.startFrame ?? 0) - groupStart,
              endFrame: (p.endFrame ?? 0) - groupStart,
            })) ?? [];
          }
          return newClip;
        });
      } else {
        offsetStart = value.startFrame ?? 0;
        clips = [value];
      }

      const filterClips = getClipsByType('filter');
      clips = clips.filter((c) => c.type !== 'filter').sort((a, b) => getClipPositionScore(a.clipId) - getClipPositionScore(b.clipId));

      filterClips.map((c) => {
        if (c.type === 'filter') {
          (c as any).score = getClipPositionScore(c.clipId);
        }
      });

      const exportClips: ExportClip[] = [];
      for (const clipItem of clips as ExportClip[]) {
        const newClip = { ...clipItem };
        if (Array.isArray((clipItem as any).preprocessors)) {
          (newClip as any).preprocessors = (clipItem as any).preprocessors.map((p: any) => {
            const convertedSrc = p?.status === 'complete' && p?.src ? convertApexCachePath(p.src) : p?.src;
            return { ...p, src: convertedSrc };
          });
        }
        if (isMapTarget && Object.prototype.hasOwnProperty.call(newClip, 'masks')) {
          (newClip as any).masks = [];
        }
        for (const filter of [...filterClips]) {
          if (getClipPositionScore(filter.clipId) < getClipPositionScore(newClip.clipId)) {
            if (!newClip.applicators) {
              newClip.applicators = [];
            }
            const newFilter = { ...filter };
            newFilter.startFrame = (newFilter.startFrame ?? 0) - offsetStart;
            newFilter.endFrame = (newFilter.endFrame ?? 0) - offsetStart;
            newClip.applicators?.push(newFilter as any);
          }
        }
        newClip.applicators = _.uniqBy(newClip.applicators ?? [], 'clipId');
        try {
          const isRenderable = newClip.type === 'image' || newClip.type === 'video';
          const isGroupChild = (newClip as any)?.groupId !== undefined && (newClip as any)?.groupId !== null;
          const originalTransform = ((newClip as any)?.transform ?? {}) as any;
          const t = { ...originalTransform } as any;
          if (isRenderable && !isGroupChild && t && typeof width === 'number' && typeof height === 'number') {
            const rawW = (Number(t.width) || 0) || width;
            const rawH = (Number(t.height) || 0) || height;
            const sx = Number.isFinite(t.scaleX) ? Number(t.scaleX) : 1;
            const sy = Number.isFinite(t.scaleY) ? Number(t.scaleY) : 1;
            const w = Math.max(0, rawW * sx);
            const h = Math.max(0, rawH * sy);
            const deg = Number.isFinite(t.rotation) ? Number(t.rotation) : 0;
            const rad = deg * Math.PI / 180;
            const c = Math.cos(rad);
            const s = Math.sin(rad);
            const x1 = w * c;           const y1 = w * s;
            const x2 = -h * s;          const y2 = h * c;
            const x3 = w * c - h * s;   const y3 = w * s + h * c;
            const minX = Math.min(0, x1, x2, x3);
            const maxX = Math.max(0, x1, x2, x3);
            const minY = Math.min(0, y1, y2, y3);
            const maxY = Math.max(0, y1, y2, y3);
            const aabbW = maxX - minX;
            const aabbH = maxY - minY;
            t.x = (width - aabbW) / 2 - minX;
            t.y = (height - aabbH) / 2 - minY;
            (newClip as any).transform = t;
          }
        } catch {}
        exportClips.unshift(newClip);
      }

      let absolutePath: string | null = null;

      if (String(input.type).startsWith('video')) {
        const frameRange = value.selectedRange ? value.selectedRange : [0, 1];
        const filePath = await getPreviewPath(`${clipId}_${input.id}_${frameRange[0]}_${frameRange[1]}`);
        if (exportClips.length === 1) {
          const result = await exportClip({
            mode: 'video',
            width: width,
            height: height,
            range: { start: frameRange[0], end: frameRange[1]},
            clip: exportClips[0],
            fps: fps,
            backgroundColor: '#000000',
            filename: filePath,
            encoderOptions: {
              format: 'webm',
              codec: 'vp9',
              preset: 'ultrafast',
              crf: 23,
              bitrate: '1000k',
              resolution: { width: width, height: height },
              alpha: true,
            },
          });
          if (typeof result === 'string') {
            absolutePath = result;
          }
        } else {
          const result = await exportSequence({
            mode: 'video',
            width: width,
            height: height,
            range: { start: frameRange[0], end: frameRange[1]},
            clips: exportClips,
            fps: fps,
            backgroundColor: '#000000',
            filename: filePath,
            encoderOptions: {
              format: 'webm',
              codec: 'vp9',
              preset: 'ultrafast',
              crf: 23,
              bitrate: '1000k',
              resolution: { width: width, height: height },
              alpha: true,
            },
          });
          if (typeof result === 'string') {
            absolutePath = result;
          }
        }
      } else {
        const frame = value.type === 'video' || value.type === 'group' ? (value as any).selectedFrame : 0;
        if (exportClips.length === 1) {
          const result = await exportClip({
            mode: 'image',
            width: width,
            height: height,
            imageFrame: frame,
            clip: exportClips[0],
            fps: fps,
            backgroundColor: '#000000',
          });
          if (result instanceof Blob) {
            const buf = new Uint8Array(await result.arrayBuffer());
            absolutePath = await savePreviewImage(buf, { fileNameHint: `${clipId}_${input.id}_${frame}` });
          }
        } else {
          const result = await exportSequence({
            mode: 'image',
            width: width,
            height: height,
            imageFrame: frame,
            clips: exportClips,
            fps: fps,
          });
          if (result instanceof Blob) {
            const buf = new Uint8Array(await result.arrayBuffer());
            absolutePath = await savePreviewImage(buf, { fileNameHint: `${clipId}_${input.id}_${frame}` });
          }
        }
      }

      (modelValues as any)[input.id] = {
        type: 'image',
        src: absolutePath,
      };
    } else if (String(input.type).startsWith('audio')) {
      const value = { ...modelValues[input.id] } as AnyClipProps & {selectedFrame?: number, selectedRange?: [number, number]};
      if (!value) continue;
      if (value.type === 'audio') {
        const mediaInfo = getMediaInfoCached(value.src);
        if (!mediaInfo) continue;

        const filePath = await getPreviewPath(`${clipId}_${input.id}`, { ext: 'mp3' });
        const frameRange = value.selectedRange ? value.selectedRange : [0, 1];
        value.startFrame = frameRange[0];
        value.endFrame = frameRange[1];
        const result = await exportClip({
          mode: 'audio',
          clip: value as any,
          range: { start: frameRange[0], end: frameRange[1] },
          fps: fps,
          filename: filePath,
        });

        if (typeof result === 'string') {
          (modelValues as any)[input.id] = {
            type: input.type,
            src: result,
          };
        }
      }
    } else if (input.type === 'random') {
      const value = (modelValues as any)[input.id];
      if (value === -1 || value === '-1') {
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
      if (t === 'image_list') {
        const listVal = Array.isArray(raw) ? raw : [];
        const paths = listVal
          .map((item: any) => {
            if (!item) return null;
            if (typeof item === 'string') return item;
            if (item && typeof item === 'object' && typeof (item as any).src === 'string') return (item as any).src;
            if (item && typeof item === 'object' && typeof (item as any).input_path === 'string') return (item as any).input_path;
            return null;
          })
          .filter((p: any) => typeof p === 'string' && p.length > 0);
        if (paths.length > 0) {
          engineInputs[input.id] = paths;
        }
        continue;
      }
      if (t.startsWith('image') || t.startsWith('video')) {
        const hasPreprocessor = Boolean((input as any)?.preprocessor_ref);
        const clipSource = clipValues[input.id];
        const resolveApplyPreprocessor = () => {
          if (clipSource && typeof clipSource === 'object') {
            if (typeof (clipSource as any).apply_preprocessor === 'boolean') {
              return (clipSource as any).apply_preprocessor;
            }
            if (typeof (clipSource as any).apply === 'boolean') {
              return (clipSource as any).apply;
            }
          }
          return undefined;
        };
        const applyPreprocessor = resolveApplyPreprocessor();
        let mediaPath: string | undefined;
        if (typeof raw === 'string') {
          mediaPath = raw;
        } else if (raw && typeof raw === 'object') {
          if (typeof (raw as any).input_path === 'string') {
            mediaPath = (raw as any).input_path;
          } else if (typeof (raw as any).src === 'string') {
            mediaPath = (raw as any).src;
          }
        }
        if (!mediaPath) continue;
        if (hasPreprocessor) {
          engineInputs[input.id] = {
            input_path: mediaPath,
            apply_preprocessor: typeof applyPreprocessor === 'boolean' ? applyPreprocessor : true,
          };
        } else {
          engineInputs[input.id] = mediaPath;
        }
      } else if (t.startsWith('audio')) {
        if (typeof raw === 'string') {
          engineInputs[input.id] = raw;
        } else if (raw && typeof raw === 'object' && (raw as any).src) {
          engineInputs[input.id] = (raw as any).src;
        }
      } else if (t === 'boolean') {
        const v = (raw as any)?.value ?? raw;
        engineInputs[input.id] = (String(v).toLowerCase() === 'true');
      } else if (t === 'number' || t === 'number+slider' || t === 'random') {
        const v = (raw as any)?.value ?? raw;
        const parsed = Number(v);
        engineInputs[input.id] = Number.isFinite(parsed) ? parsed : v;
      } else if (t === 'number_list') {
        const v = (raw as any)?.value ?? raw;
        const arr = String(v).split(/[\s,]+/).map((s) => Number(s)).filter((n) => Number.isFinite(n));
        engineInputs[input.id] = arr;
      } else {
        engineInputs[input.id] = (raw as any)?.value ?? raw;
      }
    }

    let duration = (clip?.endFrame ?? 0) - (clip?.startFrame ?? 0);
    duration = Math.max(1, duration);
    duration = Math.min(duration, ((clip as ModelClipProps)?.manifest?.spec?.max_duration_secs ?? Infinity) * fps);
    const durationSeconds = duration / fps;
    engineInputs['duration'] = `${durationSeconds}s`;
    const manifestId = (clip as ModelClipProps)?.manifest?.metadata?.id;
    const selectedExisting = (clip as ModelClipProps)?.selectedComponents || {};
    const manifestForDefaults = manifestData || (clip as ModelClipProps)?.manifest;
    const selectedDefaults = buildSelectedComponentDefaults(manifestForDefaults);
    const selectedComponents = { ...selectedDefaults, ...selectedExisting };

    if (!selectedComponents.attention) {
      selectedComponents.attention = { name: 'sdpa' };
    }

    const activeJobId = uuidv4();
    
    const res = await runEngine({ manifest_id: manifestId, inputs: engineInputs, selected_components: selectedComponents, job_id: activeJobId });
    if (res.success) {
      toast.success(`Generation started for ${(clip as ModelClipProps)?.manifest?.metadata?.name}`);
      const returnedJobId = (res.data as any)?.job_id || clipId;
      setEngineJobId(returnedJobId);
      if (returnedJobId) {
        try { await clearEngineJob(returnedJobId); } catch {}
        try { await startEngineTracking(returnedJobId); } catch {}
      }
      try {
        const persistedValues: Record<string, any> = {};
        for (const input of inputs) {
          const t = String(input.type);
          if (t.startsWith('image') || t.startsWith('video')) {
            const uiSelection = (clipValues as any)?.[input.id];
            persistedValues[input.id] = uiSelection !== undefined ? uiSelection : ((input as any)?.value ?? '');
          } else {
            const engineVal = (engineInputs as any)[input.id];
            persistedValues[input.id] = engineVal !== undefined ? engineVal : (input as any)?.value ?? '';
          }
        }
        const existingGenerations = ((clip as ModelClipProps)?.generations ?? []);
        const newGeneration = {
          jobId: activeJobId,
          modelStatus: 'pending' as const,
          src: '',
          createdAt: Date.now(),
          selectedComponents: selectedComponents,
          values: getRawModelValues(clipId),
          transform: (clip as AnyClipProps)?.transform,
        };
        if (clipId) {
          updateClip(clipId, {
            activeJobId: activeJobId,
            modelStatus: 'pending',
            generations: [...existingGenerations, newGeneration],
          } as any);
        }
      } catch {}
      try { setSelectedTab('model-progress'); } catch {}
    } else {
      toast.error(res.error || 'Failed to start generation');
    }
  } catch (err: any) {
    toast.error(err?.message || 'Failed to start generation');
  }
};


