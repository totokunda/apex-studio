import { useClipStore } from '@/lib/clip'
import { useControlsStore } from '@/lib/control'
import AudioProperties from '../properties/AudioProperties'
import { useMemo, useRef, useEffect, useState, useCallback } from 'react'
import { getMediaInfoCached } from '@/lib/media/utils'
import DurationProperties from '../properties/DurationProperties'
import PositionProperties from '../properties/PositionProperties'
import LayoutProperties from '../properties/LayoutProperties'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import AppearanceProperties from '../properties/AppearanceProperties'
import AdjustProperties from '../properties/AdjustProperties'  
import {LuChevronRight, LuChevronLeft, LuMonitorCog} from 'react-icons/lu'
import { cn } from '@/lib/utils'  
import TextProperties from '../properties/TextProperties'
import LineProperties from '../properties/LineProperties'
import PreprocessorDurationPanel from '../properties/preprocessor/PreprocessorDurationPanel'
import PreprocessorParametersPanel from '../properties/preprocessor/PreprocessorParametersPanel'
import { FaStop } from 'react-icons/fa'
import { cancelPreprocessor } from '@/lib/preprocessor/api'
import { toast } from 'sonner';

import { usePreprocessorJobActions } from '@/lib/preprocessor/api';
import { useDrawingStore } from '@/lib/drawing';
import { useViewportStore } from '@/lib/viewport';
import MaskPropertiesPanel from '../properties/mask/MaskPropertiesPanel';
import { ModelInputsProperties } from '../properties/model/ModelInputsProperties'
import { RiAiGenerate } from 'react-icons/ri'
import { ModelGenerationProperties } from '../properties/model/ModelGenerationProperties'
import ProgressPanel from '../properties/model/ProgressPanel'
import FilterProperties from '../properties/FilterProperties'
import { ImageClipProps, ModelClipProps, VideoClipProps } from '@/lib/types'
import _ from 'lodash';
import { getSchedulerComponentKey } from "@/lib/manifest/componentKey";
import { usePreprocessorsListQuery } from "@/lib/preprocessor/queries";
import type { Preprocessor } from '@/lib/preprocessor/api';
import { validatePreprocessorFrames } from '@/lib/preprocessorHelpers';
import { runEngine, cancelEngine, useEngineJobActions, useEngineJob } from '@/lib/engine/api';
import { ManifestComponent } from '@/lib/manifest/api';
import ModelComponentsProperties from '../properties/model/ModelComponentsProperties'
import OffloadProperties from '../properties/OffloadProperties'
import { v4 as uuidv4 } from 'uuid';
import FrameInterpolateProperties from '../properties/FrameInterpolateProperties'
import PreprocessorProperties from '../properties/preprocessor/PreprocessorProperties'
import LoraPanel from '../properties/model/LoraPanel'
import { runModelGeneration } from '@/lib/modelGeneration';
import { runPreprocessorJob } from '@/lib/preprocessorRun';
import { useManifestQuery } from '@/lib/manifest/queries'
import { useInputControlsStore } from '@/lib/inputControl'
interface PropertiesPanelProps {
    panelSize: number;
}

const ClipPropertiesPanel:React.FC<PropertiesPanelProps> = ({panelSize}) => {
  const {selectedClipIds, selectedMaskId} = useControlsStore();
  const {selectedPreprocessorId, getPreprocessorById, getClipFromPreprocessorId, updatePreprocessor, getClipsForGroup} = useClipStore();
  const updateClip = useClipStore((s) => s.updateClip);
  // const updateModelInput = useClipStore((s) => s.updateModelInput);
  const { fps } = useControlsStore();
  const selectedLineId = useDrawingStore((s) => s.selectedLineId);
  const tool = useViewportStore((s) => s.tool);
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const clipId = useMemo(() => selectedClipIds[selectedClipIds.length - 1], [selectedClipIds]);
  const getClipPositionScore = useClipStore((s) => s.getClipPositionScore);
  const getClipsByType = useClipStore((s) => s.getClipsByType);
  const timelines = useClipStore((s) => s.timelines);
  const getClipById = useClipStore((s) => s.getClipById);
  const clip = getClipById(clipId);

  const preprocessor = selectedPreprocessorId ? getPreprocessorById(selectedPreprocessorId) : null
  const clipType = clip?.type;
  const tabRef = useRef<HTMLDivElement>(null);  
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>(clip?.type === 'text' ? "text" : "transform");
  const { clearJob, stopTracking } = usePreprocessorJobActions();
  const { startTracking: startEngineTracking, stopTracking: stopEngineTracking, clearJob: clearEngineJob } = useEngineJobActions();
  const [isPreparingGeneration, setIsPreparingGeneration] = useState(false);
  const getModelValues = useClipStore((s) => s.getModelValues);
  const getRawModelValues = useClipStore((s) => s.getRawModelValues);
  const hasFrameInterpolate = useMemo(() => {
    if (clip?.type === 'video') return true;
    return false;
  }, [clip?.type]);

  const hasPreprocessorBrowser = useMemo(() => {
    return clip?.type === 'video' || clip?.type === 'image';
  }, [clip?.type]);

  // Preprocessor browser state
  const { data: preprocessors = [] } = usePreprocessorsListQuery({
    enabled: hasPreprocessorBrowser,
  });
  const [preprocQuery, setPreprocQuery] = useState('');
  const [preprocDetailId, setPreprocDetailId] = useState<string | null>(null);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const addPreprocessorToClip = useClipStore((s) => s.addPreprocessorToClip);
  const setSelectedPreprocessorId = useClipStore((s) => s.setSelectedPreprocessorId);
  const getPreprocessorsForClip = useClipStore((s) => s.getPreprocessorsForClip);
  const currentPreprocessors = getPreprocessorsForClip(clip?.clipId ?? '') || []

  const compatiblePreprocessors = useMemo(() => {
    const list = preprocessors ?? [];
    const q = preprocQuery.trim().toLowerCase();
    const byType = list.filter((p: Preprocessor) => {
      if (clip?.type === 'video') return !!p.supports_video;
      if (clip?.type === 'image') return !!p.supports_image;
      return false;
    });
    if (!q) return byType;
    return byType.filter((p) =>
      p.name.toLowerCase().includes(q) ||
      (p.description || '').toLowerCase().includes(q) ||
      p.category.toLowerCase().includes(q)
    );
  }, [preprocQuery, clip?.type, preprocessors]);

  const getDefaultParamValue = (param: any) => {
    if (Object.prototype.hasOwnProperty.call(param, 'default') && param.default !== undefined) return param.default;
    const t = String(param.type);
    if (t === 'int' || t === 'float') return 0;
    if (t === 'bool') return false;
    if (t === 'str') return '';
    if (t === 'category') {
      const first = Array.isArray(param.options) ? param.options[0] : undefined;
      return first ? first.value : '';
    }
    return '';
  };

  const handleAddPreprocessor = useCallback((preproc: Preprocessor) => {
    if (!clip || !clipId) return;
    if (clip.type !== 'video' && clip.type !== 'image') return;
    const clipDuration = Math.max(1, (clip.endFrame ?? 0) - (clip.startFrame ?? 0));
    const fpsVal = Math.max(1, fps || 1);
    // Prevent add if timeline fully covered by existing preprocessors
    const existingAll = getPreprocessorsForClip(clipId) || [];
    const fullyCovered = (() => {
      if (clipDuration <= 0) return true;
      const intervals = existingAll.map((p) => {
        const s = Math.max(0, p.startFrame ?? 0);
        const e = Math.max(s + 1, Math.min(clipDuration, p.endFrame ?? clipDuration));
        return [s, e] as [number, number];
      }).sort((a, b) => a[0] - b[0]);
      let coverEnd = 0;
      for (const [s, e] of intervals) {
        if (s > coverEnd) {
          return false; // found a gap
        }
        coverEnd = Math.max(coverEnd, e);
        if (coverEnd >= clipDuration) return true;
      }
      return coverEnd >= clipDuration;
    })();
    if (fullyCovered) {
      try { toast.info('No available space for another preprocessor'); } catch {}
      return;
    }
    // Always choose the largest available gap, capped to 5 seconds worth of frames
    const existing = getPreprocessorsForClip(clipId) || [];
    const intervals = existing
      .map((p) => {
        const s = Math.max(0, p.startFrame ?? 0);
        const e = Math.max(s + 1, Math.min(clipDuration, p.endFrame ?? clipDuration));
        return [s, e] as [number, number];
      })
      .sort((a, b) => a[0] - b[0]);

    // Merge overlapping intervals
    const merged: Array<[number, number]> = [];
    for (const [s, e] of intervals) {
      if (merged.length === 0 || s > merged[merged.length - 1][1]) {
        merged.push([s, e]);
      } else {
        merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], e);
      }
    }

    // Compute gaps
    const gaps: Array<[number, number]> = [];
    let prevEnd = 0;
    for (const [s, e] of merged) {
      if (s > prevEnd) gaps.push([prevEnd, s]);
      prevEnd = Math.max(prevEnd, e);
    }
    if (prevEnd < clipDuration) gaps.push([prevEnd, clipDuration]);
    if (merged.length === 0 && gaps.length === 0 && clipDuration > 0) {
      gaps.push([0, clipDuration]);
    }

    // Pick the largest gap
    let chosenGap: [number, number] | null = null;
    let maxLen = -1;
    for (const [s, e] of gaps) {
      const len = Math.max(0, e - s);
      if (len > maxLen) {
        maxLen = len;
        chosenGap = [s, e];
      }
    }

    // Fallback (shouldn't happen due to fullyCovered check), but guard anyway
    if (!chosenGap) {
      try { toast.info('No available space for another preprocessor'); } catch {}
      return;
    }

    const maxFiveSecondsFrames = Math.max(1, Math.round(5 * fpsVal));
    let start = chosenGap[0];
    let end = Math.min(chosenGap[1], start + maxFiveSecondsFrames);

    const { isValid } = validatePreprocessorFrames(start, end, 'new', existing, clipDuration);
    if (!isValid) {
      // Clamp as last resort
      start = Math.max(0, Math.min(start, clipDuration - 1));
      end = Math.max(start + 1, Math.min(end, clipDuration));
    }

    const values: Record<string, any> = {};
    if (Array.isArray(preproc.parameters)) {
      for (const p of preproc.parameters) {
        values[p.name] = getDefaultParamValue(p);
      }
    }

    const id = uuidv4();
    addPreprocessorToClip(clipId, {
      id,
      preprocessor: preproc,
      startFrame: start,
      endFrame: end,
      values,
      status: undefined,
      jobIds: [],
    });
    try { setSelectedPreprocessorId(id); } catch {}
    try { toast.success(`Added ${preproc.name}`); } catch {}
  }, [clip, clipId, fps, focusFrame, getPreprocessorsForClip, addPreprocessorToClip, setSelectedPreprocessorId]);
  const getAssetById = useClipStore((s) => s.getAssetById);

  // check if clip has audio if it is video 
  const hasDuration = useMemo(() => {
   if (!selectedPreprocessorId) {
     return true;
   }
   return false;
  }, [selectedPreprocessorId]);

  const hasFilter = useMemo(() => {
    return clip?.type === 'filter';
  }, [clip?.type]);

  const hasAudio = useMemo(() => {
    if (clip?.type === 'audio') return true;
    if (clip?.type === 'video') {
      const mediaInfo = getMediaInfoCached(clip?.assetId)
      return mediaInfo?.audio !== null
    }
    return false;
  }, [clipType, (clip as VideoClipProps | ImageClipProps)?.assetId]);

  const hasAppearance = useMemo(() => {
    if (clip?.type === 'image' || clip?.type === 'video' || clip?.type === 'shape' || clip?.type === 'text') return true;
    return false;
  }, [clip?.type]);

  const hasModel = useMemo(() => {
    if (clip?.type === 'model') return true;
    return false;
  }, [clip?.type]);

  // Engine job tracking is started explicitly when a run begins; bind UI strictly to this clip's active job.
  const activeEngineJobId = (clip as ModelClipProps | undefined)?.activeJobId ?? null;
  const { isProcessing: isEngineProcessing, isComplete: isEngineComplete, isFailed: isEngineFailed } = useEngineJob(activeEngineJobId, false);

  const clipSignature = JSON.stringify(getModelValues(clipId));

  // Ensure we have the latest manifest with download flags for components and paths
  const manifestId = (clip as ModelClipProps)?.manifest?.metadata?.id || null;
  const { data: manifestData } = useManifestQuery(manifestId);

  const buildSelectedComponentDefaults = useCallback((manifest: any): Record<string, any> => {
    const defaults: Record<string, any> = {};
    if (!manifest) return defaults;
    const components: ManifestComponent[] = (manifest?.spec?.components || []) as ManifestComponent[];

    const normalizeModelPaths = (c: ManifestComponent): Array<any> => {
      const raw = Array.isArray(c.model_path) ? c.model_path : (c.model_path ? [{ path: c.model_path }] : []);
      return (raw as any[]).map((it) => (typeof it === 'string' ? { path: it } : it)).filter((it) => it && it.path);
    };
    const isItemDownloaded = (item: any): boolean => !!(item && item.is_downloaded === true);

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
        const items = normalizeModelPaths(comp).filter((it) => isItemDownloaded(it));
        if (items.length > 0) {
          // Prefer item associated to this component's name when present, else fall back to first
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
  }, []);

  // Automatically populate/refresh default selectedComponents when downloads complete
  useEffect(() => {
    if (!hasModel || !clipId) return;
    const manifest: any = manifestData || (clip as any)?.manifest;
    if (!manifest) return;
    const defaults = buildSelectedComponentDefaults(manifest);
    const existing = (clip as ModelClipProps | undefined)?.selectedComponents || {};
    const merged = { ...defaults, ...existing };
    if (!_.isEqual(merged, existing)) {
      try { updateClip(clipId, { selectedComponents: merged } as any); } catch {}
    }
  }, [hasModel, clipId, manifestData, buildSelectedComponentDefaults, updateClip, clip]);

  // Check whether required model assets for this manifest are downloaded
  const isModelDownloaded = useMemo(() => {
    if (!hasModel) return true;
    const manifest: any = manifestData || (clip as any)?.manifest;
    if (!manifest) return false;
    const components: ManifestComponent[] = (manifest?.spec?.components || []) as ManifestComponent[];
    const normalizeModelPaths = (c: ManifestComponent): Array<any> => {
      const raw = Array.isArray(c.model_path) ? c.model_path : (c.model_path ? [{ path: c.model_path }] : []);
      return (raw as any[]).map((it) => (typeof it === 'string' ? { path: it } : it)).filter((it) => it && it.path);
    };
    const isItemDownloaded = (item: any): boolean => !!(item && item.is_downloaded === true);

    // Consider a component satisfied if either:
    // - It has model_path and at least one item is downloaded
    // - Or it has no model_path (e.g., scheduler/helper without downloadable assets)
    // - Or the component itself is flagged as downloaded
    return components.every((comp) => {
      const items = normalizeModelPaths(comp);
      if (items.length > 0) return items.some((it) => isItemDownloaded(it));
      if (typeof (comp as any).is_downloaded === 'boolean') return !!(comp as any).is_downloaded;
      return true;
    });
  }, [hasModel, manifestData, clip]);

  // Disable generate when any required model inputs are missing
  const isGenerateDisabled = useMemo(() => {
    if (!hasModel || !clip) return false;
    const manifest: any = (clip as any).manifest;
    const ui = manifest?.spec?.ui || manifest?.ui;
    if (!ui || !Array.isArray(ui.inputs)) return false;
    const values = getModelValues(clipId) || {};

    const isEmpty = (val: any) => {
      if (val === undefined || val === null) return true;
      if (typeof val === 'string' && val.trim() === '') return true;
      if (Array.isArray(val) && val.length === 0) return true;
      // Media objects may have a selection field
      if (val && typeof val === 'object' && Object.prototype.hasOwnProperty.call(val, 'selection')) {
        const sel = (val as any).selection;
        if (sel === undefined || sel === null) return true;
        if (Array.isArray(sel) && sel.length === 0) return true;
      }
      return false;
    };

    return ui.inputs.some((inp: any) => inp?.required && isEmpty(values[inp.id]));
  }, [hasModel, clipId, clipSignature]);

  const hasTransform = useMemo(() => {
    if (clip?.type === 'image' || clip?.type === 'video' || clip?.type === 'shape' || clip?.type === 'text') return true;
    if (clip?.type === 'model' && clip.assetId) return true;
    return false;
  }, [clip?.type]);

  const hasText = useMemo(() => {
    if (clip?.type === 'text') return true;
    return false;
  }, [clip?.type]);

  const hasAdjust = useMemo(() => {
    if (clip?.type === 'image' || clip?.type === 'video') return true;
    return false;
  }, [clip?.type]);

  const hasLine = useMemo(() => {
    if (clip?.type === 'draw' && selectedLineId) return true;
    return false;
  }, [clip?.type, selectedLineId]);

  const hasMask = useMemo(() => {
    // Show mask panel if a mask is selected OR if we're in mask tool mode

    if ((clip?.type === 'video' || clip?.type === 'image')) return true;
    return false;
  }, [selectedMaskId, tool, clip?.type]);

  const numVisibleTabs = useMemo(() => {
    let tabTotal = 0;
    if (hasLine) tabTotal++;
    if (hasMask) tabTotal++;
    if (hasTransform) tabTotal++;
    if (hasPreprocessorBrowser) tabTotal++;
    if (hasAudio) tabTotal++;
    if (hasDuration || hasFilter) tabTotal++;
    if (hasAppearance) tabTotal++;
    if (hasAdjust) tabTotal++;
    return tabTotal;
  }, [hasLine, hasMask, hasTransform, hasAudio, hasDuration, hasFilter, hasAppearance, hasAdjust]);

  
  useEffect(() => {
    // Automatically switch to line tab when a line is selected
    if (hasLine && selectedLineId) {
      setSelectedTab('line');
    }
  }, [selectedLineId, hasLine]);

  useEffect(() => {
    // Automatically switch to mask tab when a mask is selected
    if (hasMask && selectedMaskId) {
      setSelectedTab('mask');
    }
  }, [selectedMaskId, hasMask]);

  const checkScrollButtons = () => {
    if (tabRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = tabRef.current;
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 1);
    }
  };

  useEffect(() => {
    checkScrollButtons();
    const handleScroll = () => checkScrollButtons();
    const tabElement = tabRef.current;
    if (tabElement) {
      tabElement.addEventListener('scroll', handleScroll);
      return () => tabElement.removeEventListener('scroll', handleScroll);
    }
  }, [panelSize, numVisibleTabs]);


  const hasValidPreprocessor = useMemo(() => {
    if (selectedPreprocessorId) {
      const clip = getClipFromPreprocessorId(selectedPreprocessorId);
      
      if (!preprocessor || !clip) return false;
      return preprocessor.startFrame !== undefined && preprocessor.endFrame !== undefined && clip.type === 'video' || clip.type === 'image';
    }
    return false;
  }, [selectedPreprocessorId, getPreprocessorById, getClipFromPreprocessorId]);

  const hasPreprocessorDuration = useMemo(() => {
    if (selectedPreprocessorId) {
      
      const clip = getClipFromPreprocessorId(selectedPreprocessorId);
      if (preprocessor?.status === 'running' || preprocessor?.status === 'complete') return false;
      if (!preprocessor || !clip) return false;
      return preprocessor.startFrame !== undefined && preprocessor.endFrame !== undefined && clip.type === 'video' || clip.type === 'image';
    }
    return false;
  }, [selectedPreprocessorId, getPreprocessorById, getClipFromPreprocessorId, preprocessor]);


  const getValidTab = (currentTab: string) => {
    // Check if current tab is valid for this clip type

    if (currentTab === "line" && hasLine) return "line";
    
    if (currentTab === "text" && hasText) return "text";
    if (currentTab === "transform" && hasTransform) return "transform";
    if (currentTab === "mask" && hasMask) return "mask";
    if (currentTab === "preprocessors" && hasPreprocessorBrowser) return "preprocessors";
    if (currentTab === "audio" && hasAudio) return "audio";
    if (currentTab === "duration" && (hasDuration || hasFilter)) return "duration";
    if (currentTab === "appearance" && hasAppearance) return "appearance";
    if (currentTab === "adjust" && hasAdjust) return "adjust";
    if (currentTab === "enhance" && hasFrameInterpolate) return "enhance"; 
    if (currentTab === "preprocessor-parameters" && hasValidPreprocessor) return "preprocessor-parameters";
    if (currentTab === "preprocessor-duration" && hasValidPreprocessor && hasPreprocessorDuration) return "preprocessor-duration";
    if (currentTab === "model-inputs" && hasModel) return "model-inputs";
    if (currentTab === "model-progress" && hasModel && ((clip as ModelClipProps | undefined)?.modelStatus === 'running' || (clip as ModelClipProps | undefined)?.modelStatus === 'pending')) return "model-progress";
    if (currentTab === "model-offload" && hasModel && isModelDownloaded) return "model-offload";
    // If current tab is invalid, return first available tab
    if (hasValidPreprocessor) return "preprocessor-parameters";
    if (hasLine) return "line";
    if (hasModel) {
      if ((clip as ModelClipProps | undefined)?.modelStatus === 'running' || (clip as ModelClipProps | undefined)?.modelStatus === 'pending') return "model-progress";
      return "model-inputs";
    }
    if (hasTransform) return "transform";
    if (hasMask) return "mask";
    if (hasAudio) return "audio";
    if (hasAppearance) return "appearance";
    if (hasAdjust) return "adjust";
    if (hasFrameInterpolate) return "enhance";
    return "duration"; // fallback
  };

  useEffect(() => {
    const validTab = getValidTab(selectedTab);
    if (validTab !== selectedTab) {
      setSelectedTab(validTab);
    }
  }, [clipId, hasLine, hasMask, hasTransform, hasAudio, hasDuration, hasAppearance, hasAdjust, hasPreprocessorDuration, tool, (clip as ModelClipProps | undefined)?.modelStatus, hasFrameInterpolate]);

  // Reflect engine job lifecycle into *this clip's* modelStatus for internal gating.
  // Important: only do this when the clip actually has an active engine job id,
  // and only write when the status would actually change (prevents update loops).
  const currentModelStatus = (clip as ModelClipProps | undefined)?.modelStatus;
  const clipHasResult = !!(
    (clip as ModelClipProps | undefined)?.assetId ||
    (clip as ModelClipProps | undefined)?.previewPath
  );
  useEffect(() => {
    if (!clipId) return;
    if (!hasModel) return;
    if (!activeEngineJobId) return;

    let nextStatus: ModelClipProps["modelStatus"] | undefined;
    if (isEngineProcessing) {
      nextStatus = "running";
    } else if (isEngineComplete) {
      // Only mark complete once the result is actually present on the clip.
      // Otherwise keep it running until the timeline has something to render.
      nextStatus = clipHasResult ? "complete" : "running";
    } else if (isEngineFailed) {
      nextStatus = "failed";
    }

    if (!nextStatus) return;
    if (currentModelStatus === nextStatus) return;
    updateClip(clipId, { modelStatus: nextStatus } as any);
  }, [
    clipId,
    hasModel,
    activeEngineJobId,
    isEngineProcessing,
    isEngineComplete,
    isEngineFailed,
    clipHasResult,
    currentModelStatus,
    updateClip,
  ]);

  const [isPreparingPreprocessor, setIsPreparingPreprocessor] = useState(false);

  const handleRunPreprocessor = useCallback(async () => {
    await runPreprocessorJob({
      selectedPreprocessorId,
      fps: fps || 24,
      getPreprocessorById,
      getClipFromPreprocessorId,
      getAssetById,
      updatePreprocessor,
      clearJob,
      toast,
      setIsPreparingPreprocessor,
    });
  }, [selectedPreprocessorId, fps, getPreprocessorById, getClipFromPreprocessorId, updatePreprocessor, clearJob]);

  const handleStopPreprocessor = useCallback(async () => {
    if (!selectedPreprocessorId) return;
    const clip = getClipFromPreprocessorId(selectedPreprocessorId);
    if (!preprocessor || !clip) return;
    if (preprocessor.status !== 'running') return;
    
    // Stop tracking the job and clear it
    const jobId = preprocessor.activeJobId || selectedPreprocessorId;
    try { await cancelPreprocessor(jobId); } catch {}
    stopTracking(jobId);
    clearJob(jobId);
    
    // Update preprocessor status to idle
    updatePreprocessor(clip.clipId, preprocessor.id, { status: undefined, activeJobId: undefined, progress: 0 });
    
    toast.info(`Preprocessor ${preprocessor.preprocessor.name} stopped`);
  }, [selectedPreprocessorId, getPreprocessorById, getClipFromPreprocessorId, stopTracking, clearJob, updatePreprocessor, preprocessor]);

  const handleGenerate = useCallback(async () => {
    setIsPreparingGeneration(true);
    try {
      // Always read the latest clip from the store so newly-changed offload settings
      // (and other model settings) are included in the engine request.
      const latestClip = clipId ? getClipById(clipId) : clip;
      await runModelGeneration({
        clipId,
        clip: latestClip,
        fps: fps || 24,
        aspectRatio,
        getClipsForGroup,
        getAssetById,
        getClipsByType,
        getClipPositionScore,
        timelines,
        getModelValues,
        getRawModelValues,
        manifestData,
        runEngine,
        clearEngineJob,
        startEngineTracking,
        updateClip,
        toast,
        setSelectedTab,
      });
    } finally {
      setIsPreparingGeneration(false);
    }
  }, [
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
    setSelectedTab,
  ]);

  const isModelRunning =
    !!clip &&
    (((clip as ModelClipProps | undefined)?.modelStatus === 'running') ||
      ((clip as ModelClipProps | undefined)?.modelStatus === 'pending'));

  useEffect(() => {
    if (hasModel && isModelRunning) {
      setSelectedTab("model-progress");
    }
  }, [hasModel, isModelRunning, clipId]);

  const handleStopGeneration = useCallback(async () => {
    const targetJobId = (clip as ModelClipProps)?.activeJobId;
    if (!targetJobId) {
      updateClip(clipId, { modelStatus: undefined });
      return;
    }
    try {
      const res = await cancelEngine(targetJobId);
      if (!res.success) {
        toast.error(res.error || 'Failed to stop generation');
      } else {
        toast.info('Generation stopped');
        // Dispatch event so JobsMenu can immediately mark it as canceled
        const event = new CustomEvent("jobs-menu-reload", {
          detail: { jobId: targetJobId },
        });
        window.dispatchEvent(event);
      }
    } catch (e: any) {
      // fallthrough
    }
    try { await stopEngineTracking(targetJobId); } catch {}
    try { clearEngineJob(targetJobId); } catch {}
    try { if (clipId) updateClip(clipId, { modelStatus: undefined }); } catch {}
  }, [clipId, stopEngineTracking, clearEngineJob, (clip as ModelClipProps)?.activeJobId]);

  const height = useMemo(() => {
    if (hasValidPreprocessor || hasModel) {
      if (hasModel && !isModelDownloaded) {
        return 'calc(100% - 40px)';
      }
      return 'calc(100% - 80px)';
    }
    return '100%';
  }, [hasValidPreprocessor, hasModel, isModelDownloaded]);


  return (
    <div className="h-full w-full min-w-0 flex flex-col" style={{ position: 'relative', overflow: 'hidden' }}>
      <div className="overflow-hidden" style={{ height:height }}>
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="min-w-0 relative flex flex-col h-full">
        <div className="relative shrink-0 ">
          <LuChevronLeft onClick={() => {
            if (tabRef.current) {
              tabRef.current.scrollBy({ left: -96, behavior: 'smooth' });
            }
          }} className={cn("text-brand-light h-6 w-6 bg-brand-background-dark/90  backdrop-blur-sm border border-brand-light/10 hover:bg-brand-background z-50 transition-all duration-200 rounded-full absolute left-0 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollLeft ? "block" : "hidden")} />
          <TabsList ref={tabRef} style={{scrollbarWidth: 'none', msOverflowStyle: 'none'}} className={cn("bg-brand w-full rounded-b-none p-0 min-w-0 shrink overflow-x-auto [&::-webkit-scrollbar]:hidden")}>
            {(hasValidPreprocessor) && <TabsTrigger value="preprocessor-parameters" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Inputs</TabsTrigger>}
            {(hasValidPreprocessor && hasPreprocessorDuration) && <TabsTrigger value="preprocessor-duration" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Duration</TabsTrigger>}
            
            {(hasModel) && <TabsTrigger value="model-inputs" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Inputs</TabsTrigger>}
            {(hasModel) && isModelRunning && <TabsTrigger value="model-progress" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5  whitespace-nowrap">Progress</TabsTrigger>}
            {(hasModel) && <TabsTrigger value="model-architecture" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Architecture</TabsTrigger>}
            {(hasModel) && isModelDownloaded && <TabsTrigger value="model-offload" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Offload</TabsTrigger>}
            {(hasModel) && <TabsTrigger value="model-lora" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">LoRA</TabsTrigger>}
            {(hasModel) && !isModelRunning && <TabsTrigger value="model-generation" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Generations</TabsTrigger>}
            {(hasLine) && <TabsTrigger value="line" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Line</TabsTrigger>}
            {(hasText) && <TabsTrigger value="text" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Text</TabsTrigger>}
            {(hasTransform) && <TabsTrigger value="transform" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Transform</TabsTrigger>}
            {(hasMask) && <TabsTrigger value="mask" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Mask</TabsTrigger>}
            {(hasAudio) && <TabsTrigger value="audio" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Audio</TabsTrigger>}
            {(hasPreprocessorBrowser) && <TabsTrigger value="preprocessors" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Preprocessors</TabsTrigger>}
            {((hasDuration || hasFilter)  && !isModelRunning) && <TabsTrigger value="duration" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">
              {hasFilter ? 'Filter' : 'Duration'}
              </TabsTrigger>}
            {(hasAdjust) && <TabsTrigger value="adjust" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Adjust</TabsTrigger>}
            {(hasAppearance) && <TabsTrigger value="appearance" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Appearance</TabsTrigger>}
            {(hasFrameInterpolate) && <TabsTrigger value="enhance" className="text-brand-light text-[11px] h-9 shrink-0 px-4.5 whitespace-nowrap">Enhance</TabsTrigger>}
            

          </TabsList>
          <LuChevronRight onClick={() => {
            if (tabRef.current) {
              tabRef.current.scrollBy({ left: 96, behavior: 'smooth' });
            }
          }} className={cn("text-brand-light h-6 w-6 border border-brand-light/10 bg-brand-background-dark/90 backdrop-blur-sm hover:bg-brand-background z-10 transition-all duration-200 rounded-full absolute right-0 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollRight ? "block" : "hidden")} />
        </div>
        <ScrollArea className="flex-1 overflow-y-auto">
          <div style={{ paddingBottom: hasValidPreprocessor ? '20px' : '0' }}>
            {(hasPreprocessorBrowser) && <TabsContent value="preprocessors" className="min-w-0 m-0">
              <PreprocessorProperties
                preprocDetailId={preprocDetailId}
                setPreprocDetailId={setPreprocDetailId}
                preprocQuery={preprocQuery}
                setPreprocQuery={setPreprocQuery}
                compatiblePreprocessors={compatiblePreprocessors}
                clip={clip}
                currentPreprocessors={currentPreprocessors as any}
                onAdd={handleAddPreprocessor}
              />
            </TabsContent>}
            {(hasValidPreprocessor && selectedPreprocessorId && preprocessor) && <TabsContent value="preprocessor-parameters" className="min-w-0 m-0">
              <PreprocessorParametersPanel preprocessor={preprocessor}/>
            </TabsContent>}
            {(hasValidPreprocessor && selectedPreprocessorId && hasPreprocessorDuration) && <TabsContent value="preprocessor-duration" className="min-w-0 m-0">
              <PreprocessorDurationPanel preprocessorId={selectedPreprocessorId} />
            </TabsContent>}
          {(hasLine) && <TabsContent value="line" className="min-w-0 m-0">
            <LineProperties clipId={clipId} />
          </TabsContent>}
          {(hasText) && <TabsContent value="text" className="min-w-0 m-0">  <TextProperties clipId={clipId} /> </TabsContent>}
          {(hasTransform) && <TabsContent  value="transform" className="min-w-0 divide-y divide-brand-light/10 m-0">
            <PositionProperties clipId={clipId}  />
            <LayoutProperties clipId={clipId}  />
          </TabsContent>}
          {(hasMask) && <TabsContent value="mask" className="min-w-0 m-0">
            <MaskPropertiesPanel clipId={clipId} />
          </TabsContent>}
          {(hasAudio) && <TabsContent value="audio" className="min-w-0 m-0">
            <AudioProperties clipId={clipId} />
          </TabsContent>}
          {((hasDuration || hasFilter)  && !isModelRunning) && <TabsContent value="duration" className="min-w-0 m-0">
            {hasFilter && <FilterProperties clipId={clipId} />}
            <DurationProperties clipId={clipId} />
          </TabsContent>}
          {(hasAdjust) && <TabsContent value="adjust" className="min-w-0 m-0">
            <AdjustProperties clipId={clipId} />
          </TabsContent>}
          {(hasAppearance) && <TabsContent value="appearance" className="min-w-0 m-0">
            <AppearanceProperties clipId={clipId} />
          </TabsContent>}
          {(hasModel) && <TabsContent value="model-inputs" className="min-w-0 m-0">
            <ModelInputsProperties clipId={clipId} panelSize={panelSize} />
          </TabsContent>}
          {(hasModel) && !isModelRunning &&  <TabsContent value="model-generation" className="min-w-0 m-0">
            <ModelGenerationProperties clipId={clipId} />
          </TabsContent>}
          {(hasModel) && <TabsContent value="model-progress" className="min-w-0 m-0"> 
            <ProgressPanel clipId={clipId} />
          </TabsContent>}
          {(hasModel) && <TabsContent value="model-architecture" className="min-w-0 m-0"> 
            <ModelComponentsProperties clipId={clipId} />
          </TabsContent>}
          {(hasModel) && isModelDownloaded && <TabsContent value="model-offload" className="min-w-0 m-0">
            <OffloadProperties clipId={clipId} />
          </TabsContent>}
            {(hasFrameInterpolate) && <TabsContent value="enhance" className="min-w-0 m-0">
            <FrameInterpolateProperties clipId={clipId} />
          </TabsContent>}
          {(hasModel) && <TabsContent value="model-lora" className="min-w-0 m-0"> 
            <LoraPanel clipId={clipId} panelSize={panelSize} />
          </TabsContent>}
          </div>
        </ScrollArea>
      </Tabs>
      </div>
      {hasValidPreprocessor && (
        <div className="absolute bottom-0 left-0 right-0 p-5 bg-brand border-t border-brand-light/10" style={{ zIndex: 100, pointerEvents: 'auto' }}>
          <button
            disabled={isPreparingPreprocessor}
            onClick={preprocessor?.status === 'running' ? handleStopPreprocessor : handleRunPreprocessor}
            className={cn("w-full py-2.5 px-6 rounded-lg font-medium text-[12px] flex items-center disabled:opacity-60 text-brand-lighter disabled:cursor-not-allowed disabled:bg-brand-light/10 disabled:text-brand-light/50 justify-center gap-x-2 transition-all duration-200 shadow-lg hover:opacity-90", 
              preprocessor?.status === 'running'
                ? 'bg-red-500'
                : isPreparingPreprocessor
                  ? 'bg-brand-light/10 text-brand-light/60'
                  : 'bg-brand-accent-two-shade'
            )}
          >
            {preprocessor?.status === 'running' ? (
              <FaStop size={16} />  
            ) : (
              <LuMonitorCog size={16} />
            )}
            <span>{preprocessor?.status === 'running' ? 'Stop' : (isPreparingPreprocessor ? 'Preparing…' : 'Preprocess')}</span>
          </button>
        </div>
      )}

      {hasModel && (
        <div className={cn("absolute bottom-0 left-0 right-0  border-brand-light/5",
           isModelDownloaded ? "bg-brand  p-5 border-t" : "py-3 px-3 "
        )} style={{ zIndex: 50, pointerEvents: 'auto' }}>
          {isModelRunning ? (
            <button
              onClick={handleStopGeneration}
              className={cn(
                "w-full py-2.5 px-6 rounded-lg font-medium text-[12px] flex items-center justify-center gap-x-2 transition-all duration-200 shadow-lg",
              )}
              style={{ backgroundColor: '#DC2626', color: '#FFFFFF' }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#B91C1C' }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#DC2626' }}
            >
              <FaStop size={16} />
              <span>Stop Generating</span>
            </button>
          ) : (
            isModelDownloaded ? (
              <button
                onClick={handleGenerate}
                disabled={isGenerateDisabled || isPreparingGeneration}
                className={cn(
                  "w-full py-2.5 px-6 rounded-lg font-medium text-[12px] text-brand-lighter bg-brand-accent-two-shade flex items-center justify-center gap-x-2 transition-all duration-200 shadow-lg hover:opacity-90 disabled:opacity-60 disabled:cursor-not-allowed disabled:bg-brand-light/10 disabled:text-brand-light/50",
                )}
              >
                <RiAiGenerate size={16} />
                <span>{isPreparingGeneration ? 'Preparing…' : 'Generate'}</span>
              </button>
            ) : (
              <div className="w-full rounded-lg font-medium text-[11px] flex items-center justify-start gap-x-2  h-full transition-all duration-200 opacity-50 text-brand-light ">
                <span>Download Model To Generate</span>
              </div>
            )
          )}
        </div>
      )}
    </div>
  )
}

export default ClipPropertiesPanel
