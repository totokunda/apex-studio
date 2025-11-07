import { useClipStore } from '@/lib/clip'
import { useControlsStore } from '@/lib/control'
import AudioProperties from './AudioProperties'
import { useMemo, useRef, useEffect, useState, useCallback } from 'react'
import {  convertApexCachePath, convertUserDataPath, getMediaInfoCached } from '@/lib/media/utils'
import DurationProperties from './DurationProperties'
import PositionProperties from './PositionProperties'
import LayoutProperties from './LayoutProperties'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import AppearanceProperties from './AppearanceProperties'
import AdjustProperties from './AdjustProperties'  
import {LuChevronRight, LuChevronLeft, LuMonitorCog} from 'react-icons/lu'
import { cn } from '@/lib/utils'  
import TextProperties from './TextProperties'
import LineProperties from './LineProperties'
import PreprocessorDurationPanel from './preprocessor/PreprocessorDurationPanel'
import PreprocessorParametersPanel from './preprocessor/PreprocessorParametersPanel'
import { FaStop } from 'react-icons/fa'
import { runPreprocessor } from '@/lib/preprocessor/api'
import { toast } from 'sonner';

import { toFrameRange } from '@/lib/media/fps';
import { usePreprocessorJobActions } from '@/lib/preprocessor/api';
import { useDrawingStore } from '@/lib/drawing';
import { useViewportStore } from '@/lib/viewport';
import MaskPropertiesPanel from './mask/MaskPropertiesPanel';
import { ModelInputsProperties } from './model/ModelInputsProperties'
import { RiAiGenerate } from 'react-icons/ri'
import { ModelGenerationProperties } from './model/ModelGenerationProperties'
import ProgressPanel from './model/ProgressPanel'
import FilterProperties from './FilterProperties'
import { savePreviewImage, getPreviewPath} from '@app/preload'
import { AnyClipProps, ModelClipProps } from '@/lib/types'
import {ExportClip, exportSequence, exportClip} from '@app/export-renderer'
import _ from 'lodash';
import { BASE_LONG_SIDE } from '@/lib/settings';
import { runEngine, cancelEngine, useEngineJobActions, useEngineJob } from '@/lib/engine/api';
import { useManifest } from '@/lib/manifest/hooks';
import { ManifestComponent } from '@/lib/manifest/api';
import ModelComponentsProperties from './model/ModelComponentsProperties'
import { v4 as uuidv4 } from 'uuid';
import FrameInterpolateProperties from './FrameInterpolateProperties'
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

  const clip = useClipStore((s) => s.getClipById(clipId))
  // const getClipsByType = useClipStore((s) => s.getClipsByType);

  const preprocessor = selectedPreprocessorId ? getPreprocessorById(selectedPreprocessorId) : null
  const clipType = clip?.type;
  const tabRef = useRef<HTMLDivElement>(null);  
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>(clip?.type === 'text' ? "text" : "transform");
  const { clearJob, stopTracking } = usePreprocessorJobActions();
  const { startTracking: startEngineTracking, stopTracking: stopEngineTracking, clearJob: clearEngineJob } = useEngineJobActions();
  const [isPreparingGeneration, setIsPreparingGeneration] = useState(false);
  const [engineJobId, setEngineJobId] = useState<string | null>(null);
  const getModelValues = useClipStore((s) => s.getModelValues);
  const hasFrameInterpolate = useMemo(() => {
    if (clip?.type === 'video') return true;
    return false;
  }, [clip?.type]);

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
      const mediaInfo = getMediaInfoCached(clip?.src)
      return mediaInfo?.audio !== null
    }
    return false;
  }, [clipType, clip?.src]);

  const hasAppearance = useMemo(() => {
    if (clip?.type === 'image' || clip?.type === 'video' || clip?.type === 'shape' || clip?.type === 'text') return true;
    return false;
  }, [clip?.type]);

  const hasModel = useMemo(() => {
    if (clip?.type === 'model') return true;
    return false;
  }, [clip?.type]);

  // Engine job tracking is started explicitly when a run begins
  const effectiveJobId = engineJobId || (hasModel ? (clipId ?? null) : null);
  const { isProcessing: isEngineProcessing, isComplete: isEngineComplete, isFailed: isEngineFailed } = useEngineJob(effectiveJobId, false);

  const clipSignature = JSON.stringify(getModelValues(clipId));

  // Ensure we have the latest manifest with download flags for components and paths
  const manifestId = (clip as ModelClipProps)?.manifest?.metadata?.id || null;
  const { data: manifestData } = useManifest(manifestId);

  const buildSelectedComponentDefaults = useCallback((manifest: any): Record<string, any> => {
    const defaults: Record<string, any> = {};
    if (!manifest) return defaults;
    const components: ManifestComponent[] = (manifest?.spec?.components || []) as ManifestComponent[];

    const normalizeModelPaths = (c: ManifestComponent): Array<any> => {
      const raw = Array.isArray(c.model_path) ? c.model_path : (c.model_path ? [{ path: c.model_path }] : []);
      return (raw as any[]).map((it) => (typeof it === 'string' ? { path: it } : it)).filter((it) => it && it.path);
    };
    const isItemDownloaded = (item: any): boolean => !!(item && item.is_downloaded === true);

    components.forEach((comp) => {
      const key = comp.type || 'component';
      if (comp.type === 'scheduler' && Array.isArray(comp.scheduler_options) && comp.scheduler_options.length > 0) {
        const first = comp.scheduler_options[0];
        defaults['scheduler'] = { name: first.name, base: (first as any).base, config_path: (first as any).config_path };
      } else if (comp.model_path) {
        const items = normalizeModelPaths(comp).filter((it) => isItemDownloaded(it));
        if (items.length > 0) {
          const first = items[0];
          defaults[key] = { path: first.path, variant: first.variant, precision: first.precision, type: first.type };
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

    if (tool === 'mask' && (clip?.type === 'video' || clip?.type === 'image')) return true;
    return false;
  }, [selectedMaskId, tool, clip?.type]);

  const numVisibleTabs = useMemo(() => {
    let tabTotal = 0;
    if (hasLine) tabTotal++;
    if (hasMask) tabTotal++;
    if (hasTransform) tabTotal++;
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

  useEffect(() => {
    // Reset mapping and status when switching selected clip
    setEngineJobId(null);
    if (clipId) {
      try { updateClip(clipId, { modelStatus: undefined }); } catch {}
    }
  }, [clipId, updateClip]);

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
    if (hasMask) return "mask";
    if (currentTab === "line" && hasLine) return "line";
    if (currentTab === "mask" && hasMask) return "mask";
    if (currentTab === "text" && hasText) return "text";
    if (currentTab === "transform" && hasTransform) return "transform";
    if (currentTab === "audio" && hasAudio) return "audio";
    if (currentTab === "duration" && (hasDuration || hasFilter)) return "duration";
    if (currentTab === "appearance" && hasAppearance) return "appearance";
    if (currentTab === "adjust" && hasAdjust) return "adjust";
    if (currentTab === "enhance" && hasFrameInterpolate) return "enhance"; 
    if (currentTab === "preprocessor-parameters" && hasValidPreprocessor) return "preprocessor-parameters";
    if (currentTab === "preprocessor-duration" && hasValidPreprocessor && hasPreprocessorDuration) return "preprocessor-duration";
    if (currentTab === "model-inputs" && hasModel) return "model-inputs";
    if (currentTab === "model-progress" && hasModel && ((clip as ModelClipProps | undefined)?.modelStatus === 'running' || (clip as ModelClipProps | undefined)?.modelStatus === 'pending')) return "model-progress";
    // If current tab is invalid, return first available tab
    if (hasValidPreprocessor) return "preprocessor-parameters";
    if (hasLine) return "line";
    if (hasModel) return "model-inputs";
    if (hasTransform) return "transform";
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

  // Reflect engine job lifecycle into clip.modelStatus for internal gating
  useEffect(() => {
    if (!clipId) return;
    if (isEngineProcessing) { 
      updateClip(clipId, { modelStatus: 'running' });
    } else if (isEngineComplete) {
      updateClip(clipId, { modelStatus: 'complete' });
    } else if (isEngineFailed) {
      updateClip(clipId, { modelStatus: 'failed' });
    }
  }, [clipId, isEngineProcessing, isEngineComplete, isEngineFailed, updateClip]);

  const [isPreparingPreprocessor, setIsPreparingPreprocessor] = useState(false);

  const handleRunPreprocessor = useCallback(async () => {
    // get the preprocessor 
    if (!selectedPreprocessorId) return;
    // Clear any previous job data if present
    if (preprocessor?.activeJobId) {
      clearJob(preprocessor.activeJobId);
    }
    const clip = getClipFromPreprocessorId(selectedPreprocessorId);
    if (!preprocessor) return;
    if (!clip) return;
    if (!clip.src) return;

    // If backend is remote and src is local-like, inform user and show preparing state
    // Always show preparing state for preprocessors; toast only if an upload will actually occur
    setIsPreparingPreprocessor(true);
    try {
      const { getBackendIsRemote, getFileShouldUpload } = await import('@app/preload');
      const remoteRes = await getBackendIsRemote();
      const isRemote = !!(remoteRes && remoteRes.success && remoteRes.data?.isRemote);
      if (isRemote) {
        const su = await getFileShouldUpload(String(clip.src || ''));
        const shouldUpload = !!(su && su.success && su.data?.shouldUpload);
        if (shouldUpload) {
          toast.info('Uploading source media to server…');
        }
      }
    } catch {}

    // need to convert our startFrame and endFrame back to where it would be with real FPS
    const clipMediaInfo = getMediaInfoCached(clip.src);
    const clipFps = clipMediaInfo?.stats.video?.averagePacketRate ?? 24;
    if (preprocessor.startFrame === undefined || preprocessor.endFrame === undefined) return;
    let { start: startFrameReal, end: endFrameReal } = toFrameRange(preprocessor.startFrame, preprocessor.endFrame, fps, clipFps, clipMediaInfo?.duration ?? 0);

    // need to scale startFrame to real fps
    const clipMediaStartFrame = Math.round(((clipMediaInfo?.startFrame ?? 0) / fps) * clipFps);
    
    startFrameReal += clipMediaStartFrame;
    endFrameReal += clipMediaStartFrame;
    
    // Generate a unique job id for this preprocessor run and persist on the preprocessor
    const activeJobId = uuidv4();
    try { updatePreprocessor(clip.clipId, preprocessor.id, { status: 'running', activeJobId, jobIds: [...(preprocessor.jobIds || []), activeJobId] }); } catch {}

    const response = await runPreprocessor({
      start_frame: startFrameReal,
      end_frame: endFrameReal,
      preprocessor_name: preprocessor.preprocessor.id,
      input_path: clip.src,
      job_id: activeJobId,
      download_if_needed: true,
      params: preprocessor.values,
    });
    setIsPreparingPreprocessor(false);
    if (response.success) {
      toast.success(`Preprocessor ${preprocessor.preprocessor.name} run started successfully`);
      updatePreprocessor(clip.clipId, preprocessor.id, { status: 'running' });
    } else {
      toast.error(`Failed to run preprocessor ${preprocessor.preprocessor.name}`);
    }
  }, [selectedPreprocessorId, getPreprocessorById, getClipFromPreprocessorId, preprocessor]);

  const handleStopPreprocessor = useCallback(() => {
    if (!selectedPreprocessorId) return;
    const clip = getClipFromPreprocessorId(selectedPreprocessorId);
    if (!preprocessor || !clip) return;
    if (preprocessor.status !== 'running') return;
    
    // Stop tracking the job and clear it
    const jobId = preprocessor.activeJobId || selectedPreprocessorId;
    stopTracking(jobId);
    clearJob(jobId);
    
    // Update preprocessor status to idle
    updatePreprocessor(clip.clipId, preprocessor.id, { status: undefined, activeJobId: undefined });
    
    toast.info(`Preprocessor ${preprocessor.preprocessor.name} stopped`);
  }, [selectedPreprocessorId, getPreprocessorById, getClipFromPreprocessorId, stopTracking, clearJob, updatePreprocessor, preprocessor]);

  const handleGenerate = useCallback(async () => {
    setIsPreparingGeneration(true);
    toast.info('Preparing inputs and starting generation...');
    // get the clip Id that is of type model 
    const modelValues = getModelValues(clipId);
    if (!modelValues) return;
    const inputs = (clip as ModelClipProps)?.manifest?.spec.ui?.inputs || []; 

    const clipValues: Record<string, AnyClipProps> = {};

    // loop through modelValues to pre-export media inputs
    for (const input of inputs) {
       if (String(input.type).startsWith('image')) {
        const value = modelValues[input.id] as AnyClipProps & {selectedFrame?: number, selectedRange?: [number, number]; selection?:string};
        clipValues[input.id] = value;
        if (!value || value.selection === '') continue;
        // we need to check clip type
        
        let width = 0;
        let height = 0;
        if (value.type === 'image') {
          const mediaInfo = getMediaInfoCached(value.src);
          const filePath = convertUserDataPath(value.src);  
          value.src = filePath;
          let transform = value.originalTransform;
          width = transform?.width ?? mediaInfo?.image?.width ?? 0;
          height = transform?.height ?? mediaInfo?.image?.height ?? 0;
          value.preprocessors.forEach((p) => {
            if (p.status === 'complete' && p.src) {
              p.src = convertApexCachePath(p.src);
            }
          });
          if (!mediaInfo) continue;
        } else if (value.type === 'video') {
          const mediaInfo = getMediaInfoCached(value.src);
          let transform = value.originalTransform;
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
          // offset all clips by the group start frame moving it to 0 
          const groupStart = value.startFrame ?? 0;
          offsetStart = groupStart;
          clips = groupedClips.map((c) => {
            // we need to offset our aplicators and preprocessors too. 
            const newClip = { ...c };
            newClip.startFrame = (c.startFrame ?? 0) - groupStart;
            newClip.endFrame = (c.endFrame ?? 0) - groupStart;
            if (newClip.type === 'image') {
              const mediaInfo = getMediaInfoCached(newClip.src);
              if (!mediaInfo) return newClip;
              const filePath = convertUserDataPath(newClip.src);  
              newClip.src = filePath;
              newClip.preprocessors.forEach((p) => {
                if (p.status === 'complete' && p.src) {
                  p.src = convertApexCachePath(p.src);
                }
              });
            }
            if (Object.prototype.hasOwnProperty.call(newClip, 'preprocessors')) {
              (newClip as any).preprocessors = (c as any).preprocessors?.map((p: any) => ({ ...p, startFrame: (p.startFrame ?? 0) - groupStart, endFrame: (p.endFrame ?? 0) - groupStart })) ?? [];
            }
            return newClip;
          });
        } else {
          offsetStart = value.startFrame ?? 0;
          clips = [value];
        }

        // sort clips by position score
        const filterClips = getClipsByType('filter');
        clips = clips.filter((c) => c.type !== 'filter').sort((a, b) => getClipPositionScore(a.clipId) - getClipPositionScore(b.clipId));

        filterClips.map((c) => {
          if (c.type === 'filter') {
            (c as any).score = getClipPositionScore(c.clipId);
          }
        });

        // for every filter clip that has a lower score than a clip we add the filter as an applicator to the clip
        const exportClips: ExportClip[] = [];
        for (const clip of clips as ExportClip[]) {
          const newClip = { ...clip };
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
          // add to beginning of exportClips
          newClip.applicators = _.uniqBy(newClip.applicators ?? [], 'clipId');
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
        
        }
        else {
          const frame = value.type === 'video' || value.type === 'group' ? value.selectedFrame : 0;
          if (exportClips.length === 1) {
            
            const result = await exportClip({
              mode: 'image',
              width: width,
              height: height,
              imageFrame: frame,
              clip: exportClips[0],
              fps: fps,
            });
            
            // save the result to the file system
            if (result instanceof Blob) {
              const buf = new Uint8Array(await result.arrayBuffer());
              absolutePath = await savePreviewImage(buf, { fileNameHint: `${clipId}_${input.id}_${frame}` });
            }

          }
          else {
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

        modelValues[input.id] = {
          type: 'image',
          src: absolutePath,
        };


       } else if (String(input.type).startsWith('audio')) {
          const value = modelValues[input.id] as AnyClipProps & {selectedFrame?: number, selectedRange?: [number, number]};
          if (!value) continue;
          // we need to check clip type
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
              modelValues[input.id] = {
                type: input.type,
                src: result,
              };
            }
          }

       } 

       else if (input.type === 'random') {
          const value = modelValues[input.id] 
          if (value === -1 || value === '-1') {
            const min = input.min ?? 0;
            const max = input.max ?? Number.MAX_SAFE_INTEGER;
            const randomValue = Math.floor(Math.random() * (max - min + 1)) + min;
            modelValues[input.id] = randomValue;
          }
       }

    }
    
    // Build engine inputs and trigger generation with job_id = clipId
    try {
      const engineInputs: Record<string, any> = {};
      for (const input of inputs) {
        const raw = (modelValues as any)[input.id];
        const t = String(input.type);
        if (raw == null) continue;
        if (t.startsWith('image') || t.startsWith('video') || t.startsWith('audio')) {
          if (typeof raw === 'string') {
            engineInputs[input.id] = raw;
          } else if (raw && typeof raw === 'object' && raw.src) {
            engineInputs[input.id] = raw.src;
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

      // add duration to engine inputs
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
          try { clearEngineJob(returnedJobId); } catch {}
          try { await startEngineTracking(returnedJobId); } catch {}
        }
        try {

          // we need to replace engineInputs with clipValues
          const updatedEngineInputs = { ...engineInputs };
          for (const [key, value] of Object.entries(clipValues)) {
            updatedEngineInputs[key] = value;
          }
          const existingGenerations = ((clip as ModelClipProps)?.generations ?? []);
          const newGeneration = {
            jobId: activeJobId,
            modelStatus: 'pending' as const,
            src: '',
            createdAt: Date.now(),
            selectedComponents: selectedComponents,
            values: updatedEngineInputs,
          };
          updateClip(clipId, {
            activeJobId: activeJobId,
            modelStatus: 'pending',
            generations: [...existingGenerations, newGeneration],
          } as any);
        } catch {}
        // Switch to Progress tab for visibility
        try { setSelectedTab('model-progress'); } catch {}
      } else {
        toast.error(res.error || 'Failed to start generation');
      }
    } catch (err: any) {
      toast.error(err?.message || 'Failed to start generation');
    } finally {
      setIsPreparingGeneration(false);
    }

  }, [selectedClipIds, getModelValues, clip]);

  const handleStopGeneration = useCallback(async () => {
    const targetJobId = engineJobId || (clip as ModelClipProps)?.activeJobId;
    if (!targetJobId) return;
    try {
      const res = await cancelEngine(targetJobId);
      if (!res.success) {
        toast.error(res.error || 'Failed to stop generation');
      } else {
        toast.info('Generation stopped');
      }
    } catch (e: any) {
      // fallthrough
    }
    try { await stopEngineTracking(targetJobId); } catch {}
    try { clearEngineJob(targetJobId); } catch {}
    try { if (clipId) updateClip(clipId, { modelStatus: undefined }); } catch {}
  }, [clipId, engineJobId, stopEngineTracking, clearEngineJob, (clip as ModelClipProps)?.activeJobId]);

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
        <div className="relative flex-shrink-0 ">
          <LuChevronLeft onClick={() => {
            if (tabRef.current) {
              tabRef.current.scrollBy({ left: -96, behavior: 'smooth' });
            }
          }} className={cn("text-brand-light h-6 w-6 bg-brand-background/90 border border-brand-light/10 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute left-1 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollLeft ? "block" : "hidden")} />
          <TabsList ref={tabRef} style={{scrollbarWidth: 'none', msOverflowStyle: 'none'}} className={cn("bg-brand w-full rounded-b-none p-0 min-w-0 flex-shrink overflow-x-auto [&::-webkit-scrollbar]:hidden")}>
            {(hasValidPreprocessor) && <TabsTrigger value="preprocessor-parameters" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Inputs</TabsTrigger>}
            {(hasValidPreprocessor && hasPreprocessorDuration) && <TabsTrigger value="preprocessor-duration" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Duration</TabsTrigger>}
            {(hasModel) && <TabsTrigger value="model-inputs" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Inputs</TabsTrigger>}
            {(hasModel) && ((clip as ModelClipProps | undefined)?.modelStatus === 'running' || (clip as ModelClipProps | undefined)?.modelStatus === 'pending') && <TabsTrigger value="model-progress" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Progress</TabsTrigger>}
            {(hasModel) && <TabsTrigger value="model-architecture" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Architecture</TabsTrigger>}
            {(hasModel) && <TabsTrigger value="model-generation" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Generations</TabsTrigger>}
            {(hasLine) && <TabsTrigger value="line" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Line</TabsTrigger>}
            {(hasText) && <TabsTrigger value="text" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Text</TabsTrigger>}
            {(hasTransform && !hasMask) && <TabsTrigger value="transform" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Transform</TabsTrigger>}
            {(hasMask) && <TabsTrigger value="mask" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Mask</TabsTrigger>}
            {(hasAudio && !hasMask) && <TabsTrigger value="audio" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Audio</TabsTrigger>}
            {((hasDuration || hasFilter) && !hasMask) && <TabsTrigger value="duration" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">
              {hasFilter ? 'Filter' : 'Duration'}
              </TabsTrigger>}
            {(hasAdjust && !hasMask) && <TabsTrigger value="adjust" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Adjust</TabsTrigger>}
            {(hasAppearance && !hasMask) && <TabsTrigger value="appearance" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Appearance</TabsTrigger>}
            {(hasFrameInterpolate) && <TabsTrigger value="enhance" className="text-brand-light text-[11px] h-9 flex-shrink-0 px-4 whitespace-nowrap">Enhance</TabsTrigger>}
          </TabsList>
          <LuChevronRight onClick={() => {
            if (tabRef.current) {
              tabRef.current.scrollBy({ left: 96, behavior: 'smooth' });
            }
          }} className={cn("text-brand-light h-6 w-6 border border-brand-light/10 bg-brand-background/90 hover:bg-brand-background/100 z-10 transition-all duration-200 rounded-full absolute right-1 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollRight ? "block" : "hidden")} />
        </div>
        <ScrollArea className="flex-1 overflow-y-auto">
          <div style={{ paddingBottom: hasValidPreprocessor ? '20px' : '0' }}>
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
          {(hasTransform && !hasMask) && <TabsContent  value="transform" className="min-w-0 divide-y divide-brand-light/10 m-0">
            <PositionProperties clipId={clipId}  />
            <LayoutProperties clipId={clipId}  />
          </TabsContent>}
          {(hasMask) && <TabsContent value="mask" className="min-w-0 m-0">
            <MaskPropertiesPanel clipId={clipId} />
          </TabsContent>}
          {(hasAudio && !hasMask) && <TabsContent value="audio" className="min-w-0 m-0">
            <AudioProperties clipId={clipId} />
          </TabsContent>}
          {((hasDuration || hasFilter) && !hasMask) && <TabsContent value="duration" className="min-w-0 m-0">
            {hasFilter && <FilterProperties clipId={clipId} />}
            <DurationProperties clipId={clipId} />
          </TabsContent>}
          {(hasAdjust && !hasMask) && <TabsContent value="adjust" className="min-w-0 m-0">
            <AdjustProperties clipId={clipId} />
          </TabsContent>}
          {(hasAppearance && !hasMask) && <TabsContent value="appearance" className="min-w-0 m-0">
            <AppearanceProperties clipId={clipId} />
          </TabsContent>}
          {(hasModel) && <TabsContent value="model-inputs" className="min-w-0 m-0">
            <ModelInputsProperties clipId={clipId} panelSize={panelSize} />
          </TabsContent>}
          {(hasModel) && <TabsContent value="model-generation" className="min-w-0 m-0">
            <ModelGenerationProperties clipId={clipId} />
          </TabsContent>}
          {(hasModel) && <TabsContent value="model-progress" className="min-w-0 m-0"> 
            <ProgressPanel clipId={clipId} />
          </TabsContent>}
          {(hasModel) && <TabsContent value="model-architecture" className="min-w-0 m-0"> 
            <ModelComponentsProperties clipId={clipId} />
          </TabsContent>}
            {(hasFrameInterpolate) && <TabsContent value="enhance" className="min-w-0 m-0">
            <FrameInterpolateProperties clipId={clipId} />
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
              preprocessor?.status === 'running' ? 'bg-red-500' : 'bg-brand-accent-two-shade ',
              isPreparingPreprocessor ? 'bg-brand-light/10! text-brand-light/60!' : ''
            )}
            onMouseEnter={(e) => {
              if (preprocessor?.status === 'running') {
                e.currentTarget.style.backgroundColor = '#B91C1C';
              } else {
                e.currentTarget.style.backgroundColor = '#8E5FAF';
              }
            }}
            onMouseLeave={(e) => {
              if (preprocessor?.status === 'running') {
                e.currentTarget.style.backgroundColor = '#DC2626';
              } else {
                e.currentTarget.style.backgroundColor = '#A477C4';
              }
            }}
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
          {((clip as ModelClipProps | undefined)?.modelStatus === 'running' || (clip as ModelClipProps | undefined)?.modelStatus === 'pending') ? (
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
                  "w-full py-2.5 px-6 rounded-lg font-medium text-[12px] text-brand-light bg-brand-accent-two-shade flex items-center justify-center gap-x-2 transition-all duration-200 shadow-lg hover:opacity-90 disabled:opacity-60 disabled:cursor-not-allowed disabled:bg-brand-light/10 disabled:text-brand-light/50",
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