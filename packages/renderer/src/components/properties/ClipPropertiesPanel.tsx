import { useClipStore } from '@/lib/clip'
import { useControlsStore } from '@/lib/control'
import AudioProperties from './AudioProperties'
import { useMemo, useRef, useEffect, useState, useCallback } from 'react'
import {  getMediaInfoCached } from '@/lib/media/utils'
import DurationProperties from './DurationProperties'
import PositionProperties from './PositionProperties'
import LayoutProperties from './LayoutProperties'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import AppearanceProperties from './AppearanceProperties'
import AdjustProperties from './AdjustProperties'  
import {LuChevronRight, LuChevronLeft} from 'react-icons/lu'
import { cn } from '@/lib/utils'  
import TextProperties from './TextProperties'
import LineProperties from './LineProperties'
import PreprocessorInfoPanel from './preprocessor/PreprocessorInfoPanel'
import PreprocessorDurationPanel from './preprocessor/PreprocessorDurationPanel'
import PreprocessorParametersPanel from './preprocessor/PreprocessorParametersPanel'
import { FaPlay, FaStop } from 'react-icons/fa'
import { runPreprocessor } from '@/lib/preprocessor/api'
import { toast } from 'sonner';

import { toFrameRange } from '@/lib/media/fps';
import { usePreprocessorJobActions } from '@/lib/preprocessor/api';
import { useDrawingStore } from '@/lib/drawing';

interface PropertiesPanelProps {
    panelSize: number;
}

const ClipPropertiesPanel:React.FC<PropertiesPanelProps> = ({panelSize}) => {
  const {selectedClipIds} = useControlsStore();
  const {selectedPreprocessorId, getPreprocessorById, getClipFromPreprocessorId, updatePreprocessor} = useClipStore();
  const { fps } = useControlsStore();
  const selectedLineId = useDrawingStore((s) => s.selectedLineId);
  const clipId = useMemo(() => selectedClipIds[selectedClipIds.length - 1], [selectedClipIds]);

  const clip = useClipStore((s) => s.getClipById(clipId))
  const preprocessor = selectedPreprocessorId ? getPreprocessorById(selectedPreprocessorId) : null
  const clipType = clip?.type;
  const tabRef = useRef<HTMLDivElement>(null);  
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>(clip?.type === 'text' ? "text" : "transform");
  const { clearJob, stopTracking } = usePreprocessorJobActions();

  

  // check if clip has audio if it is video 
  const hasDuration = useMemo(() => {
   if (!selectedPreprocessorId) {
     return true;
   }
   return false;
  }, [selectedPreprocessorId]);

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

  const numVisibleTabs = useMemo(() => {
    let tabTotal = 0;
    if (hasLine) tabTotal++;
    if (hasTransform) tabTotal++;
    if (hasAudio) tabTotal++;
    if (hasDuration) tabTotal++;
    if (hasAppearance) tabTotal++;
    if (hasAdjust) tabTotal++;
    return tabTotal;
  }, [hasLine, hasTransform, hasAudio, hasDuration, hasAppearance, hasAdjust]);

  

  useEffect(() => {
    // Automatically switch to line tab when a line is selected
    if (hasLine && selectedLineId) {
      setSelectedTab('line');
    }
  }, [selectedLineId, hasLine]);

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
    if (currentTab === "audio" && hasAudio) return "audio";
    if (currentTab === "duration" && hasDuration) return "duration";
    if (currentTab === "appearance" && hasAppearance) return "appearance";
    if (currentTab === "adjust" && hasAdjust) return "adjust";
    if (currentTab === "preprocessor-info" && hasValidPreprocessor) return "preprocessor-info";
    if (currentTab === "preprocessor-parameters" && hasValidPreprocessor) return "preprocessor-parameters";
    if (currentTab === "preprocessor-duration" && hasValidPreprocessor && hasPreprocessorDuration) return "preprocessor-duration";
    // If current tab is invalid, return first available tab
    if (hasValidPreprocessor) return "preprocessor-info";
    if (hasLine) return "line";
    if (hasTransform) return "transform";
    if (hasAudio) return "audio";
    if (hasAppearance) return "appearance";
    if (hasAdjust) return "adjust";
    return "duration"; // fallback
  };

  useEffect(() => {
    const validTab = getValidTab(selectedTab);
    if (validTab !== selectedTab) {
      setSelectedTab(validTab);
    }
  }, [clipId, hasLine, hasTransform, hasAudio, hasDuration, hasAppearance, hasAdjust, hasPreprocessorDuration]);



  const handleRunPreprocessor = useCallback(async () => {
    // get the preprocessor 
    if (!selectedPreprocessorId) return;
    clearJob(selectedPreprocessorId);
    const clip = getClipFromPreprocessorId(selectedPreprocessorId);
    if (!preprocessor) return;
    if (!clip) return;
    if (!clip.src) return;

    // need to convert our startFrame and endFrame back to where it would be with real FPS
    const clipMediaInfo = getMediaInfoCached(clip.src);
    const clipFps = clipMediaInfo?.stats.video?.averagePacketRate ?? 24;
    if (preprocessor.startFrame === undefined || preprocessor.endFrame === undefined) return;
    let { start: startFrameReal, end: endFrameReal } = toFrameRange(preprocessor.startFrame, preprocessor.endFrame, fps, clipFps, clipMediaInfo?.duration ?? 0);

    // need to scale startFrame to real fps
    const clipMediaStartFrame = Math.round(((clipMediaInfo?.startFrame ?? 0) / fps) * clipFps);
    
    startFrameReal += clipMediaStartFrame;
    endFrameReal += clipMediaStartFrame;
    
    const response = await runPreprocessor({
      start_frame: startFrameReal,
      end_frame: endFrameReal,
      preprocessor_name: preprocessor.preprocessor.id,
      input_path: clip.src,
      job_id: preprocessor.id,
      download_if_needed: true,
      params: preprocessor.values,
    });
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
    stopTracking(selectedPreprocessorId);
    clearJob(selectedPreprocessorId);
    
    // Update preprocessor status to idle
    updatePreprocessor(clip.clipId, preprocessor.id, { status: undefined });
    
    toast.info(`Preprocessor ${preprocessor.preprocessor.name} stopped`);
  }, [selectedPreprocessorId, getPreprocessorById, getClipFromPreprocessorId, stopTracking, clearJob, updatePreprocessor, preprocessor]);

  return (
    <div className="h-full w-full min-w-0 flex flex-col" style={{ position: 'relative', overflow: 'hidden' }}>
      <div className="overflow-hidden" style={{ height: hasValidPreprocessor ? 'calc(100% - 90px)' : '100%' }}>
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="min-w-0 relative flex flex-col h-full">
        <div className="relative flex-shrink-0 border-b border-brand-light/5 ">
          <LuChevronLeft onClick={() => {
            if (tabRef.current) {
              tabRef.current.scrollBy({ left: -96, behavior: 'smooth' });
            }
          }} className={cn("text-brand-light h-6 w-6 bg-brand-background/90 border border-brand-light/10 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute left-1 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollLeft ? "block" : "hidden")} />
          <TabsList ref={tabRef} style={{scrollbarWidth: 'none', msOverflowStyle: 'none'}} className={cn("bg-brand w-full rounded-b-none p-0 min-w-0 flex-shrink overflow-x-auto [&::-webkit-scrollbar]:hidden")}>
            {(hasValidPreprocessor) && <TabsTrigger value="preprocessor-info" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Info</TabsTrigger>}
            {(hasValidPreprocessor) && <TabsTrigger value="preprocessor-parameters" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Parameters</TabsTrigger>}
            {(hasValidPreprocessor && hasPreprocessorDuration) && <TabsTrigger value="preprocessor-duration" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Duration</TabsTrigger>}
            {(hasLine) && <TabsTrigger value="line" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Line</TabsTrigger>}
            {(hasText) && <TabsTrigger value="text" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Text</TabsTrigger>}
            {(hasTransform) && <TabsTrigger value="transform" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Transform</TabsTrigger>}
            {(hasAudio) && <TabsTrigger value="audio" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Audio</TabsTrigger>}
            {(hasDuration) && <TabsTrigger value="duration" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Duration</TabsTrigger>}
            {(hasAdjust) && <TabsTrigger value="adjust" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Adjust</TabsTrigger>}
            {(hasAppearance) && <TabsTrigger value="appearance" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Appearance</TabsTrigger>}
          </TabsList>
          <LuChevronRight onClick={() => {
            if (tabRef.current) {
              tabRef.current.scrollBy({ left: 96, behavior: 'smooth' });
            }
          }} className={cn("text-brand-light h-6 w-6 border border-brand-light/10 bg-brand-background/90 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute right-1 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollRight ? "block" : "hidden")} />
        </div>
        <ScrollArea className="flex-1 overflow-y-auto">
          <div style={{ paddingBottom: hasValidPreprocessor ? '20px' : '0' }}>
            {(hasValidPreprocessor && selectedPreprocessorId && preprocessor) && <TabsContent value="preprocessor-info" className="min-w-0 m-0">
              <PreprocessorInfoPanel preprocessor={preprocessor} />
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
          {(hasAudio) && <TabsContent value="audio" className="min-w-0 m-0">
            <AudioProperties clipId={clipId} />
          </TabsContent>}
          {(hasDuration) && <TabsContent value="duration" className="min-w-0 m-0">
            <DurationProperties clipId={clipId} />
          </TabsContent>}
          {(hasAdjust) && <TabsContent value="adjust" className="min-w-0 m-0">
            <AdjustProperties clipId={clipId} />
          </TabsContent>}
          {(hasAppearance) && <TabsContent value="appearance" className="min-w-0 m-0">
            <AppearanceProperties clipId={clipId} />
          </TabsContent>}
          </div>
        </ScrollArea>
      </Tabs>
      </div>
      {hasValidPreprocessor && (
        <div className="absolute bottom-0 left-0 right-0 p-5 bg-brand border-t border-brand-light/10" style={{ zIndex: 100, pointerEvents: 'auto' }}>
          <button
            onClick={preprocessor?.status === 'running' ? handleStopPreprocessor : handleRunPreprocessor}
            className="w-full py-3 px-6 rounded-lg font-semibold text-[13px] flex items-center justify-center gap-x-3 transition-all duration-200 shadow-lg hover:opacity-90"
            style={{
              backgroundColor: preprocessor?.status === 'running' ? '#DC2626' : '#A477C4',
              color: '#FFFFFF'
            }}
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
              <FaStop size={12} />  
            ) : (
              <FaPlay size={12} />
            )}
            <span>{preprocessor?.status === 'running' ? 'Stop Preprocessor' : 'Run Preprocessor'}</span>
          </button>
        </div>
      )}
    </div>
  )
}

export default ClipPropertiesPanel