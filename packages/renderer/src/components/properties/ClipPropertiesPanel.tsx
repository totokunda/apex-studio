import { useClipStore } from '@/lib/clip'
import { useControlsStore } from '@/lib/control'
import AudioProperties from './AudioProperties'
import { useMemo, useRef, useEffect, useState } from 'react'
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

interface PropertiesPanelProps {
    panelSize: number;
}

const ClipPropertiesPanel:React.FC<PropertiesPanelProps> = ({panelSize}) => {
  const {selectedClipIds} = useControlsStore();
  const clipId = useMemo(() => selectedClipIds[selectedClipIds.length - 1], [selectedClipIds]);

  const clip = useClipStore((s) => s.getClipById(clipId))
  const clipType = clip?.type;
  const tabRef = useRef<HTMLDivElement>(null);  
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>(clip?.type === 'text' ? "text" : "transform");
  

  // check if clip has audio if it is video 
  const hasDuration = useMemo(() => {
    if (clip?.type === 'video' || clip?.type === 'audio') {
      return true;
    }
    return false
  }, [clip?.type, clip?.src]);

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

  const numVisibleTabs = useMemo(() => {
    let tabTotal = 0;
    if (hasTransform) tabTotal++;
    if (hasAudio) tabTotal++;
    if (hasDuration) tabTotal++;
    if (hasAppearance) tabTotal++;
    if (hasAdjust) tabTotal++;
    return tabTotal;
  }, [hasTransform, hasAudio, hasDuration, hasAppearance, hasAdjust]);

  const getValidTab = (currentTab: string) => {
    // Check if current tab is valid for this clip type
    if (currentTab === "text" && hasText) return "text";
    if (currentTab === "transform" && hasTransform) return "transform";
    if (currentTab === "audio" && hasAudio) return "audio";
    if (currentTab === "duration" && hasDuration) return "duration";
    if (currentTab === "appearance" && hasAppearance) return "appearance";
    if (currentTab === "adjust" && hasAdjust) return "adjust";
    // If current tab is invalid, return first available tab
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
  }, [clipId, hasTransform, hasAudio, hasDuration, hasAppearance, hasAdjust]);

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

  return (
    <div className="h-full w-full overflow-hidden min-w-0 flex flex-col">
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="min-w-0 relative flex flex-col h-full">
        <div className="relative flex-shrink-0 border-b border-brand-light/5 ">
          <LuChevronLeft onClick={() => {
            if (tabRef.current) {
              tabRef.current.scrollBy({ left: -96, behavior: 'smooth' });
            }
          }} className={cn("text-brand-light h-6 w-6 bg-brand-background/90 border border-brand-light/10 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute left-1 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollLeft ? "block" : "hidden")} />
          <TabsList ref={tabRef} style={{scrollbarWidth: 'none', msOverflowStyle: 'none'}} className={cn("bg-brand w-full rounded-b-none p-0 min-w-0 flex-shrink overflow-x-auto [&::-webkit-scrollbar]:hidden")}>
            {(hasText) && <TabsTrigger value="text" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Text</TabsTrigger>}
            {(hasTransform) && <TabsTrigger value="transform" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Transform</TabsTrigger>}
            {(hasAudio) && <TabsTrigger value="audio" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Audio</TabsTrigger>}
            <TabsTrigger value="duration" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Duration</TabsTrigger>
            {(hasAdjust) && <TabsTrigger value="adjust" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Adjust</TabsTrigger>}
            {(hasAppearance) && <TabsTrigger value="appearance" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Appearance</TabsTrigger>}
          </TabsList>
          <LuChevronRight onClick={() => {
            if (tabRef.current) {
              tabRef.current.scrollBy({ left: 96, behavior: 'smooth' });
            }
          }} className={cn("text-brand-light h-6 w-6 border border-brand-light/10 bg-brand-background/90 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute right-1 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollRight ? "block" : "hidden")} />
        </div>
        <ScrollArea className="flex-1">
          {(hasText) && <TabsContent value="text" className="min-w-0 m-0">  <TextProperties clipId={clipId} /> </TabsContent>}
          {(hasTransform) && <TabsContent  value="transform" className="min-w-0 divide-y divide-brand-light/10 m-0">
            <PositionProperties clipId={clipId}  />
            <LayoutProperties clipId={clipId}  />
          </TabsContent>}
          {(hasAudio) && <TabsContent value="audio" className="min-w-0 m-0">
            <AudioProperties clipId={clipId} />
          </TabsContent>}
          <TabsContent value="duration" className="min-w-0 m-0">
            <DurationProperties clipId={clipId} />
          </TabsContent>
          {(hasAdjust) && <TabsContent value="adjust" className="min-w-0 m-0">
            <AdjustProperties clipId={clipId} />
          </TabsContent>}
          {(hasAppearance) && <TabsContent value="appearance" className="min-w-0 m-0">
            <AppearanceProperties clipId={clipId} />
          </TabsContent>}
          
        </ScrollArea>
      </Tabs>
    </div>
  )
}

export default ClipPropertiesPanel