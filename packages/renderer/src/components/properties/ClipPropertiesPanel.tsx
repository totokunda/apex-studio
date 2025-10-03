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
import {LuChevronRight, LuChevronLeft} from 'react-icons/lu'
import { cn } from '@/lib/utils'  

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
  const [scrolledAmount, setScrolledAmount] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>("transform");
  

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
    if (clip?.type === 'image' || clip?.type === 'video' || clip?.type === 'shape') return true;
    return false;
  }, [clip?.type]);

  const hasTransform = useMemo(() => {
    if (clip?.type === 'image' || clip?.type === 'video' || clip?.type === 'shape') return true;
    return false;
  }, [clip?.type]);

  const numVisibleTabs = useMemo(() => {
    let tabTotal = 0;
    if (hasTransform) tabTotal++;
    if (hasAudio) tabTotal++;
    if (hasDuration) tabTotal++;
    if (hasAppearance) tabTotal++;
    return tabTotal;
  }, [hasTransform, hasAudio, hasDuration, hasAppearance]);

  const getValidTab = (currentTab: string) => {
    // Check if current tab is valid for this clip type
    if (currentTab === "transform" && hasTransform) return "transform";
    if (currentTab === "audio" && hasAudio) return "audio";
    if (currentTab === "duration" && hasDuration) return "duration";
    if (currentTab === "appearance" && hasAppearance) return "appearance";
    
    // If current tab is invalid, return first available tab
    if (hasTransform) return "transform";
    if (hasAudio) return "audio";
    if (hasDuration) return "duration";
    if (hasAppearance) return "appearance";
    return "transform"; // fallback
  };

  useEffect(() => {
    const validTab = getValidTab(selectedTab);
    if (validTab !== selectedTab) {
      setSelectedTab(validTab);
    }
  }, [clipId, hasTransform, hasAudio, hasDuration, hasAppearance]);

  useEffect(() => {
    const tabWidth = 96 * numVisibleTabs;
    const minScroll = tabWidth > panelSize ? -(tabWidth - panelSize) : 0;
    
    if (tabRef.current) {
      if (scrolledAmount < minScroll) {
        setScrolledAmount(minScroll);
      }
      setCanScrollLeft(scrolledAmount < 0);
      setCanScrollRight(scrolledAmount > minScroll);
    }
  }, [panelSize, scrolledAmount, numVisibleTabs]);

  return (
    <div  className="h-full w-full overflow-hidden min-w-0">
      <div className="flex flex-col divide-y divide-brand-light/5 pb-4 min-w-0 overflow-x-auto overflow-y-auto h-full">
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="min-w-0 relative">
        <ScrollArea>
          <LuChevronLeft onClick={() => {
            if (tabRef.current) {
              const tabWidth = 96 * numVisibleTabs;
              const minScroll = tabWidth > panelSize ? -(tabWidth - panelSize) : 0;
              const newAmount = Math.min(0, scrolledAmount + 48);
              setIsAnimating(true);
              setScrolledAmount(Math.max(minScroll, newAmount));
              setTimeout(() => setIsAnimating(false), 300);
            }
          }} className={cn("text-brand-light h-6 w-6 bg-brand-background/90 border border-brand-light/10 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute left-1 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollLeft ? "block" : "hidden")} />
        <TabsList ref={tabRef} style={{transform: `translateX(${scrolledAmount}px)`}} className={cn("bg-brand w-full rounded-b-none p-0 min-w-0 flex-shrink overflow-x-auto", isAnimating && "transition-all duration-300")}>
          {(hasTransform) && <TabsTrigger value="transform" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Transform</TabsTrigger>}
          {(hasAudio) && <TabsTrigger value="audio" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Audio</TabsTrigger>}
          {(hasDuration) && <TabsTrigger value="duration" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Duration</TabsTrigger>}
          {(hasAppearance) && <TabsTrigger value="appearance" className="text-brand-light text-xs h-10 flex-shrink-0 px-4 whitespace-nowrap">Appearance</TabsTrigger>}
        </TabsList>
        <LuChevronRight onClick={() => {
          if (tabRef.current) {
            const tabWidth = 96 * numVisibleTabs;
            const minScroll = tabWidth > panelSize ? -(tabWidth - panelSize) : 0;
            const newAmount = Math.max(minScroll, scrolledAmount - 48);
            setIsAnimating(true);
            setScrolledAmount(Math.min(0, newAmount));
            setTimeout(() => setIsAnimating(false), 300);
          }
        }} className={cn("text-brand-light h-6 w-6 border border-brand-light/10 bg-brand-background/90 hover:bg-brand-background/100 z-50 transition-all duration-200 rounded-full absolute right-1 top-1/2 -translate-y-1/2 p-1 cursor-pointer ", canScrollRight ? "block" : "hidden")} />
        </ScrollArea>
        {(hasTransform) && <TabsContent  value="transform" className="min-w-0 divide-y divide-brand-light/10">
          <PositionProperties clipId={clipId}  />
          <LayoutProperties clipId={clipId}  />
        </TabsContent>}
        {(hasAudio) && <TabsContent value="audio" className="min-w-0">
          <AudioProperties clipId={clipId} />
        </TabsContent>}
        {(hasDuration) && <TabsContent value="duration" className="min-w-0">
          <DurationProperties clipId={clipId} />
        </TabsContent>}
        {(hasAppearance) && <TabsContent value="appearance" className="min-w-0">
          <AppearanceProperties clipId={clipId} />
        </TabsContent>}
      </Tabs>
      </div>
    </div>
  )
}

export default ClipPropertiesPanel