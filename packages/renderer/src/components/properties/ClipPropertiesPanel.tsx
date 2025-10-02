import { useClipStore } from '@/lib/clip'
import { useControlsStore } from '@/lib/control'
import AudioProperties from './AudioProperties'
import { useMemo } from 'react'
import {  getMediaInfoCached } from '@/lib/media/utils'
import DurationProperties from './DurationProperties'
import PositionProperties from './PositionProperties'
import LayoutProperties from './LayoutProperties'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

const ClipPropertiesPanel = () => {
  const clipId = useControlsStore((s) => s.selectedClipIds[s.selectedClipIds.length - 1])
  const clip = useClipStore((s) => s.getClipById(clipId))
  const clipType = clip?.type;
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

  return (
    <ScrollArea className="h-full w-full dark">
      <div className="flex flex-col divide-y divide-brand-light/5 pb-4">
      <Tabs defaultValue="transform">
        <TabsList className="bg-brand w-full rounded-b-none p-0">
        <TabsTrigger value="transform" className="text-brand-light text-xs h-10">Transform</TabsTrigger>
          {(hasAudio) && <TabsTrigger value="audio" className="text-brand-light text-xs h-10">Audio</TabsTrigger>}
          {(hasDuration) && <TabsTrigger value="duration" className="text-brand-light text-xs h-10">Duration</TabsTrigger>}
        </TabsList>
        <TabsContent value="transform">
          <PositionProperties clipId={clipId} />
          <LayoutProperties clipId={clipId} />
        </TabsContent>
        {(hasAudio) && <TabsContent value="audio">
          <AudioProperties clipId={clipId} />
        </TabsContent>}
        {(hasDuration) && <TabsContent value="duration">
          <DurationProperties clipId={clipId} />
        </TabsContent>}
      </Tabs>
      </div>
    </ScrollArea>
  )
}

export default ClipPropertiesPanel