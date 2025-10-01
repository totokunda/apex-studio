import { useClipStore } from '@/lib/clip'
import { useControlsStore } from '@/lib/control'
import AudioProperties from './AudioProperties'
import { useMemo } from 'react'
import {  getMediaInfoCached } from '@/lib/media/utils'
import DurationProperties from './DurationProperties'
// Speed 
// Position
// Volume
// Duration
// Animation

const ClipPropertiesPanel = () => {
  const clipId = useControlsStore((s) => s.selectedClipIds[0])
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
    return false
  }, [clipType, clip?.src]);

  return (
    <div className="flex flex-col divide-y divide-brand-light/10">
      {(hasAudio) && <AudioProperties clipId={clipId} />}
      {(hasDuration) && <DurationProperties clipId={clipId} />}
    </div>
  )
}

export default ClipPropertiesPanel