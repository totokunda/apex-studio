import React, { useEffect, useMemo, useState } from 'react';
import { AnyClipProps } from '@/lib/types';
import { getMediaInfo, getMediaInfoCached } from '@/lib/media/utils';
import VideoPreview from './VideoPreview';
import ImagePreview from './ImagePreview';
import { BaseClipApplicator } from './apply/base';

interface DynamicModelPreviewProps {
  clip: AnyClipProps;
  rectWidth: number;
  rectHeight: number;
  applicators: BaseClipApplicator[];
  overlap: boolean;
}

const DynamicModelPreview: React.FC<DynamicModelPreviewProps> = ({ clip, rectWidth, rectHeight, applicators, overlap }) => {
  const src = (clip as any)?.src || '';
  const [tick, setTick] = useState(0);
  const [activeSrc, setActiveSrc] = useState(src);

  // Resolve info for the active source only; keep showing previous src until new info is ready
  const info = useMemo(() => getMediaInfoCached(activeSrc), [activeSrc, tick]);

  // Fallback type guess by file extension while media info is being resolved
  const typeGuess = useMemo(() => {
    const normalized = (activeSrc || '').split('?')[0].split('#')[0].toLowerCase();
    if (/\.(mp4|mov|webm|m4v|avi|mkv)$/.test(normalized)) return 'video';
    if (/\.(png|jpg|jpeg|webp|bmp|gif)$/.test(normalized)) return 'image';
    return null;
  }, [activeSrc]);

  // When src changes, prefetch media info for the new src and switch only when ready
  useEffect(() => {
    let cancelled = false;
    if (!src || src === activeSrc) return () => { cancelled = true };
    const cached = getMediaInfoCached(src);
    if (cached) {
      if (!cancelled) {
        setActiveSrc(src);
        setTick((v) => v + 1);
      }
      return () => { cancelled = true };
    }
    (async () => {
      try {
        await getMediaInfo(src);
      } finally {
        if (!cancelled) {
          setActiveSrc(src);
          setTick((v) => v + 1);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [src]);

  if (info?.video || (!info && typeGuess === 'video' && activeSrc)) {
    return (
      <VideoPreview
        key={`${clip.clipId}:${activeSrc}`}
        {...({ ...(clip as any), src: activeSrc } as any)}
        rectWidth={rectWidth}
        rectHeight={rectHeight}
        applicators={applicators}
        overlap={overlap}
      />
    );
  }
  if (info?.image || (!info && typeGuess === 'image' && activeSrc)) {
    return (
      <ImagePreview
        key={`${clip.clipId}:${activeSrc}`}
        {...({ ...(clip as any), src: activeSrc } as any)}
        rectWidth={rectWidth}
        rectHeight={rectHeight}
        applicators={applicators}
        overlap={overlap}
      />
    );
  }
  // While probing type, render nothing; effect above will trigger rerender when info is ready
  return null;
};

export default DynamicModelPreview;


