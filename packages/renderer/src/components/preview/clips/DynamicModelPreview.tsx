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

  const info = useMemo(() => getMediaInfoCached(src), [src, tick]);

  useEffect(() => {
    let cancelled = false;
    if (!info && src) {
      (async () => {
        try {
          await getMediaInfo(src);
        } finally {
          if (!cancelled) setTick((v) => v + 1);
        }
      })();
    }
    return () => {
      cancelled = true;
    };
  }, [src, info]);

  if (info?.video) {
    return (
      <VideoPreview
        key={`${clip.clipId}:${src}`}
        {...(clip as any)}
        rectWidth={rectWidth}
        rectHeight={rectHeight}
        applicators={applicators}
        overlap={overlap}
      />
    );
  }
  if (info?.image) {
    return (
      <ImagePreview
        key={`${clip.clipId}:${src}`}
        {...(clip as any)}
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


