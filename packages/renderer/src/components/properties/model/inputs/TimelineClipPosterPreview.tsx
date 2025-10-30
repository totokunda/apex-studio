import React, { useMemo } from 'react';
import { Stage, Layer, Group, Rect } from 'react-konva';
import { useClipStore } from '@/lib/clip';
import { useViewportStore } from '@/lib/viewport';
import { BASE_LONG_SIDE } from '@/lib/settings';
import { AnyClipProps } from '@/lib/types';
import VideoPreview from '@/components/preview/clips/VideoPreview';
import ImagePreview from '@/components/preview/clips/ImagePreview';
import ShapePreview from '@/components/preview/clips/ShapePreview';
import TextPreview from '@/components/preview/clips/TextPreview';
import DrawingPreview from '@/components/preview/clips/DrawingPreview';
import { getApplicatorsForClip } from '@/lib/applicator-utils';
import { useWebGLHaldClut } from '@/components/preview/webgl-filters';
import { useInputControlsStore } from '@/lib/inputControl';

const sortGroupChildren = (groupClip: AnyClipProps, allClips: AnyClipProps[]) => {
  const nested = ((groupClip as any).children as string[][] | undefined) ?? [];
  const childIdsFlat = nested.flat();
  // preserve nested order but ensure deterministic per-timeline order similar to fullscreen
  const children = childIdsFlat
    .map(id => allClips.find(c => c.clipId === id))
    .filter(Boolean) as AnyClipProps[];
  return children;
}

const TimelineClipPosterPreview: React.FC<{ clipId?: string, clip?: AnyClipProps, width: number, height: number, inputId?: string }>
  = ({ clipId, clip: clipOverride, width, height, inputId }) => {
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const getClipById = useClipStore((s) => s.getClipById);
  const clips = useClipStore((s) => s.clips);
  const clipWithinFrame = useClipStore((s) => s.clipWithinFrame);
  const haldClutInstance = useWebGLHaldClut();
  const focusFrame = useInputControlsStore((s) => s.getFocusFrame(inputId));

  const rectDims = useMemo(() => {
    const ratio = aspectRatio.width / aspectRatio.height;
    if (!Number.isFinite(ratio) || ratio <= 0) {
      return { rectWidth: 0, rectHeight: 0 };
    }
    return { rectWidth: BASE_LONG_SIDE * ratio, rectHeight: BASE_LONG_SIDE };
  }, [aspectRatio.width, aspectRatio.height]);

  const view = useMemo(() => {
    const { rectWidth, rectHeight } = rectDims;
    if (!rectWidth || !rectHeight || !width || !height) return { scale: 1, x: 0, y: 0 };
    const scaleX = width / rectWidth;
    const scaleY = height / rectHeight;
    const scale = Math.min(scaleX, scaleY);
    const x = (width - rectWidth * scale) / 2;
    const y = (height - rectHeight * scale) / 2;
    return { scale, x, y };
  }, [rectDims.rectWidth, rectDims.rectHeight, width, height]);

  const toRender = useMemo(() => {
    if (clipOverride) {
      if (clipOverride.type === 'group') return sortGroupChildren(clipOverride, clips);
      return [clipOverride];
    }
    if (!clipId) return [] as AnyClipProps[];
    const c = getClipById(clipId);
    if (!c) return [] as AnyClipProps[];
    if (c.type === 'group') return sortGroupChildren(c, clips);
    return [c];
  }, [clipOverride, clipId, getClipById, clips]);

  const getApplicators = React.useCallback((id: string) => {
    if (clipOverride) return [];
    return getApplicatorsForClip(id, { haldClutInstance });
  }, [haldClutInstance, clipOverride]);

  const { rectWidth, rectHeight } = rectDims;

  return (
    <div className="w-full h-auto rounded-[6px] flex flex-col items-center justify-start">
    <Stage key={`${width}x${height}`} width={width} height={height}>
      <Layer >
        <Group x={view.x} y={view.y} scaleX={view.scale} scaleY={view.scale} listening={false} >
          <Rect x={0} y={0} width={rectWidth} height={rectHeight} fill={'#000000'} />
          {toRender.map((clip) => {
            if (clip.type === 'group') return null;
            const startFrame = clip.startFrame || 0;
            const hasOverlap = (clip.type === 'video' || clip.type === 'image') && (startFrame > 0) ? true : false;
            const clipAtFrame = clipWithinFrame(clip, focusFrame, hasOverlap, 1);
              if (!clipAtFrame && clip.groupId) return null;
            const applicators = getApplicators(clip.clipId);
            switch (clip.type) {
              case 'video':
                return <VideoPreview key={clip.clipId} {...(clip as any)} overrideClip={clipOverride ? (clip) : undefined} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} overlap={true} inputMode={true} inputId={inputId} />
              case 'image':
                return <ImagePreview key={clip.clipId} {...(clip as any)} overrideClip={clipOverride ? (clip) : undefined} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} overlap={true} inputMode={true} inputId={inputId} />
              case 'shape':
                return <ShapePreview key={clip.clipId} {...(clip as any)} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} assetMode={true} />
              case 'text':
                return <TextPreview key={clip.clipId} {...(clip as any)} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} assetMode={true} />
              case 'draw':
                return <DrawingPreview key={clip.clipId} {...(clip as any)} rectWidth={rectWidth} rectHeight={rectHeight} assetMode={true} />
              default:
                return null;
            }
          })}
        </Group>
      </Layer>
    </Stage>
    </div>
  );
}

export default TimelineClipPosterPreview;

