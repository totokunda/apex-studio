import React, { useCallback, useMemo } from 'react';
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

// Kept for parity with other components if needed later; not used in poster sorting
// const getGroupChildren = (groupClip: AnyClipProps, allClips: AnyClipProps[]) => {
//   const nested = ((groupClip as any).children as string[][] | undefined) ?? [];
//   const childIdsFlat = nested.flat();
//   const children = childIdsFlat
//     .map(id => allClips.find(c => c.clipId === id))
//     .filter(Boolean) as AnyClipProps[];
//   return children;
// }

const TimelineClipPosterPreview: React.FC<{ clipId?: string, clip?: AnyClipProps, width: number, height: number, inputId?: string }>
  = ({ clipId, clip: clipOverride, width, height, inputId }) => {
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const getClipById = useClipStore((s) => s.getClipById);
  const clips = useClipStore((s) => s.clips);
  const clipWithinFrame = useClipStore((s) => s.clipWithinFrame);
  const haldClutInstance = useWebGLHaldClut();
  const focusFrame = useInputControlsStore((s) => s.getFocusFrame(inputId) ?? 0);

  const timelines = useClipStore((s) => s.timelines);


  const sortClips = useCallback((clipsToSort: AnyClipProps[]) => {
    // Treat each group as a single sortable unit; then expand children in defined order
    type GroupUnit = { kind: 'group'; id: string; y: number; start: number; children: AnyClipProps[] };
    type SingleUnit = { kind: 'single'; y: number; start: number; clip: AnyClipProps };

    const groups = clipsToSort.filter(c => c.type === 'group') as AnyClipProps[];
    const childrenSet = new Set<string>(
      groups.flatMap(g => {
        const nested = ((g as any).children as string[][] | undefined) ?? [];
        return nested.flat();
      })
    );

    // Build group units
    const groupUnits: GroupUnit[] = groups.map((g) => {
      const y = (timelines.find(t => t.timelineId === g.timelineId)?.timelineY) ?? 0;
      const start = g.startFrame ?? 0;
      const nested = ((g as any).children as string[][] | undefined) ?? [];
      const childIdsFlat = nested.flat();
      const children = childIdsFlat
        .map(id => clips.find(c => c.clipId === id))
        .filter(Boolean) as AnyClipProps[];
      return { kind: 'group', id: g.clipId, y, start, children };
    });

    // Build single units for non-group, non-child clips
    const singleUnits: SingleUnit[] = clipsToSort
      .filter(c => c.type !== 'group' && !childrenSet.has(c.clipId))
      .map((c) => {
        const y = (timelines.find(t => t.timelineId === c.timelineId)?.timelineY) ?? 0;
        const start = c.startFrame ?? 0;
        return { kind: 'single', y, start, clip: c };
      });

    // Sort units: lower on screen first (higher y), then earlier start
    const units = [...groupUnits, ...singleUnits].sort((a, b) => {
      if (a.y !== b.y) return b.y - a.y;
      return a.start - b.start;
    });

    // Flatten units back to clip list; for groups, expand children in their defined order
    const result: AnyClipProps[] = [];
    for (const u of units) {
      if (u.kind === 'single') {
        result.push(u.clip);
      } else {
        // Within a group, render lower timelines first (higher y), then earlier starts
        // Ensure children are ordered as in group's children list (reversed like main Preview)
        result.push(...u.children.reverse());
      }
    }

    return result;
  }, [timelines, clips])

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
      if (clipOverride.type === 'group') {
        // Let sorter handle group flattening so child order matches main Preview
        return sortClips([clipOverride]);
      }
      return [clipOverride];
    }
    if (!clipId) return [] as AnyClipProps[];
    const c = getClipById(clipId);
    if (!c) return [] as AnyClipProps[];
    if (c.type === 'group') {
      return sortClips([c]);
    }
    return [c];
  }, [clipOverride, clipId, getClipById, clips, sortClips]);

  const getApplicators = React.useCallback((id: string, frameOverride?: number) => {
    return getApplicatorsForClip(id, { haldClutInstance, focusFrameOverride: frameOverride });
  }, [haldClutInstance]);

  const { rectWidth, rectHeight } = rectDims;

  return (
    <div className="w-full h-auto rounded-[6px] flex flex-col items-center justify-start">
    <Stage key={`${width}x${height}:${clipOverride?.clipId || clipId || 'none'}`} width={width} height={height}>
      <Layer >
        <Group x={view.x} y={view.y} scaleX={view.scale} scaleY={view.scale} listening={false} >
          <Rect x={0} y={0} width={rectWidth} height={rectHeight} fill={'#000000'} />
          {toRender.map((clip) => {
            if (clip.type === 'group') return null;
            const startFrame = clip.startFrame || 0;
            const groupStart = clip.groupId ? (getClipById(clip.groupId)?.startFrame || 0) : 0;
            const relativeStart = startFrame - groupStart;
            const hasOverlap = (clip.type === 'video' || clip.type === 'image') && ((clip.groupId ? relativeStart : startFrame) > 0) ? true : false;
            const effectiveFrame = clip.groupId ? (focusFrame + groupStart) : focusFrame;
            const clipAtFrame = clipWithinFrame(clip, effectiveFrame, hasOverlap, 0);
            if (!clipAtFrame && clip.groupId) return null;

            const applicators = getApplicators(clip.clipId, effectiveFrame);
            // Use clipOverride only when an explicit override group is provided
            const overrideToUse = (clipOverride ? (clip) : undefined);

            switch (clip.type) {
              case 'video':
                return <VideoPreview key={clip.clipId} {...(clip as any)} overrideClip={overrideToUse} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} overlap={true} inputMode={true} inputId={inputId} />
              case 'image':
                return <ImagePreview key={clip.clipId} {...(clip as any)} overrideClip={overrideToUse} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} overlap={true} inputMode={true} inputId={inputId} />
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

