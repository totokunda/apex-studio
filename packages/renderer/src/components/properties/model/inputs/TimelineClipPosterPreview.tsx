import React, { useCallback, useMemo, useEffect, useRef, useState } from 'react';
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
import AudioPreview from '@/components/preview/clips/AudioPreview';
import { getMediaInfo } from '@/lib/media/utils';

// Kept for parity with other components if needed later; not used in poster sorting
// const getGroupChildren = (groupClip: AnyClipProps, allClips: AnyClipProps[]) => {
//   const nested = ((groupClip as any).children as string[][] | undefined) ?? [];
//   const childIdsFlat = nested.flat();
//   const children = childIdsFlat
//     .map(id => allClips.find(c => c.clipId === id))
//     .filter(Boolean) as AnyClipProps[];
//   return children;
// }

const TimelineClipPosterPreview: React.FC<{ clipId?: string, clip?: AnyClipProps, width: number, height: number, inputId?: string, ratioOverride?: number, audioOnly?: boolean }>
  = ({ clipId, clip: clipOverride, width, height, inputId, ratioOverride, audioOnly = false }) => {
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const getClipById = useClipStore((s) => s.getClipById);
  const clips = useClipStore((s) => s.clips);
  const clipWithinFrame = useClipStore((s) => s.clipWithinFrame);
  const haldClutInstance = useWebGLHaldClut();
  const focusFrame = useInputControlsStore((s) => s.getFocusFrame(inputId) ?? 0);

  const timelines = useClipStore((s) => s.timelines);
  const isOnTimeline = useMemo(() => {
    return getClipById(clipId ?? '') !== undefined && clipId !== undefined;
  }, [clipId, getClipById]);

  const ratioCacheByClipIdRef = useRef<Record<string, number>>({});
  const [contentRatio, setContentRatio] = useState<number | null>(null);

  const rootClip = useMemo<AnyClipProps | null>(() => {
    if (clipOverride) return clipOverride as AnyClipProps;
    if (!clipId) return null;
    return getClipById(clipId) as AnyClipProps | null;
  }, [clipOverride, clipId, getClipById]);

  useEffect(() => {
    const clip = rootClip;
    if (!clip) {
      setContentRatio(null);
      return;
    }
    if (clip.type === 'group') {
      setContentRatio(null);
      return;
    }
    const cached = ratioCacheByClipIdRef.current[clip.clipId];
    if (typeof cached === 'number' && cached > 0) {
      setContentRatio(cached);
      return;
    }
    const src = (clip as any)?.src as string | undefined;
    if (!src) {
      setContentRatio(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const info = await getMediaInfo(src);
        const w = (info as any)?.stats?.video?.width ?? (info as any)?.stats?.image?.width ?? (info as any)?.width ?? (info as any)?.streams?.[0]?.width;
        const h = (info as any)?.stats?.video?.height ?? (info as any)?.stats?.image?.height ?? (info as any)?.height ?? (info as any)?.streams?.[0]?.height;
        const r = Number(w) / Number(h);
        if (!cancelled && Number.isFinite(r) && r > 0) {
          ratioCacheByClipIdRef.current[clip.clipId] = r;
          setContentRatio(r);
        }
      } catch {
        if (!cancelled) setContentRatio(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [rootClip]);


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
    const editorRatio = aspectRatio.width / aspectRatio.height;
    const ratio = typeof ratioOverride === 'number' && ratioOverride > 0
      ? ratioOverride
      : (!isOnTimeline && typeof contentRatio === 'number' && contentRatio > 0)
        ? contentRatio
        : editorRatio;
    if (!Number.isFinite(ratio) || ratio <= 0) {
      return { rectWidth: 0, rectHeight: 0 };
    }
    return { rectWidth: BASE_LONG_SIDE * ratio, rectHeight: BASE_LONG_SIDE };
  }, [aspectRatio.width, aspectRatio.height, rootClip, contentRatio, ratioOverride]);

  const view = useMemo(() => {
    const { rectWidth, rectHeight } = rectDims;
    if (!rectWidth || !rectHeight || !width || !height) return { scale: 1, x: 0, y: 0 };
    const scaleX = width / rectWidth;
    const scaleY = height / rectHeight;
    // Use cover scaling so posters fill the container
    const scale = Math.max(scaleX, scaleY);
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
    <Stage key={`${clipOverride?.clipId || clipId || 'none'}${audioOnly ? ':audio' : ''}`} width={audioOnly? 1:width} height={audioOnly? 1:height}>
      <Layer >
        <Group x={view.x} y={view.y} scaleX={view.scale} scaleY={view.scale} listening={false} >
          <Rect x={0} y={0} width={rectWidth} height={rectHeight} fill={'#000000'} />
          {audioOnly ? null : toRender.map((clip) => {
            if (clip.type === 'group') return null;
            const startFrame = clip.startFrame || 0;
            const groupStart = clip.groupId ? (getClipById(clip.groupId)?.startFrame || 0) : 0;
            const relativeStart = startFrame - groupStart;
            const hasOverlap = (clip.type === 'video' || clip.type === 'image') && ((clip.groupId ? relativeStart : startFrame) > 0) ? true : false;
            // Use global frame for bounds/effects, but keep local focus for child playback math
            const effectiveGlobalFrame = clip.groupId ? (focusFrame + groupStart) : focusFrame;
            const clipAtFrame = clipWithinFrame(clip, effectiveGlobalFrame, hasOverlap, 0);
            if (!clipAtFrame && clip.groupId) return null;

            const applicators = getApplicators(clip.clipId, effectiveGlobalFrame);
            // Use clipOverride only when an explicit override group is provided
            const overrideToUse = (clipOverride ? (clip) : undefined);

            if (overrideToUse && overrideToUse.transform && (overrideToUse.type === 'image' || overrideToUse.type === 'video') && (overrideToUse.groupId === undefined)) {
              const t = overrideToUse.transform as any;
              const rawW = (t.width ?? rectWidth);
              const rawH = (t.height ?? rectHeight);
              const sx = (typeof t.scaleX === 'number' ? t.scaleX : 1);
              const sy = (typeof t.scaleY === 'number' ? t.scaleY : 1);
              const w = rawW * sx;
              const h = rawH * sy;
              const deg = (typeof t.rotation === 'number' ? t.rotation : 0);
              const rad = deg * Math.PI / 180;
              const c = Math.cos(rad);
              const s = Math.sin(rad);
              // corners after rotation around top-left pivot (0,0)
              const x1 = w * c;           const y1 = w * s;
              const x2 = -h * s;          const y2 = h * c;
              const x3 = w * c - h * s;   const y3 = w * s + h * c;
              const minX = Math.min(0, x1, x2, x3);
              const maxX = Math.max(0, x1, x2, x3);
              const minY = Math.min(0, y1, y2, y3);
              const maxY = Math.max(0, y1, y2, y3);
              const aabbW = maxX - minX;
              const aabbH = maxY - minY;
              t.x = (rectWidth - aabbW) / 2 - minX;
              t.y = (rectHeight - aabbH) / 2 - minY;
            }

            switch (clip.type) {
              case 'video':
                return <VideoPreview key={clip.clipId} {...(clip as any)} overrideClip={overrideToUse} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} overlap={true} inputMode={true} inputId={inputId} focusFrameOverride={focusFrame} />
              case 'image':
                return <ImagePreview key={clip.clipId} {...(clip as any)} overrideClip={overrideToUse} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} overlap={true} inputMode={true} inputId={inputId} focusFrameOverride={focusFrame} />
              case 'shape':
                return <ShapePreview key={clip.clipId} {...(clip as any)} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} assetMode={true} />
              case 'text':
                return <TextPreview key={clip.clipId} {...(clip as any)} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} assetMode={true} />
              case 'draw':
                return <DrawingPreview key={clip.clipId} {...(clip as any)} rectWidth={rectWidth} rectHeight={rectHeight} assetMode={true} applicators={applicators} />
              default:
                return null;
            }
          })}
          {toRender.map((clip) => {
            if (clip.type !== 'video' && clip.type !== 'audio') return null;
            const startFrame = clip.startFrame || 0;
            const groupStart = clip.groupId ? (getClipById(clip.groupId)?.startFrame || 0) : 0;
            const relativeStart = startFrame - groupStart;
            const hasOverlap = ((clip.groupId ? relativeStart : startFrame) > 0) ? true : false;
            const effectiveGlobalFrame = clip.groupId ? (focusFrame + groupStart) : focusFrame;
            const clipAtFrame = clipWithinFrame(clip, effectiveGlobalFrame, hasOverlap, 0);
            if (!clipAtFrame && clip.groupId) return null;
            const overrideToUse = (clipOverride ? (clip) : undefined);
            return <AudioPreview key={clip.clipId} {...(clip as any)} overrideClip={overrideToUse} overlap={hasOverlap} rectWidth={rectWidth} rectHeight={rectHeight} inputMode={true} inputId={inputId} />
          })}
        </Group>
      </Layer>
    </Stage>
    </div>
  );
}

export default TimelineClipPosterPreview;

