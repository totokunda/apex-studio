import { createRoot, Root } from 'react-dom/client';
import { createRef } from 'react';
import OffscreenPosterStage, { OffscreenPosterStageHandle } from '@/components/preview/OffscreenPosterStage';
import { AnyClipProps } from '@/lib/types';
import { useClipStore } from '@/lib/clip';
import { getMediaInfo, getMediaInfoCached } from '@/lib/media/utils';
import { BASE_LONG_SIDE } from '@/lib/settings';
import { useViewportStore } from '@/lib/viewport';

export type OffscreenFrameResult = {
  frame: number;
  canvas: HTMLCanvasElement;
};

export type OffscreenRangeOptions = {
  width?: number;
  height?: number;
  startFrame: number;
  endFrame: number; // inclusive
  step?: number;
  pixelRatio?: number;
  quality?: number;
  clipId?: string;
  clip?: AnyClipProps;
  inputId?: string;
};

export type OffscreenSingleFrameOptions = Omit<OffscreenRangeOptions, 'startFrame' | 'endFrame' | 'step'> & {
  frame: number;
};

function createHiddenContainer(width: number, height: number): HTMLDivElement {
  const container = document.createElement('div');
  container.style.position = 'absolute';
  container.style.left = '-10000px';
  container.style.top = '-10000px';
  container.style.width = `${width}px`;
  container.style.height = `${height}px`;
  container.style.opacity = '0';
  container.style.pointerEvents = 'none';
  document.body.appendChild(container);
  return container;
}

async function ensureNextFrame() {
  await new Promise(requestAnimationFrame);
}

async function resolveDimensions(opts: { width?: number; height?: number; clip?: AnyClipProps; clipId?: string }): Promise<{ width: number; height: number; baseLongSide: number; ratioOverride: number }> {
  // Compute base short side from the actual media resolution.
  // If a group clip is provided, use the smallest resolution among its media children.
  const ratio = useViewportStore.getState().aspectRatio;
  const store = useClipStore.getState();
  const clip = opts.clip ?? (opts.clipId ? store.getClipById(opts.clipId) as AnyClipProps | undefined : undefined);

  // For grouped clips, ignore individual media resolutions and use the main editor's aspect ratio.
  if (clip && clip.type === 'group') {
    const chosenRatio = ratio.width / Math.max(1, ratio.height);
    // Determine base short side prioritizing provided dimensions, otherwise use project baseline.
    let baseShortSide = 0;
    if (typeof opts.height === 'number' && isFinite(opts.height) && opts.height > 0) {
      baseShortSide = Math.floor(opts.height);
    } else if (typeof opts.width === 'number' && isFinite(opts.width) && opts.width > 0) {
      baseShortSide = Math.max(1, Math.floor(opts.width / chosenRatio));
    } else {
      baseShortSide = BASE_LONG_SIDE;
    }
    const rectW = Math.max(1, Math.floor(baseShortSide * chosenRatio));
    const rectH = Math.max(1, Math.floor(baseShortSide));
    return { width: rectW, height: rectH, baseLongSide: baseShortSide, ratioOverride: chosenRatio };
  }

  const collectMediaClips = (c?: AnyClipProps): AnyClipProps[] => {
    if (!c) return [];
    if (c.type === 'group') {
      const nested = ((c as any).children as string[][] | undefined) ?? [];
      const ids = nested.flat();
      return ids
        .map(id => store.getClipById(id))
        .filter((mc): mc is AnyClipProps => !!mc && (mc.type === 'video' || mc.type === 'image'));
    }
    return (c.type === 'video' || c.type === 'image') ? [c] : [];
  };

  const mediaClips = collectMediaClips(clip);

  let baseShortSide = 0;
  let effectiveRatio: number | null = null;
  for (const mc of mediaClips) {
    const selectedSrc = (mc as any)?.preprocessors?.find((p: any) => p.status === 'complete')?.src ?? (mc as any)?.src;
    if (typeof selectedSrc !== 'string' || !selectedSrc) continue;
    let info = getMediaInfoCached(selectedSrc);
    if (!info) {
      try { info = await getMediaInfo(selectedSrc); } catch {}
    }
    const w = (mc as any)?.mediaWidth || info?.video?.displayWidth || info?.image?.width || 0;
    const h = (mc as any)?.mediaHeight || info?.video?.displayHeight || info?.image?.height || 0;
    if (!w || !h) continue;
    const shortSideCandidate = h; // matches rect formula where rectWidth = shortSide * (W/H), rectHeight = shortSide
    if (baseShortSide === 0) baseShortSide = shortSideCandidate;
    else baseShortSide = Math.min(baseShortSide, shortSideCandidate);
    if (effectiveRatio === null) {
      const clipRatio = (mc as any)?.mediaAspectRatio;
      effectiveRatio = (typeof clipRatio === 'number' && isFinite(clipRatio) && clipRatio > 0)
        ? clipRatio
        : (w / h);
    }
  }

  if (baseShortSide <= 0) {
    // Fallback to project baseline if no media info available
    baseShortSide = BASE_LONG_SIDE;
  }

  const chosenRatio = effectiveRatio === null
    ? (ratio.width / Math.max(1, ratio.height))
    : effectiveRatio;
  const rectW = Math.max(1, Math.floor(baseShortSide * chosenRatio));
  const rectH = Math.max(1, Math.floor(baseShortSide));

  return { width: rectW, height: rectH, baseLongSide: baseShortSide, ratioOverride: chosenRatio };
}

async function waitForNonEmptyFrame(ref: React.RefObject<OffscreenPosterStageHandle | null>, pixelRatio: number, quality = 1.0, maxTries = 30): Promise<HTMLCanvasElement | null> {
  for (let i = 0; i < maxTries; i++) {
    await ensureNextFrame();
    const canvas = await ref.current?.toCanvas(pixelRatio, quality);
    if (!canvas) continue;
    const ctx = canvas.getContext('2d');
    if (!ctx) continue;
    const { width, height } = canvas;
    if (width === 0 || height === 0) continue;
    const sampleSize = Math.min(32, width, height);
    const img = ctx.getImageData(0, 0, sampleSize, sampleSize);
    const data = img.data;
    let nonBlack = false;
    for (let p = 0; p < data.length; p += 4) {
      const r = data[p], g = data[p + 1], b = data[p + 2], a = data[p + 3];
      if (a !== 0 && (r !== 0 || g !== 0 || b !== 0)) { nonBlack = true; break; }
    }
    if (nonBlack) return canvas;
  }
  // last attempt
  return (await ref.current?.toCanvas(pixelRatio, quality)) ?? null;
}

export async function* renderOffscreenFrames(options: OffscreenRangeOptions): AsyncGenerator<OffscreenFrameResult> {
  const { startFrame, endFrame, step = 1, pixelRatio = 1, quality = 1.0, clipId, clip, inputId } = options;
  const { width, height, baseLongSide, ratioOverride } = await resolveDimensions({ width: options.width, height: options.height, clip, clipId });
  const container = createHiddenContainer(width, height);
  let root: Root | null = null;
  const stageRef = createRef<OffscreenPosterStageHandle>();

  // Simple frame queue to bridge callback -> async generator
  const queue: OffscreenFrameResult[] = [];
  let resolveNext: ((v: OffscreenFrameResult | null) => void) | null = null;
  let finished = false;

  const push = (item: OffscreenFrameResult) => {
    if (resolveNext) {
      const r = resolveNext;
      resolveNext = null;
      r(item);
    } else {
      queue.push(item);
    }
  };

  const nextItem = async (): Promise<OffscreenFrameResult | null> => {
    if (queue.length > 0) return queue.shift() as OffscreenFrameResult;
    if (finished) return null;
    return await new Promise<OffscreenFrameResult | null>((resolve) => {
      resolveNext = resolve;
    });
  };

  try {
    root = createRoot(container);
    root.render(
      <OffscreenPosterStage
        ref={stageRef}
        width={width}
        height={height}
        startFrame={startFrame}
        endFrame={endFrame}
        step={step}
        inputId={inputId}
        clipId={clipId}
        clip={clip}
        baseLongSide={baseLongSide}
        ratioOverride={ratioOverride}
        // Fast, no delay, stream frames via callback
        offscreenFast={true}
        noDelay={true}
        onFrame={async (f, _canvas) => {
          // Ensure we read a fresh canvas after draw commit at requested pixelRatio/quality
          const canvas = await stageRef.current?.toCanvas(pixelRatio, quality);
          if (canvas) push({ frame: f, canvas });
        }}
        onEnd={() => { finished = true; if (resolveNext) { const r = resolveNext; resolveNext = null; r(null); } }}
      />
    );

    while (true) {
      const item = await nextItem();
      if (!item) break;
      yield item;
    }
  } finally {
    try {
      root?.unmount();
    } catch {}
    if (container.parentNode) container.parentNode.removeChild(container);
  }
}

export async function renderOffscreenFrame(options: OffscreenSingleFrameOptions): Promise<HTMLCanvasElement | null> {
  const { frame, pixelRatio = 1, quality = 1.0, clipId, clip, inputId } = options;
  const { width, height, baseLongSide, ratioOverride } = await resolveDimensions({ width: options.width, height: options.height, clip, clipId });
  const container = createHiddenContainer(width, height);
  const root = createRoot(container);
  const stageRef = createRef<OffscreenPosterStageHandle>();
  try {
    root.render(
      <OffscreenPosterStage
        ref={stageRef}
        width={width}
        height={height}
        frame={frame}
        inputId={inputId}
        clipId={clipId}
        clip={clip}
        baseLongSide={baseLongSide}
        ratioOverride={ratioOverride}
      />
    );
    const canvas = await waitForNonEmptyFrame(stageRef, pixelRatio, quality);
    return canvas ?? null;
  } finally {
    try {
      root.unmount();
    } catch {}
    if (container.parentNode) container.parentNode.removeChild(container);
  }
}


