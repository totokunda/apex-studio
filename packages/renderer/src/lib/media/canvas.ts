import { MediaInfo } from "../types";
import { CanvasDecoderContext, CanvasDecoderKey } from "./types";
import { CanvasSink, WrappedCanvas } from "mediabunny";
import { nowMs } from "./utils";
import { FramesCache, MediaCache } from "./cache";
import { PREFETCH_BACK, PREFETCH_AHEAD } from "../settings";
import { buildFrameKey, getCachedSample, getCachedSamples } from "./cache";
import { canvasDecoders, pruneStaleDecoders } from "./utils";

function getOrCreateCanvasDecoder(
  path: string,
  mediaInfo: MediaInfo,
  width: number,
  height: number,
  canBeTransparent: boolean,
): CanvasDecoderContext | null {
  if (!mediaInfo?.video) return null;
  const targetW = Math.max(
    1,
    Math.floor(width || mediaInfo.video.displayWidth || 0),
  );
  const targetH = Math.max(
    1,
    Math.floor(height || mediaInfo.video.displayHeight || 0),
  );
  const key: CanvasDecoderKey = `${path}#canvas@${targetW}x${targetH}`;
  const existing = canvasDecoders.get(key);
  const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
  if (existing) {
    existing.lastAccessTs = nowMs();
    existing.frameRate = frameRate || existing.frameRate;
    return existing;
  }
  try {
    const sink = new CanvasSink(mediaInfo.video, {
      width: targetW,
      height: targetH,
      fit: "fill",
      alpha: canBeTransparent,
    });
    const ctx: CanvasDecoderContext = {
      sink,
      inFlight: new Set<number>(),
      lastAccessTs: nowMs(),
      frameRate: frameRate || 0,
      width: targetW,
      height: targetH,
    };
    canvasDecoders.set(key, ctx);
    // Prime decoder lightly (do not await)
    void (async () => {
      try {
        const ts = 0;
        await sink.getCanvas(ts);
      } catch {}
    })();
    return ctx;
  } catch (e) {
    console.warn("[media] Failed to create CanvasSink", { path, error: e });
    return null;
  }
}

function schedulePrefetchCanvas(
  path: string,
  mediaInfo: MediaInfo,
  centerFrame: number,
  width: number,
  height: number,
) {
  const ctx = getOrCreateCanvasDecoder(path, mediaInfo, width, height, false);
  if (!ctx) return;
  const frameRate =
    ctx.frameRate || mediaInfo.stats.video?.averagePacketRate || 0;
  if (!Number.isFinite(frameRate) || frameRate <= 0) return;
  const start = Math.max(0, Math.floor(centerFrame - PREFETCH_BACK));
  const end = Math.max(start, Math.floor(centerFrame + PREFETCH_AHEAD));
  const frameCache = FramesCache.getState();
  const tasks: number[] = [];
  for (let i = start; i <= end; i++) {
    if (i === centerFrame) continue;
    if (ctx.inFlight.has(i)) continue;
    const key = buildFrameKey(path, ctx.width, ctx.height, i, true);
    if (frameCache.has(key)) continue;
    tasks.push(i);
  }
  if (tasks.length === 0) return;
  tasks.forEach((f) => ctx.inFlight.add(f));
  void (async () => {
    try {
      const timestamps = tasks.map((f) => f / frameRate);
      const iterable = (await ctx.sink.canvasesAtTimestamps(
        timestamps,
      )) as AsyncIterable<WrappedCanvas | null>;
      const frameCache = FramesCache.getState();
      for await (const wrapped of iterable) {
        if (!wrapped) continue;
        const estIndex = Math.round(wrapped.timestamp * frameRate);
        const key = buildFrameKey(path, ctx.width, ctx.height, estIndex, true);
        frameCache.put(key, wrapped);
        ctx.inFlight.delete(estIndex);
      }
    } catch (e) {
      tasks.forEach((f) => ctx.inFlight.delete(f));
    } finally {
      pruneStaleDecoders();
    }
  })();
}

export const fetchCanvasSample = async (
  path: string,
  frameIndex: number,
  width?: number,
  height?: number,
  options?: { mediaInfo?: MediaInfo; poolSize?: number; prefetch?: boolean },
): Promise<WrappedCanvas | null> => {
  const frame = getCachedSample(path, frameIndex, width, height, true);
  if (frame) return frame as WrappedCanvas;

  const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
  if (!mediaInfo || !mediaInfo.video) return null;

  width = width || mediaInfo.video?.displayWidth || 0;
  height = height || mediaInfo.video?.displayHeight || 0;
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return null;
  }

  const key = buildFrameKey(path, width, height, frameIndex, true);
  const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
  if (!Number.isFinite(frameRate) || frameRate <= 0) {
    return null;
  }
  const timestamp = frameIndex / frameRate;

  let sample: WrappedCanvas | null = null;
  try {
    const videoCanBeTransparent = await mediaInfo.video.canBeTransparent();
    const decoder = getOrCreateCanvasDecoder(
      path,
      mediaInfo,
      width,
      height,
      videoCanBeTransparent,
    );
    if (!decoder) return null;
    decoder.lastAccessTs = nowMs();
    sample = await decoder.sink.getCanvas(timestamp);
    if (!sample) {
      for (const df of [1, -1, 2, -2]) {
        try {
          sample = await decoder.sink.getCanvas(timestamp + df / frameRate);
          if (sample) break;
        } catch {}
      }
    }
  } catch (error) {
    console.warn("[media] Failed to decode canvas sample", {
      path,
      frameIndex,
      error,
    });
    sample = null;
  }

  // cache the sample
  if (sample) {
    const frameCache = FramesCache.getState();
    frameCache.put(key, sample);
    const estIndex = Math.round(sample.timestamp * frameRate);
    if (estIndex !== frameIndex) {
      const altKey = buildFrameKey(path, width, height, estIndex, true);
      frameCache.put(altKey, sample);
    }
    if (options?.prefetch) {
      schedulePrefetchCanvas(path, mediaInfo, frameIndex, width, height);
    }
  }

  return sample;
};

export const fetchCanvasSamples = async (
  path: string,
  frameIndices: number[],
  width?: number,
  height?: number,
  options?: { mediaInfo?: MediaInfo },
): Promise<(WrappedCanvas | null)[]> => {
  // make samples a list of nulls to start with
  const samples = getCachedSamples(path, frameIndices, width, height, true);

  if (samples.every((sample) => sample !== null)) {
    return samples as (WrappedCanvas | null)[];
  }

  const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
  if (!mediaInfo || !mediaInfo.video)
    return samples as (WrappedCanvas | null)[];

  width = Math.max(1, Math.floor(width || mediaInfo.video?.displayWidth || 0));
  height = Math.max(
    1,
    Math.floor(height || mediaInfo.video?.displayHeight || 0),
  );
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return samples as (WrappedCanvas | null)[];
  }
  const videoCanBeTransparent = await mediaInfo.video.canBeTransparent();
  const decoder = getOrCreateCanvasDecoder(
    path,
    mediaInfo,
    width,
    height,
    videoCanBeTransparent,
  );
  if (!decoder) return samples as (WrappedCanvas | null)[];
  decoder.lastAccessTs = nowMs();

  const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
  if (!Number.isFinite(frameRate) || frameRate <= 0) {
    return samples as (WrappedCanvas | null)[];
  }
  // Figure out which positions are still missing (preserve duplicates and order)
  const missingPositions: number[] = [];
  const frameIndexToPositions = new Map<number, number[]>();
  for (let i = 0; i < frameIndices.length; i++) {
    if (!samples[i]) {
      missingPositions.push(i);
      const fi = frameIndices[i]!;
      const arr = frameIndexToPositions.get(fi) || [];
      arr.push(i);
      frameIndexToPositions.set(fi, arr);
    }
  }

  if (missingPositions.length === 0) {
    return samples as (WrappedCanvas | null)[];
  }

  // Request canvases for all missing positions in the same order as provided
  const requestedTimestamps: number[] = missingPositions.map(
    (pos) => frameIndices[pos]! / frameRate,
  );

  const frameCache = FramesCache.getState();

  let iterable: AsyncIterable<WrappedCanvas | null> | null = null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    iterable = await decoder.sink.canvasesAtTimestamps(requestedTimestamps);
  } catch (error) {
    console.warn("[media] Failed to request canvas samples at timestamps", {
      path,
      error,
    });
    iterable = null;
  }

  // Track which positions remain unassigned and any successfully decoded sample by requested frame index
  const unassigned = new Set<number>(missingPositions);
  const firstSampleByRequestedIndex = new Map<number, WrappedCanvas>();

  // Helper to assign a sample to a position and cache under requested and estimated indices
  const assignSample = (pos: number, sample: WrappedCanvas | null) => {
    if (!unassigned.has(pos)) return; // already filled
    const reqIndex = frameIndices[pos]!;
    if (sample) {
      samples[pos] = sample;
      firstSampleByRequestedIndex.set(
        reqIndex,
        firstSampleByRequestedIndex.get(reqIndex) || sample,
      );
      const estimatedIndex = Math.round(sample.timestamp * frameRate);
      // Cache under requested index
      const reqKey = buildFrameKey(path, width!, height!, reqIndex, true);
      frameCache.put(reqKey, sample);
      // Also cache under estimated index if different
      if (estimatedIndex !== reqIndex) {
        const estKey = buildFrameKey(
          path,
          width!,
          height!,
          estimatedIndex,
          true,
        );
        frameCache.put(estKey, sample);
      }
    } else {
      samples[pos] = null;
    }
    unassigned.delete(pos);
  };

  // Consume iterable if available; be robust to out-of-order yields
  if (iterable) {
    let j = 0;
    try {
      for await (const sample of iterable as AsyncIterable<WrappedCanvas | null>) {
        // Prefer mapping by exact/nearby estimated frame index when possible
        let posToFill: number | undefined;
        if (sample) {
          const est = Math.round(sample.timestamp * frameRate);
          let arr = frameIndexToPositions.get(est);
          if (!arr || arr.length === 0)
            arr = frameIndexToPositions.get(est - 1);
          if (!arr || arr.length === 0)
            arr = frameIndexToPositions.get(est + 1);
          if (arr && arr.length > 0) {
            posToFill = arr.shift()!;
            if (arr.length === 0) frameIndexToPositions.delete(est);
          }
        }
        // Fallback: assume same order as requested timestamps
        if (posToFill === undefined && j < missingPositions.length) {
          posToFill = missingPositions[j];
        }
        if (posToFill !== undefined) {
          assignSample(posToFill, sample);
          j++;
        }
      }
    } catch (error) {
      console.warn("[media] Error while iterating decoded canvas samples", {
        path,
        error,
      });
    }
  }

  // If any positions are still unassigned, try to reuse a decoded duplicate first
  if (unassigned.size > 0) {
    for (const pos of Array.from(unassigned)) {
      const fi = frameIndices[pos]!;
      const dup = firstSampleByRequestedIndex.get(fi);
      if (dup) assignSample(pos, dup);
    }
  }

  // As a last resort, individually fetch unresolved frames
  if (unassigned.size > 0) {
    await Promise.all(
      Array.from(unassigned).map(async (pos) => {
        try {
          const fi = frameIndices[pos]!;
          const single = await fetchCanvasSample(path, fi, width, height, {
            mediaInfo,
          });
          assignSample(pos, single);
        } catch {
          assignSample(pos, null);
        }
      }),
    );
  }

  return samples as (WrappedCanvas | null)[];
};

// Return nearest cached canvas samples synchronously for quick initial rendering.
// For each requested frame index, search the cache for the closest available
// frame within a bounded radius. If none is found within the max distance,
// return null for that position.
export const getNearestCachedCanvasSamples = (
  path: string,
  frameIndices: number[],
  width?: number,
  height?: number,
  options?: { mediaInfo?: MediaInfo },
): (WrappedCanvas | null)[] => {
  const mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);

  if (!mediaInfo || !mediaInfo.video) {
    return new Array(frameIndices.length).fill(null);
  }

  const targetW = Math.max(
    1,
    Math.floor(width || mediaInfo.video?.displayWidth || 0),
  );
  const targetH = Math.max(
    1,
    Math.floor(height || mediaInfo.video?.displayHeight || 0),
  );

  const results: (WrappedCanvas | null)[] = new Array(frameIndices.length).fill(
    null,
  );

  for (let i = 0; i < frameIndices.length; i++) {
    const fi = frameIndices[i]!;
    // Try exact match first
    let found = getCachedSample(
      path,
      fi,
      targetW,
      targetH,
      true,
    ) as WrappedCanvas | null;

    if (found && (found as any).canvas) {
      results[i] = found as WrappedCanvas;
      continue;
    }
    // Search outward without distance limit
    let assigned: WrappedCanvas | null = null;
    for (let d = 1; ; d++) {
      const left = fi - d;
      if (left >= 0) {
        const sL = getCachedSample(
          path,
          left,
          targetW,
          targetH,
          true,
        ) as WrappedCanvas | null;
        if (sL && (sL as any).canvas) {
          assigned = sL;
          break;
        }
      }
      const right = fi + d;
      const sR = getCachedSample(
        path,
        right,
        targetW,
        targetH,
        true,
      ) as WrappedCanvas | null;
      if (sR && (sR as any).canvas) {
        assigned = sR;
        break;
      }

      // Safety check to prevent infinite loop if no frames are cached
      if (left < 0 && right > 10000) break;
    }
    results[i] = assigned;
  }

  return results;
};
