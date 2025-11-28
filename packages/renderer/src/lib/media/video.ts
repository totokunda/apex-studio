import { MediaInfo } from "../types";
import { VideoDecoderContext, VideoDecoderKey } from "./types";
import { CanvasSink, WrappedCanvas } from "mediabunny";
import { getMediaInfo, nowMs } from "./utils";
import { MediaCache } from "./cache";
import { videoDecoders, pruneStaleDecoders } from "./utils";

function getOrCreateVideoDecoder(
  path: string,
  mediaInfo: MediaInfo,
  canBeTransparent: boolean,
  isolationKey?: string,
): VideoDecoderContext | null {
  if (!mediaInfo?.video) return null;
  // When an isolationKey is provided, create an independent decoder per key to avoid cross-component interference
  const key: VideoDecoderKey = isolationKey
    ? `${path}#video#${isolationKey}`
    : `${path}#video`;

  const existing = videoDecoders.get(key);
  const frameRate = mediaInfo.stats.video?.averagePacketRate || 0;
  if (existing) {
    existing.lastAccessTs = nowMs();
    existing.frameRate = frameRate || existing.frameRate;
    return existing;
  }
  try {
    const sink = new CanvasSink(mediaInfo.video, {
      poolSize: 2,
      fit: "contain", // In case the video changes dimensions over time
      alpha: canBeTransparent,
    });

    const ctx: VideoDecoderContext = {
      sink,
      inFlight: new Set<number>(),
      lastAccessTs: nowMs(),
      frameRate: frameRate || 0,
    };
    videoDecoders.set(key, ctx);
    // Prime decoder lightly (do not await)
    void (async () => {
      try {
        const ts = 0;
        await sink.getCanvas(ts);
      } catch {}
    })();
    return ctx;
  } catch (e) {
    console.warn("[media] Failed to create VideoSampleSink", {
      path,
      error: e,
    });
    return null;
  }
}

export const getVideoIterator = async (
  path: string,
  options?: {
    mediaInfo?: MediaInfo;
    fps?: number;
    startIndex: number;
    endIndex?: number;
    isolationKey?: string;
  },
) => {
  try {
    const mediaInfo =
      options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo || !mediaInfo.video) throw new Error("Media info not found");
    const fps = options?.fps || mediaInfo.stats.video?.averagePacketRate || 0;

    const startTimestamp = (options?.startIndex || 0) / fps;
    const endTimestamp = options?.endIndex ? options.endIndex / fps : undefined;
    const videoCanBeTransparent = await mediaInfo.video.canBeTransparent();
    const decoder = getOrCreateVideoDecoder(
      path,
      mediaInfo,
      videoCanBeTransparent,
      options?.isolationKey,
    );
    if (!decoder) throw new Error("Decoder not found");
    decoder.lastAccessTs = nowMs();

    const stream = await decoder.sink.canvases(startTimestamp, endTimestamp);
    async function* iterate(): AsyncGenerator<WrappedCanvas | null> {
      for await (const wc of stream) {
        yield wc;
      }
    }
    return iterate();
  } finally {
    pruneStaleDecoders();
  }
};

// Frame-aligned iterator for blitting: yields one frame per render tick mapped from
// project fps space into source timestamps, honoring clip speed. Callers should
// call .next() once per timeline frame to fetch the correct video image.
export const getVideoFrameIterator = async (
  path: string,
  options: {
    mediaInfo?: MediaInfo;
    projectFps: number; // target/output fps (timeline/export fps)
    startIndex: number; // first project frame index to render (inclusive)
    endIndex?: number; // last project frame index to render (exclusive)
    speed?: number; // playback speed multiplier (e.g., 0.5x, 1x, 2x)
    isolationKey?: string; // ensure per-consumer decoder isolation
  },
): Promise<AsyncIterable<WrappedCanvas | null>> => {
  let mediaInfo = options?.mediaInfo || MediaCache.getState().getMedia(path);
  if (!mediaInfo) {
    // fetch media info
    mediaInfo = await getMediaInfo(path);
    MediaCache.getState().setMedia(path, mediaInfo);
  }

  if (!mediaInfo.video) throw new Error("Media info not found");

  const projectFps = Math.max(1, Math.floor(options.projectFps || 0));
  const speed = (() => {
    const s = Number(options?.speed ?? 1);
    return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
  })();

  const videoCanBeTransparent = await mediaInfo.video.canBeTransparent();
  const decoderMaybe = getOrCreateVideoDecoder(
    path,
    mediaInfo,
    videoCanBeTransparent,
    options?.isolationKey,
  );
  if (!decoderMaybe) throw new Error("Decoder not found");
  const decoder = decoderMaybe;
  decoder.lastAccessTs = nowMs();

  const startIndex = Math.max(0, Math.floor(options.startIndex || 0));
  const endIndex = Number.isFinite(options.endIndex as number)
    ? Math.max(0, Math.floor(options.endIndex as number))
    : undefined;

  async function* iterate(): AsyncGenerator<WrappedCanvas | null> {
    try {
      // Compute source time window based on project indices and speed
      const startTimestamp = (startIndex / projectFps) * speed;
      const endTimestamp =
        typeof endIndex === "number"
          ? (endIndex / projectFps) * speed
          : undefined;

      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const stream = await (decoder.sink as any).canvases(
        startTimestamp,
        endTimestamp,
      );
      const it = (stream as AsyncIterable<WrappedCanvas | null>)[
        Symbol.asyncIterator
      ]();

      let prev: WrappedCanvas | null = null; // last decoded <= target ts
      let next = await it.next();

      const lastIndex =
        typeof endIndex === "number" ? endIndex : Number.POSITIVE_INFINITY;
      for (let i = startIndex; i < lastIndex; i++) {
        const targetTs = (i / projectFps) * speed;
        // Advance until next frame is at or after target timestamp
        while (
          !next.done &&
          next.value &&
          typeof next.value.timestamp === "number" &&
          next.value.timestamp < targetTs
        ) {
          prev = next.value;
          next = await it.next();
        }

        let out: WrappedCanvas | null = null;
        if (
          prev &&
          typeof prev.timestamp === "number" &&
          prev.timestamp <= targetTs
        ) {
          out = prev; // duplicate last frame if needed
        } else if (!next.done) {
          out = (next as any).value ?? null;
        } else {
          out = prev; // stream ended: hold last if available
        }

        decoder.lastAccessTs = nowMs();
        yield out ?? null;
      }
    } finally {
      pruneStaleDecoders();
    }
  }

  return iterate();
};
