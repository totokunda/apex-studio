import { MediaInfo } from "../types";
import { AudioDecoderContext, AudioDecoderKey } from "./types";
import { AudioBufferSink, WrappedAudioBuffer } from "mediabunny";
import { nowMs, audioDecoders, pruneStaleDecoders } from "./utils";
import { MediaCache } from "./cache";

export const BLOCK_SIZE = 2048;

function getOrCreateAudioDecoder(
  path: string,
  mediaInfo: MediaInfo,
): AudioDecoderContext | null {
  if (!mediaInfo?.audio) return null;
  const key: AudioDecoderKey = `${path}#audio`;
  const existing = audioDecoders.get(key);
  const sampleRate = mediaInfo.audio?.sampleRate || 0;
  if (existing) {
    existing.lastAccessTs = nowMs();
    existing.sampleRate = sampleRate || existing.sampleRate;
    return existing;
  }
  try {
    const sink = new AudioBufferSink(mediaInfo.audio);

    const ctx: AudioDecoderContext = {
      sink,
      inFlight: new Set<number>(),
      lastAccessTs: nowMs(),
      sampleRate: sampleRate || 0,
    };
    audioDecoders.set(key, ctx);
    // Prime decoder lightly (do not await)
    void (async () => {
      try {
        const ts = 0;
        await sink.getBuffer(ts);
      } catch {}
    })();
    return ctx;
  } catch (e) {
    console.warn("[media] Failed to create AudioSampleSink", {
      path,
      error: e,
    });
    return null;
  }
}

export const getAudioIterator = async (
  path: string,
  options?: {
    mediaInfo?: MediaInfo;
    sampleRate?: number;
    fps?: number;
    sampleSize?: number;
    startIndex?: number;
    endIndex?: number;
  },
) => {
  try {
    const mediaInfo =
      options?.mediaInfo || MediaCache.getState().getMedia(path);
    if (!mediaInfo || !mediaInfo.audio) throw new Error("Media info not found");
    const sampleRate = mediaInfo.audio?.sampleRate || 0;
    const sampleSize = mediaInfo.audio?.sampleSize || 2048.0;
    const hasFps = !!(options?.fps && options.fps > 0);
    // Align semantics with video: when fps is provided, treat startIndex/endIndex as frame indices
    // and convert to seconds via fps. Fallback to sampleSize-based mapping when fps is unavailable.
    const startIndex = options?.startIndex || 0;
    let startTimestamp: number;
    if (hasFps) {
      startTimestamp = startIndex / (options!.fps as number);
    } else {
      startTimestamp = (sampleSize * startIndex) / sampleRate;
    }
    let endTimestamp: number | undefined = undefined;
    if (typeof options?.endIndex === "number") {
      const endIndex = options!.endIndex as number;
      endTimestamp = hasFps
        ? endIndex / (options!.fps as number)
        : (sampleSize * endIndex) / sampleRate;
    }
    const decoder = getOrCreateAudioDecoder(path, mediaInfo);
    if (!decoder) throw new Error("Decoder not found");
    decoder.lastAccessTs = nowMs();
    const stream = await decoder.sink.buffers(startTimestamp, endTimestamp);
    async function* iterate(): AsyncGenerator<WrappedAudioBuffer | null> {
      for await (const buf of stream) {
        yield buf;
      }
    }
    return iterate();
  } finally {
    pruneStaleDecoders();
  }
};
