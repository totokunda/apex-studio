import {
  EncodedPacketSink,
  Input,
  StreamSource,
  ALL_FORMATS,
  AudioSample,
} from "mediabunny";
import * as nodeFs from "node:fs/promises";

// Minimal asset shape expected from the main thread
type WorkerAsset = {
  id: string;
  type: "video" | "image" | "audio";
  path: string;
};

// Message types from main thread to worker
export type AudioWorkerMessage =
  | {
      type: "configure";
      assetId: string;
      config: {
        audioDecoderConfig: AudioDecoderConfig;
        asset: WorkerAsset;
        formatStr?: string;
        folderUuid?: string;
        userDataPath?: string;
      };
      requestId?: number;
    }
  | {
      type: "iterate";
      assetId: string;
      startTime: number;
      endTime: number;
      requestId: number;
    }
  | {
      type: "preseek";
      assetId: string;
      timestamp: number;
      requestId?: number;
    }
  | {
      type: "dispose";
      assetId?: string;
    }
  | {
      type: "ack";
      assetId?: string;
      requestId: number;
    };

// Response types from worker to main thread
export type AudioWorkerResponse =
  | {
      type: "audioData";
      // Transfer channel data as Float32Arrays (transferable)
      channelData: Float32Array[];
      sampleRate: number;
      timestamp: number;
      duration: number;
      requestId: number;
      assetId?: string;
    }
  | { type: "error"; error: string; requestId?: number; assetId?: string }
  | { type: "iterateDone"; requestId: number; assetId?: string }
  | { type: "ready"; requestId?: number; assetId?: string }
  | { type: "preseekDone"; requestId?: number; assetId?: string }
  | {
      type: "debug";
      scope: "audio-decoder-worker";
      event: string;
      assetId?: string;
      requestId?: number;
      payload?: any;
    };

// Worker-local helper: turn a file/app URL into a plain path string
function fileURLToPathInWorker(raw: string): string {
  try {
    const u = new URL(raw);

    if (u.protocol === "file:" || u.protocol === "app:") {
      const decoded = decodeURIComponent(u.pathname);
      // Windows file URL path: /C:/path -> C:/path
      if (/^\/[A-Za-z]:\//.test(decoded)) {
        return decoded.slice(1);
      }
      return decoded;
    }

    // For unsupported URL schemes, preserve the original value.
    return raw;
  } catch {
    // Not a URL, assume this is already a local filesystem path.
    return raw;
  }
}

// State per asset
type AssetState = {
  decoder: AudioDecoder | null;
  sink: EncodedPacketSink | null;
  input: Input | null;
  config: AudioDecoderConfig | null;
  currentRequestId: number;
  iterationInFlight: number;
  iterationResume: (() => void) | null;
  // Cached pre-seek position to speed up iteration start
  cachedSeekTimestamp: number | null;
  cachedKeyPacket: any | null;
  // Only use cached key packet after at least one successful iteration (avoids "Packet was not created from this track")
  hasSuccessfullyIterated: boolean;
};

const assetStates = new Map<string, AssetState>();

// Tuning constants
const MAX_ITERATION_IN_FLIGHT = 8;
const MAX_DECODE_QUEUE_SIZE = 50;

// Debug log on worker load
try {
  // @ts-ignore
  postMessage({
    type: "debug",
    scope: "audio-decoder-worker",
    event: "worker-loaded",
  } satisfies AudioWorkerResponse);
} catch {
  // Best-effort only
}

function getOrCreateState(assetId: string): AssetState {
  let state = assetStates.get(assetId);
  if (!state) {
    state = {
      decoder: null,
      sink: null,
      input: null,
      config: null,
      currentRequestId: 0,
      iterationInFlight: 0,
      iterationResume: null,
      cachedSeekTimestamp: null,
      cachedKeyPacket: null,
      hasSuccessfullyIterated: false,
    };
    assetStates.set(assetId, state);
  }
  return state;
}

function createAudioDecoder(
  state: AssetState,
  assetId: string,
  requestId: number,
  onOutput: (sample: AudioSample) => void,
) {
  const decoder = new AudioDecoder({
    output: (data) => {
      try {
        const sample = new AudioSample(data);
        onOutput(sample);
      } catch (e) {
        console.error("AudioDecoder output error", e);
      }
    },
    error: (e) => {
      console.error("AudioDecoder error", e);
      // @ts-ignore
      postMessage({
        type: "error",
        error: e.message ?? "AudioDecoder error",
        assetId,
        requestId,
      });
    },
  });
  state.decoder = decoder;
  return decoder;
}

function createNodeFileSource(filePath: string): StreamSource {
  let fileHandle: Awaited<ReturnType<typeof nodeFs.open>> | null = null;
  let knownSize: number | null = null;

  return new StreamSource({
    getSize: async () => {
      if (!fileHandle) {
        fileHandle = await nodeFs.open(filePath, "r");
      }
      if (knownSize == null) {
        const stats = await fileHandle.stat();
        knownSize = Number(stats.size);
      }
      return knownSize;
    },
    read: async (start, end) => {
      if (!fileHandle) {
        fileHandle = await nodeFs.open(filePath, "r");
      }
      const length = Math.max(0, end - start);
      const buffer = new Uint8Array(length);
      if (length === 0) return buffer;
      const readResult = await fileHandle.read(buffer, 0, length, start);
      if (readResult.bytesRead === length) return buffer;
      return buffer.subarray(0, readResult.bytesRead);
    },
    dispose: async () => {
      if (!fileHandle) return;
      try {
        await fileHandle.close();
      } catch {
        // ignore close errors in worker teardown
      } finally {
        fileHandle = null;
      }
    },
    prefetchProfile: "fileSystem",
  });
}

// Message listener
self.onmessage = async (e: MessageEvent<AudioWorkerMessage>) => {
  const msg = e.data;

  // Debug echo
  try {
    // @ts-ignore
    postMessage({
      type: "debug",
      scope: "audio-decoder-worker",
      event: "onmessage",
      assetId: (msg as any).assetId,
      requestId: (msg as any).requestId,
      payload: { type: msg.type },
    } satisfies AudioWorkerResponse);
  } catch {
    // Best-effort
  }

  try {
    switch (msg.type) {
      case "configure": {
        await handleConfigure(msg, msg.requestId);
        break;
      }
      case "iterate": {
        await handleIterate(
          msg.startTime,
          msg.endTime,
          msg.requestId,
          msg.assetId,
        );
        break;
      }
      case "preseek": {
        await handlePreseek(msg.assetId, msg.timestamp, msg.requestId);
        break;
      }
      case "ack": {
        const id = msg.assetId;
        if (!id) break;
        const state = assetStates.get(id);
        if (!state) break;
        if (msg.requestId === state.currentRequestId) {
          state.iterationInFlight--;
          if (
            state.iterationInFlight < MAX_ITERATION_IN_FLIGHT &&
            state.iterationResume
          ) {
            state.iterationResume();
            state.iterationResume = null;
          }
        }
        break;
      }
      case "dispose": {
        dispose(msg.assetId);
        break;
      }
    }
  } catch (err: any) {
    console.error("Audio Worker Error:", err);
    // @ts-ignore
    postMessage({
      type: "error",
      error: err.message,
      requestId: (msg as any).requestId || 0,
      assetId: (msg as any).assetId,
    });
  }
};

async function handleConfigure(
  msg: Extract<AudioWorkerMessage, { type: "configure" }>,
  requestId?: number,
) {
  const { assetId, config: cfg } = msg;
  const id = assetId ?? cfg.asset.id;

  if (!id) {
    throw new Error("configure message missing asset identifier");
  }

  dispose(id);
  const state = getOrCreateState(id);
  // Ensure no stale packets survive across reconfiguration.
  // A cached EncodedPacket is only valid for the exact track/sink that created it.
  state.cachedSeekTimestamp = null;
  state.cachedKeyPacket = null;
  state.hasSuccessfullyIterated = false;

  let formats = ALL_FORMATS;

  // Setup Input
  let input: Input | null = null;
  const filePath = fileURLToPathInWorker(cfg.asset.path);
  if (!filePath) {
    throw new Error("Missing file path for audio source");
  }
  input = new Input({ formats, source: createNodeFileSource(filePath) });

  state.input = input;

  const audioTrack = await state.input.getPrimaryAudioTrack();
  if (!audioTrack) throw new Error("No audio track found in worker");

  // Setup Sink
  state.sink = new EncodedPacketSink(audioTrack);

  // Store config
  state.config = cfg.audioDecoderConfig;


  // @ts-ignore
  postMessage({
    type: "ready",
    requestId,
    assetId: id,
  });
}

async function handlePreseek(
  assetId: string,
  timestamp: number,
  requestId?: number,
) {
  const state = assetStates.get(assetId);
  if (!state || !state.sink) {
    // @ts-ignore
    postMessage({
      type: "preseekDone",
      requestId,
      assetId,
    });
    return;
  }

  try {
    // Pre-fetch the key packet for this timestamp
    const keyPacket = await state.sink.getKeyPacket(timestamp, { verifyKeyPackets: true});
    state.cachedSeekTimestamp = timestamp;
    state.cachedKeyPacket = keyPacket;
  } catch (e) {
    // Ignore errors, preseek is best-effort
    state.cachedSeekTimestamp = null;
    state.cachedKeyPacket = null;
  }

  // @ts-ignore
  postMessage({
    type: "preseekDone",
    requestId,
    assetId,
  });
}

async function handleIterate(
  startTime: number,
  endTime: number,
  requestId: number,
  assetId?: string,
) {
  const id = assetId;
  if (!id) return;

  const state = assetStates.get(id);
  if (!state || !state.sink || !state.config) return;

  state.currentRequestId = requestId;
  state.iterationInFlight = 0;
  state.iterationResume = null;

  // Close previous decoder if any
  if (state.decoder && (state.decoder.state as string) !== "closed") {
    try {
      state.decoder.close();
    } catch {
      // ignore
    }
  }

  // Create new decoder with output handler
  const decoder = createAudioDecoder(state, id, requestId, (sample) => {
    const frameTime = sample.timestamp;
    if (frameTime < startTime - 0.1 || frameTime > endTime + 0.1) {
      sample.close();
      return;
    }

    state.iterationInFlight++;

    // Extract channel data as Float32Arrays for transfer
    const numChannels = sample.numberOfChannels;
    const numFrames = sample.numberOfFrames;
    const channelData: Float32Array[] = [];

    for (let ch = 0; ch < numChannels; ch++) {
      const chData = new Float32Array(numFrames);
      sample.copyTo(chData, { planeIndex: ch, format: "f32-planar" });
      channelData.push(chData);
    }

    const msg: AudioWorkerResponse = {
      type: "audioData",
      channelData,
      sampleRate: sample.sampleRate,
      timestamp: sample.timestamp,
      duration: sample.duration,
      requestId,
      assetId: id,
    };

    // Transfer the Float32Array buffers
    const transferables = channelData.map((arr) => arr.buffer);
    // @ts-ignore
    postMessage(msg, transferables);

    sample.close();
  });

  decoder.configure(state.config);

  // Track whether iteration completed without error (for cache validity)
  let iterationSucceeded = false;

  try {
    // Use cached key packet only after at least one successful iteration.
    // On first use, the cache may be stale (different track instance) causing "Packet was not created from this track".
    let keyPacket: any;
    if (
      state.hasSuccessfullyIterated &&
      state.cachedKeyPacket &&
      state.cachedSeekTimestamp !== null &&
      Math.abs(state.cachedSeekTimestamp - startTime) < 0.5
    ) {
      keyPacket = state.cachedKeyPacket;
      // Clear cache after use
      state.cachedKeyPacket = null;
      state.cachedSeekTimestamp = null;
    } else {
      // No cache hit or first iteration, do the seek
      keyPacket = await state.sink.getKeyPacket(startTime, { verifyKeyPackets: true});
    }
    const packets = state.sink.packets(keyPacket || undefined);

    for await (const packet of packets) {
      if (state.currentRequestId !== requestId) break;
      if (packet.timestamp > endTime + 1.0) break;

      // Backpressure: wait if decoder queue is full
      if (decoder.decodeQueueSize >= MAX_DECODE_QUEUE_SIZE) {
        await new Promise<void>((resolve) => {
          decoder.addEventListener("dequeue", () => resolve(), { once: true });
        });
      }

      // Flow control: Wait if too many frames are pending ack
      while (state.iterationInFlight >= MAX_ITERATION_IN_FLIGHT) {
        if (state.currentRequestId !== requestId) break;
        await new Promise<void>((r) => (state.iterationResume = r));
      }

      if (state.currentRequestId !== requestId) break;

      decoder.decode(packet.toEncodedAudioChunk());
    }

    // Flush remaining
    if (decoder.state !== "closed") {
      await decoder.flush();
    }
    iterationSucceeded = true;
  } catch (e) {
    console.warn("Audio iteration failed", e);
  } finally {
    if (iterationSucceeded) {
      state.hasSuccessfullyIterated = true;
    }
    if (decoder.state !== "closed") {
      decoder.close();
    }
  }

  // @ts-ignore
  postMessage({
    type: "iterateDone",
    requestId,
    assetId: id,
  });
}

function dispose(assetId?: string) {
  if (assetId) {
    const state = assetStates.get(assetId);
    if (!state) return;
    state.currentRequestId++;
    if (state.decoder && (state.decoder.state as string) !== "closed") {
      try {
        state.decoder.close();
      } catch {
        // ignore
      }
      state.decoder = null;
    }
    state.input = null;
    state.sink = null;
    state.iterationInFlight = 0;
    state.iterationResume = null;
    // Clear any cached packets; they may belong to a previous track instance.
    state.cachedSeekTimestamp = null;
    state.cachedKeyPacket = null;
    state.hasSuccessfullyIterated = false;
    return;
  }

  // Dispose all
  for (const [id] of assetStates) {
    dispose(id);
  }
  assetStates.clear();
}
