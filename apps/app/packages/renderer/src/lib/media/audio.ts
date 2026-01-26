import { MediaInfo } from "../types";
import { WrappedAudioBuffer } from "mediabunny";
import { pruneStaleDecoders } from "./utils";
import { MediaCache } from "./cache";
import { useProjectsStore } from "../projects";
import type {
  AudioWorkerMessage,
  AudioWorkerResponse,
} from "./audio-decoder.worker";

export const BLOCK_SIZE = 2048;

// Worker pool - one worker per audio asset for concurrent decoding
const audioWorkers = new Map<string, WorkerState>();

function getActiveFolderUuid(): string | undefined {
  try {
    return useProjectsStore.getState().getActiveProject()?.folderUuid || undefined;
  } catch {
    return undefined;
  }
}

type WorkerState = {
  worker: Worker;
  requestId: number;
  configured: boolean;
  configuredFolderUuid?: string;
  requestedFolderUuid?: string;
  configurePromise: Promise<void> | null;
  pendingResolvers: Map<
    number,
    {
      queue: WrappedAudioBuffer[];
      done: boolean;
      error: Error | null;
      resolver: (() => void) | null;
    }
  >;
  // Track pending configure requests by requestId
  pendingConfigureResolvers: Map<
    number,
    {
      resolve: () => void;
      reject: (err: Error) => void;
      timeoutId: ReturnType<typeof setTimeout>;
    }
  >;
};

function getOrCreateAudioWorker(assetId: string): WorkerState {
  const existing = audioWorkers.get(assetId);
  if (existing) {
    return existing;
  }

  const worker = new Worker(
    new URL("./audio-decoder.worker.ts", import.meta.url),
    { type: "module" },
  );

  const state: WorkerState = {
    worker,
    requestId: 0,
    configured: false,
    configuredFolderUuid: undefined,
    requestedFolderUuid: undefined,
    configurePromise: null,
    pendingResolvers: new Map<
      number,
      {
        queue: WrappedAudioBuffer[];
        done: boolean;
        error: Error | null;
        resolver: (() => void) | null;
      }
    >(),
    pendingConfigureResolvers: new Map(),
  };

  worker.onmessage = (e: MessageEvent<AudioWorkerResponse>) => {
    const msg = e.data;

    switch (msg.type) {
      case "audioData": {
        const pending = state.pendingResolvers.get(msg.requestId);
        if (!pending) return;

        // Reconstruct AudioBuffer from transferred Float32Arrays
        try {
          const numChannels = msg.channelData.length;
          const numFrames = msg.channelData[0]?.length ?? 0;

          if (numFrames > 0 && numChannels > 0) {
            const audioBuffer = new AudioBuffer({
              numberOfChannels: numChannels,
              length: numFrames,
              sampleRate: msg.sampleRate,
            });

            for (let ch = 0; ch < numChannels; ch++) {
              // Ensure we have a proper ArrayBuffer-backed Float32Array
              const channelArr = msg.channelData[ch];
              // Always copy to ensure ArrayBuffer backing (not SharedArrayBuffer)
              const safeArr = new Float32Array(channelArr.length);
              safeArr.set(channelArr);
              audioBuffer.copyToChannel(safeArr, ch);
            }

            pending.queue.push({
              buffer: audioBuffer,
              timestamp: msg.timestamp,
              duration: msg.duration,
            });
          }

          if (pending.resolver) {
            pending.resolver();
            pending.resolver = null;
          }
        } catch (err) {
          console.warn("[audio] Failed to reconstruct AudioBuffer", err);
        }

        // Send ack back to worker for flow control
        worker.postMessage({
          type: "ack",
          assetId: msg.assetId,
          requestId: msg.requestId,
        } satisfies AudioWorkerMessage);
        break;
      }

      case "iterateDone": {
        const pending = state.pendingResolvers.get(msg.requestId);
        if (pending) {
          pending.done = true;
          if (pending.resolver) {
            pending.resolver();
            pending.resolver = null;
          }
        }
        break;
      }

      case "error": {
        // Check if this is a configure error
        const configResolver = state.pendingConfigureResolvers.get(msg.requestId ?? 0);
        if (configResolver) {
          clearTimeout(configResolver.timeoutId);
          state.pendingConfigureResolvers.delete(msg.requestId ?? 0);
          state.configurePromise = null;
          configResolver.reject(new Error(msg.error));
          break;
        }
        
        // Otherwise it's an iteration error
        const pending = state.pendingResolvers.get(msg.requestId ?? 0);
        if (pending) {
          pending.error = new Error(msg.error);
          if (pending.resolver) {
            pending.resolver();
            pending.resolver = null;
          }
        }
        break;
      }

      case "ready": {
        // Handle configuration completion
        const configResolver = state.pendingConfigureResolvers.get(msg.requestId ?? 0);
        if (configResolver) {
          clearTimeout(configResolver.timeoutId);
          state.pendingConfigureResolvers.delete(msg.requestId ?? 0);
          state.configured = true;
          state.configuredFolderUuid = state.requestedFolderUuid;
          state.configurePromise = null;
          configResolver.resolve();
        }
        break;
      }

      case "debug":
        // Informational messages
        break;
    }
  };

  worker.onerror = (e) => {
    console.error("[audio-worker] Error:", e);
  };

  audioWorkers.set(assetId, state);
  return state;
}

export function disposeAudioWorker(assetId: string) {
  const state = audioWorkers.get(assetId);
  if (!state) return;

  state.worker.postMessage({
    type: "dispose",
    assetId,
  } satisfies AudioWorkerMessage);

  // Give it a moment then terminate
  setTimeout(() => {
    state.worker.terminate();
  }, 100);

  audioWorkers.delete(assetId);
}

export function disposeAllAudioWorkers() {
  for (const [assetId] of audioWorkers) {
    disposeAudioWorker(assetId);
  }
}

/**
 * Preconfigure an audio worker for the given path so it's ready when playback starts.
 * This eliminates the startup delay when calling getAudioIterator.
 * Call this early (e.g., when media info is loaded) for instant audio playback.
 */
export async function preconfigureAudioWorker(
  path: string,
  mediaInfo?: MediaInfo,
): Promise<void> {
  const info = mediaInfo || MediaCache.getState().getMedia(path);
  if (!info || !info.audio) {
    return; // No audio track, nothing to preconfigure
  }

  const assetId = path;
  const workerState = getOrCreateAudioWorker(assetId);
  const folderUuid = getActiveFolderUuid();

  // If already configured or configuring, wait for existing promise
  if (workerState.configured && workerState.configuredFolderUuid === folderUuid) {
    return;
  }
  if (workerState.configurePromise) {
    return workerState.configurePromise;
  }

  // Get decoder config
  const decoderConfig = await info.audio.getDecoderConfig();
  if (!decoderConfig) {
    return; // Can't configure without decoder config
  }

  // Determine format from path
  const ext = path.split(".").pop()?.toLowerCase() || "";
  const formatMap: Record<string, string> = {
    mp4: "mp4",
    m4a: "mp4",
    mov: "mov",
    webm: "webm",
    mkv: "mkv",
    ogg: "ogg",
    mp3: "mp3",
    wav: "wav",
    flac: "flac",
    aac: "aac",
  };
  const formatStr = formatMap[ext];

  const requestId = ++workerState.requestId;
  workerState.requestedFolderUuid = folderUuid;

  // Create configuration promise using the resolver map (no handler replacement)
  workerState.configurePromise = new Promise<void>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      workerState.pendingConfigureResolvers.delete(requestId);
      workerState.configurePromise = null;
      reject(new Error("Audio worker preconfigure timeout"));
    }, 10000);

    // Register resolver in the map - main message handler will resolve/reject
    workerState.pendingConfigureResolvers.set(requestId, {
      resolve,
      reject,
      timeoutId,
    });

    // Send configure message
    workerState.worker.postMessage({
      type: "configure",
      assetId,
      config: {
        audioDecoderConfig: decoderConfig,
        asset: {
          id: assetId,
          type: "audio",
          path,
        },
        formatStr,
        folderUuid,
      },
      requestId,
    } satisfies AudioWorkerMessage);
  });

  return workerState.configurePromise;
}

/**
 * Check if an audio worker is already preconfigured for the given path.
 */
export function isAudioWorkerPreconfigured(path: string): boolean {
  const state = audioWorkers.get(path);
  return state?.configured ?? false;
}

// Track last pre-seek position per asset to avoid redundant seeks
const lastPreseekPosition = new Map<string, number>();

/**
 * Pre-seek the audio worker to a specific timestamp so iteration starts faster.
 * Call this when the user scrubs while paused to warm up the seek position.
 * This is optional but reduces latency when playback starts.
 */
export async function preseekAudioWorker(
  path: string,
  timestamp: number,
  mediaInfo?: MediaInfo,
): Promise<void> {
  // Only pre-seek if we're configured
  let workerState = audioWorkers.get(path);
  if (!workerState?.configured) {
    // Try to preconfigure first if not done
    await preconfigureAudioWorker(path, mediaInfo);
    workerState = audioWorkers.get(path);
    if (!workerState?.configured) {
      return;
    }
  }
  
  // Avoid redundant pre-seeks to the same position (within 0.1s)
  const lastPos = lastPreseekPosition.get(path);
  if (lastPos !== undefined && Math.abs(lastPos - timestamp) < 0.1) {
    return;
  }
  lastPreseekPosition.set(path, timestamp);
  
  // Send preseek message to worker to cache the key packet
  const requestId = ++workerState.requestId;
  workerState.worker.postMessage({
    type: "preseek",
    assetId: path,
    timestamp,
    requestId,
  } satisfies AudioWorkerMessage);
  
  // Don't await response - preseek is fire-and-forget for minimal blocking
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
    if (!mediaInfo || !mediaInfo.audio)
      throw new Error("Media info not found");

    const sampleRate = mediaInfo.audio?.sampleRate || 0;
    const sampleSize = mediaInfo.audio?.sampleSize || 2048.0;
    const hasFps = !!(options?.fps && options.fps > 0);

    // Calculate timestamps
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

    // Get decoder config
    const decoderConfig = await mediaInfo.audio.getDecoderConfig();
    if (!decoderConfig) {
      throw new Error("Decoder config not found for audio track");
    }

    // Get or create worker
    const assetId = path; // Use path as asset ID
    const workerState = getOrCreateAudioWorker(assetId);
    const requestId = ++workerState.requestId;
    const folderUuid = getActiveFolderUuid();

    // Set up pending state for this request
    const pendingState = {
      queue: [] as WrappedAudioBuffer[],
      done: false,
      error: null as Error | null,
      resolver: null as (() => void) | null,
    };
    workerState.pendingResolvers.set(requestId, pendingState);

    // Wait for preconfiguration if it's in progress, or configure now
    if (workerState.configurePromise) {
      await workerState.configurePromise;
    } else if (!workerState.configured || workerState.configuredFolderUuid !== folderUuid) {
      // Not preconfigured, configure now
      const ext = path.split(".").pop()?.toLowerCase() || "";
      const formatMap: Record<string, string> = {
        mp4: "mp4",
        m4a: "mp4",
        mov: "mov",
        webm: "webm",
        mkv: "mkv",
        ogg: "ogg",
        mp3: "mp3",
        wav: "wav",
        flac: "flac",
        aac: "aac",
      };
      const formatStr = formatMap[ext];

      const configRequestId = ++workerState.requestId;
      workerState.requestedFolderUuid = folderUuid;

      // Configure worker with the audio track using resolver map (no handler replacement)
      const configPromise = new Promise<void>((resolve, reject) => {
        const timeoutId = setTimeout(() => {
          workerState.pendingConfigureResolvers.delete(configRequestId);
          reject(new Error("Audio worker configure timeout"));
        }, 10000);

        // Register resolver in the map - main message handler will resolve/reject
        workerState.pendingConfigureResolvers.set(configRequestId, {
          resolve,
          reject,
          timeoutId,
        });

        workerState.worker.postMessage({
          type: "configure",
          assetId,
          config: {
            audioDecoderConfig: decoderConfig,
            asset: {
              id: assetId,
              type: "audio",
              path,
            },
            formatStr,
            folderUuid,
          },
          requestId: configRequestId,
        } satisfies AudioWorkerMessage);
      });

      await configPromise;
    }
    // else: already configured, proceed directly

    // Start iteration
    workerState.worker.postMessage({
      type: "iterate",
      assetId,
      startTime: startTimestamp,
      endTime: endTimestamp ?? Infinity,
      requestId,
    } satisfies AudioWorkerMessage);

    // Generator that yields audio buffers as they arrive
    async function* iterate(): AsyncGenerator<WrappedAudioBuffer | null> {
      try {
        while (true) {
          // Check for errors
          if (pendingState.error) throw pendingState.error;

          // Yield queued buffers
          if (pendingState.queue.length > 0) {
            const item = pendingState.queue.shift();
            if (item) {
              const itemEnd = item.timestamp + item.duration;
              if (itemEnd > startTimestamp) {
                if (
                  endTimestamp !== undefined &&
                  item.timestamp >= endTimestamp
                ) {
                  return;
                }
                yield item;
              }
            }
            continue;
          }

          // If done and queue empty, break
          if (pendingState.done && pendingState.queue.length === 0) {
            break;
          }

          // Wait for more data
          await new Promise<void>((resolve) => {
            pendingState.resolver = resolve;
          });
        }
      } finally {
        // Clean up pending state
        workerState.pendingResolvers.delete(requestId);
      }
    }

    return iterate();
  } finally {
    pruneStaleDecoders();
  }
};
