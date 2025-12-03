import {
  EncodedPacketSink,
  EncodedPacket,
  Input,
  UrlSource,
  BlobSource,
  ALL_FORMATS,
  MP4,
  WEBM,
  QTFF,
  MATROSKA,
  OGG,
  MP3,
  WAVE,
  FLAC,
  ADTS,
} from "mediabunny";

// Define message types
export type WorkerMessage =
  | {
      type: "configure";
      config: {
        videoDecoderConfig: VideoDecoderConfig;
        source: {
          type: "url" | "blob";
          url?: string;
          blob?: Blob;
        };
        formatStr?: string;
        initialTimestamp?: number;
      };
      requestId?: number;
    }
  | { type: "seek"; timestamp: number; forceAccurate: boolean; requestId: number }
  | {
      type: "iterate";
      startTime: number;
      endTime: number;
      requestId: number;
    }
  | { type: "dispose" }
  | { type: "ack"; requestId: number };

export type WorkerResponse =
  | {
      type: "frame";
      frame: VideoFrame; // Transferable
      timestamp: number;
      duration: number;
      requestId: number; // to match with seek/iterate request
    }
  | { type: "error"; error: string; requestId?: number }
  | { type: "seekDone"; requestId: number }
  | { type: "iterateDone"; requestId: number };

// State
let decoder: VideoDecoder | null = null;
let sink: EncodedPacketSink | null = null;
let input: Input | null = null;

// Caching
const cachedDecodedFrames = new Map<number, VideoFrame>();
const MAX_CACHE_SIZE = 240; // Keep ~2 seconds of frames (at 30fps)
const keyPacketCache = new Map<number, EncodedPacket>();
let isCachingKeyPackets = false;

// Seek state
let seekTargetTimestamp: number | null = null;
let seekDone = false;
let currentRequestId = 0;
let lastSeekTime = 0;
let lastSeekTimestamp = 0;
let showingPreview = false;
let config: VideoDecoderConfig | null = null;

// Iteration flow control
let iterationInFlight = 0;
const MAX_ITERATION_IN_FLIGHT = 4;
let iterationResume: (() => void) | null = null;

// Output Handling with dynamic dispatch
let customOutputHandler: ((frame: VideoFrame) => void) | null = null;

// Helpers
function findCachedFrame(timestamp: number): VideoFrame | null {
  for (const [t, frame] of cachedDecodedFrames) {
    if (Math.abs(t - timestamp) < 0.05) return frame;
  }
  return null;
}

function cacheFrame(frame: VideoFrame) {
  const frameTime = frame.timestamp / 1e6;
  if (cachedDecodedFrames.has(frameTime)) return;

  if (cachedDecodedFrames.size >= MAX_CACHE_SIZE) {
    const firstKey = cachedDecodedFrames.keys().next().value;
    if (firstKey !== undefined) {
      cachedDecodedFrames.get(firstKey)?.close();
      cachedDecodedFrames.delete(firstKey);
    }
  }
  cachedDecodedFrames.set(frameTime, frame.clone());
}

// Background task
async function cacheKeyPackets() {
  if (isCachingKeyPackets) return;
  isCachingKeyPackets = true;
  try {
    // Lazy caching is usually sufficient and safer for performance
  } catch (e) {
    console.warn("Background keyframe caching failed", e);
  }
}

// Handler
const onFrameHandler = (frame: VideoFrame) => {
  if (customOutputHandler) {
      customOutputHandler(frame);
      return;
  }

  const frameTime = frame.timestamp / 1e6;

  // 1. Cache
  cacheFrame(frame);

  // 2. Preview
  if (showingPreview) {
    postFrame(frame, currentRequestId);
    showingPreview = false;
  }

  // 3. Check Target
  if (seekTargetTimestamp !== null) {
    if (frameTime >= seekTargetTimestamp - 0.04) {
      seekDone = true;
      postFrame(frame, currentRequestId);
      seekTargetTimestamp = null;
    }
  }

  frame.close();
};

function postFrame(frame: VideoFrame, reqId: number) {
  const clone = frame.clone();
  const msg: WorkerResponse = {
    type: "frame",
    frame: clone,
    timestamp: clone.timestamp / 1e6,
    duration: (clone.duration ?? 0) / 1e6,
    requestId: reqId,
  };
  // @ts-ignore
  postMessage(msg, [clone]);
}

// Message Listener
self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const msg = e.data;

  try {
    switch (msg.type) {
      case "configure": {
        await handleConfigure(msg.config, msg.requestId);
        break;
      }
      case "seek": {
        await handleSeek(msg.timestamp, msg.forceAccurate, msg.requestId);
        break;
      }
      case "iterate": {
        await handleIterate(msg.startTime, msg.endTime, msg.requestId);
        break;
      }
      case "ack": {
        if (msg.requestId === currentRequestId) {
          iterationInFlight--;
          if (iterationInFlight < MAX_ITERATION_IN_FLIGHT && iterationResume) {
             iterationResume();
             iterationResume = null;
          }
        }
        break;
      }
      case "dispose": {
        dispose();
        break;
      }
    }
  } catch (err: any) {
    console.error("Worker Error:", err);
    // @ts-ignore
    postMessage({ type: "error", error: err.message, requestId: msg.requestId || 0 });
  }
};

async function handleConfigure(cfg: {
  videoDecoderConfig: VideoDecoderConfig;
  source: { type: "url" | "blob"; url?: string; blob?: Blob };
  formatStr?: string;
  initialTimestamp?: number;
}, requestId?: number) {
  dispose(); // Clear previous state
  let formats = ALL_FORMATS;
  if (cfg.formatStr) {
      if (cfg.formatStr === "mp4") formats = [MP4];
      else if (cfg.formatStr === "webm") formats = [WEBM];
      else if (cfg.formatStr === "mov") formats = [QTFF];
      else if (cfg.formatStr === "mkv") formats = [MATROSKA];
      else if (cfg.formatStr === "ogg") formats = [OGG];
      else if (cfg.formatStr === "mp3") formats = [MP3];
      else if (cfg.formatStr === "wav") formats = [WAVE];
      else if (cfg.formatStr === "flac") formats = [FLAC];
      else if (cfg.formatStr === "aac") formats = [ADTS];
  }

  // 1. Setup Input
  if (cfg.source.type === "url" && cfg.source.url) {
    input = new Input({
      formats: formats,
      source: new UrlSource(new URL(cfg.source.url)),
    });
  } else if (cfg.source.type === "blob" && cfg.source.blob) {
    input = new Input({
      formats: formats,
      source: new BlobSource(cfg.source.blob),
    });
  } else {
    throw new Error("Invalid source configuration");
  }

  const videoTrack = await input.getPrimaryVideoTrack();
  if (!videoTrack) throw new Error("No video track found in worker");

  // 2. Setup Sink
  sink = new EncodedPacketSink(videoTrack);

  // 3. Setup Decoder
  config = {
    ...cfg.videoDecoderConfig,
    optimizeForLatency: true,
  };

  decoder = new VideoDecoder({
    output: onFrameHandler,
    error: (e) => {
      console.error("VideoDecoder error", e);
      // @ts-ignore
      postMessage({ type: "error", error: e.message });
    },
  });
  
  decoder.configure(config);
  cacheKeyPackets();

  if (requestId !== undefined) {
    await handleSeek(cfg.initialTimestamp ?? 0, true, requestId);
  }
}

async function handleSeek(timestamp: number, forceAccurate: boolean, requestId: number) {
  if (!decoder || !sink) return;
  
  // Cancel any active iteration handler to prevent hijacking seek output
  customOutputHandler = null;
  // Also resume any stuck flow control so iterate can exit cleanly
  if (iterationResume) {
      iterationResume();
      iterationResume = null;
  }

  currentRequestId = requestId;
  const now = performance.now();
  const timeSinceLast = now - lastSeekTime;
  const dist = Math.abs(timestamp - (lastSeekTimestamp || 0));
  lastSeekTime = now;
  lastSeekTimestamp = timestamp;

  const isFastScrubbing = !forceAccurate && timeSinceLast < 150 && dist > 0.5;

  // 1. Cache Hit
  const cached = findCachedFrame(timestamp);
  if (cached) {
    postFrame(cached, requestId);
    // @ts-ignore
    postMessage({ type: "seekDone", requestId });
    return;
  }

  seekTargetTimestamp = timestamp;
  seekDone = false;
  showingPreview = false;

  const currentPacket = await sink.getKeyPacket(timestamp, { verifyKeyPackets: false });
  if (!currentPacket) return;

  if (!keyPacketCache.has(currentPacket.timestamp)) {
    keyPacketCache.set(currentPacket.timestamp, currentPacket);
  }

  if (currentRequestId !== requestId) return;

  if ((decoder.state as string) === "closed") return;
  decoder.reset();
  if (config) decoder.configure(config);

  decoder.decode(currentPacket.toEncodedVideoChunk());


  if (isFastScrubbing) {
    showingPreview = true;
    await new Promise((r) => setTimeout(r, 80));
    if (currentRequestId !== requestId) return;
    showingPreview = false;
  }

  const packets = sink.packets(currentPacket);

  for await (const packet of packets) {
    if ((decoder.state as string) === "closed") break;
      decoder.decode(packet.toEncodedVideoChunk());
    if (seekDone) break;
    if (currentRequestId !== requestId) break;
    if (packet.timestamp > timestamp + 0.1) break;
  }

  if (forceAccurate) {
    await decoder.flush();
  }
  
  // @ts-ignore
  postMessage({ type: "seekDone", requestId });
}

async function handleIterate(startTime: number, endTime: number, requestId: number) {
    if (!decoder || !sink || !config) return;

    currentRequestId = requestId;

    const keyPacket = await sink.getKeyPacket(startTime, { verifyKeyPackets: false });
    if (!keyPacket) {
        // @ts-ignore
        postMessage({ type: "iterateDone", requestId });
        return;
    }

    if ((decoder.state as string) === "closed") return;
    decoder.reset();
    decoder.configure(config);

    // Override handler temporarily for iteration
    
    // To keep it simple in this worker version, let's reuse the decoder but clear the Seek target.
    seekTargetTimestamp = null; 
    
    // Reset flow control
    iterationInFlight = 0;
    iterationResume = null;

    const iterationHandler = (frame: VideoFrame) => {
        const frameTime = frame.timestamp / 1e6;
        if (frameTime < startTime || frameTime > endTime + 0.05) {
            frame.close();
            return;
        }
        
        iterationInFlight++;
        
        // Post immediately
        const clone = frame.clone();
        const msg = { 
            type: "frame", 
            frame: clone, 
            timestamp: frameTime, 
            duration: (frame.duration ?? 0) / 1e6,
            requestId 
        };
        // @ts-ignore
        postMessage(msg, [clone]);
        frame.close();
    };

    // Use dynamic handler instead of recreating decoder
    customOutputHandler = iterationHandler;

    try {
        if ((decoder.state as string) === "closed") {
             // @ts-ignore
             postMessage({ type: "error", error: "Decoder closed", requestId });
             return;
        }
        
        decoder.decode(keyPacket.toEncodedVideoChunk());
        const packets = sink.packets(keyPacket);
        
        for await (const packet of packets) {
            if (currentRequestId !== requestId) break;
            if (packet.timestamp > endTime + 0.1) break;
            
            if ((decoder.state as string) === "closed") break;
            
            // Flow control: Wait if too many frames are pending
            while (iterationInFlight >= MAX_ITERATION_IN_FLIGHT) {
                 if (currentRequestId !== requestId) break;
                 await new Promise<void>(r => iterationResume = r);
            }

            decoder.decode(packet.toEncodedVideoChunk());
            
            while (decoder.decodeQueueSize > 6) {
                 await new Promise(r => setTimeout(r, 5));
            }
        }
        
        if ((decoder.state as string) !== "closed") {
            await decoder.flush();
        }
    } finally {
        customOutputHandler = null;
    }
    
    // @ts-ignore
    postMessage({ type: "iterateDone", requestId });
}

function dispose() {
  currentRequestId++;
  if (decoder && (decoder.state as string) !== "closed") {
    decoder.close();
  }
  for (const frame of cachedDecodedFrames.values()) {
    frame.close();
  }
  cachedDecodedFrames.clear();
  keyPacketCache.clear();
}
