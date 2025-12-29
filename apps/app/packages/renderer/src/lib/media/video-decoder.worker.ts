import {
  EncodedPacketSink,
  EncodedPacket,
  Input,
  UrlSource,
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
// Minimal asset shape expected from the main thread. This mirrors the core
// fields of the renderer-side `Asset` type but is kept local to the worker
// for decoupling and to avoid importing renderer modules here.
type WorkerAsset = {
  id: string;
  type: "video" | "image" | "audio";
  path: string;
};

// Define message types
export type WorkerMessage =
  | {
      type: "configure";
      assetId: string;
      config: {
        videoDecoderConfig: VideoDecoderConfig;
        asset: WorkerAsset;
        formatStr?: string;
        initialTimestamp?: number;
        folderUuid?: string;
        // Optional absolute Electron userData path, passed from the renderer
        // so the worker can cheaply detect when an asset path is already
        // rooted under userData and prefer app://user-data over app://apex-cache.
        userDataPath?: string;
      };
      requestId?: number;
    }
  | {
      type: "seek";
      assetId: string;
      timestamp: number;
      forceAccurate: boolean;
      requestId: number;
    }
  | {
      type: "iterate";
      assetId: string;
      startTime: number;
      endTime: number;
      requestId: number;
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

export type WorkerResponse =
  | {
      type: "frame";
      frame: VideoFrame; // Transferable
      timestamp: number;
      duration: number;
      requestId: number; // to match with seek/iterate request
      assetId?: string;
    }
  | { type: "error"; error: string; requestId?: number; assetId?: string }
  | { type: "seekDone"; requestId: number; assetId?: string }
  | { type: "iterateDone"; requestId: number; assetId?: string }
  | { type: "ready"; requestId?: number; assetId?: string }
  // Lightweight debug messages mirrored back to the main thread so
  // worker activity is visible in the renderer devtools console.
  | {
      type: "debug";
      scope: "video-decoder-worker";
      event: string;
      assetId?: string;
      requestId?: number;
      payload?: any;
    };

// Worker-local helper: turn a file/app URL into a plain path string without
// relying on Node.js utilities. This stays compatible with browser/Electron
// worker contexts.
function fileURLToPathInWorker(raw: string): string {
  try {
    const u = new URL(raw);

    // For file:// or app:// URLs, use the pathname and strip leading slashes.
    if (u.protocol === "file:" || u.protocol === "app:") {
      return decodeURIComponent(u.pathname.replace(/^\/+/, ""));
    }

    // For other URL schemes, just normalize the pathname.
    return decodeURIComponent((u.pathname || "").replace(/^\/+/, ""));
  } catch {
    // Not a URL – assume it's already a path; just normalize leading slashes.
    return raw.replace(/^\/+/, "");
  }
}

// Best-effort helper to quickly detect a 404 for an app:// URL before we hand
// it off to mediabunny. If the HEAD request itself fails (e.g. protocol does
// not support HEAD), we treat it as "unknown" and *do not* block source
// creation – only an explicit 404 status will cause us to skip this URL.
async function isAppUrlDefinitely404(url: URL): Promise<boolean> {
  try {
    const res = await fetch(url.toString(), { method: "HEAD" });
    return res.status === 404;
  } catch {
    return false;
  }
}

// State
type AssetState = {
  decoder: VideoDecoder | null;
  alphaDecoder: VideoDecoder | null;
  sink: EncodedPacketSink | null;
  input: Input | null;

// Caching
  cachedDecodedFrames: Map<number, VideoFrame>;
  keyPacketCache: Map<number, EncodedPacket>;
  isCachingKeyPackets: boolean;

  // Alpha merge (for codecs where alpha is stored separately in packet sideData)
  alphaFramesByTimestamp: Map<number, VideoFrame>;
  pendingColorFramesByTimestamp: Map<
    number,
    {
      frame: VideoFrame;
      requestId: number;
    }
  >;
  // Reused canvases to avoid reallocating for every frame
  mergeCanvas: OffscreenCanvas | null;
  mergeCtx: OffscreenCanvasRenderingContext2D | null;
  alphaCanvas: OffscreenCanvas | null;
  alphaCtx: OffscreenCanvasRenderingContext2D | null;

  // Seek state
  seekTargetTimestamp: number | null;
  seekDone: boolean;
  currentRequestId: number;
  lastSeekTime: number;
  lastSeekTimestamp: number;
  showingPreview: boolean;
  config: VideoDecoderConfig | null;
  pendingSeekFrame: VideoFrame | null;
  pendingSeekFrameTime: number;

  // Iteration flow control
  iterationInFlight: number;
  iterationResume: (() => void) | null;

  // Output Handling with dynamic dispatch
  customOutputHandler: ((frame: VideoFrame) => void) | null;
};

const assetStates = new Map<string, AssetState>();

// Shared tuning constants
const MAX_CACHE_SIZE = 240; // Keep ~2 seconds of frames (at 30fps)
const MAX_ITERATION_IN_FLIGHT = 4;

// Emit a one-time debug message as soon as the worker script is evaluated so
// we can confirm that this exact file is being loaded by the main thread.
try {
  // @ts-ignore
  postMessage({
    type: "debug",
    scope: "video-decoder-worker",
    event: "worker-loaded",
  } satisfies WorkerResponse);
} catch {
  // Best-effort only.
}

function getOrCreateState(assetId: string): AssetState {
  let state = assetStates.get(assetId);
  if (!state) {
    state = {
      decoder: null,
      alphaDecoder: null,
      sink: null,
      input: null,
      cachedDecodedFrames: new Map<number, VideoFrame>(),
      keyPacketCache: new Map<number, EncodedPacket>(),
      isCachingKeyPackets: false,
      alphaFramesByTimestamp: new Map<number, VideoFrame>(),
      pendingColorFramesByTimestamp: new Map<
        number,
        { frame: VideoFrame; requestId: number }
      >(),
      mergeCanvas: null,
      mergeCtx: null,
      alphaCanvas: null,
      alphaCtx: null,
      seekTargetTimestamp: null,
      seekDone: false,
      currentRequestId: 0,
      lastSeekTime: 0,
      lastSeekTimestamp: 0,
      showingPreview: false,
      config: null,
      pendingSeekFrame: null,
      pendingSeekFrameTime: 0,
      iterationInFlight: 0,
      iterationResume: null,
      customOutputHandler: null,
    };
    assetStates.set(assetId, state);
  }
  return state;
}

function resetAlphaMergeQueues(state: AssetState) {
  for (const f of state.alphaFramesByTimestamp.values()) f.close();
  state.alphaFramesByTimestamp.clear();
  for (const v of state.pendingColorFramesByTimestamp.values()) v.frame.close();
  state.pendingColorFramesByTimestamp.clear();
}

function ensureMergeCanvases(state: AssetState, width: number, height: number) {
  if (
    !state.mergeCanvas ||
    state.mergeCanvas.width !== width ||
    state.mergeCanvas.height !== height
  ) {
    state.mergeCanvas = new OffscreenCanvas(width, height);
    state.mergeCtx = state.mergeCanvas.getContext("2d", {
      willReadFrequently: true,
    }) as OffscreenCanvasRenderingContext2D | null;
  }
  if (
    !state.alphaCanvas ||
    state.alphaCanvas.width !== width ||
    state.alphaCanvas.height !== height
  ) {
    state.alphaCanvas = new OffscreenCanvas(width, height);
    state.alphaCtx = state.alphaCanvas.getContext("2d", {
      willReadFrequently: true,
    }) as OffscreenCanvasRenderingContext2D | null;
  }
}

function mergeAlphaIntoColor(
  state: AssetState,
  colorFrame: VideoFrame,
  alphaFrame: VideoFrame,
): VideoFrame {
  const width = colorFrame.displayWidth || (colorFrame as any).codedWidth || 0;
  const height =
    colorFrame.displayHeight || (colorFrame as any).codedHeight || 0;
  if (!width || !height) {
    // Fallback: if we can't determine dimensions, just return color as-is.
    // Caller will close `alphaFrame`.
    return colorFrame;
  }

  ensureMergeCanvases(state, width, height);
  const ctx = state.mergeCtx;
  const aCtx = state.alphaCtx;
  if (!ctx || !aCtx || !state.mergeCanvas || !state.alphaCanvas) {
    return colorFrame;
  }

  // Render both frames into RGBA buffers via 2D canvas drawImage conversion.
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(colorFrame as any, 0, 0, width, height);
  const colorImage = ctx.getImageData(0, 0, width, height);

  aCtx.clearRect(0, 0, width, height);
  aCtx.drawImage(alphaFrame as any, 0, 0, width, height);
  const alphaImage = aCtx.getImageData(0, 0, width, height);

  const c = colorImage.data;
  const a = alphaImage.data;
  // Use the alpha frame's luminance (red channel after drawImage) as the output alpha.
  for (let i = 0; i < c.length; i += 4) {
    c[i + 3] = a[i]; // take R as alpha (grayscale)
  }

  ctx.putImageData(colorImage, 0, 0);
  return new VideoFrame(state.mergeCanvas, {
    timestamp: colorFrame.timestamp,
    duration: colorFrame.duration ?? undefined,
  });
}

function createAlphaFrameHandler(assetId: string) {
  return (alphaFrame: VideoFrame) => {
    const state = assetStates.get(assetId);
    if (!state) {
      alphaFrame.close();
      return;
    }

    const ts = alphaFrame.timestamp;
    const pending = state.pendingColorFramesByTimestamp.get(ts);
    if (pending) {
      state.pendingColorFramesByTimestamp.delete(ts);

      // If request has moved on, drop both frames.
      if (pending.requestId !== state.currentRequestId) {
        pending.frame.close();
        alphaFrame.close();
        return;
      }

      let merged: VideoFrame | null = null;
      try {
        merged = mergeAlphaIntoColor(state, pending.frame, alphaFrame);
      } catch {
        merged = pending.frame;
      } finally {
        // If mergeAlphaIntoColor returned the original color frame, don't double-close.
        if (merged !== pending.frame) pending.frame.close();
        alphaFrame.close();
      }

      // Dispatch merged frame through the normal pipeline.
      dispatchDecodedFrame(assetId, merged);
      return;
    }

    // Store alpha frame for when the corresponding color frame arrives.
    // Bound the map to avoid unbounded growth.
    if (state.alphaFramesByTimestamp.size > 120) {
      const firstKey = state.alphaFramesByTimestamp.keys().next().value;
      if (firstKey !== undefined) {
        state.alphaFramesByTimestamp.get(firstKey)?.close();
        state.alphaFramesByTimestamp.delete(firstKey);
      }
    }
    state.alphaFramesByTimestamp.set(ts, alphaFrame);
  };
}

function ensureAlphaDecoder(state: AssetState, assetId: string) {
  if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
    return;
  }
  if (!state.config) {
    return;
  }

  state.alphaDecoder = new VideoDecoder({
    output: createAlphaFrameHandler(assetId),
    error: (e) => {
      console.error("Alpha VideoDecoder error", e);
      try {
        // @ts-ignore
        postMessage({
          type: "error",
          error: e.message ?? "Alpha VideoDecoder error",
          assetId,
        });
      } catch {
        // ignore
      }
    },
  });

  // Alpha side data is encoded with the same codec, but is not "embedded alpha"
  // in the bitstream. Some platforms reject `alpha: "keep"` for such streams,
  // so we try without the alpha hint first.
  try {
    const cfgAny: any = { ...(state.config as any) };
    delete cfgAny.alpha;
    state.alphaDecoder.configure(cfgAny as VideoDecoderConfig);
  } catch (e) {
    try {
      state.alphaDecoder.configure(state.config as VideoDecoderConfig);
    } catch (e2) {
      console.error("Alpha VideoDecoder configure failed", e, e2);
      try {
        state.alphaDecoder.close();
      } catch {
        // ignore
      }
      state.alphaDecoder = null;
    }
  }
}

function dispatchDecodedFrame(assetId: string, frame: VideoFrame) {
  const state = assetStates.get(assetId);
  if (!state) {
    frame.close();
    return;
  }

  if (state.customOutputHandler) {
    state.customOutputHandler(frame);
    return;
  }

  const frameTime = frame.timestamp / 1e6;

  
  // 1. Cache
  cacheFrame(state, frame);

  // 2. Preview
  if (state.showingPreview) {
    postFrame(assetId, frame, state.currentRequestId);
    state.showingPreview = false;
  }

  // 3. Check Target
  if (state.seekTargetTimestamp !== null) {
    // Track closest frame to the desired seek position as a fallback.
    const distance = Math.abs(frameTime - state.seekTargetTimestamp);
    if (
      !state.pendingSeekFrame ||
      distance < Math.abs(state.pendingSeekFrameTime - state.seekTargetTimestamp)
    ) {
      if (state.pendingSeekFrame) {
        state.pendingSeekFrame.close();
      }
      state.pendingSeekFrame = frame.clone();
      state.pendingSeekFrameTime = frameTime;
    }

    if (frameTime >= state.seekTargetTimestamp - 0.04) {
      state.seekDone = true;
      postFrame(assetId, frame, state.currentRequestId);
      state.seekTargetTimestamp = null;
      if (state.pendingSeekFrame) {
        state.pendingSeekFrame.close();
        state.pendingSeekFrame = null;
      }
    }
  }

  frame.close();
}

// Helpers
function findCachedFrame(state: AssetState, timestamp: number): VideoFrame | null {
  for (const [t, frame] of state.cachedDecodedFrames) {
    if (Math.abs(t - timestamp) < 0.05) return frame;
  }
  return null;
}

function cacheFrame(state: AssetState, frame: VideoFrame) {
  const frameTime = frame.timestamp / 1e6;
  if (state.cachedDecodedFrames.has(frameTime)) return;

  if (state.cachedDecodedFrames.size >= MAX_CACHE_SIZE) {
    const firstKey = state.cachedDecodedFrames.keys().next().value;
    if (firstKey !== undefined) {
      state.cachedDecodedFrames.get(firstKey)?.close();
      state.cachedDecodedFrames.delete(firstKey);
    }
  }
  state.cachedDecodedFrames.set(frameTime, frame.clone());
}

// Background task
async function cacheKeyPackets(state: AssetState) {
  if (state.isCachingKeyPackets) return;
  state.isCachingKeyPackets = true;
  try {
    // Lazy caching is usually sufficient and safer for performance
  } catch (e) {
    console.warn("Background keyframe caching failed", e);
  }
}

// Handler factory – binds an assetId so multiple assets can run concurrently.
const createFrameHandler = (assetId: string) => (frame: VideoFrame) => {

  console.log("createFrameHandler", assetId, frame.timestamp);
  
  const state = assetStates.get(assetId);
  if (!state) {
    frame.close();
    return;
  }

  // If an alpha decoder is active, merge alpha frames (from sideData) into this
  // color frame before dispatching it to the rest of the worker pipeline.
  if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
    const ts = frame.timestamp;
    const alpha = state.alphaFramesByTimestamp.get(ts);
    if (alpha) {
      state.alphaFramesByTimestamp.delete(ts);
      let merged: VideoFrame | null = null;
      try {
        merged = mergeAlphaIntoColor(state, frame, alpha);
      } catch {
        merged = frame;
      } finally {
        if (merged !== frame) frame.close();
        alpha.close();
      }
      dispatchDecodedFrame(assetId, merged);
      return;
    }

    // Wait for alpha output callback to arrive.
    state.pendingColorFramesByTimestamp.set(ts, {
      frame,
      requestId: state.currentRequestId,
    });

    // Bound pending color frames to avoid unbounded growth.
    if (state.pendingColorFramesByTimestamp.size > 120) {
      const firstKey = state.pendingColorFramesByTimestamp.keys().next().value;
      if (firstKey !== undefined) {
        const v = state.pendingColorFramesByTimestamp.get(firstKey);
        v?.frame.close();
        state.pendingColorFramesByTimestamp.delete(firstKey);
      }
    }
    return;
  }

  dispatchDecodedFrame(assetId, frame);
};

function postFrame(assetId: string, frame: VideoFrame, reqId: number) {
  const clone = frame.clone();
  const msg: WorkerResponse = {
    type: "frame",
    frame: clone,
    timestamp: clone.timestamp / 1e6,
    duration: (clone.duration ?? 0) / 1e6,
    requestId: reqId,
    assetId,
  };
  // @ts-ignore
  postMessage(msg, [clone]);
}


// Message Listener
self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const msg = e.data;

  // Mirror a tiny debug summary back to the main thread so that
  // activity in this worker is visible in the normal renderer console.
  try {
    // @ts-ignore
    postMessage({
      type: "debug",
      scope: "video-decoder-worker",
      event: "onmessage",
      assetId: (msg as any).assetId,
      requestId: (msg as any).requestId,
      payload: { type: msg.type },
    } satisfies WorkerResponse);
  } catch {
    // Best-effort only; never let debug plumbing break decoding.
  }

  
  try {
    switch (msg.type) {
      case "configure": {
        await handleConfigure(msg, msg.requestId);
        break;
      }
      case "seek": {
        await handleSeek(msg.timestamp, msg.forceAccurate, msg.requestId, msg.assetId);
        break;
      }
      case "iterate": {
        await handleIterate(msg.startTime, msg.endTime, msg.requestId, msg.assetId);
        break;
      }
      
      case "ack": {
        // Per-asset iteration flow control
        const id = msg.assetId;
        if (!id) break;
        const state = assetStates.get(id);
        if (!state) break;
        if (msg.requestId === state.currentRequestId) {
          state.iterationInFlight--;
          if (state.iterationInFlight < MAX_ITERATION_IN_FLIGHT && state.iterationResume) {
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
    console.error("Worker Error:", err);
    // @ts-ignore
    postMessage({
      type: "error",
      error: err.message,
      // Some messages don't carry requestId; fall back to 0 in that case.
      requestId: (msg as any).requestId || 0,
      assetId: msg.assetId,
    });
  }
};

async function handleConfigure(
  msg: Extract<WorkerMessage, { type: "configure" }>,
  requestId?: number,
) {
  const { assetId, config: cfg } = msg;
  const id = assetId ?? cfg.asset.id;
  
  if (!id) {
    throw new Error("configure message missing asset identifier");
  }
  dispose(id);
  const state = getOrCreateState(id);
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

  // 1. Setup Input (asset-centric only)
  // Interpret the asset path as a URL-like string that mediabunny knows how
  // to open (e.g. file://, app://, http://).
  let input: Input | null = null;
  let filePath: string | null = null;
  let primarySourceDir: "user-data" | "apex-cache" = "user-data";
  let secondarySourceDir: "user-data" | "apex-cache" = "apex-cache";
  filePath = fileURLToPathInWorker(cfg.asset.path);

  const hasUserDataPrefix =
    typeof cfg.userDataPath === "string" &&
    cfg.userDataPath.length > 0 &&
    filePath.includes(cfg.userDataPath?.replace(/^\/+/, ""))

  // If the incoming asset path is explicitly rooted under Electron's userData
  // directory, prefer serving via app://user-data regardless of engine_results
  // naming. Otherwise, preserve the existing heuristic that favors apex-cache
  // for engine_results outputs and user-data for everything else.
  if (!hasUserDataPrefix && filePath.includes("engine_results")) {
    primarySourceDir = "apex-cache";
    secondarySourceDir = "user-data";
  }
  try {
    const url = new URL(`app://${primarySourceDir}/${filePath}`);
    if (cfg.folderUuid && primarySourceDir === "apex-cache") {
      url.searchParams.set("folderUuid", cfg.folderUuid);
    }
    // Skip this URL up-front if we *know* it returns a 404; otherwise fall
    // back to the previous behavior and let mediabunny attempt to open it.
    const is404 = await isAppUrlDefinitely404(url);
    if (is404) {
      throw new Error("Primary app:// URL returned 404");
    }
    input = new Input({ formats, source: new UrlSource(url) });
  } catch (e) {
    try {
      if (!filePath) {
        throw new Error("Missing file path for secondary source");
      }
      const url = new URL(`app://${secondarySourceDir}/${filePath}`);
      if (cfg.folderUuid && secondarySourceDir === "apex-cache") {
        url.searchParams.set("folderUuid", cfg.folderUuid);
      }
      const is404 = await isAppUrlDefinitely404(url);
      if (is404) {
        throw new Error("Secondary app:// URL returned 404");
      }
      input = new Input({ formats, source: new UrlSource(url) });
    } catch (e) {
      throw new Error("Failed to create input");
    }
  }
  state.input = input;

  const videoTrack = await state.input.getPrimaryVideoTrack();
  if (!videoTrack) throw new Error("No video track found in worker");


  // 2. Setup Sink
  state.sink = new EncodedPacketSink(videoTrack);

  // 3. Setup Decoder
  // NOTE: Some TS lib.dom versions don't expose `alpha` on VideoDecoderConfig yet.
  // We still set it at runtime (when supported) to preserve transparency.
  const configAny: any = {
    ...cfg.videoDecoderConfig,
    optimizeForLatency: true,
  };
  if (configAny.alpha == null) {
    configAny.alpha = "keep";
  }
  state.config = configAny as VideoDecoderConfig;
 
  state.decoder = new VideoDecoder({
    output: createFrameHandler(id),
    error: (e) => {
      console.error("VideoDecoder error", e);
      // @ts-ignore
      postMessage({
        type: "error",
        error: e.message,
        assetId: id,
      });
    },
  });

  try {
    state.decoder.configure(state.config as VideoDecoderConfig);
  } catch (e) {
    // Fallback: if the platform/codec rejects alpha preservation, retry without it.
    // This keeps broad compatibility while enabling transparency where supported.
    try {
      const fallbackConfig: any = { ...(state.config as any) };
      delete fallbackConfig.alpha;
      state.config = fallbackConfig;
      state.decoder.configure(fallbackConfig as VideoDecoderConfig);
    } catch (e2) {
      console.error("VideoDecoder configure failed", e, e2);
      // @ts-ignore
      postMessage({
        type: "error",
        error: (e2 as any)?.message ?? "VideoDecoder configure failed",
        assetId: id,
      });
      return;
    }
  }
  void cacheKeyPackets(state);
  postMessage({
    type: "ready",
    requestId,
    assetId: id,
  });
}

async function handleSeek(
  timestamp: number,
  forceAccurate: boolean,
  requestId: number,
  assetId?: string,
) {

  const id = assetId;
 
  if (!id) return;
  const state = assetStates.get(id);
  if (!state || !state.decoder || !state.sink) return;

  // Cancel any active iteration handler to prevent hijacking seek output
  state.customOutputHandler = null;
  // Also resume any stuck flow control so iterate can exit cleanly
  if (state.iterationResume) {
      state.iterationResume();
      state.iterationResume = null;
  }

  state.currentRequestId = requestId;
  resetAlphaMergeQueues(state);

  // Reset any previous seek fallback frame
  if (state.pendingSeekFrame) {
    state.pendingSeekFrame.close();
    state.pendingSeekFrame = null;
  }

  const now = performance.now();
  const timeSinceLast = now - state.lastSeekTime;
  const dist = Math.abs(timestamp - (state.lastSeekTimestamp || 0));
  state.lastSeekTime = now;
  state.lastSeekTimestamp = timestamp;

  const isFastScrubbing = !forceAccurate && timeSinceLast < 150 && dist > 0.5;

  // 1. Cache Hit
  const cached = findCachedFrame(state, timestamp);
  console.log("handleSeek cached", id, timestamp, cached);
  if (cached) {
    postFrame(id, cached, requestId);
    // @ts-ignore
    postMessage({
      type: "seekDone",
      requestId,
      assetId: id,
    });
    return;
  }

  state.seekTargetTimestamp = timestamp;
  state.seekDone = false;
  state.showingPreview = false;

  // IMPORTANT:
  // After VideoDecoder.reset()/configure() (and sometimes after flush()), WebCodecs
  // requires the *next* decode() call to be a key frame. If we ask mediabunny for
  // a "key packet" without verification, it can occasionally return a non-key
  // packet (especially during rapid scrubbing), which triggers:
  //   DataError: A key frame is required after configure() or flush().
  // So we always verify key packets for seeks.
  const currentPacket = await state.sink.getKeyPacket(timestamp, {
    verifyKeyPackets: true,
  });


  if (!currentPacket) return;

  if (!state.keyPacketCache.has(currentPacket.timestamp)) {
    state.keyPacketCache.set(currentPacket.timestamp, currentPacket);
  }


  if (state.currentRequestId !== requestId) return;


  if ((state.decoder.state as string) === "closed") return;

  if (state.config) state.decoder.configure(state.config);
  if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
    try {
      state.alphaDecoder.reset();
      // Configure alpha decoder without the alpha hint when possible.
      const cfgAny: any = { ...(state.config as any) };
      delete cfgAny.alpha;
      state.alphaDecoder.configure(cfgAny as VideoDecoderConfig);
    } catch {
      // ignore; we'll recreate lazily if needed
    }
  }

  const isKeyframeRequiredError = (e: any): boolean => {
    const msg = (e?.message ?? "").toString();
    return e?.name === "DataError" && /key\s*frame/i.test(msg);
  };

  // Decode the verified key packet first. If the runtime still complains (some
  // platforms are extra strict about "key" typing), retry by re-fetching a
  // verified key packet at the same timestamp and decoding that.
  try {
    const chunk = currentPacket.toEncodedVideoChunk();
    state.decoder.decode(chunk);
  } catch (e: any) {
    if (isKeyframeRequiredError(e)) {
      try {
        // Re-verify from the demuxer and retry once.
        const retryKey = await state.sink.getKeyPacket(timestamp, {
          verifyKeyPackets: true,
        });
        if (retryKey && (state.decoder.state as string) !== "closed") {
          state.decoder.reset();
          if (state.config) state.decoder.configure(state.config);
          state.decoder.decode(retryKey.toEncodedVideoChunk());
        }
      } catch {
        // fall through to the outer error handler
      }
    }
    throw e;
  }


  if (currentPacket.sideData?.alpha) {
    ensureAlphaDecoder(state, id);
    if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
      try {
        state.alphaDecoder.decode(currentPacket.alphaToEncodedVideoChunk());
      } catch {
        // ignore alpha decode errors; color still decodes
      }
    }
  }

  if (isFastScrubbing) {
    state.showingPreview = true;
    await new Promise((r) => setTimeout(r, 80));
    //if (state.currentRequestId !== requestId) return;
    state.showingPreview = false;
  }

  

  const packets = state.sink.packets(currentPacket);
  for await (const packet of packets) {
    if ((state.decoder.state as string) === "closed") break;
    const colorChunk = packet.toEncodedVideoChunk();
    state.decoder.decode(colorChunk);
    if (packet.sideData?.alpha) {
      ensureAlphaDecoder(state, id);
      if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
        try {
          state.alphaDecoder.decode(packet.alphaToEncodedVideoChunk());
        } catch (e) {
          console.log(state.alphaDecoder, "alpha decode error", e);
          // ignore
        }
      }
    }
    if (state.seekDone) break;
    if (state.currentRequestId !== requestId) break;
    if (packet.timestamp > timestamp + 0.1) break;
  }


  if (forceAccurate) {
    await state.decoder.flush();
    if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
      try {
        await state.alphaDecoder.flush();
      } catch {
        // ignore
      }
    }
  }

  // If we never hit the seek threshold inside onFrameHandler (e.g. end of
  // stream or sparse frames near the target), fall back to the closest frame
  // we observed so the caller always gets something for an initial seek.
  if (!state.seekDone && state.pendingSeekFrame) {
    state.seekTargetTimestamp = null;
    postFrame(id, state.pendingSeekFrame, requestId);
    // @ts-ignore - VideoFrame is available at runtime, but TS's lib typing
    // can treat it as `never` in some worker configs.
    state.pendingSeekFrame.close();
    state.pendingSeekFrame = null;
  }

  // @ts-ignore
  postMessage({
    type: "seekDone",
    requestId,
    assetId: id,
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
    if (!state || !state.decoder || !state.sink || !state.config) return;

    state.currentRequestId = requestId;
    resetAlphaMergeQueues(state);

    // Same keyframe requirement as seek: after reset/configure, the first decode
    // must be a key frame. Verify key packets to avoid intermittent DataError
    // during iteration starts.
    const keyPacket = await state.sink.getKeyPacket(startTime, {
      verifyKeyPackets: true,
    });
    if (!keyPacket) {
        // @ts-ignore
        postMessage({
          type: "iterateDone",
          requestId,
          assetId: id,
        });
        return;
    }

    if ((state.decoder.state as string) === "closed") return;
    state.decoder.reset();
    state.decoder.configure(state.config);
    if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
      try {
        state.alphaDecoder.reset();
        const cfgAny: any = { ...(state.config as any) };
        delete cfgAny.alpha;
        state.alphaDecoder.configure(cfgAny as VideoDecoderConfig);
      } catch {
        // ignore
      }
    }

    // Override handler temporarily for iteration
    
    // To keep it simple in this worker version, let's reuse the decoder but clear the Seek target.
    state.seekTargetTimestamp = null; 
    
    // Reset flow control
    state.iterationInFlight = 0;
    state.iterationResume = null;

    const iterationHandler = (frame: VideoFrame) => {
        const frameTime = frame.timestamp / 1e6;
        if (frameTime < startTime || frameTime > endTime + 0.05) {
            frame.close();
            return;
        }
        
        state.iterationInFlight++;
        
        // Post immediately
        const clone = frame.clone();
        const msg = { 
            type: "frame", 
            frame: clone, 
            timestamp: frameTime, 
            duration: (frame.duration ?? 0) / 1e6,
            requestId,
            assetId: id,
        };
        // @ts-ignore
        postMessage(msg, [clone]);
        frame.close();
    };

    // Use dynamic handler instead of recreating decoder
    state.customOutputHandler = iterationHandler;


    try {
        if ((state.decoder.state as string) === "closed") {
             // @ts-ignore
             postMessage({
               type: "error",
               error: "Decoder closed",
               requestId,
               assetId: id,
             });
             return;
        }
        
        // Always decode color into the main decoder, and alpha (when present)
        // into the alphaDecoder for later merge.
        state.decoder.decode(keyPacket.toEncodedVideoChunk());
        if (keyPacket.sideData?.alpha) {
          ensureAlphaDecoder(state, id);
          if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
            try {
              state.alphaDecoder.decode(keyPacket.alphaToEncodedVideoChunk());
            } catch {
              // ignore
            }
          }
        }
        const packets = state.sink.packets(keyPacket);
        
        for await (const packet of packets) {
            if (state.currentRequestId !== requestId) break;
            if (packet.timestamp > endTime + 0.1) break;
            
            if ((state.decoder.state as string) === "closed") break;
            
            // Flow control: Wait if too many frames are pending
            while (state.iterationInFlight >= MAX_ITERATION_IN_FLIGHT) {
                 if (state.currentRequestId !== requestId) break;
                 await new Promise<void>(r => state.iterationResume = r);
            }

            state.decoder.decode(packet.toEncodedVideoChunk());
            if (packet.sideData?.alpha) {
              ensureAlphaDecoder(state, id);
              if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
                try {
                  state.alphaDecoder.decode(packet.alphaToEncodedVideoChunk());
                } catch {
                  // ignore
                }
              }
            }
            
            while (state.decoder.decodeQueueSize > 6) {
                 await new Promise(r => setTimeout(r, 5));
            }
        }
        
        if ((state.decoder.state as string) !== "closed") {
            await state.decoder.flush();
        }
        if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
          try {
            await state.alphaDecoder.flush();
          } catch {
            // ignore
          }
        }
    } finally {
        state.customOutputHandler = null;
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
      state.decoder.close();
      state.decoder = null;
  }
    if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
      try {
        state.alphaDecoder.close();
      } catch {
        // ignore
      }
      state.alphaDecoder = null;
    }
    resetAlphaMergeQueues(state);
    for (const frame of state.cachedDecodedFrames.values()) {
    frame.close();
  }
    state.cachedDecodedFrames.clear();
    state.keyPacketCache.clear();
    state.input = null;
    state.sink = null;
    state.seekTargetTimestamp = null;
    state.pendingSeekFrame?.close();
    state.pendingSeekFrame = null;
    state.seekDone = false;
    state.customOutputHandler = null;
    state.iterationInFlight = 0;
    state.iterationResume = null;
    return;
  }

  // Dispose all assets
  for (const [id] of assetStates) {
    dispose(id);
  }
  assetStates.clear();
}
