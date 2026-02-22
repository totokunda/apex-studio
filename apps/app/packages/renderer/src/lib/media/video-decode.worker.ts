import type {
  DecodeWorkerRequest,
  DecodeWorkerResponse,
  DecodeSessionDescriptor,
  SessionId,
  JobId,
  SourceId,
  SerializedEncodedChunk,
} from "./video-decode-protocol";

type DecodeMeta = {
  requestId: number;
  jobId: JobId;
  sourceId: SourceId;
  timestampSec: number;
  durationSec: number;
  expectsAlpha: boolean;
};

type SessionRuntimeState = {
  descriptor: DecodeSessionDescriptor;
  cancelledJobs: Set<JobId>;
  createdAtMs: number;
  config: VideoDecoderConfig;
  decoder: VideoDecoder | null;
  alphaDecoder: VideoDecoder | null;
  colorMetaQueue: DecodeMeta[];
  alphaMetaQueue: DecodeMeta[];
  pendingColorByTimestamp: Map<number, { frame: VideoFrame; meta: DecodeMeta }>;
  pendingAlphaByTimestamp: Map<number, VideoFrame>;
  mergeCanvas: OffscreenCanvas | null;
  mergeCtx: OffscreenCanvasRenderingContext2D | null;
  alphaCanvas: OffscreenCanvas | null;
  alphaCtx: OffscreenCanvasRenderingContext2D | null;
};

const sessionStates = new Map<SessionId, SessionRuntimeState>();
const MAX_PENDING_MERGE_FRAMES = 120;
const MAX_DECODE_QUEUE_SIZE = 8;
const MAX_DECODE_QUEUE_WAIT_MS = 500;
const DECODE_QUEUE_SLEEP_MS = 5;

function postDecode(msg: DecodeWorkerResponse, transfer: Transferable[] = []) {
  // @ts-ignore
  postMessage(msg, transfer);
}

function postDebug(
  event: string,
  opts?: {
    sessionId?: SessionId;
    sourceId?: SourceId;
    requestId?: number;
    payload?: unknown;
  },
) {
  postDecode({
    type: "debug",
    event,
    sessionId: opts?.sessionId,
    sourceId: opts?.sourceId,
    requestId: opts?.requestId,
    payload: opts?.payload,
  });
}

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

function closeFrameSafe(frame: VideoFrame | null | undefined) {
  if (!frame) return;
  try {
    frame.close();
  } catch {
    // ignore
  }
}

function closeDecoderSafe(decoder: VideoDecoder | null | undefined) {
  if (!decoder) return;
  try {
    if ((decoder.state as string) !== "closed") {
      decoder.close();
    }
  } catch {
    // ignore
  }
}

function clearMergeQueues(state: SessionRuntimeState) {
  for (const item of state.pendingColorByTimestamp.values()) {
    closeFrameSafe(item.frame);
  }
  state.pendingColorByTimestamp.clear();
  for (const frame of state.pendingAlphaByTimestamp.values()) {
    closeFrameSafe(frame);
  }
  state.pendingAlphaByTimestamp.clear();
}

function clearSessionQueues(state: SessionRuntimeState) {
  state.colorMetaQueue.length = 0;
  state.alphaMetaQueue.length = 0;
  clearMergeQueues(state);
}

function clearSessionRuntime(state: SessionRuntimeState) {
  clearSessionQueues(state);
  closeDecoderSafe(state.decoder);
  closeDecoderSafe(state.alphaDecoder);
  state.decoder = null;
  state.alphaDecoder = null;
  state.mergeCanvas = null;
  state.mergeCtx = null;
  state.alphaCanvas = null;
  state.alphaCtx = null;
}

function isQuotaExceededError(err: any): boolean {
  return err?.name === "QuotaExceededError";
}

async function waitForDecodeQueue(
  decoder: VideoDecoder,
  maxSize: number = MAX_DECODE_QUEUE_SIZE,
) {
  let waited = 0;
  while (decoder.decodeQueueSize > maxSize && waited < MAX_DECODE_QUEUE_WAIT_MS) {
    await sleep(DECODE_QUEUE_SLEEP_MS);
    waited += DECODE_QUEUE_SLEEP_MS;
  }
}

function toEncodedVideoChunk(packet: SerializedEncodedChunk): EncodedVideoChunk {
  return new EncodedVideoChunk({
    type: packet.type,
    timestamp: packet.timestamp,
    duration: packet.duration,
    data: new Uint8Array(packet.data),
  });
}

async function decodeSerializedChunk(
  decoder: VideoDecoder,
  packet: SerializedEncodedChunk,
): Promise<boolean> {
  for (let attempt = 0; attempt < 2; attempt++) {
    await waitForDecodeQueue(decoder, MAX_DECODE_QUEUE_SIZE);
    try {
   
      decoder.decode(toEncodedVideoChunk(packet));
      return true;
    } catch (err: any) {
      if (isQuotaExceededError(err)) {
        await waitForDecodeQueue(decoder, 2);
        continue;
      }
      return false;
    }
  }
  return false;
}

function ensureMergeCanvases(state: SessionRuntimeState, width: number, height: number) {
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
  state: SessionRuntimeState,
  colorFrame: VideoFrame,
  alphaFrame: VideoFrame,
): VideoFrame {
  const width = colorFrame.displayWidth || (colorFrame as any).codedWidth || 0;
  const height = colorFrame.displayHeight || (colorFrame as any).codedHeight || 0;
  if (!width || !height) {
    return colorFrame;
  }

  ensureMergeCanvases(state, width, height);
  const ctx = state.mergeCtx;
  const aCtx = state.alphaCtx;
  if (!ctx || !aCtx || !state.mergeCanvas || !state.alphaCanvas) {
    return colorFrame;
  }

  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(colorFrame as any, 0, 0, width, height);
  const colorImage = ctx.getImageData(0, 0, width, height);

  aCtx.clearRect(0, 0, width, height);
  aCtx.drawImage(alphaFrame as any, 0, 0, width, height);
  const alphaImage = aCtx.getImageData(0, 0, width, height);

  const c = colorImage.data;
  const a = alphaImage.data;
  for (let i = 0; i < c.length; i += 4) {
    c[i + 3] = a[i];
  }
  ctx.putImageData(colorImage, 0, 0);
  return new VideoFrame(state.mergeCanvas, {
    timestamp: colorFrame.timestamp,
    duration: colorFrame.duration ?? undefined,
  });
}

function dispatchFrame(
  state: SessionRuntimeState,
  sessionId: SessionId,
  meta: DecodeMeta,
  frame: VideoFrame,
) {
  if (state.cancelledJobs.has(meta.jobId)) {
    closeFrameSafe(frame);
    postDecode({
      type: "jobCancelled",
      requestId: meta.requestId,
      sessionId,
      sourceId: meta.sourceId,
      jobId: meta.jobId,
    });
    return;
  }

  const timestampSec = frame.timestamp / 1e6;
  const durationSec = (frame.duration ?? Math.round(meta.durationSec * 1e6)) / 1e6;
  postDecode(
    {
      type: "frame",
      requestId: meta.requestId,
      sessionId,
      logicalClipId: state.descriptor.logicalClipId,
      sourceId: meta.sourceId,
      jobId: meta.jobId,
      frame,
      timestampSec,
      durationSec,
    },
    [frame],
  );
}

function trimPendingMergeMaps(state: SessionRuntimeState) {
  while (state.pendingColorByTimestamp.size > MAX_PENDING_MERGE_FRAMES) {
    const first = state.pendingColorByTimestamp.entries().next().value;
    if (!first) break;
    const [ts, value] = first as [number, { frame: VideoFrame; meta: DecodeMeta }];
    closeFrameSafe(value.frame);
    state.pendingColorByTimestamp.delete(ts);
  }
  while (state.pendingAlphaByTimestamp.size > MAX_PENDING_MERGE_FRAMES) {
    const first = state.pendingAlphaByTimestamp.entries().next().value;
    if (!first) break;
    const [ts, frame] = first as [number, VideoFrame];
    closeFrameSafe(frame);
    state.pendingAlphaByTimestamp.delete(ts);
  }
}

function onColorFrame(sessionId: SessionId, frame: VideoFrame) {
  const state = sessionStates.get(sessionId);
  if (!state) {
    closeFrameSafe(frame);
    return;
  }

  const meta = state.colorMetaQueue.shift();
  if (!meta) {
    closeFrameSafe(frame);
    postDebug("color-meta-missing", {
      sessionId,
      sourceId: state.descriptor.sourceId,
      payload: { frameTimestamp: frame.timestamp },
    });
    return;
  }

  if (meta.expectsAlpha) {
    const ts = frame.timestamp;
    const alpha = state.pendingAlphaByTimestamp.get(ts);
    if (alpha) {
      state.pendingAlphaByTimestamp.delete(ts);
      let merged: VideoFrame | null = null;
      try {
        merged = mergeAlphaIntoColor(state, frame, alpha);
      } catch {
        merged = frame;
      } finally {
        if (merged !== frame) {
          closeFrameSafe(frame);
        }
        closeFrameSafe(alpha);
      }
      dispatchFrame(state, sessionId, meta, merged);
      return;
    }

    state.pendingColorByTimestamp.set(ts, { frame, meta });
    trimPendingMergeMaps(state);
    return;
  }

  dispatchFrame(state, sessionId, meta, frame);
}

function onAlphaFrame(sessionId: SessionId, alphaFrame: VideoFrame) {
  const state = sessionStates.get(sessionId);
  if (!state) {
    closeFrameSafe(alphaFrame);
    return;
  }

  const meta = state.alphaMetaQueue.shift();
  if (meta && state.cancelledJobs.has(meta.jobId)) {
    closeFrameSafe(alphaFrame);
    return;
  }

  const ts = alphaFrame.timestamp;
  const pending = state.pendingColorByTimestamp.get(ts);
  if (pending) {
    state.pendingColorByTimestamp.delete(ts);
    let merged: VideoFrame | null = null;
    try {
      merged = mergeAlphaIntoColor(state, pending.frame, alphaFrame);
    } catch {
      merged = pending.frame;
    } finally {
      if (merged !== pending.frame) {
        closeFrameSafe(pending.frame);
      }
      closeFrameSafe(alphaFrame);
    }
    dispatchFrame(state, sessionId, pending.meta, merged);
    return;
  }

  state.pendingAlphaByTimestamp.set(ts, alphaFrame);
  trimPendingMergeMaps(state);
}

function buildConfig(descriptor: DecodeSessionDescriptor): VideoDecoderConfig {
  const cfgAny: any = {
    ...(descriptor.decoderConfig as any),
    optimizeForLatency: true,
  };
  if (cfgAny.alpha == null) {
    cfgAny.alpha = "keep";
  }
  return cfgAny as VideoDecoderConfig;
}

function createColorDecoder(state: SessionRuntimeState, sessionId: SessionId) {
  const decoder = new VideoDecoder({
    output: (frame) => onColorFrame(sessionId, frame),
    error: (e) => {
      postDecode({
        type: "error",
        requestId: 0,
        sessionId,
        sourceId: state.descriptor.sourceId,
        error: e.message ?? "VideoDecoder error",
      });
    },
  });
  state.decoder = decoder;
}

function createAlphaDecoder(state: SessionRuntimeState, sessionId: SessionId) {
  const alphaDecoder = new VideoDecoder({
    output: (frame) => onAlphaFrame(sessionId, frame),
    error: (e) => {
      postDecode({
        type: "error",
        requestId: 0,
        sessionId,
        sourceId: state.descriptor.sourceId,
        error: e.message ?? "Alpha VideoDecoder error",
      });
    },
  });
  state.alphaDecoder = alphaDecoder;
}

function ensureColorDecoderConfigured(state: SessionRuntimeState, sessionId: SessionId) {
  if (!state.decoder || (state.decoder.state as string) === "closed") {
    createColorDecoder(state, sessionId);
  }
  if (!state.decoder) {
    throw new Error(`Failed to create color decoder for session '${sessionId}'`);
  }
  const config = buildConfig(state.descriptor);
  state.config = config;
  try {
    state.decoder.configure(state.config);
  } catch {
    const fallback: any = { ...(state.config as any) };
    delete fallback.alpha;
    state.config = fallback as VideoDecoderConfig;
    state.decoder.configure(state.config);
  }
}

function ensureAlphaDecoderConfigured(state: SessionRuntimeState, sessionId: SessionId) {
  if (!state.alphaDecoder || (state.alphaDecoder.state as string) === "closed") {
    createAlphaDecoder(state, sessionId);
  }
  if (!state.alphaDecoder) return;
  const cfgAny: any = { ...(state.config as any) };
  delete cfgAny.alpha;
  try {
    state.alphaDecoder.configure(cfgAny as VideoDecoderConfig);
  } catch {
    // Best effort only; color decode can still proceed.
  }
}

function upsertSession(session: DecodeSessionDescriptor): SessionRuntimeState {
  const existing = sessionStates.get(session.sessionId);
  if (existing) {
    clearSessionRuntime(existing);
    existing.descriptor = session;
    existing.cancelledJobs.clear();
    existing.config = buildConfig(session);
    return existing;
  }
  
  const created: SessionRuntimeState = {
    descriptor: session,
    cancelledJobs: new Set<JobId>(),
    createdAtMs: performance.now(),
    config: buildConfig(session),
    decoder: null,
    alphaDecoder: null,
    colorMetaQueue: [],
    alphaMetaQueue: [],
    pendingColorByTimestamp: new Map(),
    pendingAlphaByTimestamp: new Map(),
    mergeCanvas: null,
    mergeCtx: null,
    alphaCanvas: null,
    alphaCtx: null,
  };
  sessionStates.set(session.sessionId, created);
  return created;
}

function getSessionOrError(
  sessionId: SessionId,
  requestId: number,
): SessionRuntimeState | null {
  const state = sessionStates.get(sessionId);
  if (state) return state;
  postDecode({
    type: "error",
    requestId,
    sessionId,
    error: `session '${sessionId}' not found`,
  });
  return null;
}

function handleCreateSession(msg: Extract<DecodeWorkerRequest, { type: "createSession" }>) {
  const state = upsertSession(msg.session);
  try {
    ensureColorDecoderConfigured(state, msg.session.sessionId);
  } catch (err: any) {
    postDecode({
      type: "error",
      requestId: msg.requestId,
      sessionId: msg.session.sessionId,
      sourceId: msg.session.sourceId,
      error: err?.message ?? "Failed to configure decoder session",
    });
    return;
  }
  postDebug("session-created", {
    sessionId: msg.session.sessionId,
    sourceId: msg.session.sourceId,
    requestId: msg.requestId,
    payload: {
      activeSessionCount: sessionStates.size,
      createdAtMs: state.createdAtMs,
      priority: msg.session.priority ?? "interactive",
    },
  });
  postDecode({
    type: "sessionReady",
    requestId: msg.requestId,
    sessionId: msg.session.sessionId,
    logicalClipId: msg.session.logicalClipId,
    sourceId: msg.session.sourceId,
  });
}

function handleResetSession(msg: Extract<DecodeWorkerRequest, { type: "resetSession" }>) {
  const state = getSessionOrError(msg.sessionId, msg.requestId);
  if (!state) return;
  clearSessionQueues(state);
  if (state.decoder && (state.decoder.state as string) !== "closed") {
    try {
      state.decoder.reset();
    } catch {
      // ignore
    }
  }
  if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
    try {
      state.alphaDecoder.reset();
    } catch {
      // ignore
    }
  }
  try {
    ensureColorDecoderConfigured(state, msg.sessionId);
    if (state.alphaDecoder) {
      ensureAlphaDecoderConfigured(state, msg.sessionId);
    }
  } catch (err: any) {
    postDecode({
      type: "error",
      requestId: msg.requestId,
      sessionId: msg.sessionId,
      sourceId: state.descriptor.sourceId,
      error: err?.message ?? "Failed to reset/configure session decoders",
    });
    return;
  }
  state.cancelledJobs.clear();
  postDebug("session-reset", {
    sessionId: msg.sessionId,
    sourceId: state.descriptor.sourceId,
    requestId: msg.requestId,
  });
  postDecode({
    type: "resetDone",
    requestId: msg.requestId,
    sessionId: msg.sessionId,
    sourceId: state.descriptor.sourceId,
  });
}

async function handleFlushSession(msg: Extract<DecodeWorkerRequest, { type: "flushSession" }>) {
  const state = getSessionOrError(msg.sessionId, msg.requestId);
  if (!state) return;
  try {
    if (state.decoder && (state.decoder.state as string) !== "closed") {
      await state.decoder.flush();
    }
    if (state.alphaDecoder && (state.alphaDecoder.state as string) !== "closed") {
      await state.alphaDecoder.flush();
    }
  } catch {
    // Best effort; emit debug regardless.
  }
  postDebug("session-flush", {
    sessionId: msg.sessionId,
    sourceId: state.descriptor.sourceId,
    requestId: msg.requestId,
  });
  postDecode({
    type: "flushDone",
    requestId: msg.requestId,
    sessionId: msg.sessionId,
    sourceId: state.descriptor.sourceId,
  });
}

function removeLastMetaIfMatch(queue: DecodeMeta[], meta: DecodeMeta) {
  const last = queue[queue.length - 1];
  if (!last) return;
  if (last.requestId === meta.requestId && last.jobId === meta.jobId) {
    queue.pop();
  }
}

async function handleDecodeChunk(msg: Extract<DecodeWorkerRequest, { type: "decodeChunk" }>) {
  const state = getSessionOrError(msg.sessionId, msg.requestId);
  if (!state) return;

  if (state.cancelledJobs.has(msg.jobId)) {
    postDecode({
      type: "jobCancelled",
      requestId: msg.requestId,
      sessionId: msg.sessionId,
      sourceId: msg.sourceId,
      jobId: msg.jobId,
    });
    return;
  }

  try {
    ensureColorDecoderConfigured(state, msg.sessionId);
  } catch (err: any) {
    postDecode({
      type: "error",
      requestId: msg.requestId,
      sessionId: msg.sessionId,
      sourceId: msg.sourceId,
      jobId: msg.jobId,
      error: err?.message ?? "Failed to ensure configured color decoder",
    });
    return;
  }

  const meta: DecodeMeta = {
    requestId: msg.requestId,
    jobId: msg.jobId,
    sourceId: msg.sourceId,
    timestampSec: msg.packet.timestampSec,
    durationSec: msg.packet.durationSec,
    expectsAlpha: !!msg.packet.alphaChunk,
  };

  state.colorMetaQueue.push(meta);
  const colorOk = state.decoder
    ? await decodeSerializedChunk(state.decoder, msg.packet.chunk)
    : false;
  if (!colorOk) {
    removeLastMetaIfMatch(state.colorMetaQueue, meta);
    postDecode({
      type: "error",
      requestId: msg.requestId,
      sessionId: msg.sessionId,
      sourceId: msg.sourceId,
      jobId: msg.jobId,
      error: "Failed to decode color chunk",
    });
    return;
  }

  if (msg.packet.alphaChunk) {
    try {
      ensureAlphaDecoderConfigured(state, msg.sessionId);
      const alphaMeta: DecodeMeta = { ...meta };
      state.alphaMetaQueue.push(alphaMeta);
      const alphaOk = state.alphaDecoder
        ? await decodeSerializedChunk(state.alphaDecoder, msg.packet.alphaChunk)
        : false;
      if (!alphaOk) {
        removeLastMetaIfMatch(state.alphaMetaQueue, alphaMeta);
        postDebug("alpha-chunk-decode-failed", {
          sessionId: msg.sessionId,
          sourceId: msg.sourceId,
          requestId: msg.requestId,
          payload: { jobId: msg.jobId, packetTimestampSec: msg.packet.timestampSec },
        });
      }
    } catch {
      postDebug("alpha-decoder-config-failed", {
        sessionId: msg.sessionId,
        sourceId: msg.sourceId,
        requestId: msg.requestId,
        payload: { jobId: msg.jobId },
      });
    }
  }

  postDecode({
    type: "decodeDone",
    requestId: msg.requestId,
    sessionId: msg.sessionId,
    sourceId: msg.sourceId,
    jobId: msg.jobId,
  });
}

function handleCancelJob(msg: Extract<DecodeWorkerRequest, { type: "cancelJob" }>) {
  const state = getSessionOrError(msg.sessionId, msg.requestId);
  if (!state) return;
  state.cancelledJobs.add(msg.jobId);

  state.colorMetaQueue = state.colorMetaQueue.filter((meta) => meta.jobId !== msg.jobId);
  state.alphaMetaQueue = state.alphaMetaQueue.filter((meta) => meta.jobId !== msg.jobId);

  for (const [ts, pending] of state.pendingColorByTimestamp) {
    if (pending.meta.jobId === msg.jobId) {
      closeFrameSafe(pending.frame);
      state.pendingColorByTimestamp.delete(ts);
    }
  }

  postDecode({
    type: "jobCancelled",
    requestId: msg.requestId,
    sessionId: msg.sessionId,
    sourceId: state.descriptor.sourceId,
    jobId: msg.jobId,
  });
}

function handleDisposeSession(msg: Extract<DecodeWorkerRequest, { type: "disposeSession" }>) {
  const state = sessionStates.get(msg.sessionId);
  if (!state) {
    postDecode({
      type: "error",
      requestId: msg.requestId,
      sessionId: msg.sessionId,
      error: `disposeSession: session '${msg.sessionId}' not found`,
    });
    return;
  }
  clearSessionRuntime(state);
  sessionStates.delete(msg.sessionId);
  postDebug("session-disposed", {
    sessionId: msg.sessionId,
    sourceId: state.descriptor.sourceId,
    requestId: msg.requestId,
    payload: { activeSessionCount: sessionStates.size },
  });
}

self.onmessage = async (event: MessageEvent<DecodeWorkerRequest>) => {
  const msg = event.data;
  try {
    switch (msg.type) {
      case "createSession":
        handleCreateSession(msg);
        break;
      case "resetSession":
        handleResetSession(msg);
        break;
      case "flushSession":
        await handleFlushSession(msg);
        break;
      case "decodeChunk":
        await handleDecodeChunk(msg);
        break;
      case "cancelJob":
        handleCancelJob(msg);
        break;
      case "disposeSession":
        handleDisposeSession(msg);
        break;
    }
  } catch (err: any) {
    postDecode({
      type: "error",
      requestId: (msg as any).requestId ?? 0,
      sessionId: (msg as any).sessionId,
      sourceId: (msg as any).sourceId,
      jobId: (msg as any).jobId,
      error: err?.message ?? "Unhandled decode worker error",
    });
  }
};
