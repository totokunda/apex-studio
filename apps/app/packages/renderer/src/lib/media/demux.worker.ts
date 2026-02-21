import {
  ALL_FORMATS,
  EncodedPacket,
  EncodedPacketSink,
  Input,
  UrlSource,
} from "mediabunny";
import type {
  DemuxPacketEnvelope,
  DemuxWorkerRequest,
  DemuxWorkerResponse,
  JobId,
  SourceDescriptor,
  SourceId,
} from "./video-decode-protocol";

type SourceRuntimeState = {
  descriptor: SourceDescriptor;
  input: Input | null;
  sink: EncodedPacketSink | null;
  activeJobs: Set<JobId>;
  keyPacketCache: Map<number, EncodedPacket>;
  createdAtMs: number;
};

const sourceStates = new Map<SourceId, SourceRuntimeState>();

const MAX_KEY_PACKET_CACHE = 64;
const DEFAULT_BATCH_SIZE = 8;
const MAX_BATCH_SIZE = 24;
const DEFAULT_MAX_PACKETS = 240;

function postDemux(msg: DemuxWorkerResponse, transfer: Transferable[] = []) {
  // @ts-ignore
  postMessage(msg, transfer);
}

function postDebug(
  event: string,
  opts?: { sourceId?: SourceId; requestId?: number; payload?: unknown },
) {
  postDemux({
    type: "debug",
    event,
    sourceId: opts?.sourceId,
    requestId: opts?.requestId,
    payload: opts?.payload,
  });
}

function fileURLToPathInWorker(raw: string): string {
  try {
    const u = new URL(raw);
    if (u.protocol === "file:" || u.protocol === "app:") {
      return decodeURIComponent(u.pathname.replace(/^\/+/, ""));
    }
    return decodeURIComponent((u.pathname || "").replace(/^\/+/, ""));
  } catch {
    return raw.replace(/^\/+/, "");
  }
}

async function isAppUrlDefinitely404(url: URL): Promise<boolean> {
  try {
    const res = await fetch(url.toString(), { method: "HEAD" });
    return res.status === 404;
  } catch {
    return false;
  }
}

function getOrCreateSourceState(source: SourceDescriptor): SourceRuntimeState {
  const existing = sourceStates.get(source.sourceId);
  if (existing) {
    existing.descriptor = source;
    return existing;
  }
  const created: SourceRuntimeState = {
    descriptor: source,
    input: null,
    sink: null,
    activeJobs: new Set<JobId>(),
    keyPacketCache: new Map<number, EncodedPacket>(),
    createdAtMs: performance.now(),
  };
  sourceStates.set(source.sourceId, created);
  return created;
}

function clearKeyPacketCache(state: SourceRuntimeState) {
  state.keyPacketCache.clear();
}

function rememberKeyPacket(state: SourceRuntimeState, packet: EncodedPacket) {
  const key = packet.timestamp;
  if (state.keyPacketCache.has(key)) return;
  if (state.keyPacketCache.size >= MAX_KEY_PACKET_CACHE) {
    const firstKey = state.keyPacketCache.keys().next().value;
    if (firstKey !== undefined) {
      state.keyPacketCache.delete(firstKey);
    }
  }
  state.keyPacketCache.set(key, packet);
}

function getCachedNearbyKeyPacket(
  state: SourceRuntimeState,
  timestampSec: number,
): EncodedPacket | null {
  let best: EncodedPacket | null = null;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const [ts, packet] of state.keyPacketCache) {
    const sec = ts / 1e6;
    const dist = Math.abs(sec - timestampSec);
    if (dist < bestDist) {
      best = packet;
      bestDist = dist;
    }
  }
  // Keep this strict to avoid stale/random starts.
  return bestDist <= 0.25 ? best : null;
}

function clampBatchSize(size: number | undefined): number {
  if (!Number.isFinite(size as number)) return DEFAULT_BATCH_SIZE;
  return Math.max(1, Math.min(MAX_BATCH_SIZE, Math.floor(size as number)));
}

function clampMaxPackets(maxPackets: number | undefined): number {
  if (!Number.isFinite(maxPackets as number)) return DEFAULT_MAX_PACKETS;
  return Math.max(1, Math.min(5000, Math.floor(maxPackets as number)));
}

function toSerializedChunk(chunk: EncodedVideoChunk) {
  const data = new Uint8Array(chunk.byteLength);
  chunk.copyTo(data);
  return {
    type: chunk.type,
    timestamp: chunk.timestamp,
    duration: chunk.duration ?? undefined,
    data: data.buffer,
  };
}

function toPacketEnvelope(
  sourceId: SourceId,
  requestId: number,
  jobId: JobId,
  packet: EncodedPacket,
): DemuxPacketEnvelope {
  const colorChunk = packet.toEncodedVideoChunk();
  const chunk = toSerializedChunk(colorChunk);
  const alphaChunk = packet.sideData?.alpha
    ? toSerializedChunk(packet.alphaToEncodedVideoChunk())
    : undefined;
  const timestampSec = packet.timestamp / 1e6;
  const durationSec =
    packet.duration != null ? packet.duration / 1e6 : (chunk.duration ?? 0) / 1e6;

  return {
    sourceId,
    requestId,
    jobId,
    timestampSec,
    durationSec,
    isKey: packet.type === "key",
    chunk,
    alphaChunk,
  };
}

function collectTransfersFromPackets(packets: DemuxPacketEnvelope[]): Transferable[] {
  const transfer: Transferable[] = [];
  for (const packet of packets) {
    transfer.push(packet.chunk.data);
    if (packet.alphaChunk) {
      transfer.push(packet.alphaChunk.data);
    }
  }
  return transfer;
}

async function buildInputForSource(source: SourceDescriptor): Promise<Input> {
  const filePath = fileURLToPathInWorker(source.path);

  const hasUserDataPrefix =
    typeof source.userDataPath === "string" &&
    source.userDataPath.length > 0 &&
    filePath.includes(source.userDataPath.replace(/^\/+/, ""));

  let primarySourceDir: "user-data" | "apex-cache" = "user-data";
  let secondarySourceDir: "user-data" | "apex-cache" = "apex-cache";
  if (!hasUserDataPrefix && filePath.includes("engine_results")) {
    primarySourceDir = "apex-cache";
    secondarySourceDir = "user-data";
  }

  try {
    const primary = new URL(`app://${primarySourceDir}/${filePath}`);
    if (source.folderUuid && primarySourceDir === "apex-cache") {
      primary.searchParams.set("folderUuid", source.folderUuid);
    }
    if (await isAppUrlDefinitely404(primary)) {
      throw new Error("Primary source returned 404");
    }
    return new Input({
      formats: ALL_FORMATS,
      source: new UrlSource(primary),
    });
  } catch {
    const secondary = new URL(`app://${secondarySourceDir}/${filePath}`);
    if (source.folderUuid && secondarySourceDir === "apex-cache") {
      secondary.searchParams.set("folderUuid", source.folderUuid);
    }
    if (await isAppUrlDefinitely404(secondary)) {
      throw new Error("Secondary source returned 404");
    }
    return new Input({
      formats: ALL_FORMATS,
      source: new UrlSource(secondary),
    });
  }
}

async function ensureSourceReady(source: SourceDescriptor): Promise<SourceRuntimeState> {
  const state = getOrCreateSourceState(source);
  if (state.sink) {
    return state;
  }

  state.input = await buildInputForSource(source);
  const videoTrack = await state.input.getPrimaryVideoTrack();
  if (!videoTrack) {
    throw new Error(`registerSource: no video track for '${source.sourceId}'`);
  }
  state.sink = new EncodedPacketSink(videoTrack);
  clearKeyPacketCache(state);
  return state;
}

async function getVerifiedKeyPacket(
  state: SourceRuntimeState,
  timestampSec: number,
): Promise<EncodedPacket | null> {
  if (!state.sink) return null;

  const cached = getCachedNearbyKeyPacket(state, timestampSec);
  if (cached) return cached;

  try {
    const keyPacket = await state.sink.getKeyPacket(timestampSec, {
      verifyKeyPackets: true,
    });
    if (keyPacket) {
      rememberKeyPacket(state, keyPacket);
      return keyPacket;
    }

    const nearby = await state.sink.getPacket(timestampSec);
    if (nearby) {
      const nextKey = await state.sink.getNextKeyPacket(nearby, {
        verifyKeyPackets: true,
      });
      if (nextKey) {
        rememberKeyPacket(state, nextKey);
        return nextKey;
      }
    }

    const first = await state.sink.getFirstPacket({ verifyKeyPackets: true });
    if (first) {
      rememberKeyPacket(state, first);
      return first;
    }
    return null;
  } catch {
    return null;
  }
}

async function getStartPacket(
  state: SourceRuntimeState,
  timestampSec: number,
  startAtKeyframe: boolean,
): Promise<EncodedPacket | null> {
  if (!state.sink) return null;
  if (startAtKeyframe) {
    return getVerifiedKeyPacket(state, timestampSec);
  }
  try {
    const packet = await state.sink.getPacket(timestampSec);
    if (packet?.type === "key") {
      rememberKeyPacket(state, packet);
    }
    return packet;
  } catch {
    return null;
  }
}

function isJobActive(state: SourceRuntimeState, jobId: JobId): boolean {
  return state.activeJobs.has(jobId);
}

async function handleRegisterSource(msg: Extract<DemuxWorkerRequest, { type: "registerSource" }>) {
  const state = await ensureSourceReady(msg.source);
  postDebug("source-registered", {
    sourceId: msg.source.sourceId,
    requestId: msg.requestId,
    payload: {
      createdAtMs: state.createdAtMs,
      activeSourceCount: sourceStates.size,
      keyPacketCache: state.keyPacketCache.size,
    },
  });
  postDemux({
    type: "sourceReady",
    requestId: msg.requestId,
    sourceId: msg.source.sourceId,
  });
}

function disposeSourceState(sourceId: SourceId) {
  const state = sourceStates.get(sourceId);
  if (!state) return;
  state.activeJobs.clear();
  clearKeyPacketCache(state);
  state.sink = null;
  state.input = null;
  sourceStates.delete(sourceId);
}

function handleDisposeSource(msg: Extract<DemuxWorkerRequest, { type: "disposeSource" }>) {
  if (!sourceStates.has(msg.sourceId)) {
    postDemux({
      type: "error",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      error: `disposeSource: source '${msg.sourceId}' not found`,
    });
    return;
  }

  disposeSourceState(msg.sourceId);
  postDebug("source-disposed", {
    sourceId: msg.sourceId,
    requestId: msg.requestId,
    payload: { activeSourceCount: sourceStates.size },
  });
}

async function handleGetKeyPacketAt(msg: Extract<DemuxWorkerRequest, { type: "getKeyPacketAt" }>) {
  const state = sourceStates.get(msg.sourceId);
  if (!state || !state.sink) {
    postDemux({
      type: "error",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      error: `getKeyPacketAt: source '${msg.sourceId}' not found`,
    });
    return;
  }

  const keyPacket = await getVerifiedKeyPacket(state, msg.timestampSec);
  if (!keyPacket) {
    postDemux({
      type: "error",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      error: "getKeyPacketAt: no key packet available",
    });
    return;
  }

  const packet = toPacketEnvelope(msg.sourceId, msg.requestId, -1, keyPacket);
  postDemux(
    {
      type: "keyPacket",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      packet,
    },
    [packet.chunk.data, ...(packet.alphaChunk ? [packet.alphaChunk.data] : [])],
  );
}

async function handleStreamPackets(msg: Extract<DemuxWorkerRequest, { type: "streamPackets" }>) {
  const state = sourceStates.get(msg.sourceId);
  if (!state || !state.sink) {
    postDemux({
      type: "error",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      jobId: msg.jobId,
      error: `streamPackets: source '${msg.sourceId}' not found`,
    });
    return;
  }

  const batchSize = clampBatchSize(msg.maxPackets && Math.min(msg.maxPackets, DEFAULT_BATCH_SIZE));
  const maxPackets = clampMaxPackets(msg.maxPackets);

  const startPacket = await getStartPacket(
    state,
    msg.startTimeSec,
    msg.startAtKeyframe,
  );

  if (!startPacket) {
    postDemux({
      type: "streamDone",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      jobId: msg.jobId,
    });
    return;
  }

  if (startPacket.type === "key") {
    rememberKeyPacket(state, startPacket);
  }

  state.activeJobs.add(msg.jobId);

  const batch: DemuxPacketEnvelope[] = [];
  let emitted = 0;
  try {
    for await (const packet of state.sink.packets(startPacket)) {
      if (!isJobActive(state, msg.jobId)) {
        break;
      }

      if (packet.timestamp / 1e6 > msg.endTimeSec + 0.1) {
        break;
      }

      if (packet.type === "key") {
        rememberKeyPacket(state, packet);
      }

      const envelope = toPacketEnvelope(
        msg.sourceId,
        msg.requestId,
        msg.jobId,
        packet,
      );

      batch.push(envelope);
      emitted++;

      if (batch.length >= batchSize) {
        const transfer = collectTransfersFromPackets(batch);
        const packetsOut = batch.splice(0, batch.length);
        postDemux(
          {
            type: "packets",
            requestId: msg.requestId,
            sourceId: msg.sourceId,
            jobId: msg.jobId,
            packets: packetsOut,
          },
          transfer,
        );
      }

      if (emitted >= maxPackets) {
        break;
      }
    }
  } catch (err: any) {
    postDemux({
      type: "error",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      jobId: msg.jobId,
      error: err?.message ?? "streamPackets failed",
    });
  } finally {
    if (batch.length > 0) {
      const transfer = collectTransfersFromPackets(batch);
      const packetsOut = batch.splice(0, batch.length);
      postDemux(
        {
          type: "packets",
          requestId: msg.requestId,
          sourceId: msg.sourceId,
          jobId: msg.jobId,
          packets: packetsOut,
        },
        transfer,
      );
    }

    const wasActive = state.activeJobs.delete(msg.jobId);
    if (!wasActive) {
      postDemux({
        type: "jobCancelled",
        requestId: msg.requestId,
        sourceId: msg.sourceId,
        jobId: msg.jobId,
      });
      return;
    }

    postDemux({
      type: "streamDone",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      jobId: msg.jobId,
    });
    postDebug("stream-complete", {
      sourceId: msg.sourceId,
      requestId: msg.requestId,
      payload: {
        jobId: msg.jobId,
        emittedPackets: emitted,
        maxPackets,
        batchSize,
        activeJobs: state.activeJobs.size,
      },
    });
  }
}

function handleCancelJob(msg: Extract<DemuxWorkerRequest, { type: "cancelJob" }>) {
  const state = sourceStates.get(msg.sourceId);
  if (!state) {
    postDemux({
      type: "error",
      requestId: msg.requestId,
      sourceId: msg.sourceId,
      jobId: msg.jobId,
      error: `cancelJob: source '${msg.sourceId}' not found`,
    });
    return;
  }

  state.activeJobs.delete(msg.jobId);
  postDemux({
    type: "jobCancelled",
    requestId: msg.requestId,
    sourceId: msg.sourceId,
    jobId: msg.jobId,
  });
}

self.onmessage = async (event: MessageEvent<DemuxWorkerRequest>) => {
  const msg = event.data;
  try {
    switch (msg.type) {
      case "registerSource":
        await handleRegisterSource(msg);
        break;
      case "disposeSource":
        handleDisposeSource(msg);
        break;
      case "getKeyPacketAt":
        await handleGetKeyPacketAt(msg);
        break;
      case "streamPackets":
        await handleStreamPackets(msg);
        break;
      case "cancelJob":
        handleCancelJob(msg);
        break;
    }
  } catch (err: any) {
    postDemux({
      type: "error",
      requestId: (msg as any).requestId ?? 0,
      sourceId: (msg as any).sourceId,
      jobId: (msg as any).jobId,
      error: err?.message ?? "Unhandled demux worker error",
    });
  }
};

