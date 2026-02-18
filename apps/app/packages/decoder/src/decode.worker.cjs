"use strict";

const fs = require("node:fs");
const path = require("node:path");

// ─── Addon resolution (unchanged) ───────────────────────────────────────────

function sanitizeAddonCandidates(payload) {
  if (!payload || !Array.isArray(payload.addonPathCandidates)) return [];
  return payload.addonPathCandidates.filter((v) => typeof v === "string" && v.length > 0);
}

function resolveDemuxPath(payload) {
  const fromMessage = sanitizeAddonCandidates(payload);
  const fromEnv =
    typeof process !== "undefined" && typeof process.env?.DEMUX_ADDON_PATH === "string"
      ? process.env.DEMUX_ADDON_PATH
      : "";
  const cwd = typeof process !== "undefined" && typeof process.cwd === "function" ? process.cwd() : "";
  const candidates = [
    ...fromMessage,
    fromEnv,
    path.join(cwd, "packages", "decoder", "build", "Release", "demux.node"),
    path.join(cwd, "decoder", "build", "Release", "demux.node"),
  ].filter(Boolean);

  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {
      /* ignore */
    }
  }
  return candidates[0] || "";
}

let demux = null;
const FRAME_CACHE_MAX_SIZE = 24;
const FAST_SEEK_CACHE_TOLERANCE_MICROS = 50_000;
let frameCacheUseSeq = 0;

function getDemux(payload) {
  if (demux) return demux;
  const demuxPath = resolveDemuxPath(payload);
  if (!demuxPath) throw new Error("Demux addon path could not be resolved");
  demux = require(demuxPath);
  return demux;
}

// ─── Worker state: decoder + packets ───
let state = {
  streams: null,
  packets: null,
  videoPackets: [],
  videoPacketTimestampsMicros: [],
  keyVideoPacketIndices: [],
  videoStreamIndex: -1,
  config: null,
  decoder: null,
  duration: -1,
  durationMicros: -1,
  cursorVideoPacketIndex: 0,
  eofFlushed: false,
  outputFrames: [],
  frameCache: [],
  pendingDecodeResolve: null,
  seekGeneration: 0,
};
let decodeQueue = Promise.resolve();

// ─── Utilities ──────────────────────────────────────────────────────────────

function isFiniteNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function closeFrame(frame) {
  try {
    frame.close();
  } catch {
    /* ignore */
  }
}

function clearOutputFrames() {
  for (const frame of state.outputFrames) closeFrame(frame);
  state.outputFrames = [];
}

function clearFrameCache() {
  for (const entry of state.frameCache) closeFrame(entry.frame);
  state.frameCache = [];
}

function cloneFrameSafe(frame) {
  try {
    return frame.clone();
  } catch {
    return null;
  }
}

// ─── Frame cache (unchanged logic) ──────────────────────────────────────────

function cacheFrameResult(result, sourceVideoPacketIndex) {
  if (!result?.frame) return;
  const clone = cloneFrameSafe(result.frame);
  if (!clone) return;

  const timestamp =
    isFiniteNum(result.timestamp) ? result.timestamp : clone.timestamp;
  const duration =
    isFiniteNum(result.duration) ? result.duration : clone.duration;

  const existingIndex = state.frameCache.findIndex((entry) => entry.timestamp === timestamp);
  frameCacheUseSeq += 1;
  const entry = {
    frame: clone,
    timestamp,
    duration,
    sourceVideoPacketIndex:
      isFiniteNum(sourceVideoPacketIndex)
        ? Math.max(0, Math.floor(sourceVideoPacketIndex))
        : -1,
    lastUsedSeq: frameCacheUseSeq,
  };

  if (existingIndex >= 0) {
    closeFrame(state.frameCache[existingIndex].frame);
    state.frameCache[existingIndex] = entry;
  } else {
    state.frameCache.push(entry);
  }

  while (state.frameCache.length > FRAME_CACHE_MAX_SIZE) {
    let oldestIndex = 0;
    let oldestSeq = state.frameCache[0]?.lastUsedSeq ?? 0;
    for (let i = 1; i < state.frameCache.length; i += 1) {
      const seq = state.frameCache[i]?.lastUsedSeq ?? 0;
      if (seq < oldestSeq) {
        oldestSeq = seq;
        oldestIndex = i;
      }
    }
    const [removed] = state.frameCache.splice(oldestIndex, 1);
    if (removed?.frame) closeFrame(removed.frame);
  }
}

function findCachedFrameNear(targetMicros, toleranceMicros) {
  if (!state.frameCache || state.frameCache.length === 0) return null;
  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < state.frameCache.length; i += 1) {
    const entry = state.frameCache[i];
    const ts = entry?.timestamp;
    if (typeof ts !== "number" || !Number.isFinite(ts)) continue;
    const distance = Math.abs(ts - targetMicros);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  if (bestIndex < 0 || bestDistance > toleranceMicros) return null;
  const entry = state.frameCache[bestIndex];
  const clone = cloneFrameSafe(entry.frame);
  if (!clone) return null;
  frameCacheUseSeq += 1;
  entry.lastUsedSeq = frameCacheUseSeq;
  return {
    frame: clone,
    timestamp: entry.timestamp,
    duration: entry.duration,
    sourceVideoPacketIndex: entry.sourceVideoPacketIndex,
  };
}

function shiftOutputFrameResult() {
  const frame = state.outputFrames.shift();
  if (!frame) return null;
  return { frame, timestamp: frame.timestamp, duration: frame.duration };
}

// ─── Packet / config helpers (unchanged) ────────────────────────────────────

function packetToEncodedVideoChunk(packet) {
  const data = packet.data;
  const byteLength = data?.byteLength ?? data?.length ?? 0;
  let arrayBuffer;
  if (byteLength > 0) {
    const copy = new Uint8Array(byteLength);
    copy.set(data);
    arrayBuffer = copy.buffer;
  } else {
    arrayBuffer = new ArrayBuffer(0);
  }
  return new EncodedVideoChunk({
    type: packet.isKeyFrame ? "key" : "delta",
    timestamp: packet.timestampMicros >= 0 ? packet.timestampMicros : 0,
    data: arrayBuffer,
  });
}

function buildVideoDecoderConfig(videoStream) {
  const vdc = videoStream?.videoDecoderConfig;
  if (!vdc) return null;
  const config = {
    codec: vdc.codec,
    codedWidth: vdc.codedWidth,
    codedHeight: vdc.codedHeight,
    optimizeForLatency: true,
  };
  const desc = vdc.description;
  const descLen = desc?.byteLength ?? desc?.length ?? 0;
  if (descLen > 0) {
    const copy = new Uint8Array(descLen);
    copy.set(desc);
    config.description = copy.buffer;
  }
  return config;
}

// ─── Decoder setup (unchanged output callback pattern) ──────────────────────

function setupDecoder(config) {
  if (state.pendingDecodeResolve) {
    const resolve = state.pendingDecodeResolve;
    state.pendingDecodeResolve = null;
    resolve(null);
  }
  clearOutputFrames();
  state.eofFlushed = false;

  if (state.decoder) {
    try {
      state.decoder.close();
    } catch {
      /* ignore */
    }
    state.decoder = null;
  }
  if (!config) return;

  const decoder = new VideoDecoder({
    output: (frame) => {
      if (state.pendingDecodeResolve) {
        const resolve = state.pendingDecodeResolve;
        state.pendingDecodeResolve = null;
        resolve({ frame, timestamp: frame.timestamp, duration: frame.duration });
        return;
      }
      state.outputFrames.push(frame);
    },
    error: (e) => {
      const resolve = state.pendingDecodeResolve;
      state.pendingDecodeResolve = null;
      if (resolve) resolve(null);
      console.error("[decode.worker] VideoDecoder error:", e);
    },
  });

  decoder.configure(config);
  state.decoder = decoder;
}

/**
 * If the decoder has died (error callback nulled it, or state is "closed"),
 * recreate it from the stored config so the next operation can proceed.
 */
function ensureDecoder() {
  if (!state.config) return false;
  if (!state.decoder || state.decoder.state === "closed") {
    setupDecoder(state.config);
  }
  return !!state.decoder;
}

// ─── File loading (single-pass index build) ─────────────────────────────────

function loadFile(payload) {
  const addon = getDemux(payload);
  const result = addon.loadFile(payload.filePath);
  if (!result) throw new Error("Demux returned no result");
  if (!result.streams || !result.packets) {
    throw new Error("Demux returned invalid result");
  }

  const videoStreamIndex = result.streams.findIndex((s) => s.codecType === "video");
  if (videoStreamIndex < 0) {
    throw new Error("No video stream found");
  }

  // Single pass builds all three arrays at once
  const videoPackets = [];
  const videoPacketTimestampsMicros = [];
  const keyVideoPacketIndices = [];

  for (let i = 0; i < result.packets.length; i += 1) {
    const packet = result.packets[i];
    if (packet.streamIndex !== videoStreamIndex) continue;
    const videoIdx = videoPackets.length;
    videoPackets.push(i);
    videoPacketTimestampsMicros.push(
      typeof packet.timestampMicros === "number" ? packet.timestampMicros : -1,
    );
    if (packet.isKeyFrame) {
      keyVideoPacketIndices.push(videoIdx);
    }
  }

  if (keyVideoPacketIndices.length === 0 && videoPackets.length > 0) {
    keyVideoPacketIndices.push(0);
  }
  const config = buildVideoDecoderConfig(result.streams[videoStreamIndex]);
  if (!config) {
    throw new Error("Could not build VideoDecoderConfig");
  }
  state.streams = result.streams;
  state.packets = result.packets;
  state.videoPackets = videoPackets;
  state.videoPacketTimestampsMicros = videoPacketTimestampsMicros;
  state.keyVideoPacketIndices = keyVideoPacketIndices;
  state.videoStreamIndex = videoStreamIndex;
  state.config = config;
  state.duration = result.duration ?? -1;
  state.durationMicros = result.durationMicros ?? -1;
  state.cursorVideoPacketIndex = 0;
  state.seekGeneration += 1;
  clearFrameCache();
  setupDecoder(config);
  return {
    streams: result.streams,
    videoStreamIndex,
    videoDecoderConfig: config,
    videoPacketCount: videoPackets.length,
    keyVideoPacketCount: keyVideoPacketIndices.length,
    duration: result.duration,
    durationMicros: result.durationMicros,
  };
}

// ─── Core decode (unchanged promise+timeout pattern) ────────────────────────

function decodePacket(videoPacketIndex, timeoutMs = 120) {
  const packetIndex = state.videoPackets[videoPacketIndex];
  if (packetIndex == null || packetIndex < 0) {
    return Promise.resolve(null);
  }

  const packet = state.packets[packetIndex];
  if (!packet || !state.decoder) {
    return Promise.resolve(null);
  }

  const chunk = packetToEncodedVideoChunk(packet);
  state.pendingDecodeResolve = null;

  return new Promise((resolve) => {
    const onDone = (result) => {
      if (timeout) clearTimeout(timeout);
      state.pendingDecodeResolve = null;
      resolve(result);
    };

    state.pendingDecodeResolve = onDone;
    const timeout = setTimeout(() => {
      if (state.pendingDecodeResolve === onDone) {
        state.pendingDecodeResolve = null;
        resolve(null);
      }
    }, timeoutMs);

    try {
      state.decoder.decode(chunk);
    } catch (err) {
      console.error("[decode.worker] decode() threw:", err);
      onDone(null);
    }
  });
}

function enqueueDecodeOperation(task) {
  const op = decodeQueue.then(() => task());
  decodeQueue = op.catch(() => null).then(() => null);
  return op;
}

// ─── Index lookups (binary search) ──────────────────────────────────────────

function findPacketIndexAtOrAfterTimestamp(targetMicros) {
  const ts = state.videoPacketTimestampsMicros;
  if (!ts || ts.length === 0) return -1;
  if (!isFiniteNum(targetMicros) || targetMicros < 0) return 0;

  let lo = 0;
  let hi = ts.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (ts[mid] >= 0 && ts[mid] >= targetMicros) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo < ts.length ? lo : ts.length - 1;
}

function findPreviousKeyPacketIndex(videoPacketIndex) {
  if (!state.videoPackets || state.videoPackets.length === 0) return -1;
  const keys = state.keyVideoPacketIndices;
  if (!keys || keys.length === 0) return 0;

  let lo = 0;
  let hi = keys.length - 1;
  let best = keys[0];
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (keys[mid] <= videoPacketIndex) {
      best = keys[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

function clampVideoPacketIndex(index) {
  if (!state.videoPackets || state.videoPackets.length === 0) return -1;
  if (!isFiniteNum(index)) return 0;
  const i = Math.floor(index);
  if (i < 0) return 0;
  if (i >= state.videoPackets.length) return state.videoPackets.length - 1;
  return i;
}

// ─── Seek (unchanged decode loop, uses binary search for lookups) ───────────

async function seekToTimestampMicros(timestampMicros, seekGeneration, forceAccurate) {
  if (!state.config || !state.videoPackets || state.videoPackets.length === 0) return null;
  if (!ensureDecoder()) return null;

  const isStaleSeek = () => seekGeneration !== state.seekGeneration;
  const targetMicros =
    isFiniteNum(timestampMicros) && timestampMicros >= 0
      ? timestampMicros
      : 0;

  const targetPacketIndex = clampVideoPacketIndex(findPacketIndexAtOrAfterTimestamp(targetMicros));
  if (targetPacketIndex < 0) return null;
  if (!forceAccurate) {
    const cached = findCachedFrameNear(targetMicros, FAST_SEEK_CACHE_TOLERANCE_MICROS);
    if (cached?.frame) {
      const sourcePacketIndex = clampVideoPacketIndex(cached.sourceVideoPacketIndex);
      if (sourcePacketIndex >= 0) {
        state.cursorVideoPacketIndex = Math.min(sourcePacketIndex + 1, state.videoPackets.length);
      }
      state.eofFlushed = false;
      return { frame: cached.frame, timestamp: cached.timestamp, duration: cached.duration };
    }
  }
  const startPacketIndex = clampVideoPacketIndex(findPreviousKeyPacketIndex(targetPacketIndex));

  setupDecoder(state.config);
  state.cursorVideoPacketIndex = startPacketIndex;
  state.eofFlushed = false;

  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  let packetsProcessed = 0;
  const closeEnoughMicros = forceAccurate ? 10_000 : 40_000;
  const hardPacketBudget = forceAccurate ? Number.POSITIVE_INFINITY : 220;
  const decodeTimeoutMs = forceAccurate ? 20 : 6;

  const considerCandidate = (candidate, sourceVideoPacketIndex) => {
    if (!candidate?.frame) return;
    cacheFrameResult(candidate, sourceVideoPacketIndex);
    const ts =
      isFiniteNum(candidate.timestamp)
        ? candidate.timestamp
        : targetMicros;
    const distance = Math.abs(ts - targetMicros);
    if (distance <= bestDistance) {
      if (best?.frame) closeFrame(best.frame);
      best = candidate;
      bestDistance = distance;
    } else {
      closeFrame(candidate.frame);
    }
  };

  for (let i = startPacketIndex; i <= targetPacketIndex; i += 1) {
    if (isStaleSeek()) {
      if (best?.frame) closeFrame(best.frame);
      clearOutputFrames();
      return null;
    }
    const frameResult = await decodePacket(i, decodeTimeoutMs);
    packetsProcessed += 1;
    state.cursorVideoPacketIndex = i + 1;
    considerCandidate(frameResult, i);
    let queued = shiftOutputFrameResult();
    while (queued) {
      considerCandidate(queued, i);
      queued = shiftOutputFrameResult();
    }

    if (!forceAccurate) {
      if (best && bestDistance <= closeEnoughMicros) break;
      if (packetsProcessed >= hardPacketBudget) break;
    }
  }
  if (isStaleSeek()) {
    if (best?.frame) closeFrame(best.frame);
    clearOutputFrames();
    return null;
  }
  return best;
}

// ─── Sequential decode (unchanged) ──────────────────────────────────────────

async function decodeNextFrame(endTimestampMicros) {
  if (!state.videoPackets || state.videoPackets.length === 0) return null;
  if (!ensureDecoder()) return null;

  let queued = shiftOutputFrameResult();
  if (queued) {
    cacheFrameResult(queued, Math.max(0, state.cursorVideoPacketIndex - 1));
    return queued;
  }

  while (state.cursorVideoPacketIndex < state.videoPackets.length) {
    const decodePacketIndex = state.cursorVideoPacketIndex;
    const packetIndex = state.videoPackets[decodePacketIndex];
    const packet = state.packets?.[packetIndex];
    if (
      isFiniteNum(endTimestampMicros) &&
      endTimestampMicros >= 0 &&
      isFiniteNum(packet?.timestampMicros) &&
      packet.timestampMicros > endTimestampMicros
    ) {
      return null;
    }

    const frameResult = await decodePacket(decodePacketIndex, 32);
    state.cursorVideoPacketIndex += 1;
    if (frameResult && frameResult.frame) {
      cacheFrameResult(frameResult, decodePacketIndex);
      return frameResult;
    }

    queued = shiftOutputFrameResult();
    if (queued) {
      cacheFrameResult(queued, decodePacketIndex);
      return queued;
    }
  }

  if (state.decoder && !state.eofFlushed) {
    state.eofFlushed = true;
    try {
      await state.decoder.flush();
    } catch {
      /* ignore */
    }
    queued = shiftOutputFrameResult();
    if (queued) {
      cacheFrameResult(queued, Math.max(0, state.videoPackets.length - 1));
      return queued;
    }
  }

  return null;
}

// ─── Operation dispatch (unchanged) ─────────────────────────────────────────

function run(type, payload) {
  switch (type) {
    case "loadFile":
      return loadFile(payload);
    case "decodeFrame": {
      const { videoPacketIndex } = payload || {};
      return enqueueDecodeOperation(() => decodePacket(videoPacketIndex));
    }
    case "seek": {
      const { timestampMicros, forceAccurate, __seekGeneration } = payload || {};
      const seekGeneration =
        isFiniteNum(__seekGeneration)
          ? __seekGeneration
          : state.seekGeneration;
      return enqueueDecodeOperation(() =>
        seekToTimestampMicros(timestampMicros, seekGeneration, !!forceAccurate),
      );
    }
    case "decodeNext": {
      const { endTimestampMicros } = payload || {};
      return enqueueDecodeOperation(() => decodeNextFrame(endTimestampMicros));
    }
    case "flush": {
      if (state.decoder) {
        return enqueueDecodeOperation(async () => {
          try {
            await state.decoder.flush();
          } catch {
            /* ignore */
          }
          return null;
        });
      }
      return null;
    }
    default:
      throw new Error(`Unknown decode worker op: ${type}`);
  }
}

// ─── Message handling (unchanged) ───────────────────────────────────────────

function handleMessage(message, respond) {
  const { id, type, payload } = message || {};
  if (typeof id !== "number" || !type) return;
  const send = (response, transferList) => {
    if (transferList && transferList.length > 0) {
      respond(response, transferList);
    } else {
      respond(response);
    }
  };

  const runAsync = async () => {
    try {
      let opPayload = payload || {};
      if (type === "seek") {
        state.seekGeneration += 1;
        opPayload = { ...opPayload, __seekGeneration: state.seekGeneration };
      }
      const result = await run(type, opPayload);
      if (result && result.frame) {
        const response = {
          id,
          ok: true,
          result: { frame: result.frame, timestamp: result.timestamp, duration: result.duration },
        };
        send(response, [result.frame]);
      } else {
        send({ id, ok: true, result });
      }
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      send({ id, ok: false, error: text });
    }
  };
  if (type === "loadFile") {
    try {
      const result = run(type, payload || {});
      send({ id, ok: true, result });
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      send({ id, ok: false, error: text });
    }
  } else {
    runAsync();
  }
}

if (
  typeof self !== "undefined" &&
  typeof self.addEventListener === "function" &&
  typeof self.postMessage === "function"
) {
  self.addEventListener("message", (event) => {
    handleMessage(event.data, (response, transferList) => {
      if (transferList && transferList.length > 0) {
        self.postMessage(response, transferList);
      } else {
        self.postMessage(response);
      }
    });
  });
} else {
  throw new Error("decode.worker.cjs requires Web Worker runtime");
}
