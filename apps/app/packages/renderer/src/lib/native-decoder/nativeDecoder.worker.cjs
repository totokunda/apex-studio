"use strict";

const fs = require("node:fs");
const path = require("node:path");

function sanitizeAddonCandidates(payload) {
  if (!payload || !Array.isArray(payload.addonPathCandidates)) return [];
  return payload.addonPathCandidates.filter((v) => typeof v === "string" && v.length > 0);
}

function resolveAddonPath(payload) {
  const fromMessage = sanitizeAddonCandidates(payload);
  const fromEnv =
    typeof process !== "undefined" && typeof process.env?.NATIVE_DECODER_ADDON_PATH === "string"
      ? process.env.NATIVE_DECODER_ADDON_PATH
      : "";
  const cwd = typeof process !== "undefined" && typeof process.cwd === "function" ? process.cwd() : "";
  const candidates = [
    ...fromMessage,
    fromEnv,
    path.join(cwd, "packages", "native-decoder", "build", "Release", "addon.node"),
    path.join(cwd, "native-decoder", "build", "Release", "addon.node"),
  ].filter(Boolean);

  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {
      // ignore
    }
  }
  return candidates[0] || "";
}

let addon = null;

function getAddon(payload) {
  if (addon) return addon;
  const addonPath = resolveAddonPath(payload);
  if (!addonPath) throw new Error("Native decoder addon path could not be resolved");
  addon = require(addonPath);
  return addon;
}

function decodeFrame(payload) {
  const api = getAddon(payload);
  const { filePath, width, height, timestamp, keyframeOnly, decoderKey } = payload;
  const data = new Uint8Array(width * height * 4);
  const result = api.decodeFrameInto(
    filePath,
    data,
    timestamp,
    !!keyframeOnly,
    width,
    height,
    decoderKey
  );
  return { timestamp: result.timestamp, data };
}

function decodeNext(payload) {
  const api = getAddon(payload);
  const { filePath, width, height, startTime, endTime, decoderKey } = payload;
  const data = new Uint8Array(width * height * 4);
  const start = typeof startTime === "number" && startTime >= 0 ? startTime : undefined;
  const end = typeof endTime === "number" && endTime >= 0 ? endTime : undefined;
  const result = api.decodeNextFrame(filePath, data, start, end, width, height, decoderKey);
  if (!result) return null;
  return { timestamp: result.timestamp, data };
}

function run(type, payload) {
  const api = getAddon(payload);
  switch (type) {
    case "loadFile":
      return api.loadFile(payload.filePath, payload.decoderKey);
    case "decodeFrame":
      return decodeFrame(payload);
    case "decodeNextFrame":
      return decodeNext(payload);
    default:
      throw new Error(`Unknown native decoder worker op: ${type}`);
  }
}

function handleMessage(message, respond) {
  const { id, type, payload } = message || {};
  if (typeof id !== "number" || !type) return;
  try {
    const result = run(type, payload || {});
    if (result && result.data instanceof Uint8Array) {
      respond({ id, ok: true, result }, [result.data.buffer]);
      return;
    }
    respond({ id, ok: true, result });
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    respond({ id, ok: false, error: text });
  }
}

if (
  typeof self !== "undefined" &&
  typeof self.addEventListener === "function" &&
  typeof self.postMessage === "function"
) {
  self.addEventListener("message", (event) => {
    handleMessage(event.data, (response, transferList) => {
      self.postMessage(response, transferList);
    });
  });
} else {
  throw new Error("nativeDecoder.worker.cjs requires Web Worker runtime");
}
