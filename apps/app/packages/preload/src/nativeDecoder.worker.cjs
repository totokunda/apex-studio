"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { parentPort, workerData } = require("node:worker_threads");

function resolveAddonPath() {
  const fromWorkerData = workerData && typeof workerData.addonPath === "string"
    ? workerData.addonPath
    : "";
  const fromEnv = typeof process.env?.NATIVE_DECODER_ADDON_PATH === "string"
    ? process.env.NATIVE_DECODER_ADDON_PATH
    : "";
  const cwd = process.cwd();
  const candidates = [
    fromWorkerData,
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

const addonPath = resolveAddonPath();
if (!addonPath) throw new Error("Native decoder addon path could not be resolved");
const addon = require(addonPath);

function decodeFrame(payload) {
  const { filePath, width, height, timestamp, keyframeOnly, decoderKey } = payload;
  const data = new Uint8Array(width * height * 4);
  const result = addon.decodeFrameInto(
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
  const { filePath, width, height, startTime, endTime, decoderKey } = payload;
  const data = new Uint8Array(width * height * 4);
  const start = typeof startTime === "number" && startTime >= 0 ? startTime : undefined;
  const end = typeof endTime === "number" && endTime >= 0 ? endTime : undefined;
  const result = addon.decodeNextFrame(filePath, data, start, end, width, height, decoderKey);
  if (!result) return null;
  return { timestamp: result.timestamp, data };
}

function run(type, payload) {
  switch (type) {
    case "loadFile":
      return addon.loadFile(payload.filePath, payload.decoderKey);
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

if (!parentPort) throw new Error("nativeDecoder.worker.cjs requires parentPort");

parentPort.on("message", (message) => {
  handleMessage(message, (response, transferList) => {
    parentPort.postMessage(response, transferList);
  });
});
