"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { parentPort, workerData } = require("node:worker_threads");

function resolveAddonPath() {
  const fromWorkerData = workerData && typeof workerData.addonPath === "string"
    ? workerData.addonPath
    : "";
  const cwd = process.cwd();
  const candidates = [
    fromWorkerData,
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
  const { filePath, width, height, timestamp, keyframeOnly } = payload;
  const data = new Uint8Array(width * height * 4);
  const result = addon.decodeFrameInto(filePath, data, timestamp, !!keyframeOnly);
  return { timestamp: result.timestamp, data };
}

function decodeNext(payload) {
  const { filePath, width, height, startTime, endTime } = payload;
  const data = new Uint8Array(width * height * 4);
  const start = typeof startTime === "number" ? startTime : undefined;
  const end = typeof endTime === "number" ? endTime : undefined;
  const result = addon.decodeNextFrame(filePath, data, start, end);
  if (!result) return null;
  return { timestamp: result.timestamp, data };
}

function run(type, payload) {
  switch (type) {
    case "loadFile":
      return addon.loadFile(payload.filePath);
    case "decodeFrame":
      return decodeFrame(payload);
    case "decodeNextFrame":
      return decodeNext(payload);
    default:
      throw new Error(`Unknown native decoder worker op: ${type}`);
  }
}

if (!parentPort) throw new Error("nativeDecoder.worker.cjs requires parentPort");

parentPort.on("message", (message) => {
  const { id, type, payload } = message || {};
  if (typeof id !== "number" || !type) return;
  try {
    const result = run(type, payload || {});
    if (result && result.data instanceof Uint8Array) {
      parentPort.postMessage({ id, ok: true, result }, [result.data.buffer]);
      return;
    }
    parentPort.postMessage({ id, ok: true, result });
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    parentPort.postMessage({ id, ok: false, error: text });
  }
});

