#!/usr/bin/env node

import { Worker } from "node:worker_threads";
import { performance } from "node:perf_hooks";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const repoRoot = path.resolve(__dirname, "..");

const addonPath = path.join(
  repoRoot,
  "apps",
  "app",
  "packages",
  "native-decoder",
  "build",
  "Release",
  "addon.node",
);
const workerPath = path.join(
  repoRoot,
  "apps",
  "app",
  "packages",
  "preload",
  "dist",
  "nativeDecoder.worker.cjs",
);

const filePath = process.argv[2];
const frames = Number(process.argv[3] ?? 30);

if (!filePath) {
  console.error("Usage: node scripts/native-decoder-worker-benchmark.mjs <video-file> [frames]");
  process.exit(1);
}

function summarizeLags(lags) {
  if (!lags.length) return { avgMs: 0, p95Ms: 0, maxMs: 0 };
  const arr = [...lags].sort((a, b) => a - b);
  const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
  const p95 = arr[Math.min(arr.length - 1, Math.floor(arr.length * 0.95))];
  const max = arr[arr.length - 1];
  return {
    avgMs: Number(avg.toFixed(2)),
    p95Ms: Number(p95.toFixed(2)),
    maxMs: Number(max.toFixed(2)),
  };
}

async function measure(name, fn) {
  const intervalMs = 16;
  let ticks = 0;
  let last = performance.now();
  const lags = [];
  const timer = setInterval(() => {
    const now = performance.now();
    lags.push(Math.max(0, now - last - intervalMs));
    ticks++;
    last = now;
  }, intervalMs);

  const t0 = performance.now();
  await fn();
  const elapsed = performance.now() - t0;
  clearInterval(timer);

  const expectedTicks = Math.floor(elapsed / intervalMs);
  return {
    totalMs: Number(elapsed.toFixed(1)),
    fpsEquivalent: Number((frames / (elapsed / 1000)).toFixed(2)),
    timerTicksObserved: ticks,
    timerTicksExpected: expectedTicks,
    timerTicksMissed: Math.max(0, expectedTicks - ticks),
    lag: summarizeLags(lags),
  };
}

async function directScenario() {
  const api = require(addonPath);
  const info = api.loadFile(filePath);
  const w = info.video.width;
  const h = info.video.height;
  const buf = new Uint8Array(w * h * 4);
  for (let i = 0; i < frames; i++) api.decodeFrameInto(filePath, buf, i / 30, false);
}

async function workerScenario() {
  const worker = new Worker(workerPath, { workerData: { addonPath } });
  let reqId = 1;
  const pending = new Map();

  const call = (type, payload) =>
    new Promise((resolve, reject) => {
      const id = reqId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, type, payload });
    });

  worker.on("message", (m) => {
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.ok) p.resolve(m.result);
    else p.reject(new Error(m.error));
  });
  worker.on("error", (e) => {
    for (const p of pending.values()) p.reject(e);
    pending.clear();
  });

  const info = await call("loadFile", { filePath });
  for (let i = 0; i < frames; i++) {
    await call("decodeFrame", {
      filePath,
      width: info.video.width,
      height: info.video.height,
      timestamp: i / 30,
      keyframeOnly: false,
    });
  }
  await worker.terminate();
}

const direct = await measure("direct_main_thread", directScenario);
const worker = await measure("worker_thread_decode", workerScenario);
console.log(JSON.stringify({ filePath, frames, direct, worker }, null, 2));
