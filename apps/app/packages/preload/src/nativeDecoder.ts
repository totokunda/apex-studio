/**
 * Native decoder interface backed by a dedicated worker_thread.
 *
 * This keeps decode off the renderer main thread while preserving the same
 * API shape for renderer consumers (now async).
 */

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

type DecoderWorkerRequest =
  | { id: number; type: "loadFile"; payload: { filePath: string } }
  | {
      id: number;
      type: "decodeFrame";
      payload: {
        filePath: string;
        width: number;
        height: number;
        timestamp: number;
        keyframeOnly: boolean;
      };
    }
  | {
      id: number;
      type: "decodeNextFrame";
      payload: {
        filePath: string;
        width: number;
        height: number;
        startTime: number;
        endTime: number;
      };
    };

type DecoderWorkerResponse =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };

type Pending = {
  resolve: (value: any) => void;
  reject: (reason?: unknown) => void;
};

type NativeAddon = {
  loadFile: (filePath: string) => NativeDecoderFileInfo;
  decodeFrameInto: (
    filePath: string,
    buffer: Uint8Array,
    timestamp: number,
    keyframeOnly?: boolean,
  ) => { timestamp: number };
  decodeNextFrame: (
    filePath: string,
    buffer: Uint8Array,
    startTime?: number,
    endTime?: number,
  ) => { timestamp: number } | null;
};

let decodeWorker: Worker | null = null;
let workerEnabled = true;
let directAddon: NativeAddon | null = null;
let nextRequestId = 1;
const pending = new Map<number, Pending>();

function rejectAllPending(error: Error) {
  for (const { reject } of pending.values()) reject(error);
  pending.clear();
}

function getAddonPath(): string {
  const cwd = process.cwd();
  const candidates = [
    path.join(cwd, "packages", "native-decoder", "build", "Release", "addon.node"),
    path.join(cwd, "native-decoder", "build", "Release", "addon.node"),
    path.join(__dirname, "..", "..", "native-decoder", "build", "Release", "addon.node"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {
      // ignore
    }
  }
  return candidates[0];
}

function getWorkerPath(): string {
  const candidates = [
    path.join(__dirname, "nativeDecoder.worker.cjs"),
    path.join(__dirname, "..", "src", "nativeDecoder.worker.cjs"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {
      // ignore
    }
  }
  return candidates[0];
}

function ensureDecodeWorker(): Worker {
  if (!workerEnabled) throw new Error("native decoder worker disabled");
  if (decodeWorker) return decodeWorker;

  const worker = new Worker(getWorkerPath(), {
    workerData: { addonPath: getAddonPath() },
  });

  worker.on("message", (msg: DecoderWorkerResponse) => {
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.ok === true) {
      entry.resolve(msg.result);
      return;
    }
    entry.reject(new Error(msg.error || "Native decoder worker error"));
  });

  worker.on("error", (error) => {
    decodeWorker = null;
    rejectAllPending(error);
  });

  worker.on("exit", (code) => {
    decodeWorker = null;
    if (code !== 0) {
      rejectAllPending(new Error(`Native decoder worker exited with code ${code}`));
    }
  });

  decodeWorker = worker;
  return worker;
}

function postToDecodeWorker<T>(type: DecoderWorkerRequest["type"], payload: object): Promise<T> {
  const worker = ensureDecodeWorker();
  const id = nextRequestId++;
  const request = { id, type, payload } as DecoderWorkerRequest;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage(request);
  });
}

function getDirectAddon(): NativeAddon {
  if (directAddon) return directAddon;
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  directAddon = require(getAddonPath()) as NativeAddon;
  return directAddon;
}

function runDirect<T>(type: DecoderWorkerRequest["type"], payload: any): T {
  const addon = getDirectAddon();
  if (type === "loadFile") {
    return addon.loadFile(payload.filePath) as T;
  }
  if (type === "decodeFrame") {
    const data = new Uint8Array(payload.width * payload.height * 4);
    const res = addon.decodeFrameInto(
      payload.filePath,
      data,
      payload.timestamp,
      payload.keyframeOnly,
    );
    return { timestamp: res.timestamp, data } as T;
  }
  if (type === "decodeNextFrame") {
    const data = new Uint8Array(payload.width * payload.height * 4);
    const start = payload.startTime >= 0 ? payload.startTime : undefined;
    const end = payload.endTime >= 0 ? payload.endTime : undefined;
    const res = addon.decodeNextFrame(payload.filePath, data, start, end);
    if (!res) return null as T;
    return { timestamp: res.timestamp, data } as T;
  }
  throw new Error(`Unknown decoder op: ${type}`);
}

async function callDecoder<T>(type: DecoderWorkerRequest["type"], payload: object): Promise<T> {
  if (!workerEnabled) return runDirect<T>(type, payload);
  try {
    return await postToDecodeWorker<T>(type, payload);
  } catch (error) {
    workerEnabled = false;
    if (decodeWorker) {
      try {
        decodeWorker.terminate();
      } catch {
        // ignore
      }
      decodeWorker = null;
    }
    console.warn(
      "[native-decoder] worker failed; falling back to direct addon decode:",
      error instanceof Error ? error.message : String(error),
    );
    return runDirect<T>(type, payload);
  }
}

export type NativeDecoderFileInfo = {
  format: string;
  duration: number;
  bitrate: number;
  nb_streams: number;
  hw_accelerated: boolean;
  decode_backend: "videotoolbox_direct" | "ffmpeg_hwaccel" | "ffmpeg_software";
  video?: {
    width: number;
    height: number;
    codec: string;
    pixel_format: string;
    fps: number;
    stream_index: number;
  };
  audio?: {
    codec: string;
    sample_rate: number;
    channels: number;
    stream_index: number;
  };
};

/**
 * Load a video file and return its metadata. Decoder state is cached natively by file path.
 */
export async function nativeDecoderLoadFile(filePath: string): Promise<NativeDecoderFileInfo> {
  const result = await callDecoder<NativeDecoderFileInfo>("loadFile", { filePath });
  if (!result?.video) throw new Error("No video stream found");
  return result;
}

/**
 * Decode a single frame at timestamp seconds.
 */
export async function nativeDecoderDecodeFrame(
  filePath: string,
  width: number,
  height: number,
  timestamp: number,
  keyframeOnly = false,
): Promise<{ timestamp: number; data: Uint8Array }> {
  const result = await callDecoder<{ timestamp: number; data: Uint8Array }>("decodeFrame", {
    filePath,
    width,
    height,
    timestamp,
    keyframeOnly,
  });
  const data = result.data instanceof Uint8Array ? result.data : new Uint8Array(result.data);
  return { timestamp: result.timestamp, data };
}

/**
 * Decode the next sequential frame (or seek when startTime >= 0).
 */
export async function nativeDecoderDecodeNextFrame(
  filePath: string,
  width: number,
  height: number,
  startTime = -1,
  endTime = -1,
): Promise<{ timestamp: number; data: Uint8Array } | null> {
  const result = await callDecoder<{ timestamp: number; data: Uint8Array } | null>(
    "decodeNextFrame",
    {
      filePath,
      width,
      height,
      startTime,
      endTime,
    },
  );
  if (!result) return null;
  const data = result.data instanceof Uint8Array ? result.data : new Uint8Array(result.data);
  return { timestamp: result.timestamp, data };
}
