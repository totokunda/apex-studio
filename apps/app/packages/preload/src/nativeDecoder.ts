/**
 * Native decoder interface backed by node:worker_threads.
 *
 * Renderer now owns the Web Worker transport. Preload keeps a pure
 * worker_threads transport for any preload/main-world callers.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

type DecoderWorkerRequest =
  | { id: number; type: "loadFile"; payload: { filePath: string; decoderKey?: string } }
  | {
      id: number;
      type: "decodeFrame";
      payload: {
        filePath: string;
        width: number;
        height: number;
        timestamp: number;
        keyframeOnly: boolean;
        decoderKey?: string;
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
        decoderKey?: string;
      };
    };

type DecoderWorkerResponse =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };

type Pending = {
  resolve: (value: any) => void;
  reject: (reason?: unknown) => void;
};

class DecoderWorkerOperationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DecoderWorkerOperationError";
  }
}

let decodeWorker: Worker | null = null;
let nextRequestId = 1;
const pending = new Map<number, Pending>();

function rejectAllPending(error: Error): void {
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
    entry.reject(new DecoderWorkerOperationError(msg.error || "Native decoder worker error"));
  });

  worker.on("error", (error) => {
    decodeWorker = null;
    rejectAllPending(error);
  });

  worker.on("exit", (code) => {
    decodeWorker = null;
    if (code === 0) return;
    rejectAllPending(new Error(`Native decoder worker exited with code ${code}`));
  });

  decodeWorker = worker;
  return worker;
}

function terminateDecodeWorker(): void {
  const worker = decodeWorker;
  decodeWorker = null;
  if (!worker) return;
  void worker.terminate();
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

async function callDecoder<T>(type: DecoderWorkerRequest["type"], payload: object): Promise<T> {
  try {
    return await postToDecodeWorker<T>(type, payload);
  } catch (error) {
    if (error instanceof DecoderWorkerOperationError) {
      throw error;
    }
    terminateDecodeWorker();
    throw error;
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

export async function nativeDecoderLoadFile(
  filePath: string,
  decoderKey?: string,
): Promise<NativeDecoderFileInfo> {
  const result = await callDecoder<NativeDecoderFileInfo>("loadFile", { filePath, decoderKey });
  if (!result?.video) throw new Error("No video stream found");
  return result;
}

export async function nativeDecoderDecodeFrame(
  filePath: string,
  width: number,
  height: number,
  timestamp: number,
  keyframeOnly = false,
  decoderKey?: string,
): Promise<{ timestamp: number; data: Uint8Array }> {
  const result = await callDecoder<{ timestamp: number; data: Uint8Array }>("decodeFrame", {
    filePath,
    width,
    height,
    timestamp,
    keyframeOnly,
    decoderKey,
  });
  const data = result.data instanceof Uint8Array ? result.data : new Uint8Array(result.data);
  return { timestamp: result.timestamp, data };
}

export async function nativeDecoderDecodeNextFrame(
  filePath: string,
  width: number,
  height: number,
  startTime = -1,
  endTime = -1,
  decoderKey?: string,
): Promise<{ timestamp: number; data: Uint8Array } | null> {
  const result = await callDecoder<{ timestamp: number; data: Uint8Array } | null>(
    "decodeNextFrame",
    {
      filePath,
      width,
      height,
      startTime,
      endTime,
      decoderKey,
    },
  );
  if (!result) return null;
  const data = result.data instanceof Uint8Array ? result.data : new Uint8Array(result.data);
  return { timestamp: result.timestamp, data };
}
