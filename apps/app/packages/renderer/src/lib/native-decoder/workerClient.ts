import path from "node:path";
import type { FileInfo } from "./types";

type DecoderWorkerRequest =
  | {
      id: number;
      type: "loadFile";
      payload: {
        filePath: string;
        decoderKey?: string;
        addonPathCandidates: string[];
      };
    }
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
        addonPathCandidates: string[];
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
        addonPathCandidates: string[];
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

function fileUrlToPath(url: string): string {
  const parsed = new URL(url);
  const decoded = decodeURIComponent(parsed.pathname);
  if (/^\/[A-Za-z]:/.test(decoded)) return decoded.slice(1);
  return decoded;
}

function getAddonPathCandidates(): string[] {
  const candidates = new Set<string>();
  const cwd = typeof process?.cwd === "function" ? process.cwd() : "";
  const resourcesPath = typeof process?.resourcesPath === "string" ? process.resourcesPath : "";
  const moduleDir = path.dirname(fileUrlToPath(import.meta.url));

  const add = (p: string) => {
    if (!p) return;
    candidates.add(path.normalize(p));
  };

  add(path.join(cwd, "packages", "native-decoder", "build", "Release", "addon.node"));
  add(path.join(cwd, "native-decoder", "build", "Release", "addon.node"));
  add(path.resolve(moduleDir, "../../../native-decoder/build/Release/addon.node"));
  add(path.resolve(moduleDir, "../../../../native-decoder/build/Release/addon.node"));
  add(path.resolve(moduleDir, "../../native-decoder/build/Release/addon.node"));
  add(path.join(resourcesPath, "app.asar.unpacked", "packages", "native-decoder", "build", "Release", "addon.node"));
  add(path.join(resourcesPath, "app.asar", "packages", "native-decoder", "build", "Release", "addon.node"));

  return Array.from(candidates);
}

function getWorkerUrl(): string {
  return new URL("./nativeDecoder.worker.cjs", import.meta.url).toString();
}

function ensureDecodeWorker(): Worker {
  if (decodeWorker) return decodeWorker;

  const worker = new Worker(getWorkerUrl());

  worker.addEventListener("message", (event: MessageEvent<DecoderWorkerResponse>) => {
    const msg = event.data;
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.ok) {
      entry.resolve(msg.result);
      return;
    }
    entry.reject(new DecoderWorkerOperationError(msg.error || "Native decoder worker error"));
  });

  worker.addEventListener("error", (event: ErrorEvent) => {
    decodeWorker = null;
    rejectAllPending(new Error(event.message || "Native decoder worker transport error"));
  });

  decodeWorker = worker;
  return worker;
}

function terminateDecodeWorker(): void {
  const worker = decodeWorker;
  decodeWorker = null;
  if (!worker) return;
  worker.terminate();
}

function postToDecodeWorker<T>(
  type: DecoderWorkerRequest["type"],
  payload: Record<string, unknown>,
): Promise<T> {
  const worker = ensureDecodeWorker();
  const id = nextRequestId++;
  const request = {
    id,
    type,
    payload: {
      ...(payload as object),
      addonPathCandidates: getAddonPathCandidates(),
    },
  } as DecoderWorkerRequest;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage(request);
  });
}

async function callDecoder<T>(
  type: DecoderWorkerRequest["type"],
  payload: Record<string, unknown>,
): Promise<T> {
  try {
    const result = await postToDecodeWorker<T>(type, payload);
    return result;
  } catch (error) {
    if (error instanceof DecoderWorkerOperationError) {
      throw error;
    }
    terminateDecodeWorker();
    throw error;
  }
}

export async function nativeDecoderLoadFile(
  filePath: string,
  decoderKey?: string,
): Promise<FileInfo> {
  const result = await callDecoder<FileInfo>("loadFile", { filePath, decoderKey });
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
