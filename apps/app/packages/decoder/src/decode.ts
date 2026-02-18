/**
 * Decoder client — loads media files via the demux addon and returns
 * WebCodecs-ready streams and packets.
 */

import path from "node:path";

function urlToModuleDir(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "file:") {
      const decoded = decodeURIComponent(parsed.pathname);
      const p = /^\/[A-Za-z]:/.test(decoded) ? decoded.slice(1) : decoded;
      return path.dirname(p);
    }
  } catch {
    /* ignore */
  }
  return null;
}

function toFilesystemPath(input: string): string {
  if (!input || typeof input !== "string") return input;
  const s = input.trim();
  if (!s) return input;
  try {
    if (s.startsWith("file://")) {
      const parsed = new URL(s);
      const decoded = decodeURIComponent(parsed.pathname);
      return /^\/[A-Za-z]:/.test(decoded) ? decoded.slice(1) : decoded;
    }
    if (s.startsWith("app://user-data/") || s.startsWith("app://apex-cache/")) {
      const parsed = new URL(s);
      const decoded = decodeURIComponent(parsed.pathname);
      return /^\/[A-Za-z]:/.test(decoded) ? decoded.slice(1) : decoded;
    }
  } catch {
    /* ignore */
  }
  return s;
}

export interface DemuxStream {
  index: number;
  codecType: "video" | "audio" | "unknown";
  timeBaseNum: number;
  timeBaseDen: number;
  duration: number;
  durationMicros: number;
  codec: string;
  codedWidth?: number;
  codedHeight?: number;
  videoDecoderConfig?: {
    codec: string;
    codedWidth: number;
    codedHeight: number;
    description?: ArrayBuffer;
  };
  sampleRate?: number;
  channelCount?: number;
  description?: ArrayBuffer;
}

export interface DemuxPacket {
  streamIndex: number;
  isKeyFrame: boolean;
  timestampMicros: number;
  pts: number;
  dts: number;
  data: ArrayBuffer | Uint8Array;
}

export interface DemuxResult {
  streams: DemuxStream[];
  packets?: DemuxPacket[];
  videoStreamIndex?: number;
  videoDecoderConfig?: VideoDecoderConfig;
  videoPacketCount?: number;
  duration: number;
  durationMicros?: number;
}

export type LoadFileResult = DemuxResult;

export interface DecodeFrameResult {
  frame: VideoFrame;
  timestamp: number;
  duration: number;
}

export interface DecodedFrame {
  frame: VideoFrame;
  timestamp: number; // seconds
  duration: number; // seconds
}

type WorkerRequest =
  | { id: number; type: "loadFile"; payload: { filePath: string; addonPathCandidates: string[] } }
  | { id: number; type: "decodeFrame"; payload: { videoPacketIndex: number } }
  | { id: number; type: "seek"; payload: { timestampMicros: number; forceAccurate: boolean } }
  | { id: number; type: "decodeNext"; payload: { endTimestampMicros?: number } };

type WorkerResponse =
  | { id: number; ok: true; result: DemuxResult | DecodeFrameResult | null }
  | { id: number; ok: false; error: string };

type Pending = {
  resolve: (value: DemuxResult | DecodeFrameResult | null) => void;
  reject: (reason?: unknown) => void;
};

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function getDemuxAddonPathCandidates(): string[] {
  const cwd = typeof process?.cwd === "function" ? process.cwd() : "";
  const proc = process as NodeJS.Process & { resourcesPath?: string };
  const resourcesPath = typeof proc?.resourcesPath === "string" ? proc.resourcesPath : "";
  const moduleDir = urlToModuleDir(import.meta.url);

  const candidates: string[] = [
    path.join(cwd, "packages", "decoder", "build", "Release", "demux.node"),
    path.join(cwd, "decoder", "build", "Release", "demux.node"),
    ...(moduleDir
      ? [
          path.resolve(moduleDir, "../build/Release/demux.node"),
          path.resolve(moduleDir, "../../build/Release/demux.node"),
        ]
      : []),
    ...(resourcesPath
      ? [
          path.join(resourcesPath, "app.asar.unpacked", "packages", "decoder", "build", "Release", "demux.node"),
          path.join(resourcesPath, "app.asar", "packages", "decoder", "build", "Release", "demux.node"),
        ]
      : []),
  ].filter(Boolean);

  return [...new Set(candidates)];
}

function getWorkerUrl(): string {
  return new URL("./decode.worker.cjs", import.meta.url).toString();
}

function secondsToMicros(seconds: number): number {
  if (!Number.isFinite(seconds)) return -1;
  return Math.max(0, Math.floor(seconds * 1_000_000));
}

function microsToSeconds(micros: number): number {
  if (!Number.isFinite(micros)) return 0;
  return micros / 1_000_000;
}

function toDecodedFrame(result: DecodeFrameResult | null): DecodedFrame | null {
  if (!result) return null;
  return {
    frame: result.frame,
    timestamp: microsToSeconds(result.timestamp),
    duration: microsToSeconds(result.duration),
  };
}

function ensureWorker(): Worker {
  if (worker) return worker;

  const w = new Worker(getWorkerUrl());

  w.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
    const msg = event.data;
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.ok) {
      entry.resolve(msg.result);
    } else {
      entry.reject(new Error(msg.error || "Demux worker error"));
    }
  });

  w.addEventListener("error", (event: ErrorEvent) => {
    worker = null;
    for (const { reject } of pending.values()) {
      reject(new Error(event.message || "Demux worker error"));
    }
    pending.clear();
  });

  worker = w;
  return worker;
}

function postToWorker<T>(
  type: "loadFile" | "decodeFrame" | "seek" | "decodeNext",
  payload: Record<string, unknown>,
): Promise<T> {
  const w = ensureWorker();
  const id = nextId++;
  const request = {
    id,
    type,
    payload: type === "loadFile"
      ? { ...payload, addonPathCandidates: getDemuxAddonPathCandidates() }
      : payload,
  } as WorkerRequest;
  
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    w.postMessage(request);
  });
}

/**
 * Load a media file, demux it, and set up the WebCodecs VideoDecoder in the worker.
 * Returns stream metadata and config (packets stay in worker for on-demand decode).
 */
export async function loadFile(filePathOrUrl: string): Promise<LoadFileResult> {
  const fsPath = toFilesystemPath(filePathOrUrl);
  return postToWorker<LoadFileResult>("loadFile", { filePath: fsPath });
}

/**
 * Decode a single video frame by packet index. Call loadFile first.
 * @param videoPacketIndex Index into the video packet array (0..videoPacketCount-1).
 */
export async function decodeFrame(videoPacketIndex: number): Promise<DecodeFrameResult | null> {
  return postToWorker<DecodeFrameResult | null>("decodeFrame", { videoPacketIndex });
}

/**
 * Seek to timestamp and return the closest decoded frame at/after the nearest previous key packet.
 */
export async function seek(timestampSeconds: number, forceAccurate = false): Promise<DecodedFrame | null> {
  const raw = await postToWorker<DecodeFrameResult | null>("seek", {
    timestampMicros: secondsToMicros(timestampSeconds),
    forceAccurate,
  });
  return toDecodedFrame(raw);
}

/**
 * Decode the next frame from the current cursor.
 */
export async function decodeNext(endTimestampSeconds?: number): Promise<DecodedFrame | null> {
  const raw = await postToWorker<DecodeFrameResult | null>("decodeNext", {
    endTimestampMicros:
      typeof endTimestampSeconds === "number" && Number.isFinite(endTimestampSeconds)
        ? secondsToMicros(endTimestampSeconds)
        : undefined,
  });
  return toDecodedFrame(raw);
}

/**
 * Iterate decoded frames in [startTimeSeconds, endTimeSeconds].
 */
export async function iterate(
  startTimeSeconds: number,
  endTimeSeconds: number,
  onFrame: (frame: DecodedFrame) => void | Promise<void>,
  checkCancel?: () => boolean,
): Promise<void> {
  const cancelled = () => (typeof checkCancel === "function" ? !checkCancel() : false);
  if (cancelled()) return;

  const first = await seek(startTimeSeconds, true);
  if (cancelled()) {
    if (first?.frame) first.frame.close();
    return;
  }
  if (first) {
    if (first.timestamp <= endTimeSeconds + 1e-4) {
      await onFrame(first);
    } else {
      first.frame.close();
      return;
    }
  }

  while (true) {
    if (cancelled()) return;
    const next = await decodeNext(endTimeSeconds);
    if (!next) return;
    if (next.timestamp > endTimeSeconds + 1e-4) {
      next.frame.close();
      return;
    }
    await onFrame(next);
  }
}

export function terminateWorker(): void {
  if (worker) {
    worker.terminate();
    worker = null;
  }
  pending.clear();
}
