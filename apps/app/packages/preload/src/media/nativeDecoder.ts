/**
 * Native video decoder addon bridge.
 *
 * Loads the @app/native-decoder N-API addon and exposes typed wrappers
 * that the renderer can call via the preload context bridge.
 */

import { existsSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { ipcRenderer } from "electron";

const require = createRequire(import.meta.url);

// ------------------------------------------------------------------
// Addon binary resolution (mirrors ffmpegBin.ts pattern)
// ------------------------------------------------------------------

interface NativeDecoderAddon {
  createDecoder(): number;
  configure(handle: number, opts: {
    filePath: string;
    width: number;
    height: number;
    decodeOnly?: boolean;
    outputFormat?: "rgba" | "nv12";
    bufferPool: Array<(ArrayBufferLike | Uint8Array)>;
    onFrame: (bufferIndex: number, width: number, height: number,
              timestamp: number, duration: number, requestId: number) => void;
    onError: (message: string) => void;
    onReady: () => void;
  }): void;
  seek(handle: number, timestamp: number, forceAccurate: boolean, requestId: number): Promise<void>;
  iterate(handle: number, startTime: number, endTime: number, requestId: number): Promise<void>;
  ackFrame(handle: number, bufferIndex: number): void;
  cancelCurrent(handle: number): void;
  dispose(handle: number): void;
  disposeAll(): void;
  getCapabilities(): { hwAccelMethods: string[]; preferredMethod: string };
}

let addon: NativeDecoderAddon | null = null;
const NATIVE_BUFFER_POOL_SIZE = 4;
const decoderBufferPools = new Map<number, {
  buffers: ArrayBufferLike[];
  views: Uint8Array[];
}>();

function resolveAddonPath(): string | null {
  const nodeName = process.platform === "win32"
    ? "native_decoder.node"
    : "native_decoder.node";

  const moduleDir = dirname(fileURLToPath(import.meta.url));
  const devWorkspaceCandidate = resolve(
    moduleDir,
    "..",
    "..",
    "native-decoder",
    "build",
    "Release",
    nodeName,
  );

  const candidates = [
    // Development: workspace build (prefer this to avoid stale packaged artifacts)
    devWorkspaceCandidate,
    // Production: bundled in resources
    join(process.resourcesPath ?? "", "native-decoder", nodeName),
    // Production: extraResources unpacked from asar
    join(process.resourcesPath ?? "", "app.asar.unpacked",
         "node_modules", "@app", "native-decoder", "build", "Release", nodeName),
  ];

  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  return null;
}

function loadAddon(): NativeDecoderAddon {
  if (addon) return addon;

  const addonPath = resolveAddonPath();
  if (!addonPath) {
    throw new Error(
      "native-decoder addon not found. Ensure the native-decoder package is built."
    );
  }

  addon = require(addonPath) as NativeDecoderAddon;
  return addon;
}

// ------------------------------------------------------------------
// Public API (exposed via contextBridge)
// ------------------------------------------------------------------

export function createNativeDecoder(): number {
  const loadedAddon = loadAddon();
  return loadedAddon.createDecoder();
}

export function configureNativeDecoder(
  handle: number,
  filePath: string,
  width: number,
  height: number,
  onFrame: (frameBytes: Uint8Array, width: number, height: number,
            timestamp: number, duration: number, requestId: number, bufferIndex?: number,
            pixelFormat?: "rgba" | "nv12") => void,
  onError: (message: string) => void,
  onReady: () => void,
  opts?: {
    decodeOnly?: boolean;
    outputFormat?: "rgba" | "nv12";
    copyFrameData?: boolean;
    bufferPoolSize?: number;
    manualAck?: boolean;
    preferSharedBufferPool?: boolean;
  },
): void {
  const isCloneFailure = (error: unknown): boolean => {
    const message = error instanceof Error ? error.message : String(error);
    const lowered = message.toLowerCase();
    return (
      lowered.includes("could not be cloned") ||
      lowered.includes("datacloneerror") ||
      lowered.includes("structured clone")
    );
  };

  const outputFormat = opts?.outputFormat ?? "rgba";
  const frameBytes = outputFormat === "nv12"
    ? Math.max(1, Math.floor(width * height * 3 / 2))
    : Math.max(1, width * height * 4);
  const manualAck = opts?.manualAck === true && opts?.copyFrameData === false;
  const poolSize = Math.max(2, Math.min(32, Math.floor(opts?.bufferPoolSize ?? NATIVE_BUFFER_POOL_SIZE)));
  // SharedArrayBuffer-backed frame payloads are not cloneable on some
  // Electron bridge configurations; keep this opt-in only.
  const shouldUseSharedPool =
    opts?.preferSharedBufferPool === true &&
    typeof SharedArrayBuffer !== "undefined";
  const pool = Array.from({ length: poolSize }, () => {
    if (shouldUseSharedPool) {
      return new SharedArrayBuffer(frameBytes);
    }
    return new ArrayBuffer(frameBytes);
  });
  const views = pool.map((buffer) => new Uint8Array(buffer));
  decoderBufferPools.set(handle, { buffers: pool, views });
  let forceCopyFallback = opts?.copyFrameData !== false;

  const native = loadAddon();
  try {
    native.configure(handle, {
      filePath,
      width,
      height,
      decodeOnly: opts?.decodeOnly ?? false,
      outputFormat,
      bufferPool: views,
      onFrame: (bufferIndex, frameWidth, frameHeight, timestamp, duration, requestId) => {
        if (bufferIndex < 0) {
          // decodeOnly mode: native did not materialize pixel bytes.
          try {
            onFrame(new Uint8Array(0), frameWidth, frameHeight, timestamp, duration, requestId);
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            onError(`Renderer frame callback failed: ${message}`);
          }
          return;
        }

        const state = decoderBufferPools.get(handle);
        const source = state?.views[bufferIndex];
        if (!source) {
          native.ackFrame(handle, bufferIndex);
          onError(`Invalid native frame buffer index: ${bufferIndex}`);
          return;
        }

        try {
          if (!forceCopyFallback && opts?.copyFrameData === false) {
            // Zero-copy mode: renderer must consume frame synchronously before return.
            onFrame(
              source,
              frameWidth,
              frameHeight,
              timestamp,
              duration,
              requestId,
              bufferIndex,
              outputFormat,
            );
          } else {
            // Safe default for async consumers: copy before acknowledging native buffer slot.
            const copied = new Uint8Array(source.byteLength);
            copied.set(source);
            onFrame(
              copied,
              frameWidth,
              frameHeight,
              timestamp,
              duration,
              requestId,
              bufferIndex,
              outputFormat,
            );
          }
        } catch (error) {
          // Bridge clone errors can happen on some Electron/runtime combos when
          // sending pooled frame views directly. Fall back to copied ArrayBuffer payloads.
          if (!forceCopyFallback && opts?.copyFrameData === false && isCloneFailure(error)) {
            forceCopyFallback = true;
            try {
              const copied = new Uint8Array(source.byteLength);
              copied.set(source);
              onFrame(
                copied,
                frameWidth,
                frameHeight,
                timestamp,
                duration,
                requestId,
                bufferIndex,
                outputFormat,
              );
            } catch (fallbackError) {
              const message =
                fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
              onError(`Renderer frame callback failed: ${message}`);
              native.ackFrame(handle, bufferIndex);
              return;
            }
          } else {
            const message = error instanceof Error ? error.message : String(error);
            onError(`Renderer frame callback failed: ${message}`);
            native.ackFrame(handle, bufferIndex);
            return;
          }
        }

        if (!manualAck) {
          native.ackFrame(handle, bufferIndex);
        }
      },
      onError,
      onReady,
    });
  } catch (error) {
    decoderBufferPools.delete(handle);
    throw error;
  }
}

export function seekNativeDecoder(
  handle: number,
  timestamp: number,
  forceAccurate: boolean,
  requestId: number,
): Promise<void> {
  return loadAddon().seek(handle, timestamp, forceAccurate, requestId);
}

export function iterateNativeDecoder(
  handle: number,
  startTime: number,
  endTime: number,
  requestId: number,
): Promise<void> {
  return loadAddon().iterate(handle, startTime, endTime, requestId);
}

export function cancelNativeDecoder(handle: number): void {
  loadAddon().cancelCurrent(handle);
}

export function disposeNativeDecoder(handle: number): void {
  decoderBufferPools.delete(handle);
  loadAddon().dispose(handle);
}

export function disposeAllNativeDecoders(): void {
  decoderBufferPools.clear();
  loadAddon().disposeAll();
}

export function ackNativeFrame(handle: number, bufferIndex: number): void {
  loadAddon().ackFrame(handle, bufferIndex);
}

export function getNativeDecoderCapabilities(): {
  hwAccelMethods: string[];
  preferredMethod: string;
} {
  return loadAddon().getCapabilities();
}

/**
 * Returns true if the native decoder addon is available on this platform.
 */
export function isNativeDecoderAvailable(): boolean {
  try {
    return resolveAddonPath() !== null;
  } catch {
    return false;
  }
}

/**
 * Resolve an asset URL (file://, app://, or absolute path) to an absolute
 * filesystem path that the native C++ decoder can open via avformat_open_input.
 */
export async function resolveNativeDecoderPath(urlOrPath: string): Promise<string> {
  // file:// URL -> absolute filesystem path
  if (urlOrPath.startsWith("file://")) {
    return fileURLToPath(urlOrPath);
  }

  // app:// custom protocol → resolve via main process IPC
  if (urlOrPath.startsWith("app://")) {
    const resolved = await ipcRenderer.invoke("appdir:resolve-path", urlOrPath);
    if (typeof resolved === "string" && resolved.length > 0) return resolved;
    // If resolution failed, fall through to return the raw value
  }

  // Already an absolute path or unrecognised scheme — return as-is
  return urlOrPath;
}
