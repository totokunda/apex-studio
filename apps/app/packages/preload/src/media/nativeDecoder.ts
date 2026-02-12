/**
 * Native video decoder addon bridge.
 *
 * Loads the @app/native-decoder N-API addon and exposes typed wrappers
 * that the renderer can call via the preload context bridge.
 */

import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
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
    bufferPool: SharedArrayBuffer[];
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

function resolveAddonPath(): string | null {
  const nodeName = process.platform === "win32"
    ? "native_decoder.node"
    : "native_decoder.node";

  const candidates = [
    // Production: bundled in resources
    join(process.resourcesPath ?? "", "native-decoder", nodeName),
    // Production: extraResources unpacked from asar
    join(process.resourcesPath ?? "", "app.asar.unpacked",
         "node_modules", "@app", "native-decoder", "build", "Release", nodeName),
    // Development: locally built
    join(dirname(fileURLToPath(import.meta.url)),
         "..", "..", "..", "native-decoder", "build", "Release", nodeName),
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
  return loadAddon().createDecoder();
}

export function configureNativeDecoder(
  handle: number,
  opts: {
    filePath: string;
    width: number;
    height: number;
    bufferPool: SharedArrayBuffer[];
    onFrame: (bufferIndex: number, width: number, height: number,
              timestamp: number, duration: number, requestId: number) => void;
    onError: (message: string) => void;
    onReady: () => void;
  },
): void {
  loadAddon().configure(handle, opts);
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
  loadAddon().dispose(handle);
}

export function disposeAllNativeDecoders(): void {
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
  // file:// URL → absolute filesystem path
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
