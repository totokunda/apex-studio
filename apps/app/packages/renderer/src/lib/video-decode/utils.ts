import { StreamSource } from "mediabunny";
import { fileURLToPath } from "url";
const nodeFs = require('node:fs/promises');

export function fileURLToPathInWorker(raw: string): string {
    try {
      const u = new URL(raw);
  
      // For file:// or app:// URLs, convert URL path to a local filesystem path.
      if (u.protocol === "file:" || u.protocol === "app:") {
        const decoded = decodeURIComponent(u.pathname);
        // Windows file URL path: /C:/path -> C:/path
        if (/^\/[A-Za-z]:\//.test(decoded)) {
          return decoded.slice(1);
        }
        return decoded;
      }
  
      // For unsupported URL schemes, fall back to input string.
      return raw;
    } catch {
      // Not a URL – assume it's already a local filesystem path.
      return raw;
    }
  }




export function createNodeFileSource(filePath: string): StreamSource {
    let fileHandle: any = null;
    let knownSize: number | null = null;
  
    return new StreamSource({
      getSize: async () => {
        if (!fileHandle) {
          fileHandle = await nodeFs.open(filePath, "r");
        }
        if (knownSize == null) {
          const stats = await fileHandle.stat();
          knownSize = Number(stats.size);
        }
        return knownSize;
      },
      read: async (start, end) => {
        if (!fileHandle) {
          fileHandle = await nodeFs.open(filePath, "r");
        }
        const length = Math.max(0, end - start);
        const buffer = new Uint8Array(length);
        if (length === 0) return buffer;
        const readResult = await fileHandle.read(buffer, 0, length, start);
        if (readResult.bytesRead === length) return buffer;
        return buffer.subarray(0, readResult.bytesRead);
      },
      dispose: async () => {
        if (!fileHandle) return;
        try {
          await fileHandle.close();
        } catch {
          // ignore close errors in worker teardown
        } finally {
          fileHandle = null;
        }
      },
      prefetchProfile: "fileSystem",
    });
  }
  

  // Generic binary file helpers used by the renderer (supports app://, http(s) and file://)
export const readFileBuffer = async (path: string) => {
  // Handle app:// scheme directly (served by main via AppDirProtocol)
  if (typeof path === "string" && path.startsWith("app://")) {
    const res = await fetch(path);
    if (!res.ok)
      throw new Error(
        `Failed to fetch ${path}: ${res.status} ${res.statusText}`,
      );
    const ab = await res.arrayBuffer();
    return Buffer.from(ab);
  }
  // Remote HTTP(S)
  if (typeof path === "string" && /^https?:\/\//i.test(path)) {
    const res = await fetch(path);
    if (!res.ok)
      throw new Error(
        `Failed to fetch ${path}: ${res.status} ${res.statusText}`,
      );
    const ab = await res.arrayBuffer();
    return Buffer.from(ab);
  }
  // file:// URL → local fs
  if (typeof path === "string" && path.startsWith("file://")) {
    try {
      path = fileURLToPath(path);
    } catch {
      // fall through with original
    }
  }

  try {
    const buffer = await nodeFs.readFile(path);
    return buffer;
  } catch (err) {
    // If local read failed, attempt to fetch via app://apex-cache assuming the input may be a remote absolute path
    try {
      const encodedPath = (() => {
        const p = path.startsWith("/") ? path : `/${path}`;
        return encodeURI(p);
      })();
      const appUrl = new URL(`app://apex-cache${encodedPath}`);
      const res = await fetch(appUrl);
      if (!res.ok)
        throw new Error(
          `Failed to fetch ${appUrl}: ${res.status} ${res.statusText}`,
        );
      const ab = await res.arrayBuffer();
      return Buffer.from(ab);
    } catch (e) {
      throw err instanceof Error
        ? err
        : new Error("readFileBuffer: failed to read file");
    }
  }
};