import { ipcRenderer, webUtils } from "electron";
import { promises as fsp } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import fs from "node:fs";

/**
 * Thin helpers around IPC and filesystem that are used broadly across preload.
 */

function send(channel: string, message: string) {
  return ipcRenderer.invoke(channel, message);
}

function resolvePath(inputPath: string): string {
  if (!inputPath) return "";
  if (inputPath.startsWith("~/") || inputPath === "~") {
    return join(homedir(), inputPath.slice(1));
  }
  return inputPath;
}

// Safely delete a file from disk. Accepts absolute paths or file:// URLs.
async function deleteFile(pathOrUrl: string): Promise<void> {
  try {
    let p = String(pathOrUrl || "");
    if (!p) return;
    if (p.startsWith("file://")) {
      try {
        p = fileURLToPath(p);
      } catch {
        // fall through with original
      }
    }
    await fsp.rm(p, { force: true });
  } catch {
    // swallow
  }
}

// Generic binary file helpers used by the renderer (supports app://, http(s) and file://)
const readFileBuffer = async (path: string) => {
  const original = path;
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
    const buffer = await fsp.readFile(path);
    return buffer;
  } catch (err) {
    // If local read failed, attempt to fetch via app://apex-cache assuming 'original' may be a remote absolute path
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

const readFileStream = async (path: string) => {
  const stream = await fs.createReadStream(fileURLToPath(path));
  return stream;
};

function pathToFileURLString(path: string): string {
  const fileUrl = pathToFileURL(path);
  return fileUrl.href;
}

function getPathForFile(file: File): string {
  return webUtils.getPathForFile(file);
}

export {
  send,
  resolvePath,
  deleteFile,
  readFileBuffer,
  readFileStream,
  pathToFileURLString,
  getPathForFile,
};


