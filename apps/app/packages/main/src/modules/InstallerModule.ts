import type { AppModule } from "../AppModule.js";
import { ipcMain } from "electron";
import fs from "node:fs";
import { promises as fsp } from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createZstdDecompress } from "node:zlib";
import * as tar from "tar-stream";

type ConfigResponse<T> = { success: true; data: T } | { success: false; error: string };

type BundleSource =
  | { kind: "local"; path: string }
  | { kind: "remote"; url: string; assetName?: string };

type ExtractBundleRequest = {
  source: BundleSource;
  destinationDir: string;
  jobId?: string;
};

type InstallerProgressEvent =
  | {
      phase: "download";
      downloadedBytes: number;
      totalBytes?: number;
      percent?: number;
      message?: string;
    }
  | {
      phase: "extract";
      processedBytes?: number;
      totalBytes?: number;
      percent?: number;
      entryName?: string;
      entriesExtracted?: number;
      message?: string;
    }
  | {
      phase: "status";
      message: string;
    };

function getUserFfmpegInstallDir(): string {
  // Stable, user-writable location. Preload will also check this path for discovery.
  return path.join(os.homedir(), ".apex-studio", "ffmpeg");
}

function exeName(cmd: "ffmpeg" | "ffprobe") {
  return process.platform === "win32" ? `${cmd}.exe` : cmd;
}

async function runTarExtractZst(opts: {
  archivePath: string;
  destinationDir: string;
}): Promise<void> {
  const archivePath = opts.archivePath;
  const destinationDir = opts.destinationDir;

  // tar flag support varies between bsdtar/gnu tar; try a few.
  const attempts: Array<{ args: string[]; label: string }> = [
    { label: "tar --zstd", args: ["--zstd", "-xf", archivePath, "-C", destinationDir] },
    {
      label: "tar --use-compress-program zstd",
      args: ["--use-compress-program", "zstd", "-xf", archivePath, "-C", destinationDir],
    },
    { label: "tar -I zstd", args: ["-I", "zstd", "-xf", archivePath, "-C", destinationDir] },
  ];

  let lastErr: string | null = null;
  for (const a of attempts) {
    const res = await new Promise<{ ok: boolean; stderr: string; error?: NodeJS.ErrnoException }>((resolve) => {
      const p = spawn("tar", a.args, { stdio: ["ignore", "ignore", "pipe"] });
      let stderr = "";
      p.stderr?.on("data", (d) => (stderr += String(d)));
      p.on("error", (err) => resolve({ ok: false, stderr, error: err }));
      p.on("close", (code) => resolve({ ok: code === 0, stderr }));
    });

    if (res.ok) return;
    if (res.error && (res.error as any)?.code === "ENOENT") {
      const e = new Error(`'tar' not found on PATH (needed to extract .tar.zst bundles)`);
      (e as any).code = "ENOENT";
      throw e;
    }
    lastErr = `${a.label} failed${res.stderr ? `: ${res.stderr.trim()}` : ""}`;
  }

  throw new Error(lastErr || "Failed to extract bundle");
}

function safeJoinWithinRoot(root: string, relative: string): string {
  // Normalize and ensure no traversal outside root.
  const rel = relative.replace(/^[\\/]+/, "");
  const out = path.resolve(root, rel);
  const rootResolved = path.resolve(root);
  if (out === rootResolved) return out;
  if (!out.startsWith(rootResolved + path.sep)) {
    throw new Error(`Refusing to write outside destination directory: ${relative}`);
  }
  return out;
}

async function extractTarZstWithNode(opts: {
  archivePath: string;
  destinationDir: string;
  onProgress?: (ev: InstallerProgressEvent) => void;
}): Promise<void> {
  const extract = tar.extract();
  const inflator = createZstdDecompress();

  await fsp.mkdir(opts.destinationDir, { recursive: true });

  const pending: Promise<void>[] = [];
  let entriesExtracted = 0;

  extract.on("entry", (header: any, stream: any, next: any) => {
    const done = (p: Promise<void>) => {
      pending.push(
        p.finally(() => {
          stream.resume();
          next();
        }),
      );
    };

    try {
      const name = String(header?.name || "");
      const type = String(header?.type || "file");
      const mode = typeof header?.mode === "number" ? header.mode : undefined;
      const absPath = safeJoinWithinRoot(opts.destinationDir, name);
      entriesExtracted += 1;
      opts.onProgress?.({
        phase: "extract",
        entryName: name,
        entriesExtracted,
        message: `Extracting ${name}`,
      });

      if (type === "directory") {
        done(
          (async () => {
            await fsp.mkdir(absPath, { recursive: true });
          })(),
        );
        return;
      }

      if (type === "file") {
        done(
          (async () => {
            await fsp.mkdir(path.dirname(absPath), { recursive: true });
            await pipeline(stream, fs.createWriteStream(absPath));
            if (mode !== undefined && process.platform !== "win32") {
              try {
                await fsp.chmod(absPath, mode);
              } catch {}
            }
          })(),
        );
        return;
      }

      // Ignore symlinks and other special file types for safety.
      done(Promise.resolve());
    } catch (e) {
      done(Promise.reject(e));
    }
  });

  const stat = await fsp.stat(opts.archivePath);
  const totalBytes = stat.size;
  const input = fs.createReadStream(opts.archivePath);
  let processedBytes = 0;
  let lastEmitAt = 0;
  const emitBytes = () => {
    const now = Date.now();
    if (now - lastEmitAt < 120) return;
    lastEmitAt = now;
    const percent =
      totalBytes > 0 ? Math.max(0, Math.min(1, processedBytes / totalBytes)) : undefined;
    opts.onProgress?.({
      phase: "extract",
      processedBytes,
      totalBytes,
      percent,
      message: percent !== undefined ? `Extracting… ${Math.round(percent * 100)}%` : "Extracting…",
    });
  };
  input.on("data", (chunk: string | Buffer) => {
    processedBytes += typeof chunk === "string" ? Buffer.byteLength(chunk) : chunk.length;
    emitBytes();
  });

  await new Promise<void>((resolve, reject) => {
    extract.on("finish", resolve);
    extract.on("error", reject);
    inflator.on("error", reject);
    input.on("error", reject);
    input.pipe(inflator).pipe(extract);
  });

  // Ensure all entry writes finished.
  await Promise.all(pending);
  opts.onProgress?.({
    phase: "extract",
    processedBytes: totalBytes,
    totalBytes,
    percent: 1,
    message: "Extraction complete",
  });
}

async function downloadToTempFile(opts: {
  url: string;
  assetName?: string;
  onProgress?: (ev: InstallerProgressEvent) => void;
}): Promise<string> {
  const res = await fetch(opts.url);
  if (!res.ok) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  }
  if (!res.body) throw new Error("Download failed: empty body");

  const totalBytesHeader = res.headers.get("content-length");
  const totalBytes = totalBytesHeader ? Number(totalBytesHeader) : undefined;

  const safeName = (opts.assetName || "bundle.tar.zst").replace(/[^\w.\-]+/g, "_");
  const outPath = path.join(os.tmpdir(), `apex-server-bundle-${randomUUID()}-${safeName}`);
  await fsp.mkdir(path.dirname(outPath), { recursive: true });
  const ws = fs.createWriteStream(outPath);
  const body = Readable.fromWeb(res.body as any);
  let downloadedBytes = 0;
  let lastEmitAt = 0;
  const emit = () => {
    const now = Date.now();
    if (now - lastEmitAt < 120) return;
    lastEmitAt = now;
    const percent =
      typeof totalBytes === "number" && totalBytes > 0
        ? Math.max(0, Math.min(1, downloadedBytes / totalBytes))
        : undefined;
    opts.onProgress?.({
      phase: "download",
      downloadedBytes,
      totalBytes,
      percent,
      message:
        percent !== undefined
          ? `Downloading… ${Math.round(percent * 100)}%`
          : "Downloading…",
    });
  };
  body.on("data", (chunk: string | Buffer) => {
    downloadedBytes += typeof chunk === "string" ? Buffer.byteLength(chunk) : chunk.length;
    emit();
  });
  opts.onProgress?.({
    phase: "download",
    downloadedBytes: 0,
    totalBytes,
    percent: 0,
    message: "Starting download…",
  });
  await pipeline(body, ws);
  opts.onProgress?.({
    phase: "download",
    downloadedBytes: typeof totalBytes === "number" ? totalBytes : downloadedBytes,
    totalBytes,
    percent: 1,
    message: "Download complete",
  });
  return outPath;
}

async function ensureFfmpegInstalled(): Promise<{
  installed: boolean;
  ffmpegPath: string;
  ffprobePath: string;
  method: "already-present" | "copied-from-ffmpeg-static";
}> {
  const installDir = getUserFfmpegInstallDir();
  await fsp.mkdir(installDir, { recursive: true });

  const targetFfmpeg = path.join(installDir, exeName("ffmpeg"));
  const targetFfprobe = path.join(installDir, exeName("ffprobe"));
  if (fs.existsSync(targetFfmpeg) && fs.existsSync(targetFfprobe)) {
    return {
      installed: true,
      ffmpegPath: targetFfmpeg,
      ffprobePath: targetFfprobe,
      method: "already-present",
    };
  }

  const require = createRequire(import.meta.url);

  // These packages are already dependencies in `apps/app/package.json`
  const srcFfmpeg = (() => {
    try {
      const p = require("ffmpeg-static");
      return typeof p === "string" ? p : null;
    } catch {
      return null;
    }
  })();
  const srcFfprobe = (() => {
    try {
      const mod = require("ffprobe-static");
      const p = typeof mod === "string" ? mod : mod?.path;
      return typeof p === "string" ? p : null;
    } catch {
      return null;
    }
  })();

  if (!srcFfmpeg || !fs.existsSync(srcFfmpeg)) {
    throw new Error("Could not resolve ffmpeg-static binary to install ffmpeg");
  }
  if (!srcFfprobe || !fs.existsSync(srcFfprobe)) {
    throw new Error("Could not resolve ffprobe-static binary to install ffprobe");
  }

  await fsp.copyFile(srcFfmpeg, targetFfmpeg);
  await fsp.copyFile(srcFfprobe, targetFfprobe);
  if (process.platform !== "win32") {
    await fsp.chmod(targetFfmpeg, 0o755);
    await fsp.chmod(targetFfprobe, 0o755);
  }

  return {
    installed: true,
    ffmpegPath: targetFfmpeg,
    ffprobePath: targetFfprobe,
    method: "copied-from-ffmpeg-static",
  };
}

export class InstallerModule implements AppModule {
  async enable(): Promise<void> {
    // idempotent-ish registration
    if (ipcMain.listenerCount("installer:extract-server-bundle") > 0) return;

    ipcMain.handle(
      "installer:extract-server-bundle",
      async (_evt, req: ExtractBundleRequest): Promise<ConfigResponse<{ extractedTo: string }>> => {
        try {
          const jobId = String(req?.jobId || randomUUID());
          const sendProgress = (ev: InstallerProgressEvent) => {
            _evt.sender.send(`installer:progress:${jobId}`, ev);
          };

          const destinationDir = String(req?.destinationDir || "").trim();
          if (!destinationDir) throw new Error("destinationDir is required");

          const src = req?.source;
          if (!src || (src.kind !== "local" && src.kind !== "remote")) {
            throw new Error("source is required");
          }

          await fsp.mkdir(destinationDir, { recursive: true });

          let archivePath: string;
          if (src.kind === "local") {
            archivePath = String(src.path || "").trim();
            if (!archivePath) throw new Error("local bundle path is required");
            if (!fs.existsSync(archivePath)) {
              throw new Error(`Bundle file not found: ${archivePath}`);
            }
            sendProgress({ phase: "status", message: "Using local bundle" });
          } else {
            const url = String(src.url || "").trim();
            if (!url) throw new Error("remote bundle url is required");
            sendProgress({ phase: "status", message: "Downloading bundle…" });
            archivePath = await downloadToTempFile({
              url,
              assetName: src.assetName,
              onProgress: sendProgress,
            });
          }

          // Always use Node-based extraction so we can stream progress back to the renderer.
          // (We keep the tar-based extractor around as a fallback for future use.)
          sendProgress({ phase: "status", message: "Extracting bundle…" });
          await extractTarZstWithNode({
            archivePath,
            destinationDir,
            onProgress: sendProgress,
          });

          sendProgress({ phase: "status", message: "Bundle extracted" });
          return { success: true, data: { extractedTo: destinationDir } };
        } catch (e) {
          return { success: false, error: e instanceof Error ? e.message : "Failed to extract bundle" };
        }
      },
    );

    ipcMain.handle(
      "installer:ensure-ffmpeg",
      async (): Promise<
        ConfigResponse<{
          installed: boolean;
          ffmpegPath: string;
          ffprobePath: string;
          method: "already-present" | "copied-from-ffmpeg-static";
        }>
      > => {
        try {
          const data = await ensureFfmpegInstalled();
          return { success: true, data };
        } catch (e) {
          return { success: false, error: e instanceof Error ? e.message : "Failed to install ffmpeg" };
        }
      },
    );
  }
}

export function installerModule() {
  return new InstallerModule();
}


