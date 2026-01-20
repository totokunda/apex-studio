import type { AppModule } from "../AppModule.js";
import { app, ipcMain } from "electron";
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
import net from "node:net";

type ConfigResponse<T> = { success: true; data: T } | { success: false; error: string };

type BundleSource =
  | { kind: "local"; path: string }
  | { kind: "remote"; url: string; assetName?: string };

type ExtractBundleRequest = {
  source: BundleSource;
  destinationDir: string;
  jobId?: string;
};

type RunSetupRequest = {
  /**
   * Base directory used by the Python API to persist config and downloads.
   * This becomes the APEX_HOME_DIR env var and also passes --apex_home_dir to setup.py
   */
  apexHomeDir: string;
  /**
   * The directory where the server bundle (.tar.zst) was extracted.
   * Typically contains an "apex-engine/" folder.
   */
  apiInstallDir: string;
  maskModelType?: string | null;
  installRife?: boolean;
  enableImageRenderSteps?: boolean;
  enableVideoRenderSteps?: boolean;
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

type SetupProgressPayload = {
  progress: number | null;
  message: string;
  status: string;
  metadata?: Record<string, any>;
};

function getUserFfmpegInstallDir(): string {
  // Stable, user-writable location. Preload will also check this path for discovery.
  return path.join(os.homedir(), ".apex-studio", "ffmpeg");
}

function exeName(cmd: "ffmpeg" | "ffprobe") {
  return process.platform === "win32" ? `${cmd}.exe` : cmd;
}

function resolveApiBundleRoot(apiInstallDir: string): string {
  /**
   * We want the directory that contains the bundled venv at:
   *   <bundleRoot>/apex-studio/(bin|Scripts)/python
   *
   * Bundle layouts we support (apiInstallDir is user-chosen install dir / extraction dir):
   * - <apiInstallDir>/python-api/apex-engine/apex-studio/...
   * - <apiInstallDir>/apex-engine/apex-studio/...
   * - <apiInstallDir>/python-api (already points at python-api root)
   * - <apiInstallDir> (already points at apex-engine root)
   */
  const root = String(apiInstallDir || "").trim();
  if (!root) return root;

  const candidates = [
    // Most common for current bundles:
    path.join(root, "python-api", "apex-engine"),
    // Some older/dev bundles:
    path.join(root, "apex-engine"),
    // If caller already passed python-api root:
    path.join(root, "apex-engine"),
    // If caller already passed apex-engine root:
    root,
  ];

  for (const p of candidates) {
    try {
      if (fs.existsSync(p) && fs.statSync(p).isDirectory()) {
        // Ensure it looks like a bundle root by checking for apex-studio folder.
        const marker = path.join(p, "apex-studio");
        if (fs.existsSync(marker) && fs.statSync(marker).isDirectory()) return p;
      }
    } catch {}
  }

  // Fall back to root if we can't confidently detect it.
  return root;
}

function resolveExtractedPythonExe(bundleRoot: string): string | null {
  // Mirror the structure used in packaged builds:
  //   <bundleRoot>/apex-studio/bin/python (mac/linux)
  //   <bundleRoot>/apex-studio/Scripts/python.exe (win)
  const base = path.join(bundleRoot, "apex-studio");
  const candidates =
    process.platform === "win32"
      ? [path.join(base, "Scripts", "python.exe")]
      : [path.join(base, "bin", "python"), path.join(base, "bin", "python3")];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }
  return null;
}

function resolveDevApiRoot(): string | null {
  // Locate <repo>/apps/api by looking for pyproject.toml.
  const cwd = process.cwd();
  const candidates = [
    // common from repo root
    path.resolve(cwd, "apps", "api"),
    // common when running from apps/app
    path.resolve(cwd, "..", "api"),
    // fallback: traverse up a couple levels
    path.resolve(cwd, "..", "..", "apps", "api"),
  ];
  for (const p of candidates) {
    try {
      const marker = path.join(p, "pyproject.toml");
      if (fs.existsSync(marker)) return p;
    } catch {}
  }
  return null;
}

function resolveSetupScriptPath(bundleRoot: string): string | null {
  // Prefer a setup.py shipped alongside the bundle root (if present), otherwise fall back to dev repo path.
  const fromBundle = path.join(bundleRoot, "scripts", "setup.py");
  try {
    if (fs.existsSync(fromBundle)) return fromBundle;
  } catch {}

  // Common bundle layout: <installDir>/python-api/apex-engine is bundleRoot, but setup.py lives in
  // <installDir>/python-api/scripts/setup.py.
  try {
    const p = path.join(path.dirname(bundleRoot), "scripts", "setup.py");
    if (fs.existsSync(p)) return p;
  } catch {}

  const devApiRoot = resolveDevApiRoot();
  if (devApiRoot) {
    const p = path.join(devApiRoot, "scripts", "setup.py");
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }

  // Packaged fallback: some builds may ship python-api sources in resourcesPath.
  try {
    const resources = process.resourcesPath;
    const p = path.join(resources, "python-api", "scripts", "setup.py");
    if (fs.existsSync(p)) return p;
  } catch {}

  // Last resort: the repo root path if available (for certain dev configurations)
  try {
    const p = path.resolve(process.cwd(), "apps", "api", "scripts", "setup.py");
    if (fs.existsSync(p)) return p;
  } catch {}

  return null;
}

function appendEnvPath(existing: string | undefined, add: string): string {
  const sep = process.platform === "win32" ? ";" : ":";
  const a = String(add || "").trim();
  if (!a) return existing || "";
  const e = String(existing || "").trim();
  if (!e) return a;
  // Avoid duplicates in simple cases
  if (e.split(sep).includes(a)) return e;
  return `${a}${sep}${e}`;
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

async function safeRemovePath(p: string): Promise<void> {
  const target = String(p || "").trim();
  if (!target) return;
  try {
    await fsp.rm(target, { recursive: true, force: true });
  } catch {}
}

async function isNonEmptyDir(p: string): Promise<boolean> {
  try {
    const st = await fsp.stat(p);
    if (!st.isDirectory()) return false;
    const entries = await fsp.readdir(p);
    return entries.length > 0;
  } catch {
    return false;
  }
}

function assertSafeInstallDir(destinationDir: string): void {
  const dest = path.resolve(String(destinationDir || "").trim());
  if (!dest) throw new Error("destinationDir is required");
  const root = path.parse(dest).root;
  if (dest === root) throw new Error(`Refusing to install into filesystem root: ${dest}`);
  // Prevent foot-guns like installing into home dir directly.
  const home = os.homedir();
  if (path.resolve(dest) === path.resolve(home)) {
    throw new Error(`Refusing to install into home directory: ${dest}`);
  }
}

async function validateExtractedBundleOrThrow(opts: {
  installDir: string;
  timeoutMs?: number;
}): Promise<void> {
  const installDir = path.resolve(String(opts.installDir || "").trim());
  if (!installDir) throw new Error("installDir is required");

  const bundleRoot = resolveApiBundleRoot(installDir);
  const pythonExe = resolveExtractedPythonExe(bundleRoot);
  if (!pythonExe) {
    throw new Error(`Could not find bundled Python executable under: ${path.join(bundleRoot, "apex-studio")}`);
  }

  // Keep this lightweight: we only need to prove the interpreter can start.
  const timeoutMs = typeof opts.timeoutMs === "number" ? opts.timeoutMs : 8000;
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const child = spawn(pythonExe, ["-c", "import sys; print(sys.version)"], {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        // Avoid user/site/path pollution.
        PYTHONNOUSERSITE: "1",
        PYTHONDONTWRITEBYTECODE: "1",
      },
    });

    let stdout = "";
    let stderr = "";
    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            try {
              // Try hard to avoid hanging the installer. On some systems "killed" processes
              // can become awkward to reap; we still force-settle the promise on timeout.
              child.kill("SIGKILL");
            } catch {}
            if (settled) return;
            settled = true;
            reject(new Error(`Bundle validation timed out after ${timeoutMs}ms (python did not exit)`));
          }, timeoutMs)
        : null;

    child.stdout?.on("data", (b) => (stdout += b.toString("utf-8")));
    child.stderr?.on("data", (b) => (stderr += b.toString("utf-8")));

    child.on("error", (err) => {
      if (timer) clearTimeout(timer);
      if (settled) return;
      settled = true;
      reject(err);
    });
    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      if (settled) return;
      settled = true;
      if (code === 0) return resolve();
      const out = `${stdout}\n${stderr}`.trim();
      reject(new Error(`Bundle validation failed (exit ${code}). Output:\n${out}`));
    });
  });
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
  try {
    await pipeline(body, ws);
  } catch (e) {
    // If the pipeline errors mid-stream, ensure we don't leave a partial temp file behind.
    try {
      ws.destroy();
    } catch {}
    await safeRemovePath(outPath);
    throw e;
  }
  opts.onProgress?.({
    phase: "download",
    downloadedBytes: typeof totalBytes === "number" ? totalBytes : downloadedBytes,
    totalBytes,
    percent: 1,
    message: "Download complete",
  });
  return outPath;
}

async function removeAppleDoubleFilesRecursively(opts: {
  rootDir: string;
  onStatus?: (msg: string) => void;
}): Promise<{ removedFiles: number; removedDirs: number }> {
  /**
   * Defensive post-extraction cleanup.
   *
   * Some macOS workflows (Finder, certain zip/tar behaviors, network drives) can inject
   * AppleDouble metadata files (`._*`) and `__MACOSX/` folders into archives.
   *
   * These can break Python libs that scan `*.py` (e.g. transformers) if `._*.py` exists.
   */
  const root = String(opts.rootDir || "").trim();
  if (!root) return { removedFiles: 0, removedDirs: 0 };

  let removedFiles = 0;
  let removedDirs = 0;

  const walk = async (dir: string) => {
    let entries: fs.Dirent[];
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }

    await Promise.all(
      entries.map(async (ent) => {
        const name = ent.name;
        const full = path.join(dir, name);
        if (ent.isDirectory()) {
          if (name === "__MACOSX") {
            try {
              await fsp.rm(full, { recursive: true, force: true });
              removedDirs += 1;
            } catch {}
            return;
          }
          await walk(full);
          return;
        }
        if (!ent.isFile()) return;
        if (!name.startsWith("._")) return;
        try {
          await fsp.unlink(full);
          removedFiles += 1;
        } catch {}
      }),
    );
  };

  opts.onStatus?.("Cleaning up macOS metadata files (._*) …");
  await walk(root);
  if (removedFiles > 0 || removedDirs > 0) {
    opts.onStatus?.(`Cleanup complete (removed ${removedFiles} file(s), ${removedDirs} dir(s))`);
  }
  return { removedFiles, removedDirs };
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

async function removeAppleDoublePyFiles(opts: {
  pythonExe: string;
  onStatus?: (msg: string) => void;
}): Promise<{ scannedDirs: number; removedFiles: number }> {
  /**
   * On macOS it’s common for archives/copies to accidentally include AppleDouble files like:
   *   transformers/._some_module.py
   * These are binary metadata blobs (not UTF-8), but `transformers` scans `*.py` and will
   * crash with UnicodeDecodeError when it hits them.
   *
   * We proactively delete them from site-packages before running setup.py.
   */
  const pythonExe = opts.pythonExe;
  const onStatus = opts.onStatus;

  const venvRoot =
    process.platform === "win32"
      ? path.dirname(path.dirname(pythonExe)) // .../apex-studio/Scripts/python.exe -> .../apex-studio
      : path.dirname(path.dirname(pythonExe)); // .../apex-studio/bin/python -> .../apex-studio

  const sitePackageCandidates: string[] = [];
  if (process.platform === "win32") {
    sitePackageCandidates.push(path.join(venvRoot, "Lib", "site-packages"));
  } else {
    // venvRoot/lib/pythonX.Y/site-packages
    try {
      const libDir = path.join(venvRoot, "lib");
      const entries = await fsp.readdir(libDir, { withFileTypes: true });
      for (const e of entries) {
        if (!e.isDirectory()) continue;
        const name = e.name;
        if (!/^python\\d+\\.\\d+$/.test(name)) continue;
        sitePackageCandidates.push(path.join(libDir, name, "site-packages"));
      }
    } catch {}
  }

  const roots = sitePackageCandidates.filter((p) => {
    try {
      return fs.existsSync(p) && fs.statSync(p).isDirectory();
    } catch {
      return false;
    }
  });

  let scannedDirs = 0;
  let removedFiles = 0;

  const walk = async (dir: string): Promise<void> => {
    scannedDirs += 1;
    let entries: fs.Dirent[];
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    await Promise.all(
      entries.map(async (ent) => {
        const name = ent.name;
        const full = path.join(dir, name);
        if (ent.isDirectory()) {
          await walk(full);
          return;
        }
        if (!ent.isFile()) return;
        if (!name.startsWith("._") || !name.endsWith(".py")) return;
        try {
          await fsp.unlink(full);
          removedFiles += 1;
        } catch {}
      }),
    );
  };

  for (const sp of roots) {
    // Target only transformers to keep this fast and avoid touching unrelated packages.
    const transformersDir = path.join(sp, "transformers");
    try {
      if (fs.existsSync(transformersDir) && fs.statSync(transformersDir).isDirectory()) {
        onStatus?.("Cleaning up macOS metadata files in transformers…");
        await walk(transformersDir);
      }
    } catch {}
  }

  if (removedFiles > 0) {
    onStatus?.(`Removed ${removedFiles} macOS metadata file(s) that can break transformers import`);
  }
  return { scannedDirs, removedFiles };
}

export class InstallerModule implements AppModule {
  async enable(): Promise<void> {
    // idempotent-ish registration
    if (ipcMain.listenerCount("installer:extract-server-bundle") > 0) return;

    ipcMain.handle(
      "installer:extract-server-bundle",
      async (_evt, req: ExtractBundleRequest): Promise<ConfigResponse<{ extractedTo: string }>> => {
        const tempPathsToCleanup: string[] = [];
        let stagingDirToCleanup: string | null = null;
        try {
          const jobId = String(req?.jobId || randomUUID());
          const sendProgress = (ev: InstallerProgressEvent) => {
            _evt.sender.send(`installer:progress:${jobId}`, ev);
          };

          const destinationDir = String(req?.destinationDir || "").trim();
          if (!destinationDir) throw new Error("destinationDir is required");
          assertSafeInstallDir(destinationDir);
          const destinationAbs = path.resolve(destinationDir);

          const src = req?.source;
          if (!src || (src.kind !== "local" && src.kind !== "remote")) {
            throw new Error("source is required");
          }

          // If destination already contains an install, use a staging directory so we can
          // validate the new bundle before swapping it into place.
          const hasExistingInstall = await isNonEmptyDir(destinationAbs);
          const effectiveDestinationDir = hasExistingInstall
            ? path.join(path.dirname(destinationAbs), `.apex-install-staging-${randomUUID()}`)
            : destinationAbs;
          if (hasExistingInstall) {
            stagingDirToCleanup = effectiveDestinationDir;
            sendProgress({
              phase: "status",
              message: "Existing install detected; extracting into staging directory for safe reinstall…",
            });
          }
          await fsp.mkdir(effectiveDestinationDir, { recursive: true });

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
            // Track temp artifacts we create so we can always delete them later.
            tempPathsToCleanup.push(archivePath);
          }

          // Always use Node-based extraction so we can stream progress back to the renderer.
          // (We keep the tar-based extractor around as a fallback for future use.)
          sendProgress({ phase: "status", message: "Extracting bundle…" });
          await extractTarZstWithNode({
            archivePath,
            destinationDir: effectiveDestinationDir,
            onProgress: sendProgress,
          });

          // Defensive cleanup: remove AppleDouble `._*` files that can slip into archives and break runtime.
          try {
            await removeAppleDoubleFilesRecursively({
              rootDir: effectiveDestinationDir,
              onStatus: (m) => sendProgress({ phase: "status", message: m }),
            });
          } catch {}

          sendProgress({ phase: "status", message: "Validating extracted bundle…" });
          await validateExtractedBundleOrThrow({ installDir: effectiveDestinationDir, timeoutMs: 10_000 });

          // Safe reinstall swap:
          // - Only now do we remove the previous install and move the staged dir into place.
          if (hasExistingInstall) {
            sendProgress({ phase: "status", message: "Replacing existing install…" });
            try {
              await fsp.rm(destinationAbs, { recursive: true, force: true });
            } catch {}
            // Rename is atomic when source+dest are on the same filesystem (we ensure sibling paths).
            await fsp.rename(effectiveDestinationDir, destinationAbs);
            stagingDirToCleanup = null; // now lives at destinationAbs
          }

          sendProgress({ phase: "status", message: "Bundle extracted" });
          return { success: true, data: { extractedTo: destinationAbs } };
        } catch (e) {
          return { success: false, error: e instanceof Error ? e.message : "Failed to extract bundle" };
        } finally {
          if (stagingDirToCleanup) {
            try {
              await safeRemovePath(stagingDirToCleanup);
            } catch {}
          }
          if (tempPathsToCleanup.length > 0) {
            try {
              // Best-effort cleanup: never fail the overall operation because temp cleanup failed.
              await Promise.all(tempPathsToCleanup.map((p) => safeRemovePath(p)));
            } catch {}
          }
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

    ipcMain.handle(
      "installer:run-setup",
      async (
        _evt,
        req: RunSetupRequest,
      ): Promise<ConfigResponse<{ jobId: string }>> => {
        const jobId = String(req?.jobId || randomUUID());
        const sendStatus = (message: string) => {
          try {
            _evt.sender.send(`installer:progress:${jobId}`, { phase: "status", message });
          } catch {}
        };
        const sendSetupProgress = (payload: SetupProgressPayload) => {
          try {
            _evt.sender.send(`installer:setup-progress:${jobId}`, payload);
          } catch {}
        };
        try {
          const apexHomeDir = String(req?.apexHomeDir || "").trim();
          const apiInstallDir = String(req?.apiInstallDir || "").trim();
          if (!apexHomeDir) throw new Error("apexHomeDir is required");
          if (!apiInstallDir) throw new Error("apiInstallDir is required");

          await fsp.mkdir(apexHomeDir, { recursive: true });

          const bundleRoot = resolveApiBundleRoot(apiInstallDir);
          if (!bundleRoot) throw new Error("Failed to resolve API bundle root");
          if (!fs.existsSync(bundleRoot)) {
            throw new Error(`API bundle root does not exist: ${bundleRoot}`);
          }

          const pythonExe = resolveExtractedPythonExe(bundleRoot);
          if (!pythonExe) {
            throw new Error(
              `Could not find bundled Python executable under: ${path.join(bundleRoot, "apex-studio")}`,
            );
          }

          const setupScript = resolveSetupScriptPath(bundleRoot);
          console.log("setupScript", setupScript);
          if (!setupScript) {
            throw new Error(
              "Could not locate setup.py. Expected it under the extracted bundle (scripts/setup.py) or the dev repo (apps/api/scripts/setup.py).",
            );
          }

          const args: string[] = [
            setupScript,
            "--apex_home_dir",
            apexHomeDir,
            "--job_id",
            jobId,
          ];

          const maskModelType = (req?.maskModelType ?? "").toString().trim();
          if (maskModelType) {
            args.push("--mask_model_type", maskModelType);
          }
          if (req?.installRife) args.push("--install_rife");
          if (req?.enableImageRenderSteps) args.push("--enable_image_render_steps");
          if (req?.enableVideoRenderSteps) args.push("--enable_video_render_steps");

          const env: NodeJS.ProcessEnv = {
            ...process.env,
            APEX_HOME_DIR: apexHomeDir,
            PYTHONUNBUFFERED: "1",
            APEX_SETUP_PROGRESS_JSON: "1",
            // Ensure `import src` works when running setup.py from the repo scripts directory.
            PYTHONPATH: appendEnvPath(process.env.PYTHONPATH, bundleRoot),
          };

          sendStatus("Starting setup (models + config) …");

          // Work around a known macOS packaging/extraction gotcha where AppleDouble files (._*.py)
          // sneak into site-packages and crash transformers import.
          try {
            await removeAppleDoublePyFiles({ pythonExe, onStatus: sendStatus });
          } catch {}


          const child = spawn(pythonExe, args, {
            cwd: bundleRoot,
            env,
            stdio: ["ignore", "pipe", "pipe"],
            detached: false,
          });

          const sendSetupLogLine = (kind: "stdout" | "stderr", line: string) => {
            const trimmed = String(line || "").replace(/\s+$/, "");
            if (!trimmed) return;

            // JSON progress events are emitted by setup.py when APEX_SETUP_PROGRESS_JSON=1.
            // We forward them to the renderer so it can update installer phases without websockets.
            if (kind === "stdout" && trimmed.startsWith("{") && trimmed.endsWith("}")) {
              try {
                const obj = JSON.parse(trimmed) as SetupProgressPayload;
                if (obj && typeof obj === "object" && typeof (obj as any).message === "string") {
                  sendSetupProgress(obj);
                }
              } catch {}
            }

            const msg = `[setup:${kind}] ${trimmed}`.slice(0, 1200);
            // Emit to renderer (for DevTools console + debugging).
            sendStatus(msg);
            // Also log in main process console.
            try {
              // eslint-disable-next-line no-console
              console.log(msg);
            } catch {}
          };

          const makeLineEmitter = (kind: "stdout" | "stderr") => {
            let buf = "";
            return (d: Buffer | string) => {
              buf += typeof d === "string" ? d : d.toString("utf8");
              // Split on both \n and \r\n
              const parts = buf.split(/\r?\n/);
              buf = parts.pop() ?? "";
              for (const p of parts) sendSetupLogLine(kind, p);
              // Prevent unbounded growth if a process prints huge non-newline output.
              if (buf.length > 16_000) {
                sendSetupLogLine(kind, buf.slice(0, 16_000));
                buf = "";
              }
            };
          };

          let stderr = "";
          const emitStdout = makeLineEmitter("stdout");
          const emitStderr = makeLineEmitter("stderr");
          child.stdout?.setEncoding?.("utf8");
          child.stderr?.setEncoding?.("utf8");
          child.stdout?.on("data", emitStdout);
          child.stderr?.on("data", (d) => {
            stderr += String(d);
            if (stderr.length > 20_000) stderr = stderr.slice(-20_000);
            emitStderr(d as any);
          });

          child.on("error", (err) => {
            sendStatus(`Setup process failed to start: ${err instanceof Error ? err.message : String(err)}`);
            // Ensure the renderer can resolve/reject even if setup.py never emitted JSON progress.
            sendSetupProgress({
              progress: null,
              status: "error",
              message: `Setup process failed to start: ${
                err instanceof Error ? err.message : String(err)
              }`,
              metadata: { task: "setup" },
            });
          });

          child.on("close", (code) => {
            if (code === 0) {
              sendStatus("Setup process finished");
              // Some reinstall / no-op paths may exit without emitting JSON progress.
              // Always emit a final "setup complete" event so the renderer can finish.
              sendSetupProgress({
                progress: 1,
                status: "complete",
                message: "Setup complete",
                metadata: { task: "setup", task_progress: 1 },
              });
              return;
            }
            const suffix = stderr.trim() ? ` (stderr: ${stderr.trim().slice(0, 800)})` : "";
            const msg = `Setup process exited with code ${code}${suffix}`;
            sendStatus(msg);
            // Ensure the renderer can resolve/reject even if setup.py never emitted JSON progress.
            sendSetupProgress({
              progress: null,
              status: "error",
              message: msg,
              metadata: { task: "setup" },
            });
          });
          return { success: true, data: { jobId } };
        } catch (e) {
          const msg = e instanceof Error ? e.message : "Failed to run setup";
          sendStatus(`Setup failed: ${msg}`);
          return { success: false, error: msg };
        }
      },
    );
  }
}

export function installerModule() {
  return new InstallerModule();
}


