import { sha256sum } from "./nodeCrypto.js";
import { versions } from "./versions.js";
import { processMediaTo24 } from "./media/process.js";
import {
  renameMediaPairInRoot,
  deleteMediaPairInRoot,
} from "./media/renameDelete.js";
import { ipcRenderer, webUtils } from "electron";
import { getMediaRootAbsolute } from "./media/root.js";
import { symlinksDir, proxyDir } from "./media/paths.js";
import { promises as fsp } from "node:fs";
import fs from "node:fs";
import { join, basename, extname, dirname } from "node:path";
import { homedir } from "node:os";
import { pathToFileURL, fileURLToPath } from "node:url";
import {
  AUDIO_EXTS,
  IMAGE_EXTS,
  VIDEO_EXTS,
  getLowercaseExtension,
} from "./media/fileExts.js";
import { ensureUniqueNameSync } from "./media/links.js";
import { ClipType } from "../../renderer/src/lib/types.js";
import { fetchFilters } from "./filters/fetch.js";
import { spawn } from "node:child_process";
// import os from 'node:os';

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
      } catch {}
    }
    await fsp.rm(p, { force: true });
  } catch {
    // swallow
  }
}

async function renameMediaPair(
  convertedOldName: string,
  convertedNewName: string,
) {
  return renameMediaPairInRoot(
    getMediaRootAbsolute(),
    convertedOldName,
    convertedNewName,
  );
}

async function deleteMediaPair(convertedName: string) {
  return deleteMediaPairInRoot(getMediaRootAbsolute(), convertedName);
}

async function pickMediaPaths(options: {
  directory?: boolean;
  filters?: { name: string; extensions: string[] }[];
  title?: string;
}): Promise<string[]> {
  try {
    const paths = await ipcRenderer.invoke("dialog:pick-media", options ?? {});
    if (Array.isArray(paths)) return paths as string[];
    return [];
  } catch {
    return [];
  }
}

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
    } catch {}
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

export type ConvertedMediaItem = {
  name: string;
  absPath: string;
  assetUrl: string;
  dateAddedMs: number;
  type: ClipType;
  hasProxy?: boolean;
};

type ProxyIndex = {
  [symlinkName: string]: string; // maps symlink name to proxy filename
};

function isSupportedExt(ext: string): boolean {
  return VIDEO_EXTS.has(ext) || IMAGE_EXTS.has(ext) || AUDIO_EXTS.has(ext);
}

async function loadProxyIndex(): Promise<ProxyIndex> {
  try {
    const root = getMediaRootAbsolute();
    const indexPath = join(root, ".proxy-index.json");
    const content = await fsp.readFile(indexPath, "utf8");
    return JSON.parse(content);
  } catch {
    return {};
  }
}

async function saveProxyIndex(index: ProxyIndex): Promise<void> {
  const root = getMediaRootAbsolute();
  const indexPath = join(root, ".proxy-index.json");
  await fsp.writeFile(indexPath, JSON.stringify(index, null, 2), "utf8");
}

async function ensureMediaDirs(): Promise<{
  root: string;
  symlinksAbs: string;
  proxyAbs: string;
}> {
  const root = getMediaRootAbsolute();
  const symlinksAbs = symlinksDir(root);
  const proxyAbs = proxyDir(root);
  if (!fs.existsSync(symlinksAbs))
    fs.mkdirSync(symlinksAbs, { recursive: true });
  if (!fs.existsSync(proxyAbs)) fs.mkdirSync(proxyAbs, { recursive: true });
  return { root, symlinksAbs, proxyAbs };
}

async function ensurePreviewDir(): Promise<{
  root: string;
  previewsAbs: string;
}> {
  const root = getMediaRootAbsolute();
  const previewsAbs = join(root, "previews");
  if (!fs.existsSync(previewsAbs))
    fs.mkdirSync(previewsAbs, { recursive: true });
  return { root, previewsAbs };
}

async function savePreviewImage(
  data: ArrayBuffer | Uint8Array | string,
  options?: { fileNameHint?: string },
): Promise<string> {
  const { previewsAbs } = await ensurePreviewDir();

  let buffer: Buffer;
  let ext = "png";
  if (typeof data === "string") {
    const m = /^data:image\/(png|jpeg);base64,(.*)$/i.exec(data.trim());
    if (m && m[2]) {
      ext = m[1] === "jpeg" ? "jpg" : "png";
      buffer = Buffer.from(m[2], "base64");
    } else {
      // Treat string as raw path not supported here
      throw new Error(
        "savePreviewImage: expected data URL string or binary buffer",
      );
    }
  } else if (data instanceof Uint8Array) {
    buffer = Buffer.from(data);
  } else {
    buffer = Buffer.from(new Uint8Array(data));
  }

  const baseName = (options?.fileNameHint || `preview_${Date.now()}`)
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const desired = baseName.toLowerCase().endsWith(`.${ext}`)
    ? baseName
    : `${baseName}.${ext}`;

  let fileName = desired;
  let attempt = 0;
  while (fs.existsSync(join(previewsAbs, fileName))) {
    attempt += 1;
    const stem = desired.replace(/\.[^.]+$/, "");
    fileName = `${stem}_${attempt}.${ext}`;
  }

  const absPath = join(previewsAbs, fileName);
  await fsp.writeFile(absPath, buffer);
  return absPath;
}

// Save an image buffer or data URL directly to a specified absolute path.
async function saveImageToPath(
  data: ArrayBuffer | Uint8Array | string,
  outPath: string,
): Promise<string> {
  let buffer: Buffer;
  if (typeof data === "string") {
    const m = /^data:image\/(png|jpe?g|webp);base64,(.*)$/i.exec(data.trim());
    if (m && m[2]) {
      buffer = Buffer.from(m[2], "base64");
    } else {
      throw new Error(
        "saveImageToPath: expected data URL string or binary buffer",
      );
    }
  } else if (data instanceof Uint8Array) {
    buffer = Buffer.from(data);
  } else {
    buffer = Buffer.from(new Uint8Array(data));
  }

  let p = String(outPath || "");
  if (!p) throw new Error("saveImageToPath: outPath is required");
  if (p.startsWith("file://")) {
    try {
      p = fileURLToPath(p);
    } catch {
      // fall through with original
    }
  }

  const dir = dirname(p);
  if (!fs.existsSync(dir)) {
    await fsp.mkdir(dir, { recursive: true });
  }

  await fsp.writeFile(p, buffer);
  return p;
}

// Save arbitrary audio buffer to previews directory
async function savePreviewAudio(
  data: ArrayBuffer | Uint8Array | string,
  options?: { fileNameHint?: string; ext?: string },
): Promise<string> {
  const { previewsAbs } = await ensurePreviewDir();

  let buffer: Buffer;
  let ext = (options?.ext || "wav").replace(/^\./, "");
  if (typeof data === "string") {
    const m = /^data:audio\/(wav|mp3|aac|ogg);base64,(.*)$/i.exec(data.trim());
    if (m && m[2]) {
      ext = m[1].toLowerCase();
      buffer = Buffer.from(m[2], "base64");
    } else {
      throw new Error(
        "savePreviewAudio: expected audio data URL or binary buffer",
      );
    }
  } else if (data instanceof Uint8Array) {
    buffer = Buffer.from(data);
  } else {
    buffer = Buffer.from(new Uint8Array(data));
  }

  const baseName = (options?.fileNameHint || `audio_${Date.now()}`)
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const desired = baseName.toLowerCase().endsWith(`.${ext}`)
    ? baseName
    : `${baseName}.${ext}`;

  let fileName = desired;
  let attempt = 0;
  while (fs.existsSync(join(previewsAbs, fileName))) {
    attempt += 1;
    const stem = desired.replace(/\.[^.]+$/, "");
    fileName = `${stem}_${attempt}.${ext}`;
  }

  const absPath = join(previewsAbs, fileName);
  await fsp.writeFile(absPath, buffer);
  return absPath;
}

// Encode provided WAV buffer to MP3 and save under previews, returning absolute path
async function exportAudioMp3FromWav(
  data: ArrayBuffer | Uint8Array | string,
  options?: { fileNameHint?: string },
): Promise<string> {
  const wavPath = await savePreviewAudio(data, {
    fileNameHint: (options?.fileNameHint || "audio") + ".wav",
    ext: "wav",
  });
  const { previewsAbs } = await ensurePreviewDir();
  const baseName = (options?.fileNameHint || `audio_${Date.now()}`)
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const desired = baseName.toLowerCase().endsWith(".mp3")
    ? baseName
    : `${baseName}.mp3`;
  let outName = desired;
  let attempt = 0;
  while (fs.existsSync(join(previewsAbs, outName))) {
    attempt += 1;
    const stem = desired.replace(/\.[^.]+$/, "");
    outName = `${stem}_${attempt}.mp3`;
  }
  const outAbs = join(previewsAbs, outName);

  await new Promise<void>((resolve, reject) => {
    const args = [
      "-y",
      "-i",
      wavPath,
      "-c:a",
      "libmp3lame",
      "-b:a",
      "192k",
      outAbs,
    ];
    const ff = spawn("ffmpeg", args);
    let stderr = "";
    ff.stderr.setEncoding("utf8");
    ff.stderr.on("data", (d) => {
      stderr += String(d);
    });
    ff.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg failed (${code}): ${stderr}`));
    });
    ff.on("error", (err) => reject(err));
  });

  return outAbs;
}

type AudioMixClipSpec = {
  src: string;
  startFrame?: number;
  endFrame?: number;
  trimStart?: number;
  volumeDb?: number;
  fadeInSec?: number;
  fadeOutSec?: number;
  speed?: number;
};

function buildAtempoChain(rate: number): string[] {
  const parts: string[] = [];
  const clamp = (x: number, lo: number, hi: number) =>
    Math.max(lo, Math.min(hi, x));
  let r = Math.max(
    0.1,
    Math.min(5, Number.isFinite(rate) && rate > 0 ? rate : 1),
  );
  if (r === 1) return [];
  // Each atempo must be between 0.5 and 2.0; chain to achieve r
  if (r > 1) {
    while (r > 2.0 + 1e-6) {
      parts.push("2.0");
      r /= 2.0;
    }
    parts.push(r.toFixed(5));
  } else {
    while (r < 0.5 - 1e-6) {
      parts.push("0.5");
      r /= 0.5;
    }
    parts.push(clamp(r, 0.5, 2.0).toFixed(5));
  }
  return parts;
}

// Detect whether a media file has at least one audio stream using ffprobe.
// If ffprobe is unavailable or returns an unexpected error, we conservatively
// assume audio is present so behavior degrades to the previous implementation.
async function hasAudioStream(path: string): Promise<boolean> {
  return await new Promise<boolean>((resolve) => {
    try {
      const ff = spawn("ffprobe", [
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        path,
      ]);
      let stdout = "";
      let stderr = "";
      ff.stdout.setEncoding("utf8");
      ff.stderr.setEncoding("utf8");
      ff.stdout.on("data", (d) => {
        stdout += String(d);
      });
      ff.stderr.on("data", (d) => {
        stderr += String(d);
      });
      ff.on("close", (code) => {
        // Typical "no audio" case: ffprobe complains that the stream specifier
        // matches no streams.
        if (stderr.includes("matches no streams")) {
          resolve(false);
          return;
        }
        if (code === 0 && stdout.trim().length > 0) {
          resolve(true);
          return;
        }
        // Unexpected error: assume audio exists to preserve legacy behavior.
        resolve(true);
      });
      ff.on("error", () => {
        // If ffprobe itself fails (missing, permissions, etc), keep old behavior.
        resolve(true);
      });
    } catch {
      resolve(true);
    }
  });
}

async function renderAudioMixWithFfmpeg(
  clips: AudioMixClipSpec[],
  options: {
    fps: number;
    exportStartFrame: number;
    exportEndFrame: number;
    outFormat?: "wav" | "mp3";
    fileNameHint?: string;
    filename?: string;
  },
): Promise<string | null> {
  if (!Array.isArray(clips) || clips.length === 0) return null;
  const fps = Math.max(1, Math.trunc(options.fps || 24));
  const exportStart = Math.max(0, Math.trunc(options.exportStartFrame || 0));
  const exportEnd = Math.max(
    exportStart + 1,
    Math.trunc(options.exportEndFrame || exportStart + 1),
  );
  const durationSec = Math.max(0.001, (exportEnd - exportStart) / fps);
  const outFormat = options.outFormat || "wav";
  const { previewsAbs } = await ensurePreviewDir();

  const baseName = (options.fileNameHint || `audio_mix_${Date.now()}`)
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const desired = baseName.toLowerCase().endsWith(`.${outFormat}`)
    ? baseName
    : `${baseName}.${outFormat}`;
  let outName = options.filename ? options.filename : desired;
  let attempt = 0;

  let outAbs: string | undefined = options.filename;

  if (!options.filename) {
    while (fs.existsSync(join(previewsAbs, outName))) {
      attempt += 1;
      const stem = desired.replace(/\.[^.]+$/, "");
      outName = `${stem}_${attempt}.${outFormat}`;
    }
    outAbs = join(previewsAbs, outName);
  }
  if (!outAbs) {
    throw new Error("No output file path provided");
  }
  
  const args: string[] = ["-y"];
  // Inputs – only include sources that actually have an audio stream so we don't
  // reference nonexistent pads like "[2:a]" for silent video files.
  type ActiveClip = { clip: AudioMixClipSpec; inputIndex: number };
  const activeClips: ActiveClip[] = [];
  let nextInputIndex = 0;
  for (const c of clips) {
    let p = c.src;
    try {
      if (typeof p === "string" && p.startsWith("file://"))
        p = fileURLToPath(p);
    } catch {}
    // eslint-disable-next-line no-await-in-loop
    const hasAudio = await hasAudioStream(p);
    if (!hasAudio) continue;
    args.push("-i", p);
    activeClips.push({ clip: c, inputIndex: nextInputIndex });
    nextInputIndex += 1;
  }

  if (activeClips.length === 0) {
    // No usable audio inputs – nothing to mix.
    return null;
  }

  const filterParts: string[] = [];
  const outLabels: string[] = [];
  activeClips.forEach(({ clip: c, inputIndex }) => {
    const volDb = Number.isFinite(c.volumeDb as number)
      ? (c.volumeDb as number)
      : 0;
    const fadeIn = Math.max(0, Number(c.fadeInSec || 0));
    const fadeOut = Math.max(0, Number(c.fadeOutSec || 0));
    const speed = Math.max(0.1, Math.min(5, Number(c.speed || 1)));
    const clipStart = Math.max(0, Math.trunc(c.startFrame ?? 0));
    const clipEnd = Math.max(
      clipStart,
      Math.trunc((c.endFrame as number) ?? clipStart),
    );

    // Normalize trimStart to a finite, non-negative frame count.
    const trimStartFramesRaw = Number(c.trimStart ?? 0);
    const trimStartFrames = Number.isFinite(trimStartFramesRaw)
      ? Math.max(0, Math.trunc(trimStartFramesRaw))
      : 0;

    // Project-frame position at which the underlying media frame 0 would have appeared
    // before any timeline trims. This stays stable as the user trims the head of the clip.
    const realStartFrame = clipStart - trimStartFrames;

    // Intersection of this clip's visible range with the export range.
    const visibleStart = Math.max(exportStart, clipStart);
    const visibleEnd = Math.max(visibleStart, Math.min(exportEnd, clipEnd));

    // Duration of this clip within the export timeline.
    const clipTimelineDurSec = Math.max(0, (visibleEnd - visibleStart) / fps);

    // Delay from exportStart until this clip becomes audible on the mixed timeline.
    const delayMs = Math.max(
      0,
      Math.round(((visibleStart - exportStart) / fps) * 1000),
    );

    // Where to start reading from inside the source (in frames/seconds).
    const mediaStartFrames = Math.max(0, visibleStart - realStartFrame);
    const mediaStartSec = mediaStartFrames / fps;
    const atempoChain = buildAtempoChain(speed);
    const labelIn = `[${inputIndex}:a]`;
    const labelOut = `[a${inputIndex}]`;
    const chain: string[] = [];
    chain.push("aresample=async=1");
    chain.push(`atrim=start=${mediaStartSec.toFixed(6)}`);
    chain.push("asetpts=PTS-STARTPTS");
    if (atempoChain.length > 0) {
      for (const r of atempoChain) chain.push(`atempo=${r}`);
    }
    if (clipTimelineDurSec > 0) {
      chain.push(`atrim=end=${clipTimelineDurSec.toFixed(6)}`);
      chain.push("asetpts=PTS-STARTPTS");
    }
    if (fadeIn > 0) chain.push(`afade=t=in:st=0:d=${fadeIn.toFixed(6)}`);
    if (fadeOut > 0 && clipTimelineDurSec > 0)
      chain.push(
        `afade=t=out:st=${Math.max(0, clipTimelineDurSec - fadeOut).toFixed(6)}:d=${fadeOut.toFixed(6)}`,
      );
    if (volDb !== 0) chain.push(`volume=${volDb.toFixed(4)}dB`);
    // adelay requires per-channel values
    chain.push(`adelay=${delayMs}|${delayMs}`);
    filterParts.push(
      `${labelIn}${chain.length ? chain.join(",") : "anull"}${labelOut}`,
    );
    outLabels.push(labelOut);
  });

  // Add a silent base track spanning the entire export duration so that
  // gaps between clips are emitted as silence and the mixed audio always
  // covers the full export range.
  const baseLabel = "[base]";
  filterParts.unshift(
    `anullsrc=r=48000:cl=stereo:d=${durationSec.toFixed(6)}${baseLabel}`,
  );
  outLabels.unshift(baseLabel);

  // Mix and confine to export duration
  if (outLabels.length === 1) {
    const single = outLabels[0];
    filterParts.push(
      `${single}atrim=end=${durationSec.toFixed(6)},asetpts=PTS-STARTPTS[aout]`,
    );
  } else {
    const mixLabel = "[amix]";
    filterParts.push(
      `${outLabels.join("")}amix=inputs=${outLabels.length}:normalize=0:duration=longest${mixLabel}`,
    );
    filterParts.push(
      `${mixLabel}atrim=end=${durationSec.toFixed(6)},asetpts=PTS-STARTPTS[aout]`,
    );
  }

  args.push("-filter_complex", filterParts.join(";"));
  args.push("-map", "[aout]");
  args.push("-ar", "48000");
  if (outFormat === "wav") {
    args.push("-c:a", "pcm_s16le");
  } else {
    args.push("-c:a", "libmp3lame", "-b:a", "192k");
  }
  args.push(outAbs);

  console.log(args, "args");

  await new Promise<void>((resolve, reject) => {
    const ff = spawn("ffmpeg", args);
    let stderr = "";
    ff.stderr.setEncoding("utf8");
    ff.stderr.on("data", (d) => {
      stderr += String(d);
    });
    ff.on("close", (code) => {
      console.log(code, "code");
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg failed (${code}): ${stderr}`));
    });
    ff.on("error", (err) => reject(err));
  });

  return outAbs;
}

// Compute a fully-qualified path inside the previews directory from an id or local path
async function getPreviewPath(
  idOrPath: string,
  options?: { ext?: string; ensureUnique?: boolean },
): Promise<string> {
  const { previewsAbs } = await ensurePreviewDir();
  const ensureUnique = options?.ensureUnique !== false;
  const desiredExt = (options?.ext || "mp4").replace(/^\./, "");

  const isAbsolutePath =
    typeof idOrPath === "string" && idOrPath.startsWith("/");

  let baseName: string;
  if (isAbsolutePath) {
    // If already inside previews dir, return as-is; otherwise move under previews by basename
    if (idOrPath === previewsAbs || idOrPath.startsWith(previewsAbs + "/")) {
      baseName = idOrPath.slice(previewsAbs.length + 1);
    } else {
      baseName = basename(idOrPath);
    }
  } else {
    baseName = String(idOrPath);
  }

  // Sanitize and ensure extension
  baseName =
    baseName.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") ||
    `preview_${Date.now()}`;

  const hasExt = !!extname(baseName);
  const desired = hasExt ? baseName : `${baseName}.${desiredExt}`;

  // If absolute input was already fully inside previewsAbs and had subdirs, keep subdir; else place at root
  let targetPath =
    isAbsolutePath && idOrPath.startsWith(previewsAbs + "/")
    ? join(previewsAbs, baseName)
    : join(previewsAbs, desired);

  if (!ensureUnique) return targetPath;

  // Ensure uniqueness by suffixing _N before extension
  if (!fs.existsSync(targetPath)) return targetPath;
  const stem = desired.replace(/\.[^.]+$/, "");
  const ext = extname(desired) || `.${desiredExt}`;
  let attempt = 1;
  while (fs.existsSync(join(previewsAbs, `${stem}_${attempt}${ext}`))) {
    attempt += 1;
  }
  targetPath = join(previewsAbs, `${stem}_${attempt}${ext}`);
  return targetPath;
}

// Streaming MP4 encoding (image2pipe)
type EncodeSession = {
  id: string;
  fps: number;
  outAbs: string;
  proc: import("child_process").ChildProcessWithoutNullStreams;
  closed: boolean;
};

const encodeSessions: Map<string, EncodeSession> = new Map();

// Advanced export encoder (codec/bitrate/format/resolution/filename)
type ExportVideoOpenOptions = {
  fps: number;
  width: number;
  height: number;
  filename: string; // absolute path preferred
  format?: "mp4" | "mov" | "mkv" | "webm";
  codec?: "h264" | "hevc" | "prores" | "vp9" | "av1";
  preset?:
    | "ultrafast"
    | "superfast"
    | "veryfast"
    | "faster"
    | "fast"
    | "medium"
    | "slow"
    | "slower"
    | "veryslow";
  crf?: number; // quality mode for supported codecs
  bitrate?: string; // e.g. '8M'
  resolution?: { width: number; height: number };
  alpha?: boolean; // preserve transparency
  audioPath?: string; // optional absolute path to audio to mux
};

type AdvancedEncodeSession = EncodeSession & {
  options: ExportVideoOpenOptions;
  stderr: string;
  args: string[];
};

async function exportVideoOpen(
  options: ExportVideoOpenOptions,
): Promise<{ sessionId: string; outAbs: string }> {
  const outAbs = options.filename;
  const outDir = dirname(outAbs);
  try {
    await fsp.mkdir(outDir, { recursive: true });
  } catch {}

  const fps = Math.max(1, Math.trunc(Number(options.fps) || 24));
  const targetW = Math.max(
    1,
    Math.trunc(options.resolution?.width || options.width),
  );
  const targetH = Math.max(
    1,
    Math.trunc(options.resolution?.height || options.height),
  );
  const format = (options.format ||
    (outAbs.split(".").pop() as any) ||
    "mp4") as Required<ExportVideoOpenOptions>["format"];
  const codec = (options.codec ||
    (format === "webm"
      ? "vp9"
      : "h264")) as Required<ExportVideoOpenOptions>["codec"];
  const preset = options.preset || "medium";
  const wantsAlpha = !!options.alpha;

  // Validate alpha/container/codec compatibility early
  if (wantsAlpha) {
    if (codec === "h264" || codec === "hevc") {
      throw new Error(
        'Alpha transparency is not supported with H.264/HEVC. Use codec "prores" (format mov) or "vp9" (format webm/mkv), or "av1" (format webm/mkv).',
      );
    }
    if (codec === "prores" && format !== "mov") {
      throw new Error('ProRes with alpha requires format "mov".');
    }
    if (codec === "vp9" && !(format === "webm" || format === "mkv")) {
      throw new Error('VP9 with alpha requires format "webm" or "mkv".');
    }
    if (codec === "av1" && (format === "mp4" || format === "mov")) {
      throw new Error(
        'AV1 with alpha is recommended in "webm" or "mkv" containers, not mp4/mov.',
      );
    }
  }

  const args: string[] = [
    "-y",
    "-f",
    "image2pipe",
    "-framerate",
    String(fps),
    "-vcodec",
    "png",
    "-i",
    "pipe:0",
  ];
  const hasAudio =
    typeof options.audioPath === "string" && options.audioPath.length > 0;
  if (hasAudio) {
    args.push("-i", options.audioPath as string);
  }

  const useCodec = codec;
  switch (useCodec) {
    case "h264":
      if (wantsAlpha) {
        // H.264 does not support alpha in common containers; advise using mov+prores or webm+vp9
        throw new Error(
          'Alpha transparency is not supported with H.264. Use codec "prores" with format "mov" or codec "vp9" with format "webm".',
        );
      }
      args.push("-c:v", "libx264", "-preset", preset);
      if (typeof options.crf === "number") {
        args.push(
          "-crf",
          String(Math.max(0, Math.min(51, Math.trunc(options.crf)))),
        );
        args.push("-pix_fmt", "yuv420p");
      } else if (options.bitrate) {
        args.push(
          "-b:v",
          options.bitrate,
          "-maxrate",
          options.bitrate,
          "-bufsize",
          options.bitrate,
        );
        args.push("-pix_fmt", "yuv420p");
      } else {
        args.push("-crf", "18", "-pix_fmt", "yuv420p");
      }
      if (format === "mp4" || format === "mov") {
        args.push("-movflags", "+faststart");
      }
      break;
    case "hevc":
      if (wantsAlpha) {
        // HEVC alpha is not widely supported; recommend alternatives
        throw new Error(
          'Alpha transparency is not reliably supported with HEVC. Use codec "prores" with format "mov" or codec "vp9" with format "webm".',
        );
      }
      args.push("-c:v", "libx265", "-preset", preset);
      if (format === "mp4" || format === "mov") {
        args.push("-tag:v", "hvc1");
      }
      if (typeof options.crf === "number") {
        args.push(
          "-crf",
          String(Math.max(0, Math.min(51, Math.trunc(options.crf)))),
        );
        args.push("-pix_fmt", "yuv420p");
      } else if (options.bitrate) {
        args.push("-b:v", options.bitrate);
        args.push("-pix_fmt", "yuv420p");
      } else {
        args.push("-crf", "20", "-pix_fmt", "yuv420p");
      }
      break;
    case "prores":
      if (wantsAlpha) {
        // ProRes 4444 (profile 4) with alpha
        args.push("-c:v", "prores_ks", "-profile:v", "4");
        args.push("-pix_fmt", "yuva444p10le");
        if (options.bitrate) args.push("-b:v", options.bitrate);
      } else {
        args.push("-c:v", "prores_ks", "-profile:v", "3");
        if (options.bitrate) args.push("-b:v", options.bitrate);
        args.push("-pix_fmt", "yuv422p10le");
      }
      break;
    case "vp9":
      args.push("-c:v", "libvpx-vp9", "-row-mt", "1");
      if (typeof options.crf === "number") {
        args.push(
          "-crf",
          String(Math.max(0, Math.min(63, Math.trunc(options.crf)))),
        );
        args.push("-b:v", "0");
      } else if (options.bitrate) {
        args.push("-b:v", options.bitrate);
      } else {
        args.push("-crf", "32", "-b:v", "0");
      }
      args.push("-pix_fmt", wantsAlpha ? "yuva420p" : "yuv420p");
      break;
    case "av1":
      args.push("-c:v", "libaom-av1", "-cpu-used", "6");
      if (typeof options.crf === "number") {
        args.push(
          "-crf",
          String(Math.max(0, Math.min(63, Math.trunc(options.crf)))),
        );
        args.push("-b:v", "0");
      } else if (options.bitrate) {
        args.push("-b:v", options.bitrate);
      } else {
        args.push("-crf", "28", "-b:v", "0");
      }
      args.push("-pix_fmt", wantsAlpha ? "yuva420p" : "yuv420p");
      break;
  }

  // Ensure even dimensions for 4:2:0 pixel formats (required by encoders)
  const isChroma420 = useCodec !== "prores";
  const finalW = isChroma420 ? targetW & ~1 : targetW;
  const finalH = isChroma420 ? targetH & ~1 : targetH;
  if (finalW !== options.width || finalH !== options.height) {
    args.push("-vf", `scale=${finalW}:${finalH}:flags=lanczos`);
  }

  if ((format === "mp4" || format === "mov") && !wantsAlpha) {
    args.push("-movflags", "+faststart");
  }

  // Audio codec settings and stream mapping
  if (hasAudio) {
    if (format === "mp4" || format === "mov") {
      args.push("-c:a", "aac", "-b:a", "192k", "-ar", "48000");
    } else if (format === "webm" || format === "mkv") {
      args.push("-c:a", "libopus", "-b:a", "128k", "-ar", "48000");
    } else {
      args.push("-c:a", "aac", "-b:a", "192k", "-ar", "48000");
    }
  }

  // Force container format to avoid ffmpeg guessing when filename has no extension
  const container = (() => {
    switch (format) {
      case "mp4":
        return "mp4";
      case "mov":
        return "mov";
      case "webm":
        return "webm";
      case "mkv":
        return "matroska";
      default:
        return undefined;
    }
  })();
  if (container) {
    args.push("-f", container);
  }

  args.push(outAbs);

  const proc = spawn("ffmpeg", args);
  const sessionId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const session: AdvancedEncodeSession = {
    id: sessionId,
    fps,
    outAbs,
    proc,
    closed: false,
    options,
    stderr: "",
    args,
  } as AdvancedEncodeSession;
  encodeSessions.set(sessionId, session);

  const markClosed = () => {
    const s = encodeSessions.get(sessionId);
    if (s) s.closed = true;
  };
  proc.on("close", markClosed);
  proc.on("exit", markClosed);
  proc.on("error", markClosed);

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const onReady = () => {
      if (settled) return;
      settled = true;
      proc.stderr?.off("data", onData);
      resolve();
    };
    const onData = (chunk: any) => {
      const s = String(chunk || "");
      session.stderr += s;
      if (
        s.includes("Input #0") ||
        s.includes("image2pipe") ||
        s.includes("pipe:0")
      )
        onReady();
    };
    const onEarlyExit = () => {
      if (settled) return;
      settled = true;
      proc.stderr?.off("data", onData);
      reject(new Error("ffmpeg exited before initialization"));
    };
    proc.stderr?.on("data", onData);
    proc.once("close", onEarlyExit);
    proc.once("error", onEarlyExit);
    setTimeout(() => {
      if (!settled) onReady();
    }, 250);
  });

  return { sessionId, outAbs };
}

async function exportVideoAppend(
  sessionId: string,
  data: ArrayBuffer | Uint8Array | string,
): Promise<void> {
  const s = encodeSessions.get(sessionId);
  if (!s || s.closed)
    throw new Error("exportVideoAppend: invalid or closed session");

  let buffer: Buffer;
  if (typeof data === "string") {
    const m = /^data:image\/png;base64,(.*)$/i.exec(data.trim());
    if (m && m[1]) buffer = Buffer.from(m[1], "base64");
    else
      throw new Error(
        "exportVideoAppend: expected PNG data URL or binary buffer",
      );
  } else if (data instanceof Uint8Array) {
    buffer = Buffer.from(data);
  } else {
    buffer = Buffer.from(new Uint8Array(data));
  }

  if (s.proc.exitCode !== null || s.proc.killed || s.proc.stdin.destroyed) {
    const stderr = (s as any).stderr || "";
    const a = (s as any).args as string[] | undefined;
    const msg =
      `exportVideoAppend: encoder closed.\nargs: ${a ? a.join(" ") : "(unknown)"}\n` +
      (stderr ? `ffmpeg stderr:\n${stderr}` : "");
    throw new Error(msg);
  }
  await new Promise<void>((resolve, reject) => {
    let finished = false;
    const onError = (err: any) => {
      if (finished) return;
      finished = true;
      const stderr = (s as any).stderr || "";
      const a = (s as any).args as string[] | undefined;
      const msg =
        `${String(err || "stdin error")}\nargs: ${a ? a.join(" ") : "(unknown)"}\n` +
        (stderr ? `ffmpeg stderr:\n${stderr}` : "");
      reject(new Error(msg));
    };
    s.proc.stdin.once("error", onError);
    const ok = s.proc.stdin.write(buffer, (err) => {
      s.proc.stdin.off("error", onError);
      if (err) onError(err);
      else resolve();
    });
    if (!ok) {
      s.proc.stdin.once("drain", () => {
        s.proc.stdin.off("error", onError);
        if (!finished) resolve();
      });
    }
  });
}

async function exportVideoClose(sessionId: string): Promise<string> {
  const s = encodeSessions.get(sessionId);
  if (!s) throw new Error("exportVideoClose: invalid session");
  if (s.closed) {
    encodeSessions.delete(sessionId);
    return s.outAbs;
  }
  return await new Promise<string>((resolve, reject) => {
    let stderr = "";
    s.proc.stderr.setEncoding("utf8");
    s.proc.stderr.on("data", (d) => {
      stderr += String(d);
    });
    s.proc.on("close", (code) => {
      encodeSessions.delete(sessionId);
      if (code === 0) resolve(s.outAbs);
      else reject(new Error(`ffmpeg failed (${code}): ${stderr}`));
    });
    s.proc.on("error", (err) => {
      encodeSessions.delete(sessionId);
      reject(err);
    });
    try {
      s.proc.stdin.end();
    } catch {}
  });
}

async function exportVideoAbort(sessionId: string): Promise<void> {
  const s = encodeSessions.get(sessionId);
  if (!s) return;
  try {
    s.proc.stdin.destroy();
  } catch {}
  try {
    s.proc.kill("SIGKILL");
  } catch {}
  try {
    await fsp.unlink(s.outAbs);
  } catch {}
  encodeSessions.delete(sessionId);
}
// Save a sequence of PNG frames to a temporary folder, encode to mp4 with ffmpeg, then cleanup temp frames
async function savePreviewVideoFromFrames(
  frames: Array<ArrayBuffer | Uint8Array | string>,
  options?: { fps?: number; fileNameHint?: string },
): Promise<string> {
  const { previewsAbs } = await ensurePreviewDir();
  const fps =
    options?.fps && Number.isFinite(options.fps)
      ? Math.max(1, Math.trunc(options.fps as number))
      : 24;

  // Create a unique temp directory under previews to hold frame PNGs
  const tempDirBase = `frames_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const tempDir = join(previewsAbs, tempDirBase);
  await fsp.mkdir(tempDir, { recursive: true });

  // Normalize buffers and write numbered PNGs
  let index = 0;
  for (const data of frames) {
    let buffer: Buffer;
    if (typeof data === "string") {
      const m = /^data:image\/png;base64,(.*)$/i.exec(data.trim());
      if (m && m[1]) {
        buffer = Buffer.from(m[1], "base64");
      } else {
        throw new Error(
          "savePreviewVideoFromFrames: expected PNG data URL or binary buffer",
        );
      }
    } else if (data instanceof Uint8Array) {
      buffer = Buffer.from(data);
    } else {
      buffer = Buffer.from(new Uint8Array(data));
    }
    const fileName = `frame_${String(index).padStart(6, "0")}.png`;
    await fsp.writeFile(join(tempDir, fileName), buffer);
    index += 1;
  }

  if (index === 0) {
    // Cleanup temp dir if nothing was written
    try {
      await fsp.rm(tempDir, { recursive: true, force: true });
    } catch {}
    throw new Error("savePreviewVideoFromFrames: no frames provided");
  }

  // Choose output file name (ensure uniqueness)
  const baseName = (options?.fileNameHint || `preview_${Date.now()}`)
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const desired = baseName.toLowerCase().endsWith(".mp4")
    ? baseName
    : `${baseName}.mp4`;
  let outName = desired;
  let attempt = 0;
  while (fs.existsSync(join(previewsAbs, outName))) {
    attempt += 1;
    const stem = desired.replace(/\.[^.]+$/, "");
    outName = `${stem}_${attempt}.mp4`;
  }
  const outAbs = join(previewsAbs, outName);

  // Build ffmpeg args to read numbered PNGs and encode to H.264
  const inputPattern = join(tempDir, "frame_%06d.png");
  const args = [
    "-y",
    "-framerate",
    String(fps),
    "-i",
    inputPattern,
    "-c:v",
    "libx264",
    "-preset",
    "fast",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    outAbs,
  ];

  // Run ffmpeg
  await new Promise<void>((resolve, reject) => {
    const ff = spawn("ffmpeg", args);
    let stderr = "";
    ff.stderr.setEncoding("utf8");
    ff.stderr.on("data", (d) => {
      stderr += String(d);
    });
    ff.on("close", async (code) => {
      // Cleanup temp frames directory regardless of success/failure
      try {
        await fsp.rm(tempDir, { recursive: true, force: true });
      } catch {}
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`ffmpeg failed (${code}): ${stderr}`));
      }
    });
    ff.on("error", async (err) => {
      try {
        await fsp.rm(tempDir, { recursive: true, force: true });
      } catch {}
      reject(err);
    });
  });

  return outAbs;
}

async function listConvertedMedia(): Promise<ConvertedMediaItem[]> {
  const { symlinksAbs, proxyAbs } = await ensureMediaDirs();
  const proxyIndex = await loadProxyIndex();
  
  let entries: import("node:fs").Dirent[] = [];
  try {
    entries = (await fsp.readdir(symlinksAbs, {
      withFileTypes: true,
    })) as unknown as import("node:fs").Dirent[];
  } catch {
    entries = [] as any;
  }

  const items: ConvertedMediaItem[] = [];
  for (const e of entries) {
    const name = e.name;
    const ext = getLowercaseExtension(name);
    if (!isSupportedExt(ext)) continue;
    
    let absPath = join(symlinksAbs, name);
    let hasProxy = false;
    
    // Check if this file has a proxy
    if (proxyIndex[name]) {
      const proxyPath = join(proxyAbs, proxyIndex[name]);
      if (fs.existsSync(proxyPath)) {
        absPath = proxyPath;
        hasProxy = true;
      }
    }
    
    let dateAddedMs = Date.now();
    try {
      const st = await fsp.lstat(join(symlinksAbs, name));
      dateAddedMs =
        st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
    } catch {}

    const assetUrl = pathToFileURL(absPath).href;
    const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
      ? "video"
      : IMAGE_EXTS.has(ext)
        ? "image"
        : AUDIO_EXTS.has(ext)
          ? "audio"
          : "video";

    items.push({ name, absPath, assetUrl, dateAddedMs, type, hasProxy });
  }

  items.sort((a, b) =>
    a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
  );
  return items;
}

// List generated media (model generations, engine outputs, etc.) from engine_results
// under relevant cache roots. We combine:
// - The configured cache path from getCachePath()
// - Sibling apex-cache and apex-cache-remote directories (typically under userData)
// and aggregate any engine_results found there.
async function listGeneratedMedia(): Promise<ConvertedMediaItem[]> {
  try {
    const cacheRes = await getCachePath();
    const cachePath =
      cacheRes?.success && cacheRes.data?.cache_path
        ? cacheRes.data.cache_path
        : null;

    const userDataRes = await getUserDataPath();
    const userDataDir =
      userDataRes?.success && userDataRes.data?.user_data
        ? userDataRes.data.user_data
        : null;

    const roots = new Set<string>();
    if (cachePath) {
      roots.add(cachePath);
    }
    if (userDataDir) {
      // Always include apex-cache and apex-cache-remote under Electron's userData directory.
      roots.add(join(userDataDir, "apex-cache"));
      roots.add(join(userDataDir, "apex-cache-remote"));
    }

    const collected: ConvertedMediaItem[] = [];

    for (const root of roots) {
      const engineResultsAbs = join(root, "engine_results");
      if (!fs.existsSync(engineResultsAbs)) {
        continue;
      }

      let entries: import("node:fs").Dirent[] = [];
      try {
        entries = (await fsp.readdir(engineResultsAbs, {
          withFileTypes: true,
        })) as unknown as import("node:fs").Dirent[];
      } catch {
        entries = [] as any;
      }

      if (entries.length === 0) continue;

      for (const e of entries) {
        const entryName = e.name;
        const entryPath = join(engineResultsAbs, entryName);

        if (e.isDirectory()) {
          // Engine results are typically stored in subdirectories per job; scan one level deep.
          let jobEntries: import("node:fs").Dirent[] = [];
          try {
            jobEntries = (await fsp.readdir(entryPath, {
              withFileTypes: true,
            })) as unknown as import("node:fs").Dirent[];
          } catch {
            jobEntries = [] as any;
          }

          for (const je of jobEntries) {
            const name = je.name;
            const ext = getLowercaseExtension(name);
            if (!isSupportedExt(ext)) continue;

            // Only keep final result media, skip intermediate preview_* files.
            const stem = name.replace(/\.[^.]+$/, "");
            if (!stem.toLowerCase().startsWith("result")) continue;

            const absPath = join(entryPath, name);

            let dateAddedMs = Date.now();
            try {
              const st = await fsp.lstat(absPath);
              dateAddedMs =
                st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
            } catch {}

            const assetUrl = pathToFileURL(absPath).href;
            const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
              ? "video"
              : IMAGE_EXTS.has(ext)
                ? "image"
                : AUDIO_EXTS.has(ext)
                  ? "audio"
                  : "video";

            const displayName = `${entryName}/${name}`;
            collected.push({ name: displayName, absPath, assetUrl, dateAddedMs, type });
          }
        } else {
          // Also allow flat files directly under engine_results, just in case.
          const name = entryName;
          const ext = getLowercaseExtension(name);
          if (!isSupportedExt(ext)) continue;

          const stem = name.replace(/\.[^.]+$/, "");
          if (!stem.toLowerCase().startsWith("result")) continue;

          const absPath = entryPath;

          let dateAddedMs = Date.now();
          try {
            const st = await fsp.lstat(absPath);
            dateAddedMs =
              st.birthtime?.getTime?.() ?? st.mtime?.getTime?.() ?? dateAddedMs;
          } catch {}

          const assetUrl = pathToFileURL(absPath).href;
          const type: ConvertedMediaItem["type"] = VIDEO_EXTS.has(ext)
            ? "video"
            : IMAGE_EXTS.has(ext)
              ? "image"
              : AUDIO_EXTS.has(ext)
                ? "audio"
                : "video";

          collected.push({ name, absPath, assetUrl, dateAddedMs, type });
        }
      }
    }

    if (collected.length === 0) return [];

    // Deduplicate by absolute path in case multiple roots overlap
    const byPath = new Map<string, ConvertedMediaItem>();
    for (const it of collected) {
      if (!byPath.has(it.absPath)) {
        byPath.set(it.absPath, it);
      }
    }

    const unique = Array.from(byPath.values());
    // Most recent first for generations
    unique.sort((a, b) => (b.dateAddedMs ?? 0) - (a.dateAddedMs ?? 0));
    return unique;
  } catch {
    return [];
  }
}

async function importMediaPaths(
  inputAbsPaths: string[],
  resolution?: string,
): Promise<number> {
  const { symlinksAbs } = await ensureMediaDirs();
  let count = 0;
  for (const srcAbs of inputAbsPaths) {
    const fileName = srcAbs.split(/[/\\]/g).pop();
    if (!fileName) continue;
    const ext = getLowercaseExtension(fileName);
    if (!isSupportedExt(ext)) continue;
    const uniqueName = ensureUniqueNameSync(symlinksAbs, fileName);
    const dstAbs = join(symlinksAbs, uniqueName);
    try {
      await fsp.rm(dstAbs, { force: true });
      await fsp.symlink(srcAbs, dstAbs);
    } catch (error) {
      console.error("Error creating symlink:", error);
      try {
        await fsp.copyFile(srcAbs, dstAbs);
      } catch (copyError) {
        console.error("Error copying file:", copyError);
        continue;
      }
    }
    count += 1;
  }
  return count;
}

async function ensureUniqueConvertedName(desiredName: string): Promise<string> {
  const { symlinksAbs } = await ensureMediaDirs();
  return ensureUniqueNameSync(symlinksAbs, desiredName);
}

// Reveal a media item in the OS file manager using its symlinked path.
async function revealMediaItemInFolder(fileName: string): Promise<void> {
  try {
    if (!fileName) return;
    const { symlinksAbs } = await ensureMediaDirs();
    const symlinkPath = join(symlinksAbs, fileName);
    // Prefer the original target path of the symlink (outside appdata), falling
    // back to the symlink itself if it is not a symlink for some reason.
    let targetPath = symlinkPath;
    try {
      const st = await fsp.lstat(symlinkPath);
      if (st.isSymbolicLink()) {
        targetPath = await fsp.realpath(symlinkPath);
      }
    } catch {
      // If the symlink entry is missing, just try with the constructed path
    }

    await ipcRenderer.invoke("files:reveal-in-folder", targetPath);
  } catch (e) {
    // Best-effort; log but don't surface to the UI
    // eslint-disable-next-line no-console
    console.error("revealMediaItemInFolder failed", e);
  }
}

// Reveal an arbitrary absolute path in the OS file manager.
async function revealPathInFolder(absPath: string): Promise<void> {
  try {
    if (!absPath) return;
    await ipcRenderer.invoke("files:reveal-in-folder", absPath);
  } catch (e) {
    // Best-effort; log but don't surface to the UI
    // eslint-disable-next-line no-console
    console.error("revealPathInFolder failed", e);
  }
}

function getPathForFile(file: File): string {
  return webUtils.getPathForFile(file);
}

async function createProxy(
  fileName: string,
  resolution: string = "480p",
): Promise<void> {
  const { symlinksAbs, proxyAbs } = await ensureMediaDirs();
  const symlinkPath = join(symlinksAbs, fileName);
  
  if (!fs.existsSync(symlinkPath)) {
    throw new Error("Source file not found");
  }
  
  // Resolve the actual file path from the symlink
  const realPath = await fsp.realpath(symlinkPath);
  
  // Generate proxy filename
  const baseName = basename(fileName, extname(fileName));
  const proxyFileName = `${baseName}_proxy.mp4`;
  const proxyPath = join(proxyAbs, proxyFileName);
  
  // Parse target height
  const parseHeight = (res: string): number => {
    const s = res.trim().toLowerCase();
    if (s === "720" || s === "720p" || s === "hd") return 720;
    if (s === "1080" || s === "1080p") return 1080;
    const raw = s.endsWith("p") ? s.slice(0, -1) : s;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : 480;
  };
  
  const targetHeight = parseHeight(resolution);
  
  return new Promise((resolve, reject) => {
    const args = [
      "-i",
      realPath,
      "-vf",
      `scale=-2:${targetHeight}`,
      "-r",
      "24",
      "-c:v",
      "libx264",
      "-preset",
      "medium",
      "-crf",
      "23",
      "-c:a",
      "aac",
      "-b:a",
      "128k",
      "-ar",
      "48000",
      "-y",
      proxyPath,
    ];
    
    const ffmpeg = spawn("ffmpeg", args);
    
    let stderr = "";
    ffmpeg.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    
    ffmpeg.on("close", async (code) => {
      if (code === 0) {
        const proxyIndex = await loadProxyIndex();
        proxyIndex[fileName] = proxyFileName;
        await saveProxyIndex(proxyIndex);
        resolve();
      } else {
        reject(new Error(`FFmpeg failed with code ${code}: ${stderr}`));
      }
    });
    
    ffmpeg.on("error", (err) => {
      reject(err);
    });
  });
}

async function removeProxy(fileName: string): Promise<void> {
  const { proxyAbs } = await ensureMediaDirs();
  const proxyIndex = await loadProxyIndex();
  
  if (proxyIndex[fileName]) {
    const proxyPath = join(proxyAbs, proxyIndex[fileName]);
    if (fs.existsSync(proxyPath)) {
      await fsp.rm(proxyPath, { force: true });
    }
    delete proxyIndex[fileName];
    await saveProxyIndex(proxyIndex);
  }
}

// Config API functions
interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

async function getBackendUrl(): Promise<ConfigResponse<{ url: string }>> {
  return await ipcRenderer.invoke("backend:get-url");
}

async function setBackendUrl(
  url: string,
): Promise<ConfigResponse<{ url: string }>> {
  return await ipcRenderer.invoke("backend:set-url", url);
}

async function getBackendIsRemote(): Promise<
  ConfigResponse<{ isRemote: boolean }>
> {
  return await ipcRenderer.invoke("backend:is-remote");
}

async function getFileShouldUpload(
  inputPath: string,
): Promise<ConfigResponse<{ shouldUpload: boolean }>> {
  return await ipcRenderer.invoke("files:should-upload", inputPath);
}

async function getFileIsUploaded(
  inputPath: string,
): Promise<ConfigResponse<{ isUploaded: boolean }>> {
  return await ipcRenderer.invoke("files:is-uploaded", inputPath);
}

async function getHomeDir(): Promise<ConfigResponse<{ home_dir: string }>> {
  return await ipcRenderer.invoke("config:get-home-dir");
}

async function setHomeDir(
  homeDir: string,
): Promise<ConfigResponse<{ home_dir: string }>> {
  return await ipcRenderer.invoke("config:set-home-dir", homeDir);
}

async function getUserDataPath(): Promise<ConfigResponse<{ user_data: string }>> {
  return await ipcRenderer.invoke("config:get-user-data-path");
}

async function getTorchDevice(): Promise<ConfigResponse<{ device: string }>> {
  return await ipcRenderer.invoke("config:get-torch-device");
}

async function setTorchDevice(
  device: string,
): Promise<ConfigResponse<{ device: string }>> {
  return await ipcRenderer.invoke("config:set-torch-device", device);
}

async function getCachePath(): Promise<ConfigResponse<{ cache_path: string }>> {
  return await ipcRenderer.invoke("config:get-cache-path");
}

async function setCachePath(
  cachePath: string,
): Promise<ConfigResponse<{ cache_path: string }>> {
  return await ipcRenderer.invoke("config:set-cache-path", cachePath);
}

// Preprocessor API functions
async function listPreprocessors(
  checkDownloaded: boolean = true,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:list", checkDownloaded);
}

async function deletePreprocessor(name: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:delete", name);
}

// Components API functions
async function downloadComponents(
  paths: string[],
  savePath?: string,
  jobId?: string,
): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  return await ipcRenderer.invoke(
    "components:download",
    paths,
    savePath,
    jobId,
  );
}

async function deleteComponent(
  targetPath: string,
): Promise<ConfigResponse<{ status: string; path: string }>> {
  return await ipcRenderer.invoke("components:delete", targetPath);
}

async function getComponentsStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:status", jobId);
}

async function cancelComponents(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:cancel", jobId);
}

async function connectComponentsWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("components:connect-ws", jobId);
}

async function disconnectComponentsWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("components:disconnect-ws", jobId);
}

function onComponentsWebSocketUpdate(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`components:ws-update:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`components:ws-update:${jobId}`, listener);
}

function onComponentsWebSocketStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`components:ws-status:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`components:ws-status:${jobId}`, listener);
}

function onComponentsWebSocketError(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`components:ws-error:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`components:ws-error:${jobId}`, listener);
}

async function getPreprocessor(name: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:get", name);
}

async function downloadPreprocessor(
  name: string,
  jobId?: string,
): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  return await ipcRenderer.invoke("preprocessor:download", name, jobId);
}

async function createMask(request: {
  input_path: string;
  frame_number?: number;
  tool: string;
  points?: Array<{ x: number; y: number }>;
  point_labels?: Array<number>;
  box?: { x1: number; y1: number; x2: number; y2: number };
  multimask_output?: boolean;
  simplify_tolerance?: number;
  model_type?: string;
  id?: string;
}): Promise<
  ConfigResponse<{
    status: string;
    contours?: Array<Array<number>>;
    message?: string;
  }>
> {
  // check if input_path is a file url
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  return await ipcRenderer.invoke("mask:create", request);
}

type MaskTrackStreamEvent =
  | { frame_number: number; contours: Array<Array<number>> }
  | { status: "error"; error: string };

async function trackMask(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: "forward" | "backward" | "both";
  model_type?: string;
}): Promise<ReadableStream<MaskTrackStreamEvent>> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  const { streamId } = (await ipcRenderer.invoke("mask:track", request)) as {
    streamId: string;
  };
  const parsedStream = new ReadableStream<MaskTrackStreamEvent>({
    start(controller) {
      const onChunk = (_e: unknown, data: any) => {
        controller.enqueue(data as MaskTrackStreamEvent);
      };
      const onError = (_e: unknown, err: any) => {
        controller.enqueue({
          status: "error",
          error: err?.message || "Unknown error",
        });
      };
      const onEnd = () => {
        controller.close();
        ipcRenderer.removeAllListeners(`mask:track:chunk:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track:error:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track:end:${streamId}`);
      };
      ipcRenderer.on(`mask:track:chunk:${streamId}`, onChunk);
      ipcRenderer.on(`mask:track:error:${streamId}`, onError);
      ipcRenderer.on(`mask:track:end:${streamId}`, onEnd);
    },
    cancel() {
      ipcRenderer.invoke("mask:track:cancel", streamId).catch(() => {});
      ipcRenderer.removeAllListeners(`mask:track:chunk:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track:error:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track:end:${streamId}`);
    },
  });

  return parsedStream;
}

// Lower-level helpers to assemble stream from renderer (avoid returning ReadableStream across contextBridge)
async function startMaskTrack(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: "forward" | "backward" | "both";
  model_type?: string;
}): Promise<{ streamId: string }> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  
  return await ipcRenderer.invoke("mask:track", request);
}

function onMaskTrackChunk(
  streamId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`mask:track:chunk:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track:chunk:${streamId}`, listener);
}

function onMaskTrackError(
  streamId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`mask:track:error:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track:error:${streamId}`, listener);
}

function onMaskTrackEnd(streamId: string, callback: () => void): () => void {
  const listener = () => callback();
  ipcRenderer.on(`mask:track:end:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track:end:${streamId}`, listener);
}

async function cancelMaskTrack(streamId: string): Promise<void> {
  await ipcRenderer.invoke("mask:track:cancel", streamId);
}

// Shapes tracking (streaming shapeBounds)
type ShapeTrackStreamEvent =
  | { frame_number: number; shapeBounds?: any }
  | { status: "error"; error: string };

async function trackShapes(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: "forward" | "backward" | "both";
  model_type?: string;
}): Promise<ReadableStream<ShapeTrackStreamEvent>> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  const { streamId } = (await ipcRenderer.invoke(
    "mask:track-shapes",
    request,
  )) as { streamId: string };
  const parsedStream = new ReadableStream<ShapeTrackStreamEvent>({
    start(controller) {
      const onChunk = (_e: unknown, data: any) => {
        controller.enqueue(data as ShapeTrackStreamEvent);
      };
      const onError = (_e: unknown, err: any) => {
        controller.enqueue({
          status: "error",
          error: err?.message || "Unknown error",
        });
      };
      const onEnd = () => {
        controller.close();
        ipcRenderer.removeAllListeners(`mask:track-shapes:chunk:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track-shapes:error:${streamId}`);
        ipcRenderer.removeAllListeners(`mask:track-shapes:end:${streamId}`);
      };
      ipcRenderer.on(`mask:track-shapes:chunk:${streamId}`, onChunk);
      ipcRenderer.on(`mask:track-shapes:error:${streamId}`, onError);
      ipcRenderer.on(`mask:track-shapes:end:${streamId}`, onEnd);
    },
    cancel() {
      ipcRenderer.invoke("mask:track:cancel", streamId).catch(() => {});
      ipcRenderer.removeAllListeners(`mask:track-shapes:chunk:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track-shapes:error:${streamId}`);
      ipcRenderer.removeAllListeners(`mask:track-shapes:end:${streamId}`);
    },
  });
  return parsedStream;
}

async function startMaskTrackShapes(request: {
  id: string;
  input_path: string;
  frame_start: number;
  anchor_frame?: number;
  frame_end: number;
  direction?: "forward" | "backward" | "both";
  model_type?: string;
  shape_type?: string;
}): Promise<{ streamId: string }> {
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  return await ipcRenderer.invoke("mask:track-shapes", request);
}

function onMaskTrackShapesChunk(
  streamId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`mask:track-shapes:chunk:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track-shapes:chunk:${streamId}`, listener);
}

function onMaskTrackShapesError(
  streamId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`mask:track-shapes:error:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track-shapes:error:${streamId}`, listener);
}

function onMaskTrackShapesEnd(
  streamId: string,
  callback: () => void,
): () => void {
  const listener = () => callback();
  ipcRenderer.on(`mask:track-shapes:end:${streamId}`, listener);
  return () =>
    ipcRenderer.removeListener(`mask:track-shapes:end:${streamId}`, listener);
}

async function runPreprocessor(request: {
  preprocessor_name: string;
  input_path: string;
  job_id?: string;
  download_if_needed?: boolean;
  params?: Record<string, any>;
  start_frame?: number;
  end_frame?: number;
}): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  // check if input_path is a file url
  if (request.input_path.startsWith("file://")) {
    request.input_path = fileURLToPath(request.input_path);
  }
  return await ipcRenderer.invoke("preprocessor:run", request);
}

async function getPreprocessorStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:status", jobId);
}

async function getPreprocessorResult(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:result", jobId);
}

async function connectPreprocessorWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:connect-ws", jobId);
}

async function disconnectPreprocessorWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:disconnect-ws", jobId);
}

function onPreprocessorWebSocketUpdate(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-update:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`preprocessor:ws-update:${jobId}`, listener);
}

function onPreprocessorWebSocketStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-status:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`preprocessor:ws-status:${jobId}`, listener);
}

function onPreprocessorWebSocketError(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-error:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`preprocessor:ws-error:${jobId}`, listener);
}

async function cancelPreprocessor(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:cancel", jobId);
}

// Unified job helpers
async function jobStatus(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:status", jobId);
}

async function jobCancel(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:cancel", jobId);
}

// Ray job inspection helpers (aggregated across subsystems)
async function listRayJobs(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("ray:jobs:list");
}

async function getRayJob(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("ray:jobs:get", jobId);
}

async function cancelRayJob(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("ray:jobs:cancel", jobId);
}

async function cancelAllRayJobs(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("ray:jobs:cancel-all");
}

// Unified WebSocket helpers (renderer-wide API)
async function wsConnect(
  key: string,
  pathOrUrl: string,
): Promise<ConfigResponse<{ key: string }>> {
  return await ipcRenderer.invoke("ws:connect", { key, pathOrUrl });
}

async function wsDisconnect(
  key: string,
): Promise<ConfigResponse<{ message: string }>> {
  return await ipcRenderer.invoke("ws:disconnect", key);
}

async function wsStatus(
  key: string,
): Promise<ConfigResponse<{ key: string; connected: boolean }>> {
  return await ipcRenderer.invoke("ws:status", key);
}

function onWsUpdate(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-update:${key}`;
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

function onWsStatus(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-status:${key}`;
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

function onWsError(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-error:${key}`;
  const listener = (_event: any, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

function pathToFileURLString(path: string): string {
  const fileUrl = pathToFileURL(path);
  return fileUrl.href;
}

// System API functions
async function getSystemMemory(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("system:memory");
}

// Manifest API functions
async function listManifestModelTypes(): Promise<ConfigResponse<any>> {
  const res = await ipcRenderer.invoke("manifest:types");
  return res;
}

async function listManifests(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:list");
}

async function listManifestsByModel(
  model: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:list-by-model", model);
}

async function listManifestsByType(
  modelType: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:list-by-type", modelType);
}

async function listManifestsByModelAndType(
  model: string,
  modelType: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke(
    "manifest:list-by-model-and-type",
    model,
    modelType,
  );
}

async function getManifest(manifestId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:get", manifestId);
}

// Fetch a specific part of a manifest (dot-separated path)
async function getManifestPart<T = any>(
  manifestId: string,
  pathDot?: string,
): Promise<ConfigResponse<T>> {
  const qp =
    typeof pathDot === "string" && pathDot.length > 0 ? pathDot : undefined;
  return await ipcRenderer.invoke("manifest:get-part", manifestId, qp);
}

async function validateAndRegisterCustomModelPath(
  manifestId: string,
  componentIndex: number,
  name: string | undefined,
  path: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:validate-custom-model-path", {
    manifest_id: manifestId,
    component_index: componentIndex,
    name,
    path,
  });
}

async function deleteCustomModelPath(
  manifestId: string,
  componentIndex: number,
  path: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:delete-custom-model-path", {
    manifest_id: manifestId,
    component_index: componentIndex,
    path,
  });
}

async function updateManifestLoraScale(
  manifestId: string,
  loraIndex: number,
  scale: number,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:update-lora-scale", {
    manifest_id: manifestId,
    lora_index: loraIndex,
    scale,
  });
}

async function updateManifestLoraName(
  manifestId: string,
  loraIndex: number,
  name: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:update-lora-name", {
    manifest_id: manifestId,
    lora_index: loraIndex,
    name,
  });
}

async function deleteManifestLora(
  manifestId: string,
  loraIndex: number,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:delete-lora", {
    manifest_id: manifestId,
    lora_index: loraIndex,
  });
}

// Engine API functions
function _normalizeInputsForIpc(obj: any): any {
  if (obj == null) return obj;
  if (typeof obj === "string") {
    if (obj.startsWith("file://")) {
      try {
        return fileURLToPath(obj);
      } catch {
        return obj;
      }
    }
    return obj;
  }
  if (Array.isArray(obj)) return obj.map(_normalizeInputsForIpc);
  if (typeof obj === "object") {
    const out: any = {};
    for (const [k, v] of Object.entries(obj))
      out[k] = _normalizeInputsForIpc(v);
    return out;
  }
  return obj;
}

async function runEngine(request: {
  manifest_id?: string;
  yaml_path?: string;
  inputs: Record<string, any>;
  selected_components?: Record<string, any>;
  job_id?: string;
}): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  const payload = { 
    ...request, 
    inputs: _normalizeInputsForIpc(request.inputs || {}),
    selected_components: _normalizeInputsForIpc(
      request.selected_components || {},
    ),
  };
  return await ipcRenderer.invoke("engine:run", payload);
}

async function getEngineStatus(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("engine:status", jobId);
}

async function getEngineResult(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("engine:result", jobId);
}

async function cancelEngine(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("engine:cancel", jobId);
}

// Postprocessor API functions
type RunPostprocessorRequest = {
  method: "frame-interpolate";
  input_path: string;
  target_fps: number;
  job_id?: string;
  exp?: number;
  scale?: number;
};

async function runPostprocessor(
  request: RunPostprocessorRequest,
): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  if (request.input_path.startsWith("file://")) {
    try {
      request.input_path = fileURLToPath(request.input_path);
    } catch {}
  }
  return await ipcRenderer.invoke("postprocessor:run", request);
}

async function getPostprocessorStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:status", jobId);
}

async function cancelPostprocessor(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:cancel", jobId);
}

async function connectPostprocessorWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await wsConnect(`postprocessor:${jobId}`, `/ws/job/${jobId}`);
}

async function disconnectPostprocessorWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await wsDisconnect(`postprocessor:${jobId}`);
}

function onPostprocessorWebSocketUpdate(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsUpdate(`postprocessor:${jobId}`, callback);
}

function onPostprocessorWebSocketStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsStatus(`postprocessor:${jobId}`, callback);
}

function onPostprocessorWebSocketError(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsError(`postprocessor:${jobId}`, callback);
}

async function startUnifiedDownload(request: {
  item_type: "component" | "lora" | "preprocessor";
  source: string | string[];
  save_path?: string;
  job_id?: string;
  manifest_id?: string;
  lora_name?: string;
}): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  return await ipcRenderer.invoke("download:start", request);
}

async function resolveUnifiedDownload(request: {
  item_type: "component" | "lora" | "preprocessor";
  source: string | string[];
  save_path?: string;
}): Promise<
  ConfigResponse<{
    job_id: string;
    exists: boolean;
    running: boolean;
    downloaded: boolean;
    bucket: string;
    save_dir: string;
    source: string | string[];
  }>
> {
  return await ipcRenderer.invoke("download:resolve", request);
}

async function getUnifiedDownloadStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("download:status", jobId);
}

async function cancelUnifiedDownload(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("download:cancel", jobId);
}

function onUnifiedDownloadUpdate(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsUpdate(`download:${jobId}`, callback);
}

function onUnifiedDownloadStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsStatus(`download:${jobId}`, callback);
}

function onUnifiedDownloadError(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsError(`download:${jobId}`, callback);
}

async function resolveUnifiedDownloadBatch(request: {
  item_type: "component" | "lora" | "preprocessor";
  sources: Array<string | string[]>;
  save_path?: string;
}): Promise<
  ConfigResponse<{
    results: Array<{
      job_id: string;
      exists: boolean;
      running: boolean;
      downloaded: boolean;
      bucket: string;
      save_dir: string;
      source: string | string[];
    }>;
  }>
> {
  return await ipcRenderer.invoke("download:resolve-batch", request);
}

async function connectUnifiedDownloadWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await wsConnect(`download:${jobId}`, `/ws/job/${jobId}`);
}

async function disconnectUnifiedDownloadWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await wsDisconnect(`download:${jobId}`);
}

async function deleteDownload(request: {
  path: string;
  item_type?: "component" | "lora" | "preprocessor";
  source?: string | string[];
  save_path?: string;
}): Promise<
  ConfigResponse<{
    path: string;
    status: string;
    removed_mapping?: boolean;
    unmarked?: boolean;
  }>
> {
  return await ipcRenderer.invoke("download:delete", request);
}

export {
  sha256sum,
  versions,
  processMediaTo24,
  renameMediaPair,
  deleteMediaPair,
  resolvePath,
  send,
  getMediaRootAbsolute,
  listConvertedMedia,
  listGeneratedMedia,
  importMediaPaths,
  ensureUniqueConvertedName,
  revealMediaItemInFolder,
  revealPathInFolder,
  ensurePreviewDir,
  savePreviewImage,
  saveImageToPath,
  savePreviewVideoFromFrames,
  getPreviewPath,
  exportVideoOpen,
  exportVideoAppend,
  exportVideoClose,
  exportVideoAbort,
  savePreviewAudio,
  exportAudioMp3FromWav,
  renderAudioMixWithFfmpeg,
  pickMediaPaths,
  getLowercaseExtension,
  readFileBuffer,
  readFileStream,
  pathToFileURL,
  fileURLToPath,
  fetchFilters,
  getPathForFile,
  createProxy,
  removeProxy,
  wsConnect,
  wsDisconnect,
  wsStatus,
  onWsUpdate,
  onWsStatus,
  onWsError,
  getBackendUrl,
  setBackendUrl,
  getBackendIsRemote,
  getFileShouldUpload,
  getFileIsUploaded,
  getHomeDir,
  setHomeDir,
  getTorchDevice,
  setTorchDevice,
  getCachePath,
  setCachePath,
  getSystemMemory,
  deleteFile,
  listPreprocessors,
  getPreprocessor,
  downloadPreprocessor,
  runPreprocessor,
  getPreprocessorStatus,
  getPreprocessorResult,
  connectPreprocessorWebSocket,
  disconnectPreprocessorWebSocket,
  onPreprocessorWebSocketUpdate,
  onPreprocessorWebSocketStatus,
  onPreprocessorWebSocketError,
  downloadComponents,
  deleteComponent,
  getComponentsStatus,
  cancelComponents,
  connectComponentsWebSocket,
  disconnectComponentsWebSocket,
  onComponentsWebSocketUpdate,
  onComponentsWebSocketStatus,
  onComponentsWebSocketError,
  pathToFileURLString,
  cancelPreprocessor,
  createMask,
  trackMask,
  startMaskTrack,
  onMaskTrackChunk,
  onMaskTrackError,
  onMaskTrackEnd,
  cancelMaskTrack,
  trackShapes,
  startMaskTrackShapes,
  onMaskTrackShapesChunk,
  onMaskTrackShapesError,
  onMaskTrackShapesEnd,
  listManifestModelTypes,
  listManifests,
  listManifestsByModel,
  listManifestsByType,
  listManifestsByModelAndType,
  getManifest,
  getManifestPart,
  validateAndRegisterCustomModelPath,
  deleteCustomModelPath,
  updateManifestLoraScale,
  updateManifestLoraName,
  deleteManifestLora,
  runEngine,
  getEngineStatus,
  getEngineResult,
  cancelEngine,
  deletePreprocessor,
  runPostprocessor,
  getPostprocessorStatus,
  cancelPostprocessor,
  connectPostprocessorWebSocket,
  disconnectPostprocessorWebSocket,
  onPostprocessorWebSocketUpdate,
  onPostprocessorWebSocketStatus,
  onPostprocessorWebSocketError,
  startUnifiedDownload,
  resolveUnifiedDownload,
  resolveUnifiedDownloadBatch,
  getUnifiedDownloadStatus,
  cancelUnifiedDownload,
  onUnifiedDownloadUpdate,
  onUnifiedDownloadStatus,
  onUnifiedDownloadError,
  connectUnifiedDownloadWebSocket,
  disconnectUnifiedDownloadWebSocket,
  deleteDownload,
  listRayJobs,
  getRayJob,
  cancelRayJob,
  cancelAllRayJobs,
};
