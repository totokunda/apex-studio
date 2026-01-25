import fs from "node:fs";
import { promises as fsp } from "node:fs";
import { basename, dirname, extname, isAbsolute, join, sep } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { ipcRenderer } from "electron";
import { getMediaRootAbsolute } from "./root.js";
import { resolveFfmpegCommand } from "./ffmpegBin.js";

// Local helper mirror of ensurePreviewDir from the previous monolith.
async function ensurePreviewDir(): Promise<{
  root: string;
  previewsAbs: string;
}> {
  const root = getMediaRootAbsolute();
  const previewsAbs = join(root, "previews");
  if (!fs.existsSync(previewsAbs)) fs.mkdirSync(previewsAbs, { recursive: true });
  return { root, previewsAbs };
}

async function cleanupOldPreviews(options?: {
  maxAgeMs?: number;
}): Promise<{ deleted: number; scanned: number; previewsAbs: string }> {
  const { previewsAbs } = await ensurePreviewDir();
  const maxAgeMs =
    typeof options?.maxAgeMs === "number" && Number.isFinite(options.maxAgeMs)
      ? Math.max(0, options.maxAgeMs)
      : 24 * 60 * 60 * 1000;

  const cutoff = Date.now() - maxAgeMs;
  let deleted = 0;
  let scanned = 0;

  let entries: fs.Dirent[] = [];
  try {
    entries = await fsp.readdir(previewsAbs, { withFileTypes: true });
  } catch {
    return { deleted: 0, scanned: 0, previewsAbs };
  }

  for (const ent of entries) {
    // Only consider direct children of the previews directory.
    const name = ent?.name;
    if (!name) continue;
    if (name === "." || name === "..") continue;

    const abs = join(previewsAbs, name);
    scanned += 1;

    try {
      const st = await fsp.lstat(abs);
      const ts =
        st.birthtime?.getTime?.() ??
        st.mtime?.getTime?.() ??
        st.ctime?.getTime?.() ??
        0;
      if (!Number.isFinite(ts) || ts <= 0) continue;
      if (ts > cutoff) continue;

      if (ent.isDirectory()) {
        await fsp.rm(abs, { recursive: true, force: true });
      } else {
        await fsp.rm(abs, { force: true });
      }
      deleted += 1;
    } catch {
      // ignore per-entry issues (permissions, races, etc.)
    }
  }

  return { deleted, scanned, previewsAbs };
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
      toFfmpegPath(wavPath),
      "-c:a",
      "libmp3lame",
      "-b:a",
      "192k",
      toFfmpegPath(outAbs),
    ];
    const ff = spawn(resolveFfmpegCommand("ffmpeg"), args);
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

export type ExportVideoTranscodeOptions = {
  videoSrc: string;
  /**
   * Optional separate audio source. If omitted, we attempt to use audio from the video source.
   * Supports `file://...`, `app://...`, or absolute paths.
   */
  audioSrc?: string;
  /**
   * Whether to include audio at all.
   * - If false, the output will be silent.
   */
  includeAudio?: boolean;
  /**
   * If true and `audioSrc` is not provided, attempt to use embedded audio from the video source.
   * If false, we will NOT include embedded audio (even if present).
   */
  allowEmbeddedAudio?: boolean;
  outAbs: string; // absolute path preferred
  fps: number; // output fps (timeline/export fps)
  /**
   * Source-time trim expressed in seconds BEFORE speed is applied.
   * This should already include any timeline trim offsets (trimStart/srcStartFrame + rangeStart*speed).
   */
  srcStartSec: number;
  /**
   * Source-time duration in seconds BEFORE speed is applied.
   * For timeline duration D and speed S, this is typically D*S.
   */
  srcDurationSec: number;
  /**
   * Playback speed multiplier. 2 => 2x faster, 0.5 => half speed.
   */
  speed?: number;
  /**
   * Optional normalized crop in [0,1] space (relative to input frame).
   */
  crop?: { x: number; y: number; width: number; height: number };
  /**
   * Target output size. If omitted, ffmpeg will keep the (cropped) source size.
   */
  width?: number;
  height?: number;
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
  crf?: number;
  bitrate?: string;
  alpha?: boolean;
};

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function safeFixed(n: number, digits: number): string {
  const x = Number.isFinite(n) ? n : 0;
  return x.toFixed(digits);
}

function toFfmpegPath(p: string): string {
  let s = String(p || "");
  if (!s) return s;

  // Resolve file:// URLs to filesystem paths.
  if (s.startsWith("file://")) {
    try {
      s = fileURLToPath(s);
    } catch {
      // keep original
    }
  }

  // ffmpeg/ffprobe on Windows is generally happier with forward slashes,
  // and can choke on extended-length path prefixes (\\?\).
  if (process.platform === "win32") {
    if (s.startsWith("\\\\?\\UNC\\")) {
      // \\?\UNC\server\share\path -> \\server\share\path
      s = "\\\\" + s.slice("\\\\?\\UNC\\".length);
    } else if (s.startsWith("\\\\?\\")) {
      // \\?\C:\path -> C:\path
      s = s.slice("\\\\?\\".length);
    }

    if (s.includes("\\")) {
      const wasUnc = s.startsWith("\\\\");
      s = s.replace(/\\/g, "/");
      if (wasUnc) s = s.replace(/^\/{4}/, "//");
    }
  }

  return s;
}

async function exportVideoTranscodeWithFfmpeg(
  options: ExportVideoTranscodeOptions,
): Promise<string> {
  const fps = Math.max(1, Math.trunc(Number(options.fps) || 24));
  const speedRaw = Number(options.speed ?? 1);
  const speed =
    Number.isFinite(speedRaw) && speedRaw > 0
      ? Math.min(5, Math.max(0.1, speedRaw))
      : 1;
  const srcStartSec = Math.max(0, Number(options.srcStartSec || 0));
  const srcDurationSec = Math.max(0.001, Number(options.srcDurationSec || 0));

  const videoPath = await resolveFfmpegInputPath(options.videoSrc);
  const audioPath = options.audioSrc
    ? await resolveFfmpegInputPath(options.audioSrc)
    : undefined;

  const outAbs = options.outAbs;
  const outPathForFfmpeg = toFfmpegPath(outAbs);
  if (toFfmpegPath(videoPath) === outPathForFfmpeg) {
    throw new Error(
      "Export failed: source path is the same as the output path. Please export to a different file than the source media.",
    );
  }
  if (audioPath && toFfmpegPath(audioPath) === outPathForFfmpeg) {
    throw new Error(
      "Export failed: source path is the same as the output path. Please export to a different file than the source media.",
    );
  }

  // Determine container/codec defaults
  const format = (options.format ||
    (outAbs.split(".").pop() as any) ||
    "mp4") as NonNullable<ExportVideoTranscodeOptions["format"]>;
  const codec = (options.codec ||
    (format === "webm" ? "vp9" : "h264")) as NonNullable<
    ExportVideoTranscodeOptions["codec"]
  >;
  const preset = options.preset || "medium";
  const wantsAlpha = !!options.alpha;

  // Mirror constraints from exportVideoOpen
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

  const args: string[] = ["-y"];

  // Inputs
  args.push("-i", toFfmpegPath(videoPath));
  const audioInputIndex = audioPath ? 1 : 0;
  if (audioPath) args.push("-i", toFfmpegPath(audioPath));

  // Build video filterchain
  const vfParts: string[] = [];
  vfParts.push(
    `trim=start=${safeFixed(srcStartSec, 6)}:duration=${safeFixed(srcDurationSec, 6)}`,
  );
  vfParts.push("setpts=PTS-STARTPTS");
  if (speed !== 1) {
    // speed up: reduce PTS; slow down: increase PTS
    vfParts.push(`setpts=PTS/${safeFixed(speed, 6)}`);
  }

  // Crop (normalized)
  if (options.crop) {
    const cx = clamp01(options.crop.x);
    const cy = clamp01(options.crop.y);
    const cw = Math.max(0, clamp01(options.crop.width));
    const ch = Math.max(0, clamp01(options.crop.height));
    if (cw > 0 && ch > 0) {
      vfParts.push(
        `crop=iw*${safeFixed(cw, 6)}:ih*${safeFixed(ch, 6)}:iw*${safeFixed(cx, 6)}:ih*${safeFixed(cy, 6)}`,
      );
    }
  }

  // Scale (if requested)
  const targetW = options.width ? Math.max(1, Math.trunc(options.width)) : 0;
  const targetH = options.height ? Math.max(1, Math.trunc(options.height)) : 0;
  if (targetW > 0 && targetH > 0) {
    // Ensure even dimensions for chroma 4:2:0 codecs
    const isChroma420 = codec !== "prores";
    const finalW = isChroma420 ? targetW & ~1 : targetW;
    const finalH = isChroma420 ? targetH & ~1 : targetH;
    vfParts.push(`scale=${finalW}:${finalH}:flags=lanczos`);
  }

  // Output fps
  vfParts.push(`fps=${fps}`);

  const filterParts: string[] = [];
  filterParts.push(`[0:v]${vfParts.join(",")}[vout]`);

  // Audio: optional; if present apply trim + atempo chain to match speed
  const includeAudio = options.includeAudio !== false;
  const allowEmbeddedAudio = options.allowEmbeddedAudio !== false;
  const hasAudio = includeAudio
    ? await (async () => {
        if (audioPath) return true;
        if (!allowEmbeddedAudio) return false;
        return await hasAudioStreamStrict(videoPath);
      })()
    : false;

  if (hasAudio) {
    const afParts: string[] = [];
    afParts.push(
      `atrim=start=${safeFixed(srcStartSec, 6)}:duration=${safeFixed(srcDurationSec, 6)}`,
    );
    afParts.push("asetpts=PTS-STARTPTS");
    if (speed !== 1) {
      const chain = buildAtempoChain(speed);
      for (const r of chain) afParts.push(`atempo=${r}`);
    }
    filterParts.push(`[${audioInputIndex}:a]${afParts.join(",")}[aout]`);
  }

  args.push("-filter_complex", filterParts.join(";"));
  args.push("-map", "[vout]");
  if (hasAudio) args.push("-map", "[aout]");
  else args.push("-an");

  // Video encoding settings (mirror exportVideoOpen)
  switch (codec) {
    case "h264":
      if (wantsAlpha) {
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
      break;
    case "hevc":
      if (wantsAlpha) {
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

  if (format === "mp4" || format === "mov") {
    args.push("-movflags", "+faststart");
  }

  // Audio encoding
  if (hasAudio) {
    if (format === "webm" || format === "mkv") {
      args.push("-c:a", "libopus", "-b:a", "128k", "-ar", "48000");
    } else {
      args.push("-c:a", "aac", "-b:a", "192k", "-ar", "48000");
    }
  }

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
  if (container) args.push("-f", container);

  args.push(outPathForFfmpeg);

  await new Promise<void>((resolve, reject) => {
    const ff = spawn(resolveFfmpegCommand("ffmpeg"), args);
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

// Detect whether a media file has at least one audio stream using ffprobe.
// If ffprobe is unavailable or returns an unexpected error, we conservatively
// assume audio is present so behavior degrades to the previous implementation.
async function hasAudioStream(path: string): Promise<boolean> {
  return await new Promise<boolean>((resolve) => {
    try {
      const ff = spawn(resolveFfmpegCommand("ffprobe"), [
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        toFfmpegPath(path),
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
        if (stderr.includes("matches no streams")) {
          resolve(false);
          return;
        }
        if (code === 0 && stdout.trim().length > 0) {
          resolve(true);
          return;
        }
        resolve(true);
      });
      ff.on("error", () => {
        resolve(true);
      });
    } catch {
      resolve(true);
    }
  });
}

// Strict audio detection for transcode: if ffprobe fails, assume NO audio.
// This avoids failing the whole export for video-only sources when ffprobe
// is unavailable or errors.
async function hasAudioStreamStrict(path: string): Promise<boolean> {
  return await new Promise<boolean>((resolve) => {
    try {
      const ff = spawn(resolveFfmpegCommand("ffprobe"), [
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        toFfmpegPath(path),
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
        if (stderr.includes("matches no streams")) {
          resolve(false);
          return;
        }
        if (code === 0 && stdout.trim().length > 0) {
          resolve(true);
          return;
        }
        resolve(false);
      });
      ff.on("error", () => resolve(false));
    } catch {
      resolve(false);
    }
  });
}

async function resolveFfmpegInputPath(src: string): Promise<string> {
  let p = src;
  try {
    if (typeof p === "string" && p.startsWith("app://")) {
      const resolved = await ipcRenderer.invoke("appdir:resolve-path", p);
      if (typeof resolved === "string" && resolved.length > 0) {
        return toFfmpegPath(resolved);
      }
    }
    if (typeof p === "string" && p.startsWith("file://")) {
      try {
        p = fileURLToPath(p);
      } catch {
        // fall through with original
      }
    }
  } catch {
    // fall back to original src on any IPC error
  }
  return toFfmpegPath(p);
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
  type ActiveClip = { clip: AudioMixClipSpec; inputIndex: number };
  const activeClips: ActiveClip[] = [];
  let nextInputIndex = 0;
  for (const c of clips) {
    const p = await resolveFfmpegInputPath(c.src);
    // eslint-disable-next-line no-await-in-loop
    const hasAudio = await hasAudioStream(p);
    if (!hasAudio) continue;
    args.push("-i", toFfmpegPath(p));
    activeClips.push({ clip: c, inputIndex: nextInputIndex });
    nextInputIndex += 1;
  }

  if (activeClips.length === 0) {
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

    const trimStartFramesRaw = Number(c.trimStart ?? 0);
    const trimStartFrames = Number.isFinite(trimStartFramesRaw)
      ? Math.max(0, Math.trunc(trimStartFramesRaw))
      : 0;

    const realStartFrame = clipStart - trimStartFrames;

    const visibleStart = Math.max(exportStart, clipStart);
    const visibleEnd = Math.max(visibleStart, Math.min(exportEnd, clipEnd));

    const clipTimelineDurSec = Math.max(0, (visibleEnd - visibleStart) / fps);

    const delayMs = Math.max(
      0,
      Math.round(((visibleStart - exportStart) / fps) * 1000),
    );

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
    chain.push(`adelay=${delayMs}|${delayMs}`);
    filterParts.push(
      `${labelIn}${chain.length ? chain.join(",") : "anull"}${labelOut}`,
    );
    outLabels.push(labelOut);
  });

  const baseLabel = "[base]";
  filterParts.unshift(
    `anullsrc=r=48000:cl=stereo:d=${durationSec.toFixed(6)}${baseLabel}`,
  );
  outLabels.unshift(baseLabel);

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
  args.push(toFfmpegPath(outAbs));

  await new Promise<void>((resolve, reject) => {
    const ff = spawn(resolveFfmpegCommand("ffmpeg"), args);
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

type EncodeSession = {
  id: string;
  fps: number;
  outAbs: string;
  proc: import("child_process").ChildProcessWithoutNullStreams;
  closed: boolean;
};

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
  crf?: number;
  bitrate?: string;
  resolution?: { width: number; height: number };
  alpha?: boolean;
  audioPath?: string;
};

type AdvancedEncodeSession = EncodeSession & {
  options: ExportVideoOpenOptions;
  stderr: string;
  args: string[];
};

const encodeSessions: Map<string, EncodeSession> = new Map();

async function exportVideoOpen(
  options: ExportVideoOpenOptions,
): Promise<{ sessionId: string; outAbs: string }> {
  const outAbs = options.filename;
  const outDir = dirname(outAbs);
  try {
    await fsp.mkdir(outDir, { recursive: true });
  } catch {
    // ignore
  }

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
    (format === "webm" ? "vp9" : "h264")) as Required<ExportVideoOpenOptions>["codec"];
  const preset = options.preset || "medium";
  const wantsAlpha = !!options.alpha;

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
  let audioPath: string | undefined;
  if (hasAudio) {
    audioPath = await resolveFfmpegInputPath(options.audioPath as string);
    const outPathForFfmpeg = toFfmpegPath(outAbs);
    if (toFfmpegPath(audioPath) === outPathForFfmpeg) {
      throw new Error(
        "Export failed: audio source path is the same as the output path. Please export to a different file than the source media.",
      );
    }
    args.push("-i", toFfmpegPath(audioPath));
  }

  const useCodec = codec;
  switch (useCodec) {
    case "h264":
      if (wantsAlpha) {
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

  const isChroma420 = useCodec !== "prores";
  const finalW = isChroma420 ? targetW & ~1 : targetW;
  const finalH = isChroma420 ? targetH & ~1 : targetH;
  if (finalW !== options.width || finalH !== options.height) {
    args.push("-vf", `scale=${finalW}:${finalH}:flags=lanczos`);
  }

  if ((format === "mp4" || format === "mov") && !wantsAlpha) {
    args.push("-movflags", "+faststart");
  }

  if (hasAudio) {
    if (format === "mp4" || format === "mov") {
      args.push("-c:a", "aac", "-b:a", "192k", "-ar", "48000");
    } else if (format === "webm" || format === "mkv") {
      args.push("-c:a", "libopus", "-b:a", "128k", "-ar", "48000");
    } else {
      args.push("-c:a", "aac", "-b:a", "192k", "-ar", "48000");
    }
  }

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

  args.push(toFfmpegPath(outAbs));

  const proc = spawn(resolveFfmpegCommand("ffmpeg"), args);
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
      (session as any).stderr += s;
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
    } catch {
      // ignore
    }
  });
}

async function exportVideoAbort(sessionId: string): Promise<void> {
  const s = encodeSessions.get(sessionId);
  if (!s) return;
  try {
    s.proc.stdin.destroy();
  } catch {
    // ignore
  }
  try {
    s.proc.kill("SIGKILL");
  } catch {
    // ignore
  }
  try {
    await fsp.unlink(s.outAbs);
  } catch {
    // ignore
  }
  encodeSessions.delete(sessionId);
}

async function savePreviewVideoFromFrames(
  frames: Array<ArrayBuffer | Uint8Array | string>,
  options?: { fps?: number; fileNameHint?: string },
): Promise<string> {
  const { previewsAbs } = await ensurePreviewDir();
  const fps =
    options?.fps && Number.isFinite(options.fps)
      ? Math.max(1, Math.trunc(options.fps as number))
      : 24;

  const tempDirBase = `frames_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const tempDir = join(previewsAbs, tempDirBase);
  await fsp.mkdir(tempDir, { recursive: true });

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
    try {
      await fsp.rm(tempDir, { recursive: true, force: true });
    } catch {
      // ignore
    }
    throw new Error("savePreviewVideoFromFrames: no frames provided");
  }

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

  const inputPattern = toFfmpegPath(join(tempDir, "frame_%06d.png"));
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
    toFfmpegPath(outAbs),
  ];

  await new Promise<void>((resolve, reject) => {
    const ff = spawn(resolveFfmpegCommand("ffmpeg"), args);
    let stderr = "";
    ff.stderr.setEncoding("utf8");
    ff.stderr.on("data", (d) => {
      stderr += String(d);
    });
    ff.on("close", async (code) => {
      try {
        await fsp.rm(tempDir, { recursive: true, force: true });
      } catch {
        // ignore cleanup error
      }
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`ffmpeg failed (${code}): ${stderr}`));
      }
    });
    ff.on("error", async (err) => {
      try {
        await fsp.rm(tempDir, { recursive: true, force: true });
      } catch {
        // ignore
      }
      reject(err);
    });
  });

  return outAbs;
}

async function getPreviewPath(
  idOrPath: string,
  options?: { ext?: string; ensureUnique?: boolean },
): Promise<string> {
  const { previewsAbs } = await ensurePreviewDir();
  const ensureUnique = options?.ensureUnique !== false;
  const desiredExt = (options?.ext || "mp4").replace(/^\./, "");

  let maybePath = String(idOrPath || "");
  if (maybePath.startsWith("file://")) {
    try {
      maybePath = fileURLToPath(maybePath);
    } catch {
      // keep original
    }
  }

  const isAbsolutePath = typeof maybePath === "string" && isAbsolute(maybePath);

  let baseName: string;
  if (isAbsolutePath) {
    const prefix = previewsAbs.endsWith(sep) ? previewsAbs : previewsAbs + sep;
    if (maybePath === previewsAbs || maybePath.startsWith(prefix)) {
      baseName = maybePath.slice(prefix.length);
    } else {
      baseName = basename(maybePath);
    }
  } else {
    baseName = String(idOrPath);
  }

  baseName =
    baseName.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") ||
    `preview_${Date.now()}`;

  const hasExt = !!extname(baseName);
  const desired = hasExt ? baseName : `${baseName}.${desiredExt}`;

  let targetPath =
    isAbsolutePath && idOrPath.startsWith(previewsAbs + "/")
      ? join(previewsAbs, baseName)
      : join(previewsAbs, desired);

  if (!ensureUnique) return targetPath;

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

export {
  ensurePreviewDir,
  cleanupOldPreviews,
  savePreviewImage,
  saveImageToPath,
  savePreviewAudio,
  exportAudioMp3FromWav,
  renderAudioMixWithFfmpeg,
  exportVideoTranscodeWithFfmpeg,
  exportVideoOpen,
  exportVideoAppend,
  exportVideoClose,
  exportVideoAbort,
  savePreviewVideoFromFrames,
  getPreviewPath,
};


