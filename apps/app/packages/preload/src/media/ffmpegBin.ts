import { existsSync } from "node:fs";
import { join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

function tryRequire<T = unknown>(name: string): T | null {
  try {
    return require(name) as T;
  } catch {
    return null;
  }
}

function getBundledBinaryPath(cmd: "ffmpeg" | "ffprobe"): string | null {
  const exeName = process.platform === "win32" ? `${cmd}.exe` : cmd;
  const p = join(process.resourcesPath, "ffmpeg", exeName);
  return existsSync(p) ? p : null;
}

function getDevStaticBinaryPath(cmd: "ffmpeg" | "ffprobe"): string | null {
  if (cmd === "ffmpeg") {
    const p = tryRequire<string>("ffmpeg-static");
    return p && existsSync(p) ? p : null;
  }
  // ffprobe-static exports either { path } or a string depending on version.
  const mod = tryRequire<any>("ffprobe-static");
  const p = typeof mod === "string" ? mod : mod?.path;
  return typeof p === "string" && existsSync(p) ? p : null;
}

export function resolveFfmpegCommand(cmd: "ffmpeg" | "ffprobe"): string {
  // Explicit override (useful for debugging / custom installs)
  const override =
    cmd === "ffmpeg"
      ? process.env.APEX_FFMPEG_PATH || process.env.FFMPEG_PATH
      : process.env.APEX_FFPROBE_PATH || process.env.FFPROBE_PATH;
  if (override && existsSync(override)) return override;

  // Packaged build: prefer the bundled binary in resources.
  const bundled = getBundledBinaryPath(cmd);
  if (bundled) return bundled;

  // Dev: fall back to ffmpeg-static / ffprobe-static if installed.
  const dev = getDevStaticBinaryPath(cmd);
  if (dev) return dev;

  // Final fallback: rely on PATH.
  return cmd;
}


