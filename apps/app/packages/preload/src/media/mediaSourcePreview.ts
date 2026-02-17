import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

type MediaSourceSession = {
  process: ChildProcess;
  firstChunk?: Buffer;
  ended: boolean;
  stderrTail: string;
};

type StartPayload = {
  path: string;
  seekTo: number;
  fps?: number;
  size?: number;
  scope?: string;
};

type StartResult = {
  sessionId?: string;
  mimeCodec?: string;
  error?: string;
};

const require = createRequire(import.meta.url);
const sessions = new Map<string, MediaSourceSession>();
const pendingStartsByScope = new Map<string, { token: string; process: ChildProcess }>();
const MIME_CODEC = 'video/mp4; codecs="avc1.42C01F"';

const tryRequire = <T = unknown>(name: string): T | null => {
  try {
    return require(name) as T;
  } catch {
    return null;
  }
};

const resolveFfmpegCommand = () => {
  const cmd = "ffmpeg";
  const exeName = process.platform === "win32" ? `${cmd}.exe` : cmd;

  const override = process.env.APEX_FFMPEG_PATH || process.env.FFMPEG_PATH;
  if (override && existsSync(override)) return override;

  const userInstalled = join(homedir(), ".apex-studio", "ffmpeg", exeName);
  if (existsSync(userInstalled)) return userInstalled;

  const bundled = join(process.resourcesPath, "ffmpeg", exeName);
  if (existsSync(bundled)) return bundled;

  const devStatic = tryRequire<string>("ffmpeg-static");
  if (typeof devStatic === "string" && existsSync(devStatic)) return devStatic;

  return cmd;
};

const clampFps = (value: unknown) => {
  if (!Number.isFinite(value as number)) return undefined;
  return Math.max(1, Math.min(60, Math.round(value as number)));
};

const clampSize = (value: unknown) => {
  if (!Number.isFinite(value as number)) return undefined;
  return Math.max(64, Math.min(1920, Math.round(value as number)));
};

const toFilePath = (input: string) => {
  if (input.startsWith("file://")) return fileURLToPath(input);
  return input;
};

const killProcess = (process: ChildProcess) => {
  try {
    process.kill("SIGKILL");
  } catch {
    // best-effort
  }
};

const clearPendingStart = (scope: string, token: string) => {
  const pending = pendingStartsByScope.get(scope);
  if (pending?.token === token) {
    pendingStartsByScope.delete(scope);
  }
};

const registerPendingStart = (scope: string, token: string, process: ChildProcess) => {
  const previous = pendingStartsByScope.get(scope);
  if (previous) {
    killProcess(previous.process);
  }
  pendingStartsByScope.set(scope, { token, process });
};

const getVideoFilters = ({
  fps,
  size,
  forceColorspace,
}: {
  fps?: number;
  size?: number;
  forceColorspace?: boolean;
}) => {
  const scaleOptions: string[] = [];
  if (size != null) {
    scaleOptions.push(
      `${size}:${size}:flags=lanczos:force_original_aspect_ratio=decrease:force_divisible_by=2`,
    );
  }
  scaleOptions.push(
    "in_color_matrix=auto:in_range=auto:out_color_matrix=bt709:out_range=tv",
  );
  return [
    ...(fps != null ? [`fps=${fps}`] : []),
    ...(forceColorspace ? ["colorspace=iall=bt709:all=bt709"] : []),
    `scale=${scaleOptions.join(":")}`,
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
    "format=yuv420p",
  ];
};

const getArgs = ({
  path,
  seekTo,
  fps,
  size,
  forceColorspace,
}: {
  path: string;
  seekTo: number;
  fps?: number;
  size?: number;
  forceColorspace?: boolean;
}) => [
  "-hide_banner",
  "-loglevel",
  "error",
  "-fflags",
  "+nobuffer+flush_packets+discardcorrupt",
  "-avioflags",
  "direct",
  "-flush_packets",
  "1",
  "-ss",
  String(seekTo),
  "-i",
  path,
  "-fps_mode",
  "passthrough",
  "-map_metadata",
  "-1",
  "-map_chapters",
  "-1",
  "-an",
  "-vf",
  getVideoFilters({ fps, size, forceColorspace }).join(","),
  "-c:v",
  "libx264",
  "-preset",
  "ultrafast",
  "-tune",
  "zerolatency",
  "-crf",
  "10",
  "-g",
  "1",
  "-f",
  "mp4",
  "-movflags",
  "+frag_keyframe+empty_moov+default_base_moof",
  "-",
];

const appendStderrTail = (session: MediaSourceSession, chunk: unknown) => {
  const next = String(chunk ?? "");
  if (!next) return;
  session.stderrTail = `${session.stderrTail}${next}`.slice(-8000);
};

const waitForFirstChunk = async (process: ChildProcess, timeoutMs: number) =>
  new Promise<Buffer>((resolve, reject) => {
    const stdout = process.stdout;
    if (!stdout) {
      reject(new Error("MediaSource process missing stdout"));
      return;
    }

    let settled = false;
    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("Timed out waiting for first MediaSource chunk"));
    }, timeoutMs);

    const onData = (chunk: Buffer) => {
      if (settled) return;
      settled = true;
      cleanup();
      stdout.pause();
      resolve(chunk);
    };
    const onExit = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("MediaSource ffmpeg exited before first chunk"));
    };
    const onError = (err: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    };
    const cleanup = () => {
      clearTimeout(timeout);
      stdout.off("data", onData);
      process.off("exit", onExit);
      process.off("error", onError);
    };

    stdout.once("data", onData);
    process.once("exit", onExit);
    process.once("error", onError);
    stdout.resume();
  });

const readChunk = async (session: MediaSourceSession) =>
  new Promise<Uint8Array | null>((resolve, reject) => {
    if (session.firstChunk) {
      const first = session.firstChunk;
      session.firstChunk = undefined;
      resolve(new Uint8Array(first.buffer, first.byteOffset, first.byteLength));
      return;
    }

    if (session.ended) {
      resolve(null);
      return;
    }

    const stdout = session.process.stdout;
    if (!stdout) {
      session.ended = true;
      resolve(null);
      return;
    }

    const onData = (chunk: Buffer) => {
      cleanup();
      stdout.pause();
      resolve(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength));
    };
    const onError = (err: Error) => {
      cleanup();
      reject(err);
    };
    const onClose = () => {
      cleanup();
      session.ended = true;
      resolve(null);
    };
    const onExit = () => {
      cleanup();
      session.ended = true;
      resolve(null);
    };
    const cleanup = () => {
      stdout.off("data", onData);
      stdout.off("error", onError);
      stdout.off("close", onClose);
      session.process.off("exit", onExit);
    };

    stdout.once("data", onData);
    stdout.once("error", onError);
    stdout.once("close", onClose);
    session.process.once("exit", onExit);
    stdout.resume();
  });

const killSession = (sessionId: string) => {
  const session = sessions.get(sessionId);
  if (!session) return;
  sessions.delete(sessionId);
  session.ended = true;
  killProcess(session.process);
};

export async function previewMseStart(rawPayload: StartPayload): Promise<StartResult> {
  const payload = rawPayload || ({} as StartPayload);
  const path = toFilePath(String(payload.path || ""));
  const scope = String(payload.scope || "global");
  const seekTo = Number.isFinite(payload.seekTo) ? Math.max(0, payload.seekTo) : 0;
  const fps = clampFps(payload.fps);
  const size = clampSize(payload.size);

  if (!path) {
    return { error: "Missing media path for MediaSource stream" };
  }

  const attemptStart = async (forceColorspace?: boolean) => {
    const args = getArgs({ path, seekTo, fps, size, forceColorspace });
    const process = spawn(resolveFfmpegCommand(), args, {
      stdio: ["ignore", "pipe", "pipe"],
    });
    const token = randomUUID();
    registerPendingStart(scope, token, process);

    const session: MediaSourceSession = {
      process,
      ended: false,
      stderrTail: "",
    };
    process.stderr?.setEncoding("utf8");
    process.stderr?.on("data", (d) => appendStderrTail(session, d));
    process.once("exit", () => {
      session.ended = true;
    });

    try {
      const firstChunk = await waitForFirstChunk(process, 20_000);
      clearPendingStart(scope, token);
      session.firstChunk = firstChunk;
      return { session };
    } catch (err) {
      clearPendingStart(scope, token);
      killProcess(process);
      const message =
        err instanceof Error ? err.message : "Failed to start MediaSource stream";
      return {
        error: session.stderrTail
          ? `${message}. ffmpeg stderr: ${session.stderrTail}`
          : message,
      };
    }
  };

  const firstTry = await attemptStart(false);
  let session = firstTry.session;
  let error = "error" in firstTry ? firstTry.error : "Failed to start MediaSource stream";

  if (
    !session &&
    /Unsupported input|no path between colorspaces|colorspace|swscaler/i.test(
      String(error || ""),
    )
  ) {
    const secondTry = await attemptStart(true);
    session = secondTry.session;
    if ("error" in secondTry) {
      error = secondTry.error;
    }
  }

  if (!session) {
    return { error: error || "Failed to start MediaSource stream" };
  }

  const sessionId = randomUUID();
  sessions.set(sessionId, session);
  return { sessionId, mimeCodec: MIME_CODEC };
}

export async function previewMseRead(sessionId: string): Promise<Uint8Array | null> {
  const session = sessions.get(sessionId);
  if (!session) return null;

  const chunk = await readChunk(session);
  if (!chunk || chunk.byteLength === 0) {
    killSession(sessionId);
    return null;
  }
  return chunk;
}

export async function previewMseAbort(sessionId?: string): Promise<void> {
  if (!sessionId) return;
  killSession(sessionId);
}

