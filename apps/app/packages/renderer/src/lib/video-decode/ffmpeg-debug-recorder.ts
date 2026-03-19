/**
 * Debug recorder that pipes all frames drawn by any scheduler to ffmpeg.
 * Spans across multiple renderers/clips. Start on first scheduler start,
 * stop and close the mp4 when the last scheduler stops.
 */

const { spawn } = require("child_process");
const path = require("path");
const os = require("os");

let ffmpegProcess: ReturnType<typeof spawn> | null = null;
let outputPath: string | null = null;
let width: number = 0;
let height: number = 0;
let targetFps: number = 24;
const activeSchedulers = new Set<string>();

function extractRgbaFromFrame(
  frame: VideoFrame,
  targetW?: number,
  targetH?: number
): Uint8Array | null {
  try {
    const w = frame.codedWidth;
    const h = frame.codedHeight;
    const outW = targetW ?? w;
    const outH = targetH ?? h;
    const canvas = new OffscreenCanvas(outW, outH);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(frame, 0, 0, w, h, 0, 0, outW, outH);
    const imageData = ctx.getImageData(0, 0, outW, outH);
    return new Uint8Array(imageData.data);
  } catch {
    return null;
  }
}

function ensureFfmpegStarted(w: number, h: number, fps: number): boolean {
  if (ffmpegProcess && width === w && height === h) return true;
  if (ffmpegProcess) {
    try {
      ffmpegProcess.stdin?.end();
      ffmpegProcess = null;
    } catch {
      /* ignore */
    }
  }

  width = w;
  height = h;
  targetFps = fps;
  outputPath = path.join(
    os.tmpdir(),
    `apex-debug-frames-${Date.now()}.mp4`
  );

  console.log("[ffmpeg-debug-recorder] ensureFfmpegStarted", outputPath);

  try {
    ffmpegProcess = spawn(
      "ffmpeg",
      [
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        `${width}x${height}`,
        "-r",
        String(targetFps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        outputPath,
      ],
      {
        stdio: ["pipe", "ignore", "pipe"],
        detached: true, // Survive worker termination so ffmpeg can finish writing
      }
    );
    ffmpegProcess.unref(); // Allow worker to exit without waiting for ffmpeg

    ffmpegProcess.stderr?.on("data", (chunk: Buffer) => {
      // Optional: log ffmpeg stderr for debugging
      if (process.env.APEX_DEBUG_FFMPEG) {
        console.debug("[ffmpeg-debug-recorder]", chunk.toString());
      }
    });

    ffmpegProcess.on("error", (err: Error) => {
      console.error("[ffmpeg-debug-recorder] ffmpeg error:", err);
      ffmpegProcess = null;
    });

    ffmpegProcess.on("close", (code: number, signal: string | null) => {
      const pathToLog = outputPath;
      ffmpegProcess = null;
      outputPath = null;
      console.log(
        `[ffmpeg-debug-recorder] ffmpeg close: code=${code} signal=${signal} path=${pathToLog ?? "n/a"}`
      );
      if (code === 0 && pathToLog) {
        console.log(`[ffmpeg-debug-recorder] Wrote ${pathToLog}`);
      }
    });

    ffmpegProcess.stdin?.on("finish", () => {
      console.log("[ffmpeg-debug-recorder] stdin finish (EOF sent)");
    });

    ffmpegProcess.stdin?.on("error", (err: Error) => {
      console.error("[ffmpeg-debug-recorder] stdin error:", err);
    });

    return true;
  } catch (err) {
    console.error("[ffmpeg-debug-recorder] Failed to spawn ffmpeg:", err);
    return false;
  }
}

export function startRecording(id: string, fps: number = 24): void {
  activeSchedulers.add(id);
  targetFps = fps;
}

export function writeFrame(id: string, frame: VideoFrame, fps: number = 24): void {
  if (activeSchedulers.size === 0) return;
  const w = frame.codedWidth;
  const h = frame.codedHeight;
  if (w <= 0 || h <= 0) return;

  if (!ensureFfmpegStarted(w, h, fps)) {
    console.error("[ffmpeg-debug-recorder] Failed to ensure ffmpeg started");
    return;
  }

  // Scale to initial dimensions if frame size differs (e.g. different clip)
  const rgba =
    width > 0 && height > 0
      ? extractRgbaFromFrame(frame, width, height)
      : extractRgbaFromFrame(frame);
  if (!rgba || !ffmpegProcess?.stdin?.writable) return;

  try {
    ffmpegProcess.stdin.write(Buffer.from(rgba));
  } catch (err) {
    console.error("[ffmpeg-debug-recorder] Write error:", err);
  }
}

export function stopRecording(id: string): void {
  activeSchedulers.delete(id);
  if (activeSchedulers.size === 0 && ffmpegProcess?.stdin?.writable) {
    console.log("[ffmpeg-debug-recorder] Stopping: ending stdin, waiting for ffmpeg to finish");
    try {
      ffmpegProcess.stdin.end();
      // Do NOT null ffmpegProcess here - let the 'close' handler do it when ffmpeg exits
    } catch (err) {
      console.error("[ffmpeg-debug-recorder] Stop recording error:", err);
      ffmpegProcess = null;
      outputPath = null;
    }
  }
}
