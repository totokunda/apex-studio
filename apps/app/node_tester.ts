import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const require = createRequire(import.meta.url);

const addonPath =
  "/Users/tosinkuye/apex-workspace/apex-studio/apps/app/packages/native-decoder/build/Release/native_decoder.node";
const testVideoPath =
  process.argv[2] ??
  "/Users/tosinkuye/Downloads/YTDown.com_YouTube_DJ-Khaled-Wild-Thoughts-Official-Video-f_Media_fyaI4-5849w_001_1080p.mp4";

const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 720;
const POOL_SIZE = 4;
const NATIVE_DECODE_ONLY = process.env.NATIVE_DECODE_ONLY !== "0";
const NATIVE_OUTPUT_FORMAT =
  process.env.NATIVE_OUTPUT_FORMAT === "nv12" ? "nv12" : "rgba";
const NATIVE_FRAME_BATCH_SIZE = Math.max(
  1,
  Number(process.env.NATIVE_FRAME_BATCH_SIZE ?? "32") || 32,
);
const NATIVE_SUPPRESS_FRAME_CALLBACKS =
  process.env.NATIVE_SUPPRESS_FRAME_CALLBACKS === "1";

type NativeDecoderAddon = {
  createDecoder(): number;
  configure(
    handle: number,
    opts: {
      filePath: string;
      width: number;
      height: number;
      decodeOnly?: boolean;
      outputFormat?: "rgba" | "nv12";
      suppressFrameCallbacks?: boolean;
      frameBatchSize?: number;
      bufferPool: ArrayBuffer[];
      onFrame: (
        bufferIndex: number,
        width: number,
        height: number,
        timestamp: number,
        duration: number,
        requestId: number,
      ) => void;
      onFrameBatch?: (framesFlat: Float64Array) => void;
      onError: (message: string) => void;
      onReady: () => void;
    },
  ): void;
  iterate(
    handle: number,
    startTime: number,
    endTime: number,
    requestId: number,
  ): Promise<void>;
  ackFrame(handle: number, bufferIndex: number): void;
  dispose(handle: number): void;
  disposeAll(): void;
  getCapabilities(): { hwAccelMethods: string[]; preferredMethod: string };
};

type FrameAudit = {
  totalCallbacks?: number;
  iterateCallbacks?: number;
  nonIterateCallbacks?: number;
  outputCallbacks?: number;
  packetsSubmitted?: number;
  outOfOrderTimestamps?: number;
  duplicateTimestamps?: number;
  maxBackwardJumpSeconds?: number;
  readbackChecksum?: number;
};

type BenchResult = {
  label: string;
  frames: number;
  wallSeconds: number;
  mediaSeconds: number;
  fps: number;
  realtimeFactor: number;
  expectedFrames?: number;
  frameAudit?: FrameAudit;
};

type VideoProbe = {
  expectedFrames?: number;
  width?: number;
  height?: number;
  durationSeconds?: number;
  fps?: number;
};

type WebCodecsMode = "decode-only" | "rgba-readback";

const addon = require(addonPath) as NativeDecoderAddon;
const ffprobeStatic = require("ffprobe-static") as { path: string };

function parseFps(raw: string | undefined): number | null {
  if (!raw || raw === "0/0") return null;
  const parts = raw.split("/");
  if (parts.length !== 2) return null;
  const num = Number(parts[0]);
  const den = Number(parts[1]);
  if (!Number.isFinite(num) || !Number.isFinite(den) || den === 0) return null;
  const value = num / den;
  return Number.isFinite(value) && value > 0 ? value : null;
}

async function probeVideo(videoPath: string): Promise<VideoProbe> {
  const ffprobePath = ffprobeStatic?.path;
  if (!ffprobePath) return {};

  const args = [
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-count_frames",
    "-show_entries",
    "stream=nb_read_frames,avg_frame_rate,r_frame_rate,duration,width,height",
    "-of",
    "json",
    videoPath,
  ];

  return new Promise<VideoProbe>((resolve) => {
    const child = spawn(ffprobePath, args, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";

    child.stdout.on("data", (d) => {
      out += d.toString();
    });

    child.on("error", () => resolve({}));
    child.on("close", () => {
      try {
        const parsed = JSON.parse(out);
        const stream = parsed?.streams?.[0];
        if (!stream) {
          resolve({});
          return;
        }

        const width = Number(stream.width);
        const height = Number(stream.height);
        const duration = Number(stream.duration);
        const fps = parseFps(stream.avg_frame_rate) ?? parseFps(stream.r_frame_rate) ?? undefined;

        let expectedFrames: number | undefined;
        const nbReadFrames = Number(stream.nb_read_frames);
        if (Number.isFinite(nbReadFrames) && nbReadFrames > 0) {
          expectedFrames = Math.round(nbReadFrames);
        } else if (Number.isFinite(duration) && duration > 0 && fps && fps > 0) {
          expectedFrames = Math.round(duration * fps);
        }

        resolve({
          expectedFrames,
          width: Number.isFinite(width) && width > 0 ? Math.round(width) : undefined,
          height: Number.isFinite(height) && height > 0 ? Math.round(height) : undefined,
          durationSeconds: Number.isFinite(duration) && duration > 0 ? duration : undefined,
          fps,
        });
      } catch {
        resolve({});
      }
    });
  });
}

function getTargetDimensions(probe: VideoProbe): { width: number; height: number } {
  if (probe.width && probe.height && probe.width > 0 && probe.height > 0) {
    return { width: probe.width, height: probe.height };
  }
  return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT };
}

function makeBufferPool(
  width: number,
  height: number,
  decodeOnly: boolean,
  outputFormat: "rgba" | "nv12",
): ArrayBuffer[] {
  const bytesPerFrame = decodeOnly
    ? 4
    : outputFormat === "nv12"
      ? Math.max(1, Math.floor((width * height * 3) / 2))
      : Math.max(1, width * height * 4);
  return Array.from({ length: POOL_SIZE }, () => new ArrayBuffer(bytesPerFrame));
}

async function runNativeBenchmark(videoPath: string, probe: VideoProbe): Promise<BenchResult> {
  console.log("\n[native] Loaded addon from:", addonPath);
  console.log("[native] Capabilities:", addon.getCapabilities());
  console.log(
    "[native] Mode:",
    NATIVE_DECODE_ONLY
      ? "decode-only (no frame conversion)"
      : `output=${NATIVE_OUTPUT_FORMAT}`,
  );
  console.log("[native] suppress callbacks:", NATIVE_SUPPRESS_FRAME_CALLBACKS);

  const { width, height } = getTargetDimensions(probe);
  console.log(`[native] Target output: ${width}x${height}`);
  if (typeof probe.expectedFrames === "number") {
    console.log("[native] Expected frames (ffprobe):", probe.expectedFrames);
  } else {
    console.log("[native] Expected frames (ffprobe): unavailable");
  }

  const handle = addon.createDecoder();
  const pool = makeBufferPool(width, height, NATIVE_DECODE_ONLY, NATIVE_OUTPUT_FORMAT);
  console.log("[native] Decoder handle:", handle);

  let readyResolve!: () => void;
  let readyReject!: (e: unknown) => void;
  const readyPromise = new Promise<void>((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });

  let frameCount = 0;
  const ITERATE_REQUEST_ID = 2001;
  const ITERATE_START = 0;
  const ITERATE_END = 1e9;
  let firstIterTs: number | null = null;
  let lastIterTs = 0;
  let totalCallbacks = 0;
  let iterateCallbacks = 0;
  let nonIterateCallbacks = 0;
  let outOfOrderTimestamps = 0;
  let duplicateTimestamps = 0;
  let maxBackwardJumpSeconds = 0;
  let lastIterTimestamp: number | null = null;
  const processNativeFrame = (
    bufferIndex: number,
    timestamp: number,
    duration: number,
    requestId: number,
  ) => {
    totalCallbacks += 1;
    if (requestId === ITERATE_REQUEST_ID) {
      iterateCallbacks += 1;
      frameCount += 1;
      if (firstIterTs == null) firstIterTs = timestamp;
      lastIterTs = timestamp + duration;

      if (lastIterTimestamp != null) {
        const delta = timestamp - lastIterTimestamp;
        if (delta < 0) {
          outOfOrderTimestamps += 1;
          const backwardJump = Math.abs(delta);
          if (backwardJump > maxBackwardJumpSeconds) {
            maxBackwardJumpSeconds = backwardJump;
          }
        }
        if (Math.abs(delta) < 1e-9) {
          duplicateTimestamps += 1;
        }
      }
      lastIterTimestamp = timestamp;
    } else {
      nonIterateCallbacks += 1;
    }

    if (bufferIndex >= 0) {
      addon.ackFrame(handle, bufferIndex);
    }
  };

  addon.configure(handle, {
    filePath: videoPath,
    width,
    height,
    decodeOnly: NATIVE_DECODE_ONLY,
    outputFormat: NATIVE_OUTPUT_FORMAT,
    suppressFrameCallbacks: NATIVE_SUPPRESS_FRAME_CALLBACKS,
    frameBatchSize: NATIVE_FRAME_BATCH_SIZE,
    bufferPool: pool,
    onFrame: (bufferIndex, _w, _h, timestamp, duration, requestId) => {
      processNativeFrame(bufferIndex, timestamp, duration, requestId);
    },
    onFrameBatch: (framesFlat) => {
      for (let i = 0; i + 5 < framesFlat.length; i += 6) {
        const bufferIndex = Number(framesFlat[i]);
        const timestamp = Number(framesFlat[i + 3]);
        const duration = Number(framesFlat[i + 4]);
        const requestId = Number(framesFlat[i + 5]);
        processNativeFrame(bufferIndex, timestamp, duration, requestId);
      }
    },
    onError: (message) => {
      readyReject(new Error(message));
    },
    onReady: () => {
      readyResolve();
    },
  });

  await readyPromise;

  const t0 = performance.now();
  await addon.iterate(handle, ITERATE_START, ITERATE_END, ITERATE_REQUEST_ID);
  const t1 = performance.now();

  const wallSeconds = (t1 - t0) / 1000;
  const estimatedFrameCount = (NATIVE_SUPPRESS_FRAME_CALLBACKS && typeof probe.expectedFrames === "number")
    ? probe.expectedFrames
    : frameCount;
  const mediaSeconds = firstIterTs == null
    ? (probe.durationSeconds ?? 0)
    : Math.max(0, lastIterTs - firstIterTs);
  const fps = wallSeconds > 0 ? estimatedFrameCount / wallSeconds : 0;
  const realtimeFactor = wallSeconds > 0 ? mediaSeconds / wallSeconds : 0;

  addon.dispose(handle);

  return {
    label: NATIVE_DECODE_ONLY ? "native-decode-only" : `native-${NATIVE_OUTPUT_FORMAT}`,
    frames: estimatedFrameCount,
    wallSeconds,
    mediaSeconds,
    fps,
    realtimeFactor,
    expectedFrames: probe.expectedFrames,
    frameAudit: {
      totalCallbacks,
      iterateCallbacks,
      nonIterateCallbacks,
      outOfOrderTimestamps,
      duplicateTimestamps,
      maxBackwardJumpSeconds,
    },
  };
}

function electronBenchScript(
  videoPath: string,
  mediabunnyPath: string,
  mode: WebCodecsMode,
  targetWidth: number,
  targetHeight: number,
): string {
  const label = mode === "decode-only" ? "webcodecs-decode" : "webcodecs-rgba-readback";

  const rendererScript = `
(async () => {
  const { Input, UrlSource, EncodedPacketSink, ALL_FORMATS } = require(${JSON.stringify(mediabunnyPath)});
  if (typeof VideoDecoder === "undefined") {
    throw new Error("VideoDecoder is unavailable in renderer context");
  }

  const mode = ${JSON.stringify(mode)};
  const targetWidth = ${targetWidth};
  const targetHeight = ${targetHeight};
  const sourceUrl = new URL("/video", location.href).toString();

  const input = new Input({
    formats: ALL_FORMATS,
    source: new UrlSource(sourceUrl),
  });

  const track = await input.getPrimaryVideoTrack();
  if (!track) {
    throw new Error("No primary video track found");
  }

  const sink = new EncodedPacketSink(track);
  const cfg = await track.getDecoderConfig();
  if (!cfg) {
    throw new Error("No WebCodecs decoder config returned for this asset. Codec is likely unsupported in this Electron runtime.");
  }

  const support = await VideoDecoder.isConfigSupported(cfg);
  if (!support.supported) {
    const codec = cfg.codec || "unknown";
    throw new Error("WebCodecs does not support this decoder config (codec=" + codec + ") in this Electron runtime.");
  }
  const decoderConfig = support.config || cfg;

  let frames = 0;
  let firstTs = null;
  let lastTs = 0;
  let outputCallbacks = 0;
  let packetsSubmitted = 0;
  let outOfOrderTimestamps = 0;
  let duplicateTimestamps = 0;
  let maxBackwardJumpSeconds = 0;
  let lastOutputTs = null;
  let readbackChecksum = 0;

  let canvas = null;
  let ctx = null;

  let rejectErr;
  const decoder = new VideoDecoder({
    output: (frame) => {
      outputCallbacks += 1;
      const ts = frame.timestamp / 1e6;
      const dur = (frame.duration || 0) / 1e6;

      if (lastOutputTs != null) {
        const delta = ts - lastOutputTs;
        if (delta < 0) {
          outOfOrderTimestamps += 1;
          const backwardJump = Math.abs(delta);
          if (backwardJump > maxBackwardJumpSeconds) {
            maxBackwardJumpSeconds = backwardJump;
          }
        }
        if (Math.abs(delta) < 1e-9) {
          duplicateTimestamps += 1;
        }
      }
      lastOutputTs = ts;

      if (mode === "rgba-readback") {
        const frameW = frame.displayWidth || frame.codedWidth || 1;
        const frameH = frame.displayHeight || frame.codedHeight || 1;
        const outW = Math.max(1, targetWidth || frameW);
        const outH = Math.max(1, targetHeight || frameH);

        if (!canvas) {
          canvas = document.createElement("canvas");
          canvas.width = outW;
          canvas.height = outH;
          ctx = canvas.getContext("2d", { willReadFrequently: true });
          if (!ctx) {
            throw new Error("Failed to create 2D context for WebCodecs readback benchmark");
          }
        }

        if (canvas.width !== outW || canvas.height !== outH) {
          canvas.width = outW;
          canvas.height = outH;
        }

        ctx.drawImage(frame, 0, 0, outW, outH);
        const image = ctx.getImageData(0, 0, outW, outH);
        const data = image.data;
        const last = data.length > 0 ? data[data.length - 1] : 0;
        readbackChecksum = (readbackChecksum + data[0] + last) >>> 0;
      }

      if (firstTs === null) firstTs = ts;
      lastTs = ts + dur;
      frames += 1;
      frame.close();
    },
    error: (err) => {
      rejectErr = err;
    },
  });

  decoder.configure({ ...decoderConfig, optimizeForLatency: true });

  const t0 = performance.now();
  for await (const packet of sink.packets()) {
    if (rejectErr) throw rejectErr;
    packetsSubmitted += 1;
    decoder.decode(packet.toEncodedVideoChunk());
    if (decoder.decodeQueueSize > 8) {
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  await decoder.flush();
  decoder.close();
  const t1 = performance.now();

  const wallSeconds = (t1 - t0) / 1000;
  const mediaSeconds = firstTs === null ? 0 : Math.max(0, lastTs - firstTs);
  const fps = wallSeconds > 0 ? frames / wallSeconds : 0;
  const realtimeFactor = wallSeconds > 0 ? mediaSeconds / wallSeconds : 0;

  return {
    label: ${JSON.stringify(label)},
    frames,
    wallSeconds,
    mediaSeconds,
    fps,
    realtimeFactor,
    frameAudit: {
      outputCallbacks,
      packetsSubmitted,
      outOfOrderTimestamps,
      duplicateTimestamps,
      maxBackwardJumpSeconds,
      readbackChecksum,
    },
  };
})()
`;

  return `
const { app, BrowserWindow } = require("electron");
const http = require("node:http");
const fs = require("node:fs");
const videoPath = ${JSON.stringify(videoPath)};

function parseRangeHeader(rangeHeader, totalSize) {
  const match = /^bytes=(\\d*)-(\\d*)$/.exec(rangeHeader || "");
  if (!match) return null;

  const startStr = match[1];
  const endStr = match[2];
  let start = startStr === "" ? NaN : Number(startStr);
  let end = endStr === "" ? NaN : Number(endStr);

  if (Number.isNaN(start) && Number.isNaN(end)) return null;
  if (Number.isNaN(start)) {
    const suffix = end;
    if (!Number.isFinite(suffix) || suffix <= 0) return null;
    start = Math.max(0, totalSize - suffix);
    end = totalSize - 1;
  } else if (Number.isNaN(end)) {
    end = totalSize - 1;
  }

  if (start < 0 || end < start || start >= totalSize) return null;
  end = Math.min(end, totalSize - 1);
  return { start, end };
}

async function run() {
  await app.whenReady();
  const stat = fs.statSync(videoPath);
  if (!stat.isFile()) {
    throw new Error("Video path is not a file: " + videoPath);
  }
  const totalSize = stat.size;

  const server = http.createServer((req, res) => {
    const reqUrl = req.url || "/";
    if (reqUrl.startsWith("/video")) {
      const range = parseRangeHeader(req.headers.range, totalSize);
      res.setHeader("Accept-Ranges", "bytes");
      res.setHeader("Cache-Control", "no-store");
      res.setHeader("Content-Type", "application/octet-stream");

      if (range) {
        const { start, end } = range;
        const chunkLen = end - start + 1;
        res.statusCode = 206;
        res.setHeader("Content-Range", "bytes " + start + "-" + end + "/" + totalSize);
        res.setHeader("Content-Length", String(chunkLen));
        if (req.method === "HEAD") {
          res.end();
          return;
        }

        const stream = fs.createReadStream(videoPath, { start, end });
        stream.on("error", (e) => {
          res.statusCode = 500;
          res.end(String(e && e.message ? e.message : e));
        });
        stream.pipe(res);
        return;
      }

      res.statusCode = 200;
      res.setHeader("Content-Length", String(totalSize));
      if (req.method === "HEAD") {
        res.end();
        return;
      }
      const stream = fs.createReadStream(videoPath);
      stream.on("error", (e) => {
        res.statusCode = 500;
        res.end(String(e && e.message ? e.message : e));
      });
      stream.pipe(res);
      return;
    }

    const html = "<!doctype html><html><body>webcodecs-bench</body></html>";
    res.statusCode = 200;
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.setHeader("Content-Length", String(Buffer.byteLength(html)));
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    res.end(html);
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });

  const addr = server.address();
  const benchUrl = "http://127.0.0.1:" + addr.port + "/";

  const win = new BrowserWindow({
    show: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      sandbox: false,
      webSecurity: false,
    },
  });

  try {
    await win.loadURL(benchUrl);
    const result = await win.webContents.executeJavaScript(
      ${JSON.stringify(rendererScript)}
    );

    console.log("__WEB_BENCH__" + JSON.stringify(result));
    await win.destroy();
    await new Promise((resolve) => server.close(resolve));
    app.exit(0);
  } catch (err) {
    const msg = err && err.stack ? err.stack : String(err);
    console.error("__WEB_BENCH_ERR__" + msg);
    try { await win.destroy(); } catch {}
    try { await new Promise((resolve) => server.close(resolve)); } catch {}
    app.exit(1);
  }
}

run();
`;
}

async function runWebCodecsBenchmark(
  videoPath: string,
  probe: VideoProbe,
  mode: WebCodecsMode,
): Promise<BenchResult> {
  const electronBinary = require("electron") as string;
  const appDir = fileURLToPath(new URL(".", import.meta.url));
  const mediabunnyPath = require.resolve("mediabunny", { paths: [appDir] }) as string;
  const { width, height } = getTargetDimensions(probe);

  const script = electronBenchScript(videoPath, mediabunnyPath, mode, width, height);
  const tmpDir = mkdtempSync(join(tmpdir(), `apex-webcodecs-${mode}-`));
  const entryFile = join(tmpDir, "electron-bench.cjs");
  writeFileSync(entryFile, script, "utf8");

  return new Promise<BenchResult>((resolve, reject) => {
    const child = spawn(electronBinary, [entryFile], {
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });

    let out = "";
    let err = "";

    child.stdout.on("data", (d) => {
      const text = d.toString();
      out += text;
      process.stdout.write(`[${mode}] ${text}`);
    });

    child.stderr.on("data", (d) => {
      const text = d.toString();
      err += text;
      process.stderr.write(`[${mode}] ${text}`);
    });

    child.on("close", (code) => {
      const marker = "__WEB_BENCH__";
      const idx = out.lastIndexOf(marker);

      if (idx >= 0) {
        try {
          const json = out.slice(idx + marker.length).trim().split("\n")[0];
          const parsed = JSON.parse(json) as BenchResult;
          parsed.expectedFrames = probe.expectedFrames;
          rmSync(tmpDir, { recursive: true, force: true });
          resolve(parsed);
          return;
        } catch (e) {
          rmSync(tmpDir, { recursive: true, force: true });
          reject(new Error(`Failed to parse WebCodecs benchmark JSON: ${String(e)}`));
          return;
        }
      }

      rmSync(tmpDir, { recursive: true, force: true });
      reject(
        new Error(
          `WebCodecs benchmark failed (mode=${mode}, exit=${code}). stderr: ${err || "<empty>"}`,
        ),
      );
    });

    child.on("error", (e) => {
      rmSync(tmpDir, { recursive: true, force: true });
      reject(e);
    });
  });
}

function printResult(result: BenchResult): void {
  console.log(`\n[${result.label}] frames: ${result.frames}`);
  console.log(`[${result.label}] wall (s): ${result.wallSeconds.toFixed(3)}`);
  console.log(`[${result.label}] media (s): ${result.mediaSeconds.toFixed(3)}`);
  console.log(`[${result.label}] throughput (fps): ${result.fps.toFixed(2)}`);
  console.log(`[${result.label}] realtime factor (x): ${result.realtimeFactor.toFixed(2)}`);

  if (typeof result.expectedFrames === "number") {
    const diff = result.frames - result.expectedFrames;
    const ok = Math.abs(diff) <= 1;
    console.log(
      `[${result.label}] expected frames: ${result.expectedFrames} (delta=${diff}, verify=${ok ? "PASS" : "FAIL"})`,
    );
  }

  if (result.frameAudit) {
    if (typeof result.frameAudit.totalCallbacks === "number") {
      console.log(`[${result.label}] native onFrame total: ${result.frameAudit.totalCallbacks}`);
      console.log(
        `[${result.label}] native onFrame iterate/non-iterate: ${result.frameAudit.iterateCallbacks}/${result.frameAudit.nonIterateCallbacks}`,
      );
    }

    if (typeof result.frameAudit.outputCallbacks === "number") {
      console.log(`[${result.label}] WebCodecs output callbacks: ${result.frameAudit.outputCallbacks}`);
      console.log(`[${result.label}] WebCodecs packets submitted: ${result.frameAudit.packetsSubmitted}`);
    }

    if (typeof result.frameAudit.outOfOrderTimestamps === "number") {
      const maxBackward = result.frameAudit.maxBackwardJumpSeconds ?? 0;
      console.log(
        `[${result.label}] frame order issues: outOfOrder=${result.frameAudit.outOfOrderTimestamps}, duplicateTs=${result.frameAudit.duplicateTimestamps}, maxBackwardJump=${maxBackward.toFixed(6)}s`,
      );
    }

    if (typeof result.frameAudit.readbackChecksum === "number") {
      console.log(`[${result.label}] readback checksum: ${result.frameAudit.readbackChecksum}`);
    }
  }
}

function printComparison(left: BenchResult, right: BenchResult, name: string): void {
  const fpsRatio = right.fps > 0 ? left.fps / right.fps : 0;
  const rtRatio = right.realtimeFactor > 0 ? left.realtimeFactor / right.realtimeFactor : 0;
  console.log(`\n[compare:${name}] ${left.label}/${right.label} fps ratio: ${fpsRatio.toFixed(2)} x`);
  console.log(`[compare:${name}] ${left.label}/${right.label} realtime ratio: ${rtRatio.toFixed(2)} x`);
}

async function main(): Promise<void> {
  console.log("Video:", testVideoPath);

  const probe = await probeVideo(testVideoPath);
  const { width, height } = getTargetDimensions(probe);
  console.log(`[probe] source=${probe.width ?? "?"}x${probe.height ?? "?"}, target=${width}x${height}, fps=${probe.fps ?? "?"}, duration=${probe.durationSeconds ?? "?"}, expectedFrames=${probe.expectedFrames ?? "?"}`);

  const native = await runNativeBenchmark(testVideoPath, probe);
  printResult(native);

  const webDecode = await runWebCodecsBenchmark(testVideoPath, probe, "decode-only");
  printResult(webDecode);

  const webReadback = await runWebCodecsBenchmark(testVideoPath, probe, "rgba-readback");
  printResult(webReadback);

  printComparison(native, webDecode, "decode-only");
  printComparison(native, webReadback, "rgba-readback");
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  try {
    addon.disposeAll();
  } catch {
    // ignore cleanup failure
  }
  process.exit(1);
});
