import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ALL_FORMATS, BlobSource, Input } from "mediabunny";
import {
  pickMediaPaths,
  readFileBuffer,
  createNativeDecoder,
  configureNativeDecoder,
  seekNativeDecoder,
  iterateNativeDecoder,
  ackNativeFrame,
  cancelNativeDecoder,
  disposeNativeDecoder,
} from "@app/preload";
import { NativeFrameUploader } from "@/lib/media/native-frame-uploader";

type EngineId = "native" | "webcodecs";
type TabId = "compare" | EngineId;
const ENGINES: EngineId[] = ["native", "webcodecs"];

const ENGINE_LABEL: Record<EngineId, string> = {
  native: "Native Decoder",
  webcodecs: "WebCodecs + mediabunny",
};

const NATIVE_POOL_TARGET_BYTES = 320 * 1024 * 1024;
const NATIVE_POOL_MIN = 4;
const NATIVE_POOL_MAX = 16;
const BENCHMARK_RENDER_SAMPLE_EVERY_N_FRAMES = 30;
const BENCHMARK_UI_UPDATE_INTERVAL_MS = 120;

type PendingKind = "play" | "seek" | "scrub" | "pause" | "benchmark";
type NativePixelFormat = "rgba" | "nv12";

interface ProbeResult {
  duration: number;
  fps: number;
  width: number;
  height: number;
  videoDecoderConfig: VideoDecoderConfig;
}

interface DecodedFramePayload {
  requestId: number;
  timestamp: number;
  duration: number;
  width: number;
  height: number;
  rgba?: Uint8Array;
  nativePixelFormat?: NativePixelFormat;
  videoFrame?: VideoFrame;
}

interface ConfigureRequest {
  filePath: string;
  width: number;
  height: number;
  videoDecoderConfig: VideoDecoderConfig;
  onFrame: (payload: DecodedFramePayload) => void;
  onError: (message: string) => void;
  onReady: () => void;
}

interface DecoderAdapter {
  configure(request: ConfigureRequest): Promise<void>;
  seek(timestamp: number, forceAccurate: boolean, requestId: number): Promise<void>;
  iterate(
    startTime: number,
    endTime: number,
    requestId: number,
    mode?: "play" | "benchmark",
  ): Promise<void>;
  cancel(): void;
  dispose(): void;
}

function computeNativeBufferPoolSize(width: number, height: number): number {
  const frameBytes = Math.max(1, Math.floor(width * height * 3 / 2));
  const sizedByBudget = Math.floor(NATIVE_POOL_TARGET_BYTES / frameBytes);
  return Math.max(NATIVE_POOL_MIN, Math.min(NATIVE_POOL_MAX, sizedByBudget));
}

class NativeSandboxAdapter implements DecoderAdapter {
  private handle: number | null = null;
  private ackStrategy: "immediate" | "vsync" = "immediate";
  private readonly outputFormat: NativePixelFormat = "nv12";

  async configure(request: ConfigureRequest): Promise<void> {
    this.dispose();
    const handle = createNativeDecoder();
    const poolSize = computeNativeBufferPoolSize(request.width, request.height);
    this.handle = handle;

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      try {
        configureNativeDecoder(
          handle,
          request.filePath,
          request.width,
          request.height,
          (frameBytes, width, height, timestamp, duration, requestId, bufferIndex, pixelFormat) => {
            request.onFrame({
              requestId,
              timestamp,
              duration,
              width,
              height,
              rgba: frameBytes,
              nativePixelFormat: pixelFormat ?? this.outputFormat,
            });

            if (typeof bufferIndex === "number" && bufferIndex >= 0) {
              if (this.ackStrategy === "vsync") {
                requestAnimationFrame(() => {
                  if (this.handle === handle) {
                    ackNativeFrame(handle, bufferIndex);
                  }
                });
              } else {
                ackNativeFrame(handle, bufferIndex);
              }
            }
          },
          (message) => {
            request.onError(message);
            if (!settled) {
              settled = true;
              reject(new Error(message));
            }
          },
          () => {
            request.onReady();
            if (!settled) {
              settled = true;
              resolve();
            }
          },
          {
            copyFrameData: false,
            bufferPoolSize: poolSize,
            outputFormat: this.outputFormat,
            manualAck: true,
            // Shared buffers remove bridge copy overhead when cross-origin isolation is enabled.
            preferSharedBufferPool: typeof window !== "undefined" && window.crossOriginIsolated,
          },
        );
      } catch (error) {
        if (!settled) {
          settled = true;
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      }
    });
  }

  async seek(timestamp: number, forceAccurate: boolean, requestId: number): Promise<void> {
    if (this.handle == null) throw new Error("Native decoder is not configured");
    this.ackStrategy = "immediate";
    await seekNativeDecoder(this.handle, timestamp, forceAccurate, requestId);
  }

  async iterate(
    startTime: number,
    endTime: number,
    requestId: number,
    mode: "play" | "benchmark" = "play",
  ): Promise<void> {
    if (this.handle == null) throw new Error("Native decoder is not configured");
    this.ackStrategy = mode === "benchmark" ? "immediate" : "vsync";
    await iterateNativeDecoder(this.handle, startTime, endTime, requestId);
  }

  cancel(): void {
    this.ackStrategy = "immediate";
    if (this.handle != null) {
      cancelNativeDecoder(this.handle);
    }
  }

  dispose(): void {
    this.ackStrategy = "immediate";
    if (this.handle != null) {
      disposeNativeDecoder(this.handle);
      this.handle = null;
    }
  }
}

class WebCodecsSandboxAdapter implements DecoderAdapter {
  private worker: Worker | null = null;
  private readonly assetId = `sandbox-${Math.random().toString(36).slice(2)}`;
  private activeIterateRequestId: number | null = null;
  private onFrame: ((payload: DecodedFramePayload) => void) | null = null;
  private onError: ((message: string) => void) | null = null;
  private onReady: (() => void) | null = null;
  private pendingSeek = new Map<
    number,
    { resolve: () => void; reject: (error: Error) => void }
  >();
  private pendingIterate = new Map<
    number,
    { resolve: () => void; reject: (error: Error) => void }
  >();

  async configure(request: ConfigureRequest): Promise<void> {
    this.dispose();
    this.onFrame = request.onFrame;
    this.onError = request.onError;
    this.onReady = request.onReady;
    this.activeIterateRequestId = null;

    const worker = new Worker(
      new URL("../../lib/media/video-decoder.worker.ts", import.meta.url),
      { type: "module" },
    );
    this.worker = worker;

    await new Promise<void>((resolve, reject) => {
      let settled = false;

      const fail = (message: string) => {
        this.onError?.(message);
        if (!settled) {
          settled = true;
          reject(new Error(message));
        }
      };

      worker.onmessage = (event: MessageEvent<any>) => {
        const msg = event.data;

        if (msg?.type === "frame" && msg.frame) {
          const frame = msg.frame as VideoFrame;
          const width = frame.displayWidth || (frame as any).codedWidth || request.width;
          const height = frame.displayHeight || (frame as any).codedHeight || request.height;
          try {
            this.onFrame?.({
              requestId: Number(msg.requestId ?? 0),
              timestamp: Number(msg.timestamp ?? 0),
              duration: Number(msg.duration ?? 0),
              width: Number(width),
              height: Number(height),
              videoFrame: frame,
            });
          } finally {
            const shouldAck =
              this.activeIterateRequestId !== null &&
              this.activeIterateRequestId === Number(msg.requestId ?? -1);
            if (shouldAck) {
              worker.postMessage({
                type: "ack",
                assetId: this.assetId,
                requestId: Number(msg.requestId ?? 0),
              });
            }
            try {
              frame.close();
            } catch {
              // ignore
            }
          }
          return;
        }

        if (msg?.type === "ready") {
          this.onReady?.();
          if (!settled) {
            settled = true;
            resolve();
          }
          return;
        }

        if (msg?.type === "seekDone") {
          const reqId = Number(msg.requestId ?? -1);
          const pending = this.pendingSeek.get(reqId);
          if (pending) {
            this.pendingSeek.delete(reqId);
            pending.resolve();
          }
          return;
        }

        if (msg?.type === "iterateDone") {
          const reqId = Number(msg.requestId ?? -1);
          const pending = this.pendingIterate.get(reqId);
          if (pending) {
            this.pendingIterate.delete(reqId);
            pending.resolve();
          }
          if (this.activeIterateRequestId === reqId) {
            this.activeIterateRequestId = null;
          }
          return;
        }

        if (msg?.type === "error") {
          const message = String(msg.error ?? "Unknown worker error");
          const reqId = Number(msg.requestId ?? -1);
          const seekPending = this.pendingSeek.get(reqId);
          if (seekPending) {
            this.pendingSeek.delete(reqId);
            seekPending.reject(new Error(message));
          }
          const iteratePending = this.pendingIterate.get(reqId);
          if (iteratePending) {
            this.pendingIterate.delete(reqId);
            iteratePending.reject(new Error(message));
          }
          fail(message);
        }
      };

      worker.onerror = (event: ErrorEvent) => {
        fail(event.message || "WebCodecs worker crashed");
      };

      const config: VideoDecoderConfig = {
        ...request.videoDecoderConfig,
        codedWidth: request.width,
        codedHeight: request.height,
      };

      worker.postMessage({
        type: "configure",
        assetId: this.assetId,
        config: {
          videoDecoderConfig: config,
          asset: {
            id: this.assetId,
            type: "video",
            path: request.filePath,
          },
          initialTimestamp: 0,
        },
        requestId: 0,
      });
    });
  }

  async seek(timestamp: number, forceAccurate: boolean, requestId: number): Promise<void> {
    if (!this.worker) throw new Error("WebCodecs decoder is not configured");
    this.activeIterateRequestId = null;
    await new Promise<void>((resolve, reject) => {
      this.pendingSeek.set(requestId, { resolve, reject });
      this.worker!.postMessage({
        type: "seek",
        assetId: this.assetId,
        timestamp,
        forceAccurate,
        requestId,
      });
    });
  }

  async iterate(
    startTime: number,
    endTime: number,
    requestId: number,
    _mode: "play" | "benchmark" = "play",
  ): Promise<void> {
    if (!this.worker) throw new Error("WebCodecs decoder is not configured");
    this.activeIterateRequestId = requestId;
    await new Promise<void>((resolve, reject) => {
      this.pendingIterate.set(requestId, { resolve, reject });
      this.worker!.postMessage({
        type: "iterate",
        assetId: this.assetId,
        startTime,
        endTime,
        requestId,
      });
    });
  }

  cancel(): void {
    // No explicit cancel message exists for worker mode.
    // New request IDs supersede in-flight operations.
  }

  dispose(): void {
    for (const pending of this.pendingSeek.values()) {
      pending.reject(new Error("WebCodecs decoder disposed"));
    }
    for (const pending of this.pendingIterate.values()) {
      pending.reject(new Error("WebCodecs decoder disposed"));
    }
    this.pendingSeek.clear();
    this.pendingIterate.clear();
    this.activeIterateRequestId = null;

    if (this.worker) {
      this.worker.postMessage({ type: "dispose", assetId: this.assetId });
      this.worker.terminate();
      this.worker = null;
    }
  }
}

interface BenchmarkResult {
  wallSeconds: number;
  decodedFrames: number;
  renderedFrames: number;
  decodeFps: number;
  renderFps: number;
  realtimeFactor: number;
  firstFrameLatencyMs: number | null;
  renderedAllFrames: boolean;
  renderSamplingStep: number;
}

interface PlayerViewState {
  ready: boolean;
  playing: boolean;
  benchmarking: boolean;
  frames: number;
  renderedFrames: number;
  currentTime: number;
  lastPlayLatencyMs: number | null;
  lastSeekLatencyMs: number | null;
  lastScrubLatencyMs: number | null;
  lastPauseLatencyMs: number | null;
  throughputFps: number;
  realtimeFactor: number;
  errors: number;
  lastError: string;
  lastBenchmark: BenchmarkResult | null;
}

const INITIAL_PLAYER_STATE: PlayerViewState = {
  ready: false,
  playing: false,
  benchmarking: false,
  frames: 0,
  renderedFrames: 0,
  currentTime: 0,
  lastPlayLatencyMs: null,
  lastSeekLatencyMs: null,
  lastScrubLatencyMs: null,
  lastPauseLatencyMs: null,
  throughputFps: 0,
  realtimeFactor: 0,
  errors: 0,
  lastError: "",
  lastBenchmark: null,
};

interface PendingOperation {
  kind: PendingKind;
  startedAtMs: number;
}

interface PlayRunState {
  requestId: number;
  startedAtMs: number;
  startFrameCount: number;
  startMediaTime: number;
}

interface BenchmarkRunState {
  requestId: number;
  startedAtMs: number;
  startFrameCount: number;
  startRenderedCount: number;
  startMediaTime: number;
  firstFrameLatencyMs: number | null;
}

interface PlayerRuntime {
  adapter: DecoderAdapter | null;
  requestId: number;
  duration: number;
  currentTime: number;
  frameCount: number;
  renderedCount: number;
  lastUiUpdateAtMs: number;
  pending: Map<number, PendingOperation>;
  activeIterationRequestId: number | null;
  activePlayRun: PlayRunState | null;
  activeBenchmark: BenchmarkRunState | null;
}

function createPlayerRuntime(): PlayerRuntime {
  return {
    adapter: null,
    requestId: 0,
    duration: 0,
    currentTime: 0,
    frameCount: 0,
    renderedCount: 0,
    lastUiUpdateAtMs: 0,
    pending: new Map<number, PendingOperation>(),
    activeIterationRequestId: null,
    activePlayRun: null,
    activeBenchmark: null,
  };
}

function createAdapter(engine: EngineId): DecoderAdapter {
  return engine === "native"
    ? new NativeSandboxAdapter()
    : new WebCodecsSandboxAdapter();
}

function formatMs(value: number | null): string {
  return value == null ? "-" : `${value.toFixed(2)} ms`;
}

function formatSeconds(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(3)} s` : "-";
}

function formatFps(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(2)} fps` : "-";
}

async function probeFile(filePath: string): Promise<ProbeResult> {
  const bytes = await readFileBuffer(filePath);
  const binary = new Uint8Array(bytes);
  const input = new Input({
    formats: ALL_FORMATS,
    source: new BlobSource(new Blob([binary])),
  });

  const videoTrack = await input.getPrimaryVideoTrack();
  if (!videoTrack) {
    throw new Error("No video track found in selected file.");
  }

  const canDecode = await videoTrack.canDecode();
  if (!canDecode) {
    throw new Error("Video track cannot be decoded by WebCodecs on this machine.");
  }

  const videoDecoderConfig = await videoTrack.getDecoderConfig();
  if (!videoDecoderConfig) {
    throw new Error("Unable to derive VideoDecoderConfig.");
  }

  const duration = Number((await input.computeDuration()) ?? 0);
  const packetStats = await videoTrack.computePacketStats(1000);

  const width = Number(
    videoDecoderConfig.codedWidth ?? (videoTrack as any).displayWidth ?? 0,
  );
  const height = Number(
    videoDecoderConfig.codedHeight ?? (videoTrack as any).displayHeight ?? 0,
  );

  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error("Invalid video dimensions reported by decoder config.");
  }

  const fps = Number(
    (packetStats as any)?.frameRate ??
      (packetStats as any)?.averageFrameRate ??
      (videoTrack as any)?.frameRate ??
      0,
  );
  return {
    duration: Number.isFinite(duration) && duration > 0 ? duration : 0,
    fps: Number.isFinite(fps) && fps > 0 ? fps : 0,
    width,
    height,
    videoDecoderConfig,
  };
}

export default function DecoderSandbox() {
  const [activeTab, setActiveTab] = useState<TabId>("compare");
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadError, setLoadError] = useState<string>("");
  const [players, setPlayers] = useState<Record<EngineId, PlayerViewState>>({
    native: { ...INITIAL_PLAYER_STATE },
    webcodecs: { ...INITIAL_PLAYER_STATE },
  });

  const nativeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const webCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const nativeUploaderRef = useRef<NativeFrameUploader | null>(null);
  const nativeUploaderCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const runtimesRef = useRef<Record<EngineId, PlayerRuntime>>({
    native: createPlayerRuntime(),
    webcodecs: createPlayerRuntime(),
  });
  const scrubTimersRef = useRef<Record<EngineId, number | null>>({
    native: null,
    webcodecs: null,
  });

  const patchPlayer = useCallback(
    (engine: EngineId, updater: (prev: PlayerViewState) => PlayerViewState) => {
      setPlayers((prev) => ({ ...prev, [engine]: updater(prev[engine]) }));
    },
    [],
  );

  const getCanvasRef = useCallback(
    (engine: EngineId) => (engine === "native" ? nativeCanvasRef : webCanvasRef),
    [],
  );

  const drawFrame = useCallback(
    (engine: EngineId, payload: DecodedFramePayload): boolean => {
      const canvas = getCanvasRef(engine).current;
      if (!canvas) return false;

      if (engine === "native" && payload.rgba) {
        if (
          !nativeUploaderRef.current ||
          nativeUploaderCanvasRef.current !== canvas
        ) {
          nativeUploaderRef.current?.dispose();
          nativeUploaderRef.current = new NativeFrameUploader(canvas);
          nativeUploaderCanvasRef.current = canvas;
        }
        if (payload.nativePixelFormat === "nv12") {
          const expected = Math.floor(payload.width * payload.height * 3 / 2);
          if (payload.rgba.byteLength >= expected) {
            nativeUploaderRef.current.uploadNV12(payload.rgba, payload.width, payload.height);
            return true;
          }
        } else {
          const expected = payload.width * payload.height * 4;
          if (payload.rgba.byteLength >= expected) {
            nativeUploaderRef.current.upload(payload.rgba, payload.width, payload.height);
            return true;
          }
        }
      }

      if (canvas.width !== payload.width || canvas.height !== payload.height) {
        canvas.width = payload.width;
        canvas.height = payload.height;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return false;

      if (payload.videoFrame) {
        ctx.drawImage(payload.videoFrame, 0, 0, payload.width, payload.height);
        return true;
      }

      if (payload.rgba && payload.rgba.byteLength >= payload.width * payload.height * 4) {
        const expected = payload.width * payload.height * 4;
        const src = new Uint8ClampedArray(payload.rgba.buffer, payload.rgba.byteOffset, expected);
        const clamped = new Uint8ClampedArray(expected);
        clamped.set(src);
        const imageData = new ImageData(clamped, payload.width, payload.height);
        ctx.putImageData(imageData, 0, 0);
        return true;
      }

      return false;
    },
    [getCanvasRef],
  );

  const completePendingOperation = useCallback(
    (engine: EngineId, requestId: number) => {
      const runtime = runtimesRef.current[engine];
      const pending = runtime.pending.get(requestId);
      if (!pending) return;
      runtime.pending.delete(requestId);

      const latencyMs = performance.now() - pending.startedAtMs;
      patchPlayer(engine, (prev) => {
        const next = { ...prev };
        if (pending.kind === "play") next.lastPlayLatencyMs = latencyMs;
        if (pending.kind === "seek") next.lastSeekLatencyMs = latencyMs;
        if (pending.kind === "scrub") next.lastScrubLatencyMs = latencyMs;
        if (pending.kind === "pause") next.lastPauseLatencyMs = latencyMs;
        if (pending.kind === "benchmark" && runtime.activeBenchmark) {
          runtime.activeBenchmark.firstFrameLatencyMs = latencyMs;
        }
        return next;
      });
    },
    [patchPlayer],
  );

  const handleEngineError = useCallback(
    (engine: EngineId, message: string) => {
      patchPlayer(engine, (prev) => ({
        ...prev,
        playing: false,
        benchmarking: false,
        errors: prev.errors + 1,
        lastError: message,
      }));
    },
    [patchPlayer],
  );

  const finalizeActivePlayRun = useCallback(
    (engine: EngineId, requestId: number) => {
      const runtime = runtimesRef.current[engine];
      const run = runtime.activePlayRun;
      if (!run || run.requestId !== requestId) return;

      const wallSeconds = Math.max(0.0001, (performance.now() - run.startedAtMs) / 1000);
      const frameDelta = Math.max(0, runtime.frameCount - run.startFrameCount);
      const mediaDelta = Math.max(0, runtime.currentTime - run.startMediaTime);

      runtime.activePlayRun = null;
      patchPlayer(engine, (prev) => ({
        ...prev,
        currentTime: runtime.currentTime,
        frames: runtime.frameCount,
        renderedFrames: runtime.renderedCount,
        throughputFps: frameDelta / wallSeconds,
        realtimeFactor: mediaDelta / wallSeconds,
      }));
    },
    [patchPlayer],
  );

  const finalizeBenchmarkRun = useCallback(
    (engine: EngineId, requestId: number) => {
      const runtime = runtimesRef.current[engine];
      const run = runtime.activeBenchmark;
      if (!run || run.requestId !== requestId) return;

      const wallSeconds = Math.max(0.0001, (performance.now() - run.startedAtMs) / 1000);
      const decodedFrames = Math.max(0, runtime.frameCount - run.startFrameCount);
      const renderedFrames = Math.max(0, runtime.renderedCount - run.startRenderedCount);
      const mediaDelta = Math.max(0, runtime.currentTime - run.startMediaTime);
      const result: BenchmarkResult = {
        wallSeconds,
        decodedFrames,
        renderedFrames,
        decodeFps: decodedFrames / wallSeconds,
        renderFps: renderedFrames / wallSeconds,
        realtimeFactor: mediaDelta / wallSeconds,
        firstFrameLatencyMs: run.firstFrameLatencyMs,
        renderedAllFrames:
          BENCHMARK_RENDER_SAMPLE_EVERY_N_FRAMES <= 1
            ? renderedFrames === decodedFrames
            : true,
        renderSamplingStep: BENCHMARK_RENDER_SAMPLE_EVERY_N_FRAMES,
      };

      runtime.activeBenchmark = null;
      patchPlayer(engine, (prev) => ({
        ...prev,
        currentTime: runtime.currentTime,
        frames: runtime.frameCount,
        renderedFrames: runtime.renderedCount,
        benchmarking: false,
        lastBenchmark: result,
      }));
    },
    [patchPlayer],
  );

  const handleFrame = useCallback(
    (engine: EngineId, payload: DecodedFramePayload) => {
      const runtime = runtimesRef.current[engine];
      if (payload.requestId !== runtime.requestId) {
        return;
      }

      runtime.currentTime = payload.timestamp;
      runtime.frameCount += 1;

      const benchmarking = runtime.activeBenchmark?.requestId === payload.requestId;
      const shouldRender =
        !benchmarking ||
        runtime.frameCount % BENCHMARK_RENDER_SAMPLE_EVERY_N_FRAMES === 0;
      const rendered = shouldRender ? drawFrame(engine, payload) : false;
      if (rendered) {
        runtime.renderedCount += 1;
      }

      completePendingOperation(engine, payload.requestId);
      const now = performance.now();
      const shouldPatchUi =
        !benchmarking ||
        rendered ||
        now - runtime.lastUiUpdateAtMs >= BENCHMARK_UI_UPDATE_INTERVAL_MS;
      if (shouldPatchUi) {
        runtime.lastUiUpdateAtMs = now;
        patchPlayer(engine, (prev) => ({
          ...prev,
          currentTime: runtime.currentTime,
          frames: runtime.frameCount,
          renderedFrames: runtime.renderedCount,
        }));
      }
    },
    [completePendingOperation, drawFrame, patchPlayer],
  );

  const disposeEngine = useCallback(
    (engine: EngineId) => {
      const runtime = runtimesRef.current[engine];
      runtime.adapter?.dispose();
      runtime.adapter = null;
      runtime.requestId = 0;
      runtime.duration = 0;
      runtime.currentTime = 0;
      runtime.frameCount = 0;
      runtime.renderedCount = 0;
      runtime.lastUiUpdateAtMs = 0;
      runtime.pending.clear();
      runtime.activeIterationRequestId = null;
      runtime.activePlayRun = null;
      runtime.activeBenchmark = null;
      patchPlayer(engine, () => ({ ...INITIAL_PLAYER_STATE }));
    },
    [patchPlayer],
  );

  const configureEngine = useCallback(
    async (engine: EngineId, filePath: string, info: ProbeResult) => {
      disposeEngine(engine);

      const adapter = createAdapter(engine);
      const runtime = runtimesRef.current[engine];
      runtime.adapter = adapter;
      runtime.duration = info.duration;

      await adapter.configure({
        filePath,
        width: info.width,
        height: info.height,
        videoDecoderConfig: info.videoDecoderConfig,
        onFrame: (payload) => handleFrame(engine, payload),
        onError: (message) => handleEngineError(engine, message),
        onReady: () => {
          patchPlayer(engine, (prev) => ({ ...prev, ready: true }));
        },
      });
    },
    [disposeEngine, handleEngineError, handleFrame, patchPlayer],
  );

  const seekEngine = useCallback(
    (engine: EngineId, timestamp: number, kind: PendingKind, forceAccurate: boolean) => {
      const runtime = runtimesRef.current[engine];
      if (!runtime.adapter) return;

      const clampedTimestamp = Math.max(0, Math.min(timestamp, runtime.duration || timestamp));

      if (runtime.activePlayRun) {
        finalizeActivePlayRun(engine, runtime.activePlayRun.requestId);
      }
      if (runtime.activeBenchmark) {
        finalizeBenchmarkRun(engine, runtime.activeBenchmark.requestId);
      }

      runtime.adapter.cancel();
      runtime.pending.clear();

      const requestId = runtime.requestId + 1;
      runtime.requestId = requestId;
      runtime.currentTime = clampedTimestamp;
      runtime.lastUiUpdateAtMs = 0;
      runtime.activeIterationRequestId = null;
      runtime.activePlayRun = null;
      runtime.activeBenchmark = null;
      runtime.pending.set(requestId, { kind, startedAtMs: performance.now() });

      patchPlayer(engine, (prev) => ({
        ...prev,
        playing: false,
        benchmarking: false,
        currentTime: clampedTimestamp,
      }));

      void runtime.adapter
        .seek(clampedTimestamp, forceAccurate, requestId)
        .then(() => {
          completePendingOperation(engine, requestId);
        })
        .catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          handleEngineError(engine, message);
        });
    },
    [completePendingOperation, finalizeActivePlayRun, finalizeBenchmarkRun, handleEngineError, patchPlayer],
  );

  const startIterateEngine = useCallback(
    (engine: EngineId, mode: "play" | "benchmark", startTime: number, endTime: number) => {
      const runtime = runtimesRef.current[engine];
      if (!runtime.adapter) return;

      const clampedStart = Math.max(0, Math.min(startTime, runtime.duration || startTime));
      const clampedEnd = Math.max(clampedStart, Math.min(endTime, runtime.duration || endTime));

      runtime.adapter.cancel();
      runtime.pending.clear();

      const requestId = runtime.requestId + 1;
      runtime.requestId = requestId;
      runtime.currentTime = clampedStart;
      runtime.lastUiUpdateAtMs = 0;
      runtime.activeIterationRequestId = requestId;

      runtime.activePlayRun = {
        requestId,
        startedAtMs: performance.now(),
        startFrameCount: runtime.frameCount,
        startMediaTime: runtime.currentTime,
      };

      runtime.pending.set(requestId, {
        kind: mode === "play" ? "play" : "benchmark",
        startedAtMs: performance.now(),
      });

      if (mode === "benchmark") {
        runtime.activeBenchmark = {
          requestId,
          startedAtMs: performance.now(),
          startFrameCount: runtime.frameCount,
          startRenderedCount: runtime.renderedCount,
          startMediaTime: runtime.currentTime,
          firstFrameLatencyMs: null,
        };
      } else {
        runtime.activeBenchmark = null;
      }

      patchPlayer(engine, (prev) => ({
        ...prev,
        playing: true,
        benchmarking: mode === "benchmark",
        lastError: "",
      }));

      void runtime.adapter
        .iterate(clampedStart, clampedEnd, requestId, mode)
        .then(() => {
          finalizeActivePlayRun(engine, requestId);
          finalizeBenchmarkRun(engine, requestId);
          const rt = runtimesRef.current[engine];
          if (rt.activeIterationRequestId === requestId) {
            rt.activeIterationRequestId = null;
            patchPlayer(engine, (prev) => ({ ...prev, playing: false, benchmarking: false }));
          }
        })
        .catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          handleEngineError(engine, message);
          finalizeActivePlayRun(engine, requestId);
          finalizeBenchmarkRun(engine, requestId);
        });
    },
    [finalizeActivePlayRun, finalizeBenchmarkRun, handleEngineError, patchPlayer],
  );

  const playEngine = useCallback(
    (engine: EngineId) => {
      const runtime = runtimesRef.current[engine];
      startIterateEngine(engine, "play", runtime.currentTime, runtime.duration);
    },
    [startIterateEngine],
  );

  const pauseEngine = useCallback(
    (engine: EngineId) => {
      const runtime = runtimesRef.current[engine];
      seekEngine(engine, runtime.currentTime, "pause", false);
    },
    [seekEngine],
  );

  const runBenchmarkEngine = useCallback(
    (engine: EngineId) => {
      const runtime = runtimesRef.current[engine];
      startIterateEngine(engine, "benchmark", 0, runtime.duration);
    },
    [startIterateEngine],
  );

  const scheduleScrub = useCallback(
    (engine: EngineId, timestamp: number) => {
      const existing = scrubTimersRef.current[engine];
      if (existing != null) {
        window.clearTimeout(existing);
      }
      scrubTimersRef.current[engine] = window.setTimeout(() => {
        seekEngine(engine, timestamp, "scrub", false);
        scrubTimersRef.current[engine] = null;
      }, 40);
    },
    [seekEngine],
  );

  const loadFile = useCallback(
    async (filePath: string) => {
      setLoading(true);
      setLoadError("");
      try {
        const info = await probeFile(filePath);
        setSelectedFile(filePath);
        setProbe(info);

        await Promise.allSettled(
          ENGINES.map(async (engine) => {
            await configureEngine(engine, filePath, info);
          }),
        );

        for (const engine of ENGINES) {
          seekEngine(engine, 0, "seek", true);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setLoadError(message);
      } finally {
        setLoading(false);
      }
    },
    [configureEngine, seekEngine],
  );

  const onPickFile = useCallback(async () => {
    const paths = await pickMediaPaths({
      title: "Choose media file",
      directory: false,
    });
    if (!paths.length) return;
    await loadFile(paths[0]!);
  }, [loadFile]);

  useEffect(() => {
    return () => {
      for (const engine of ENGINES) {
        const timer = scrubTimersRef.current[engine];
        if (timer != null) {
          window.clearTimeout(timer);
          scrubTimersRef.current[engine] = null;
        }
      }
      for (const engine of ENGINES) {
        const runtime = runtimesRef.current[engine];
        runtime.adapter?.dispose();
      }
      nativeUploaderRef.current?.dispose();
      nativeUploaderRef.current = null;
      nativeUploaderCanvasRef.current = null;
    };
  }, []);

  const nativeBench = players.native.lastBenchmark;
  const webBench = players.webcodecs.lastBenchmark;
  const benchmarkRatio = useMemo(() => {
    if (!nativeBench || !webBench) return null;
    const decodeRatio = webBench.decodeFps > 0 ? nativeBench.decodeFps / webBench.decodeFps : 0;
    const renderRatio = webBench.renderFps > 0 ? nativeBench.renderFps / webBench.renderFps : 0;
    return { decodeRatio, renderRatio };
  }, [nativeBench, webBench]);

  const renderSingleTab = (engine: EngineId) => {
    const state = players[engine];

    return (
      <div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
          <button onClick={() => playEngine(engine)} disabled={!probe || loading || !state.ready}>
            Play
          </button>
          <button onClick={() => pauseEngine(engine)} disabled={!probe || loading || !state.ready}>
            Pause
          </button>
          <button
            onClick={() => seekEngine(engine, 0, "seek", true)}
            disabled={!probe || loading || !state.ready}
          >
            Seek 0s
          </button>
          <button
            onClick={() => runBenchmarkEngine(engine)}
            disabled={!probe || loading || !state.ready || state.benchmarking}
          >
            {state.benchmarking ? "Running..." : "Run Full-Speed Iterate"}
          </button>
        </div>

        {probe && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ marginBottom: 6 }}>
              Scrub ({ENGINE_LABEL[engine]}): {state.currentTime.toFixed(3)}s / {probe.duration.toFixed(3)}s
            </div>
            <input
              type="range"
              min={0}
              max={probe.duration}
              step={0.01}
              value={Math.max(0, Math.min(state.currentTime, probe.duration))}
              style={{ width: "100%" }}
              onChange={(e) => {
                const next = Number(e.target.value);
                scheduleScrub(engine, next);
              }}
              onPointerUp={(e) => {
                const next = Number((e.target as HTMLInputElement).value);
                seekEngine(engine, next, "seek", true);
              }}
            />
          </div>
        )}

        <EnginePanel title={ENGINE_LABEL[engine]} state={state} canvasRef={getCanvasRef(engine)} />
      </div>
    );
  };

  return (
    <div
      style={{
        height: "100vh",
        background: "#111",
        color: "#eee",
        padding: 16,
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        overflowY: "auto",
        overflowX: "hidden",
        boxSizing: "border-box",
      }}
    >
      <h1 style={{ margin: "0 0 8px 0" }}>Decoder Sandbox</h1>
      <div style={{ marginBottom: 12 }}>
        Tabbed native vs WebCodecs comparison with full-speed frame iteration and render audit.
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
        <button onClick={onPickFile} disabled={loading}>
          {loading ? "Loading..." : "Open File"}
        </button>
        <button
          onClick={() => {
            runBenchmarkEngine("native");
            runBenchmarkEngine("webcodecs");
          }}
          disabled={!probe || loading || players.native.benchmarking || players.webcodecs.benchmarking}
        >
          Run Full-Speed Iterate (Both)
        </button>
        <button
          onClick={() => {
            playEngine("native");
            playEngine("webcodecs");
          }}
          disabled={!probe || loading}
        >
          Play Both
        </button>
        <button
          onClick={() => {
            pauseEngine("native");
            pauseEngine("webcodecs");
          }}
          disabled={!probe || loading}
        >
          Pause Both
        </button>
      </div>

      <div style={{ marginBottom: 6 }}>File: {selectedFile || "-"}</div>
      {probe && (
        <div style={{ marginBottom: 12 }}>
          Probe: {probe.width}x{probe.height} | source fps={probe.fps.toFixed(3)} | duration={" "}
          {formatSeconds(probe.duration)}
        </div>
      )}
      {loadError ? <div style={{ color: "#ff7d7d", marginBottom: 12 }}>{loadError}</div> : null}

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <TabButton label="Compare" active={activeTab === "compare"} onClick={() => setActiveTab("compare")} />
        <TabButton
          label="Native"
          active={activeTab === "native"}
          onClick={() => setActiveTab("native")}
        />
        <TabButton
          label="WebCodecs"
          active={activeTab === "webcodecs"}
          onClick={() => setActiveTab("webcodecs")}
        />
      </div>

      {activeTab === "compare" ? (
        <div>
          <div style={{ marginBottom: 12 }}>
            {benchmarkRatio ? (
              <div>
                Last benchmark ratio (native/webcodecs): decode={benchmarkRatio.decodeRatio.toFixed(2)}x, render={" "}
                {benchmarkRatio.renderRatio.toFixed(2)}x
              </div>
            ) : (
              <div>Run full-speed iterate on both decoders to populate direct ratio.</div>
            )}
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
              gap: 16,
            }}
          >
            <EnginePanel
              title={ENGINE_LABEL.native}
              state={players.native}
              canvasRef={nativeCanvasRef}
            />
            <EnginePanel
              title={ENGINE_LABEL.webcodecs}
              state={players.webcodecs}
              canvasRef={webCanvasRef}
            />
          </div>
        </div>
      ) : (
        renderSingleTab(activeTab)
      )}
    </div>
  );
}

function TabButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 12px",
        border: "1px solid #444",
        background: active ? "#2a2a2a" : "#171717",
        color: "#eee",
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
}

function EnginePanel({
  title,
  state,
  canvasRef,
}: {
  title: string;
  state: PlayerViewState;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}) {
  const benchmark = state.lastBenchmark;
  const renderAudit = benchmark?.renderSamplingStep && benchmark.renderSamplingStep > 1
    ? true
    : state.frames > 0
      ? state.renderedFrames === state.frames
      : true;

  return (
    <div style={{ border: "1px solid #333", padding: 12, borderRadius: 8 }}>
      <div style={{ marginBottom: 8, fontWeight: 700 }}>{title}</div>
      <canvas
        ref={canvasRef}
        width={640}
        height={360}
        style={{
          width: "100%",
          maxHeight: 360,
          background: "#000",
          border: "1px solid #222",
          marginBottom: 10,
        }}
      />
      <div>ready: {String(state.ready)}</div>
      <div>playing: {String(state.playing)}</div>
      <div>benchmarking: {String(state.benchmarking)}</div>
      <div>current time: {formatSeconds(state.currentTime)}</div>
      <div>onFrame callbacks: {state.frames}</div>
      <div>rendered frames: {state.renderedFrames}</div>
      <div>
        render audit (all callbacks rendered):
        {" "}
        {renderAudit ? "PASS" : "FAIL"}
      </div>
      <div>play latency: {formatMs(state.lastPlayLatencyMs)}</div>
      <div>pause latency: {formatMs(state.lastPauseLatencyMs)}</div>
      <div>seek latency: {formatMs(state.lastSeekLatencyMs)}</div>
      <div>scrub latency: {formatMs(state.lastScrubLatencyMs)}</div>
      <div>last iterate throughput: {formatFps(state.throughputFps)}</div>
      <div>last iterate realtime factor: {state.realtimeFactor.toFixed(2)}x</div>
      <div>errors: {state.errors}</div>
      {state.lastError ? <div style={{ color: "#ff7d7d" }}>last error: {state.lastError}</div> : null}

      {benchmark ? (
        <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid #2a2a2a" }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>Full-Speed Iterate Result</div>
          <div>decoded frames: {benchmark.decodedFrames}</div>
          <div>rendered frames: {benchmark.renderedFrames}</div>
          <div>decode fps: {formatFps(benchmark.decodeFps)}</div>
          <div>render fps: {formatFps(benchmark.renderFps)}</div>
          <div>render sampling: every {benchmark.renderSamplingStep} frame(s)</div>
          <div>realtime factor: {benchmark.realtimeFactor.toFixed(2)}x</div>
          <div>first-frame latency: {formatMs(benchmark.firstFrameLatencyMs)}</div>
          <div>rendered every decoded frame: {benchmark.renderedAllFrames ? "PASS" : "FAIL"}</div>
          <div>wall time: {formatSeconds(benchmark.wallSeconds)}</div>
        </div>
      ) : null}
    </div>
  );
}
