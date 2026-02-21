import { Asset, MediaInfo } from "../types";
import {
  ADTS,
  FLAC,
  MATROSKA,
  MP3,
  MP4,
  OGG,
  QTFF,
  WAVE,
  WEBM,
} from "mediabunny";
import { getUserDataPath as getUserDataPathPreload } from "@app/preload";
import type {
  DecodePriority,
  DecodeWorkerRequest,
  DecodeWorkerResponse,
  DemuxPacketEnvelope,
  DemuxWorkerRequest,
  DemuxWorkerResponse,
  JobId,
  SourceDescriptor,
  SourceId,
} from "./video-decode-protocol";

type FrameCallbackData = {
  canvas: VideoFrame;
  timestamp: number;
  duration: number;
};

export type VideoDecoderAddAssetOptions = {
  mediaInfo: MediaInfo;
  videoDecoderConfig: VideoDecoderConfig;
  folderUuid?: string;
  initialTimestamp?: number;
  onFrame?: (data: FrameCallbackData) => void;
  onError?: (error: Error) => void;
  onReady?: () => void;
  logicalId?: string;
};

type IterationState = {
  requestId: number;
  shouldDelay?: (timestamp: number) => Promise<void>;
  checkCancel?: () => boolean;
  resolve: () => void;
  reject: (err: any) => void;
  frameProcessingPromise: Promise<void>;
};

type PendingSeek = { resolve: () => void; reject: (err: any) => void };

type SeekCandidateFrame = {
  frame: VideoFrame;
  timestampSec: number;
  durationSec: number;
};

type ActiveJobBase = {
  kind: "seek" | "iterate";
  jobId: JobId;
  requestId: number;
  startTimeSec: number;
  endTimeSec: number;
  nextDemuxStartSec: number;
  lastSegmentStartSec: number;
  startAtKeyframe: boolean;
  demuxRequestInFlight: boolean;
  demuxDone: boolean;
  currentSegmentPackets: number;
  segmentPacketLimit: number;
  packetQueue: DemuxPacketEnvelope[];
  decodeInFlight: number;
  flushRequested: boolean;
  flushDone: boolean;
  cancelled: boolean;
  lastQueuedTimestampSec: number;
  lastQueuedDurationSec: number;
};

type SeekJobState = ActiveJobBase & {
  kind: "seek";
  targetTimestampSec: number;
  bestCandidate: SeekCandidateFrame | null;
  resolved: boolean;
};

type IterateJobState = ActiveJobBase & {
  kind: "iterate";
};

type ActiveJobState = SeekJobState | IterateJobState;

type AssetState = {
  decoderId: string;
  sourceId: SourceId;
  sessionId: string;
  logicalClipId: string;
  decodeWorkerIndex: number;
  folderUuid?: string;
  formatStr?: string;
  asset: Asset;
  mediaInfo: MediaInfo;
  videoDecoderConfig: VideoDecoderConfig;
  onFrame?: (data: FrameCallbackData) => void;
  onError?: (error: Error) => void;
  onReady?: () => void;
  initialized: boolean;
  initializedPromise: Promise<void>;
  initializedResolve: (() => void) | null;
  currentRequestId: number;
  pendingSeeks: Map<number, PendingSeek>;
  activeIteration: IterationState | null;
  activeJob: ActiveJobState | null;
  disposed: boolean;
};

type SourceRefState = {
  sourceId: SourceId;
  descriptor: SourceDescriptor;
  refCount: number;
  registered: boolean;
  registerPromise: Promise<void> | null;
};

type PendingUnary = {
  expectedType: string;
  resolve: (msg: any) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

type DecodeWorkerRuntime = {
  worker: Worker;
  pendingUnary: Map<number, PendingUnary>;
  lastDebug: unknown;
  lastError: string | null;
};

type DemuxStreamMeta = {
  jobId: JobId;
  maxPackets: number;
};

type DecodeChunkMeta = {
  assetId: string;
  jobId: JobId;
  workerIndex: number;
};

const REQUEST_TIMEOUT_MS = 20000;
const SEEK_SEGMENT_PACKETS = 120;
const ITERATE_SEGMENT_PACKETS = 48;
const MAX_DECODE_IN_FLIGHT_PER_JOB = 6;
const DEMUX_LOW_WATER = 10;
const EPSILON_SEC = 1e-4;

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

function closeFrameSafe(frame: VideoFrame | null | undefined) {
  if (!frame) return;
  try {
    frame.close();
  } catch {
    // ignore
  }
}

function hashString(value: string): number {
  let h = 0;
  for (let i = 0; i < value.length; i++) {
    h = (h << 5) - h + value.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

function normalizeTargetConfig(
  mediaInfo: MediaInfo,
  videoDecoderConfig: VideoDecoderConfig,
): VideoDecoderConfig {
  const { codedWidth, codedHeight } = videoDecoderConfig;
  let targetWidth = codedWidth || 0;
  let targetHeight = codedHeight || 0;

  if (!targetWidth || !targetHeight) {
    targetWidth = mediaInfo.video?.codedWidth || 0;
    targetHeight = mediaInfo.video?.codedHeight || 0;
  }

  if (targetWidth && targetHeight) {
    const isLandscape = targetWidth >= targetHeight;
    const shortSide = isLandscape ? targetHeight : targetWidth;
    if (shortSide > 720) {
      const scale = 720 / shortSide;
      targetWidth = Math.round(targetWidth * scale);
      targetHeight = Math.round(targetHeight * scale);
      targetWidth = targetWidth - (targetWidth % 2);
      targetHeight = targetHeight - (targetHeight % 2);
    }
  }

  const normalized: VideoDecoderConfig = {
    ...videoDecoderConfig,
    codedWidth: targetWidth,
    codedHeight: targetHeight,
  };
  const normalizedAny = normalized as any;
  if (normalizedAny.alpha == null) {
    normalizedAny.alpha = "keep";
  }
  return normalized;
}

function inferFormatStr(mediaInfo: MediaInfo): string | undefined {
  const fmt = mediaInfo.format;
  if (!fmt) return undefined;
  if (fmt === MP4) return "mp4";
  if (fmt === WEBM) return "webm";
  if (fmt === QTFF) return "mov";
  if (fmt === MATROSKA) return "mkv";
  if (fmt === OGG) return "ogg";
  if (fmt === MP3) return "mp3";
  if (fmt === WAVE) return "wav";
  if (fmt === FLAC) return "flac";
  if (fmt === ADTS) return "aac";
  return undefined;
}

let cachedUserDataPath: string | null = null;
let userDataPathInitPromise: Promise<void> | null = null;

async function ensureUserDataPathLoaded(): Promise<void> {
  if (cachedUserDataPath || userDataPathInitPromise) {
    return userDataPathInitPromise ?? Promise.resolve();
  }
  userDataPathInitPromise = (async () => {
    try {
      const res: any = await getUserDataPathPreload();
      if (res?.success && res.data?.user_data) {
        cachedUserDataPath = res.data.user_data;
      }
    } catch {
      // Best-effort only.
    }
  })();
  return userDataPathInitPromise;
}

void ensureUserDataPathLoaded();

export class VideoDecodeFarmCoordinator {
  private demuxWorker: Worker;
  private decodeWorkers: DecodeWorkerRuntime[] = [];
  private assets = new Map<string, AssetState>();
  private sourceRefs = new Map<SourceId, SourceRefState>();
  private activeJobsById = new Map<JobId, string>();
  private demuxPendingUnary = new Map<number, PendingUnary>();
  private demuxStreamRequestMeta = new Map<number, DemuxStreamMeta>();
  private decodeChunkRequestMeta = new Map<number, DecodeChunkMeta>();
  private nextRequestId = 1;
  private nextJobId = 1;
  private lastDemuxDebug: unknown = null;
  private lastDemuxError: string | null = null;
  private _seekTotal = 0;
  private _seekAccurate = 0;
  private _seekFast = 0;

  constructor(options?: { poolSize?: number }) {
    const poolSize = Math.max(1, Math.min(8, Math.floor(options?.poolSize ?? 1)));

    this.demuxWorker = new Worker(new URL("./demux.worker.ts", import.meta.url), {
      type: "module",
    });
    this.demuxWorker.onmessage = (e: MessageEvent<DemuxWorkerResponse>) => {
      this.handleDemuxMessage(e.data);
    };
    this.demuxWorker.onerror = (err) => {
      // eslint-disable-next-line no-console
      console.error("[VideoDecodeFarm] demux worker error", err.message || err);
    };
    this.demuxWorker.onmessageerror = (err) => {
      // eslint-disable-next-line no-console
      console.error("[VideoDecodeFarm] demux worker message error", err);
    };

    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(new URL("./video-decode.worker.ts", import.meta.url), {
        type: "module",
      });
      const runtime: DecodeWorkerRuntime = {
        worker,
        pendingUnary: new Map(),
        lastDebug: null,
        lastError: null,
      };
      worker.onmessage = (e: MessageEvent<DecodeWorkerResponse>) => {
        this.handleDecodeMessage(i, e.data);
      };
      worker.onerror = (err) => {
        // eslint-disable-next-line no-console
        console.error(`[VideoDecodeFarm] decode worker ${i} error`, err.message || err);
      };
      worker.onmessageerror = (err) => {
        // eslint-disable-next-line no-console
        console.error(`[VideoDecodeFarm] decode worker ${i} message error`, err);
      };
      this.decodeWorkers.push(runtime);
    }
  }

  public hasAsset(assetId: string): boolean {
    return this.assets.has(assetId);
  }

  public addAsset(asset: Asset, options: VideoDecoderAddAssetOptions): void {
    const decoderId = options.logicalId ?? asset.id;
    const normalizedConfig = normalizeTargetConfig(
      options.mediaInfo,
      options.videoDecoderConfig,
    );
    const formatStr = inferFormatStr(options.mediaInfo);
    const decodeWorkerIndex = hashString(decoderId) % this.decodeWorkers.length;

    const state = this.createAssetState({
      decoderId,
      sourceId: asset.id,
      sessionId: decoderId,
      logicalClipId: decoderId,
      decodeWorkerIndex,
      folderUuid: options.folderUuid,
      formatStr,
      asset,
      mediaInfo: options.mediaInfo,
      videoDecoderConfig: normalizedConfig,
      onFrame: options.onFrame,
      onError: options.onError,
      onReady: options.onReady,
    });

    const existing = this.assets.get(decoderId);
    if (existing) {
      this.disposeAsset(decoderId);
    }

    this.assets.set(decoderId, state);
    void this.initializeAsset(state).catch((err: any) => {
      state.disposed = true;
      this.assets.delete(decoderId);
      this.releaseSourceRef(state.sourceId);
      this.postDecode(state.decodeWorkerIndex, {
        type: "disposeSession",
        requestId: this.nextReqId(),
        sessionId: state.sessionId,
      });
      if (state.initializedResolve) {
        state.initializedResolve();
        state.initializedResolve = null;
      }
      this.emitAssetError(state, err?.message ?? "Failed to initialize decode session");
    });
  }

  public updateAssetHandlers(
    assetId: string,
    handlers: {
      onFrame?: (data: FrameCallbackData) => void;
      onError?: (error: Error) => void;
      onReady?: () => void;
    },
  ): void {
    const state = this.assets.get(assetId);
    if (!state) return;
    if (handlers.onFrame) state.onFrame = handlers.onFrame;
    if (handlers.onError) state.onError = handlers.onError;
    if (handlers.onReady) {
      if (state.initialized) {
        handlers.onReady();
      } else {
        state.onReady = handlers.onReady;
      }
    }
  }

  public async seek(
    assetId: string,
    timestamp: number,
    forceAccurate: boolean = false,
  ): Promise<void> {
    const state = this.assets.get(assetId);
    if (!state || state.disposed) return;

    await state.initializedPromise;
    if (state.disposed) return;

    this._seekTotal++;
    if (forceAccurate) this._seekAccurate++;
    else this._seekFast++;

    const requestId = ++state.currentRequestId;

    return new Promise<void>((resolve, reject) => {
      state.pendingSeeks.set(requestId, { resolve, reject });
      this.cancelActiveJob(state, "new-request");
      this.startSeekJob(state, requestId, timestamp, forceAccurate);
    });
  }

  public async iterate(
    assetId: string,
    startTime: number,
    endTime: number,
    shouldDelay?: (timestamp: number) => Promise<void>,
    checkCancel?: () => boolean,
  ): Promise<void> {
    const state = this.assets.get(assetId);
    if (!state || state.disposed) return;

    await state.initializedPromise;
    if (state.disposed) return;

    const requestId = ++state.currentRequestId;

    return new Promise<void>((resolve, reject) => {
      this.cancelActiveJob(state, "new-request");
      state.activeIteration = {
        requestId,
        shouldDelay,
        checkCancel,
        resolve,
        reject,
        frameProcessingPromise: Promise.resolve(),
      };
      this.startIterateJob(state, requestId, startTime, endTime);
    });
  }

  public disposeAsset(assetId: string): void {
    const state = this.assets.get(assetId);
    if (!state) return;

    state.disposed = true;
    this.cancelActiveJob(state, "dispose");

    if (state.activeIteration) {
      state.activeIteration.reject(new Error("Asset disposed"));
      state.activeIteration = null;
    }
    for (const pending of state.pendingSeeks.values()) {
      pending.reject(new Error("Asset disposed"));
    }
    state.pendingSeeks.clear();

    this.releaseSourceRef(state.sourceId);
    this.assets.delete(assetId);

    const reqId = this.nextReqId();
    this.postDecode(state.decodeWorkerIndex, {
      type: "disposeSession",
      requestId: reqId,
      sessionId: state.sessionId,
    });
  }

  public disposeAll(): void {
    for (const assetId of Array.from(this.assets.keys())) {
      this.disposeAsset(assetId);
    }
    this.assets.clear();
    this.sourceRefs.clear();
    this.activeJobsById.clear();
    this.demuxPendingUnary.clear();
    this.demuxStreamRequestMeta.clear();
    this.decodeChunkRequestMeta.clear();
    this.demuxWorker.terminate();
    for (const runtime of this.decodeWorkers) {
      runtime.pendingUnary.clear();
      runtime.worker.terminate();
    }
    this.decodeWorkers = [];
  }

  public getSeekStats(): { total: number; accurate: number; fast: number } {
    return {
      total: this._seekTotal,
      accurate: this._seekAccurate,
      fast: this._seekFast,
    };
  }

  public getDecoderDebugStats(): any {
    const assets: Array<Record<string, any>> = [];
    for (const [assetId, state] of this.assets) {
      assets.push({
        assetId,
        sourceId: state.sourceId,
        sessionId: state.sessionId,
        decodeWorkerIndex: state.decodeWorkerIndex,
        initialized: state.initialized,
        pendingSeeks: state.pendingSeeks.size,
        hasActiveIteration: !!state.activeIteration,
        activeJob: state.activeJob
          ? {
              kind: state.activeJob.kind,
              jobId: state.activeJob.jobId,
              requestId: state.activeJob.requestId,
              queueSize: state.activeJob.packetQueue.length,
              decodeInFlight: state.activeJob.decodeInFlight,
              demuxDone: state.activeJob.demuxDone,
              demuxRequestInFlight: state.activeJob.demuxRequestInFlight,
              flushRequested: state.activeJob.flushRequested,
              flushDone: state.activeJob.flushDone,
              nextDemuxStartSec: state.activeJob.nextDemuxStartSec,
              endTimeSec: state.activeJob.endTimeSec,
            }
          : null,
      });
    }

    return {
      mode: "decode-farm",
      seekStats: this.getSeekStats(),
      demux: {
        pendingUnary: this.demuxPendingUnary.size,
        pendingStreamRequests: this.demuxStreamRequestMeta.size,
        lastDebug: this.lastDemuxDebug,
        lastError: this.lastDemuxError,
      },
      decode: this.decodeWorkers.map((runtime, index) => ({
        index,
        pendingUnary: runtime.pendingUnary.size,
        lastDebug: runtime.lastDebug,
        lastError: runtime.lastError,
      })),
      jobs: {
        activeCount: this.activeJobsById.size,
        pendingDecodeChunks: this.decodeChunkRequestMeta.size,
      },
      assets,
    };
  }

  private createAssetState(params: Omit<AssetState, "initialized" | "initializedPromise" | "initializedResolve" | "currentRequestId" | "pendingSeeks" | "activeIteration" | "activeJob" | "disposed">): AssetState {
    const state: AssetState = {
      ...params,
      initialized: false,
      initializedPromise: Promise.resolve(),
      initializedResolve: null,
      currentRequestId: 0,
      pendingSeeks: new Map(),
      activeIteration: null,
      activeJob: null,
      disposed: false,
    };
    state.initializedPromise = new Promise<void>((resolve) => {
      state.initializedResolve = () => {
        if (state.initialized) return;
        state.initialized = true;
        resolve();
      };
    });
    return state;
  }

  private async initializeAsset(state: AssetState): Promise<void> {
    await ensureUserDataPathLoaded();
    const sourceDescriptor: SourceDescriptor = {
      sourceId: state.sourceId,
      assetId: state.asset.id,
      path: state.asset.path,
      folderUuid: state.folderUuid,
      userDataPath: cachedUserDataPath ?? undefined,
      formatStr: state.formatStr,
    };
    await this.ensureSourceRegistered(state.sourceId, sourceDescriptor);
    if (state.disposed) return;

    await this.requestDecodeUnary(state.decodeWorkerIndex, {
      type: "createSession",
      requestId: this.nextReqId(),
      session: {
        sessionId: state.sessionId,
        logicalClipId: state.logicalClipId,
        sourceId: state.sourceId,
        decoderConfig: state.videoDecoderConfig,
        priority: "interactive",
      },
    }, "sessionReady");

    if (state.disposed) return;

    if (state.initializedResolve) {
      state.initializedResolve();
      state.initializedResolve = null;
    }
    if (state.onReady) {
      const cb = state.onReady;
      state.onReady = undefined;
      cb();
    }
  }

  private async ensureSourceRegistered(
    sourceId: SourceId,
    descriptor: SourceDescriptor,
  ): Promise<void> {
    const existing = this.sourceRefs.get(sourceId);
    if (existing) {
      existing.refCount++;
      existing.descriptor = descriptor;
      if (existing.registerPromise) {
        await existing.registerPromise;
      }
      return;
    }

    const sourceState: SourceRefState = {
      sourceId,
      descriptor,
      refCount: 1,
      registered: false,
      registerPromise: null,
    };
    this.sourceRefs.set(sourceId, sourceState);

    sourceState.registerPromise = (async () => {
      await this.requestDemuxUnary(
        {
          type: "registerSource",
          requestId: this.nextReqId(),
          source: descriptor,
        },
        "sourceReady",
      );
      sourceState.registered = true;
      sourceState.registerPromise = null;
    })();

    try {
      await sourceState.registerPromise;
    } catch (err) {
      this.sourceRefs.delete(sourceId);
      throw err;
    }
  }

  private releaseSourceRef(sourceId: SourceId) {
    const sourceState = this.sourceRefs.get(sourceId);
    if (!sourceState) return;
    sourceState.refCount--;
    if (sourceState.refCount > 0) return;
    this.sourceRefs.delete(sourceId);
    this.postDemux({
      type: "disposeSource",
      requestId: this.nextReqId(),
      sourceId,
    });
  }

  private startSeekJob(
    state: AssetState,
    requestId: number,
    timestampSec: number,
    forceAccurate: boolean,
  ) {
    const safeTarget = Number.isFinite(timestampSec) ? Math.max(0, timestampSec) : 0;
    const endWindowSec = forceAccurate ? 4.0 : 2.0;

    const job: SeekJobState = {
      kind: "seek",
      jobId: this.nextJobId++,
      requestId,
      startTimeSec: safeTarget,
      endTimeSec: safeTarget + endWindowSec,
      nextDemuxStartSec: safeTarget,
      lastSegmentStartSec: safeTarget,
      startAtKeyframe: true,
      demuxRequestInFlight: false,
      demuxDone: false,
      currentSegmentPackets: 0,
      segmentPacketLimit: SEEK_SEGMENT_PACKETS,
      packetQueue: [],
      decodeInFlight: 0,
      flushRequested: false,
      flushDone: false,
      cancelled: false,
      lastQueuedTimestampSec: safeTarget - 10,
      lastQueuedDurationSec: 1 / 30,
      targetTimestampSec: safeTarget,
      bestCandidate: null,
      resolved: false,
    };

    state.activeJob = job;
    this.activeJobsById.set(job.jobId, state.decoderId);
    void this.prepareAndRunJob(state, job, "interactive");
  }

  private startIterateJob(
    state: AssetState,
    requestId: number,
    startTimeSec: number,
    endTimeSec: number,
  ) {
    const safeStart = Number.isFinite(startTimeSec) ? Math.max(0, startTimeSec) : 0;
    const safeEnd = Number.isFinite(endTimeSec) ? Math.max(safeStart, endTimeSec) : safeStart;

    const job: IterateJobState = {
      kind: "iterate",
      jobId: this.nextJobId++,
      requestId,
      startTimeSec: safeStart,
      endTimeSec: safeEnd,
      nextDemuxStartSec: safeStart,
      lastSegmentStartSec: safeStart,
      startAtKeyframe: true,
      demuxRequestInFlight: false,
      demuxDone: false,
      currentSegmentPackets: 0,
      segmentPacketLimit: ITERATE_SEGMENT_PACKETS,
      packetQueue: [],
      decodeInFlight: 0,
      flushRequested: false,
      flushDone: false,
      cancelled: false,
      lastQueuedTimestampSec: safeStart - 10,
      lastQueuedDurationSec: 1 / 30,
    };

    state.activeJob = job;
    this.activeJobsById.set(job.jobId, state.decoderId);
    void this.prepareAndRunJob(state, job, "realtime");
  }

  private async prepareAndRunJob(
    state: AssetState,
    job: ActiveJobState,
    priority: DecodePriority,
  ) {
    try {
      await this.requestDecodeUnary(
        state.decodeWorkerIndex,
        {
          type: "resetSession",
          requestId: this.nextReqId(),
          sessionId: state.sessionId,
        },
        "resetDone",
      );
      if (!this.isActiveJob(state, job.jobId)) return;
      this.requestNextDemuxSegment(state, job, priority);
    } catch (err: any) {
      this.failActiveJob(
        state,
        new Error(err?.message ?? "Failed to prepare decode session"),
      );
    }
  }

  private requestNextDemuxSegment(
    state: AssetState,
    job: ActiveJobState,
    priority: DecodePriority,
  ) {
    if (job.cancelled || job.demuxDone || job.demuxRequestInFlight) return;
    if (job.nextDemuxStartSec > job.endTimeSec + EPSILON_SEC) {
      job.demuxDone = true;
      this.maybeFlushOrComplete(state, job, priority);
      return;
    }

    const reqId = this.nextReqId();
    job.currentSegmentPackets = 0;
    job.demuxRequestInFlight = true;
    job.lastSegmentStartSec = job.nextDemuxStartSec;
    this.demuxStreamRequestMeta.set(reqId, {
      jobId: job.jobId,
      maxPackets: job.segmentPacketLimit,
    });

    this.postDemux({
      type: "streamPackets",
      requestId: reqId,
      sourceId: state.sourceId,
      jobId: job.jobId,
      startTimeSec: job.nextDemuxStartSec,
      endTimeSec: job.endTimeSec,
      startAtKeyframe: job.startAtKeyframe,
      maxPackets: job.segmentPacketLimit,
      priority,
    });
    job.startAtKeyframe = false;
  }

  private pumpJobDecodes(state: AssetState, job: ActiveJobState, priority: DecodePriority) {
    if (!this.isActiveJob(state, job.jobId) || job.cancelled) return;

    while (
      job.packetQueue.length > 0 &&
      job.decodeInFlight < MAX_DECODE_IN_FLIGHT_PER_JOB
    ) {
      const packet = job.packetQueue.shift();
      if (!packet) break;
      this.dispatchDecodeChunk(state, job, packet);
    }

    if (
      !job.demuxDone &&
      !job.demuxRequestInFlight &&
      job.packetQueue.length <= DEMUX_LOW_WATER
    ) {
      this.requestNextDemuxSegment(state, job, priority);
    }

    this.maybeFlushOrComplete(state, job, priority);
  }

  private maybeFlushOrComplete(
    state: AssetState,
    job: ActiveJobState,
    priority: DecodePriority,
  ) {
    if (!this.isActiveJob(state, job.jobId) || job.cancelled) return;
    if (job.kind === "seek" && job.resolved) return;
    if (job.demuxRequestInFlight) return;
    if (!job.demuxDone) return;
    if (job.packetQueue.length > 0) return;
    if (job.decodeInFlight > 0) return;

    if (!job.flushRequested) {
      job.flushRequested = true;
      void this.requestDecodeUnary(
        state.decodeWorkerIndex,
        {
          type: "flushSession",
          requestId: this.nextReqId(),
          sessionId: state.sessionId,
        },
        "flushDone",
      )
        .catch(() => undefined)
        .then(async () => {
          if (!this.isActiveJob(state, job.jobId)) return;
          job.flushDone = true;
          // Give the decoder output callback a tiny window to surface final frames.
          await sleep(0);
          this.completeJob(state, job, priority);
        });
      return;
    }

    if (job.flushDone) {
      this.completeJob(state, job, priority);
    }
  }

  private dispatchDecodeChunk(
    state: AssetState,
    job: ActiveJobState,
    packet: DemuxPacketEnvelope,
  ) {
    if (!this.isActiveJob(state, job.jobId) || job.cancelled) return;

    const requestId = this.nextReqId();
    job.decodeInFlight++;
    this.decodeChunkRequestMeta.set(requestId, {
      assetId: state.decoderId,
      jobId: job.jobId,
      workerIndex: state.decodeWorkerIndex,
    });

    const transfer: Transferable[] = [packet.chunk.data];
    if (packet.alphaChunk) transfer.push(packet.alphaChunk.data);

    this.postDecode(
      state.decodeWorkerIndex,
      {
        type: "decodeChunk",
        requestId,
        sessionId: state.sessionId,
        sourceId: state.sourceId,
        jobId: job.jobId,
        packet,
      },
      transfer,
    );
  }

  private handleDemuxMessage(msg: DemuxWorkerResponse) {
    if (msg.type === "debug") {
      this.lastDemuxDebug = msg;
      if ((window as any).__apexVideoDecoderDebug === true) {
        // eslint-disable-next-line no-console
        console.debug("[VideoDecodeFarm] demux debug", msg);
      }
      return;
    }

    if (msg.type === "error") {
      this.lastDemuxError = msg.error;
      const pending = this.demuxPendingUnary.get(msg.requestId);
      if (pending) {
        clearTimeout(pending.timer);
        this.demuxPendingUnary.delete(msg.requestId);
        pending.reject(new Error(msg.error));
        return;
      }
      if (typeof msg.jobId === "number") {
        const assetId = this.activeJobsById.get(msg.jobId);
        if (assetId) {
          const state = this.assets.get(assetId);
          if (state) {
            this.failActiveJob(state, new Error(msg.error));
            return;
          }
        }
      }
      return;
    }

    const pending = this.demuxPendingUnary.get(msg.requestId);
    if (pending && msg.type === pending.expectedType) {
      clearTimeout(pending.timer);
      this.demuxPendingUnary.delete(msg.requestId);
      pending.resolve(msg);
      return;
    }

    if (msg.type === "packets") {
      const state = this.getAssetStateForJob(msg.jobId);
      if (!state) return;
      const job = state.activeJob;
      if (!job || job.jobId !== msg.jobId || job.cancelled) return;

      const streamMeta = this.demuxStreamRequestMeta.get(msg.requestId);
      if (streamMeta) {
        job.currentSegmentPackets += msg.packets.length;
      }

      for (const packet of msg.packets) {
        if (packet.timestampSec <= job.lastQueuedTimestampSec + EPSILON_SEC) {
          continue;
        }
        job.packetQueue.push(packet);
        job.lastQueuedTimestampSec = packet.timestampSec;
        job.lastQueuedDurationSec = Math.max(packet.durationSec || 0, 1 / 120);
        job.nextDemuxStartSec =
          job.lastQueuedTimestampSec + job.lastQueuedDurationSec + EPSILON_SEC;
      }

      const priority: DecodePriority = job.kind === "iterate" ? "realtime" : "interactive";
      this.pumpJobDecodes(state, job, priority);
      return;
    }

    if (msg.type === "streamDone") {
      const state = this.getAssetStateForJob(msg.jobId);
      const streamMeta = this.demuxStreamRequestMeta.get(msg.requestId);
      this.demuxStreamRequestMeta.delete(msg.requestId);
      if (!state) return;
      const job = state.activeJob;
      if (!job || job.jobId !== msg.jobId || job.cancelled) return;

      job.demuxRequestInFlight = false;
      const maxPackets = streamMeta?.maxPackets ?? job.segmentPacketLimit;
      const hadMore = job.currentSegmentPackets >= maxPackets;
      const canContinue = job.nextDemuxStartSec <= job.endTimeSec + EPSILON_SEC;
      const progressed = job.nextDemuxStartSec > job.lastSegmentStartSec + EPSILON_SEC;
      if (!progressed && canContinue) {
        // Some containers can repeatedly return the same packet around boundaries.
        // Nudge forward to avoid stalling a seek/iterate job on one timestamp.
        job.nextDemuxStartSec = Math.min(
          job.endTimeSec + EPSILON_SEC,
          job.lastSegmentStartSec + 0.25,
        );
      }
      if (!hadMore || !canContinue) {
        job.demuxDone = true;
      }

      const priority: DecodePriority = job.kind === "iterate" ? "realtime" : "interactive";
      this.pumpJobDecodes(state, job, priority);
      return;
    }

    if (msg.type === "jobCancelled") {
      // Best-effort informational response.
      return;
    }
  }

  private handleDecodeMessage(workerIndex: number, msg: DecodeWorkerResponse) {
    const runtime = this.decodeWorkers[workerIndex];
    if (!runtime) return;

    if (msg.type === "debug") {
      runtime.lastDebug = msg;
      if ((window as any).__apexVideoDecoderDebug === true) {
        // eslint-disable-next-line no-console
        console.debug(`[VideoDecodeFarm] decode debug w${workerIndex}`, msg);
      }
      return;
    }

    if (msg.type === "error") {
      runtime.lastError = msg.error;
      const pending = msg.requestId ? runtime.pendingUnary.get(msg.requestId) : undefined;
      if (pending) {
        clearTimeout(pending.timer);
        runtime.pendingUnary.delete(msg.requestId);
        pending.reject(new Error(msg.error));
        return;
      }

      const chunkMeta = msg.requestId
        ? this.decodeChunkRequestMeta.get(msg.requestId)
        : undefined;
      if (chunkMeta) {
        this.decodeChunkRequestMeta.delete(msg.requestId as number);
        const state = this.assets.get(chunkMeta.assetId);
        if (state?.activeJob?.jobId === chunkMeta.jobId) {
          state.activeJob.decodeInFlight = Math.max(0, state.activeJob.decodeInFlight - 1);
          this.failActiveJob(state, new Error(msg.error));
        }
        return;
      }

      const state = msg.sessionId ? this.assets.get(msg.sessionId) : undefined;
      if (state) {
        this.emitAssetError(state, msg.error);
      }
      return;
    }

    const pending = runtime.pendingUnary.get(msg.requestId);
    if (pending && msg.type === pending.expectedType) {
      clearTimeout(pending.timer);
      runtime.pendingUnary.delete(msg.requestId);
      pending.resolve(msg);
      return;
    }

    if (msg.type === "decodeDone") {
      const meta = this.decodeChunkRequestMeta.get(msg.requestId);
      if (!meta) return;
      this.decodeChunkRequestMeta.delete(msg.requestId);
      const state = this.assets.get(meta.assetId);
      if (!state) return;
      const job = state.activeJob;
      if (!job || job.jobId !== meta.jobId || job.cancelled) return;
      job.decodeInFlight = Math.max(0, job.decodeInFlight - 1);
      const priority: DecodePriority = job.kind === "iterate" ? "realtime" : "interactive";
      this.pumpJobDecodes(state, job, priority);
      return;
    }

    if (msg.type === "frame") {
      const state = this.assets.get(msg.sessionId);
      if (!state) {
        closeFrameSafe(msg.frame);
        return;
      }
      const job = state.activeJob;
      if (!job || job.jobId !== msg.jobId || job.cancelled) {
        closeFrameSafe(msg.frame);
        return;
      }

      if (job.kind === "seek") {
        this.handleSeekFrame(state, job, msg.frame, msg.timestampSec, msg.durationSec);
      } else {
        this.handleIterateFrame(state, job, msg.frame, msg.timestampSec, msg.durationSec);
      }
      return;
    }

    if (msg.type === "jobCancelled") {
      return;
    }
  }

  private handleSeekFrame(
    state: AssetState,
    job: SeekJobState,
    frame: VideoFrame,
    timestampSec: number,
    durationSec: number,
  ) {
    if (!this.isActiveJob(state, job.jobId) || job.cancelled) {
      closeFrameSafe(frame);
      return;
    }

    if (timestampSec + EPSILON_SEC >= job.targetTimestampSec) {
      if (job.bestCandidate) {
        closeFrameSafe(job.bestCandidate.frame);
        job.bestCandidate = null;
      }
      this.dispatchFrameToClient(state, frame, timestampSec, durationSec);
      job.resolved = true;
      this.resolveSeekRequest(state, job.requestId);
      this.stopJobWorkers(state, job);
      this.clearActiveJob(state, job.jobId);
      return;
    }

    if (job.bestCandidate) {
      closeFrameSafe(job.bestCandidate.frame);
    }
    job.bestCandidate = { frame, timestampSec, durationSec };
  }

  private handleIterateFrame(
    state: AssetState,
    job: IterateJobState,
    frame: VideoFrame,
    timestampSec: number,
    durationSec: number,
  ) {
    const iteration = state.activeIteration;
    if (!iteration || iteration.requestId !== job.requestId) {
      closeFrameSafe(frame);
      return;
    }

    iteration.frameProcessingPromise = iteration.frameProcessingPromise
      .then(async () => {
        if (!this.isActiveJob(state, job.jobId) || job.cancelled) {
          closeFrameSafe(frame);
          return;
        }

        if (iteration.checkCancel && !iteration.checkCancel()) {
          closeFrameSafe(frame);
          this.cancelActiveJob(state, "new-request");
          return;
        }

        if (iteration.shouldDelay) {
          await iteration.shouldDelay(timestampSec);
        }

        if (iteration.checkCancel && !iteration.checkCancel()) {
          closeFrameSafe(frame);
          this.cancelActiveJob(state, "new-request");
          return;
        }

        this.dispatchFrameToClient(state, frame, timestampSec, durationSec);
      })
      .catch((err) => {
        closeFrameSafe(frame);
        this.failActiveJob(state, err instanceof Error ? err : new Error(String(err)));
      });
  }

  private completeJob(
    state: AssetState,
    job: ActiveJobState,
    priority: DecodePriority,
  ) {
    if (!this.isActiveJob(state, job.jobId) || job.cancelled) return;

    if (job.kind === "seek") {
      if (!job.resolved) {
        if (job.bestCandidate) {
          const candidate = job.bestCandidate;
          job.bestCandidate = null;
          this.dispatchFrameToClient(
            state,
            candidate.frame,
            candidate.timestampSec,
            candidate.durationSec,
          );
          this.resolveSeekRequest(state, job.requestId);
        } else {
          this.resolveSeekRequest(state, job.requestId);
        }
      }
      this.clearActiveJob(state, job.jobId);
      return;
    }

    const iteration = state.activeIteration;
    if (iteration && iteration.requestId === job.requestId) {
      iteration.frameProcessingPromise
        .then(() => {
          iteration.resolve();
        })
        .catch((err) => {
          iteration.reject(err instanceof Error ? err : new Error(String(err)));
        })
        .finally(() => {
          if (state.activeIteration?.requestId === iteration.requestId) {
            state.activeIteration = null;
          }
        });
    }

    this.clearActiveJob(state, job.jobId);

    // Keep pulling if a caller immediately started another job while finalize was pending.
    const nextJob = state.activeJob;
    if (nextJob) {
      this.pumpJobDecodes(state, nextJob, priority);
    }
  }

  private clearActiveJob(state: AssetState, jobId: JobId) {
    if (state.activeJob?.jobId !== jobId) return;
    const job = state.activeJob;
    if (job.kind === "seek" && job.bestCandidate) {
      closeFrameSafe(job.bestCandidate.frame);
      job.bestCandidate = null;
    }
    state.activeJob = null;
    this.activeJobsById.delete(jobId);
    this.cleanupJobRequestMetadata(jobId);
  }

  private cleanupJobRequestMetadata(jobId: JobId) {
    for (const [requestId, meta] of this.demuxStreamRequestMeta) {
      if (meta.jobId === jobId) {
        this.demuxStreamRequestMeta.delete(requestId);
      }
    }
    for (const [requestId, meta] of this.decodeChunkRequestMeta) {
      if (meta.jobId === jobId) {
        this.decodeChunkRequestMeta.delete(requestId);
      }
    }
  }

  private stopJobWorkers(state: AssetState, job: ActiveJobState) {
    const demuxCancelRequest: DemuxWorkerRequest = {
      type: "cancelJob",
      requestId: this.nextReqId(),
      sourceId: state.sourceId,
      jobId: job.jobId,
    };
    this.postDemux(demuxCancelRequest);

    this.postDecode(state.decodeWorkerIndex, {
      type: "cancelJob",
      requestId: this.nextReqId(),
      sessionId: state.sessionId,
      jobId: job.jobId,
    });
  }

  private cancelActiveJob(
    state: AssetState,
    mode: "new-request" | "dispose" | "error",
  ) {
    const job = state.activeJob;
    if (!job) return;

    job.cancelled = true;
    this.stopJobWorkers(state, job);

    for (const packet of job.packetQueue) {
      // packet ArrayBuffers are GC-managed; just drop references.
      void packet;
    }
    job.packetQueue.length = 0;

    if (job.kind === "seek") {
      if (job.bestCandidate) {
        closeFrameSafe(job.bestCandidate.frame);
        job.bestCandidate = null;
      }
      if (!job.resolved) {
        const pending = state.pendingSeeks.get(job.requestId);
        if (pending) {
          state.pendingSeeks.delete(job.requestId);
          if (mode === "new-request") {
            pending.reject(new Error("Seek superseded"));
          } else if (mode === "dispose") {
            pending.reject(new Error("Asset disposed"));
          } else {
            pending.reject(new Error("Seek cancelled"));
          }
        }
      }
    } else if (state.activeIteration && state.activeIteration.requestId === job.requestId) {
      const iteration = state.activeIteration;
      if (mode === "new-request") {
        iteration.resolve();
      } else if (mode === "dispose") {
        iteration.reject(new Error("Asset disposed"));
      } else {
        iteration.reject(new Error("Iteration cancelled"));
      }
      state.activeIteration = null;
    }

    this.clearActiveJob(state, job.jobId);
  }

  private failActiveJob(state: AssetState, error: Error) {
    const job = state.activeJob;
    if (!job) {
      this.emitAssetError(state, error.message);
      return;
    }

    job.cancelled = true;
    this.stopJobWorkers(state, job);

    if (job.kind === "seek") {
      const pending = state.pendingSeeks.get(job.requestId);
      if (pending) {
        state.pendingSeeks.delete(job.requestId);
        pending.reject(error);
      }
    } else if (state.activeIteration && state.activeIteration.requestId === job.requestId) {
      state.activeIteration.reject(error);
      state.activeIteration = null;
    }

    this.clearActiveJob(state, job.jobId);
    this.emitAssetError(state, error.message);
  }

  private resolveSeekRequest(state: AssetState, requestId: number) {
    const pending = state.pendingSeeks.get(requestId);
    if (!pending) return;
    state.pendingSeeks.delete(requestId);
    pending.resolve();
  }

  private dispatchFrameToClient(
    state: AssetState,
    frame: VideoFrame,
    timestampSec: number,
    durationSec: number,
  ) {
    try {
      if (state.onFrame) {
        state.onFrame({
          canvas: frame,
          timestamp: timestampSec,
          duration: durationSec,
        });
      }
    } catch (err: any) {
      this.emitAssetError(state, err?.message ?? "onFrame callback failed");
    } finally {
      closeFrameSafe(frame);
    }
  }

  private emitAssetError(state: AssetState, error: string) {
    if (state.onError) {
      state.onError(new Error(error));
      return;
    }
    // eslint-disable-next-line no-console
    console.error("[VideoDecodeFarm]", state.decoderId, error);
  }

  private getAssetStateForJob(jobId: JobId): AssetState | null {
    const assetId = this.activeJobsById.get(jobId);
    if (!assetId) return null;
    return this.assets.get(assetId) ?? null;
  }

  private isActiveJob(state: AssetState, jobId: JobId): boolean {
    return !!state.activeJob && state.activeJob.jobId === jobId;
  }

  private async requestDemuxUnary(
    msg: DemuxWorkerRequest,
    expectedType: DemuxWorkerResponse["type"],
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.demuxPendingUnary.delete(msg.requestId);
        reject(
          new Error(
            `Demux request timed out (${msg.type} -> ${expectedType}, requestId=${msg.requestId})`,
          ),
        );
      }, REQUEST_TIMEOUT_MS);

      this.demuxPendingUnary.set(msg.requestId, {
        expectedType,
        resolve,
        reject,
        timer,
      });
      this.postDemux(msg);
    });
  }

  private async requestDecodeUnary(
    workerIndex: number,
    msg: DecodeWorkerRequest,
    expectedType: DecodeWorkerResponse["type"],
  ): Promise<any> {
    const runtime = this.decodeWorkers[workerIndex];
    if (!runtime) {
      throw new Error(`Decode worker ${workerIndex} is unavailable`);
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        runtime.pendingUnary.delete(msg.requestId);
        reject(
          new Error(
            `Decode request timed out (${msg.type} -> ${expectedType}, requestId=${msg.requestId})`,
          ),
        );
      }, REQUEST_TIMEOUT_MS);

      runtime.pendingUnary.set(msg.requestId, {
        expectedType,
        resolve,
        reject,
        timer,
      });
      this.postDecode(workerIndex, msg);
    });
  }

  private postDemux(msg: DemuxWorkerRequest, transfer: Transferable[] = []) {
    this.demuxWorker.postMessage(msg, transfer);
  }

  private postDecode(
    workerIndex: number,
    msg: DecodeWorkerRequest,
    transfer: Transferable[] = [],
  ) {
    const runtime = this.decodeWorkers[workerIndex];
    runtime.worker.postMessage(msg, transfer);
  }

  private nextReqId(): number {
    const id = this.nextRequestId;
    this.nextRequestId += 1;
    return id;
  }
}
