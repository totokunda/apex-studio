import { Asset, MediaInfo } from "../types";
import {
    UrlSource,
    BlobSource,
    MP4,
    WEBM,
    QTFF,
    OGG,
    MATROSKA,
    MP3,
    WAVE,
    FLAC,
    ADTS,
} from "mediabunny";
import {
    getUserDataPath as getUserDataPathPreload,
    createNativeDecoder,
    configureNativeDecoder,
    seekNativeDecoder,
    iterateNativeDecoder,
    ackNativeFrame,
    cancelNativeDecoder,
    disposeNativeDecoder,
    disposeAllNativeDecoders,
    isNativeDecoderAvailable,
    resolveNativeDecoderPath,
} from "@app/preload";
import { NativeFrameUploader } from "./native-frame-uploader";

// ---------------------------------------------------------------------------
// Shared canvas type used by both native (HTMLCanvasElement) and worker
// (VideoFrame) paths. Consumers use ctx.drawImage() which accepts both.
// ---------------------------------------------------------------------------
type DecodedCanvas = VideoFrame | HTMLCanvasElement;

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

function computeFormatStr(fmt: any): string | undefined {
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

function normalizeDimensions(
    videoDecoderConfig: VideoDecoderConfig,
    mediaInfo: MediaInfo,
): { targetWidth: number; targetHeight: number } {
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

    return { targetWidth, targetHeight };
}

// ---------------------------------------------------------------------------
// Native decoder state (per-asset)
// ---------------------------------------------------------------------------

interface NativeDecoderAssetState {
    nativeHandle: number;
    uploader: NativeFrameUploader;
}

type NativePixelFormat = "rgba" | "nv12";

// Detect once at module load whether the native path is usable.
const nativeAvailable: boolean = (() => {
    try {
        return isNativeDecoderAvailable();
    } catch {
        return false;
    }
})();

// ---------------------------------------------------------------------------
// VideoFrameDecoder — single-asset decoder
// ---------------------------------------------------------------------------

interface VideoFrameDecoderOptions {
    mediaInfo: MediaInfo;
    videoDecoderConfig: VideoDecoderConfig;
    onFrame: (data: { canvas: DecodedCanvas; timestamp: number; duration: number }) => void;
    onError: (error: Error) => void;
    initialTimestamp?: number;
    onReady?: () => void;
}

export class VideoFrameDecoder {
    private worker: Worker | null = null;
    private currentRequestId = 0;
    private initialized = false;
    private initializedPromise: Promise<void>;
    private initializedResolve: (() => void) | null = null;
    private onReadyCallback?: () => void;
    private onFrameCallback: (data: { canvas: DecodedCanvas; timestamp: number; duration: number }) => void;
    private onErrorCallback: (error: Error) => void;
    private readonly useNative: boolean;
    private native: NativeDecoderAssetState | null = null;

    private activeIteration: {
        requestId: number;
        shouldDelay?: (timestamp: number) => Promise<void>;
        checkCancel?: () => boolean;
        resolve: () => void;
        reject: (err: any) => void;
        frameProcessingPromise: Promise<void>;
    } | null = null;

    constructor(options: VideoFrameDecoderOptions) {
        this.onFrameCallback = options.onFrame;
        this.onErrorCallback = options.onError;
        this.onReadyCallback = options.onReady;
        this.useNative = nativeAvailable;

        this.initializedPromise = new Promise<void>((resolve) => {
            this.initializedResolve = () => {
                if (this.initialized) return;
                this.initialized = true;
                resolve();
            };
        });

        const { targetWidth, targetHeight } = normalizeDimensions(
            options.videoDecoderConfig,
            options.mediaInfo,
        );

        if (this.useNative) {
            this.initNative(options, targetWidth, targetHeight);
        } else {
            this.initWorker(options, targetWidth, targetHeight);
        }
    }

    // -----------------------------------------------------------------------
    // Native path init
    // -----------------------------------------------------------------------

    private async initNative(
        options: VideoFrameDecoderOptions,
        targetWidth: number,
        targetHeight: number,
    ): Promise<void> {
        try {
            if (targetWidth <= 0 || targetHeight <= 0) {
                throw new Error("Native decoder requires non-zero target dimensions");
            }
            const handle = createNativeDecoder();
            const uploader = new NativeFrameUploader();

            this.native = { nativeHandle: handle, uploader };

            const filePath = await resolveNativeDecoderPath(options.mediaInfo.path);

            configureNativeDecoder(
                handle,
                filePath,
                targetWidth,
                targetHeight,
                (frameBytes, width, height, timestamp, duration, requestId, bufferIndex, pixelFormat) => {
                    this.handleNativeFrame(
                        frameBytes,
                        width,
                        height,
                        timestamp,
                        duration,
                        requestId,
                        bufferIndex,
                        pixelFormat ?? "rgba",
                    );
                },
                (message) => {
                    if (this.activeIteration) {
                        this.activeIteration.reject(new Error(message));
                        this.activeIteration = null;
                    } else {
                        this.onErrorCallback(new Error(message));
                    }
                },
                () => {
                    if (this.initializedResolve) {
                        this.initializedResolve();
                        this.initializedResolve = null;
                    }
                    if (this.onReadyCallback) {
                        this.onReadyCallback();
                        this.onReadyCallback = undefined;
                    }
                },
                {
                    outputFormat: "nv12",
                    copyFrameData: false,
                    manualAck: true,
                    preferSharedBufferPool:
                        typeof window !== "undefined" && window.crossOriginIsolated,
                },
            );
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[VideoFrameDecoder] Native decoder init failed, falling back to worker:", err);
            (this as any).useNative = false;
            this.native = null;
            this.initWorker(options, targetWidth, targetHeight);
        }
    }

    private handleNativeFrame(
        frameBytes: Uint8Array,
        width: number,
        height: number,
        timestamp: number,
        duration: number,
        requestId: number,
        bufferIndex?: number,
        pixelFormat: NativePixelFormat = "rgba",
    ): void {
        const native = this.native!;
        const ack = () => {
            if (typeof bufferIndex === "number" && bufferIndex >= 0) {
                ackNativeFrame(native.nativeHandle, bufferIndex);
            }
        };
        const upload = (): HTMLCanvasElement => {
            if (pixelFormat === "nv12") {
                return native.uploader.uploadNV12(frameBytes, width, height);
            }
            return native.uploader.upload(frameBytes, width, height);
        };

        // Iteration frames
        if (this.activeIteration && requestId === this.activeIteration.requestId) {
            const iteration = this.activeIteration;

            iteration.frameProcessingPromise = iteration.frameProcessingPromise
                .then(async () => {
                    try {
                        if (iteration.checkCancel && !iteration.checkCancel()) {
                            return;
                        }
                        if (iteration.shouldDelay) {
                            await iteration.shouldDelay(timestamp);
                        }
                        if (iteration.checkCancel && !iteration.checkCancel()) {
                            return;
                        }

                        const canvas = upload();
                        this.onFrameCallback({ canvas, timestamp, duration });
                    } finally {
                        ack();
                    }
                })
                .catch((err) => {
                    console.error("Native frame processing error", err);
                });
            return;
        }

        // Seek/preview frames
        if (requestId !== undefined && requestId !== this.currentRequestId) {
            ack();
            return;
        }

        try {
            const canvas = upload();
            this.onFrameCallback({ canvas, timestamp, duration });
        } finally {
            ack();
        }
    }

    // -----------------------------------------------------------------------
    // Worker path init
    // -----------------------------------------------------------------------

    private initWorker(
        options: VideoFrameDecoderOptions,
        targetWidth: number,
        targetHeight: number,
    ): void {
        this.worker = new Worker(new URL("./video-decoder.worker.ts", import.meta.url), {
            type: "module",
        });

        this.worker.onmessage = (e) => {
            const msg = e.data;

            if (msg.type === "frame") {
                if (this.activeIteration && msg.requestId === this.activeIteration.requestId) {
                    const iteration = this.activeIteration;

                    iteration.frameProcessingPromise = iteration.frameProcessingPromise
                        .then(async () => {
                            if (iteration.checkCancel && !iteration.checkCancel()) {
                                msg.frame.close();
                                return;
                            }
                            if (iteration.shouldDelay) {
                                await iteration.shouldDelay(msg.timestamp);
                            }
                            if (iteration.checkCancel && !iteration.checkCancel()) {
                                msg.frame.close();
                                return;
                            }

                            this.onFrameCallback({
                                canvas: msg.frame,
                                timestamp: msg.timestamp,
                                duration: msg.duration,
                            });
                            msg.frame.close();
                            this.worker!.postMessage({ type: "ack", requestId: msg.requestId });
                        })
                        .catch((err) => {
                            console.error("Frame processing error", err);
                            msg.frame.close();
                        });
                    return;
                }

                if (msg.requestId !== undefined && msg.requestId !== this.currentRequestId) {
                    msg.frame.close();
                    return;
                }

                this.onFrameCallback({
                    canvas: msg.frame,
                    timestamp: msg.timestamp,
                    duration: msg.duration,
                });
                msg.frame.close();
            } else if (msg.type === "ready") {
                if (this.initializedResolve) {
                    this.initializedResolve();
                    this.initializedResolve = null;
                }
                if (this.onReadyCallback) {
                    this.onReadyCallback();
                    this.onReadyCallback = undefined;
                }
            } else if (msg.type === "seekDone") {
                if (this.initializedResolve) {
                    this.initializedResolve();
                    this.initializedResolve = null;
                }
            } else if (msg.type === "error") {
                if (this.activeIteration && msg.requestId === this.activeIteration.requestId) {
                    this.activeIteration.reject(new Error(msg.error));
                    this.activeIteration = null;
                } else {
                    this.onErrorCallback(new Error(msg.error));
                }
            } else if (msg.type === "iterateDone") {
                if (this.activeIteration && msg.requestId === this.activeIteration.requestId) {
                    const iteration = this.activeIteration;
                    iteration.frameProcessingPromise.then(() => {
                        iteration.resolve();
                    });
                    this.activeIteration = null;
                }
            }
        };

        const config = {
            ...options.videoDecoderConfig,
            codedWidth: targetWidth,
            codedHeight: targetHeight,
        };

        let sourceConfig: { type: "url" | "blob"; url?: string; blob?: Blob } | null = null;

        if (options.mediaInfo.originalInput) {
            const src = (options.mediaInfo.originalInput).source;
            if (src instanceof UrlSource) {
                sourceConfig = { type: "url", url: (src as any)._url.toString() };
            } else if (src instanceof BlobSource) {
                sourceConfig = { type: "blob", blob: (src as any)._blob };
            }
        }

        if (!sourceConfig) {
            const path = options.mediaInfo.path;
            if (path.startsWith("blob:")) {
                sourceConfig = { type: "url", url: path };
            } else {
                sourceConfig = { type: "url", url: path };
            }
        }

        const formatStr = computeFormatStr(options.mediaInfo.format);

        this.worker.postMessage({
            type: "configure",
            config: {
                videoDecoderConfig: config,
                source: sourceConfig,
                formatStr,
                initialTimestamp: options.initialTimestamp ?? 0,
            },
            requestId: ++this.currentRequestId,
        });
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    public isReady(): boolean {
        return this.initialized;
    }

    public async waitUntilReady(): Promise<void> {
        if (this.initialized) return;
        return this.initializedPromise;
    }

    public async seek(timestamp: number, forceAccurate: boolean = false): Promise<void> {
        const reqId = ++this.currentRequestId;

        if (this.useNative && this.native) {
            return seekNativeDecoder(this.native.nativeHandle, timestamp, forceAccurate, reqId);
        }

        this.worker!.postMessage({
            type: "seek",
            timestamp,
            forceAccurate,
            requestId: reqId,
        });
    }

    public async iterate(
        startTime: number,
        endTime: number,
        shouldDelay?: (timestamp: number) => Promise<void>,
        checkCancel?: () => boolean,
    ): Promise<void> {
        const reqId = ++this.currentRequestId;

        if (this.useNative && this.native) {
            return new Promise<void>((resolve, reject) => {
                this.activeIteration = {
                    requestId: reqId,
                    shouldDelay,
                    checkCancel,
                    resolve,
                    reject,
                    frameProcessingPromise: Promise.resolve(),
                };

                iterateNativeDecoder(this.native!.nativeHandle, startTime, endTime, reqId)
                    .then(() => {
                        if (this.activeIteration && this.activeIteration.requestId === reqId) {
                            const iteration = this.activeIteration;
                            this.activeIteration = null;
                            iteration.frameProcessingPromise.then(() => iteration.resolve());
                        }
                    })
                    .catch((err) => {
                        if (this.activeIteration && this.activeIteration.requestId === reqId) {
                            this.activeIteration.reject(err);
                            this.activeIteration = null;
                        }
                    });
            });
        }

        return new Promise<void>((resolve, reject) => {
            this.activeIteration = {
                requestId: reqId,
                shouldDelay,
                checkCancel,
                resolve,
                reject,
                frameProcessingPromise: Promise.resolve(),
            };

            this.worker!.postMessage({
                type: "iterate",
                startTime,
                endTime,
                requestId: reqId,
            });
        });
    }

    public dispose() {
        this.currentRequestId++;

        if (this.native) {
            cancelNativeDecoder(this.native.nativeHandle);
            disposeNativeDecoder(this.native.nativeHandle);
            this.native.uploader.dispose();
            this.native = null;
        }

        if (this.worker) {
            this.worker.postMessage({ type: "dispose" });
            this.worker.terminate();
            this.worker = null;
        }
    }
}

// ---------------------------------------------------------------------------
// VideoDecoderManager — multi-asset decoder
// ---------------------------------------------------------------------------

interface VideoDecoderManagerAssetState {
    asset: Asset;
    mediaInfo: MediaInfo;
    videoDecoderConfig: VideoDecoderConfig;
    onFrame?: (data: { canvas: DecodedCanvas; timestamp: number; duration: number }) => void;
    onError?: (error: Error) => void;
    onReady?: () => void;
    initialized: boolean;
    initializedPromise: Promise<void>;
    initializedResolve: (() => void) | null;
    currentRequestId: number;
    activeIteration: {
        requestId: number;
        shouldDelay?: (timestamp: number) => Promise<void>;
        checkCancel?: () => boolean;
        resolve: () => void;
        reject: (err: any) => void;
        frameProcessingPromise: Promise<void>;
    } | null;
    pendingSeeks: Map<number, { resolve: () => void; reject: (err: any) => void }>;
    native?: NativeDecoderAssetState;
}

// Cached Electron userData path, resolved once via preload and then reused for
// all worker configurations. This lets the worker cheaply detect when an
// incoming asset path already lives under the userData tree and prefer the
// faster app://user-data host over app://apex-cache.
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
            // Best-effort only; keep cachedUserDataPath as null on failure.
        }
    })();
    return userDataPathInitPromise;
}

// Kick off userData resolution in the background; we don't block decoder usage
// on this, and fall back to previous heuristics until it's available.
void ensureUserDataPathLoaded();

export class VideoDecoderManager {
    private _worker: Worker | null = null;
    private assets = new Map<string, VideoDecoderManagerAssetState>();
    private readonly useNative: boolean;

    private createAssetState(params: {
        asset: Asset;
        mediaInfo: MediaInfo;
        videoDecoderConfig: VideoDecoderConfig;
        onFrame?: (data: { canvas: DecodedCanvas; timestamp: number; duration: number }) => void;
        onError?: (error: Error) => void;
        onReady?: () => void;
    }): VideoDecoderManagerAssetState {
        const state: VideoDecoderManagerAssetState = {
            asset: params.asset,
            mediaInfo: params.mediaInfo,
            videoDecoderConfig: params.videoDecoderConfig,
            onFrame: params.onFrame,
            onError: params.onError,
            onReady: params.onReady,
            initialized: false,
            initializedPromise: Promise.resolve() as Promise<void>,
            initializedResolve: null,
            currentRequestId: 0,
            activeIteration: null,
            pendingSeeks: new Map(),
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

    constructor() {
        this.useNative = nativeAvailable;

        if (!this.useNative) {
            // Eagerly create the worker for the WebCodecs fallback path.
            this.ensureWorker();
        }
    }

    // -----------------------------------------------------------------------
    // Lazy worker creation
    // -----------------------------------------------------------------------

    private ensureWorker(): Worker {
        if (this._worker) return this._worker;

        // IMPORTANT: keep the inline `new Worker(new URL(...))` form.
        // Vite's worker transform relies on recognizing this pattern.
        this._worker = new Worker(new URL("./video-decoder.worker.ts", import.meta.url), {
            type: "module",
        });

        this._worker.onerror = (err) => {
            console.error("[VideoDecoderManager] worker error", err.message || err);
        };
        this._worker.onmessageerror = (err) => {
            console.error("[VideoDecoderManager] worker message error", err);
        };

        this._worker.onmessage = (e: MessageEvent) => {
            this.handleWorkerMessage(e.data);
        };

        return this._worker;
    }

    private handleWorkerMessage(msg: any): void {
        const assetId: string | undefined = msg.assetId;
        const client = assetId ? this.assets.get(assetId) : undefined;

        if (!client) {
            if (msg.type === "frame" && msg.frame) {
                try { msg.frame.close(); } catch { /* ignore */ }
            }
            return;
        }

        if (msg.type === "frame") {
            if (client.activeIteration && msg.requestId === client.activeIteration.requestId) {
                const iteration = client.activeIteration;

                iteration.frameProcessingPromise = iteration.frameProcessingPromise
                    .then(async () => {
                        if (iteration.checkCancel && !iteration.checkCancel()) {
                            msg.frame.close();
                            return;
                        }
                        if (iteration.shouldDelay) {
                            await iteration.shouldDelay(msg.timestamp);
                        }
                        if (iteration.checkCancel && !iteration.checkCancel()) {
                            msg.frame.close();
                            return;
                        }

                        if (client.onFrame) {
                            client.onFrame({
                                canvas: msg.frame,
                                timestamp: msg.timestamp,
                                duration: msg.duration,
                            });
                        }
                        msg.frame.close();

                        this.ensureWorker().postMessage({
                            type: "ack",
                            assetId,
                            requestId: msg.requestId,
                        });
                    })
                    .catch((err: any) => {
                        console.error("Frame processing error (manager)", err);
                        msg.frame.close();
                    });
                return;
            }

            if (msg.requestId !== undefined && msg.requestId !== client.currentRequestId) {
                msg.frame.close();
                return;
            }

            if (client.onFrame) {
                client.onFrame({
                    canvas: msg.frame,
                    timestamp: msg.timestamp,
                    duration: msg.duration,
                });
            }
            msg.frame.close();
        } else if (msg.type === "ready") {
            if (!client.initialized) {
                client.initialized = true;
                if (client.initializedResolve) {
                    client.initializedResolve();
                    client.initializedResolve = null;
                }
                if (client.onReady) {
                    client.onReady();
                    client.onReady = undefined;
                }
            }
        } else if (msg.type === "seekDone") {
            const pending = client.pendingSeeks.get(msg.requestId);
            if (pending) {
                client.pendingSeeks.delete(msg.requestId);
                pending.resolve();
            }
            if (!client.initialized) {
                client.initialized = true;
                if (client.initializedResolve) {
                    client.initializedResolve();
                    client.initializedResolve = null;
                }
                if (client.onReady) {
                    client.onReady();
                    client.onReady = undefined;
                }
            }
        } else if (msg.type === "error") {
            if (client.activeIteration && msg.requestId === client.activeIteration.requestId) {
                client.activeIteration.reject(new Error(msg.error));
                client.activeIteration = null;
            } else {
                const pending = msg.requestId ? client.pendingSeeks.get(msg.requestId) : undefined;
                if (pending) {
                    client.pendingSeeks.delete(msg.requestId);
                    pending.reject(new Error(msg.error));
                } else if (client.onError) {
                    client.onError(new Error(msg.error));
                } else {
                    console.error("[VideoDecoderManager] Unhandled error", msg.error);
                }
            }
        } else if (msg.type === "iterateDone") {
            if (client.activeIteration && msg.requestId === client.activeIteration.requestId) {
                const iteration = client.activeIteration;
                iteration.frameProcessingPromise.then(() => {
                    iteration.resolve();
                });
                client.activeIteration = null;
            }
        }
    }

    // -----------------------------------------------------------------------
    // addAsset
    // -----------------------------------------------------------------------

    public addAsset(
        asset: Asset,
        options: {
            mediaInfo: MediaInfo;
            videoDecoderConfig: VideoDecoderConfig;
            folderUuid?: string;
            initialTimestamp?: number;
            onFrame?: (data: { canvas: DecodedCanvas; timestamp: number; duration: number }) => void;
            onError?: (error: Error) => void;
            onReady?: () => void;
            logicalId?: string;
        },
    ): void {
        const decoderId = options.logicalId ?? asset.id;

        const { targetWidth, targetHeight } = normalizeDimensions(
            options.videoDecoderConfig,
            options.mediaInfo,
        );

        const normalizedConfig: VideoDecoderConfig = {
            ...options.videoDecoderConfig,
            codedWidth: targetWidth,
            codedHeight: targetHeight,
        };
        const normalizedAny = normalizedConfig as any;
        if (normalizedAny.alpha == null) {
            normalizedAny.alpha = "keep";
        }

        const assetClient = this.createAssetState({
            asset,
            mediaInfo: options.mediaInfo,
            videoDecoderConfig: normalizedConfig,
            onFrame: options.onFrame,
            onError: options.onError,
            onReady: options.onReady,
        });

        this.assets.set(decoderId, assetClient);

        if (this.useNative) {
            this.addAssetNative(decoderId, asset, options, assetClient, targetWidth, targetHeight);
        } else {
            this.addAssetWorker(decoderId, asset, options, assetClient, normalizedConfig, targetWidth, targetHeight);
        }
    }

    private async addAssetNative(
        decoderId: string,
        asset: Asset,
        options: {
            mediaInfo: MediaInfo;
            initialTimestamp?: number;
            onReady?: () => void;
        },
        assetClient: VideoDecoderManagerAssetState,
        targetWidth: number,
        targetHeight: number,
    ): Promise<void> {
        try {
            if (targetWidth <= 0 || targetHeight <= 0) {
                throw new Error("Native decoder requires non-zero target dimensions");
            }
            
            const handle = createNativeDecoder();

            const uploader = new NativeFrameUploader();

            assetClient.native = { nativeHandle: handle, uploader };

            const filePath = await resolveNativeDecoderPath(asset.path);

            configureNativeDecoder(
                handle,
                filePath,
                targetWidth,
                targetHeight,
                (frameBytes, width, height, timestamp, duration, requestId, bufferIndex, pixelFormat) => {
                    this.handleNativeFrame(
                        assetClient,
                        frameBytes,
                        width,
                        height,
                        timestamp,
                        duration,
                        requestId,
                        bufferIndex,
                        pixelFormat ?? "rgba",
                    );
                },
                (message) => {
                    if (assetClient.activeIteration) {
                        assetClient.activeIteration.reject(new Error(message));
                        assetClient.activeIteration = null;
                    } else if (assetClient.onError) {
                        assetClient.onError(new Error(message));
                    }
                },
                () => {
                    if (!assetClient.initialized) {
                        assetClient.initialized = true;
                        if (assetClient.initializedResolve) {
                            assetClient.initializedResolve();
                            assetClient.initializedResolve = null;
                        }
                        if (assetClient.onReady) {
                            assetClient.onReady();
                            assetClient.onReady = undefined;
                        }
                    }
                },
                {
                    outputFormat: "nv12",
                    copyFrameData: false,
                    manualAck: true,
                    preferSharedBufferPool:
                        typeof window !== "undefined" && window.crossOriginIsolated,
                },
            );
        } catch (err) {
            console.warn("[VideoDecoderManager] Native decoder failed for", decoderId, ", falling back to worker:", err);
            assetClient.native = undefined;
            // Fall back to worker path
            this.addAssetWorker(decoderId, asset, options as any, assetClient, assetClient.videoDecoderConfig, targetWidth, targetHeight);
        }
    }

    private handleNativeFrame(
        client: VideoDecoderManagerAssetState,
        frameBytes: Uint8Array,
        width: number,
        height: number,
        timestamp: number,
        duration: number,
        requestId: number,
        bufferIndex?: number,
        pixelFormat: NativePixelFormat = "rgba",
    ): void {
        const native = client.native!;
        const ack = () => {
            if (typeof bufferIndex === "number" && bufferIndex >= 0) {
                ackNativeFrame(native.nativeHandle, bufferIndex);
            }
        };
        const upload = (): HTMLCanvasElement => {
            if (pixelFormat === "nv12") {
                return native.uploader.uploadNV12(frameBytes, width, height);
            }
            return native.uploader.upload(frameBytes, width, height);
        };

        // Iteration frames
        if (client.activeIteration && requestId === client.activeIteration.requestId) {
            const iteration = client.activeIteration;

            iteration.frameProcessingPromise = iteration.frameProcessingPromise
                .then(async () => {
                    try {
                        if (iteration.checkCancel && !iteration.checkCancel()) {
                            return;
                        }
                        if (iteration.shouldDelay) {
                            await iteration.shouldDelay(timestamp);
                        }
                        if (iteration.checkCancel && !iteration.checkCancel()) {
                            return;
                        }

                        const canvas = upload();
                        if (client.onFrame) {
                            client.onFrame({ canvas, timestamp, duration });
                        }
                    } finally {
                        ack();
                    }
                })
                .catch((err) => {
                    console.error("Native frame processing error (manager)", err);
                });
            return;
        }

        // Seek/preview frames
        if (requestId !== undefined && requestId !== client.currentRequestId) {
            ack();
            return;
        }

        try {
            const canvas = upload();
            if (client.onFrame) {
                client.onFrame({ canvas, timestamp, duration });
            }
        } finally {
            ack();
        }
    }

    private addAssetWorker(
        decoderId: string,
        asset: Asset,
        options: {
            mediaInfo: MediaInfo;
            folderUuid?: string;
            initialTimestamp?: number;
        },
        assetClient: VideoDecoderManagerAssetState,
        normalizedConfig: VideoDecoderConfig,
        _targetWidth: number,
        _targetHeight: number,
    ): void {
        const worker = this.ensureWorker();
        const reqId = ++assetClient.currentRequestId;

        const formatStr = computeFormatStr(options.mediaInfo.format);

        const config: any = {
            videoDecoderConfig: normalizedConfig,
            asset: {
                id: asset.id,
                type: asset.type,
                path: asset.path,
            },
            folderUuid: options.folderUuid,
            formatStr,
            initialTimestamp: options.initialTimestamp ?? 0,
        };

        if (cachedUserDataPath) {
            config.userDataPath = cachedUserDataPath;
        }

        worker.postMessage({
            type: "configure",
            assetId: decoderId,
            config,
            requestId: reqId,
        });
    }

    // -----------------------------------------------------------------------
    // disposeAsset / hasAsset / updateAssetHandlers
    // -----------------------------------------------------------------------

    public disposeAsset(assetId: string): void {
        const client = this.assets.get(assetId);
        if (!client) return;

        if (client.activeIteration) {
            client.activeIteration.reject(new Error("Asset disposed"));
            client.activeIteration = null;
        }
        for (const { reject } of client.pendingSeeks.values()) {
            reject(new Error("Asset disposed"));
        }
        client.pendingSeeks.clear();

        if (client.native) {
            cancelNativeDecoder(client.native.nativeHandle);
            disposeNativeDecoder(client.native.nativeHandle);
            client.native.uploader.dispose();
            client.native = undefined;
        } else if (this._worker) {
            this._worker.postMessage({ type: "dispose", assetId });
        }

        this.assets.delete(assetId);
    }

    public hasAsset(assetId: string): boolean {
        return this.assets.has(assetId);
    }

    public updateAssetHandlers(
        assetId: string,
        handlers: {
            onFrame?: (data: { canvas: DecodedCanvas; timestamp: number; duration: number }) => void;
            onError?: (error: Error) => void;
            onReady?: () => void;
        },
    ): void {
        const client = this.assets.get(assetId);
        if (!client) return;

        if (handlers.onFrame) client.onFrame = handlers.onFrame;
        if (handlers.onError) client.onError = handlers.onError;

        if (handlers.onReady) {
            if (client.initialized) {
                handlers.onReady();
            } else {
                client.onReady = handlers.onReady;
            }
        }
    }

    // -----------------------------------------------------------------------
    // seek / iterate
    // -----------------------------------------------------------------------

    public async seek(assetId: string, timestamp: number, forceAccurate: boolean = false): Promise<void> {
        const client = this.assets.get(assetId);
        if (!client) return;

        const reqId = ++client.currentRequestId;

        if (client.native) {
            // Native path — the C++ addon's seek() returns a Promise directly.
            // The onFrame callback handles frame delivery.
            return seekNativeDecoder(client.native.nativeHandle, timestamp, forceAccurate, reqId);
        }

        // Worker path
        return new Promise<void>((resolve, reject) => {
            client.pendingSeeks.set(reqId, { resolve, reject });

            this.ensureWorker().postMessage({
                type: "seek",
                assetId,
                timestamp,
                forceAccurate,
                requestId: reqId,
            });
        });
    }

    public async iterate(
        assetId: string,
        startTime: number,
        endTime: number,
        shouldDelay?: (timestamp: number) => Promise<void>,
        checkCancel?: () => boolean,
    ): Promise<void> {
        const client = this.assets.get(assetId);
        if (!client) return;

        const reqId = ++client.currentRequestId;

        if (client.native) {
            return new Promise<void>((resolve, reject) => {
                client.activeIteration = {
                    requestId: reqId,
                    shouldDelay,
                    checkCancel,
                    resolve,
                    reject,
                    frameProcessingPromise: Promise.resolve(),
                };

                iterateNativeDecoder(client.native!.nativeHandle, startTime, endTime, reqId)
                    .then(() => {
                        if (client.activeIteration && client.activeIteration.requestId === reqId) {
                            const iteration = client.activeIteration;
                            client.activeIteration = null;
                            iteration.frameProcessingPromise.then(() => iteration.resolve());
                        }
                    })
                    .catch((err) => {
                        if (client.activeIteration && client.activeIteration.requestId === reqId) {
                            client.activeIteration.reject(err);
                            client.activeIteration = null;
                        }
                    });
            });
        }

        // Worker path
        return new Promise<void>((resolve, reject) => {
            client.activeIteration = {
                requestId: reqId,
                shouldDelay,
                checkCancel,
                resolve,
                reject,
                frameProcessingPromise: Promise.resolve(),
            };

            this.ensureWorker().postMessage({
                type: "iterate",
                assetId,
                startTime,
                endTime,
                requestId: reqId,
            });
        });
    }

    // -----------------------------------------------------------------------
    // disposeAll
    // -----------------------------------------------------------------------

    public disposeAll(): void {
        for (const id of this.assets.keys()) {
            this.disposeAsset(id);
        }

        if (this.useNative) {
            disposeAllNativeDecoders();
        }

        if (this._worker) {
            this._worker.terminate();
            this._worker = null;
        }

        this.assets.clear();
    }
}
