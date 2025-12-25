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
import { getUserDataPath as getUserDataPathPreload } from "@app/preload";

interface VideoFrameDecoderOptions {
    mediaInfo: MediaInfo;
    videoDecoderConfig: VideoDecoderConfig;
    onFrame: ({
        canvas,
        timestamp,
        duration,
    }: {
        canvas: VideoFrame;
        timestamp: number;
        duration: number;
    }) => void;
    onError: (error: Error) => void;
    initialTimestamp?: number;
    onReady?: () => void;
}

export class VideoFrameDecoder {
    private worker: Worker;
    private currentRequestId = 0;
    private initialized = false;
    private initializedPromise: Promise<void>;
    private initializedResolve: (() => void) | null = null;
    private onReadyCallback?: () => void;
    private onFrameCallback: (data: { canvas: VideoFrame; timestamp: number; duration: number }) => void;
    private onErrorCallback: (error: Error) => void;

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

        // Setup "ready" tracking that resolves once the worker has finished
        // initial configuration. Callers can await this or use the onReady
        // callback; both are driven directly by worker messages.
        this.initializedPromise = new Promise<void>((resolve) => {
            this.initializedResolve = () => {
                if (this.initialized) return;
                this.initialized = true;
                resolve();
            };
        });

        this.worker = new Worker(new URL("./video-decoder.worker.ts", import.meta.url), {
            type: "module",
        });

        this.worker.onmessage = (e) => {
            const msg = e.data;


            if (msg.type === "frame") {
                // Handle Iteration Frames
                if (this.activeIteration && msg.requestId === this.activeIteration.requestId) {
                    const iteration = this.activeIteration;
        
                    // Chain processing to ensure correct timing and flow control
                    iteration.frameProcessingPromise = iteration.frameProcessingPromise.then(async () => {
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
                            canvas: msg.frame, // Ownership transferred here usually, but here we just pass it
                            timestamp: msg.timestamp,
                            duration: msg.duration,
                        });
                        
                        // We do NOT close the frame here if onFrameCallback consumes/closes it.
                        // But looking at VideoPreview.tsx, it renders it to canvas.
                        // `VideoFrame` needs to be closed. VideoPreview calls drawWrappedCanvas -> context.drawImage.
                        // It doesn't close it explicitly? 
                        // Wait, original VideoDecoder closed it: `frame.close()` at end of handler.
                        // `VideoPreview` receives `wc` with `canvas: VideoFrame`.
                        // `drawWrappedCanvas` does NOT close it.
                        // So we MUST close it here.
                        msg.frame.close();

                        // Send Ack to resume worker
                        this.worker.postMessage({ type: "ack", requestId: msg.requestId });
                    }).catch(err => {
                        console.error("Frame processing error", err);
                        msg.frame.close();
                    });
                    
                    return;
                }

                // Handle Seek/Preview Frames
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
                // "ready" from the worker means the decoder is configured
                // and ready for seeks. Notify both the promise and optional callback.
                if (this.initializedResolve) {
                    this.initializedResolve();
                    this.initializedResolve = null;
                }
                if (this.onReadyCallback) {
                    this.onReadyCallback();
                    this.onReadyCallback = undefined;
                }
            } else if (msg.type === "seekDone") {
                // Fallback: in case we ever have an older worker without "ready",
                // treat the first seekDone as making the decoder ready for callers
                // that are awaiting waitUntilReady().
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
                     // Wait for queue to drain
                     iteration.frameProcessingPromise.then(() => {
                         iteration.resolve();
                     });
                     this.activeIteration = null;
                 }
            }
        };
        
        // Calculate target dimensions (max 720p) logic preserved from original
        const { codedWidth, codedHeight } = options.videoDecoderConfig;
        let targetWidth = codedWidth || 0;
        let targetHeight = codedHeight || 0;

        if (!targetWidth || !targetHeight) {
             targetWidth = options.mediaInfo.video?.codedWidth || 0;
             targetHeight = options.mediaInfo.video?.codedHeight || 0;
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

        const config = {
            ...options.videoDecoderConfig,
            codedWidth: targetWidth,
            codedHeight: targetHeight,
        };

        // Extract Source
        let sourceConfig: { type: "url" | "blob"; url?: string; blob?: Blob } | null = null;
        
        // Try to get source from originalInput if available (best)
        if (options.mediaInfo.originalInput) {
             const src = (options.mediaInfo.originalInput).source;
             if (src instanceof UrlSource) {
                 sourceConfig = { type: "url", url: (src as any)._url.toString() };
             } else if (src instanceof BlobSource) {
                 sourceConfig = { type: "blob", blob: (src as any)._blob };
             }
        }
        
        // Fallback: Infer from path if originalInput is missing
        if (!sourceConfig) {
             const path = options.mediaInfo.path;
             if (path.startsWith("blob:")) {
                 // We can't fetch a blob: URL easily in a worker if it wasn't passed as a Blob object,
                 // but if it's a blob: URL created by URL.createObjectURL, it MIGHT work if same origin.
                 sourceConfig = { type: "url", url: path };
        } else {
                 // Assume it's a URL (file:// or http:// or app://)
                 // Note: app:// might need special handling if worker doesn't support it,
                 // but typically custom schemes work if the environment supports them.
                 sourceConfig = { type: "url", url: path };
                }
            }

        // Determine format string to speed up worker init
        let formatStr: string | undefined;
        const fmt = options.mediaInfo.format;
        if (fmt) {
            if (fmt === MP4) formatStr = "mp4";
            else if (fmt === WEBM) formatStr = "webm";
            else if (fmt === QTFF) formatStr = "mov";
            else if (fmt === MATROSKA) formatStr = "mkv";
            else if (fmt === OGG) formatStr = "ogg";
            else if (fmt === MP3) formatStr = "mp3";
            else if (fmt === WAVE) formatStr = "wav";
            else if (fmt === FLAC) formatStr = "flac";
            else if (fmt === ADTS) formatStr = "aac";
        }

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

    /**
     * Returns true once the worker has fully configured the decoder
     * and completed the initial seek.
     */
    public isReady(): boolean {
        return this.initialized;
    }

    /**
     * Await this to ensure the worker-side decoder is fully initialized
     * (configure + initial seekDone) before issuing dependent operations.
     */
    public async waitUntilReady(): Promise<void> {
        if (this.initialized) return;
        return this.initializedPromise;
    }

    public async seek(timestamp: number, forceAccurate: boolean = false): Promise<void> {
        const reqId = ++this.currentRequestId;
        this.worker.postMessage({
            type: "seek",
            timestamp,
            forceAccurate,
            requestId: reqId
        });
    }

    public async iterate(
        startTime: number, 
        endTime: number, 
        shouldDelay?: (timestamp: number) => Promise<void>,
        checkCancel?: () => boolean
    ): Promise<void> {
        const reqId = ++this.currentRequestId;
        
        return new Promise<void>((resolve, reject) => {
             this.activeIteration = {
                 requestId: reqId,
                 shouldDelay,
                 checkCancel,
                 resolve,
                 reject,
                 frameProcessingPromise: Promise.resolve()
             };

             this.worker.postMessage({
                type: "iterate",
                startTime,
                endTime,
                requestId: reqId
            });
        });
    }
    
    public dispose() {
        this.currentRequestId++;
        this.worker.postMessage({ type: "dispose" });
        this.worker.terminate();
    }
}

interface VideoDecoderManagerAssetState {
    asset: Asset;
    mediaInfo: MediaInfo;
    videoDecoderConfig: VideoDecoderConfig;
    onFrame?: (data: { canvas: VideoFrame; timestamp: number; duration: number }) => void;
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
    private worker: Worker;
    private assets = new Map<string, VideoDecoderManagerAssetState>();

    private createAssetState(params: {
        asset: Asset;
        mediaInfo: MediaInfo;
        videoDecoderConfig: VideoDecoderConfig;
        onFrame?: (data: { canvas: VideoFrame; timestamp: number; duration: number }) => void;
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
            initializedPromise: Promise.resolve() as Promise<void>, // overwritten below
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
        let workerUrl = new URL("./video-decoder.worker.ts", import.meta.url);

        this.worker = new Worker(workerUrl, {
            type: "module",
        });

        // Make worker lifecycle issues extremely obvious in the renderer console.
        this.worker.onerror = (err) => {
            // eslint-disable-next-line no-console
            console.error("[VideoDecoderManager] worker error", err.message || err);
        };

        this.worker.onmessageerror = (err) => {
            // eslint-disable-next-line no-console
            console.error("[VideoDecoderManager] worker message error", err);
        };

        this.worker.onmessage = (e: MessageEvent) => {
            const msg = e.data;

            
            const assetId: string | undefined = msg.assetId;
            const client = assetId ? this.assets.get(assetId) : undefined;

            if (!client) {
                // Best effort cleanup of stray frames
                if (msg.type === "frame" && msg.frame) {
                    try {
                        msg.frame.close();
                    } catch {
                        // ignore
                    }
                }
                return;
            }

            if (msg.type === "frame") {
                // Iteration frames for this asset
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

                            // Ack this frame to allow the worker to resume iteration
                            this.worker.postMessage({
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

                // Seek/preview frames for this asset
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
                } else {
                    // If no handler is registered yet, just close the frame.
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
                // Also treat first seekDone as "ready" fallback
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
                // Route iteration-specific errors first
                if (client.activeIteration && msg.requestId === client.activeIteration.requestId) {
                    client.activeIteration.reject(new Error(msg.error));
                    client.activeIteration = null;
                } else {
                    const pending = msg.requestId ? client.pendingSeeks.get(msg.requestId) : undefined;
                    if (pending) {
                        client.pendingSeeks.delete(msg.requestId);
                        pending.reject(new Error(msg.error));
                    } else {
                        if (client.onError) {
                            client.onError(new Error(msg.error));
                        } else {
                            console.error("[VideoDecoderManager] Unhandled error", msg.error);
                        }
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
        };
    }

    /**
     * Register a new asset with the shared worker-backed decoder.
     */
    public addAsset(
        asset: Asset,
        options: {
            mediaInfo: MediaInfo;
            videoDecoderConfig: VideoDecoderConfig;
            folderUuid?: string;
            initialTimestamp?: number;
            onFrame?: (data: { canvas: VideoFrame; timestamp: number; duration: number }) => void;
            onError?: (error: Error) => void;
            onReady?: () => void;
            /**
             * Optional logical identifier that scopes the decoder instance.
             * This allows multiple independent decoders to share the same
             * underlying Asset (e.g. several clips using the same file) without
             * overwriting each other's handlers or request state.
             *
             * When provided, this id is used as the manager/worker key, while
             * the real Asset.id is still passed through in the asset metadata.
             */
            logicalId?: string;
        },
    ): void {
        const decoderId = options.logicalId ?? asset.id;

        // Normalize target dimensions (same logic as VideoFrameDecoder)
        const { codedWidth, codedHeight } = options.videoDecoderConfig;
        let targetWidth = codedWidth || 0;
        let targetHeight = codedHeight || 0;

        if (!targetWidth || !targetHeight) {
            targetWidth = options.mediaInfo.video?.codedWidth || 0;
            targetHeight = options.mediaInfo.video?.codedHeight || 0;
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

        const normalizedConfig: VideoDecoderConfig = {
            ...options.videoDecoderConfig,
            codedWidth: targetWidth,
            codedHeight: targetHeight,
        };
        // Prefer keeping alpha for formats that support it (e.g. WebM with alpha).
        // Some TS lib.dom versions don't expose `alpha` on VideoDecoderConfig yet,
        // but the runtime may support it. The worker will fall back if unsupported.
        const normalizedAny = normalizedConfig as any;
        if (normalizedAny.alpha == null) {
            normalizedAny.alpha = "keep";
        }

        // Determine format string to speed up worker init
        let formatStr: string | undefined;
        const fmt = options.mediaInfo.format;
        if (fmt) {
            if (fmt === MP4) formatStr = "mp4";
            else if (fmt === WEBM) formatStr = "webm";
            else if (fmt === QTFF) formatStr = "mov";
            else if (fmt === MATROSKA) formatStr = "mkv";
            else if (fmt === OGG) formatStr = "ogg";
            else if (fmt === MP3) formatStr = "mp3";
            else if (fmt === WAVE) formatStr = "wav";
            else if (fmt === FLAC) formatStr = "flac";
            else if (fmt === ADTS) formatStr = "aac";
        }

        // Track per-asset (or per-logical-id) state in the manager
        const assetClient = this.createAssetState({
            asset,
            mediaInfo: options.mediaInfo,
            videoDecoderConfig: normalizedConfig,
            onFrame: options.onFrame,
            onError: options.onError,
            onReady: options.onReady,
        });

        this.assets.set(decoderId, assetClient);

        // Send configure to the shared worker
        const reqId = ++assetClient.currentRequestId;

        const config: any = {
            videoDecoderConfig: normalizedConfig,
            asset: {
                // Preserve the real Asset identity in metadata so worker-side
                // path/format logic continues to function as before.
                id: asset.id,
                type: asset.type,
                path: asset.path,
            },
            folderUuid: options.folderUuid,
            formatStr,
            initialTimestamp: options.initialTimestamp ?? 0,
        };

        // If we've already resolved the Electron userData path via preload,
        // pass it through so the worker can cheaply detect when an incoming
        // asset path is rooted under userData and prefer app://user-data.
        if (cachedUserDataPath) {
            config.userDataPath = cachedUserDataPath;
        }

        this.worker.postMessage({
            type: "configure",
            assetId: decoderId,
            config,
            requestId: reqId,
        });
    }

    /**
     * Dispose a single asset and release any resources associated with it.
     */
    public disposeAsset(assetId: string): void {
        const client = this.assets.get(assetId);
        if (!client) return;

        // Cancel any active iteration promises
        if (client.activeIteration) {
            client.activeIteration.reject(new Error("Asset disposed"));
            client.activeIteration = null;
        }
        // Reject any pending seeks
        for (const { reject } of client.pendingSeeks.values()) {
            reject(new Error("Asset disposed"));
        }
        client.pendingSeeks.clear();

        this.assets.delete(assetId);
        this.worker.postMessage({ type: "dispose", assetId });
    }

    /**
     * Returns true if an asset with the given id has already been registered.
     */
    public hasAsset(assetId: string): boolean {
        return this.assets.has(assetId);
    }

    /**
     * Update just the handlers (onFrame, onError, onReady) for an existing asset
     * without reconfiguring the underlying decoder/worker state.
     */
    public updateAssetHandlers(
        assetId: string,
        handlers: {
            onFrame?: (data: { canvas: VideoFrame; timestamp: number; duration: number }) => void;
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
                // If the asset is already initialized, invoke immediately.
                handlers.onReady();
            } else {
                client.onReady = handlers.onReady;
            }
        }
    
    }

    /**
     * Perform a seek for a specific asset.
     */
    public async seek(assetId: string, timestamp: number, forceAccurate: boolean = false): Promise<void> {
        const client = this.assets.get(assetId);
        if (!client) {
            return;
        }

        const reqId = ++client.currentRequestId;

        return new Promise<void>((resolve, reject) => {
            client.pendingSeeks.set(reqId, { resolve, reject });

            this.worker.postMessage({
                type: "seek",
                assetId,
                timestamp,
                forceAccurate,
                requestId: reqId,
            });
        });
    }

    /**
     * Iterate frames for a specific asset over a time range.
     */
    public async iterate(
        assetId: string,
        startTime: number,
        endTime: number,
        shouldDelay?: (timestamp: number) => Promise<void>,
        checkCancel?: () => boolean,
    ): Promise<void> {

        const client = this.assets.get(assetId);
        if (!client) {
            return;
        }

        const reqId = ++client.currentRequestId;

        return new Promise<void>((resolve, reject) => {
            client.activeIteration = {
                requestId: reqId,
                shouldDelay,
                checkCancel,
                resolve,
                reject,
                frameProcessingPromise: Promise.resolve(),
            };

            this.worker.postMessage({
                type: "iterate",
                assetId,
                startTime,
                endTime,
                requestId: reqId,
            });
        });
    }

    /**
     * Dispose all assets and terminate the underlying worker.
     */
    public disposeAll(): void {
        for (const id of this.assets.keys()) {
            this.disposeAsset(id);
        }
        this.worker.terminate();
        this.assets.clear();
    }
}

