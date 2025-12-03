import { MediaInfo } from "../types";
import { UrlSource, BlobSource, MP4, WEBM, QTFF, OGG, MATROSKA, MP3, WAVE, FLAC, ADTS } from "mediabunny";

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
}

export class VideoFrameDecoder {
    private worker: Worker;
    private currentRequestId = 0;
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
