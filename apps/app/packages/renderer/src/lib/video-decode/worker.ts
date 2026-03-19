import { Input, ALL_FORMATS, UrlSource, EncodedPacketSink, Source, PacketStats, EncodedPacket } from 'mediabunny';
import { WebGLRenderer } from './renderer_webgl';
import { Canvas2DRenderer } from './renderer_2d';
import { fileURLToPathInWorker, createNodeFileSource, readFileBuffer } from './utils';
import { FilterClipProps, VideoClipProps } from '../types';
import { WebGLHaldClut } from '@/components/preview/webgl-filters/hald-clut';
import { MergeAlphaState, createMergeAlphaState, mergeAlphaIntoColor } from './merge-alpha';

const { existsSync } = require('fs');

const VERSION = 1;  
const DATABASE_NAME = "media-packets";

type InitPayload = {
    type: "init"
    data: {
        canvasId: string;
        id: string;
        canvas?: OffscreenCanvas;
        path: string;
        renderer: "webgl2" | "webgpu" | "2d";
        width: number;
        height: number;
    }
}

type SeekPayload = {
    type: "seek"
    data: {
        id: string;
        speed: number;
        targetFps: number;
        timestamp: number;
    }
}

type IteratePayload = {
    type: "iterate"
    data: {
        id: string;
        startTimestamp: number;
        endTimestamp?: number;
        speed: number;
        targetFps: number;
        playbackState?: {
            startWallTime: number;
            startFocusFrame: number;
            isPlaying: boolean;
            mainNow: number;
        };
    }
}

type PausePayload = {
    type: "pause"
    data: {
        id: string;
    }
}

type DestroyPayload = {
    type: "destroy"
    data: {
        id: string;
        canvasId: string;
    }
}

type UpdateRendererPayload = {
    type: "update"
    data: {
        id: string;
        maskFrame: number;
        clip: VideoClipProps;
        focusFrame: number;
        filters: FilterClipProps[];
        useMask: boolean;
    }
}

type PreloadPayload = {
    type: "preload"
    data: {
        id: string;
        startTimestamp: number;
        endTimestamp?: number;
        secondsToPrefetch: number;
        targetFps: number;
        speed: number;
        playbackState?: {
            startWallTime: number;
            startFocusFrame: number;
            isPlaying: boolean;
            mainNow: number;
        };
    }
}

interface VideoState {
    id: string;
    input: Input;
    sink: EncodedPacketSink;
    duration: number;
    packets: EncodedPacket[];
    keyframePackets: EncodedPacket[];  // Sorted by timestamp, for O(log k) seek
    videoConfig: VideoDecoderConfig;
    packetStats: PacketStats;
    hasAlpha: boolean;
}

type RenderState = {
    id: string;
    type: "seek";
    timestamp: number;
    stopDecode: boolean;
    stopRender: boolean;
} | {
    id: string;
    type: "iterate";
    startTimestamp: number;
    endTimestamp: number;
    timingFunc: (() => Promise<void>) | undefined;
    stopDecode: boolean;
    stopRender: boolean;
    preload?: boolean;
}


const videoStates = new Map<string, VideoState>();
const canvasStates = new Map<string, OffscreenCanvas>();
const encodedPacketSinkStates = new Map<string, EncodedPacketSink>();
const renderStates = new Map<string, RenderState>();
const decoderStates = new Map<string, VideoDecoder>();
const alphaDecoderStates = new Map<string, VideoDecoder>();
const lastPacketStates = new Map<string, EncodedPacket>();
const renderers = new Map<string, WebGLRenderer | Canvas2DRenderer>();
const decoderInUseStates = new Map<string, boolean>();
const alphaAssetStates = new Map<string, MergeAlphaState>();
const alphaFramesByTimestamp = new Map<string, Map<number, VideoFrame>>();
const pendingColorFramesByTimestamp = new Map<string, Map<number, VideoFrame>>();

const seekQueueStates:Map<string, SeekPayload | null> = new Map();
const seekInProgressStates:Map<string, boolean> = new Map();

const haldClutInstance = new WebGLHaldClut(readFileBuffer);

type Payload = InitPayload | SeekPayload | IteratePayload | PausePayload | DestroyPayload | UpdateRendererPayload | PreloadPayload;

const resolveSource = (path: string): Source => {
    const filePath = fileURLToPathInWorker(path);
    if (!existsSync(filePath)) {
        const url = new URL(`app://user-data/${filePath}`);
        return new UrlSource(url);
    } else {
        return createNodeFileSource(filePath);
    }
}

const getOrCreateMergeAlphaState = (id: string): MergeAlphaState => {
    let state = alphaAssetStates.get(id);
    if (!state) {
        state = createMergeAlphaState();
        alphaAssetStates.set(id, state);
    }
    return state;
}

const drawToRenderer = (id: string, frame: VideoFrame, renderState: RenderState) => {
    const renderer = renderers.get(id);
    if (!renderer) { frame.close(); return; }
    try {
      renderer.draw(frame, renderState);
    } catch (e) {
      frame.close();
      throw e;
    }
  };

const handleMessage = (event: MessageEvent<Payload>): void => {
    const payload = event.data;
    switch (payload.type) {
        case "init":
            init(payload);
            break;
        case "seek":
            processSeekQueue(payload);
            break;
        case "iterate":
            iterate(payload);
            break;
        case "pause":
            pause(payload);
            break;
        case "destroy":
            destroy(payload);
            break;
        case "update":
            updateRenderer(payload);
            break;
    }
}

const init = async (payload: InitPayload): Promise<void> => {
    let { canvasId, canvas, path, id, renderer: rendererName = "2d", width, height } = payload.data;
    const source = resolveSource(path);

    if (!canvas) {
        // ensure canvas is present in canvasStates
        canvas = canvasStates.get(canvasId);
        if (!canvas) {
            console.error("Canvas not found");
            return;
        }
    } else {
        canvasStates.set(canvasId, canvas);
    }

    const input = new Input({
        formats: ALL_FORMATS,
        source: source
    });

    const videoTrack = await input.getPrimaryVideoTrack();
    if (!videoTrack) {
        console.error("No video track found");
        return;
    }

    const duration = await videoTrack.computeDuration();
    const packetStats = await videoTrack.computePacketStats();
    const videoConfig = await videoTrack.getDecoderConfig();

    let renderer: WebGLRenderer | Canvas2DRenderer;

    if (rendererName === "webgl2") {
        renderer = new WebGLRenderer(id, self.postMessage.bind(self), "webgl2", canvas, width, height);
    } else if (rendererName === "2d") {  
        renderer = new Canvas2DRenderer(id, self.postMessage.bind(self), canvas, width, height, haldClutInstance);
    } else {
        console.error("Invalid renderer");
        return;
    }

    renderers.set(id, renderer);

    let sink: EncodedPacketSink;

    if (!videoConfig) {
        console.error("No video config found");
        return;
    }
    
    if (encodedPacketSinkStates.has(path)) {
        sink = encodedPacketSinkStates.get(path)!;
    } else {
        sink = new EncodedPacketSink(videoTrack);
        encodedPacketSinkStates.set(path, sink);
    }

    const request = indexedDB.open(DATABASE_NAME, VERSION);

    request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains("media-timestamps")) {
            db.createObjectStore("media-timestamps");
        }
    };

    request.onsuccess = async (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Try to reuse cached timestamps first
        const existing = await new Promise<EncodedPacket[] | undefined>((resolve, reject) => {
            const tx = db.transaction("media-timestamps", "readonly");
            const store = tx.objectStore("media-timestamps");
            const req = store.get(id);
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });

        let packets: EncodedPacket[];
        let hasAlpha = false;

        if (existing && existing.length > 0) {
            // Reuse cached timestamps - skip expensive sink iteration
            packets = existing;
        } else {
            // No cache: collect from sink and store for future reuse
            const newPackets: EncodedPacket[] = [];
            for await (const packet of sink.packets(undefined, undefined, {  verifyKeyPackets: true })) {
                let emptyPacket = new EncodedPacket(new Uint8Array([]), 
                packet.type, 
                packet.timestamp, 
                packet.duration, 
                packet.sequenceNumber, 
                packet.byteLength, 
                packet.sideData);
                newPackets.push(emptyPacket);
                if (packet.sideData?.alpha) {
                    hasAlpha = true;
                }
            }

            packets = newPackets;

            // ensure sorted by timestamp ascending
            packets.sort((a, b) => a.timestamp - b.timestamp);

            await new Promise<void>((resolve, reject) => {
                const tx = db.transaction("media-timestamps", "readwrite");
                const store = tx.objectStore("media-timestamps");
                store.put(newPackets, id);
                tx.oncomplete = () => resolve();
                tx.onerror = () => reject(tx.error);
            });
        }

        const keyframePackets = packets.filter(p => p.type === "key");
    
        videoStates.set(id, {
            id,
            input,
            sink,
            duration,
            packets: packets,
            keyframePackets,
            videoConfig: videoConfig,
            packetStats: packetStats,
            hasAlpha: hasAlpha
        });

        if (hasAlpha) {
            pendingColorFramesByTimestamp.set(id, new Map());
            alphaFramesByTimestamp.set(id, new Map());
        }

        const onColorFrame = async (frame: VideoFrame) => {
           
            let renderState = renderStates.get(id);
            let videoState = videoStates.get(id);
                
            if (!renderState || !videoState || !renderer) {
                frame.close();
                console.error("Render state not found or video state not found");
                return;
            }

            if (!videoState.hasAlpha) {
                return drawToRenderer(id, frame, renderState);
            }

            const ts = frame.timestamp;
            const alphaFrames = alphaFramesByTimestamp.get(id);
            const alpha = alphaFrames?.get(ts);
            
            if (alpha) {
                alphaFrames?.delete(ts);
                const alphaAssetState = getOrCreateMergeAlphaState(id);
                let merged: VideoFrame | undefined;
                try {
                    merged = mergeAlphaIntoColor(alphaAssetState, frame, alpha);
                } catch (e) {
                    // we fallback to the original frame if the merge fails
                    merged = frame;
                } finally {
                    if (merged !== frame) frame.close();
                    alpha.close();
                }
                return drawToRenderer(id, merged, renderState);
            }
            pendingColorFramesByTimestamp.get(id)?.set(ts, frame);
  
        }

        const onAlphaFrame = async (alphaFrame: VideoFrame) => {
            let videoState = videoStates.get(id);
            const renderState = renderStates.get(id);

            if (!videoState || !renderState) {
                alphaFrame.close();
                console.error("Video state not found");
                return;
            }

           const ts = alphaFrame.timestamp;
           const pendingColorFrames = pendingColorFramesByTimestamp.get(id);
           const pendingFrame = pendingColorFrames?.get(ts);
           if (pendingFrame) {
             const mergeState = getOrCreateMergeAlphaState(id);
             let merged: VideoFrame | undefined;
             try {
                merged = mergeAlphaIntoColor(mergeState, pendingFrame, alphaFrame);
             } catch (e) {
                merged = pendingFrame;
             } finally {
                if (merged !== pendingFrame) pendingFrame.close();
                alphaFrame.close();
             }
             return drawToRenderer(id, merged, renderState);
          } 

          const alphaFrames = alphaFramesByTimestamp.get(id);
          if (alphaFrames) {
            if (alphaFrames.size >= 60) {
            const firstKey = alphaFrames?.keys().next().value;
              if (firstKey !== undefined) {
                alphaFrames?.get(firstKey)?.close();
                alphaFrames?.delete(firstKey);
              }
            }
            alphaFrames.set(ts, alphaFrame);
          }
        }

        const onError = (error: Error) => {
            console.log(error);
        }

        const [decoder, alphaDecoder] = createDecoderState(videoStates.get(id)!, onColorFrame, onAlphaFrame, onError);
        decoderStates.set(id, decoder);

        if (alphaDecoder) {
            alphaDecoderStates.set(id, alphaDecoder);
        }

         
        self.postMessage({ type: "init_complete", data: { id, duration } });
    };
}

const binarySearch = (packets: EncodedPacket[], timestamp: number): number => {
    let lo = 0, hi = packets.length - 1;
    let best = 0;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (packets[mid].timestamp <= timestamp) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return best;
}

const createDecoderState = (videoState: VideoState, onFrame: (frame: VideoFrame) => void, onAlphaFrame: (frame: VideoFrame) => void, onError: (error: Error) => void): [VideoDecoder, VideoDecoder | undefined] => {
    let alphaDecoder: VideoDecoder | undefined;
    const decoder = new VideoDecoder({
        output: (frame) => {
            onFrame(frame);
        },
        error: (error) => {
            onError(error);
        }
    });

    if (videoState.hasAlpha) {
        alphaDecoder = new VideoDecoder({
            output: (frame) => {
                onAlphaFrame(frame);
            },
            error: (error) => { 
                onError(error);
            }
        });
        const cfgAny = { ...(videoState.videoConfig as any) };
        delete cfgAny.alpha;
        alphaDecoder.configure(cfgAny);
    }

    decoder.configure(videoState.videoConfig);
    return [decoder, alphaDecoder];
}

const processSeekQueue = async (payload: SeekPayload): Promise<void> => {
    const { id } = payload.data;
    const seekInProgress = seekInProgressStates.get(id);

    if (seekInProgress) {
        seekQueueStates.set(id, payload);  // Always store the latest seek request
        return;
    }

    seekInProgressStates.set(id, true);
    const itemToProcess = seekQueueStates.get(id) ?? payload;
    seekQueueStates.delete(id);  // Remove from queue since we're processing it

    try {
        await seek(itemToProcess);
    } finally {
        seekInProgressStates.set(id, false);
        const next = seekQueueStates.get(id);
        if (next) {
            seekQueueStates.delete(id);
            processSeekQueue(next);
        }
    }
}

const seek = async (payload: SeekPayload): Promise<void> => {
    const { id, timestamp, speed, targetFps } = payload.data;

    const videoState = videoStates.get(id);
    const renderer = renderers.get(id);

    const sourceFps = videoState?.packetStats.averagePacketRate ?? 30;
    renderer?.setup({ speed, sourceFps, targetFps, startTimestamp: timestamp });

    renderer?.stop?.();

    // clear previous render states
    renderStates.delete(id);

    let renderState = {
        id,
        type: "seek",
        timestamp,
        stopDecode: false,
        stopRender: false
    } as RenderState;
    
    renderStates.set(id, renderState);

    if (!videoState) {
        console.error("Video state not found");
        return;
    }

    const sink = videoState.sink;
    // Binary search: rightmost keyframe with timestamp <= target (O(log k) vs O(n))
    const keyframes = videoState.keyframePackets;

    if (keyframes.length === 0) {
        console.error("No keyframes found");
        return;
    }

    let keypacket: EncodedPacket | undefined;
    const keypacketIndex = binarySearch(keyframes, timestamp);
    const currentKeyframe = keyframes[keypacketIndex];

    keypacket = await sink.getPacket(currentKeyframe.timestamp) ?? undefined;

    if (!keypacket) {
        console.error("Cannot find keypacket");
        return;
    }

    const metadataOnly = { metadataOnly: true };
    const targetPacket = await sink.getPacket(timestamp, metadataOnly);
    const endPacket = targetPacket ? await sink.getNextPacket(targetPacket, metadataOnly) ?? undefined : undefined;

    const decoder = decoderStates.get(id);
    if (!decoder) {
        console.error("Decoder not found");
        return;
    }

    if (renderState.type === "seek") {
        renderState.timestamp = targetPacket?.timestamp ?? timestamp;
    }

    if (decoderInUseStates.get(id) === true) {
        console.warn("Decoder already in use, skipping seek");
        return;
    }

    decoderInUseStates.set(id, true);

    try {
        for await (const packet of sink.packets(keypacket, endPacket)) {
            decoder.decode(packet.toEncodedVideoChunk());
        
            if (packet.sideData?.alpha) {
                const alphaDecoder = alphaDecoderStates.get(id);
                if (alphaDecoder && alphaDecoder.state !== "closed") {
                    try {
                        alphaDecoder.decode(packet.alphaToEncodedVideoChunk());
                    } catch {
                        // ignore alpha decode errors; color still decodes
                    }
                }
            }
    
            lastPacketStates.set(id, packet);
            if (renderState.stopDecode === true) {
                break;
            }
        }
        
        // flush decoders
        await decoder.flush();
        const alphaDecoder = alphaDecoderStates.get(id);
        if (alphaDecoder && alphaDecoder.state !== "closed") {
            try {
                await alphaDecoder.flush();
            } catch {
                // ignore
            }
        }
    } catch (e) {
        // reset and configure decoders
        decoder.reset();
        decoder.configure(videoState.videoConfig);
        const alphaDecoder = alphaDecoderStates.get(id);
        if (alphaDecoder) {
            alphaDecoder.reset();
            alphaDecoder.configure(videoState.videoConfig);
        }
    }
    finally {
        decoderInUseStates.set(id, false);
    }
}


const iterate = async (payload: IteratePayload): Promise<void> => {
    const { id, startTimestamp, endTimestamp, speed, targetFps, playbackState } = payload.data;
    const videoState = videoStates.get(id);

    const sourceFps = videoState?.packetStats.averagePacketRate ?? 30;
    // clear previous render states
    const previousRenderState = renderStates.get(id);
    renderStates.delete(id);
    const renderer = renderers.get(id);

    if (!(previousRenderState && previousRenderState.type === "iterate" && previousRenderState.preload === true && previousRenderState.startTimestamp === startTimestamp && previousRenderState.endTimestamp === endTimestamp)) {
        renderer?.stop?.();
        renderer?.setup({ speed, sourceFps, targetFps, startTimestamp, playbackState });
    } else {
        renderer?.startPlayback();
    }
  
    let renderState = {
        id,
        type: "iterate",
        startTimestamp,
        endTimestamp,
        stopDecode: false,
        stopRender: false
    } as RenderState;

    renderStates.set(id, renderState);

    if (!videoState) {
        return;
    }

    const sink = videoState.sink;
    const keyframes = videoState.keyframePackets;
    const decoder = decoderStates.get(id);

    if (!decoder) {
        console.error("Decoder not found");
        return;
    }

    const startKeyframeIndex = binarySearch(keyframes, startTimestamp);
    const startKeyFrame = keyframes[startKeyframeIndex];
    let startKeyFramePacket = await sink.getPacket(startKeyFrame.timestamp) ?? undefined;

    const metadataOnly = { metadataOnly: true };
    const targetPacket = endTimestamp ? await sink.getPacket(endTimestamp, metadataOnly) : undefined;
    const endPacket = targetPacket
    ? await sink.getNextPacket(targetPacket, metadataOnly) ?? undefined : undefined;

    if (!startKeyFramePacket) {
        console.error("Cannot find start keyframe packet");
        return;
    }

    let packetCount = 0;
    const fps = Math.floor(Math.round(sourceFps) / 2);
    const iterator = sink.packets(startKeyFramePacket, endPacket);

    while (decoderInUseStates.get(id) === true) {
        await new Promise(r => requestAnimationFrame(r));
    }

    decoderInUseStates.set(id, true);

    try {
        for await (const packet of iterator) {
            decoder.decode(packet.toEncodedVideoChunk());
    
            if (packet.sideData?.alpha) {
                const alphaDecoder = alphaDecoderStates.get(id);
                if (alphaDecoder && alphaDecoder.state !== "closed") {
                    try {
                        alphaDecoder.decode(packet.alphaToEncodedVideoChunk());
                    } catch {
                        // ignore alpha decode errors; color still decodes
                    }
                }
            }
    
            if (renderState.stopDecode === true) {
                decoder.reset();
                decoder.configure(videoState.videoConfig);
                break;
            }
    
            if (packetCount % fps === 0) {
                await new Promise(r => requestAnimationFrame(r));
            }
    
            packetCount++;
        }
    
        // flush decoders
        await decoder.flush();
        const alphaDecoder = alphaDecoderStates.get(id);
        if (alphaDecoder && alphaDecoder.state !== "closed") {
            try {
                await alphaDecoder.flush();
            } catch {
                // ignore
            }
        }
    }  catch (e) {
        // reset and configure decoders
        decoder.reset();
        decoder.configure(videoState.videoConfig);
        const alphaDecoder = alphaDecoderStates.get(id);
        if (alphaDecoder) {
            alphaDecoder.reset();
            alphaDecoder.configure(videoState.videoConfig);
        }
    } finally {
        decoderInUseStates.set(id, false);
    }
}

const pause = async (payload: PausePayload): Promise<void> => {
    const { id } = payload.data;
    const renderState = renderStates.get(id);
    const renderer = renderers.get(id);

    if (!renderState) return;

    renderState.stopDecode = true;
    renderState.stopRender = true;
    renderer?.stop?.();
}

const destroy = async (payload: DestroyPayload): Promise<void> => {
    const { id, canvasId } = payload.data;
    const renderer = renderers.get(id);
    renderer?.stop?.();
    const renderState = renderStates.get(id);
    const decoderState = decoderStates.get(id);

    if (!renderState) {
        console.error("Render state not found");
        return;
    }   

    renderState.stopDecode = true;
    renderState.stopRender = true;

    videoStates.delete(id);
    canvasStates.delete(canvasId);
    encodedPacketSinkStates.delete(id);
    renderStates.delete(id);
    decoderStates.delete(id);
    lastPacketStates.delete(id);
    renderers.delete(id);
    seekQueueStates.delete(id);
    seekInProgressStates.delete(id);

    try {
        await decoderState?.close();
    } catch (e) {
        console.error("Error closing decoder", e);
    }

}

const updateRenderer = async (payload: UpdateRendererPayload): Promise<void> => {
    const { id, maskFrame, clip, focusFrame, filters, useMask = true } = payload.data;
    const renderer = renderers.get(id);
    try {
        const signature = renderer?.getCurrentSignature();
        let newSignature: string | null = null;
        if (signature !== null) {
            newSignature = renderer?.createUpdateSignature(maskFrame, clip, focusFrame, filters, useMask) ?? null;
            
            if (signature === newSignature) {
                return;
            }
        } else {
            // first update
            newSignature = renderer?.createUpdateSignature(maskFrame, clip, focusFrame, filters, useMask) ?? null;
        }
        await renderer?.update(maskFrame, clip, focusFrame, filters, useMask);
        renderer?.setCurrentSignature(newSignature);
        self.postMessage({ type: "update_complete", data: { id, success: true } });
    } catch (e) {
        console.error("Error updating renderer", e);
        self.postMessage({ type: "update_complete", data: { id, success: false } });
    }
}

self.onmessage = handleMessage;
