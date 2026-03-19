import { Input, ALL_FORMATS, UrlSource, EncodedPacketSink, Source, PacketStats, EncodedPacket } from 'mediabunny';
import { WebGLRenderer } from './renderer_webgl';
import { Canvas2DRenderer } from './renderer_2d';
import { fileURLToPathInWorker, createNodeFileSource, readFileBuffer } from './utils';
import { FilterClipProps, VideoClipProps } from '../types';
import { WebGLHaldClut } from '@/components/preview/webgl-filters/hald-clut';

const { existsSync } = require('fs');

const VERSION = 1;  
const DATABASE_NAME = "media-packets";

type InitPayload = {
    type: "init"
    data: {
        id: string;
        canvas: OffscreenCanvas;
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
const lastPacketStates = new Map<string, EncodedPacket>();
const renderers = new Map<string, WebGLRenderer | Canvas2DRenderer>();
const decoderInUseStates = new Map<string, boolean>();
const requestDecoderStates = new Map<string, boolean>();

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
        case "preload":
            preload(payload);
            break;
    }
}

const init = async (payload: InitPayload): Promise<void> => {
    const { canvas, path, id, renderer: rendererName = "2d", width, height } = payload.data;
    const source = resolveSource(path);
    canvasStates.set(id, canvas);

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
            packetStats: packetStats
        });


        const onFrame = async (frame: VideoFrame) => {
           
            if (!renderer) {
                frame.close();
                return;
            }
    
            let renderState = renderStates.get(id);
                
            if (!renderState) {
                frame.close();
                console.error("Render state not found");
                return;
            }
    
            try {
                renderer.draw(frame, renderState);
            } catch (e) {
                // draw threw - frame was never enqueued (iterate) or closed (seek)
                frame.close();
                throw e;
            }

        }

        const onError = (error: Error) => {
            console.log(error);
        }

        const decoder = createDecoderState(videoStates.get(id)!, onFrame, onError);
        decoderStates.set(id, decoder);

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

const createDecoderState = (videoState: VideoState, onFrame: (frame: VideoFrame) => void, onError: (error: Error) => void): VideoDecoder => {
    const decoder = new VideoDecoder({
        output: (frame) => {
            onFrame(frame);
        },
        error: (error) => {
            onError(error);
        }
    });

    decoder.configure(videoState.videoConfig);
    return decoder;
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


    for await (const packet of sink.packets(keypacket, endPacket)) {
        decoder.decode(packet.toEncodedVideoChunk());
        lastPacketStates.set(id, packet);
        if (renderState.stopDecode === true) {
            break;
        }
    }
    
    await decoder.flush();

}

const preload = async (payload: PreloadPayload): Promise<void> => {
    let { id, startTimestamp, endTimestamp:realEndTimestamp, secondsToPrefetch, targetFps, speed, playbackState } = payload.data;
    const videoState = videoStates.get(id);
    const sink = videoState?.sink;
    const keyframes = videoState?.keyframePackets;
    const decoder = decoderStates.get(id);
    const sourceFps = videoState?.packetStats.averagePacketRate ?? 30;
    renderStates.delete(id);
    const renderer = renderers.get(id);

    renderer?.stop?.();
    renderer?.setup({ speed, sourceFps, targetFps, startTimestamp, playbackState, accumlateOnly: true });
    const endTimestamp = startTimestamp + secondsToPrefetch;

    if (!sink || !keyframes || !decoder) {
        console.error("Sink, keyframes, or decoder not found");
        return;
    }

    let renderState = {
        id,
        type: "iterate",
        startTimestamp,
        endTimestamp: realEndTimestamp ?? endTimestamp,
        stopDecode: false,
        stopRender: false,
        preload: true
    } as RenderState;

    renderStates.set(id, renderState);

    const startKeyframeIndex = binarySearch(keyframes, startTimestamp);
    const startKeyFrame = keyframes[startKeyframeIndex];
    let startKeyFramePacket = await sink.getPacket(startKeyFrame.timestamp) ?? undefined;

    const metadataOnly = { metadataOnly: true };
    const targetPacket = await sink.getPacket(endTimestamp, metadataOnly);
    const endPacket = targetPacket ? await sink.getNextPacket(targetPacket, metadataOnly) ?? undefined : undefined;
    const iterator = sink.packets(startKeyFramePacket, endPacket);

   
    // prefetch the packets
    if (decoderInUseStates.get(id) === true || requestDecoderStates.get(id) === true) {  // if decoder is in use, don't prefetch
        return;
    }

    decoderInUseStates.set(id, true);

    for await (const packet of iterator) {
        if (requestDecoderStates.get(id) === true) {  // if decoder is in use, don't prefetch
            break;
        }
        // we can yield on every packet since we don't really care about speed here, we just want to prefetch the packets
        await new Promise(r => requestAnimationFrame(r));
        decoder.decode(packet.toEncodedVideoChunk());
    }

    await decoder.flush();
    decoderInUseStates.set(id, false);

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

    requestDecoderStates.set(id, true);

    while (decoderInUseStates.get(id) === true) {
        await new Promise(r => requestAnimationFrame(r));
    }

    requestDecoderStates.set(id, false);
    decoderInUseStates.set(id, true);
    
    for await (const packet of iterator) {
        decoder.decode(packet.toEncodedVideoChunk());

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

    await decoder.flush();
    decoderInUseStates.set(id, false);
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
    const { id } = payload.data;
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
    canvasStates.delete(id);
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
