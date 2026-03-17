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
}

const videoStates = new Map<string, VideoState>();
const canvasStates = new Map<string, OffscreenCanvas>();
const encodedPacketSinkStates = new Map<string, EncodedPacketSink>();
const renderStates = new Map<string, RenderState>();
const decoderStates = new Map<string, VideoDecoder>();
const lastPacketStates = new Map<string, EncodedPacket>();
const renderers = new Map<string, WebGLRenderer | Canvas2DRenderer>();

const seekQueueStates:Map<string, SeekPayload | null> = new Map();
const seekInProgressStates:Map<string, boolean> = new Map();

const haldClutInstance = new WebGLHaldClut(readFileBuffer);

type Payload = InitPayload | SeekPayload | IteratePayload | PausePayload | DestroyPayload | UpdateRendererPayload;

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
        throw new Error("No video track found");
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
        throw new Error("Invalid renderer");
    }

    renderers.set(id, renderer);

    let sink: EncodedPacketSink;

    if (!videoConfig) {
        throw new Error("No video config found");
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
                throw new Error("Render state not found");
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
    renderer?.stop?.();


    const sourceFps = videoState?.packetStats.averagePacketRate ?? 30;
    renderer?.setup({ speed, sourceFps, targetFps, startTimestamp: timestamp });

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
        throw new Error("Video state not found");
    }

    const sink = videoState.sink;
    // Binary search: rightmost keyframe with timestamp <= target (O(log k) vs O(n))
    const keyframes = videoState.keyframePackets;

    if (keyframes.length === 0) {
        throw new Error("No keyframes found");
    }

    let keypacket: EncodedPacket | undefined;
    const keypacketIndex = binarySearch(keyframes, timestamp);
    const currentKeyframe = keyframes[keypacketIndex];

    keypacket = await sink.getPacket(currentKeyframe.timestamp) ?? undefined;

    if (!keypacket) throw new Error("Cannot find keypacket")

    const metadataOnly = { metadataOnly: true };
    const targetPacket = await sink.getPacket(timestamp, metadataOnly);
    const endPacket = targetPacket ? await sink.getNextPacket(targetPacket, metadataOnly) ?? undefined : undefined;

    const decoder = decoderStates.get(id);
    if (!decoder) {
        throw new Error("Decoder not found");
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

const iterate = async (payload: IteratePayload): Promise<void> => {
    const { id, startTimestamp, endTimestamp, speed, targetFps } = payload.data;
    const videoState = videoStates.get(id);

    const sourceFps = videoState?.packetStats.averagePacketRate ?? 30;
    // clear previous render states
    renderStates.delete(id);
    const renderer = renderers.get(id);
    renderer?.setup({ speed, sourceFps, targetFps, startTimestamp });
    if (!renderer) {
        throw new Error("Renderer not found");
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
        throw new Error("Video state not found");
    }

    const sink = videoState.sink;
    const keyframes = videoState.keyframePackets;
    const decoder = decoderStates.get(id);

    if (!decoder) {
        throw new Error("Decoder not found");
    }

    const startKeyframeIndex = binarySearch(keyframes, startTimestamp);
    const startKeyFrame = keyframes[startKeyframeIndex];
    let startKeyFramePacket: EncodedPacket | undefined;

    startKeyFramePacket = await sink.getPacket(startKeyFrame.timestamp) ?? undefined;

    const metadataOnly = { metadataOnly: true };
    const targetPacket = endTimestamp ? await sink.getPacket(endTimestamp, metadataOnly) : undefined;
    const endPacket = targetPacket
    ? await sink.getNextPacket(targetPacket, metadataOnly) ?? undefined : undefined;

    if (!startKeyFramePacket) {
        throw new Error("Cannot find start keyframe packet");
    }

    let packetCount = 0;
    const fps = Math.round(videoStates.get(id)?.packetStats.averagePacketRate ?? 30);

    for await (const packet of sink.packets(startKeyFramePacket, endPacket)) {
        decoder.decode(packet.toEncodedVideoChunk());

        if (renderState.stopDecode === true) {
            decoder.reset();
            decoder.configure(videoState.videoConfig);
            break;
        }

        if (packetCount % fps === 0) {
            await new Promise(r => setTimeout(r, 0));
        }

        packetCount++;
    }

    await decoder.flush();

}

const pause = async (payload: PausePayload): Promise<void> => {
    const { id } = payload.data;
    const renderState = renderStates.get(id);
    const renderer = renderers.get(id);

    if (!renderState) {
        throw new Error("Render state not found");
    }

    renderState.stopDecode = true;
    renderState.stopRender = true;
    renderer?.stop?.();
}

const destroy = async (payload: DestroyPayload): Promise<void> => {
    const { id } = payload.data;
    const renderer = renderers.get(id);
    renderer?.stop?.();
    const renderState = renderStates.get(id);
    if (!renderState) {
        return;
    }   

    renderState.stopDecode = true;
    renderState.stopRender = true;
    try {
        await decoderStates.get(id)?.close();
    } catch (e) {
        console.error("Error closing decoder", e);
    }

    videoStates.delete(id);
    canvasStates.delete(id);
    encodedPacketSinkStates.delete(id);
    renderStates.delete(id);
    decoderStates.delete(id);
    lastPacketStates.delete(id);
    renderers.delete(id);
    seekQueueStates.delete(id);
    seekInProgressStates.delete(id);
}

const updateRenderer = async (payload: UpdateRendererPayload): Promise<void> => {
    const { id, maskFrame, clip, focusFrame, filters, useMask = true } = payload.data;
    const renderer = renderers.get(id);
    try {
        await renderer?.update(maskFrame, clip, focusFrame, filters, useMask);
        self.postMessage({ type: "update_complete", data: { id, success: true } });
    } catch (e) {
        console.error("Error updating renderer", e);
        self.postMessage({ type: "update_complete", data: { id, success: false } });
    }
}

self.onmessage = handleMessage;
