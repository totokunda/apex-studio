import { FilterClipProps, VideoClipProps } from "../types";

const worker = new Worker(new URL('./dist/worker.js', import.meta.url), {
    type: "module",
});

interface InitParams {
    canvas: HTMLCanvasElement;
    sourceOrPath: string;
    id: string;
    renderer: "webgl2" | "webgpu" | "2d";
    onInitComplete?: (data: { id: string, duration: number }) => void;
    onFrame?: (data:{
        id: string,
        timestamp: number
    }) => void;
    onUpdateComplete?: (data: { id: string, success: boolean }) => void;
    width: number;
    height: number;
}

interface SeekParams {
    timestamp: number;
    id: string;
    speed: number;
    targetFps: number;
}

interface IterateParams {
    startTimestamp: number;
    endTimestamp?: number;
    id: string;
    speed: number;
    targetFps: number;
    playbackState?: {
        startWallTime: number;
        startFocusFrame: number;
        isPlaying: boolean;
        mainNow: number;
    };
}

interface PauseParams {
    id: string;
}

interface DestroyParams {
    id: string;
}

interface UpdateRendererParams {
    id: string;
    maskFrame: number;
    clip: VideoClipProps;
    focusFrame: number;
    filters: FilterClipProps[];
    useMask: boolean;
}


class VideoDecoderModule {
    private worker: Worker;
    private transferredCanvases = new WeakSet<HTMLCanvasElement>();
    private initCallbacks = new Map<string, (data: { id: string, duration: number }) => void>();
    private frameCallbacks = new Map<string, (data: { id: string, timestamp: number }) => void>();
    private updateCallbacks = new Map<string, (data: { id: string, success: boolean }) => void>();

    constructor() {
        this.worker = worker;
        this.worker.addEventListener("message", (e: MessageEvent) => {
            if (e.data?.type === "init_complete") {
                const { id } = e.data.data;
                const cb = this.initCallbacks.get(id);
                if (cb) {
                    cb(e.data.data);
                }
            }
            if (e.data?.type === "frame") {
                const { id, timestamp } = e.data.data;
                const cb = this.frameCallbacks.get(id);
                if (cb) {
                    cb({ id, timestamp });
                }
            }
            if (e.data?.type === "update_complete") {
                const { id, success } = e.data.data;
                const cb = this.updateCallbacks.get(id);
                if (cb) {
                    cb({ id, success });
                }
            }
        });
    }

    init(params: InitParams) {
        const { canvas, sourceOrPath, id, renderer, onInitComplete, onFrame, onUpdateComplete, width, height } = params;

        if (this.transferredCanvases.has(canvas)) {
            return;
        }

        if (onInitComplete) {
            this.initCallbacks.set(id, onInitComplete);
        }

        if (onFrame) {

            this.frameCallbacks.set(id, onFrame);
        }

        if (onUpdateComplete) {
            this.updateCallbacks.set(id, onUpdateComplete);
        }

        // check if the canvas is already transferred 
        const offscreenCanvas = canvas.transferControlToOffscreen();
        this.transferredCanvases.add(canvas);

        this.worker.postMessage({
            type: "init",
            data: {
                canvas: offscreenCanvas,
                path:sourceOrPath,
                id,
                renderer,
                width,
                height
            }
        }, [offscreenCanvas]);
    }

    seek(params: SeekParams) {
        const { timestamp, id, speed, targetFps } = params;
        this.worker.postMessage({
            type: "seek",
            data: {
                id,
                timestamp,
                speed,
                targetFps
            }
        });

    }

    iterate(params: IterateParams) {
        const { startTimestamp, endTimestamp, id, speed, targetFps, playbackState } = params;

        this.worker.postMessage({
            type: "iterate",
            data: {
                id,
                startTimestamp,
                endTimestamp,
                speed,
                targetFps,
                playbackState
            }
        });

    }

    pause(params: PauseParams) {
        const { id } = params;
        this.worker.postMessage({
            type: "pause",
            data: {
                id
            }
        });
    }

    destroy(params: DestroyParams) {
        const { id } = params;

        this.worker.postMessage({
            type: "destroy",
            data: {
                id
            }
        });
    }

    updateRenderer(params: UpdateRendererParams) {
        const { id, maskFrame, clip, focusFrame, filters, useMask } = params;
        this.worker.postMessage({
            type: "update",
            data: {
                id, maskFrame, clip, focusFrame, filters, useMask
            }
        });
    }

}

export default VideoDecoderModule;