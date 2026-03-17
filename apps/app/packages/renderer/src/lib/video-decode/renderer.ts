export interface Canvas2DRendererSetup {
  speed?: number;
  sourceFps?: number;
  targetFps?: number;
  startTimestamp?: number;
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
  stopDecode: boolean;
  stopRender: boolean;
}

type QueuedFrame = { frame: VideoFrame; timestamp: number };

class Scheduler {
  private targetFps: number;
  private speed: number;
  private drawFrame: (frame: VideoFrame) => void;
  private postMessage: (message: any) => void;
  private startWallTime: number = 0;
  private startTimestamp: number = 0;
  private running: boolean = false;
  private frameQueue: QueuedFrame[] = [];
  private lastDisplayedFrame: VideoFrame | null = null;
  private lastDrawWallTime: number = 0;
  private id: string;

  constructor(id: string, targetFps: number, speed: number, drawFrame: (frame: VideoFrame) => void, postMessage: (message: any) => void) {
    this.id = id;
    this.targetFps = targetFps;
    this.speed = speed;
    this.drawFrame = drawFrame;
    this.postMessage = postMessage;
  }

  start(startTimestamp: number): void {
    this.stop();
    this.startWallTime = performance.now();
    this.startTimestamp = startTimestamp;
    this.lastDrawWallTime = 0;
    this.running = true;
    this.tick();
  }

  stop(): void {
    this.running = false;
    this.lastDisplayedFrame?.close();
    this.lastDisplayedFrame = null;
    for (const { frame } of this.frameQueue) {
      frame.close();
    }
    this.frameQueue = [];
  }

  enqueueFrame(frame: VideoFrame): void {
    const timestamp = frame.timestamp / 1e6;
    this.frameQueue.push({ frame, timestamp });
  }

  private tick = (): void => {
    if (!this.running) return;

    const now = performance.now();
    const minDrawIntervalMs = 1000 / (this.targetFps * this.speed);
    const canDraw = (now - this.lastDrawWallTime) >= minDrawIntervalMs;

    if (canDraw) {
      const elapsedSeconds = (now - this.startWallTime) / 1000;
      const currentMediaTime = this.startTimestamp + elapsedSeconds * this.speed;

      const frameToDisplay = this.selectFrame(currentMediaTime);
      if (frameToDisplay) {
        if (this.lastDisplayedFrame && this.lastDisplayedFrame !== frameToDisplay.frame) {
          this.lastDisplayedFrame.close();
        }
        this.drawFrame(frameToDisplay.frame);
        this.lastDisplayedFrame = frameToDisplay.frame;
        this.lastDrawWallTime = now;
        this.discardFramesUpTo(frameToDisplay.timestamp);
        this.postMessage({ type: "frame", data: { id: this.id, timestamp: frameToDisplay.frame.timestamp } });
      }
    }

    if (this.running) {
      requestAnimationFrame(this.tick);
    }
  };

  private selectFrame(currentMediaTime: number): QueuedFrame | null {
    let best: QueuedFrame | null = null;
    for (const item of this.frameQueue) {
      if (item.timestamp <= currentMediaTime) {
        best = item;
      } else {
        break;
      }
    }
    return best;
  }

  private discardFramesUpTo(keepTimestamp: number): void {
    while (this.frameQueue.length > 0 && this.frameQueue[0].timestamp <= keepTimestamp) {
      const item = this.frameQueue.shift()!;
      if (item.frame !== this.lastDisplayedFrame) {
        item.frame.close();
      }
    }
  }
}

abstract class Renderer {
  #speed: number = 1;
  #sourceFps: number = 30;
  #targetFps: number = 30;
  #displayInterval: number | null = null;
  #scheduler: Scheduler | null = null;
  #postMessage: (message: any) => void;
  #id: string;

  constructor(id: string, postMessage: (message: any) => void) {
    this.#id = id;
    this.#postMessage = postMessage;
  }

  

  setup({ speed, sourceFps, targetFps, startTimestamp }: Canvas2DRendererSetup): void {
    this.#speed = speed ?? this.#speed;
    this.#sourceFps = sourceFps ?? this.#sourceFps;
    this.#targetFps = targetFps ?? this.#targetFps;
    this.#displayInterval = this.#speed / this.#targetFps;

    this.#scheduler?.stop();
    this.#scheduler = new Scheduler(this.#id, this.#targetFps, this.#speed, this.drawFrame.bind(this), this.#postMessage.bind(this));

    if (startTimestamp !== undefined) {
      this.#scheduler.start(startTimestamp);
    }
  }

  stop(): void {
    this.#scheduler?.stop();
  }
  
  abstract drawFrame(frame: VideoFrame): void;
  abstract update(...args: any[]): void;

  public async draw(frame: VideoFrame, renderState: RenderState): Promise<void> {
    if (!this.#displayInterval) {
      throw new Error("Display interval not set. Please call setup() first.");
    }

    const { type } = renderState;
    const frameTimestamp = frame.timestamp / 1e6;

    if (type === "seek") {
      const { timestamp: seekTimestamp } = renderState;
      if (Math.abs(frameTimestamp - seekTimestamp) < 0.01 && !renderState.stopRender) {
        this.drawFrame(frame);
        renderState.stopDecode = true;
        this.#postMessage({ type: "frame", data: { id: renderState.id, timestamp: frame.timestamp } });
      }
      frame.close();
    } else if (type === "iterate") {
      const { startTimestamp, endTimestamp } = renderState;
      if (frameTimestamp >= startTimestamp && !renderState.stopRender) {
        this.#scheduler?.enqueueFrame(frame);
      } else {
        frame.close();
      }
      if (endTimestamp && frameTimestamp >= endTimestamp) {
        renderState.stopDecode = true;
      }
    }
  }
}

export { Renderer };
