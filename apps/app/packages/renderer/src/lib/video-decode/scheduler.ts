
type QueuedFrame = { frame: VideoFrame; timestamp: number };


class Scheduler {
    private targetFps: number;
    private speed: number;
    private drawFrame: (frame: VideoFrame) => void;
    private postMessage: (message: any) => void;
    private startWallTime: number = 0;
    private startTimestamp: number = 0;
    private running: boolean = false;
    public tickScheduled: boolean = false;
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
  
    start(startTimestamp: number, playbackState?: { startWallTime: number; startFocusFrame: number; isPlaying: boolean; mainNow: number }, options?:{
        accumlateOnly?: boolean;
    }): void {
      this.stop();
      // need to sync mainNow with 
      const workerNow = performance.now();
      if (playbackState) {
        const mainNow = playbackState.mainNow;
        const offset = mainNow - workerNow;
        this.startWallTime = playbackState.startWallTime - offset;
        this.startTimestamp = startTimestamp;
      } else {
        this.startWallTime = performance.now();
        this.startTimestamp = startTimestamp;
      }
      
      this.lastDrawWallTime = 0;
      this.running = true;
      if (!options?.accumlateOnly) {
        this.tickScheduled = true;
        this.tick();
      }
    }

    beginTicking(): void {
        if (this.running && !this.tickScheduled) {
          this.tickScheduled = true;
          this.tick();
    }
   }
  
    stop(): void {
      this.running = false;
      this.tickScheduled = false;
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
          this.postMessage({ type: "frame", data: { id: this.id, timestamp: frameToDisplay.timestamp } });
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


export default Scheduler;