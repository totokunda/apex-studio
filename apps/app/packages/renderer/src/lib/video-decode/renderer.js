class Scheduler {
  targetFps;
  speed;
  drawFrame;
  postMessage;
  startWallTime = 0;
  startTimestamp = 0;
  running = false;
  frameQueue = [];
  lastDisplayedFrame = null;
  lastDrawWallTime = 0;
  id;
  constructor(id, targetFps, speed, drawFrame, postMessage) {
    this.id = id;
    this.targetFps = targetFps;
    this.speed = speed;
    this.drawFrame = drawFrame;
    this.postMessage = postMessage;
  }
  start(startTimestamp) {
    this.stop();
    this.startWallTime = performance.now();
    this.startTimestamp = startTimestamp;
    this.lastDrawWallTime = 0;
    this.running = true;
    this.tick();
  }
  stop() {
    this.running = false;
    this.lastDisplayedFrame?.close();
    this.lastDisplayedFrame = null;
    for (const { frame } of this.frameQueue) {
      frame.close();
    }
    this.frameQueue = [];
  }
  enqueueFrame(frame) {
    const timestamp = frame.timestamp / 1e6;
    this.frameQueue.push({ frame, timestamp });
  }
  tick = () => {
    if (!this.running) return;
    const now = performance.now();
    const minDrawIntervalMs = 1e3 / (this.targetFps * this.speed);
    const canDraw = now - this.lastDrawWallTime >= minDrawIntervalMs;
    if (canDraw) {
      const elapsedSeconds = (now - this.startWallTime) / 1e3;
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
  selectFrame(currentMediaTime) {
    let best = null;
    for (const item of this.frameQueue) {
      if (item.timestamp <= currentMediaTime) {
        best = item;
      } else {
        break;
      }
    }
    return best;
  }
  discardFramesUpTo(keepTimestamp) {
    while (this.frameQueue.length > 0 && this.frameQueue[0].timestamp <= keepTimestamp) {
      const item = this.frameQueue.shift();
      if (item.frame !== this.lastDisplayedFrame) {
        item.frame.close();
      }
    }
  }
}
class Renderer {
  #speed = 1;
  #sourceFps = 30;
  #targetFps = 30;
  #displayInterval = null;
  #scheduler = null;
  #postMessage;
  #id;
  constructor(id, postMessage) {
    this.#id = id;
    this.#postMessage = postMessage;
  }
  setup({ speed, sourceFps, targetFps, startTimestamp }) {
    this.#speed = speed ?? this.#speed;
    this.#sourceFps = sourceFps ?? this.#sourceFps;
    this.#targetFps = targetFps ?? this.#targetFps;
    this.#displayInterval = this.#speed / this.#targetFps;
    this.#scheduler?.stop();
    this.#scheduler = new Scheduler(this.#id, this.#targetFps, this.#speed, this.drawFrame.bind(this), this.#postMessage.bind(this));
    if (startTimestamp !== void 0) {
      this.#scheduler.start(startTimestamp);
    }
  }
  stop() {
    this.#scheduler?.stop();
  }
  async draw(frame, renderState) {
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
export {
  Renderer
};
