import Scheduler from "./scheduler";
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
  setup({ speed, sourceFps, targetFps, startTimestamp, playbackState, accumlateOnly = false }) {
    this.#speed = speed ?? this.#speed;
    this.#sourceFps = sourceFps ?? this.#sourceFps;
    this.#targetFps = targetFps ?? this.#targetFps;
    this.#displayInterval = this.#speed / this.#targetFps;
    this.#scheduler?.stop();
    this.#scheduler = new Scheduler(this.#id, this.#targetFps, this.#speed, this.drawFrame.bind(this), this.#postMessage.bind(this));
    if (startTimestamp !== void 0) {
      this.#scheduler.start(startTimestamp, playbackState, { accumlateOnly });
    }
  }
  startPlayback() {
    this.#scheduler?.beginTicking();
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
        this.#postMessage({ type: "frame", data: { id: renderState.id, timestamp: frameTimestamp } });
      }
      frame.close();
    } else if (type === "iterate") {
      const { startTimestamp, endTimestamp } = renderState;
      if (frameTimestamp >= startTimestamp && !renderState.stopRender && (!endTimestamp || frameTimestamp <= endTimestamp)) {
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
