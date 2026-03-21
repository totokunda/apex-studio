import { clipSignature } from "../types";
import { Renderer } from "./renderer";
import { CompositorShader } from "@/components/preview/webgl-filters";
function filtersSignature(filters) {
  if (!filters?.length) return "";
  return filters.map(
    (f) => [
      f.clipId ?? "",
      f.smallPath ?? "",
      f.fullPath ?? "",
      f.intensity ?? 100,
      f.startFrame ?? 0,
      f.endFrame ?? 0
    ].join(",")
  ).join("|");
}
function clipRenderSignature(clip) {
  const base = clipSignature(clip);
  const adj = [
    clip?.brightness,
    clip?.contrast,
    clip?.hue,
    clip?.saturation,
    clip?.blur,
    clip?.sharpness,
    clip?.noise,
    clip?.vignette,
    clip?.colorTintColor,
    clip?.colorTintIntensity,
    clip?.scanLines,
    clip?.chromaticAberration,
    clip?.interlace,
    clip?.pixelate,
    clip?.jitter
  ].join(",");
  return `${base};${adj}`;
}
class Canvas2DRenderer extends Renderer {
  #canvas;
  #ctx;
  #width;
  #height;
  #compositor = null;
  #filterParams = null;
  #maskFrame = null;
  #clip = null;
  #focusFrame = null;
  #filters = [];
  #useMask = true;
  #haldClutInstance = null;
  currentSignature = null;
  constructor(id, postMessage, canvas, width, height, haldClutInstance) {
    super(id, postMessage);
    this.#canvas = canvas;
    this.#ctx = canvas.getContext("2d");
    this.#width = width;
    this.#height = height;
    this.#compositor = new CompositorShader();
    this.#haldClutInstance = haldClutInstance;
  }
  setCurrentSignature(signature) {
    this.currentSignature = signature;
  }
  getCurrentSignature() {
    return this.currentSignature;
  }
  createUpdateSignature(maskFrame, clip, focusFrame, filters, useMask) {
    return [
      maskFrame,
      focusFrame,
      useMask ? "1" : "0",
      clipRenderSignature(clip),
      filtersSignature(filters)
    ].join("::");
  }
  async update(maskFrame, clip, focusFrame, filters, useMask) {
    this.#filterParams = {
      brightness: clip?.brightness,
      contrast: clip?.contrast,
      hue: clip?.hue,
      saturation: clip?.saturation,
      blur: clip?.blur,
      sharpness: clip?.sharpness,
      noise: clip?.noise,
      vignette: clip?.vignette,
      colorTintColor: clip?.colorTintColor,
      colorTintIntensity: clip?.colorTintIntensity,
      scanLines: clip?.scanLines,
      chromaticAberration: clip?.chromaticAberration,
      interlace: clip?.interlace,
      pixelate: clip?.pixelate,
      jitter: clip?.jitter
    };
    this.#maskFrame = maskFrame;
    this.#clip = clip;
    this.#focusFrame = focusFrame;
    this.#filters = filters;
    this.#useMask = useMask;
    await Promise.all(this.#filters.map(async (filter) => {
      const path = filter.smallPath || filter.fullPath || "";
      if (path) {
        await this.#haldClutInstance?.preloadClut(path);
      }
    }));
  }
  drawFrame(frame) {
    this.#canvas.width = this.#width;
    this.#canvas.height = this.#height;
    this.#ctx.drawImage(frame, 0, 0, this.#canvas.width, this.#canvas.height);
    let workingCanvas = null;
    let workingCtx = null;
    if (this.#filterParams && this.#clip && this.#maskFrame !== null && this.#focusFrame !== null) {
      workingCanvas = new OffscreenCanvas(this.#width, this.#height);
      workingCtx = workingCanvas.getContext("2d");
      if (!workingCtx) return;
      workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
      workingCtx.drawImage(this.#canvas, 0, 0);
      const result = this.#compositor?.apply(
        this.#canvas,
        {
          filterParams: this.#filterParams,
          masks: this.#useMask ? this.#clip?.masks ?? [] : [],
          clip: this.#clip,
          maskFrame: this.#maskFrame,
          focusFrame: this.#focusFrame,
          useOriginalTransform: true
        }
      );
      if (result) {
        try {
          workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
          workingCtx.drawImage(result, 0, 0, workingCanvas.width, workingCanvas.height);
          this.#ctx.drawImage(workingCanvas, 0, 0, this.#canvas.width, this.#canvas.height);
        } catch {
        }
      }
    }
    if (this.#filters.length === 0) return;
    if (!workingCanvas || !workingCtx) {
      workingCanvas = new OffscreenCanvas(this.#width, this.#height);
      workingCtx = workingCanvas.getContext("2d");
    }
    let processedCanvas = workingCanvas;
    for (const filter of this.#filters) {
      const path = filter.smallPath || filter.fullPath || "";
      const intensity = filter.intensity !== void 0 ? filter.intensity / 100 : 1;
      if (!this.isInFrameRange(filter)) {
        continue;
      }
      const result = this.#haldClutInstance?.apply(processedCanvas, path, intensity);
      if (result && result !== processedCanvas) {
        workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
        workingCtx.drawImage(result, 0, 0, workingCanvas.width, workingCanvas.height);
        processedCanvas = workingCanvas;
      }
    }
    this.#ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height);
    this.#ctx.drawImage(processedCanvas, 0, 0, this.#canvas.width, this.#canvas.height);
  }
  isInFrameRange(filter) {
    const focusFrame = this.#focusFrame ?? 0;
    const startFrame = filter.startFrame ?? 0;
    const endFrame = filter.endFrame ?? 0;
    return focusFrame >= startFrame && focusFrame <= endFrame;
  }
}
export {
  Canvas2DRenderer
};
