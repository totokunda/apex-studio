import {  FilterClipProps, VideoClipProps } from "../types";
import { Renderer } from "./renderer";
import { CompositorShader, FilterParams, WebGLHaldClut } from "@/components/preview/webgl-filters";

class Canvas2DRenderer extends Renderer {
  
  #canvas: OffscreenCanvas;
  #ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  #width: number;
  #height: number;
  #compositor: CompositorShader | null = null;
  #filterParams: FilterParams | null = null;
  #maskFrame: number | null = null;
  #clip: VideoClipProps | null = null;
  #focusFrame: number | null = null;
  #filters: FilterClipProps[] = [];
  #useMask: boolean = true;
  #haldClutInstance: WebGLHaldClut | null = null;
  currentSignature: string | null = null;


  constructor(id: string, postMessage: (message: any) => void, canvas: OffscreenCanvas, width: number, height: number, haldClutInstance: WebGLHaldClut | null) {
    super(id, postMessage);
    this.#canvas = canvas;
    this.#ctx = canvas.getContext("2d")!;
    this.#width = width;
    this.#height = height;
    this.#compositor = new CompositorShader();
    this.#haldClutInstance = haldClutInstance;
  }

  setCurrentSignature(signature: string | null): void {
    this.currentSignature = signature;
  }
  
  getCurrentSignature(): string | null {
    return this.currentSignature;
  }

  createUpdateSignature(maskFrame: number, clip: VideoClipProps, focusFrame: number, filters: FilterClipProps[], useMask: boolean): string {
    const signature = JSON.stringify({
      maskFrame,
      clip,
      focusFrame,
      filters,
      useMask,
    });
    return signature;
  }

  async update(maskFrame: number, clip: VideoClipProps, focusFrame: number, filters: FilterClipProps[], useMask: boolean): Promise<void> {
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
      jitter: clip?.jitter,
    }
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

  drawFrame(frame: VideoFrame): void {
    this.#canvas.width = this.#width;
    this.#canvas.height = this.#height;
    this.#ctx!.drawImage(frame, 0, 0, this.#canvas.width, this.#canvas.height);

    let workingCanvas: OffscreenCanvas | null = null;
    let workingCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;

    if (this.#filterParams && this.#clip && this.#maskFrame !== null && this.#focusFrame !== null) {
      workingCanvas = new OffscreenCanvas(this.#width, this.#height);
      workingCtx = workingCanvas.getContext("2d");
      if (!workingCtx) return;
      workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
      workingCtx.drawImage(this.#canvas, 0, 0);
      const result = this.#compositor?.apply(
        this.#canvas as unknown as HTMLCanvasElement,
        {
          filterParams: this.#filterParams,
          masks: this.#useMask ? this.#clip?.masks ?? [] : [],
          clip: this.#clip,
          maskFrame: this.#maskFrame,
          focusFrame: this.#focusFrame,
          useOriginalTransform: true,
        }
      )

      if (result) {
        try {
          workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
          workingCtx.drawImage(result, 0, 0, workingCanvas.width, workingCanvas.height);
          this.#ctx!.drawImage(workingCanvas, 0, 0, this.#canvas.width, this.#canvas.height);
        } catch {
          /* ignore */
        }
        
      }
    }

    if (this.#filters.length === 0) return;

    if (!workingCanvas || !workingCtx) {
      workingCanvas = new OffscreenCanvas(this.#width, this.#height);
      workingCtx = workingCanvas.getContext("2d");
    }

    let processedCanvas: HTMLCanvasElement | OffscreenCanvas = workingCanvas;
    
    for (const filter of this.#filters) {
      const path = filter.smallPath || filter.fullPath || "";
      const intensity = filter.intensity !== undefined ? filter.intensity / 100 : 1;  
      // check if we are in range
      if (!this.isInFrameRange(filter)) {
        continue;
      }
      const result = this.#haldClutInstance?.apply(processedCanvas as unknown as HTMLCanvasElement, path, intensity) as HTMLCanvasElement | OffscreenCanvas;
      if (result && result !== processedCanvas) {
        workingCtx!.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
        workingCtx!.drawImage(result, 0, 0, workingCanvas.width, workingCanvas.height);
        processedCanvas = workingCanvas;
      }
   }

    this.#ctx!.clearRect(0, 0, this.#canvas.width, this.#canvas.height);
    this.#ctx!.drawImage(processedCanvas, 0, 0, this.#canvas.width, this.#canvas.height);
  }

  isInFrameRange(filter: FilterClipProps): boolean {
    const focusFrame = this.#focusFrame ?? 0;
    const startFrame = filter.startFrame ?? 0;
    const endFrame = filter.endFrame ?? 0;
    return focusFrame >= startFrame && focusFrame <= endFrame;
  }

}

export { Canvas2DRenderer };
