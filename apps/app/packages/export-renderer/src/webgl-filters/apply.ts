import { CompositorShader } from "../../../renderer/src/components/preview/webgl-filters/compositor";

export interface FilterParams {
  brightness?: number; // -100 to 100
  contrast?: number; // -100 to 100
  hue?: number; // -100 to 100
  saturation?: number; // -100 to 100
  blur?: number; // 0 to 100
  noise?: number; // 0 to 100
  sharpness?: number; // 0 to 100
  vignette?: number; // 0 to 100
  // Found Footage / Stylize
  colorTintColor?: string; // hex color e.g. "#00ff4c"
  colorTintIntensity?: number; // 0 to 100
  scanLines?: number; // 0 to 100
  chromaticAberration?: number; // 0 to 100
  interlace?: number; // 0 to 100
  pixelate?: number; // 0 to 100
  jitter?: number; // 0 to 100
}

const blitResultToSource = (
  ctx: CanvasRenderingContext2D,
  sourceCanvas: HTMLCanvasElement,
  result: HTMLCanvasElement,
) => {
  ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  ctx.imageSmoothingEnabled = true;
  // @ts-ignore
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(result, 0, 0, sourceCanvas.width, sourceCanvas.height);
};

let compositorInstance: CompositorShader | null = null;

const getCompositor = (): CompositorShader => {
  if (!compositorInstance) {
    compositorInstance = new CompositorShader();
  }
  return compositorInstance;
};

const hasAnyActiveFilter = (params: FilterParams): boolean => {
  return (
    (params.brightness !== undefined && params.brightness !== 0) ||
    (params.contrast !== undefined && params.contrast !== 0) ||
    (params.hue !== undefined && params.hue !== 0) ||
    (params.saturation !== undefined && params.saturation !== 0) ||
    (params.blur !== undefined && params.blur > 0) ||
    (params.noise !== undefined && params.noise > 0) ||
    (params.sharpness !== undefined && params.sharpness > 0) ||
    (params.vignette !== undefined && params.vignette > 0) ||
    (params.colorTintIntensity !== undefined &&
      params.colorTintIntensity > 0 &&
      !!params.colorTintColor) ||
    (params.scanLines !== undefined && params.scanLines > 0) ||
    (params.chromaticAberration !== undefined &&
      params.chromaticAberration > 0) ||
    (params.interlace !== undefined && params.interlace > 0) ||
    (params.pixelate !== undefined && params.pixelate > 0) ||
    (params.jitter !== undefined && params.jitter > 0)
  );
};

export const disposeWebGLFilters = (): void => {
  if (compositorInstance) {
    compositorInstance.dispose();
    compositorInstance = null;
  }
};

export function applyWebGLFilters(
  sourceCanvas: HTMLCanvasElement,
  params: FilterParams,
): HTMLCanvasElement {
  const ctx = sourceCanvas.getContext("2d");
  if (!ctx) return sourceCanvas;
  if (!hasAnyActiveFilter(params)) return sourceCanvas;

  const compositor = getCompositor();
  const compositorResult = compositor.apply(sourceCanvas, {
    filterParams: params,
  });

  if (compositorResult && compositorResult !== sourceCanvas) {
    blitResultToSource(ctx, sourceCanvas, compositorResult);
  }

  return sourceCanvas;
}
