import { WebGLBlur } from "../../../renderer/src/components/preview/webgl-filters/blur";
import { WebGLBrightness } from "../../../renderer/src/components/preview/webgl-filters/brightness";
import { WebGLContrast } from "../../../renderer/src/components/preview/webgl-filters/contrast";
import { WebGLHueSaturation } from "../../../renderer/src/components/preview/webgl-filters/hue-saturation";
import { WebGLNoise } from "../../../renderer/src/components/preview/webgl-filters/noise";
import { WebGLSharpness } from "../../../renderer/src/components/preview/webgl-filters/sharpness";
import { WebGLVignette } from "../../../renderer/src/components/preview/webgl-filters/vignette";

export interface FilterParams {
  brightness?: number; // -100 to 100
  contrast?: number; // -100 to 100
  hue?: number; // -100 to 100
  saturation?: number; // -100 to 100
  blur?: number; // 0 to 100
  noise?: number; // 0 to 100
  sharpness?: number; // 0 to 100
  vignette?: number; // 0 to 100
}

export function applyWebGLFilters(
  sourceCanvas: HTMLCanvasElement,
  params: FilterParams,
): HTMLCanvasElement {
  const ctx = sourceCanvas.getContext("2d");
  if (!ctx) return sourceCanvas;

  let currentCanvas: HTMLCanvasElement = sourceCanvas;

  const disposeFns: Array<() => void> = [];

  try {
    // Brightness
    if (params.brightness && params.brightness !== 0) {
      const filter = new WebGLBrightness();
      disposeFns.push(() => filter.dispose());
      const brightnessValue = params.brightness / 100;
      const result = filter.apply(currentCanvas, brightnessValue);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Contrast
    if (params.contrast && params.contrast !== 0) {
      const filter = new WebGLContrast();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.contrast);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Hue & Saturation
    if (
      (params.hue && params.hue !== 0) ||
      (params.saturation && params.saturation !== 0)
    ) {
      const filter = new WebGLHueSaturation();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(
        currentCanvas,
        params.hue ?? 0,
        params.saturation ?? 0,
      );
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Blur
    if (params.blur && params.blur > 0) {
      const filter = new WebGLBlur();
      disposeFns.push(() => filter.dispose());
      const blurRadius = (params.blur / 100) * 10;
      const result = filter.apply(currentCanvas, blurRadius);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Sharpness
    if (params.sharpness && params.sharpness > 0) {
      const filter = new WebGLSharpness();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.sharpness);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Noise
    if (params.noise && params.noise > 0) {
      const filter = new WebGLNoise();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.noise);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Vignette
    if (params.vignette && params.vignette > 0) {
      const filter = new WebGLVignette();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.vignette);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    return sourceCanvas;
  } finally {
    for (const d of disposeFns) d();
  }
}
