import { WebGLBlur } from "../../../renderer/src/components/preview/webgl-filters/blur";
import { WebGLBrightness } from "../../../renderer/src/components/preview/webgl-filters/brightness";
import { WebGLChromaticAberration } from "../../../renderer/src/components/preview/webgl-filters/chromatic-aberration";
import { WebGLColorTint } from "../../../renderer/src/components/preview/webgl-filters/color-tint";
import { WebGLContrast } from "../../../renderer/src/components/preview/webgl-filters/contrast";
import { WebGLHueSaturation } from "../../../renderer/src/components/preview/webgl-filters/hue-saturation";
import { WebGLInterlace } from "../../../renderer/src/components/preview/webgl-filters/interlace";
import { WebGLJitter } from "../../../renderer/src/components/preview/webgl-filters/jitter";
import { WebGLNoise } from "../../../renderer/src/components/preview/webgl-filters/noise";
import { WebGLPixelate } from "../../../renderer/src/components/preview/webgl-filters/pixelate";
import { WebGLScanLines } from "../../../renderer/src/components/preview/webgl-filters/scan-lines";
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
  // Found Footage / Stylize
  colorTintColor?: string; // hex color e.g. "#00ff4c"
  colorTintIntensity?: number; // 0 to 100
  scanLines?: number; // 0 to 100
  chromaticAberration?: number; // 0 to 100
  interlace?: number; // 0 to 100
  pixelate?: number; // 0 to 100
  jitter?: number; // 0 to 100
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

    // Color Tint
    if (
      params.colorTintIntensity &&
      params.colorTintIntensity > 0 &&
      params.colorTintColor
    ) {
      const filter = new WebGLColorTint();
      disposeFns.push(() => filter.dispose());
      const hex = params.colorTintColor.replace("#", "");
      const r = parseInt(hex.substring(0, 2), 16) / 255;
      const g = parseInt(hex.substring(2, 4), 16) / 255;
      const b = parseInt(hex.substring(4, 6), 16) / 255;
      const result = filter.apply(
        currentCanvas,
        r,
        g,
        b,
        params.colorTintIntensity,
      );
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Scan Lines
    if (params.scanLines && params.scanLines > 0) {
      const filter = new WebGLScanLines();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.scanLines);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Chromatic Aberration
    if (params.chromaticAberration && params.chromaticAberration > 0) {
      const filter = new WebGLChromaticAberration();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.chromaticAberration);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Interlace
    if (params.interlace && params.interlace > 0) {
      const filter = new WebGLInterlace();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.interlace);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Pixelate
    if (params.pixelate && params.pixelate > 0) {
      const filter = new WebGLPixelate();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.pixelate);
      ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      ctx.save();
      ctx.scale(1, -1);
      ctx.drawImage(result, 0, -sourceCanvas.height);
      ctx.restore();
      currentCanvas = sourceCanvas;
    }

    // Jitter
    if (params.jitter && params.jitter > 0) {
      const filter = new WebGLJitter();
      disposeFns.push(() => filter.dispose());
      const result = filter.apply(currentCanvas, params.jitter);
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
