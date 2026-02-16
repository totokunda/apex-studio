/**
 * Custom hook for managing and applying WebGL filters
 * Provides a reusable interface for applying GPU-accelerated filters to canvas elements
 */

import { useRef, useCallback, useEffect } from "react";
import {
  WebGLBlur,
  WebGLBrightness,
  WebGLChromaticAberration,
  WebGLColorTint,
  WebGLContrast,
  WebGLHueSaturation,
  WebGLInterlace,
  WebGLJitter,
  WebGLNoise,
  WebGLPixelate,
  WebGLScanLines,
  WebGLSharpness,
  WebGLVignette,
} from "./index";

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

export interface ApplyFilterOptions {
  // "passthrough": return the latest GPU canvas to avoid an extra copy.
  // "source": copy final output back to sourceCanvas for in-place compatibility.
  output?: "passthrough" | "source";
}

export function useWebGLFilters() {
  const webglBlurRef = useRef<WebGLBlur | null>(null);
  const webglBrightnessRef = useRef<WebGLBrightness | null>(null);
  const webglChromaticAberrationRef =
    useRef<WebGLChromaticAberration | null>(null);
  const webglColorTintRef = useRef<WebGLColorTint | null>(null);
  const webglContrastRef = useRef<WebGLContrast | null>(null);
  const webglHueSaturationRef = useRef<WebGLHueSaturation | null>(null);
  const webglInterlaceRef = useRef<WebGLInterlace | null>(null);
  const webglJitterRef = useRef<WebGLJitter | null>(null);
  const webglNoiseRef = useRef<WebGLNoise | null>(null);
  const webglPixelateRef = useRef<WebGLPixelate | null>(null);
  const webglScanLinesRef = useRef<WebGLScanLines | null>(null);
  const webglSharpnessRef = useRef<WebGLSharpness | null>(null);
  const webglVignetteRef = useRef<WebGLVignette | null>(null);

  /**
   * Apply WebGL filters to a canvas in sequence
   * Returns the filtered canvas or the original if no filters need to be applied
   */
  const applyFilters = useCallback(
    (
      sourceCanvas: HTMLCanvasElement,
      params: FilterParams,
      options?: ApplyFilterOptions,
    ): HTMLCanvasElement => {
      let currentCanvas: HTMLCanvasElement = sourceCanvas;
      let hasAnyFilter = false;

      // Apply brightness
      if (params.brightness && params.brightness !== 0) {
        hasAnyFilter = true;
        if (!webglBrightnessRef.current) {
          webglBrightnessRef.current = new WebGLBrightness();
        }
        const brightnessValue = params.brightness / 100; // Map -100..100 to -1..1
        currentCanvas = webglBrightnessRef.current.apply(
          currentCanvas,
          brightnessValue,
        );
      }

      // Apply contrast
      if (params.contrast && params.contrast !== 0) {
        hasAnyFilter = true;
        if (!webglContrastRef.current) {
          webglContrastRef.current = new WebGLContrast();
        }
        currentCanvas = webglContrastRef.current.apply(
          currentCanvas,
          params.contrast,
        );
      }

      // Apply hue and saturation
      if (
        (params.hue && params.hue !== 0) ||
        (params.saturation && params.saturation !== 0)
      ) {
        hasAnyFilter = true;
        if (!webglHueSaturationRef.current) {
          webglHueSaturationRef.current = new WebGLHueSaturation();
        }
        currentCanvas = webglHueSaturationRef.current.apply(
          currentCanvas,
          params.hue ?? 0,
          params.saturation ?? 0,
        );
      }

      // Apply blur
      if (params.blur && params.blur > 0) {
        hasAnyFilter = true;
        if (!webglBlurRef.current) {
          webglBlurRef.current = new WebGLBlur();
        }
        const blurRadius = (params.blur / 100) * 10; // Map 0-100 to 0-10 radius
        currentCanvas = webglBlurRef.current.apply(currentCanvas, blurRadius);
      }

      // Apply sharpness
      if (params.sharpness && params.sharpness > 0) {
        hasAnyFilter = true;
        if (!webglSharpnessRef.current) {
          webglSharpnessRef.current = new WebGLSharpness();
        }
        currentCanvas = webglSharpnessRef.current.apply(
          currentCanvas,
          params.sharpness,
        );
      }

      // Apply noise
      if (params.noise && params.noise > 0) {
        hasAnyFilter = true;
        if (!webglNoiseRef.current) {
          webglNoiseRef.current = new WebGLNoise();
        }
        currentCanvas = webglNoiseRef.current.apply(currentCanvas, params.noise);
      }

      // Apply vignette
      if (params.vignette && params.vignette > 0) {
        hasAnyFilter = true;
        if (!webglVignetteRef.current) {
          webglVignetteRef.current = new WebGLVignette();
        }
        currentCanvas = webglVignetteRef.current.apply(
          currentCanvas,
          params.vignette,
        );
      }

      // Apply color tint
      if (
        params.colorTintIntensity &&
        params.colorTintIntensity > 0 &&
        params.colorTintColor
      ) {
        hasAnyFilter = true;
        if (!webglColorTintRef.current) {
          webglColorTintRef.current = new WebGLColorTint();
        }
        // Parse hex color to RGB components (0-1)
        const hex = params.colorTintColor.replace("#", "");
        const r = parseInt(hex.substring(0, 2), 16) / 255;
        const g = parseInt(hex.substring(2, 4), 16) / 255;
        const b = parseInt(hex.substring(4, 6), 16) / 255;
        currentCanvas = webglColorTintRef.current.apply(
          currentCanvas,
          r,
          g,
          b,
          params.colorTintIntensity,
        );
      }

      // Apply scan lines
      if (params.scanLines && params.scanLines > 0) {
        hasAnyFilter = true;
        if (!webglScanLinesRef.current) {
          webglScanLinesRef.current = new WebGLScanLines();
        }
        currentCanvas = webglScanLinesRef.current.apply(
          currentCanvas,
          params.scanLines,
        );
      }

      // Apply chromatic aberration
      if (params.chromaticAberration && params.chromaticAberration > 0) {
        hasAnyFilter = true;
        if (!webglChromaticAberrationRef.current) {
          webglChromaticAberrationRef.current = new WebGLChromaticAberration();
        }
        currentCanvas = webglChromaticAberrationRef.current.apply(
          currentCanvas,
          params.chromaticAberration,
        );
      }

      // Apply interlace
      if (params.interlace && params.interlace > 0) {
        hasAnyFilter = true;
        if (!webglInterlaceRef.current) {
          webglInterlaceRef.current = new WebGLInterlace();
        }
        currentCanvas = webglInterlaceRef.current.apply(
          currentCanvas,
          params.interlace,
        );
      }

      // Apply pixelate
      if (params.pixelate && params.pixelate > 0) {
        hasAnyFilter = true;
        if (!webglPixelateRef.current) {
          webglPixelateRef.current = new WebGLPixelate();
        }
        currentCanvas = webglPixelateRef.current.apply(
          currentCanvas,
          params.pixelate,
        );
      }

      // Apply jitter
      if (params.jitter && params.jitter > 0) {
        hasAnyFilter = true;
        if (!webglJitterRef.current) {
          webglJitterRef.current = new WebGLJitter();
        }
        currentCanvas = webglJitterRef.current.apply(
          currentCanvas,
          params.jitter,
        );
      }

      if (!hasAnyFilter) {
        return sourceCanvas;
      }

      if (options?.output === "passthrough") {
        return currentCanvas;
      }

      if (currentCanvas !== sourceCanvas) {
        const ctx = sourceCanvas.getContext("2d");
        if (!ctx) return currentCanvas;
        if (
          sourceCanvas.width !== currentCanvas.width ||
          sourceCanvas.height !== currentCanvas.height
        ) {
          sourceCanvas.width = currentCanvas.width;
          sourceCanvas.height = currentCanvas.height;
        }
        ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        ctx.imageSmoothingEnabled = true;
        // @ts-ignore
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(currentCanvas, 0, 0, sourceCanvas.width, sourceCanvas.height);
      }

      return sourceCanvas;
    },
    [],
  );

  /**
   * Check if any filters are active
   */
  const hasActiveFilters = useCallback((params: FilterParams): boolean => {
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
  }, []);

  /**
   * Cleanup WebGL resources on unmount
   */
  useEffect(() => {
    return () => {
      if (webglBlurRef.current) {
        webglBlurRef.current.dispose();
        webglBlurRef.current = null;
      }
      if (webglBrightnessRef.current) {
        webglBrightnessRef.current.dispose();
        webglBrightnessRef.current = null;
      }
      if (webglChromaticAberrationRef.current) {
        webglChromaticAberrationRef.current.dispose();
        webglChromaticAberrationRef.current = null;
      }
      if (webglColorTintRef.current) {
        webglColorTintRef.current.dispose();
        webglColorTintRef.current = null;
      }
      if (webglContrastRef.current) {
        webglContrastRef.current.dispose();
        webglContrastRef.current = null;
      }
      if (webglHueSaturationRef.current) {
        webglHueSaturationRef.current.dispose();
        webglHueSaturationRef.current = null;
      }
      if (webglInterlaceRef.current) {
        webglInterlaceRef.current.dispose();
        webglInterlaceRef.current = null;
      }
      if (webglJitterRef.current) {
        webglJitterRef.current.dispose();
        webglJitterRef.current = null;
      }
      if (webglNoiseRef.current) {
        webglNoiseRef.current.dispose();
        webglNoiseRef.current = null;
      }
      if (webglPixelateRef.current) {
        webglPixelateRef.current.dispose();
        webglPixelateRef.current = null;
      }
      if (webglScanLinesRef.current) {
        webglScanLinesRef.current.dispose();
        webglScanLinesRef.current = null;
      }
      if (webglSharpnessRef.current) {
        webglSharpnessRef.current.dispose();
        webglSharpnessRef.current = null;
      }
      if (webglVignetteRef.current) {
        webglVignetteRef.current.dispose();
        webglVignetteRef.current = null;
      }
    };
  }, []);

  return {
    applyFilters,
    hasActiveFilters,
  };
}
