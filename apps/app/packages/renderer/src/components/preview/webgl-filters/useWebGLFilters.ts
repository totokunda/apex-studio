/**
 * Custom hook for managing and applying WebGL filters
 * Provides a reusable interface for applying GPU-accelerated filters to canvas elements
 */

import { useRef, useCallback, useEffect } from "react";
import {
  WebGLBlur,
  WebGLBrightness,
  WebGLContrast,
  WebGLHueSaturation,
  WebGLNoise,
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
}

export function useWebGLFilters() {
  const webglBlurRef = useRef<WebGLBlur | null>(null);
  const webglBrightnessRef = useRef<WebGLBrightness | null>(null);
  const webglContrastRef = useRef<WebGLContrast | null>(null);
  const webglHueSaturationRef = useRef<WebGLHueSaturation | null>(null);
  const webglNoiseRef = useRef<WebGLNoise | null>(null);
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
    ): HTMLCanvasElement => {
      const ctx = sourceCanvas.getContext("2d");
      if (!ctx) return sourceCanvas;

      let currentCanvas: HTMLCanvasElement = sourceCanvas;

      // Apply brightness
      if (params.brightness && params.brightness !== 0) {
        if (!webglBrightnessRef.current) {
          webglBrightnessRef.current = new WebGLBrightness();
        }
        const brightnessValue = params.brightness / 100; // Map -100..100 to -1..1
        const result = webglBrightnessRef.current.apply(
          currentCanvas,
          brightnessValue,
        );

        // Copy result back to source canvas with flip
        ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        ctx.save();
        ctx.scale(1, -1);
        ctx.drawImage(result, 0, -sourceCanvas.height);
        ctx.restore();
        currentCanvas = sourceCanvas;
      }

      // Apply contrast
      if (params.contrast && params.contrast !== 0) {
        if (!webglContrastRef.current) {
          webglContrastRef.current = new WebGLContrast();
        }
        const result = webglContrastRef.current.apply(
          currentCanvas,
          params.contrast,
        );

        // Copy result back to source canvas with flip
        ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        ctx.save();
        ctx.scale(1, -1);
        ctx.drawImage(result, 0, -sourceCanvas.height);
        ctx.restore();
        currentCanvas = sourceCanvas;
      }

      // Apply hue and saturation
      if (
        (params.hue && params.hue !== 0) ||
        (params.saturation && params.saturation !== 0)
      ) {
        if (!webglHueSaturationRef.current) {
          webglHueSaturationRef.current = new WebGLHueSaturation();
        }
        const result = webglHueSaturationRef.current.apply(
          currentCanvas,
          params.hue ?? 0,
          params.saturation ?? 0,
        );

        // Copy result back to source canvas with flip
        ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        ctx.save();
        ctx.scale(1, -1);
        ctx.drawImage(result, 0, -sourceCanvas.height);
        ctx.restore();
        currentCanvas = sourceCanvas;
      }

      // Apply blur
      if (params.blur && params.blur > 0) {
        if (!webglBlurRef.current) {
          webglBlurRef.current = new WebGLBlur();
        }
        const blurRadius = (params.blur / 100) * 10; // Map 0-100 to 0-10 radius
        const result = webglBlurRef.current.apply(currentCanvas, blurRadius);

        // Copy result back to source canvas with flip
        ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        ctx.save();
        ctx.scale(1, -1);
        ctx.drawImage(result, 0, -sourceCanvas.height);
        ctx.restore();
        currentCanvas = sourceCanvas;
      }

      // Apply sharpness
      if (params.sharpness && params.sharpness > 0) {
        if (!webglSharpnessRef.current) {
          webglSharpnessRef.current = new WebGLSharpness();
        }
        const result = webglSharpnessRef.current.apply(
          currentCanvas,
          params.sharpness,
        );

        // Copy result back to source canvas with flip
        ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        ctx.save();
        ctx.scale(1, -1);
        ctx.drawImage(result, 0, -sourceCanvas.height);
        ctx.restore();
        currentCanvas = sourceCanvas;
      }

      // Apply noise
      if (params.noise && params.noise > 0) {
        if (!webglNoiseRef.current) {
          webglNoiseRef.current = new WebGLNoise();
        }
        const result = webglNoiseRef.current.apply(currentCanvas, params.noise);

        // Copy result back to source canvas with flip
        ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        ctx.save();
        ctx.scale(1, -1);
        ctx.drawImage(result, 0, -sourceCanvas.height);
        ctx.restore();
        currentCanvas = sourceCanvas;
      }

      // Apply vignette
      if (params.vignette && params.vignette > 0) {
        if (!webglVignetteRef.current) {
          webglVignetteRef.current = new WebGLVignette();
        }
        const result = webglVignetteRef.current.apply(
          currentCanvas,
          params.vignette,
        );

        // Copy result back to source canvas with flip
        ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        ctx.save();
        ctx.scale(1, -1);
        ctx.drawImage(result, 0, -sourceCanvas.height);
        ctx.restore();
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
      (params.vignette !== undefined && params.vignette > 0)
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
      if (webglContrastRef.current) {
        webglContrastRef.current.dispose();
        webglContrastRef.current = null;
      }
      if (webglHueSaturationRef.current) {
        webglHueSaturationRef.current.dispose();
        webglHueSaturationRef.current = null;
      }
      if (webglNoiseRef.current) {
        webglNoiseRef.current.dispose();
        webglNoiseRef.current = null;
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
