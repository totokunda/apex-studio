/**
 * WebGL Compositor - Unified pipeline: masks → blur → combined effects → HALD CLUT(s)
 * Single call with all inputs, one canvas out.
 */

import { WebGLFilterBase } from "./WebGLFilterBase";
import { WebGLBlur } from "./blur";
import type { WebGLHaldClut } from "./hald-clut";
import type { FilterParams } from "./useWebGLFilters";
import { getSharedMaskEngines } from "../mask/sharedMaskEngines";
import { getLocalFrame } from "@/lib/clip";
import type { AnyClipProps, MaskClipProps } from "@/lib/types";

import fragmentShaderBase from "./combined-effects.frag.glsl?raw";

const vertexShader = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;

  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

const ALL_EFFECT_DEFINES = [
  "APPLY_BRIGHTNESS",
  "APPLY_CONTRAST",
  "APPLY_HUE_SATURATION",
  "APPLY_COLOR_TINT",
  "APPLY_CHROMATIC_ABERRATION",
  "APPLY_JITTER",
  "APPLY_PIXELATE",
  "APPLY_NOISE",
  "APPLY_INTERLACE",
  "APPLY_SCAN_LINES",
  "APPLY_SHARPNESS",
  "APPLY_VIGNETTE",
];

function buildFragmentShader(): string {
  const defines = ALL_EFFECT_DEFINES.map((d) => `#define ${d}`).join("\n");
  return `${defines}\n${fragmentShaderBase}`;
}

export interface ClutEntry {
  /** HALD CLUT image path (must be preloaded via haldClutInstance.preloadClut) */
  path: string;
  /** Blend intensity 0–1 (default 1.0) */
  intensity?: number;
}

export interface CompositorOptions {
  /** Filter parameters (brightness, contrast, etc.) */
  filterParams: FilterParams;
  /** Masks to apply (shape, lasso, touch) before effects */
  masks?: MaskClipProps[];
  /** Frame index for mask keyframes (uses focusFrame if clip provided) */
  maskFrame?: number;
  /** Clip for transform context and getLocalFrame when maskFrame omitted */
  clip?: AnyClipProps;
  /** Use clip.originalTransform for mask positioning (default true) */
  useOriginalTransform?: boolean;
  /** Focus frame when deriving maskFrame from clip */
  focusFrame?: number;
  /** HALD CLUT(s) to apply after effects. Requires haldClutInstance. */
  cluts?: ClutEntry[];
  /** Shared WebGLHaldClut instance (e.g. from useWebGLHaldClut()). Required when using cluts. */
  haldClutInstance?: WebGLHaldClut | null;
}

const DEFAULT_MASK_CONTEXT_KEY = "preview-webgl-mask-shared";

export class CompositorShader extends WebGLFilterBase {
  private program: WebGLProgram | null = null;
  private blurFilter: WebGLBlur | null = null;
  private interlaceSeed = 0;
  private noiseSeed = 0;

  constructor() {
    super();
    this.initProgram();
  }

  private initProgram() {
    this.program = this.createProgram(vertexShader, buildFragmentShader());
  }

  protected onContextLost(): void {
    super.onContextLost();
    this.program = null;
    if (this.blurFilter) {
      this.blurFilter.dispose();
      this.blurFilter = null;
    }
  }

  protected onContextRestored(): void {
    super.onContextRestored();
    this.initProgram();
  }

  private hasAnyEffect(params: FilterParams): boolean {
    return (
      (params.brightness !== undefined && params.brightness !== 0) ||
      (params.contrast !== undefined && params.contrast !== 0) ||
      (params.hue !== undefined && params.hue !== 0) ||
      (params.saturation !== undefined && params.saturation !== 0) ||
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
      (params.jitter !== undefined && params.jitter > 0) ||
      (params.blur !== undefined && params.blur > 0)
    );
  }

  private applyMasks(
    sourceCanvas: HTMLCanvasElement,
    masks: MaskClipProps[],
    frame: number,
    clip?: AnyClipProps,
    useOriginalTransform = true,
  ): HTMLCanvasElement {
    if (masks.length === 0) return sourceCanvas;

    const { shape: shapeMask, lasso: lassoMask, touch: touchMask } =
      getSharedMaskEngines(DEFAULT_MASK_CONTEXT_KEY);

    let currentCanvas: HTMLCanvasElement = sourceCanvas;

    for (let index = 0; index < masks.length; index += 1) {
      const mask = masks[index];
      const effectiveMask =
        index === 0 ? mask : { ...mask, backgroundColorEnabled: false };
      const baseTransform =
        mask.transform ?? clip?.originalTransform ?? clip?.transform;
      let nextCanvas = currentCanvas;

      if (mask.tool === "shape") {
        nextCanvas = shapeMask.apply(
          currentCanvas,
          effectiveMask,
          frame,
          clip?.transform,
          useOriginalTransform ? clip?.originalTransform : undefined,
          baseTransform,
        );
      } else if (mask.tool === "lasso") {
        nextCanvas = lassoMask.apply(
          currentCanvas,
          effectiveMask,
          frame,
          clip?.transform,
          useOriginalTransform ? clip?.originalTransform : undefined,
          baseTransform,
        );
      } else if (mask.tool === "touch") {
        nextCanvas = touchMask.apply(
          currentCanvas,
          effectiveMask,
          frame,
          clip?.transform,
          useOriginalTransform ? clip?.originalTransform : undefined,
          baseTransform,
        );
      }

      if (nextCanvas && nextCanvas !== currentCanvas) {
        currentCanvas = nextCanvas;
      }
    }

    return currentCanvas;
  }

  /**
   * Apply the full pipeline: masks → blur → combined effects.
   * Pass everything in, get one canvas out.
   *
   * Can be called with (sourceCanvas, options) or (sourceCanvas, filterParams) for filters-only.
   */
  public apply(
    sourceCanvas: HTMLCanvasElement,
    optionsOrParams: CompositorOptions | FilterParams,
  ): HTMLCanvasElement {
    const options: CompositorOptions =
      "filterParams" in optionsOrParams
        ? optionsOrParams
        : { filterParams: optionsOrParams };

    const {
      filterParams: params,
      masks = [],
      maskFrame: providedFrame,
      clip,
      useOriginalTransform = true,
      focusFrame = 0,
      cluts = [],
      haldClutInstance,
    } = options;

    const hasMasks = masks.length > 0;
    const hasEffects = this.hasAnyEffect(params);
    const hasCluts =
      cluts.length > 0 &&
      !!haldClutInstance &&
      cluts.some((c) => (c.intensity ?? 1) > 0);

    if (!hasMasks && !hasEffects && !hasCluts) {
      return sourceCanvas;
    }

    let currentCanvas: HTMLCanvasElement = sourceCanvas;

    

    if (hasEffects) {
      // 2a. Blur is two-pass separable - run before combined effects
      if (params.blur && params.blur > 0) {
      if (!this.blurFilter) {
        this.blurFilter = new WebGLBlur();
      }
        const blurRadius = (params.blur / 100) * 10;
        currentCanvas = this.blurFilter.apply(currentCanvas, blurRadius);
      }

      // 2b. Combined effects pass
      const hasCombinedEffects =
        (params.brightness !== undefined && params.brightness !== 0) ||
        (params.contrast !== undefined && params.contrast !== 0) ||
        (params.hue !== undefined && params.hue !== 0) ||
        (params.saturation !== undefined && params.saturation !== 0) ||
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
        (params.jitter !== undefined && params.jitter > 0);

      if (hasCombinedEffects) {
        const gl = this.ensureContext();
        if (!gl || !this.program) {
          return currentCanvas;
        }

        this.resizeCanvas(currentCanvas.width, currentCanvas.height);

        const texture = this.createTextureFromCanvas(currentCanvas);
        if (!texture) return currentCanvas;

        gl.useProgram(this.program);
        this.setupAttributes(this.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
        gl.uniform2f(
          gl.getUniformLocation(this.program, "u_resolution"),
          this.canvas.width,
          this.canvas.height,
        );

        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_brightness"),
          (params.brightness ?? 0) / 100,
        );
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_contrast"),
          (params.contrast ?? 0) / 100,
        );
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_hue"),
          ((params.hue ?? 0) * 3.6) / 360,
        );
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_saturation"),
          (params.saturation ?? 0) / 100,
        );

        let r = 1,
          g = 1,
          b = 1;
        if (params.colorTintColor) {
          const hex = params.colorTintColor.replace("#", "");
          r = parseInt(hex.substring(0, 2), 16) / 255;
          g = parseInt(hex.substring(2, 4), 16) / 255;
          b = parseInt(hex.substring(4, 6), 16) / 255;
        }
        gl.uniform3f(
          gl.getUniformLocation(this.program, "u_tintColor"),
          r,
          g,
          b,
        );
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_tintIntensity"),
          (params.colorTintIntensity ?? 0) / 100,
        );

        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_amount"),
          (params.chromaticAberration ?? 0) / 100,
        );

        const maxShift = 0.02;
        const offsetX = (Math.random() - 0.5) * 2 * maxShift;
        const offsetY = (Math.random() - 0.5) * 2 * maxShift;
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_jitterIntensity"),
          (params.jitter ?? 0) / 100,
        );
        gl.uniform2f(
          gl.getUniformLocation(this.program, "u_offset"),
          offsetX,
          offsetY,
        );

        const pixelSize = 1 + ((params.pixelate ?? 0) / 100) * 19;
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_pixelSize"),
          pixelSize,
        );

        this.noiseSeed = Math.random() * 1000;
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_noise"),
          (params.noise ?? 0) / 100,
        );
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_seed"),
          this.noiseSeed,
        );

        this.interlaceSeed = Math.random() * 1000;
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_interlaceIntensity"),
          (params.interlace ?? 0) / 100,
        );
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_interlaceSeed"),
          this.interlaceSeed,
        );

        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_scanLinesIntensity"),
          (params.scanLines ?? 0) / 100,
        );
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_sharpness"),
          (params.sharpness ?? 0) / 100,
        );
        gl.uniform1f(
          gl.getUniformLocation(this.program, "u_vignette"),
          (params.vignette ?? 0) / 100,
        );

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.deleteTexture(texture);

        currentCanvas = this.canvas;
      }
    }


    if (hasMasks) {
      const frame =
        providedFrame ??
        (clip ? getLocalFrame(focusFrame, clip) : focusFrame);
      currentCanvas = this.applyMasks(
        currentCanvas,
        masks,
        frame,
        clip,
        useOriginalTransform,
      );
    }

    // 3. HALD CLUT(s) - final color grading
    if (hasCluts && haldClutInstance) {
      for (const { path, intensity = 1 } of cluts) {
        if (intensity <= 0) continue;
        currentCanvas = haldClutInstance.apply(currentCanvas, path, intensity);
      }
    }

    return currentCanvas;
  }

  public dispose() {
    if (this.blurFilter) {
      this.blurFilter.dispose();
      this.blurFilter = null;
    }
    const gl = this.gl;
    if (gl && this.program) {
      gl.deleteProgram(this.program);
    }
    this.program = null;
    super.dispose();
  }
}
