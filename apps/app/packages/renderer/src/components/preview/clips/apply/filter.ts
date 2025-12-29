import { FilterClipProps } from "@/lib/types";
import { WebGLHaldClut } from "@/components/preview/webgl-filters/hald-clut";
import { BaseClipApplicator } from "./base";

/**
 * FilterPreview class for applying Hald CLUT filters to canvases
 * Supports chaining by returning the modified canvas from apply()
 * Assumes the CLUT has already been preloaded into the haldClutInstance
 *
 * @example
 * const filterApplicator = new FilterPreview(filterClip);
 * const canvas = filterApplicator.apply(inputCanvas, haldClutInstance, 1.0);
 * // Chain multiple filters:
 * const filter1 = new FilterPreview(filterClip1);
 * const filter2 = new FilterPreview(filterClip2);
 * let result = filter1.apply(canvas, haldClutInstance);
 * result = filter2.apply(result, haldClutInstance);
 */
export class FilterPreview extends BaseClipApplicator<FilterClipProps> {
  private strength: number = 1.0;
  private haldClutInstance: WebGLHaldClut | null = null;
  constructor(clip: FilterClipProps) {
    super(clip);
  }

  setHaldClutInstance(haldClutInstance: WebGLHaldClut | null) {
    this.haldClutInstance = haldClutInstance;
  }

  setStrength(strength: number) {
    this.strength = strength;
  }

  /**
   * Returns true if required CLUT resources are already available on GPU
   */
  isReady(): boolean {
    if (!this.haldClutInstance) return false;
    const filterPath = this.clip.fullPath || this.clip.smallPath;
    if (!filterPath) return true;
    try {
      return this.haldClutInstance.isClutLoaded(filterPath);
    } catch {
      return false;
    }
  }

  /**
   * Ensure CLUT resources are preloaded before applying the filter
   */
  async ensureResources(): Promise<void> {
    if (!this.haldClutInstance) return;
    const filterPath = this.clip.fullPath || this.clip.smallPath;
    if (!filterPath) return;
    if (this.haldClutInstance.isClutLoaded(filterPath)) return;
    try {
      await this.haldClutInstance.preloadClut(filterPath);
    } catch (e) {
      console.warn("[FilterPreview] Failed to preload CLUT", filterPath, e);
    }
  }

  /**
   * Applies the filter to the provided canvas if the current frame is within the clip's range
   * NOTE: Assumes the CLUT has already been loaded. Call haldClutInstance.loadClut() before using this.
   * @param canvas - The canvas to apply the filter to
   * @param haldClutInstance - The shared WebGLHaldClut instance with preloaded CLUT
   * @param strength - Filter strength/intensity (0-1, default: 1.0)
   * @returns HTMLCanvasElement - The canvas with filter applied (or original if not in range)
   */
  apply(canvas: HTMLCanvasElement): HTMLCanvasElement {
    // Validate clip type
    if (!this.clip || this.clip.type !== "filter") {
      console.warn("[FilterPreview] Invalid filter clip");
      return canvas;
    }

    // Check if the current frame is within the filter clip's range
    if (!this.isInFrameRange()) {
      return canvas;
    }

    // If no Hald CLUT instance available, return unchanged
    if (!this.haldClutInstance) {
      console.warn("[FilterPreview] No WebGLHaldClut instance available");
      return canvas;
    }

    // Get the filter path (use fullPath for highest quality, matching preload order)
    const filterPath = this.clip.fullPath || this.clip.smallPath;

    if (!filterPath) {
      console.warn("[FilterPreview] Filter clip has no filter path");
      return canvas;
    }

    try {
      // Apply the filter with the specified strength (assumes CLUT is already loaded)

      const filteredCanvas = this.haldClutInstance.apply(
        canvas,
        filterPath,
        this.strength,
      );

      // Copy the filtered result back to the original canvas to maintain reference
      const ctx = canvas.getContext("2d");
      if (ctx && filteredCanvas !== canvas) {
        // Ensure canvas dimensions match
        if (
          canvas.width !== filteredCanvas.width ||
          canvas.height !== filteredCanvas.height
        ) {
          canvas.width = filteredCanvas.width;
          canvas.height = filteredCanvas.height;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(filteredCanvas, 0, 0);
      }

      return canvas;
    } catch (error) {
      console.error("[FilterPreview] Error applying filter:", error);
      return canvas;
    }
  }

  /**
   * Gets the filter path for this clip
   * @param useFullPath - If true, returns fullPath; otherwise returns smallPath
   * @returns The filter path or null if not available
   */
  getFilterPath(useFullPath: boolean = true): string | null {
    const path = useFullPath
      ? this.clip.fullPath || this.clip.smallPath
      : this.clip.smallPath || this.clip.fullPath;
    return path ?? null;
  }
}
