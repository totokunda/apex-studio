import { FilterClipProps } from "@/lib/types";
import { useControlsStore } from "@/lib/control";

/**
 * Base class for all clip preview applications
 * Provides common functionality for applying effects to canvases with chaining support
 *
 * @template T - The specific clip props type (VideoClipProps, FilterClipProps, etc.)
 *
 * @example
 * class MyEffectApplicator extends BaseClipApplicator<VideoClipProps> {
 *   apply(canvas: HTMLCanvasElement, param: number): HTMLCanvasElement {
 *     if (!this.isInFrameRange()) return canvas;
 *     // Apply effect logic...
 *     return canvas;
 *   }
 * }
 */
export abstract class BaseClipApplicator<
  T extends FilterClipProps = FilterClipProps,
> {
  protected clip: T;
  private focusFrameOverride?: number;

  constructor(clip: T) {
    this.clip = clip;
  }

  /**
   * Abstract method that subclasses must implement to apply their specific effect
   * All implementations should return an HTMLCanvasElement to support chaining
   * @param args - Variable arguments depending on the specific applicator
   * @returns HTMLCanvasElement - The processed canvas for chaining
   */
  abstract apply(...args: any[]): HTMLCanvasElement;

  /**
   * Checks if the current frame is within the clip's range
   * @returns boolean - True if the current frame is within the clip's range
   */
  protected isInFrameRange(): boolean {
    const focusFrame =
      typeof this.focusFrameOverride === "number"
        ? this.focusFrameOverride
        : useControlsStore.getState().focusFrame;
    const startFrame = this.clip.startFrame ?? 0;
    const endFrame = this.clip.endFrame ?? 0;
    return focusFrame >= startFrame && focusFrame <= endFrame;
  }

  /**
   * Get the clip instance
   * @returns The current clip props
   */
  getClip(): T {
    return this.clip;
  }

  getIntensity(): number {
    return this.clip.intensity ?? 1;
  }

  /**
   * Update clip properties and return this for chaining
   * @param updates - Partial clip props to merge
   * @returns this - Returns the instance for method chaining
   */
  updateClip(updates: Partial<T>): this {
    this.clip = { ...this.clip, ...updates };
    return this;
  }

  /**
   * Get the start frame of the clip
   * @returns number - The start frame
   */
  getStartFrame(): number {
    return this.clip.startFrame ?? 0;
  }

  /**
   * Get the end frame of the clip
   * @returns number - The end frame
   */
  getEndFrame(): number {
    return this.clip.endFrame ?? 0;
  }

  /**
   * Override the focus frame used for in-range checks (e.g., input-mode playback)
   */
  setFocusFrameOverride(frame?: number): this {
    this.focusFrameOverride = frame;
    return this;
  }

  /**
   * Get the clip duration in frames
   * @returns number - The duration in frames
   */
  getDuration(): number {
    return this.getEndFrame() - this.getStartFrame();
  }
}
