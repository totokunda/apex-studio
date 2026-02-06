import { getSharedMaskEngines } from "../../../renderer/src/components/preview/mask/sharedMaskEngines";

type MaskTool = "lasso" | "shape" | "draw" | "touch";

interface ClipTransform {
  x: number;
  y: number;
  width: number;
  height: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  cornerRadius: number;
  opacity: number;
}

export interface MaskData {
  lassoPoints?: number[];
  shapeBounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation?: number;
    scaleX?: number;
    scaleY?: number;
    shapeType?: "rectangle" | "ellipse" | "polygon" | "star";
  };
  contours?: number[][];
}

export interface MaskClipProps {
  id: string;
  tool: MaskTool;
  keyframes: Map<number, MaskData> | Record<number, MaskData>;
  transform?: ClipTransform;
  backgroundColor?: string;
  backgroundOpacity?: number;
  backgroundColorEnabled?: boolean;
  maskColor?: string;
  maskOpacity?: number;
  maskColorEnabled?: boolean;
}

export interface AnyClipLike {
  clipId: string;
  type: "video" | "image" | string;
  startFrame?: number;
  endFrame?: number;
  trimStart?: number;
  transform?: ClipTransform;
  originalTransform?: ClipTransform;
}

export function applyMasksToCanvas(
  sourceCanvas: HTMLCanvasElement,
  opts: {
    focusFrame: number;
    masks: MaskClipProps[];
    clip?: AnyClipLike | null;
    disabled?: boolean;
    debug?: { download?: boolean; annotateBounds?: boolean; filename?: string };
    maskFrameOverride?: number;
  },
): void {
  const { focusFrame, masks, clip, disabled, debug, maskFrameOverride } = opts;
  if (disabled || !masks || masks.length === 0) {
    return;
  }

  // Resolve frame: images use frame 0; videos use local frame relative to clip start/trim
  const getLocalFrame = (globalFrame: number, c: AnyClipLike): number => {
    const startFrame = c.startFrame ?? 0;
    const trimStart = isFinite(c.trimStart ?? 0) ? (c.trimStart ?? 0) : 0;
    const realStartFrame = startFrame + trimStart;
    return globalFrame - realStartFrame;
  };

  let frame =
    typeof maskFrameOverride === "number"
      ? Math.max(0, Math.floor(maskFrameOverride))
      : focusFrame;
  if (clip && typeof maskFrameOverride !== "number") {
    frame =
      clip.type === "video"
        ? Math.max(0, Math.round(getLocalFrame(focusFrame, clip)))
        : 0;
  }

  // Working canvas (offscreen) to compose masks
  const workingCanvas = document.createElement("canvas");
  workingCanvas.width = sourceCanvas.width;
  workingCanvas.height = sourceCanvas.height;
  const workingCtx = workingCanvas.getContext("2d");
  if (!workingCtx) {
    return;
  }

  workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
  workingCtx.drawImage(sourceCanvas, 0, 0);

  // Use a shared WebGL context/engines to avoid Chromium WebGL context limits.
  // Keep export isolated from preview by using a separate context key.
  const { shape: shapeMask, lasso: lassoMask, touch: touchMask } =
    getSharedMaskEngines("export-webgl-mask-shared");

  {
    let acc: HTMLCanvasElement = workingCanvas;
    for (let i = 0; i < masks.length; i++) {
      const mask = masks[i];
      // For subsequent masks, disable background so we compose masks instead of overwriting
      const effectiveMask: MaskClipProps =
        i === 0 ? mask : { ...mask, backgroundColorEnabled: false };
      const baseTransform =
        mask.transform ?? clip?.originalTransform ?? clip?.transform;

      let maskedCanvas: HTMLCanvasElement = acc;
      if (effectiveMask.tool === "shape") {
        maskedCanvas = shapeMask.apply(
          acc,
          effectiveMask as any,
          frame,
          clip?.transform as any,
          clip?.originalTransform as any,
          baseTransform as any,
          debug,
        );
      } else if (effectiveMask.tool === "lasso") {
        maskedCanvas = lassoMask.apply(
          acc,
          effectiveMask as any,
          frame,
          clip?.transform as any,
          clip?.originalTransform as any,
          baseTransform as any,
          debug,
        );
      } else if (effectiveMask.tool === "touch") {
        maskedCanvas = touchMask.apply(
          acc,
          effectiveMask as any,
          frame,
          clip?.transform as any,
          clip?.originalTransform as any,
          baseTransform as any,
          debug,
        );
      }

      if (maskedCanvas !== acc) {
        const ctx = acc.getContext("2d");
        if (!ctx) continue;
        ctx.clearRect(0, 0, acc.width, acc.height);
        ctx.drawImage(maskedCanvas, 0, 0);
      }
    }

    // Copy back to the source canvas (mutate input)
    const srcCtx = sourceCanvas.getContext("2d");
    if (srcCtx) {
      srcCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
      srcCtx.drawImage(workingCanvas, 0, 0);
    }
  }
}
