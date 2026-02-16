import { AnyClipProps, MaskClipProps } from "@/lib/types";
import { useCallback, useEffect, useRef } from "react";
import { getLocalFrame } from "@/lib/clip";
import { getSharedMaskEngines } from "./sharedMaskEngines";

interface WebGLMaskProps {
  focusFrame: number;
  masks: MaskClipProps[];
  disabled: boolean;
  debug?: { download?: boolean; annotateBounds?: boolean; filename?: string };
  clip?: AnyClipProps;
  useOriginalTransform?: boolean;
}

export function useWebGLMask({
  focusFrame,
  masks,
  disabled,
  debug,
  clip,
  useOriginalTransform = true,
}: WebGLMaskProps) {
  // NOTE: Do NOT create a unique WebGL context per clip/preview.
  // Large timelines will exceed Chromium's WebGL context limit and cause context loss.
  const sharedContextKeyRef = useRef<string>("preview-webgl-mask-shared");

  const maskWorkingCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Cleanup on unmount (intentionally does NOT dispose shared WebGL engines)
  useEffect(() => {
    return () => {
      maskWorkingCanvasRef.current = null;
    };
  }, []);

  const applyMask = useCallback(
    (sourceCanvas: HTMLCanvasElement, frame?: number) => {
      if (disabled || masks.length === 0) {
        return sourceCanvas;
      }

      const { shape: shapeMask, lasso: lassoMask, touch: touchMask } =
        getSharedMaskEngines(sharedContextKeyRef.current);

      if (frame === undefined) {
        if (clip) {
          frame = getLocalFrame(focusFrame, clip);
        } else {
          frame = focusFrame;
        }
      }

      let workingCanvas = maskWorkingCanvasRef.current;
      if (!workingCanvas) {
        workingCanvas = document.createElement("canvas");
        maskWorkingCanvasRef.current = workingCanvas;
      }
      if (
        workingCanvas.width !== sourceCanvas.width ||
        workingCanvas.height !== sourceCanvas.height
      ) {
        workingCanvas.width = sourceCanvas.width;
        workingCanvas.height = sourceCanvas.height;
      }
      const workingCtx = workingCanvas.getContext("2d");
      if (!workingCtx) {
        return sourceCanvas;
      }
      workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
      workingCtx.drawImage(sourceCanvas, 0, 0);

      let currentCanvas: HTMLCanvasElement = workingCanvas;

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
            debug,
          );
        } else if (mask.tool === "lasso") {
          nextCanvas = lassoMask.apply(
            currentCanvas,
            effectiveMask,
            frame,
            clip?.transform,
            useOriginalTransform ? clip?.originalTransform : undefined,
            baseTransform,
            debug,
          );
        } else if (mask.tool === "touch") {
          nextCanvas = touchMask.apply(
            currentCanvas,
            effectiveMask,
            frame,
            clip?.transform,
            useOriginalTransform ? clip?.originalTransform : undefined,
            baseTransform,
            debug,
          );
        }

        if (nextCanvas && nextCanvas !== currentCanvas) {
          currentCanvas = nextCanvas;
        }
      }

      return currentCanvas;
    },
    [focusFrame, masks, disabled, debug, clip, useOriginalTransform],
  );

  return { applyMask };
}
