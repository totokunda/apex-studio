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

      return masks.reduce((acc, mask, index) => {
        const effectiveMask =
          index === 0 ? mask : { ...mask, backgroundColorEnabled: false };
        const baseTransform =
          mask.transform ?? clip?.originalTransform ?? clip?.transform;
        if (mask.tool === "shape") {
          // For subsequent masks, disable background so we compose masks instead of overwriting

          // Apply clipTransform to shapeBounds
          const maskedCanvas = shapeMask.apply(
            acc,
            effectiveMask,
            frame,
            clip?.transform,
            useOriginalTransform ? clip?.originalTransform : undefined,
            baseTransform,
            debug,
          );
          // If maskedCanvas is the same as acc, it means WebGL failed and returned source unchanged
          if (maskedCanvas === acc) {
            return acc;
          }

          const ctx = acc.getContext("2d");
          if (!ctx) return acc;

          ctx.clearRect(0, 0, acc.width, acc.height);
          ctx.drawImage(maskedCanvas, 0, 0);

          return acc;
        } else if (mask.tool === "lasso") {
          const maskedCanvas = lassoMask.apply(
            acc,
            effectiveMask,
            frame,
            clip?.transform,
            useOriginalTransform ? clip?.originalTransform : undefined,
            baseTransform,
            debug,
          );
          if (maskedCanvas === acc) {
            return acc;
          }
          const ctx = acc.getContext("2d");
          if (!ctx) return acc;

          ctx.clearRect(0, 0, acc.width, acc.height);
          ctx.drawImage(maskedCanvas, 0, 0);

          return acc;
        } else if (mask.tool === "touch") {
          const maskedCanvas = touchMask.apply(
            acc,
            effectiveMask,
            frame,
            clip?.transform,
            useOriginalTransform ? clip?.originalTransform : undefined,
            baseTransform,
            debug,
          );
          if (maskedCanvas === acc) {
            return acc;
          }
          const ctx = acc.getContext("2d");
          if (!ctx) return acc;
          ctx.clearRect(0, 0, acc.width, acc.height);
          ctx.drawImage(maskedCanvas, 0, 0);
          return acc;
        }
        return acc;
      }, workingCanvas);
    },
    [focusFrame, masks, disabled, debug, clip, useOriginalTransform],
  );

  return { applyMask };
}
