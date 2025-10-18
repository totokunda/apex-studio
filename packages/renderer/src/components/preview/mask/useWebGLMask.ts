import { AnyClipProps, MaskClipProps } from "@/lib/types";
import { useCallback, useEffect, useRef } from "react";
import { ShapeMask } from "./shape";
import { LassoMask } from "./lasso";
import { TouchMask } from "./touch";
import { getLocalFrame } from "@/lib/clip";

interface WebGLMaskProps {
    focusFrame: number;
    masks: MaskClipProps[];
    disabled: boolean;
    debug?: { download?: boolean; annotateBounds?: boolean; filename?: string };
    clip?: AnyClipProps;
    }

export function useWebGLMask({
    focusFrame,
    masks,
    disabled,
    debug,
    clip,
}: WebGLMaskProps) {
    // create a ref of our shape mask 
    const shapeMaskRef = useRef<ShapeMask | null>(null);
    const lassoMaskRef = useRef<LassoMask | null>(null);
    const touchMaskRef = useRef<TouchMask | null>(null);
    // Lazy initialization of mask to reduce WebGL context count
    if (!shapeMaskRef.current) {
        shapeMaskRef.current = new ShapeMask();
    }
    if (!lassoMaskRef.current) {
        lassoMaskRef.current = new LassoMask();
    }
    if (!touchMaskRef.current) {
        touchMaskRef.current = new TouchMask();
    }
    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (shapeMaskRef.current) {
                shapeMaskRef.current.dispose();
                shapeMaskRef.current = null;
            }
            if (lassoMaskRef.current) {
                lassoMaskRef.current.dispose();
                lassoMaskRef.current = null;
            }
            if (touchMaskRef.current) {
                touchMaskRef.current.dispose();
                touchMaskRef.current = null;
            }
        };
    }, []);
    
    const applyMask = useCallback((sourceCanvas: HTMLCanvasElement, frame?:number) => {
        if (!shapeMaskRef.current) return sourceCanvas;
        if (disabled || masks.length === 0) {
            return sourceCanvas;
        }

        if (frame === undefined) {
            if (clip) {
               frame = getLocalFrame(focusFrame, clip);
            }
            else {
                frame = focusFrame;
            }
        }

        const workingCanvas = sourceCanvas.cloneNode(false) as HTMLCanvasElement;
        workingCanvas.width = sourceCanvas.width;
        workingCanvas.height = sourceCanvas.height;
        const workingCtx = workingCanvas.getContext('2d');
        if (!workingCtx) {
            return sourceCanvas;
        }
        workingCtx.drawImage(sourceCanvas, 0, 0);

        return masks.reduce((acc, mask, index) => {
            const effectiveMask = index === 0
                ? mask
                : { ...mask, backgroundColorEnabled: false };
            const baseTransform = mask.transform ?? clip?.originalTransform ?? clip?.transform;
            if (mask.tool === 'shape' && shapeMaskRef.current) {
                // For subsequent masks, disable background so we compose masks instead of overwriting

                // Apply clipTransform to shapeBounds
                const maskedCanvas = shapeMaskRef.current.apply(acc, effectiveMask, frame, clip?.transform, baseTransform, debug);
                // If maskedCanvas is the same as acc, it means WebGL failed and returned source unchanged
                if (maskedCanvas === acc) {
                    return acc;
                }

                const ctx = acc.getContext('2d');
                if (!ctx) return acc;

                ctx.clearRect(0, 0, acc.width, acc.height);
                ctx.drawImage(maskedCanvas, 0, 0);

                return acc;
            } else if (mask.tool === 'lasso' && lassoMaskRef.current) {
                const maskedCanvas = lassoMaskRef.current.apply(acc, effectiveMask, frame, clip?.transform, baseTransform, debug);
                if (maskedCanvas === acc) {
                    return acc;
                }
                const ctx = acc.getContext('2d');
                if (!ctx) return acc;

                ctx.clearRect(0, 0, acc.width, acc.height);
                ctx.drawImage(maskedCanvas, 0, 0);

                return acc;
            } else if (mask.tool === 'touch' && touchMaskRef.current) {
                const maskedCanvas = touchMaskRef.current.apply(acc, effectiveMask, frame, clip?.transform, baseTransform, debug);
                if (maskedCanvas === acc) {
                    return acc;
                }
                const ctx = acc.getContext('2d');
                if (!ctx) return acc;
                ctx.clearRect(0, 0, acc.width, acc.height);
                ctx.drawImage(maskedCanvas, 0, 0);
                return acc;
            }
            return acc;
        }, workingCanvas);
    }, [focusFrame, masks, disabled, debug, clip]);

    return { applyMask };

}
