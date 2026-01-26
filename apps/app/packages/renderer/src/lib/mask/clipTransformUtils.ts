import { MaskClipProps } from "../types";
import { ClipTransform } from "../types";
import { MaskData } from "../types";
import { getUncroppedTransform, sanitizeScale, transformDimension, transformRotation } from "./transformUtils";

const transformsEqual = (a?: ClipTransform, b?: ClipTransform): boolean => {
    if (!a || !b) return false;
    return (
      a.x === b.x &&
      a.y === b.y &&
      a.width === b.width &&
      a.height === b.height &&
      a.scaleX === b.scaleX &&
      a.scaleY === b.scaleY &&
      a.rotation === b.rotation &&
      a.cornerRadius === b.cornerRadius &&
      a.opacity === b.opacity && 
      (a.crop?.x === b.crop?.x && a.crop?.y === b.crop?.y && a.crop?.width === b.crop?.width && a.crop?.height === b.crop?.height)
    );
  };


export const transformShapeBoundsProportional = (
    bounds: NonNullable<MaskData["shapeBounds"]>,
    from: ClipTransform,
    to: ClipTransform,
  ): NonNullable<MaskData["shapeBounds"]> => {
    from = getUncroppedTransform(from);
    to = getUncroppedTransform(to);

    const fSx = sanitizeScale(from.scaleX);
    const fSy = sanitizeScale(from.scaleY);
    const fW = (from.width ?? 0) * fSx;
    const fH = (from.height ?? 0) * fSy;
  
    const tSx = sanitizeScale(to.scaleX);
    const tSy = sanitizeScale(to.scaleY);
    const tW = (to.width ?? from.width ?? 0) * tSx;
    const tH = (to.height ?? from.height ?? 0) * tSy;
  
    const newX = transformDimension(bounds.x, fW, tW || fW);
    const newY = transformDimension(bounds.y, fH, tH || fH);
    const newWidth = transformDimension(bounds.width, fW, tW || fW);
    const newHeight = transformDimension(bounds.height, fH, tH || fH);
    const newRotation = transformRotation(bounds.rotation ?? 0, from, to);
    

    return {
      ...bounds,
      x: newX,
      y: newY,
      width: newWidth,
      height: newHeight,
      rotation: newRotation,
    };
  };

export const transformFlatPointsProportional = (
  points: number[],
  from: ClipTransform,
  to: ClipTransform
): number[] => {
  if (!Array.isArray(points) || points.length === 0) {
    return points.slice();
  }
  
  from = getUncroppedTransform(from);
  to = getUncroppedTransform(to);

  const fSx = sanitizeScale(from.scaleX);
  const fSy = sanitizeScale(from.scaleY);
  const fW = (from.width ?? 0) * fSx;
  const fH = (from.height ?? 0) * fSy;

  const tSx = sanitizeScale(to.scaleX);
  const tSy = sanitizeScale(to.scaleY);
  const tW = (to.width ?? from.width ?? 0) * tSx;
  const tH = (to.height ?? from.height ?? 0) * tSy;

  const next: number[] = new Array(points.length);
  for (let i = 0; i < points.length; i += 2) {
      next[i] = transformDimension(points[i], fW, tW || fW);
      next[i + 1] = transformDimension(points[i + 1], fH, tH || fH);
  }
  return next;
};

export const transformContoursProportional = (
  contours: number[][],
  from: ClipTransform,
  to: ClipTransform
): number[][] => {
  if (!Array.isArray(contours) || contours.length === 0) {
    return contours.slice();
  }
  return contours.map(contour => transformFlatPointsProportional(contour, from, to));
};

export const transformTouchPointsProportional = (
  points: Array<{ x: number; y: number; label: 0 | 1 }>,
  from: ClipTransform,
  to: ClipTransform
): Array<{ x: number; y: number; label: 0 | 1 }> => {
  if (!Array.isArray(points) || points.length === 0) {
    return points.slice();
  }

  from = getUncroppedTransform(from);
  to = getUncroppedTransform(to);

  const fSx = sanitizeScale(from.scaleX);
  const fSy = sanitizeScale(from.scaleY);
  const fW = (from.width ?? 0) * fSx;
  const fH = (from.height ?? 0) * fSy;

  const tSx = sanitizeScale(to.scaleX);
  const tSy = sanitizeScale(to.scaleY);
  const tW = (to.width ?? from.width ?? 0) * tSx;
  const tH = (to.height ?? from.height ?? 0) * tSy;

  return points.map(p => ({
      ...p,
      x: transformDimension(p.x, fW, tW || fW),
      y: transformDimension(p.y, fH, tH || fH)
  }));
};

export const transformMaskDataProportional = (
    data: MaskData,
    from: ClipTransform,
    to: ClipTransform,
  ): MaskData => {
    const nextData: MaskData = { ...data };
  
    if (data.shapeBounds) {
      nextData.shapeBounds = transformShapeBoundsProportional(data.shapeBounds, from, to);
    }
    if (data.lassoPoints) {
      nextData.lassoPoints = transformFlatPointsProportional(data.lassoPoints, from, to);
    }
    if (data.contours) {
      nextData.contours = transformContoursProportional(data.contours, from, to);
    }
    if (data.touchPoints) {
      nextData.touchPoints = transformTouchPointsProportional(data.touchPoints, from, to);
    }

  
    return nextData;
  };

export const remapMaskWithClipTransformProportional = (
    mask: MaskClipProps,
    from: ClipTransform,
    to: ClipTransform,
  ): MaskClipProps => {
    if (transformsEqual(from, to)) {
      return { ...mask, transform: { ...to } };
    }
  
    const transformFrame = (data: MaskData | undefined): MaskData | undefined => {
      if (!data) return data;
      return transformMaskDataProportional(data, from, to);
    };
  
    let keyframes: MaskClipProps["keyframes"];
    if (mask.keyframes instanceof Map) {
      const updated = new Map<number, MaskData>();
      mask.keyframes.forEach((value, key) => {
        updated.set(key, transformFrame(value) ?? {});
      });
      keyframes = updated;
    } else {
      const updated: Record<number, MaskData> = {};
      Object.keys(mask.keyframes).forEach((key) => {
        const numericKey = Number(key);
        updated[numericKey] =
          transformFrame(
            (mask.keyframes as Record<number, MaskData>)[numericKey],
          ) ?? {};
      });
      keyframes = updated;
    }
  
    return {
      ...mask,
      keyframes,
      transform: { ...to },
      lastModified: mask.lastModified,
    };
  };
