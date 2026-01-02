import { ClipTransform, MaskClipProps, MaskData } from "@/lib/types";

const isFiniteNumber = (value: number | undefined | null): value is number =>
  typeof value === "number" && Number.isFinite(value);

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

export const sanitizeScale = (value: number | undefined): number => {
  if (!isFiniteNumber(value)) return 1;
  const v = value as number;
  if (Math.abs(v) < 1e-6) {
    return v < 0 ? -1e-6 : 1e-6;
  }
  return v;
};

const rotate = (x: number, y: number, angleDeg: number) => {
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return {
    x: x * cos - y * sin,
    y: x * sin + y * cos,
  };
};

export const transformPoint = (
  x: number,
  y: number,
  from: ClipTransform,
  to: ClipTransform,
): { x: number; y: number } => {
  const fRot = isFiniteNumber(from.rotation) ? from.rotation : 0;
  const fSx = sanitizeScale(from.scaleX);
  const fSy = sanitizeScale(from.scaleY);
  const fW = (from.width ?? 0) * fSx;
  const fH = (from.height ?? 0) * fSy;

  const tRot = isFiniteNumber(to.rotation) ? to.rotation : 0;
  const tSx = sanitizeScale(to.scaleX);
  const tSy = sanitizeScale(to.scaleY);
  const tW = (to.width ?? from.width ?? 0) * tSx;
  const tH = (to.height ?? from.height ?? 0) * tSy;

  if (!fW || !fH || !isFiniteNumber(fW) || !isFiniteNumber(fH)) {
    const deltaX = (to.x ?? 0) - (from.x ?? 0);
    const deltaY = (to.y ?? 0) - (from.y ?? 0);
    return {
      x: x + deltaX,
      y: y + deltaY,
    };
  }

  // Relative to from.x, from.y
  let dx = x - (from.x ?? 0);
  let dy = y - (from.y ?? 0);

  // Un-rotate
  if (fRot !== 0) {
    const unrotated = rotate(dx, dy, -fRot);
    dx = unrotated.x;
    dy = unrotated.y;
  }

  const relX = dx / fW;
  const relY = dy / fH;

  // Scale
  let tx = relX * tW;
  let ty = relY * tH;

  // Rotate
  if (tRot !== 0) {
    const rotated = rotate(tx, ty, tRot);
    tx = rotated.x;
    ty = rotated.y;
  }

  return {
    x: (to.x ?? 0) + tx,
    y: (to.y ?? 0) + ty,
  };
};

export const transformDimension = (
  value: number,
  fromSize: number,
  toSize: number,
): number => {
  if (!fromSize || !isFiniteNumber(fromSize)) {
    return value;
  }
  return (value * (toSize || fromSize)) / fromSize;
};

export const transformFlatPoints = (
  points: number[],
  from: ClipTransform,
  to: ClipTransform,
): number[] => {
  from = getUncroppedTransform(from);
  to = getUncroppedTransform(to);
  if (!Array.isArray(points) || points.length === 0) {
    return points.slice();
  }
  const next: number[] = new Array(points.length);
  for (let i = 0; i < points.length; i += 2) {
    const x = points[i];
    const y = points[i + 1];
    if (!isFiniteNumber(x) || !isFiniteNumber(y)) {
      next[i] = x;
      next[i + 1] = y;
      continue;
    }
    const transformed = transformPoint(x, y, from, to);
    next[i] = transformed.x;
    next[i + 1] = transformed.y;
  }
  return next;
};

export const transformContours = (
  contours: number[][],
  from: ClipTransform,
  to: ClipTransform,
): number[][] => {
  if (!Array.isArray(contours) || contours.length === 0) {
    return contours.slice();
  }
  return contours.map((contour) =>
    transformFlatPoints(contour ?? [], from, to),
  );
};

export const transformTouchPoints = (
  points: Array<{ x: number; y: number; label: 0 | 1 }>,
  from: ClipTransform,
  to: ClipTransform,
): Array<{ x: number; y: number; label: 0 | 1 }> => {
  from = getUncroppedTransform(from);
  to = getUncroppedTransform(to);
  if (!Array.isArray(points) || points.length === 0) {
    return points.slice();
  }
  return points.map((point) => {
    if (!isFiniteNumber(point.x) || !isFiniteNumber(point.y)) {
      return point;
    }
    const transformed = transformPoint(point.x, point.y, from, to);
    return { ...point, ...transformed };
  });
};

export const transformRotation = (
  rotation: number,
  from: ClipTransform,
  to: ClipTransform,
): number => {
  const normalize = (angle: number): number => {
    let a = ((angle % 360) + 360) % 360; // [0, 360)
    if (a > 180) a -= 360; // (-180, 180]
    return a;
  };

  const fromRotation = isFiniteNumber(from.rotation) ? from.rotation : 0;
  const toRotation = isFiniteNumber(to.rotation) ? to.rotation : 0;
  const deltaRotation = normalize(toRotation - fromRotation);
  return rotation + deltaRotation;
};

export const getUncroppedTransform = (
  transform: ClipTransform,
  offsetXY: boolean = true,
): ClipTransform => {
  if (!transform.crop) {
    return transform;
  }
  let uncroppedWidth = transform.width / transform.crop.width;
  let uncroppedHeight = transform.height / transform.crop.height;
  let offsetX = transform.crop.x * uncroppedWidth;
  let offsetY = transform.crop.y * uncroppedHeight;

  let finalX = transform.x;
  let finalY = transform.y;

  if (offsetXY) {
    const rot = isFiniteNumber(transform.rotation) ? transform.rotation : 0;
    if (rot !== 0) {
      const rotated = rotate(offsetX, offsetY, rot);
      finalX -= rotated.x;
      finalY -= rotated.y;
    } else {
      finalX -= offsetX;
      finalY -= offsetY;
    }
  }

  return {
    ...transform,
    width: uncroppedWidth,
    height: uncroppedHeight,
    x: finalX,
    y: finalY,
  };
};

export const transformShapeBounds = (
  bounds: NonNullable<MaskData["shapeBounds"]>,
  from: ClipTransform,
  to: ClipTransform,
): NonNullable<MaskData["shapeBounds"]> => {
  from = getUncroppedTransform(from);
  to = getUncroppedTransform(to);
  const topLeft = transformPoint(bounds.x, bounds.y, from, to);

  const fSx = sanitizeScale(from.scaleX);
  const fSy = sanitizeScale(from.scaleY);
  const fW = (from.width ?? 0) * fSx;
  const fH = (from.height ?? 0) * fSy;

  const tSx = sanitizeScale(to.scaleX);
  const tSy = sanitizeScale(to.scaleY);
  const tW = (to.width ?? from.width ?? 0) * tSx;
  const tH = (to.height ?? from.height ?? 0) * tSy;

  const newWidth = transformDimension(bounds.width, fW, tW || fW);
  const newHeight = transformDimension(bounds.height, fH, tH || fH);
  const newRotation = transformRotation(bounds.rotation ?? 0, from, to);

  return {
    ...bounds,
    x: topLeft.x,
    y: topLeft.y,
    width: newWidth,
    height: newHeight,
    rotation: newRotation,
  };
};

export const transformMaskData = (
  data: MaskData,
  from: ClipTransform,
  to: ClipTransform,
): MaskData => {
  const nextData: MaskData = { ...data };

  if (data.shapeBounds) {
    nextData.shapeBounds = transformShapeBounds(data.shapeBounds, from, to);
  }
  if (data.lassoPoints) {
    nextData.lassoPoints = transformFlatPoints(data.lassoPoints, from, to);
  }
  if (data.contours) {
    nextData.contours = transformContours(data.contours, from, to);
  }
  if (data.touchPoints) {
    nextData.touchPoints = transformTouchPoints(data.touchPoints, from, to);
  }
  if (data.touchBox) {
    const topLeft = transformPoint(
      data.touchBox.x1,
      data.touchBox.y1,
      from,
      to,
    );
    const bottomRight = transformPoint(
      data.touchBox.x2,
      data.touchBox.y2,
      from,
      to,
    );
    nextData.touchBox = {
      x1: topLeft.x,
      y1: topLeft.y,
      x2: bottomRight.x,
      y2: bottomRight.y,
    };
  }

  return nextData;
};

export const remapMaskWithClipTransform = (
  mask: MaskClipProps,
  from: ClipTransform,
  to: ClipTransform,
): MaskClipProps => {
  if (transformsEqual(from, to)) {
    return { ...mask, transform: { ...to } };
  }

  const transformFrame = (data: MaskData | undefined): MaskData | undefined => {
    if (!data) return data;
    return transformMaskData(data, from, to);
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

export const projectContoursBetweenTransforms = (
  contours: number[][],
  from?: ClipTransform,
  to?: ClipTransform,
): number[][] => {
  if (!from || !to || !Array.isArray(contours) || contours.length === 0) {
    return contours;
  }
  return transformContours(contours, from, to);
};

export const projectTouchPointsBetweenTransforms = (
  points: Array<{ x: number; y: number; label: 0 | 1 }>,
  from?: ClipTransform,
  to?: ClipTransform,
): Array<{ x: number; y: number; label: 0 | 1 }> => {
  if (!from || !to || !Array.isArray(points) || points.length === 0) {
    return points;
  }
  return transformTouchPoints(points, from, to);
};

/**
 * Remap mask for MediaDialog - handles the case where the input media may have
 * different dimensions than BASE_LONG_SIDE (600), while keeping the mask in the
 * exact same relative visual position.
 *
 * This function:
 * 1. Uses the mask's stored transform (or originalClipTransform) as the source coordinate system
 * 2. Calculates the relative position of shape bounds within the source transform
 * 3. Maps that relative position to the new transform's coordinate space
 * 4. Scales dimensions proportionally to maintain visual coverage
 */
export const remapMaskForMediaDialog = (
  mask: MaskClipProps,
  originalClipTransform: ClipTransform,
  newClipTransform: ClipTransform,
): MaskClipProps => {
  // Use the mask's stored transform if available, otherwise use the originalClipTransform
  const sourceTransform = (mask.transform as ClipTransform) || originalClipTransform;

  // If transforms are effectively the same, just update the mask's transform reference
  if (transformsEqual(sourceTransform, newClipTransform)) {
    return { ...mask, transform: { ...newClipTransform } };
  }

  // Get uncropped transforms for proper coordinate calculation
  const fromUncropped = getUncroppedTransform(sourceTransform);
  const toUncropped = getUncroppedTransform(newClipTransform);

  // Calculate the actual rendered sizes of source and target
  const fromScaleX = sanitizeScale(fromUncropped.scaleX);
  const fromScaleY = sanitizeScale(fromUncropped.scaleY);
  const fromWidth = (fromUncropped.width ?? 0) * fromScaleX;
  const fromHeight = (fromUncropped.height ?? 0) * fromScaleY;

  const toScaleX = sanitizeScale(toUncropped.scaleX);
  const toScaleY = sanitizeScale(toUncropped.scaleY);
  const toWidth = (toUncropped.width ?? fromUncropped.width ?? 0) * toScaleX;
  const toHeight = (toUncropped.height ?? fromUncropped.height ?? 0) * toScaleY;

  // Calculate scale ratios between source and target
  const scaleRatioX = fromWidth > 0 ? toWidth / fromWidth : 1;
  const scaleRatioY = fromHeight > 0 ? toHeight / fromHeight : 1;

  // Transform shapeBounds for MediaDialog
  const transformShapeBoundsForDialog = (
    bounds: NonNullable<MaskData["shapeBounds"]>,
  ): NonNullable<MaskData["shapeBounds"]> => {
    // Calculate relative position within source transform (0-1 normalized)
    const relX = fromWidth > 0 ? (bounds.x - (fromUncropped.x ?? 0)) / fromWidth : 0;
    const relY = fromHeight > 0 ? (bounds.y - (fromUncropped.y ?? 0)) / fromHeight : 0;

    // Map relative position to target transform coordinate space
    const newX = (toUncropped.x ?? 0) + relX * toWidth;
    const newY = (toUncropped.y ?? 0) + relY * toHeight;

    // Scale dimensions proportionally
    const newWidth = bounds.width * scaleRatioX;
    const newHeight = bounds.height * scaleRatioY;

    // Handle rotation delta
    const fromRotation = isFiniteNumber(fromUncropped.rotation) ? fromUncropped.rotation : 0;
    const toRotation = isFiniteNumber(toUncropped.rotation) ? toUncropped.rotation : 0;
    const rotationDelta = toRotation - fromRotation;
    const newRotation = (bounds.rotation ?? 0) + rotationDelta;

    return {
      ...bounds,
      x: newX,
      y: newY,
      width: newWidth,
      height: newHeight,
      rotation: newRotation,
    };
  };

  // Transform flat points (for lasso)
  const transformFlatPointsForDialog = (points: number[]): number[] => {
    if (!Array.isArray(points) || points.length === 0) return points.slice();

    const next: number[] = new Array(points.length);
    for (let i = 0; i < points.length; i += 2) {
      const x = points[i];
      const y = points[i + 1];
      if (!isFiniteNumber(x) || !isFiniteNumber(y)) {
        next[i] = x;
        next[i + 1] = y;
        continue;
      }
      // Calculate relative position and map to target space
      const relX = fromWidth > 0 ? (x - (fromUncropped.x ?? 0)) / fromWidth : 0;
      const relY = fromHeight > 0 ? (y - (fromUncropped.y ?? 0)) / fromHeight : 0;
      next[i] = (toUncropped.x ?? 0) + relX * toWidth;
      next[i + 1] = (toUncropped.y ?? 0) + relY * toHeight;
    }
    return next;
  };

  // Transform contours
  const transformContoursForDialog = (contours: number[][]): number[][] => {
    if (!Array.isArray(contours) || contours.length === 0) return contours.slice();
    return contours.map((contour) => transformFlatPointsForDialog(contour ?? []));
  };

  // Transform touch points
  const transformTouchPointsForDialog = (
    points: Array<{ x: number; y: number; label: 0 | 1 }>,
  ): Array<{ x: number; y: number; label: 0 | 1 }> => {
    if (!Array.isArray(points) || points.length === 0) return points.slice();
    return points.map((point) => {
      if (!isFiniteNumber(point.x) || !isFiniteNumber(point.y)) return point;
      const relX = fromWidth > 0 ? (point.x - (fromUncropped.x ?? 0)) / fromWidth : 0;
      const relY = fromHeight > 0 ? (point.y - (fromUncropped.y ?? 0)) / fromHeight : 0;
      return {
        ...point,
        x: (toUncropped.x ?? 0) + relX * toWidth,
        y: (toUncropped.y ?? 0) + relY * toHeight,
      };
    });
  };

  // Transform MaskData for a single keyframe
  const transformMaskDataForDialog = (data: MaskData): MaskData => {
    const nextData: MaskData = { ...data };

    if (data.shapeBounds) {
      nextData.shapeBounds = transformShapeBoundsForDialog(data.shapeBounds);
    }
    if (data.lassoPoints) {
      nextData.lassoPoints = transformFlatPointsForDialog(data.lassoPoints);
    }
    if (data.contours) {
      nextData.contours = transformContoursForDialog(data.contours);
    }
    if (data.touchPoints) {
      nextData.touchPoints = transformTouchPointsForDialog(data.touchPoints);
    }
    if (data.touchBox) {
      const relX1 = fromWidth > 0 ? (data.touchBox.x1 - (fromUncropped.x ?? 0)) / fromWidth : 0;
      const relY1 = fromHeight > 0 ? (data.touchBox.y1 - (fromUncropped.y ?? 0)) / fromHeight : 0;
      const relX2 = fromWidth > 0 ? (data.touchBox.x2 - (fromUncropped.x ?? 0)) / fromWidth : 0;
      const relY2 = fromHeight > 0 ? (data.touchBox.y2 - (fromUncropped.y ?? 0)) / fromHeight : 0;
      nextData.touchBox = {
        x1: (toUncropped.x ?? 0) + relX1 * toWidth,
        y1: (toUncropped.y ?? 0) + relY1 * toHeight,
        x2: (toUncropped.x ?? 0) + relX2 * toWidth,
        y2: (toUncropped.y ?? 0) + relY2 * toHeight,
      };
    }

    return nextData;
  };

  // Process all keyframes
  let keyframes: MaskClipProps["keyframes"];
  if (mask.keyframes instanceof Map) {
    const updated = new Map<number, MaskData>();
    mask.keyframes.forEach((value, key) => {
      updated.set(key, transformMaskDataForDialog(value));
    });
    keyframes = updated;
  } else {
    const updated: Record<number, MaskData> = {};
    Object.keys(mask.keyframes).forEach((key) => {
      const numericKey = Number(key);
      const frameData = (mask.keyframes as Record<number, MaskData>)[numericKey];
      if (frameData) {
        updated[numericKey] = transformMaskDataForDialog(frameData);
      }
    });
    keyframes = updated;
  }

  return {
    ...mask,
    keyframes,
    transform: { ...newClipTransform },
    lastModified: mask.lastModified,
  };
};
