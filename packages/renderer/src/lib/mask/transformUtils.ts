import { ClipTransform, MaskClipProps, MaskData } from '@/lib/types';

const isFiniteNumber = (value: number | undefined | null): value is number =>
  typeof value === 'number' && Number.isFinite(value);

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
    a.opacity === b.opacity
  );
};

const transformPoint = (
  x: number,
  y: number,
  from: ClipTransform,
  to: ClipTransform
): { x: number; y: number } => {
  const deltaX = (to.x ?? 0) - (from.x ?? 0);
  const deltaY = (to.y ?? 0) - (from.y ?? 0);

  const fromWidth = from.width ?? 0;
  const fromHeight = from.height ?? 0;
  const toWidth = to.width ?? fromWidth;
  const toHeight = to.height ?? fromHeight;

  if (!fromWidth || !fromHeight || !isFiniteNumber(fromWidth) || !isFiniteNumber(fromHeight)) {
    return {
      x: x + deltaX,
      y: y + deltaY,
    };
  }

  const relX = (x - (from.x ?? 0)) / fromWidth;
  const relY = (y - (from.y ?? 0)) / fromHeight;

  return {
    x: (to.x ?? 0) + relX * (toWidth || fromWidth),
    y: (to.y ?? 0) + relY * (toHeight || fromHeight),
  };
};

const transformDimension = (value: number, fromSize: number, toSize: number): number => {
  if (!fromSize || !isFiniteNumber(fromSize)) {
    return value;
  }
  return value * (toSize || fromSize) / fromSize;
};

const transformFlatPoints = (points: number[], from: ClipTransform, to: ClipTransform): number[] => {
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

const transformContours = (contours: number[][], from: ClipTransform, to: ClipTransform): number[][] => {
  if (!Array.isArray(contours) || contours.length === 0) {
    return contours.slice();
  }
  return contours.map((contour) => transformFlatPoints(contour ?? [], from, to));
};

const transformTouchPoints = (
  points: Array<{ x: number; y: number; label: 0 | 1 }>,
  from: ClipTransform,
  to: ClipTransform
): Array<{ x: number; y: number; label: 0 | 1 }> => {
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

const transformRotation = (rotation: number, from: ClipTransform, to: ClipTransform): number => {
  const normalize = (angle: number): number => {
    let a = ((angle % 360) + 360) % 360; // [0, 360)
    if (a > 180) a -= 360; // (-180, 180]
    return a;
  };

  const fromRotation = isFiniteNumber(from.rotation) ? from.rotation : 0;
  const toRotation = isFiniteNumber(to.rotation) ? to.rotation : 0;
  const deltaRotation = normalize(toRotation - fromRotation);
  return normalize(rotation + deltaRotation);
}

const transformShapeBounds = (
  bounds: NonNullable<MaskData['shapeBounds']>,
  from: ClipTransform,
  to: ClipTransform
): NonNullable<MaskData['shapeBounds']> => {
  const topLeft = transformPoint(bounds.x, bounds.y, from, to);
  const newWidth = transformDimension(bounds.width, from.width ?? 0, to.width ?? from.width ?? 0);
  const newHeight = transformDimension(bounds.height, from.height ?? 0, to.height ?? from.height ?? 0);

  return {
    ...bounds,
    x: topLeft.x,
    y: topLeft.y,
    width: newWidth,
    height: newHeight,
  };
};

const transformMaskData = (data: MaskData, from: ClipTransform, to: ClipTransform): MaskData => {
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
    const topLeft = transformPoint(data.touchBox.x1, data.touchBox.y1, from, to);
    const bottomRight = transformPoint(data.touchBox.x2, data.touchBox.y2, from, to);
    nextData.touchBox = {
      x1: topLeft.x,
      y1: topLeft.y,
      x2: bottomRight.x,
      y2: bottomRight.y,
    };
  }
  if (data.drawStrokes) {
    nextData.drawStrokes = data.drawStrokes.map((stroke) => ({
      ...stroke,
      points: transformFlatPoints(stroke.points, from, to),
    }));
  }

  return nextData;
};

export const remapMaskWithClipTransform = (
  mask: MaskClipProps,
  from: ClipTransform,
  to: ClipTransform
): MaskClipProps => {
  if (transformsEqual(from, to)) {
    return { ...mask, transform: { ...to } };
  }

  const transformFrame = (data: MaskData | undefined): MaskData | undefined => {
    if (!data) return data;
    return transformMaskData(data, from, to);
  };

  let keyframes: MaskClipProps['keyframes'];
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
      updated[numericKey] = transformFrame((mask.keyframes as Record<number, MaskData>)[numericKey]) ?? {};
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
  to?: ClipTransform
): number[][] => {
  if (!from || !to || !Array.isArray(contours) || contours.length === 0) {
    return contours;
  }
  return transformContours(contours, from, to);
};

export const projectTouchPointsBetweenTransforms = (
  points: Array<{ x: number; y: number; label: 0 | 1 }>,
  from?: ClipTransform,
  to?: ClipTransform
): Array<{ x: number; y: number; label: 0 | 1 }> => {
  if (!from || !to || !Array.isArray(points) || points.length === 0) {
    return points;
  }
  return transformTouchPoints(points, from, to);
};
