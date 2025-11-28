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
    a.opacity === b.opacity
  );
};

const getSafeNumber = (value: number | undefined, fallback: number): number =>
  isFiniteNumber(value) ? value! : fallback;

const sanitizeScale = (value: number | undefined): number => {
  if (!isFiniteNumber(value)) return 1;
  const v = value as number;
  if (Math.abs(v) < 1e-6) {
    return v < 0 ? -1e-6 : 1e-6;
  }
  return v;
};

const getScale = (
  transform: ClipTransform,
): { scaleX: number; scaleY: number } => ({
  scaleX: sanitizeScale(transform.scaleX),
  scaleY: sanitizeScale(transform.scaleY),
});

const getBaseSize = (
  transform: ClipTransform,
): { width: number; height: number } => ({
  width: getSafeNumber(transform.width, 0),
  height: getSafeNumber(transform.height, 0),
});

const getActualSize = (
  transform: ClipTransform,
): { width: number; height: number } => {
  const base = getBaseSize(transform);
  const scale = getScale(transform);
  return {
    width: base.width * scale.scaleX,
    height: base.height * scale.scaleY,
  };
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

const transformPoint = (
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

const transformDimension = (
  value: number,
  fromSize: number,
  toSize: number,
): number => {
  if (!fromSize || !isFiniteNumber(fromSize)) {
    return value;
  }
  return (value * (toSize || fromSize)) / fromSize;
};

const transformFlatPoints = (
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

const transformContours = (
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

const transformTouchPoints = (
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

const transformRotation = (
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
