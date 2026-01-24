export type KonvaCornerRadius = number | number[];

/**
 * Konva (via CanvasRenderingContext2D.arc) will throw if radius is negative.
 * This clamps cornerRadius to:
 * - >= 0
 * - <= min(width, height) / 2 (when dimensions are provided)
 *
 * Accepts both number and array forms used by Konva (e.g. [tl, tr, br, bl]).
 */
export function sanitizeCornerRadius(
  cornerRadius: unknown,
  width?: number,
  height?: number,
): KonvaCornerRadius {
  const wOk = typeof width === "number" && Number.isFinite(width) && width > 0;
  const hOk = typeof height === "number" && Number.isFinite(height) && height > 0;
  const maxR = wOk && hOk ? Math.min(width!, height!) / 2 : undefined;

  const clampOne = (v: unknown): number => {
    const n = typeof v === "number" ? v : Number(v);
    if (!Number.isFinite(n)) return 0;
    const nonNeg = Math.max(0, n);
    return typeof maxR === "number" ? Math.min(nonNeg, maxR) : nonNeg;
  };

  if (Array.isArray(cornerRadius)) {
    return cornerRadius.map(clampOne);
  }

  return clampOne(cornerRadius);
}


