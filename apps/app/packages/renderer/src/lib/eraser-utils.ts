type Pt = { x: number; y: number };

interface LineTransform {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number; // in degrees
}

const eps = 1e-7;

/**
 * Transform a point from world space to a line's local coordinate space.
 * This inverts the line's transform (translation, rotation, scale).
 */
function worldToLocal(worldPt: Pt, transform: LineTransform): Pt {
  // Step 1: Undo translation
  let x = worldPt.x - transform.x;
  let y = worldPt.y - transform.y;

  // Step 2: Undo rotation (rotate by -angle)
  const angleRad = -((transform.rotation * Math.PI) / 180);
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  const rotatedX = x * cos - y * sin;
  const rotatedY = x * sin + y * cos;

  // Step 3: Undo scale
  const localX = rotatedX / transform.scaleX;
  const localY = rotatedY / transform.scaleY;

  return { x: localX, y: localY };
}

/**
 * Transform eraser points from world space to a line's local space,
 * and also adjust the eraser radius for the line's scale.
 */
export function transformEraserToLocal(
  worldEraserPts: Pt[],
  transform: LineTransform,
  worldRadius: number,
): { localPts: Pt[]; localRadius: number } {
  const localPts = worldEraserPts.map((pt) => worldToLocal(pt, transform));

  // Adjust radius for scale - use average of scaleX and scaleY
  // In local space, the eraser needs to be scaled inversely
  const avgScale = (transform.scaleX + transform.scaleY) / 2;
  const localRadius = worldRadius / avgScale;

  return { localPts, localRadius };
}

function dist2(a: Pt, b: Pt) {
  const dx = a.x - b.x,
    dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function lerpPt(a: Pt, b: Pt, t: number): Pt {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
}

function pointInsideAnyCircle(p: Pt, eraserPts: Pt[], r: number): boolean {
  const r2 = r * r;
  for (let i = 0; i < eraserPts.length; i++) {
    if (dist2(p, eraserPts[i]) <= r2) return true;
  }
  return false;
}

/**
 * Return all intersection t values in [0,1] where segment AB intersects circle (C,r)
 * Solve |A + t*(B-A) - C|^2 = r^2
 */
function segmentCircleIntersectionTs(A: Pt, B: Pt, C: Pt, r: number): number[] {
  const dx = B.x - A.x,
    dy = B.y - A.y;
  const fx = A.x - C.x,
    fy = A.y - C.y;

  const a = dx * dx + dy * dy;
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - r * r;

  const disc = b * b - 4 * a * c;
  if (disc < 0) return [];
  if (a < eps) return []; // degenerate segment

  const s = Math.sqrt(Math.max(0, disc));
  const t1 = (-b - s) / (2 * a);
  const t2 = (-b + s) / (2 * a);

  const ts: number[] = [];
  if (t1 >= -eps && t1 <= 1 + eps) ts.push(Math.min(1, Math.max(0, t1)));
  if (t2 >= -eps && t2 <= 1 + eps) ts.push(Math.min(1, Math.max(0, t2)));
  // merge nearly-identical roots
  ts.sort((u, v) => u - v);
  const uniq: number[] = [];
  for (const t of ts) {
    if (uniq.length === 0 || Math.abs(t - uniq[uniq.length - 1]) > 1e-6)
      uniq.push(t);
  }
  return uniq;
}

/**
 * Given segment AB and many circles (eraser samples), compute all boundary ts.
 * We add 0 and 1 and the circle intersection ts, then we keep sub-intervals that lie OUTSIDE all circles.
 */
function keepIntervalsForSegment(
  A: Pt,
  B: Pt,
  eraserPts: Pt[],
  r: number,
): Array<[number, number]> {
  // Quick reject: if every circle center is far from segment bbox by > r, you can skip exact tests.
  // (Naive version here; feel free to optimize with a grid/kd-tree later.)
  const ts = new Set<number>([0, 1]);
  for (let i = 0; i < eraserPts.length; i++) {
    const its = segmentCircleIntersectionTs(A, B, eraserPts[i], r);
    for (const t of its) ts.add(t);
  }
  const sorted = Array.from(ts).sort((a, b) => a - b);
  const keep: Array<[number, number]> = [];

  for (let i = 0; i < sorted.length - 1; i++) {
    const t0 = sorted[i],
      t1 = sorted[i + 1];
    // sample midpoint to classify
    const tm = (t0 + t1) * 0.5;
    const Pm = lerpPt(A, B, tm);
    if (!pointInsideAnyCircle(Pm, eraserPts, r)) {
      keep.push([t0, t1]);
    }
  }
  return keep;
}

/**
 * Erase a polyline by union of eraser circles.
 * Returns array of polylines (each as a flat points array) to render as separate Konva.Line.
 * Input points is flat [x0,y0,x1,y1,...].
 *
 * @param points - Line points in the line's local coordinate space
 * @param eraserPts - Eraser points (should be in the same coordinate space as points)
 * @param radius - Eraser radius (should be in the same coordinate space)
 */
export function erasePolylineByEraser(
  points: number[],
  eraserPts: Pt[],
  radius: number,
): number[][] {
  if (points.length < 4 || eraserPts.length === 0) return [points.slice()];

  // Convert to Pt[]
  const P: Pt[] = [];
  for (let i = 0; i < points.length; i += 2)
    P.push({ x: points[i], y: points[i + 1] });

  const outPolylines: Pt[][] = [];
  let current: Pt[] = [];

  const pushCurrentIfValid = () => {
    if (current.length >= 2) {
      // dedupe contiguous identical points
      const cleaned: Pt[] = [current[0]];
      for (let i = 1; i < current.length; i++) {
        const a = cleaned[cleaned.length - 1],
          b = current[i];
        if (Math.hypot(a.x - b.x, a.y - b.y) > 1e-6) cleaned.push(b);
      }
      if (cleaned.length >= 2) outPolylines.push(cleaned);
    }
    current = [];
  };

  for (let i = 0; i < P.length - 1; i++) {
    const A = P[i],
      B = P[i + 1];
    const intervals = keepIntervalsForSegment(A, B, eraserPts, radius);

    if (intervals.length === 0) {
      // segment fully erased -> break the polyline
      pushCurrentIfValid();
      continue;
    }

    // stitch kept sub-segments
    for (const [t0, t1] of intervals) {
      const S = lerpPt(A, B, t0);
      const E = lerpPt(A, B, t1);

      if (current.length === 0) {
        current.push(S, E);
      } else {
        // Avoid duplicating the seam point
        const last = current[current.length - 1];
        if (Math.hypot(last.x - S.x, last.y - S.y) > 1e-6) {
          // gap -> previous part ends, start new
          pushCurrentIfValid();
          current.push(S, E);
        } else {
          // continuous -> extend
          current.push(E);
        }
      }
    }
  }
  pushCurrentIfValid();

  // back to flat arrays
  return outPolylines.map((poly) => {
    const flat: number[] = [];
    for (const p of poly) {
      flat.push(p.x, p.y);
    }
    return flat;
  });
}

/**
 * Check if two line segments can be merged based on their properties
 */
function canMergeLines(line1: any, line2: any): boolean {
  return (
    line1.tool === line2.tool &&
    line1.stroke === line2.stroke &&
    line1.strokeWidth === line2.strokeWidth &&
    line1.opacity === line2.opacity &&
    line1.smoothing === line2.smoothing &&
    line1.transform.x === line2.transform.x &&
    line1.transform.y === line2.transform.y &&
    line1.transform.scaleX === line2.transform.scaleX &&
    line1.transform.scaleY === line2.transform.scaleY &&
    line1.transform.rotation === line2.transform.rotation &&
    line1.transform.opacity === line2.transform.opacity
  );
}

/**
 * Check if two points are within tolerance distance
 */
function pointsMatch(p1: Pt, p2: Pt, tolerance: number): boolean {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y) <= tolerance;
}

/**
 * Convert flat points array to Pt array
 */
function flatToPts(flat: number[]): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i < flat.length; i += 2) {
    pts.push({ x: flat[i], y: flat[i + 1] });
  }
  return pts;
}

/**
 * Convert Pt array to flat points array
 */
function ptsToFlat(pts: Pt[]): number[] {
  const flat: number[] = [];
  for (const p of pts) {
    flat.push(p.x, p.y);
  }
  return flat;
}

/**
 * Merge line segments that share endpoints and have matching properties.
 * This reduces fragmentation after erasing operations.
 *
 * Endpoints are considered matching if they're within a distance equal to the line's strokeWidth,
 * which provides appropriate tolerance for lines of different thicknesses.
 *
 * @param lines - Array of line objects with points and properties
 * @param aggressive - If true, performs multiple merge passes until no more merges possible
 */
export function mergeConnectedLines(
  lines: any[],
  aggressive: boolean = false,
): any[] {
  if (lines.length <= 1) return lines;

  let merged = [...lines];
  let didMerge = true;
  let iterations = 0;
  const maxIterations = aggressive ? 10 : 1;

  while (didMerge && iterations < maxIterations) {
    didMerge = false;
    iterations++;

    const newMerged: any[] = [];
    const used = new Set<number>();

    for (let i = 0; i < merged.length; i++) {
      if (used.has(i)) continue;

      const line1 = merged[i];
      const pts1 = flatToPts(line1.points);

      if (pts1.length < 2) {
        used.add(i);
        continue;
      }

      let currentLine = { ...line1, points: [...line1.points] };
      let currentPts = [...pts1];
      let foundMatch = true;

      // Calculate merge tolerance based on stroke width (use strokeWidth as tolerance)
      const tolerance = currentLine.strokeWidth;

      // Keep trying to extend this line
      while (foundMatch && currentPts.length >= 2) {
        foundMatch = false;
        const start = currentPts[0];
        const end = currentPts[currentPts.length - 1];

        for (let j = i + 1; j < merged.length; j++) {
          if (used.has(j)) continue;

          const line2 = merged[j];
          if (!canMergeLines(currentLine, line2)) continue;

          const pts2 = flatToPts(line2.points);
          if (pts2.length < 2) continue;

          const start2 = pts2[0];
          const end2 = pts2[pts2.length - 1];

          // Check all possible connection patterns
          if (pointsMatch(end, start2, tolerance)) {
            // line1 end connects to line2 start
            currentPts = [...currentPts, ...pts2.slice(1)];
            currentLine.points = ptsToFlat(currentPts);
            used.add(j);
            foundMatch = true;
            didMerge = true;
            break;
          } else if (pointsMatch(end, end2, tolerance)) {
            // line1 end connects to line2 end (reversed)
            currentPts = [...currentPts, ...pts2.slice(0, -1).reverse()];
            currentLine.points = ptsToFlat(currentPts);
            used.add(j);
            foundMatch = true;
            didMerge = true;
            break;
          } else if (pointsMatch(start, end2, tolerance)) {
            // line1 start connects to line2 end
            currentPts = [...pts2, ...currentPts.slice(1)];
            currentLine.points = ptsToFlat(currentPts);
            used.add(j);
            foundMatch = true;
            didMerge = true;
            break;
          } else if (pointsMatch(start, start2, tolerance)) {
            // line1 start connects to line2 start (line1 reversed)
            currentPts = [...currentPts.reverse(), ...pts2.slice(1)];
            currentLine.points = ptsToFlat(currentPts);
            used.add(j);
            foundMatch = true;
            didMerge = true;
            break;
          }
        }
      }

      used.add(i);
      newMerged.push(currentLine);
    }

    merged = newMerged;
  }

  return merged;
}
