// Helper function to calculate polygon area (shoelace formula)
export const calculateArea = (points: number[]) => {
  let area = 0;
  const n = points.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const xi = points[i * 2];
    const yi = points[i * 2 + 1];
    const xj = points[j * 2];
    const yj = points[j * 2 + 1];
    area += xi * yj - xj * yi;
  }
  return Math.abs(area / 2);
};

// Helper function to check if a point is inside a polygon
export const isPointInPolygon = (
  x: number,
  y: number,
  polygonPoints: number[],
) => {
  let inside = false;
  const n = polygonPoints.length / 2;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygonPoints[i * 2];
    const yi = polygonPoints[i * 2 + 1];
    const xj = polygonPoints[j * 2];
    const yj = polygonPoints[j * 2 + 1];

    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
};

// Helper function to check if one polygon is completely inside another
export const isPolygonInsidePolygon = (
  innerPoints: number[],
  outerPoints: number[],
) => {
  // Check if all points of inner polygon are inside outer polygon
  const n = innerPoints.length / 2;
  for (let i = 0; i < n; i++) {
    const x = innerPoints[i * 2];
    const y = innerPoints[i * 2 + 1];
    if (!isPointInPolygon(x, y, outerPoints)) {
      return false;
    }
  }
  return true;
};

// Helper function to check if two line segments intersect
export const doSegmentsIntersect = (
  p1x: number,
  p1y: number,
  p2x: number,
  p2y: number,
  p3x: number,
  p3y: number,
  p4x: number,
  p4y: number,
) => {
  const denom = (p4y - p3y) * (p2x - p1x) - (p4x - p3x) * (p2y - p1y);
  if (Math.abs(denom) < 1e-10) return false; // Parallel or coincident

  const ua = ((p4x - p3x) * (p1y - p3y) - (p4y - p3y) * (p1x - p3x)) / denom;
  const ub = ((p2x - p1x) * (p1y - p3y) - (p2y - p1y) * (p1x - p3x)) / denom;

  return ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1;
};

// Helper function to check if two polygons intersect
export const doPolygonsIntersect = (points1: number[], points2: number[]) => {
  const n1 = points1.length / 2;
  const n2 = points2.length / 2;

  // Check if any edges intersect
  for (let i = 0; i < n1; i++) {
    const j = (i + 1) % n1;
    const p1x = points1[i * 2];
    const p1y = points1[i * 2 + 1];
    const p2x = points1[j * 2];
    const p2y = points1[j * 2 + 1];

    for (let k = 0; k < n2; k++) {
      const l = (k + 1) % n2;
      const p3x = points2[k * 2];
      const p3y = points2[k * 2 + 1];
      const p4x = points2[l * 2];
      const p4y = points2[l * 2 + 1];

      if (doSegmentsIntersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)) {
        return true;
      }
    }
  }

  // Check if one polygon is inside the other
  if (isPointInPolygon(points1[0], points1[1], points2)) return true;
  if (isPointInPolygon(points2[0], points2[1], points1)) return true;

  return false;
};

// Helper function to compute convex hull (Graham scan algorithm)
export const computeConvexHull = (points: number[]) => {
  const pts: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < points.length; i += 2) {
    pts.push({ x: points[i], y: points[i + 1] });
  }

  if (pts.length < 3) return points;

  // Find the point with lowest y (and leftmost if tie)
  let minIdx = 0;
  for (let i = 1; i < pts.length; i++) {
    if (
      pts[i].y < pts[minIdx].y ||
      (pts[i].y === pts[minIdx].y && pts[i].x < pts[minIdx].x)
    ) {
      minIdx = i;
    }
  }

  // Swap to put lowest point first
  [pts[0], pts[minIdx]] = [pts[minIdx], pts[0]];
  const pivot = pts[0];

  // Sort by polar angle
  const sorted = [
    pivot,
    ...pts.slice(1).sort((a, b) => {
      const angleA = Math.atan2(a.y - pivot.y, a.x - pivot.x);
      const angleB = Math.atan2(b.y - pivot.y, b.x - pivot.x);
      if (Math.abs(angleA - angleB) < 1e-10) {
        const distA = (a.x - pivot.x) ** 2 + (a.y - pivot.y) ** 2;
        const distB = (b.x - pivot.x) ** 2 + (b.y - pivot.y) ** 2;
        return distA - distB;
      }
      return angleA - angleB;
    }),
  ];

  // Build hull
  const hull = [sorted[0], sorted[1]];

  const ccw = (
    p1: { x: number; y: number },
    p2: { x: number; y: number },
    p3: { x: number; y: number },
  ) => {
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
  };

  for (let i = 2; i < sorted.length; i++) {
    while (
      hull.length > 1 &&
      ccw(hull[hull.length - 2], hull[hull.length - 1], sorted[i]) <= 0
    ) {
      hull.pop();
    }
    hull.push(sorted[i]);
  }

  // Convert back to flat array
  const result: number[] = [];
  for (const pt of hull) {
    result.push(pt.x, pt.y);
  }
  return result;
};

// Helper function to merge two polygons (using convex hull)
export const mergePolygons = (points1: number[], points2: number[]) => {
  // Combine all points
  const allPoints = [...points1, ...points2];
  // Return convex hull of all points
  return computeConvexHull(allPoints);
};
