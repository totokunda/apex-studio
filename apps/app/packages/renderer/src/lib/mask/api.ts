import {
  createMask as createMaskPreload,
  startMaskTrack,
  onMaskTrackChunk,
  onMaskTrackError,
  onMaskTrackEnd,
  cancelMaskTrack as cancelMaskTrackPreload,
  startMaskTrackShapes,
  onMaskTrackShapesChunk,
  onMaskTrackShapesError,
  onMaskTrackShapesEnd,
} from "@app/preload";
import { toast } from "sonner";
import { useEffect, useRef, useState, useCallback } from "react";
import type { ClipTransform, MediaInfo } from "../types";

export interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface MaskRequest {
  id?: string;
  input_path: string;
  frame_number?: number;
  tool: string;
  points?: Array<{ x: number; y: number }>;
  point_labels?: Array<number>;
  box?: { x1: number; y1: number; x2: number; y2: number };
  multimask_output?: boolean;
  simplify_tolerance?: number;
}

export interface MaskResponse {
  status: string;
  contours?: Array<Array<number>>;
  message?: string;
}

const maskCache = new Map<string, MaskResponse>();

function stableStringify(value: unknown): string {
  const seen = new WeakSet<object>();

  const helper = (input: unknown): unknown => {
    if (input === null || typeof input !== "object") {
      return input;
    }

    const typedInput = input as object;

    if (seen.has(typedInput)) {
      return;
    }

    seen.add(typedInput);

    if (Array.isArray(input)) {
      return input.map((item) => helper(item));
    }

    const result: Record<string, unknown> = {};
    const record = input as Record<string, unknown>;
    for (const key of Object.keys(record).sort()) {
      const nestedValue = record[key];
      if (typeof nestedValue === "undefined") {
        continue;
      }
      result[key] = helper(nestedValue);
    }

    return result;
  };

  return JSON.stringify(helper(value));
}

function createMaskCacheKey(request: MaskRequest): string {
  return stableStringify(request);
}

// Helper function to convert flat array [x1, y1, x2, y2, ...] to array of {x, y} objects
export function flatArrayToPoints(
  flatArray: number[],
): Array<{ x: number; y: number }> {
  const points: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < flatArray.length; i += 2) {
    if (i + 1 < flatArray.length) {
      points.push({ x: flatArray[i], y: flatArray[i + 1] });
    }
  }
  return points;
}

// Helper function to convert array of {x, y} objects to flat array [x1, y1, x2, y2, ...]
export function pointsToFlatArray(
  points: Array<{ x: number; y: number }>,
): number[] {
  const flatArray: number[] = [];
  for (const point of points) {
    flatArray.push(point.x, point.y);
  }
  return flatArray;
}

// Helper function to normalize points from display coordinates to media coordinates
// This accounts for aspect-fit letterboxing, user transforms, and optional crops
export function normalizePoints(
  points: Array<{ x: number; y: number }>,
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  filterOutOfBounds: boolean = true,
  clipTransform?: ClipTransform,
): Array<{ x: number; y: number }> {
  const mediaWidth =
    mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight =
    mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;

  if (!mediaWidth || !mediaHeight) {
    console.warn("Media dimensions not available, using display dimensions");
    return points;
  }

  // If clipTransform is provided, use it directly
  // Otherwise calculate aspect-fit dimensions within the rect
  let imageX, imageY, imageWidth, imageHeight;

  if (clipTransform) {
    const sx = clipTransform.scaleX || 1;
    const sy = clipTransform.scaleY || 1;
    const cropW =
      clipTransform.crop?.width && clipTransform.crop.width > 0
        ? clipTransform.crop.width
        : 1;
    const cropH =
      clipTransform.crop?.height && clipTransform.crop.height > 0
        ? clipTransform.crop.height
        : 1;
    const uncroppedWidth = (clipTransform.width * sx) / cropW;
    const uncroppedHeight = (clipTransform.height * sy) / cropH;
    const cropX = clipTransform.crop?.x || 0;
    const cropY = clipTransform.crop?.y || 0;

    imageWidth = uncroppedWidth;
    imageHeight = uncroppedHeight;
    // Shift the image rect so (0,0) corresponds to the top-left of the uncropped media
    imageX = clipTransform.x - cropX * uncroppedWidth;
    imageY = clipTransform.y - cropY * uncroppedHeight;
  } else {
    // Calculate aspect-fit dimensions (letterbox/pillarbox)
    const mediaAspect = mediaWidth / mediaHeight;
    const rectAspect = displayWidth / displayHeight;

    if (rectAspect > mediaAspect) {
      // Letterbox (black bars on left/right)
      imageHeight = displayHeight;
      imageWidth = displayHeight * mediaAspect;
      imageX = (displayWidth - imageWidth) / 2;
      imageY = 0;
    } else {
      // Pillarbox (black bars on top/bottom)
      imageWidth = displayWidth;
      imageHeight = displayWidth / mediaAspect;
      imageX = 0;
      imageY = (displayHeight - imageHeight) / 2;
    }
  }

  // Convert points from canvas coordinates to image-local coordinates
  const normalizedPoints = points.map((point) => {
    // First, translate to image-local coordinates
    const localX = point.x - imageX;
    const localY = point.y - imageY;

    // Then scale to media dimensions
    const mediaX = (localX / imageWidth) * mediaWidth;
    const mediaY = (localY / imageHeight) * mediaHeight;

    return { x: mediaX, y: mediaY };
  });

  // Filter out points outside media bounds if requested
  if (filterOutOfBounds) {
    return normalizedPoints.filter(
      (point) =>
        point.x >= 0 &&
        point.x <= mediaWidth &&
        point.y >= 0 &&
        point.y <= mediaHeight,
    );
  }

  return normalizedPoints;
}

// Helper function to normalize points AND their labels together
// This ensures labels stay in sync when points are filtered
export function normalizePointsWithLabels(
  points: Array<{ x: number; y: number }>,
  labels: Array<number>,
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  filterOutOfBounds: boolean = true,
  clipTransform?: ClipTransform,
): { points: Array<{ x: number; y: number }>; labels: Array<number> } {
  const mediaWidth =
    mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight =
    mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;

  if (!mediaWidth || !mediaHeight) {
    console.warn("Media dimensions not available, using display dimensions");
    return { points, labels };
  }

  // If clipTransform is provided, use it directly
  // Otherwise calculate aspect-fit dimensions within the rect
  let imageX, imageY, imageWidth, imageHeight;

  if (clipTransform) {
    imageX = clipTransform.x;
    imageY = clipTransform.y;
    imageWidth = clipTransform.width * (clipTransform.scaleX || 1);
    imageHeight = clipTransform.height * (clipTransform.scaleY || 1);
    if (clipTransform.crop) {
      imageWidth = imageWidth / (clipTransform.crop.width || 1);
      imageHeight = imageHeight / (clipTransform.crop.height || 1);
      let imageXOffset = clipTransform.crop.x * imageWidth;
      let imageYOffset = clipTransform.crop.y * imageHeight;
      imageX -= imageXOffset;
      imageY -= imageYOffset;
    }
  } else {
    // Calculate aspect-fit dimensions (letterbox/pillarbox)
    const mediaAspect = mediaWidth / mediaHeight;
    const rectAspect = displayWidth / displayHeight;

    if (rectAspect > mediaAspect) {
      // Letterbox (black bars on left/right)
      imageHeight = displayHeight;
      imageWidth = displayHeight * mediaAspect;
      imageX = (displayWidth - imageWidth) / 2;
      imageY = 0;
    } else {
      // Pillarbox (black bars on top/bottom)
      imageWidth = displayWidth;
      imageHeight = displayWidth / mediaAspect;
      imageX = 0;
      imageY = (displayHeight - imageHeight) / 2;
    }
  }

  // Convert points from canvas coordinates to image-local coordinates
  // and track which ones are in bounds
  const normalizedData: Array<{
    point: { x: number; y: number };
    label: number;
    inBounds: boolean;
  }> = points.map((point, i) => {
    // First, translate to image-local coordinates
    const localX = point.x - imageX;
    const localY = point.y - imageY;

    // Then scale to media dimensions
    const mediaX = (localX / imageWidth) * mediaWidth;
    const mediaY = (localY / imageHeight) * mediaHeight;

    const normalizedPoint = { x: mediaX, y: mediaY };
    const inBounds =
      mediaX >= 0 &&
      mediaX <= mediaWidth &&
      mediaY >= 0 &&
      mediaY <= mediaHeight;

    return {
      point: normalizedPoint,
      label: labels[i] !== undefined ? labels[i] : 1, // Default to positive if label missing
      inBounds,
    };
  });

  // Filter out points outside media bounds if requested
  if (filterOutOfBounds) {
    const filtered = normalizedData.filter((item) => item.inBounds);
    return {
      points: filtered.map((item) => item.point),
      labels: filtered.map((item) => item.label),
    };
  }

  return {
    points: normalizedData.map((item) => item.point),
    labels: normalizedData.map((item) => item.label),
  };
}

// Helper function to normalize box from display coordinates to media coordinates
// This accounts for aspect-fit letterboxing and user transforms
export function normalizeBox(
  box: { x1: number; y1: number; x2: number; y2: number },
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  clipTransform?: ClipTransform,
): { x1: number; y1: number; x2: number; y2: number } {
  const mediaWidth =
    mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight =
    mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;

  if (!mediaWidth || !mediaHeight) {
    console.warn("Media dimensions not available, using display dimensions");
    return box;
  }

  // If clipTransform is provided, use it directly
  // Otherwise calculate aspect-fit dimensions within the rect
  let imageX, imageY, imageWidth, imageHeight;

  if (clipTransform) {
    imageX = clipTransform.x;
    imageY = clipTransform.y;
    imageWidth = clipTransform.width * (clipTransform.scaleX || 1);
    imageHeight = clipTransform.height * (clipTransform.scaleY || 1);
  } else {
    // Calculate aspect-fit dimensions (letterbox/pillarbox)
    const mediaAspect = mediaWidth / mediaHeight;
    const rectAspect = displayWidth / displayHeight;

    if (rectAspect > mediaAspect) {
      // Letterbox (black bars on left/right)
      imageHeight = displayHeight;
      imageWidth = displayHeight * mediaAspect;
      imageX = (displayWidth - imageWidth) / 2;
      imageY = 0;
    } else {
      // Pillarbox (black bars on top/bottom)
      imageWidth = displayWidth;
      imageHeight = displayWidth / mediaAspect;
      imageX = 0;
      imageY = (displayHeight - imageHeight) / 2;
    }
  }

  // Convert box corners from canvas coordinates to image-local coordinates
  const local_x1 = box.x1 - imageX;
  const local_y1 = box.y1 - imageY;
  const local_x2 = box.x2 - imageX;
  const local_y2 = box.y2 - imageY;

  // Scale to media dimensions
  const media_x1 = (local_x1 / imageWidth) * mediaWidth;
  const media_y1 = (local_y1 / imageHeight) * mediaHeight;
  const media_x2 = (local_x2 / imageWidth) * mediaWidth;
  const media_y2 = (local_y2 / imageHeight) * mediaHeight;

  return {
    x1: Math.max(0, Math.min(media_x1, media_x2)),
    y1: Math.max(0, Math.min(media_y1, media_y2)),
    x2: Math.min(mediaWidth, Math.max(media_x1, media_x2)),
    y2: Math.min(mediaHeight, Math.max(media_y1, media_y2)),
  };
}

// Helper function to denormalize contours from media coordinates to display coordinates
// This accounts for aspect-fit letterboxing and user transforms
export function denormalizeContours(
  contours: Array<Array<number>>,
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  clipTransform?: ClipTransform,
): Array<Array<number>> {
  const mediaWidth =
    mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight =
    mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;

  if (!mediaWidth || !mediaHeight) {
    return contours;
  }

  // If clipTransform is provided, use it directly
  // Otherwise calculate aspect-fit dimensions within the rect
  let imageX, imageY, imageWidth, imageHeight;

  if (clipTransform) {
    imageX = clipTransform.x;
    imageY = clipTransform.y;
    imageWidth = clipTransform.width * (clipTransform.scaleX || 1);
    imageHeight = clipTransform.height * (clipTransform.scaleY || 1);
    if (clipTransform.crop) {
      imageWidth = imageWidth / (clipTransform.crop.width || 1);
      imageHeight = imageHeight / (clipTransform.crop.height || 1);
      let imageXOffset = clipTransform.crop.x * imageWidth;
      let imageYOffset = clipTransform.crop.y * imageHeight;
      imageX -= imageXOffset;
      imageY -= imageYOffset;
    }
  } else {
    // Calculate aspect-fit dimensions (letterbox/pillarbox)
    const mediaAspect = mediaWidth / mediaHeight;
    const rectAspect = displayWidth / displayHeight;

    if (rectAspect > mediaAspect) {
      // Letterbox (black bars on left/right)
      imageHeight = displayHeight;
      imageWidth = displayHeight * mediaAspect;
      imageX = (displayWidth - imageWidth) / 2;
      imageY = 0;
    } else {
      // Pillarbox (black bars on top/bottom)
      imageWidth = displayWidth;
      imageHeight = displayWidth / mediaAspect;
      imageX = 0;
      imageY = (displayHeight - imageHeight) / 2;
    }
  }

  return contours.map((contour) => {
    const denormalized: number[] = [];
    for (let i = 0; i < contour.length; i += 2) {
      // Scale from media to image-local coordinates
      const localX = (contour[i] / mediaWidth) * imageWidth;
      const localY = (contour[i + 1] / mediaHeight) * imageHeight;

      const canvasX = localX + imageX;
      const canvasY = localY + imageY;

      denormalized.push(canvasX, canvasY);
    }
    return denormalized;
  });
}

// Helper to denormalize a shape bounds dict from media coordinates to display coordinates
export function denormalizeShapeBounds(
  bounds: {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation?: number;
    shapeType?: string;
    scaleX?: number;
    scaleY?: number;
  },
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  clipTransform?: ClipTransform,
): {
  x: number;
  y: number;
  width: number;
  height: number;
  rotation?: number;
  shapeType?: string;
  scaleX?: number;
  scaleY?: number;
} {
  const mediaWidth =
    mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight =
    mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;

  if (!mediaWidth || !mediaHeight) return bounds;

  // If rectangle comes from center-based bounds (backend), convert to top-left pivot
  // so WebGL rendering (which rotates around top-left) aligns with the intended shape.
  let pivotBounds = bounds;
  if (bounds.shapeType === "rectangle" && typeof bounds.rotation === "number") {
    const cx = bounds.x + bounds.width / 2;
    const cy = bounds.y + bounds.height / 2;
    const w = Math.max(1, bounds.width);
    const h = Math.max(1, bounds.height);
    const theta = (bounds.rotation * Math.PI) / 180;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    // Pivot is the rotated position of the unrotated top-left corner (-w/2, -h/2)
    const pivotX = cx - (w / 2) * c + (h / 2) * s;
    const pivotY = cy - (w / 2) * s - (h / 2) * c;
    pivotBounds = {
      x: pivotX,
      y: pivotY,
      width: w,
      height: h,
      rotation: bounds.rotation,
      shapeType: bounds.shapeType,
      scaleX: bounds.scaleX,
      scaleY: bounds.scaleY,
    };
  }

  let imageX: number, imageY: number, imageWidth: number, imageHeight: number;
  if (clipTransform) {
    imageX = clipTransform.x;
    imageY = clipTransform.y;
    imageWidth = clipTransform.width * (clipTransform.scaleX || 1);
    imageHeight = clipTransform.height * (clipTransform.scaleY || 1);
  } else {
    const mediaAspect = mediaWidth / mediaHeight;
    const rectAspect = displayWidth / displayHeight;
    if (rectAspect > mediaAspect) {
      imageHeight = displayHeight;
      imageWidth = displayHeight * mediaAspect;
      imageX = (displayWidth - imageWidth) / 2;
      imageY = 0;
    } else {
      imageWidth = displayWidth;
      imageHeight = displayWidth / mediaAspect;
      imageX = 0;
      imageY = (displayHeight - imageHeight) / 2;
    }
  }

  // Center-based shapes: ellipse, polygon (triangle), star
  const type = (bounds.shapeType || "").toLowerCase();
  const isCenterShape =
    type === "ellipse" ||
    type === "polygon" ||
    type === "triangle" ||
    type === "star";

  if (isCenterShape) {
    // Scale center position
    const localCX = (bounds.x / mediaWidth) * imageWidth + imageX;
    const localCY = (bounds.y / mediaHeight) * imageHeight + imageY;
    // Scale size from media to image
    let localW = (bounds.width / mediaWidth) * imageWidth;
    let localH = (bounds.height / mediaHeight) * imageHeight;

    if (type === "star") {
      // Square bounds for star
      const s = Math.max(1, Math.min(localW, localH));
      localW = s;
      localH = s;
    } else if (type === "polygon" || type === "triangle") {
      // Enforce width/height ratio for triangle
      const ratio = 1.1543665517482078; // width / height
      // Fit the largest rect with this ratio inside the given box
      const maxH = Math.max(1, Math.min(localH, localW / ratio));
      const maxW = Math.max(1, ratio * maxH);
      localW = maxW;
      localH = maxH;
    }

    // Return top-left pivot for consistency with shader, which computes center from x,y + 0.5*width/height
    return {
      x: localCX - localW / 2,
      y: localCY - localH / 2,
      width: localW,
      height: localH,
      rotation: bounds.rotation,
      shapeType: bounds.shapeType,
      scaleX: bounds.scaleX,
      scaleY: bounds.scaleY,
    };
  }

  // Rectangle (top-left pivot semantics)
  const localX = (pivotBounds.x / mediaWidth) * imageWidth;
  const localY = (pivotBounds.y / mediaHeight) * imageHeight;
  const localW = (pivotBounds.width / mediaWidth) * imageWidth;
  const localH = (pivotBounds.height / mediaHeight) * imageHeight;

  const canvasX = localX + imageX;
  const canvasY = localY + imageY;

  return {
    x: canvasX,
    y: canvasY,
    width: localW,
    height: localH,
    rotation: pivotBounds.rotation,
    shapeType: pivotBounds.shapeType,
    scaleX: pivotBounds.scaleX,
    scaleY: pivotBounds.scaleY,
  };
}

// Hook-based API for automatic mask creation
export interface UseMaskOptions {
  id?: string;
  inputPath: string;
  frameNumber?: number;
  tool: string;
  points?: number[] | Array<{ x: number; y: number }>;
  pointLabels?: Array<number>;
  box?: { x1: number; y1: number; x2: number; y2: number };
  displayWidth?: number;
  displayHeight?: number;
  mediaInfo?: MediaInfo;
  clipTransform?: ClipTransform;
  multimaskOutput?: boolean;
  simplifyTolerance?: number;
  enabled?: boolean;
  debounceMs?: number;
}

export interface UseMaskResult {
  data: MaskResponse | null;
  error: string | null;
  loading: boolean;
  refetch: () => Promise<void>;
}

export function useMask(options: UseMaskOptions): UseMaskResult {
  const {
    id,
    inputPath,
    frameNumber,
    tool,
    points,
    pointLabels,
    box,
    displayWidth,
    displayHeight,
    mediaInfo,
    clipTransform,
    multimaskOutput = true,
    simplifyTolerance = 1.0,
    enabled = true,
    debounceMs = 250,
  } = options;

  const [data, setData] = useState<MaskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchMask = useCallback(async () => {
    if (!enabled || !inputPath || !tool) {
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    let cacheKey: string | null = null;
    let controller: AbortController | null = null;

    try {
      // Convert and normalize points (and labels if provided)
      let normalizedPoints: Array<{ x: number; y: number }> | undefined;
      let normalizedLabels: Array<number> | undefined;

      if (points && points.length > 0) {
        // Convert flat array to point objects if needed
        const pointObjects =
          Object.prototype.hasOwnProperty.call(points[0], "x") &&
          Object.prototype.hasOwnProperty.call(points[0], "y")
            ? (points as Array<{ x: number; y: number }>)
            : flatArrayToPoints(points as number[]);

        // Normalize if mediaInfo and display dimensions are provided
        if (mediaInfo && displayWidth && displayHeight) {
          // If we have labels, normalize points and labels together to keep them in sync
          if (pointLabels && pointLabels.length > 0) {
            const result = normalizePointsWithLabels(
              pointObjects,
              pointLabels,
              displayWidth,
              displayHeight,
              mediaInfo,
              true,
              clipTransform,
            );
            normalizedPoints = result.points;
            normalizedLabels = result.labels;
          } else {
            // No labels, just normalize points
            normalizedPoints = normalizePoints(
              pointObjects,
              displayWidth,
              displayHeight,
              mediaInfo,
              true,
              clipTransform,
            );
          }

          // If all points were filtered out, don't make the request
          if (normalizedPoints.length === 0 && pointObjects.length > 0) {
            setError("All points are outside the media bounds");
            setLoading(false);
            return;
          }
        } else {
          normalizedPoints = pointObjects;
          normalizedLabels = pointLabels;
        }
      }

      // Normalize box if provided
      let normalizedBox:
        | { x1: number; y1: number; x2: number; y2: number }
        | undefined;
      if (box && mediaInfo && displayWidth && displayHeight) {
        normalizedBox = normalizeBox(
          box,
          displayWidth,
          displayHeight,
          mediaInfo,
          clipTransform,
        );
      } else {
        normalizedBox = box;
      }

      const request: MaskRequest = {
        id: id || undefined,
        input_path: inputPath,
        frame_number: frameNumber,
        tool,
        points: normalizedPoints,
        point_labels: normalizedLabels || pointLabels, // Use normalized labels if available
        box: normalizedBox,
        multimask_output: multimaskOutput,
        simplify_tolerance: simplifyTolerance,
      };

      cacheKey = createMaskCacheKey(request);

      const cachedResponse = maskCache.get(cacheKey);
      if (cachedResponse) {
        setData(cachedResponse);
        setError(null);
        setLoading(false);
        return;
      }

      if (abortControllerRef.current) {
        // @ts-ignore
        abortControllerRef.current.abort();
      }

      controller = new AbortController();
      abortControllerRef.current = controller;

      setLoading(true);
      setError(null);

      const response = await createMaskPreload(request);

      if (controller?.signal.aborted) {
        return;
      }

      if (response.success && response.data) {
        // Denormalize contours back to display coordinates if needed
        let finalData = response.data;
        if (finalData.contours && mediaInfo && displayWidth && displayHeight) {
          finalData = {
            ...finalData,
            contours: denormalizeContours(
              finalData.contours,
              displayWidth,
              displayHeight,
              mediaInfo,
              clipTransform,
            ),
          };
        }
        if (cacheKey) {
          maskCache.set(cacheKey, finalData);
        }
        setData(finalData);
        setError(null);
      } else {
        const errorMessage = response.error || "Failed to create mask";
        setError(errorMessage);
        setData(null);
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error occurred";
        setError(errorMessage);
        setData(null);
      }
    } finally {
      setLoading(false);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }, [
    enabled,
    inputPath,
    frameNumber,
    tool,
    points,
    pointLabels,
    box,
    displayWidth,
    displayHeight,
    mediaInfo,
    clipTransform,
    multimaskOutput,
    simplifyTolerance,
  ]);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    // Clear existing debounce timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Set new debounce timer
    debounceTimerRef.current = setTimeout(() => {
      fetchMask();
    }, debounceMs);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchMask, debounceMs, enabled]);

  return {
    data,
    error,
    loading,
    refetch: fetchMask,
  };
}

// Legacy function for backward compatibility
export async function createMask(
  request: MaskRequest,
): Promise<ConfigResponse<MaskResponse>> {
  try {
    const { getBackendIsRemote, getFileShouldUpload } =
      await import("@app/preload");
    const remoteRes = await getBackendIsRemote();
    const isRemote = !!(
      remoteRes &&
      remoteRes.success &&
      remoteRes.data?.isRemote
    );
    if (isRemote) {
      const su = await getFileShouldUpload(String(request.input_path || ""));
      const shouldUpload = !!(su && su.success && su.data?.shouldUpload);
      if (shouldUpload) {
        toast.info("Uploading source media for mask…");
      }
    }
  } catch {}
  return await createMaskPreload(request);
}

export interface StreamedMaskFrame {
  stream_id: string;
  frame_number: number;
  contours: Array<Array<number>>;
}

// In-memory registry of active mask tracking sessions (renderer process scope)
type MaskTrackSession = {
  streamId: string;
  requestMeta: {
    id: string;
    frame_start: number;
    anchor_frame?: number;
    frame_end: number;
    direction?: "forward" | "backward" | "both";
    max_frames?: number;
  };
  seenFrames: Set<number>;
  totalFrames: number;
  ended: boolean;
};

const activeMaskTrackSessions: Map<string, MaskTrackSession> = new Map(); // key: mask id

export function getActiveMaskTrackSession(
  maskId: string,
): MaskTrackSession | undefined {
  return activeMaskTrackSessions.get(maskId);
}

export function attachToMaskTrack(
  maskId: string,
  options?: {
    onProgress?: (progress: number) => void;
    onFrame?: (frame: StreamedMaskFrame) => void;
    onEnd?: () => void;
  },
): (() => void) | null {
  const session = activeMaskTrackSessions.get(maskId);
  if (!session || session.ended) return null;
  const { streamId } = session;
  const { onProgress, onFrame, onEnd } = options || {};

  const offChunk = onMaskTrackChunk(streamId, (evt: any) => {
    if (evt && typeof evt.frame_number === "number") {
      session.seenFrames.add(evt.frame_number);
      if (onFrame)
        onFrame({
          frame_number: evt.frame_number,
          contours: Array.isArray(evt.contours) ? evt.contours : [],
          stream_id: streamId,
        });
      if (onProgress && session.totalFrames > 0) {
        const progress = Math.min(
          1,
          session.seenFrames.size / session.totalFrames,
        );
        onProgress(progress);
      }
    }
  });
  const offError = onMaskTrackError(streamId, (_err: any) => {
    // keep session for possible retries, but mark ended if main tears down
  });
  const offEnd = onMaskTrackEnd(streamId, () => {
    session.ended = true;
    activeMaskTrackSessions.delete(maskId);
    offChunk();
    offError();
    offEnd();
    if (onEnd) onEnd();
  });

  // Provide detach to unsubscribe without affecting the underlying stream
  return () => {
    try {
      offChunk();
    } catch {}
    try {
      offError();
    } catch {}
    try {
      offEnd();
    } catch {}
  };
}

export async function trackMask(
  request: {
    id: string;
    input_path: string;
    frame_start: number;
    anchor_frame: number;
    frame_end: number;
    direction?: "forward" | "backward" | "both";
    model_type?: string;
    max_frames?: number;
    // Optional seed inputs so the backend can (re)create the anchor mask before tracking.
    // This is important when `/system/free-memory` has cleared SAM2 state.
    tool?: "touch" | "lasso" | "shape";
    points?: Array<{ x: number; y: number }>;
    point_labels?: Array<number>;
    box?: { x1: number; y1: number; x2: number; y2: number };
    simplify_tolerance?: number;
    shape_type?: string;
  },
  options?: {
    signal?: AbortSignal;
    onProgress?: (progress: number) => void;
    onFrame?: (frame: StreamedMaskFrame) => void;
  },
): Promise<void> {
  const { signal, onProgress, onFrame } = options || {};
  // Inform user about potential upload delay
  try {
    const { getBackendIsRemote, getFileShouldUpload } =
      await import("@app/preload");
    const remoteRes = await getBackendIsRemote();
    const isRemote = !!(
      remoteRes &&
      remoteRes.success &&
      remoteRes.data?.isRemote
    );
    if (isRemote) {
      const su = await getFileShouldUpload(String(request.input_path || ""));
      const shouldUpload = !!(su && su.success && su.data?.shouldUpload);
      if (shouldUpload) {
        toast.info("Uploading source media for mask tracking…");
      }
    }
  } catch {}

  const { streamId } = await startMaskTrack(request as any);
  const anchor = request.anchor_frame ?? request.frame_start;
  const low = Math.min(request.frame_start, request.frame_end);
  const high = Math.max(request.frame_start, request.frame_end);
  let totalFrames: number;
  if (request.direction === "both") {
    let lowerBound = low;
    let upperBound = high;
    if (request.max_frames && request.max_frames > 0) {
      lowerBound = Math.max(low, anchor - request.max_frames);
      upperBound = Math.min(high, anchor + request.max_frames);
    }
    totalFrames = Math.max(1, upperBound - lowerBound + 1);
  } else {
    const rawSpan = Math.abs(request.frame_end - anchor);
    const span =
      request.max_frames && request.max_frames > 0
        ? Math.min(rawSpan, request.max_frames)
        : rawSpan;
    totalFrames = span + 1;
  }
  const seenFrames = new Set<number>();

  // Register session for reconnection
  activeMaskTrackSessions.set(request.id, {
    streamId,
    requestMeta: {
      id: request.id,
      frame_start: request.frame_start,
      anchor_frame: request.anchor_frame,
      frame_end: request.frame_end,
      direction: request.direction,
      max_frames: request.max_frames,
    },
    seenFrames,
    totalFrames,
    ended: false,
  });

  const abortHandler = () => {
    try {
      cancelMaskTrackPreload(streamId);
    } catch {}
  };
  if (signal) {
    if (signal.aborted) {
      abortHandler();
      throw new DOMException("Aborted", "AbortError");
    }
    signal.addEventListener("abort", abortHandler, { once: true });
  }

  try {
    await new Promise<void>((resolve, reject) => {
      const offChunk = onMaskTrackChunk(streamId, (evt: any) => {
        if (evt && evt.status === "error") {
          offChunk();
          offError();
          offEnd();
          reject(new Error(evt.error || "Mask tracking error"));
          return;
        }
        if (evt && typeof evt.frame_number === "number") {
          if (onFrame) {
            onFrame({
              frame_number: evt.frame_number,
              contours: Array.isArray(evt.contours) ? evt.contours : [],
              stream_id: streamId,
            });
          }
          if (!seenFrames.has(evt.frame_number)) {
            seenFrames.add(evt.frame_number);
            if (onProgress && totalFrames > 0) {
              const progress = Math.min(1, seenFrames.size / totalFrames);
              onProgress(progress);
            }
          }
        }
      });
      const offError = onMaskTrackError(streamId, (err: any) => {
        offChunk();
        offError();
        offEnd();
        // Mark session ended on error
        const sess = activeMaskTrackSessions.get(request.id);
        if (sess) {
          sess.ended = true;
          activeMaskTrackSessions.delete(request.id);
        }
        reject(new Error(err?.message || "Mask tracking error"));
      });
      const offEnd = onMaskTrackEnd(streamId, () => {
        offChunk();
        offError();
        offEnd();
        const sess = activeMaskTrackSessions.get(request.id);
        if (sess) {
          sess.ended = true;
          activeMaskTrackSessions.delete(request.id);
        }
        resolve();
      });
    });

    // Ensure we report completion
    if (onProgress && totalFrames > 0) {
      onProgress(1);
    }
  } finally {
    if (signal) {
      signal.removeEventListener("abort", abortHandler);
    }
  }
}

export async function cancelMaskTrack(streamId: string): Promise<void> {
  await cancelMaskTrackPreload(streamId);
}

export default trackMask;

export interface StreamedShapeFrame {
  stream_id: string;
  frame_number: number;
  shapeBounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation?: number;
    shapeType?: string;
    scaleX?: number;
    scaleY?: number;
  };
}

export function attachToShapeTrack(
  maskId: string,
  options?: {
    onProgress?: (progress: number) => void;
    onFrame?: (frame: StreamedShapeFrame) => void;
    onEnd?: () => void;
  },
): (() => void) | null {
  const session = activeMaskTrackSessions.get(maskId);
  if (!session || session.ended) return null;
  const { streamId } = session;
  const { onProgress, onFrame, onEnd } = options || {};

  const offChunk = onMaskTrackShapesChunk(streamId, (evt: any) => {
    if (evt && typeof evt.frame_number === "number") {
      session.seenFrames.add(evt.frame_number);
      if (onFrame)
        onFrame({
          frame_number: evt.frame_number,
          shapeBounds: evt.shapeBounds,
          stream_id: streamId,
        });
      if (onProgress && session.totalFrames > 0) {
        const progress = Math.min(
          1,
          session.seenFrames.size / session.totalFrames,
        );
        onProgress(progress);
      }
    }
  });
  const offError = onMaskTrackShapesError(streamId, (_err: any) => {});
  const offEnd = onMaskTrackShapesEnd(streamId, () => {
    session.ended = true;
    activeMaskTrackSessions.delete(maskId);
    offChunk();
    offError();
    offEnd();
    if (onEnd) onEnd();
  });

  return () => {
    try {
      offChunk();
    } catch {}
    try {
      offError();
    } catch {}
    try {
      offEnd();
    } catch {}
  };
}

export async function trackShapes(
  request: {
    id: string;
    input_path: string;
    frame_start: number;
    anchor_frame: number;
    frame_end: number;
    direction?: "forward" | "backward" | "both";
    model_type?: string;
    max_frames?: number;
    shape_type?: string;
  },
  options?: {
    signal?: AbortSignal;
    onProgress?: (progress: number) => void;
    onFrame?: (frame: StreamedShapeFrame) => void;
  },
): Promise<void> {
  const { signal, onProgress, onFrame } = options || {};
  // Inform user about potential upload delay
  try {
    const { getBackendIsRemote, getFileShouldUpload } =
      await import("@app/preload");
    const remoteRes = await getBackendIsRemote();
    const isRemote = !!(
      remoteRes &&
      remoteRes.success &&
      remoteRes.data?.isRemote
    );
    if (isRemote) {
      const su = await getFileShouldUpload(String(request.input_path || ""));
      const shouldUpload = !!(su && su.success && su.data?.shouldUpload);
      if (shouldUpload) {
        toast.info("Uploading source media for mask tracking…");
      }
    }
  } catch {}

  const { streamId } = await startMaskTrackShapes(request);
  const anchor = request.anchor_frame ?? request.frame_start;
  const low = Math.min(request.frame_start, request.frame_end);
  const high = Math.max(request.frame_start, request.frame_end);
  let totalFrames: number;
  if (request.direction === "both") {
    let lowerBound = low;
    let upperBound = high;
    if (request.max_frames && request.max_frames > 0) {
      lowerBound = Math.max(low, anchor - request.max_frames);
      upperBound = Math.min(high, anchor + request.max_frames);
    }
    totalFrames = Math.max(1, upperBound - lowerBound + 1);
  } else {
    const rawSpan = Math.abs(request.frame_end - anchor);
    const span =
      request.max_frames && request.max_frames > 0
        ? Math.min(rawSpan, request.max_frames)
        : rawSpan;
    totalFrames = span + 1;
  }
  const seenFrames = new Set<number>();

  activeMaskTrackSessions.set(request.id, {
    streamId,
    requestMeta: {
      id: request.id,
      frame_start: request.frame_start,
      anchor_frame: request.anchor_frame,
      frame_end: request.frame_end,
      direction: request.direction,
      max_frames: request.max_frames,
    },
    seenFrames,
    totalFrames,
    ended: false,
  });

  const abortHandler = () => {
    try {
      cancelMaskTrackPreload(streamId);
    } catch {}
  };
  if (signal) {
    if (signal.aborted) {
      abortHandler();
      throw new DOMException("Aborted", "AbortError");
    }
    signal.addEventListener("abort", abortHandler, { once: true });
  }

  try {
    await new Promise<void>((resolve, reject) => {
      const offChunk = onMaskTrackShapesChunk(streamId, (evt: any) => {
        if (evt && evt.status === "error") {
          offChunk();
          offError();
          offEnd();
          reject(new Error(evt.error || "Shape tracking error"));
          return;
        }
        if (evt && typeof evt.frame_number === "number") {
          if (onFrame) {
            onFrame({
              frame_number: evt.frame_number,
              shapeBounds: evt.shapeBounds,
              stream_id: streamId,
            });
          }
          if (!seenFrames.has(evt.frame_number)) {
            seenFrames.add(evt.frame_number);
            if (onProgress && totalFrames > 0) {
              const progress = Math.min(1, seenFrames.size / totalFrames);
              onProgress(progress);
            }
          }
        }
      });
      const offError = onMaskTrackShapesError(streamId, (err: any) => {
        offChunk();
        offError();
        offEnd();
        const sess = activeMaskTrackSessions.get(request.id);
        if (sess) {
          sess.ended = true;
          activeMaskTrackSessions.delete(request.id);
        }
        reject(new Error(err?.message || "Shape tracking error"));
      });
      const offEnd = onMaskTrackShapesEnd(streamId, () => {
        offChunk();
        offError();
        offEnd();
        const sess = activeMaskTrackSessions.get(request.id);
        if (sess) {
          sess.ended = true;
          activeMaskTrackSessions.delete(request.id);
        }
        resolve();
      });
    });

    if (onProgress && totalFrames > 0) onProgress(1);
  } finally {
    if (signal) signal.removeEventListener("abort", abortHandler);
  }
}

// Convenience workflow: create mask first, then start tracking when successful
export async function createThenTrackMask(
  args: {
    create: MaskRequest; // lasso or shape; points/box should already be normalized if needed
    track: {
      id: string;
      input_path: string;
      frame_start: number;
      anchor_frame: number;
      frame_end: number;
      direction?: "forward" | "backward" | "both";
      model_type?: string;
      max_frames?: number;
    };
  },
  options?: {
    signal?: AbortSignal;
    onProgress?: (progress: number) => void;
    onFrame?: (frame: StreamedMaskFrame) => void;
  },
): Promise<void> {
  const { create, track } = args;

  // Step 1: create the initial mask (lasso: polygon; shape: box)
  const createResp = await createMask(create);

  if (
    !createResp.success ||
    !createResp.data ||
    createResp.data.status !== "success"
  ) {
    const msg =
      createResp.error || createResp.data?.message || "Mask creation failed";
    throw new Error(msg);
  }

  // Step 2: start streaming tracking updates
  await trackMask(
    {
      id: track.id,
      input_path: track.input_path,
      frame_start: track.frame_start,
      anchor_frame: track.anchor_frame,
      frame_end: track.frame_end,
      direction: track.direction,
      model_type: track.model_type,
      max_frames: track.max_frames,
    },
    {
      signal: options?.signal,
      onProgress: options?.onProgress,
      onFrame: options?.onFrame,
    },
  );
}
