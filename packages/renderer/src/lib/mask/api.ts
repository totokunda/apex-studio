import {
  createMask as createMaskPreload,
  startMaskTrack,
  onMaskTrackChunk,
  onMaskTrackError,
  onMaskTrackEnd,
  cancelMaskTrack as cancelMaskTrackPreload,
} from '@app/preload';
import {useEffect, useRef, useState, useCallback} from 'react';
import type {MediaInfo} from '../types';

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
  points?: Array<{x: number, y: number}>;
  point_labels?: Array<number>;
  box?: {x1: number, y1: number, x2: number, y2: number};
  multimask_output?: boolean;
  simplify_tolerance?: number;
  model_type?: string;
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
    if (input === null || typeof input !== 'object') {
      return input;
    }

    const typedInput = input as object;

    if (seen.has(typedInput)) {
      return;
    }

    seen.add(typedInput);

    if (Array.isArray(input)) {
      return input.map(item => helper(item));
    }

    const result: Record<string, unknown> = {};
    const record = input as Record<string, unknown>;
    for (const key of Object.keys(record).sort()) {
      const nestedValue = record[key];
      if (typeof nestedValue === 'undefined') {
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
export function flatArrayToPoints(flatArray: number[]): Array<{x: number, y: number}> {
  const points: Array<{x: number, y: number}> = [];
  for (let i = 0; i < flatArray.length; i += 2) {
    if (i + 1 < flatArray.length) {
      points.push({x: flatArray[i], y: flatArray[i + 1]});
    }
  }
  return points;
}

// Helper function to convert array of {x, y} objects to flat array [x1, y1, x2, y2, ...]
export function pointsToFlatArray(points: Array<{x: number, y: number}>): number[] {
  const flatArray: number[] = [];
  for (const point of points) {
    flatArray.push(point.x, point.y);
  }
  return flatArray;
}

// Helper function to normalize points from display coordinates to media coordinates
// This accounts for aspect-fit letterboxing and user transforms
export function normalizePoints(
  points: Array<{x: number, y: number}>,
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  filterOutOfBounds: boolean = true,
  clipTransform?: {x: number, y: number, width: number, height: number, scaleX?: number, scaleY?: number, rotation?: number}
): Array<{x: number, y: number}> {
  
  const mediaWidth = mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight = mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;
  
  if (!mediaWidth || !mediaHeight) {
    console.warn('Media dimensions not available, using display dimensions');
    return points;
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
  
  // Convert points from canvas coordinates to image-local coordinates
  const normalizedPoints = points.map(point => {
    // First, translate to image-local coordinates
    const localX = point.x - imageX;
    const localY = point.y - imageY;
    
    // Then scale to media dimensions
    const mediaX = (localX / imageWidth) * mediaWidth;
    const mediaY = (localY / imageHeight) * mediaHeight;
    
    return {x: mediaX, y: mediaY};
  });
  
  // Filter out points outside media bounds if requested
  if (filterOutOfBounds) {
    return normalizedPoints.filter(point => 
      point.x >= 0 && point.x <= mediaWidth &&
      point.y >= 0 && point.y <= mediaHeight
    );
  }
  
  return normalizedPoints;
}

// Helper function to normalize points AND their labels together
// This ensures labels stay in sync when points are filtered
export function normalizePointsWithLabels(
  points: Array<{x: number, y: number}>,
  labels: Array<number>,
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  filterOutOfBounds: boolean = true,
  clipTransform?: {x: number, y: number, width: number, height: number, scaleX?: number, scaleY?: number, rotation?: number}
): {points: Array<{x: number, y: number}>, labels: Array<number>} {
  
  const mediaWidth = mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight = mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;
  
  if (!mediaWidth || !mediaHeight) {
    console.warn('Media dimensions not available, using display dimensions');
    return {points, labels};
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
  
  // Convert points from canvas coordinates to image-local coordinates
  // and track which ones are in bounds
  const normalizedData: Array<{point: {x: number, y: number}, label: number, inBounds: boolean}> = points.map((point, i) => {
    // First, translate to image-local coordinates
    const localX = point.x - imageX;
    const localY = point.y - imageY;
    
    // Then scale to media dimensions
    const mediaX = (localX / imageWidth) * mediaWidth;
    const mediaY = (localY / imageHeight) * mediaHeight;
    
    const normalizedPoint = {x: mediaX, y: mediaY};
    const inBounds = mediaX >= 0 && mediaX <= mediaWidth && mediaY >= 0 && mediaY <= mediaHeight;
    
    return {
      point: normalizedPoint,
      label: labels[i] !== undefined ? labels[i] : 1, // Default to positive if label missing
      inBounds
    };
  });
  
  // Filter out points outside media bounds if requested
  if (filterOutOfBounds) {
    const filtered = normalizedData.filter(item => item.inBounds);
    return {
      points: filtered.map(item => item.point),
      labels: filtered.map(item => item.label)
    };
  }
  
  return {
    points: normalizedData.map(item => item.point),
    labels: normalizedData.map(item => item.label)
  };
}

// Helper function to normalize box from display coordinates to media coordinates
// This accounts for aspect-fit letterboxing and user transforms
export function normalizeBox(
  box: {x1: number, y1: number, x2: number, y2: number},
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  clipTransform?: {x: number, y: number, width: number, height: number, scaleX?: number, scaleY?: number, rotation?: number}
): {x1: number, y1: number, x2: number, y2: number} {
  const mediaWidth = mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight = mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;
  
  if (!mediaWidth || !mediaHeight) {
    console.warn('Media dimensions not available, using display dimensions');
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
    y2: Math.min(mediaHeight, Math.max(media_y1, media_y2))
  };
}

// Helper function to denormalize contours from media coordinates to display coordinates
// This accounts for aspect-fit letterboxing and user transforms
export function denormalizeContours(
  contours: Array<Array<number>>,
  displayWidth: number,
  displayHeight: number,
  mediaInfo: MediaInfo,
  clipTransform?: {x: number, y: number, width: number, height: number, scaleX?: number, scaleY?: number, rotation?: number}
): Array<Array<number>> {
  const mediaWidth = mediaInfo.video?.displayWidth || mediaInfo.image?.width || displayWidth;
  const mediaHeight = mediaInfo.video?.displayHeight || mediaInfo.image?.height || displayHeight;
  
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
  
  return contours.map(contour => {
    const denormalized: number[] = [];
    for (let i = 0; i < contour.length; i += 2) {
      // Scale from media to image-local coordinates
      const localX = (contour[i] / mediaWidth) * imageWidth;
      const localY = (contour[i + 1] / mediaHeight) * imageHeight;
      
      // Translate to canvas coordinates
      const canvasX = localX + imageX;
      const canvasY = localY + imageY;
      
      denormalized.push(canvasX, canvasY);
    }
    return denormalized;
  });
}

// Hook-based API for automatic mask creation
export interface UseMaskOptions {
  id?: string;
  inputPath: string;
  frameNumber?: number;
  tool: string;
  points?: number[] | Array<{x: number, y: number}>;
  pointLabels?: Array<number>;
  box?: {x1: number, y1: number, x2: number, y2: number};
  displayWidth?: number;
  displayHeight?: number;
  mediaInfo?: MediaInfo;
  clipTransform?: {x: number, y: number, width: number, height: number, scaleX?: number, scaleY?: number, rotation?: number};
  multimaskOutput?: boolean;
  simplifyTolerance?: number;
  modelType?: string;
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
    modelType = 'sam2_base_plus',
    enabled = true,
    debounceMs = 250
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
      let normalizedPoints: Array<{x: number, y: number}> | undefined;
      let normalizedLabels: Array<number> | undefined;
      
      if (points && points.length > 0) {
        // Convert flat array to point objects if needed
        const pointObjects = Object.prototype.hasOwnProperty.call(points[0], 'x') 
            && Object.prototype.hasOwnProperty.call(points[0], 'y') ? points as Array<{x: number, y: number}> : flatArrayToPoints(points as number[]); 
        

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
              clipTransform
            );
            normalizedPoints = result.points;
            normalizedLabels = result.labels;
          } else {
            // No labels, just normalize points
            normalizedPoints = normalizePoints(pointObjects, displayWidth, displayHeight, mediaInfo, true, clipTransform);
          }
          
          // If all points were filtered out, don't make the request
          if (normalizedPoints.length === 0 && pointObjects.length > 0) {
            setError('All points are outside the media bounds');
            setLoading(false);
            return;
          }
        } else {
          normalizedPoints = pointObjects;
          normalizedLabels = pointLabels;
        }
      }
      
      // Normalize box if provided
      let normalizedBox: {x1: number, y1: number, x2: number, y2: number} | undefined;
      if (box && mediaInfo && displayWidth && displayHeight) {
        normalizedBox = normalizeBox(box, displayWidth, displayHeight, mediaInfo, clipTransform);
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
        model_type: modelType
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
            contours: denormalizeContours(finalData.contours, displayWidth, displayHeight, mediaInfo, clipTransform)
          };
        }
        if (cacheKey) {
          maskCache.set(cacheKey, finalData);
        }
        setData(finalData);
        setError(null);
      } else {
        const errorMessage = response.error || 'Failed to create mask';
        setError(errorMessage);
        setData(null);
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
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
    modelType
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
    refetch: fetchMask
  };
}

// Legacy function for backward compatibility
export async function createMask(request: MaskRequest): Promise<ConfigResponse<MaskResponse>> {
  return await createMaskPreload(request);
}

export interface StreamedMaskFrame {
  stream_id:string;
  frame_number: number;
  contours: Array<Array<number>>;
}

export async function trackMask(
  request: {
    id: string;
    input_path: string;
    frame_start: number;
    anchor_frame: number;
    frame_end: number;
    direction?: 'forward' | 'backward' | 'both';
    model_type?: string;
    max_frames?: number;
  },
  options?: {
    signal?: AbortSignal;
    onProgress?: (progress: number) => void;
    onFrame?: (frame: StreamedMaskFrame) => void;
  }
): Promise<void> {
  const { signal, onProgress, onFrame } = options || {};

  const { streamId } = await startMaskTrack(request as any);
  const anchor = request.anchor_frame ?? request.frame_start;
  const low = Math.min(request.frame_start, request.frame_end);
  const high = Math.max(request.frame_start, request.frame_end);
  let totalFrames: number;
  if (request.direction === 'both') {
    let lowerBound = low;
    let upperBound = high;
    if (request.max_frames && request.max_frames > 0) {
      lowerBound = Math.max(low, anchor - request.max_frames);
      upperBound = Math.min(high, anchor + request.max_frames);
    }
    totalFrames = Math.max(1, upperBound - lowerBound + 1);
  } else {
    const rawSpan = Math.abs(request.frame_end - anchor);
    const span = request.max_frames && request.max_frames > 0 ? Math.min(rawSpan, request.max_frames) : rawSpan;
    totalFrames = span + 1;
  }
  const seenFrames = new Set<number>();

  const abortHandler = () => {
    try {
      cancelMaskTrackPreload(streamId);
    } catch {}
  };
  if (signal) {
    if (signal.aborted) {
      abortHandler();
      throw new DOMException('Aborted', 'AbortError');
    }
    signal.addEventListener('abort', abortHandler, { once: true });
  }

  try {
    await new Promise<void>((resolve, reject) => {
      const offChunk = onMaskTrackChunk(streamId, (evt: any) => {
        if (evt && evt.status === 'error') {
          offChunk();
          offError();
          offEnd();
          reject(new Error(evt.error || 'Mask tracking error'));
          return;
        }
        if (evt && typeof evt.frame_number === 'number') {
          if (onFrame) {
            onFrame({ frame_number: evt.frame_number, contours: Array.isArray(evt.contours) ? evt.contours : [], stream_id:streamId});
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
        reject(new Error(err?.message || 'Mask tracking error'));
      });
      const offEnd = onMaskTrackEnd(streamId, () => {
        offChunk();
        offError();
        offEnd();
        resolve();
      });
    });

    // Ensure we report completion
    if (onProgress && totalFrames > 0) {
      onProgress(1);
    }
  } finally {
    if (signal) {
      signal.removeEventListener('abort', abortHandler);
    }
  }
}


export async function cancelMaskTrack(streamId: string): Promise<void> {
  await cancelMaskTrackPreload(streamId);
}

export default trackMask;
