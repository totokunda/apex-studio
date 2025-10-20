import { useClipStore } from '@/lib/clip';
import { useControlsStore } from '@/lib/control';
import { MaskClipProps, MaskData, MaskTrackingDirection, MediaInfo } from '@/lib/types';
import React, { useCallback,  useMemo, useRef, useState } from 'react';
import { LuChevronDown, LuPlay } from 'react-icons/lu';
import { BsStop } from "react-icons/bs";
import {  getLocalFrame } from '@/lib/clip';
import { getMediaInfoCached } from '@/lib/media/utils';
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuRadioItem, DropdownMenuRadioGroup } from '@/components/ui/dropdown-menu';
import { trackMask as trackMaskApi, denormalizeContours, cancelMaskTrack, getActiveMaskTrackSession, attachToMaskTrack, createThenTrackMask, normalizePoints, normalizeBox, trackShapes, createMask as createMaskApi, denormalizeShapeBounds } from '@/lib/mask/api';
import PropertiesSlider from '../PropertiesSlider';
import Konva from 'konva';
Konva

interface MaskTrackingPropertiesProps {
  mask: MaskClipProps;
  clipId: string;
}

const MaskTrackingProperties: React.FC<MaskTrackingPropertiesProps> = ({ mask, clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId));
  const updateClip = useClipStore((s) => s.updateClip);
  const updateMaskKeyframes = useClipStore((s) => s.updateMaskKeyframes);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const mediaInfoRef = useRef<MediaInfo | null>(getMediaInfoCached(clip?.src || ''));
  const streamingIdRef = useRef<string | null>(null);
  const detachStreamListenersRef = useRef<(() => void) | null>(null);

  const [tracking, setTracking] = useState(false);
  const [trackingProgress, setTrackingProgress] = useState(0);
  const [trackingError, setTrackingError] = useState<string | null>(null);
  const fps = useControlsStore((s) => s.fps || 24);

  const isVideoClip = clip?.type === 'video';

  const localFrame = useMemo(() => {
    if (!clip || !isVideoClip) return 0;
    return Math.max(0, Math.round(getLocalFrame(focusFrame, clip)));
  }, [clip, focusFrame, isVideoClip]);

  const clipDuration = useMemo(() => {
    if (!clip || !isVideoClip) return 0;
    const realEnd = Math.max(0, (clip.endFrame ?? 0) - (clip.framesToGiveEnd ?? 0));
    const realStart = Math.max(0, (clip.startFrame ?? 0) + (clip.framesToGiveStart ?? 0));
    return Math.max(0, realEnd - realStart);
  }, [clip, isVideoClip]);

  const maxForwardAvailable = useMemo(() => {
    if (!isVideoClip) return 0;
    return Math.max(0, clipDuration - localFrame);
  }, [clipDuration, localFrame, isVideoClip]);

  const maxBackwardAvailable = useMemo(() => {
    if (!isVideoClip) return 0;
    return Math.max(0, localFrame);
  }, [localFrame, isVideoClip]);

  const sliderUpperBound = useMemo(() => {
    if (!isVideoClip) return 0;
    switch (mask.trackingDirection) {
      case 'forward':
        return maxForwardAvailable;
      case 'backward':
        return maxBackwardAvailable;
      default:
        return Math.max(maxForwardAvailable, maxBackwardAvailable);
    }
  }, [isVideoClip, mask.trackingDirection, maxForwardAvailable, maxBackwardAvailable]);

  const clampedSliderUpper = Math.max(0, Math.floor(sliderUpperBound));
  const sliderDisplayMax = Math.max(1, clampedSliderUpper || 1);
  const storedMaxFrames = mask.maxTrackingFrames ?? sliderDisplayMax;
  const sliderDisplayValue = Math.min(Math.max(1, storedMaxFrames), sliderDisplayMax);
  const effectiveMaxFrames = clampedSliderUpper > 0 ? Math.min(sliderDisplayValue, clampedSliderUpper) : 0;

  const updateMask = (updates: Partial<MaskClipProps>) => {
    if (!clip || !mask) return;
    
    const masks = (clip as any).masks || [];
    const updatedMasks = masks.map((m: MaskClipProps) =>
      m.id === mask.id ? { ...m, ...updates } : m
    );
    
    updateClip(clipId, { masks: updatedMasks });
  };



  const isLassoOrShape = mask?.tool === 'lasso' || mask?.tool === 'shape' || (mask?.tool as any) === 'rectangle';
  const canTrack = Boolean(
    clip &&
    mask &&
    (mask.tool === 'touch' || isLassoOrShape) &&
    typeof focusFrame === 'number' &&
    mask.keyframes &&
    mediaInfoRef.current
  );

  const handleTrackMask = useCallback(async () => {
    if (!canTrack || !clip || !mask) return;

    const isVideo = clip.type === 'video';
    const direction: MaskTrackingDirection = mask.trackingDirection ?? 'both';

    const clipPath = (clip as any).src as string | undefined;
    if (!clipPath) {
      setTrackingError('Clip source unavailable for mask tracking');
      return;
    }

    const width = mediaInfoRef.current?.video?.displayWidth;
    const height = mediaInfoRef.current?.video?.displayHeight;

    const maxFramesToTrack = effectiveMaxFrames > 0 ? effectiveMaxFrames : undefined;

    type TrackingTask = {
      anchor_frame: number;
      direction: MaskTrackingDirection;
      frame_start: number;
      frame_end: number;
    };


    

    const clipFps = mediaInfoRef.current?.stats.video?.averagePacketRate ?? 24;
    const mediaStartFrame = Math.round((mediaInfoRef.current?.startFrame ?? 0) / fps * clipFps)

    const anchorLocalFrame = Math.round(localFrame / fps * clipFps);
    

    const clipLocalDuration = Math.max(0, Math.round(clipDuration / fps * clipFps));

    const framesForward = Math.round(((maxFramesToTrack ?? maxForwardAvailable) / fps) * clipFps);
    const framesBackward = Math.round(((maxFramesToTrack ?? maxBackwardAvailable) / fps) * clipFps);


    let task: TrackingTask | null = null;

    if (direction === 'forward') {
      if (!isVideo || framesForward > 0) {
        task = {
          anchor_frame: anchorLocalFrame + mediaStartFrame,
          direction: 'forward',
          frame_start: anchorLocalFrame + mediaStartFrame,
          frame_end: anchorLocalFrame + mediaStartFrame + Math.max(0, framesForward),
        };
      }
    }
    else if (direction === 'backward') {
      if (!isVideo || framesBackward > 0) {
        task = {
          anchor_frame: anchorLocalFrame + mediaStartFrame,
          direction: 'backward',
          frame_start: anchorLocalFrame + mediaStartFrame,
          frame_end: anchorLocalFrame + mediaStartFrame - Math.max(0, framesBackward),
        };
      }
    } 
    else if (direction === 'both') {
      const maxIndex = Math.max(0, clipLocalDuration - 1);
      const backward = Math.min(anchorLocalFrame, Math.max(0, framesBackward));
      const forward = Math.min(Math.max(0, framesForward - 1), Math.max(0, maxIndex - anchorLocalFrame));
      task = {
        direction: 'both',
        anchor_frame: anchorLocalFrame + mediaStartFrame,
        frame_start: anchorLocalFrame + mediaStartFrame - backward,
        frame_end: anchorLocalFrame + mediaStartFrame + forward,
      };
    } 


    if (!task) {
      setTrackingError('No frames available for tracking in the selected direction');
      return;
    }



    setTracking(true);
    setTrackingProgress(0);
    setTrackingError(null);


    const abortController = new AbortController();
    // Track last saved contours in this tracking session to avoid duplicate keyframes
    let lastSavedContours: Array<Array<number>> | undefined;

    try {
        await trackMaskApi(
          {
            id: mask.id,
            input_path: clipPath as string,
            anchor_frame: task.anchor_frame,
            frame_start: task.frame_start,
            frame_end: task.frame_end,
            direction: task.direction,
          },
          {
            signal: abortController.signal,
            onProgress: (progress) => {
              const overallProgress = progress;
              setTrackingProgress(overallProgress);
            },
            onFrame: ({ frame_number, contours, stream_id }) => {
              if (!mediaInfoRef.current || !clip) return;

              streamingIdRef.current = stream_id

   
              const denormalized = denormalizeContours(contours, width ?? 0, height ?? 0, mediaInfoRef.current, clip.transform);
              let mediaStartFrame = (mediaInfoRef.current.startFrame ?? 0)
              const local = Math.round(frame_number / clipFps * fps) - mediaStartFrame;

              

              const nearlyEqual = (a: number, b: number, eps = 1e-3) => Math.abs(a - b) <= eps;
              const contoursEqual = (a?: Array<Array<number>>, b?: Array<Array<number>>): boolean => {
                if (!a && !b) return true;
                if (!a || !b) return false;
                if (a.length !== b.length) return false;
                for (let i = 0; i < a.length; i++) {
                  const pa = a[i];
                  const pb = b[i];
                  if (pa.length !== pb.length) return false;
                  for (let j = 0; j < pa.length; j++) {
                    if (!nearlyEqual(pa[j], pb[j])) return false;
                  }
                }
                return true;
              };

              if (contoursEqual(lastSavedContours, denormalized)) {
                return;
              }

              updateMaskKeyframes(clip.clipId, mask.id, (existing) => {
                const keyframes = existing instanceof Map ? new Map(existing) : { ...(existing as Record<number, MaskData>) };
                const current = keyframes instanceof Map ? keyframes.get(local) : (keyframes as Record<number, MaskData>)[local];
                const nextData: MaskData = { ...(current ?? {}), contours: denormalized };
                if (keyframes instanceof Map) { keyframes.set(local, nextData); lastSavedContours = denormalized; return keyframes; }
                (keyframes as Record<number, MaskData>)[local] = nextData; lastSavedContours = denormalized; return keyframes;
              });
            },
          }
        );

    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setTrackingError('Mask tracking cancelled');
      } else {
        if (err instanceof Error && err.message === 'This operation was aborted') {
          return;
        }
        setTrackingError(err instanceof Error ? err.message : 'Failed to track mask');
      }
    } finally {
      streamingIdRef.current = null;
      setTracking(false);
    }
  }, [canTrack, clip, mask, focusFrame, mediaInfoRef, updateMaskKeyframes, updateMask]);

  const handleCreateThenTrackMask = useCallback(async () => {
    if (!canTrack || !clip || !mask || !isLassoOrShape) return;

    const isVideo = clip.type === 'video';
    const direction: MaskTrackingDirection = mask.trackingDirection ?? 'both';
    const clipPath = (clip as any).src as string | undefined;
    if (!clipPath) {
      setTrackingError('Clip source unavailable for mask tracking');
      return;
    }
    const width = mediaInfoRef.current?.video?.displayWidth;
    const height = mediaInfoRef.current?.video?.displayHeight;
    const clipFps = mediaInfoRef.current?.stats.video?.averagePacketRate ?? 24;
    const mediaStartFrame = Math.round((mediaInfoRef.current?.startFrame ?? 0) / fps * clipFps);
    const anchorLocalFrame = Math.round(localFrame / fps * clipFps);
    const frameNumber = anchorLocalFrame + mediaStartFrame;

    const clipLocalDuration = Math.max(0, Math.round(clipDuration / fps * clipFps));
    const framesForward = Math.round(((effectiveMaxFrames ?? maxForwardAvailable) / fps) * clipFps);
    const framesBackward = Math.round(((effectiveMaxFrames ?? maxBackwardAvailable) / fps) * clipFps);

    type TrackingTask = {
      anchor_frame: number;
      direction: MaskTrackingDirection;
      frame_start: number;
      frame_end: number;
    };
    let task: TrackingTask | null = null;
    if (direction === 'forward') {
      if (!isVideo || framesForward > 0) {
        task = {
          anchor_frame: anchorLocalFrame + mediaStartFrame,
          direction: 'forward',
          frame_start: anchorLocalFrame + mediaStartFrame,
          frame_end: anchorLocalFrame + mediaStartFrame + Math.max(0, framesForward),
        };
      }
    } else if (direction === 'backward') {
      if (!isVideo || framesBackward > 0) {
        task = {
          anchor_frame: anchorLocalFrame + mediaStartFrame,
          direction: 'backward',
          frame_start: anchorLocalFrame + mediaStartFrame,
          frame_end: anchorLocalFrame + mediaStartFrame - Math.max(0, framesBackward),
        };
      }
    } else {
      const maxIndex = Math.max(0, clipLocalDuration - 1);
      const backward = Math.min(anchorLocalFrame, Math.max(0, framesBackward));
      const forward = Math.min(Math.max(0, framesForward - 1), Math.max(0, maxIndex - anchorLocalFrame));
      task = {
        direction: 'both',
        anchor_frame: anchorLocalFrame + mediaStartFrame,
        frame_start: anchorLocalFrame + mediaStartFrame - backward,
        frame_end: anchorLocalFrame + mediaStartFrame + forward,
      };
    }

    if (!task) {
      setTrackingError('No frames available for tracking in the selected direction');
      return;
    }

    // pick the active keyframe data at or before current localFrame
    const getActiveKeyframeData = (): MaskData | undefined => {
      const kf = mask.keyframes as any;
      const entries: Array<[number, any]> = kf instanceof Map ? Array.from(kf.entries()) : Object.entries(kf).map(([k, v]) => [Number(k), v]);
      if (entries.length === 0) return undefined;
      entries.sort((a, b) => a[0] - b[0]);
      let chosen = entries[0][1];
      for (const [k, v] of entries) {
        if (k <= localFrame) chosen = v;
        else break;
      }
      return chosen as MaskData;
    };

    const activeData = getActiveKeyframeData();
    if (!activeData) {
      setTrackingError('No keyframe data available to seed tracking');
      return;
    }

     // get the shape type 
     const keyframes = mask.keyframes;
     // get the first keyframe
     let shapeType: string = 'rectangle';
     if (keyframes instanceof Map) {
       const firstKeyframe = keyframes.entries().next().value;
       if (firstKeyframe) {
         shapeType = firstKeyframe[1].shapeBounds?.shapeType || 'rectangle';
       }
     } else {
       const firstKeyframe = keyframes[0];
       if (firstKeyframe) {
         shapeType = firstKeyframe.shapeBounds?.shapeType || 'rectangle';
       }
     }

    // Build create request
    const createReq: any = {
      id: mask.id,
      input_path: clipPath,
      frame_number: frameNumber,
      tool: mask.tool === 'shape' ? 'shape' : 'lasso',
      simplify_tolerance: 1.0,

    };

    if ((mask.tool === 'shape') && activeData.shapeBounds) {
      const b = activeData.shapeBounds as any;
      const scaleX = b.scaleX ?? 1;
      const scaleY = b.scaleY ?? 1;
      const x1 = b.x;
      const y1 = b.y;
      const x2 = b.x + (b.width * scaleX);
      const y2 = b.y + (b.height * scaleY);
      if (mediaInfoRef.current && width && height) {
        createReq.box = normalizeBox({ x1, y1, x2, y2 }, width, height, mediaInfoRef.current, clip.transform);
      } else {
        createReq.box = { x1, y1, x2, y2 };
      }
      createReq.shape_type = shapeType;
    } else if (mask.tool === 'lasso' && activeData.lassoPoints && Array.isArray(activeData.lassoPoints) && activeData.lassoPoints.length >= 6) {
      const lassoPts = [] as Array<{ x: number, y: number }>;
      for (let i = 0; i < activeData.lassoPoints.length; i += 2) {
        const x = activeData.lassoPoints[i];
        const y = activeData.lassoPoints[i + 1];
        if (typeof x === 'number' && typeof y === 'number') lassoPts.push({ x, y });
      }
      createReq.points = mediaInfoRef.current && width && height
        ? normalizePoints(lassoPts, width, height, mediaInfoRef.current, true, clip.transform)
        : lassoPts;
    } else {
      setTrackingError('Missing lasso or shape data to seed tracking');
      return;
    }

    setTracking(true);
    setTrackingProgress(0);
    setTrackingError(null);

    const abortController = new AbortController();
    let lastSavedContours: Array<Array<number>> | undefined;
    let lastSavedLassoPoints: Array<number> | undefined;
    try {
      if ((mask.tool === 'shape')) {
        // 1) Create initial mask only (no streaming track here)
        const created = await createMaskApi(createReq);
        if (!created.success || !created.data || created.data.status !== 'success') {
          throw new Error(created.error || created.data?.message || 'Shape mask creation failed');
        }

        // 2) Start shapes stream to write shapeBounds per frame
        await trackShapes(
          {
            id: mask.id,
            input_path: clipPath,
            anchor_frame: task.anchor_frame,
            frame_start: task.frame_start,
            frame_end: task.frame_end,
            direction: task.direction,
            shape_type: shapeType,
          },
          {
            signal: abortController.signal,
            onProgress: (p) => setTrackingProgress(p),
            onFrame: ({ frame_number, shapeBounds, stream_id }) => {
              streamingIdRef.current = stream_id;
              const local = Math.round((frame_number / clipFps) * fps) - (mediaInfoRef.current?.startFrame ?? 0);
              // Denormalize bounds for canvas coordinates
              const denormBounds = (mediaInfoRef.current && width && height && shapeBounds)
                ? denormalizeShapeBounds(shapeBounds, width, height, mediaInfoRef.current, clip.transform)
                : shapeBounds;
              updateMaskKeyframes(clip.clipId, mask.id, (existing) => {
                const keyframes = existing instanceof Map ? new Map(existing) : { ...(existing as Record<number, MaskData>) };
                const current = keyframes instanceof Map ? keyframes.get(local) : (keyframes as Record<number, MaskData>)[local];
                const nextData: MaskData = { ...(current ?? {}), shapeBounds: denormBounds } as any;
                if (keyframes instanceof Map) {
                  keyframes.set(local, nextData);
                  return keyframes;
                }
                (keyframes as Record<number, MaskData>)[local] = nextData;
                return keyframes;
              });
            },
          }
        );
      } else {
        await createThenTrackMask(
          {
            create: createReq,
            track: {
              id: mask.id,
              input_path: clipPath,
              anchor_frame: task.anchor_frame,
              frame_start: task.frame_start,
              frame_end: task.frame_end,
              direction: task.direction,
            },
          },
          {
            signal: abortController.signal,
            onProgress: (progress) => setTrackingProgress(progress),
            onFrame: ({ frame_number, contours, stream_id }) => {
              if (!mediaInfoRef.current || !clip) return;
              streamingIdRef.current = stream_id;
              const denormalized = denormalizeContours(contours, width ?? 0, height ?? 0, mediaInfoRef.current, clip.transform);
              const local = Math.round((frame_number / clipFps) * fps) - (mediaInfoRef.current?.startFrame ?? 0);
              if (mask.tool === 'lasso') {
                const area = (poly: number[]) => {
                  let a = 0; for (let i = 0; i < poly.length; i += 2) {
                    const x1 = poly[i], y1 = poly[i+1];
                    const j = (i + 2) % poly.length;
                    const x2 = poly[j], y2 = poly[j+1];
                    a += x1 * y2 - x2 * y1;
                  } return Math.abs(a) * 0.5;
                };
                const largest = (denormalized || []).reduce((best: number[] | null, c) => {
                  const arr = c as number[]; return !best || area(arr) > area(best) ? arr : best;
                }, null as number[] | null) || [];
                const equalFlat = (a?: number[], b?: number[]) => {
                  if (!a && !b) return true; if (!a || !b) return false; if (a.length !== b.length) return false;
                  for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-3) return false; return true;
                };
                if (equalFlat(lastSavedLassoPoints, largest)) return;
                updateMaskKeyframes(clip.clipId, mask.id, (existing) => {
                  const keyframes = existing instanceof Map ? new Map(existing) : { ...(existing as Record<number, MaskData>) };
                  const current = keyframes instanceof Map ? keyframes.get(local) : (keyframes as Record<number, MaskData>)[local];
                  const nextData: MaskData = { ...(current ?? {}), lassoPoints: largest } as any;
                  if (keyframes instanceof Map) { keyframes.set(local, nextData); lastSavedLassoPoints = largest; return keyframes; }
                  (keyframes as Record<number, MaskData>)[local] = nextData; lastSavedLassoPoints = largest; return keyframes;
                });
              } else {
                const nearlyEqual = (a: number, b: number, eps = 1e-3) => Math.abs(a - b) <= eps;
                const contoursEqual = (a?: Array<Array<number>>, b?: Array<Array<number>>): boolean => {
                  if (!a && !b) return true; if (!a || !b) return false; if (a.length !== b.length) return false;
                  for (let i = 0; i < a.length; i++) {
                    const pa = a[i], pb = b[i]; if (pa.length !== pb.length) return false;
                    for (let j = 0; j < pa.length; j++) if (!nearlyEqual(pa[j], pb[j])) return false;
                  }
                  return true;
                };
                if (contoursEqual(lastSavedContours, denormalized)) return;
                updateMaskKeyframes(clip.clipId, mask.id, (existing) => {
                  const keyframes = existing instanceof Map ? new Map(existing) : { ...(existing as Record<number, MaskData>) };
                  const current = keyframes instanceof Map ? keyframes.get(local) : (keyframes as Record<number, MaskData>)[local];
                  const nextData: MaskData = { ...(current ?? {}), contours: denormalized };
                  if (keyframes instanceof Map) { keyframes.set(local, nextData); lastSavedContours = denormalized; return keyframes; }
                  (keyframes as Record<number, MaskData>)[local] = nextData; lastSavedContours = denormalized; return keyframes;
                });
              }
            },
          }
        );
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setTrackingError('Mask tracking cancelled');
      } else {
        if (err instanceof Error && err.message === 'This operation was aborted') {
          return;
        }
        setTrackingError(err instanceof Error ? err.message : 'Failed to create then track mask');
      }
    } finally {
      streamingIdRef.current = null;
      setTracking(false);
    }
  }, [canTrack, clip, mask, mask.keyframes, isLassoOrShape, localFrame, clipDuration, effectiveMaxFrames, maxForwardAvailable, maxBackwardAvailable, fps, mediaInfoRef, updateMaskKeyframes]);


  const handleStopTracking = useCallback(async () => {
    if (!tracking) return;
    setTracking(false);
    setTrackingProgress(0);
    setTrackingError(null);
    if (detachStreamListenersRef.current) {
      try { detachStreamListenersRef.current(); } catch {}
      detachStreamListenersRef.current = null;
    }
    const activeStreamId = streamingIdRef.current || getActiveMaskTrackSession(mask.id || '')?.streamId || null;
    if (activeStreamId) {
      cancelMaskTrack(activeStreamId);
    }
    streamingIdRef.current = null;
  }, [tracking]);



  if (!mask) {
    return (
      <div className="p-4 px-5">
        <p className="text-brand-light/50 text-[11px]">No mask selected</p>
      </div>
    );
  }

  
  React.useEffect(() => {
    if (!mask?.id || !clip) return;
    const session = getActiveMaskTrackSession(mask.id);
    if (!session || session.ended) return;

    setTracking(true);
    setTrackingError(null);
    if (session.totalFrames > 0) {
      setTrackingProgress(Math.min(1, session.seenFrames.size / session.totalFrames));
    }
    streamingIdRef.current = session.streamId;

    const width = mediaInfoRef.current?.video?.displayWidth;
    const height = mediaInfoRef.current?.video?.displayHeight;
    const clipFps = mediaInfoRef.current?.stats.video?.averagePacketRate ?? 24;
    const mediaStartFrame = (mediaInfoRef.current?.startFrame ?? 0);

    let lastSavedContours: Array<Array<number>> | undefined;

    const detach = attachToMaskTrack(mask.id, {
      onProgress: (p) => setTrackingProgress(p),
      onFrame: ({ frame_number, contours, stream_id }) => {
        if (!mediaInfoRef.current || !clip) return;
        streamingIdRef.current = stream_id;
        const denormalized = denormalizeContours(
          contours,
          width ?? 0,
          height ?? 0,
          mediaInfoRef.current,
          clip.transform
        );
        const local = Math.round((frame_number / clipFps) * fps) - mediaStartFrame;

        const nearlyEqual = (a: number, b: number, eps = 1e-3) => Math.abs(a - b) <= eps;
        const contoursEqual = (a?: Array<Array<number>>, b?: Array<Array<number>>): boolean => {
          if (!a && !b) return true;
          if (!a || !b) return false;
          if (a.length !== b.length) return false;
          for (let i = 0; i < a.length; i++) {
            const pa = a[i];
            const pb = b[i];
            if (pa.length !== pb.length) return false;
            for (let j = 0; j < pa.length; j++) {
              if (!nearlyEqual(pa[j], pb[j])) return false;
            }
          }
          return true;
        };

        updateMaskKeyframes(clip.clipId, mask.id, (existing) => {
          const keyframes = existing instanceof Map ? new Map(existing) : { ...(existing as Record<number, MaskData>) };
          const current = keyframes instanceof Map ? keyframes.get(local) : (keyframes as Record<number, MaskData>)[local];
          const currentContours = current?.contours as Array<Array<number>> | undefined;
          if (contoursEqual(currentContours, denormalized) || contoursEqual(lastSavedContours, denormalized)) {
            return keyframes;
          }
          const nextData: MaskData = { ...(current ?? {}), contours: denormalized };
          if (keyframes instanceof Map) {
            keyframes.set(local, nextData);
            lastSavedContours = denormalized;
            return keyframes;
          }
          (keyframes as Record<number, MaskData>)[local] = nextData;
          lastSavedContours = denormalized;
          return keyframes;
        });
      },
      onEnd: () => {
        streamingIdRef.current = null;
        setTracking(false);
      },
    });

    detachStreamListenersRef.current = detach;
    return () => {
      if (detachStreamListenersRef.current) {
        try { detachStreamListenersRef.current(); } catch {}
        detachStreamListenersRef.current = null;
      }
    };
  }, [mask?.id, clip, fps, updateMaskKeyframes]);

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 flex flex-col gap-y-4 px-5 w-full">
        <h4 className="text-brand-light text-[12px] font-medium text-start">Mask Tracking</h4>
        <div className="flex flex-col items-start w-full gap-y-3">
        <PropertiesSlider 
          labelClass="text-brand-light text-[11px] font-medium text-start"
          toFixed={0}
          suffix=" F"
          label="Max Frames"
          value={sliderDisplayValue}
          min={1}
          max={sliderDisplayMax}
          step={1}
          onChange={(value: number) => updateMask({ maxTrackingFrames: value })}
          disabled={!isVideoClip || sliderDisplayMax <= 1}
        />
        <div className="w-full">
          <h3 className="text-brand-light text-[11px] font-medium text-start">
            Direction
          </h3>
        <DropdownMenu>
            <DropdownMenuTrigger asChild className="w-full">
              <button className="w-full mt-2 flex flex-row items-center justify-between gap-x-2 px-3 py-2 cursor-pointer rounded-md bg-brand border border-brand-light/20 hover:bg-brand-light/10 transition-all duration-200">
                <span className="text-brand-light text-[11px] font-medium">
                  {mask.trackingDirection === 'forward' ? 'Forward' : mask.trackingDirection === 'backward' ? 'Backward' : 'Both'}
                </span>
                <LuChevronDown className="w-3 h-3 text-brand-light" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className='w-[var(--radix-dropdown-menu-trigger-width)] dark font-poppins bg-brand-background'>
              <DropdownMenuRadioGroup value={mask.trackingDirection ?? 'both'} onValueChange={(value: string) => updateMask({ trackingDirection: value as MaskTrackingDirection })}>
                <DropdownMenuRadioItem value='forward' className='w-full'>
                  <span className="text-brand-light text-[11px] font-medium">Forward</span>
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value='backward' className='w-full'>
                  <span className="text-brand-light text-[11px] font-medium">Backward</span>
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value='both' className='w-full'>
                  <span className="text-brand-light text-[11px] font-medium">Both</span>
                </DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
        </DropdownMenu>
        </div>
        </div>
        <button
          className="w-full flex flex-row items-center gap-x-2 px-2 py-2.5 cursor-pointer rounded-md justify-center bg-brand border border-brand-light/20 hover:bg-brand-light/10 transition-all duration-200"
          onClick={() => {
            if (tracking) {
              handleStopTracking();
            } else {
              if (isLassoOrShape) {
                handleCreateThenTrackMask();
              } else {
                handleTrackMask();
              }
            }
          }}
        >
          {tracking ? <BsStop className="w-3.5 h-3.5 text-brand-light" /> : <LuPlay className="w-3 h-3 text-brand-light" />}
          <span className="text-brand-light text-[11px] font-medium">{tracking ? 'Stop Tracking' : 'Track Mask'}</span>
        </button>

        {tracking && (
          <div className="mb-1">
            <div className="flex flex-row items-center justify-between mb-3">
            <p className="text-brand-light text-[10px] font-medium text-start">
              Tracking 
            </p>
            <p className="text-brand-light text-[10px] font-medium text-start">
              {Math.round((trackingProgress || 0) * 100)}%
            </p>
            </div>
            <div className="w-full h-1.5 rounded-full bg-brand-light/10 overflow-hidden">
              <div
                className="h-full bg-brand-light transition-all duration-150"
                style={{ width: `${Math.min(100, Math.round((trackingProgress || 0) * 100))}%` }}
              />
            </div>
            
          </div>
        )}

        {trackingError && !tracking && (
          <p className="mt-2 text-[10px] text-red-400">{trackingError}</p>
        )}

      </div>
    </div>
  );
};

export default MaskTrackingProperties;
