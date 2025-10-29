import { MediaInfo, VideoClipProps} from '@/lib/types'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {Image, Transformer, Group, Line} from 'react-konva'
import {getVideoIterator} from '@/lib/media/video'
import {getMediaInfo, getMediaInfoCached} from '@/lib/media/utils'
import {fetchCanvasSample} from '@/lib/media/canvas'
import { useControlsStore } from '@/lib/control';
import Konva from 'konva';
import { useViewportStore } from '@/lib/viewport';
import { useClipStore, getLocalFrame } from '@/lib/clip';
import { WrappedCanvas } from 'mediabunny';
import { useWebGLFilters } from '@/components/preview/webgl-filters';
import { BaseClipApplicator } from './apply/base'
import _ from 'lodash';
import { useWebGLMask } from '../mask/useWebGLMask'

// (prefetch helper removed by request; timeline-driven rendering only)

const VideoPreview: React.FC<VideoClipProps & {framesToPrefetch?: number, rectWidth: number, rectHeight: number, applicators: BaseClipApplicator[], overlap: boolean}> = ({ src, clipId, startFrame = 0, framesToPrefetch: _framesToPrefetch = 32, rectWidth, rectHeight, trimStart, speed: _speed, applicators, overlap}) => {
    const mediaInfo = useRef<MediaInfo | null>(getMediaInfoCached(src) || null);
    const focusFrame = useControlsStore((state) => state.focusFrame);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const originalFrameRef = useRef<HTMLCanvasElement | null>(null); // Store unfiltered frame
    const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const imageRef = useRef<Konva.Image>(null);
    const transformerRef = useRef<Konva.Transformer>(null);
    const drawTokenRef = useRef(0);
    const suppressUntilRef = useRef<number>(0);
    const { applyFilters } = useWebGLFilters();
    const currentFrame = useMemo(() => focusFrame - startFrame + (trimStart || 0), [focusFrame, startFrame, trimStart]);
    const speed = useMemo(() => {
      const s = Number(_speed ?? 1);
      return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
    }, [_speed]);
    const tool = useViewportStore((s) => s.tool);
    const scale = useViewportStore((s) => s.scale);
    const position = useViewportStore((s) => s.position);
    const setClipTransform = useClipStore((s) => s.setClipTransform);
    const clipTransform = useClipStore((s) => s.getClipTransform(clipId));
    const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
    const addClipSelection = useControlsStore((s) => s.addClipSelection);
    const clearSelection = useControlsStore((s) => s.clearSelection);
    const {selectedClipIds, isFullscreen, fps:srcFps} = useControlsStore();
    const isSelected = useMemo(() => selectedClipIds.includes(clipId), [clipId, selectedClipIds]);
    const lastSelectedSrcRef = useRef<string | null>(null); 
    const cachedPreprocessorRangeRef = useRef<{startFrame: number, endFrame: number, selectedSrc: string, frameOffset: number} | null>(null);
    const addedTimestampRef = useRef<number | undefined>(undefined); // last timestamp rendered
    const clip = useClipStore((s) => s.getClipById(clipId)) as VideoClipProps;

    const { applyMask } = useWebGLMask({
        focusFrame: focusFrame,
        masks: clip?.masks || [],
        disabled: tool === 'mask',
        clip: clip,
    });

    const {
        selectedSrc,
        frameOffset,
    } = useMemo(() => {
        // Check if we can use the cached result

        // Cache miss - recalculate
        if (!clip.preprocessors || clip.preprocessors.length === 0) {
            cachedPreprocessorRangeRef.current = null;
            addedTimestampRef.current = 0;
            return {selectedSrc: src, frameOffset: 0};
        }

        if (cachedPreprocessorRangeRef.current && 
            currentFrame >= cachedPreprocessorRangeRef.current.startFrame && 
            currentFrame <= cachedPreprocessorRangeRef.current.endFrame) {
            return {
                selectedSrc: cachedPreprocessorRangeRef.current.selectedSrc,
                frameOffset: cachedPreprocessorRangeRef.current.frameOffset
            };
        }

        
        
        // go through the preprocessors and find the one that is within the focus frame
        // adjust preprocessor ranges by trimStart to match currentFrame's reference frame
        const cliptrimStart = trimStart || 0;
        for (const preprocessor of clip.preprocessors) {
            if (preprocessor.startFrame !== undefined && preprocessor.endFrame !== undefined && preprocessor.status === 'complete' && preprocessor.src) {
                const adjustedStartFrame = preprocessor.startFrame + cliptrimStart;
                const adjustedEndFrame = preprocessor.endFrame + cliptrimStart;
                
                if (currentFrame >= adjustedStartFrame && currentFrame <= adjustedEndFrame) {
                    const startSec = preprocessor.startFrame / srcFps;
                    addedTimestampRef.current = startSec;

                    cachedPreprocessorRangeRef.current = {
                        startFrame: adjustedStartFrame,
                        endFrame: adjustedEndFrame,
                        selectedSrc: preprocessor.src,
                        frameOffset: preprocessor.startFrame,
                    };
                    
                    return {selectedSrc: preprocessor.src, frameOffset: preprocessor.startFrame};
                }
            }
        }

        cachedPreprocessorRangeRef.current = null;
        addedTimestampRef.current = 0;
        return {selectedSrc: src, frameOffset: 0};
    }, [clip?.preprocessors, src, currentFrame, trimStart]);

    // Use refs to store current filter values to avoid callback recreation
    const filterParamsRef = useRef({
        brightness: clip?.brightness,
        contrast: clip?.contrast,
        hue: clip?.hue,
        saturation: clip?.saturation,
        blur: clip?.blur,
        sharpness: clip?.sharpness,
        noise: clip?.noise,
        vignette: clip?.vignette
    });

    // Use ref to store current applicators to avoid callback recreation
    const applicatorsRef = useRef(applicators);
    const maskFrameForCurrentFocus = useMemo(() => {
        if (clip) {
            return Math.max(0, Math.floor(getLocalFrame(focusFrame, clip)));
        }
        return Math.max(0, Math.floor(currentFrame));
    }, [clip, focusFrame, currentFrame]);

    const aspectRatio = useMemo(() => {
      const originalWidth = mediaInfo.current?.video?.displayWidth || 0;
      const originalHeight = mediaInfo.current?.video?.displayHeight || 0;
      if (!originalWidth || !originalHeight) return 16/9;
      const aspectRatio = originalWidth / originalHeight;
      return aspectRatio;
    }, [mediaInfo.current?.video?.displayWidth, mediaInfo.current?.video?.displayHeight]);

    const groupRef = useRef<Konva.Group>(null);
    const SNAP_THRESHOLD_PX = 4; // pixels at screen scale
    const [guides, setGuides] = useState({
        vCenter: false,
        hCenter: false,
        v25: false,
        v75: false,
        h25: false,
        h75: false,
        left: false,
        right: false,
        top: false,
        bottom: false,
    });
    const [isInteracting, setIsInteracting] = useState(false);
    const [isRotating, setIsRotating] = useState(false);
    const [isTransforming, setIsTransforming] = useState(false);
    const iteratorRef = useRef<AsyncIterable<WrappedCanvas | null> | null>(null);
    const isPlaying = useControlsStore((s) => s.isPlaying);
    const fps = useControlsStore((s) => s.fps);
    const currentStartFrameRef = useRef<number>(0);
    const lastRenderedFrameRef = useRef<number>(-1);

    // Update refs when values change
    useEffect(() => {
        filterParamsRef.current = {
            brightness: clip?.brightness,
            contrast: clip?.contrast,
            hue: clip?.hue,
            saturation: clip?.saturation,
            blur: clip?.blur,
            sharpness: clip?.sharpness,
            noise: clip?.noise,
            vignette: clip?.vignette
        };
        applicatorsRef.current = applicators;
    }, [clip?.brightness, clip?.contrast, clip?.hue, clip?.saturation, clip?.blur, clip?.sharpness, clip?.noise, clip?.vignette, applicators, applicators.length]);

    const updateGuidesAndMaybeSnap = useCallback((opts: { snap: boolean }) => {
        if (isRotating) return; // disable guides/snapping while rotating
        const node = imageRef.current;
        const group = groupRef.current;
        if (!node || !group) return;
        const thresholdLocal = SNAP_THRESHOLD_PX / Math.max(0.0001, scale);
        const client = node.getClientRect({ skipShadow: true, skipStroke: true, relativeTo: group as any });
        const centerX = client.x + client.width / 2;
        const centerY = client.y + client.height / 2;
        const dxToVCenter = (rectWidth / 2) - centerX;
        const dyToHCenter = (rectHeight / 2) - centerY;
        const dxToV25 = (rectWidth * 0.25) - centerX;
        const dxToV75 = (rectWidth * 0.75) - centerX;
        const dyToH25 = (rectHeight * 0.25) - centerY;
        const dyToH75 = (rectHeight * 0.75) - centerY;
        const distVCenter = Math.abs(dxToVCenter);
        const distHCenter = Math.abs(dyToHCenter);
        const distV25 = Math.abs(dxToV25);
        const distV75 = Math.abs(dxToV75);
        const distH25 = Math.abs(dyToH25);
        const distH75 = Math.abs(dyToH75);
        const distLeft = Math.abs(client.x - 0);
        const distRight = Math.abs((client.x + client.width) - rectWidth);
        const distTop = Math.abs(client.y - 0);
        const distBottom = Math.abs((client.y + client.height) - rectHeight);

        const nextGuides = {
            vCenter: distVCenter <= thresholdLocal,
            hCenter: distHCenter <= thresholdLocal,
            v25: distV25 <= thresholdLocal,
            v75: distV75 <= thresholdLocal,
            h25: distH25 <= thresholdLocal,
            h75: distH75 <= thresholdLocal,
            left: distLeft <= thresholdLocal,
            right: distRight <= thresholdLocal,
            top: distTop <= thresholdLocal,
            bottom: distBottom <= thresholdLocal,
        };
        setGuides(nextGuides);

        if (opts.snap) {
            let deltaX = 0;
            let deltaY = 0;
            if (nextGuides.vCenter) {
                deltaX += dxToVCenter;
            } else if (nextGuides.v25) {
                deltaX += dxToV25;
            } else if (nextGuides.v75) {
                deltaX += dxToV75;
            } else if (nextGuides.left) {
                deltaX += -client.x;
            } else if (nextGuides.right) {
                deltaX += rectWidth - (client.x + client.width);
            }
            if (nextGuides.hCenter) {
                deltaY += dyToHCenter;
            } else if (nextGuides.h25) {
                deltaY += dyToH25;
            } else if (nextGuides.h75) {
                deltaY += dyToH75;
            } else if (nextGuides.top) {
                deltaY += -client.y;
            } else if (nextGuides.bottom) {
                deltaY += rectHeight - (client.y + client.height);
            }
            if (deltaX !== 0 || deltaY !== 0) {
                node.x(node.x() + deltaX);
                node.y(node.y() + deltaY);
                setClipTransform(clipId, { x: node.x(), y: node.y() });
            }
        }
    }, [rectWidth, rectHeight, scale, setClipTransform, clipId, isRotating]);

    const transformerBoundBoxFunc = useCallback((_oldBox: any, newBox: any) => {
        if (isRotating) return newBox; // do not snap bounds while rotating
        // Convert absolute newBox to local coordinates of the content group (rect space)
        const invScale = 1 / Math.max(0.0001, scale);
        const local = {
            x: (newBox.x - position.x) * invScale,
            y: (newBox.y - position.y) * invScale,
            width: newBox.width * invScale,
            height: newBox.height * invScale,
        };
        const thresholdLocal = SNAP_THRESHOLD_PX * invScale;

        const left = local.x;
        const right = local.x + local.width;
        const top = local.y;
        const bottom = local.y + local.height;
        const v25 = rectWidth * 0.25;
        const v75 = rectWidth * 0.75;
        const h25 = rectHeight * 0.25;
        const h75 = rectHeight * 0.75;

        // Snap left edge to 0, 25%, 75%
        if (Math.abs(left - 0) <= thresholdLocal) {
            local.x = 0;
            local.width = right - local.x;
        } else if (Math.abs(left - v25) <= thresholdLocal) {
            local.x = v25;
            local.width = right - local.x;
        } else if (Math.abs(left - v75) <= thresholdLocal) {
            local.x = v75;
            local.width = right - local.x;
        }
        // Snap right edge to rectWidth, 75%, 25%
        if (Math.abs(rectWidth - right) <= thresholdLocal) {
            local.width = rectWidth - local.x;
        } else if (Math.abs(v75 - right) <= thresholdLocal) {
            local.width = v75 - local.x;
        } else if (Math.abs(v25 - right) <= thresholdLocal) {
            local.width = v25 - local.x;
        }
        // Snap top edge to 0, 25%, 75%
        if (Math.abs(top - 0) <= thresholdLocal) {
            local.y = 0;
            local.height = bottom - local.y;
        } else if (Math.abs(top - h25) <= thresholdLocal) {
            local.y = h25;
            local.height = bottom - local.y;
        } else if (Math.abs(top - h75) <= thresholdLocal) {
            local.y = h75;
            local.height = bottom - local.y;
        }
        // Snap bottom edge to rectHeight, 75%, 25%
        if (Math.abs(rectHeight - bottom) <= thresholdLocal) {
            local.height = rectHeight - local.y;
        } else if (Math.abs(h75 - bottom) <= thresholdLocal) {
            local.height = h75 - local.y;
        } else if (Math.abs(h25 - bottom) <= thresholdLocal) {
            local.height = h25 - local.y;
        }

        // Convert back to absolute space
        let adjusted = {
            ...newBox,
            x: position.x + local.x * scale,
            y: position.y + local.y * scale,
            width: local.width * scale,
            height: local.height * scale,
        };


        // Prevent negative or zero sizes in absolute space just in case
        const MIN_SIZE_ABS = 1e-3;
        if (adjusted.width < MIN_SIZE_ABS) adjusted.width = MIN_SIZE_ABS;
        if (adjusted.height < MIN_SIZE_ABS) adjusted.height = MIN_SIZE_ABS;

        return adjusted;
    }, [rectWidth, rectHeight, scale, position.x, position.y, isRotating, aspectRatio]);

    // Create canvas once
    useEffect(() => {
        if (!canvasRef.current) {
            canvasRef.current = document.createElement('canvas');
        }
        return () => {
            canvasRef.current = null;
            originalFrameRef.current = null;
            processingCanvasRef.current = null;
        };
    }, []);

    useEffect(() => {
        if (!isSelected) return;
        const tr = transformerRef.current;
        const img = imageRef.current;
        if (!tr || !img) return;
        const raf = requestAnimationFrame(() => {
            tr.nodes([img]);
            if (typeof (tr as any).forceUpdate === 'function') {
                (tr as any).forceUpdate();
            }
            tr.getLayer()?.batchDraw?.();
        });
        return () => cancelAnimationFrame(raf);
    }, [isSelected]);

    useEffect(() => {
        let cancelled = false;
        if (lastSelectedSrcRef.current === selectedSrc) return;
        lastSelectedSrcRef.current = selectedSrc;
        // Force redraw on source switch: reset last rendered frame and clear cached original frame
        lastRenderedFrameRef.current = -1;
        originalFrameRef.current = null;
        // @ts-ignore
        iteratorRef.current?.return?.();
        iteratorRef.current = null;
        let info = getMediaInfoCached(selectedSrc);
        if (!info) {
        (async () => {
            try {
                if (!info) info = await getMediaInfo(selectedSrc);
                if (!cancelled) {
                    mediaInfo.current = info;
                    // Media info arrived; force immediate redraw
                    lastRenderedFrameRef.current = -1;
                    try { void (seekAndDrawRef.current?.()); } catch {}
                }
            } catch (e) {
                console.error(e);
                }
            })();
        } else {
            mediaInfo.current = info;
            // Have cached info; force immediate redraw
            lastRenderedFrameRef.current = -1;
            try { void (seekAndDrawRef.current?.()); } catch {}
        }
        return () => { cancelled = true };
    }, [selectedSrc]);

    // Compute aspect-fit display size and offsets within the preview rect
    const {displayWidth, displayHeight, offsetX, offsetY} = useMemo(() => {
        const originalWidth = mediaInfo.current?.video?.displayWidth || 0;
        const originalHeight = mediaInfo.current?.video?.displayHeight || 0;
        if (!originalWidth || !originalHeight || !rectWidth || !rectHeight) {
            return { displayWidth: 0, displayHeight: 0, offsetX: 0, offsetY: 0 };
        }
        const aspectRatio = originalWidth / originalHeight;
        let dw = rectWidth;
        let dh = rectHeight;
        if (rectWidth / rectHeight > aspectRatio) {
            dw = rectHeight * aspectRatio;
        } else {
            dh = rectWidth / aspectRatio;
        }
        const ox = (rectWidth - dw) / 2;
        const oy = (rectHeight - dh) / 2;
        return { displayWidth: dw, displayHeight: dh, offsetX: ox, offsetY: oy };
    }, [mediaInfo.current?.video?.displayWidth, mediaInfo.current?.video?.displayHeight, rectWidth, rectHeight]);

    // Initialize default transform if missing
    useEffect(() => {
        if (!clipTransform && displayWidth && displayHeight) {
            setClipTransform(clipId, { x: offsetX, y: offsetY, width: displayWidth, height: displayHeight, scaleX: 1, scaleY: 1, rotation: 0 });
        }
    }, [clipTransform, displayWidth, displayHeight, offsetX, offsetY, clipId, setClipTransform]);

    // Ensure canvas matches display size for crisp rendering
    useEffect(() => {
        if (!canvasRef.current) return;
        if (!displayWidth || !displayHeight) return;
        const canvas = canvasRef.current;
        const w = Math.floor(displayWidth);
        const h = Math.floor(displayHeight);
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
    }, [displayWidth, displayHeight]);

    const ensureProcessingCanvas = useCallback((width: number, height: number) => {
        let canvas = processingCanvasRef.current;
        if (!canvas) {
            canvas = document.createElement('canvas');
            processingCanvasRef.current = canvas;
        }
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        return canvas;
    }, []);

    const drawWrappedCanvas = useCallback((wc: WrappedCanvas, maskFrame?: number) => {
        let canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.imageSmoothingEnabled = true;
        // @ts-ignore
        ctx.imageSmoothingQuality = 'high';
        try { ctx.drawImage(wc.canvas, 0, 0, canvas.width, canvas.height); } catch {}
        
        // Store the original unfiltered frame for filter adjustments while paused
        if (!originalFrameRef.current) {
            originalFrameRef.current = document.createElement('canvas');
        }
        if (originalFrameRef.current.width !== canvas.width || originalFrameRef.current.height !== canvas.height) {
            originalFrameRef.current.width = canvas.width;
            originalFrameRef.current.height = canvas.height;
        }

        const origCtx = originalFrameRef.current.getContext('2d');
        if (origCtx) {
            origCtx.clearRect(0, 0, canvas.width, canvas.height);
            origCtx.drawImage(canvas, 0, 0);
        }

        const workingCanvas = ensureProcessingCanvas(canvas.width, canvas.height);
        const workingCtx = workingCanvas.getContext('2d');
        if (!workingCtx) return;
        
        workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
        workingCtx.drawImage(canvas, 0, 0);
        
        // Apply masks before running filters/applicators so downstream operations see masked pixels
        const maskedCanvas = applyMask(workingCanvas, maskFrame);
        if (maskedCanvas !== workingCanvas) {
            workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
            try { workingCtx.drawImage(maskedCanvas, 0, 0, workingCanvas.width, workingCanvas.height); } catch {}
        }
        
        // Apply WebGL filters for better performance (fast enough for real-time playback)
        // Use ref values to avoid callback recreation on filter/applicator changes
        applyFilters(workingCanvas, filterParamsRef.current);
        
        // Apply applicators to canvas
        let processedCanvas = workingCanvas;
        for (const applicator of applicatorsRef.current) {
            const result = applicator.apply(processedCanvas);
            // Ensure result is copied back to working canvas for chaining
            if (result !== processedCanvas) {
                workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
                workingCtx.drawImage(result, 0, 0, workingCanvas.width, workingCanvas.height);
                processedCanvas = workingCanvas;
            }
        }
        
        // Always draw the final processed result back to display canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(processedCanvas, 0, 0, canvas.width, canvas.height);
        
        imageRef.current?.getLayer()?.batchDraw?.();
    }, [applyFilters, applyMask, applicators.length, ensureProcessingCanvas]);

    const seekAndDraw = useCallback(async () => {
        if (!canvasRef.current) return;
        if (!mediaInfo) return;
        if (!displayWidth || !displayHeight) return;
        // If playback has started, do not run paused seek rendering to avoid cancelling live decode
        if (useControlsStore.getState().isPlaying) return;
        const clipFps = mediaInfo.current?.stats.video?.averagePacketRate || fps || 0;
        const projectFps = fps || 0;
        if (!Number.isFinite(clipFps) || clipFps <= 0) return;
        if (!Number.isFinite(projectFps) || projectFps <= 0) return;
        
        // Map from project fps space to native clip fps space
        const isUsingPreprocessorSrc = selectedSrc !== src;
        const adjustedCurrentFrame = isUsingPreprocessorSrc ? currentFrame - (trimStart || 0) : currentFrame;
        const idealFrame = Math.max(0, adjustedCurrentFrame - frameOffset) * Math.max(0.1, speed);
        const actualFrame = Math.round((idealFrame / projectFps) * clipFps);
        const totalFrames = Math.max(0, Math.floor((mediaInfo.current?.duration || 0) * clipFps));
        const targetFrame = Math.max(0, Math.min(totalFrames, actualFrame)) + Math.round(((mediaInfo.current?.startFrame || 0) / projectFps) * clipFps);
        
        // Skip if we already rendered this frame
        if (lastRenderedFrameRef.current === targetFrame) return;
        
        // Cancel any ongoing paused seek operations (do not interfere with live decode token)
        const myToken = ++drawTokenRef.current;
        
        try {
            // Use direct frame fetch for much faster seeking
            const wc = await fetchCanvasSample(selectedSrc, targetFrame, displayWidth, displayHeight, { mediaInfo: mediaInfo.current || undefined });
            
            // Check if operation was cancelled
            if (myToken !== drawTokenRef.current) return;
            
            if (wc) {
                drawWrappedCanvas(wc, maskFrameForCurrentFocus);
                lastRenderedFrameRef.current = targetFrame;
            }
        } catch (e) {
            console.warn('[video] seek draw failed', e);
        }
    }, [mediaInfo, fps, selectedSrc, src, displayWidth, displayHeight, currentFrame, drawWrappedCanvas, speed, frameOffset, trimStart, maskFrameForCurrentFocus, clip?.masks, clip?.preprocessors, applicators.length]);

    const startRendering = useCallback(async () => {
        if (!canvasRef.current) return;
        if (!mediaInfo.current) return;
        if (!displayWidth || !displayHeight) return;
        const clipFps = mediaInfo.current?.stats.video?.averagePacketRate || fps || 0;
        const projectFps = fps || 0;
        if (!Number.isFinite(clipFps) || clipFps <= 0) return;
        if (!Number.isFinite(projectFps) || projectFps <= 0) return;
        
        // Map from project fps space to native clip fps space
        // When using a preprocessor src, we need to subtract trimStart since it's already in currentFrame
        const isUsingPreprocessorSrc = selectedSrc !== src;
        const adjustedCurrentFrame = isUsingPreprocessorSrc ? currentFrame - (trimStart || 0) : currentFrame;
        const idealStartFrame = Math.max(0, adjustedCurrentFrame - frameOffset) * Math.max(0.1, speed);
        const actualStartFrame = Math.round((idealStartFrame / projectFps) * clipFps);
        const totalFrames = Math.max(0, Math.floor((mediaInfo.current?.duration || 0) * clipFps));
        const startIdx = Math.max(0, Math.min(totalFrames, actualStartFrame)) + Math.round(((mediaInfo.current?.startFrame || 0) / projectFps) * clipFps);
        currentStartFrameRef.current = startIdx;
        lastRenderedFrameRef.current = startIdx - 1;

        const myToken = ++drawTokenRef.current;
        // @ts-ignore
        iteratorRef.current?.return?.();
        const targetEndFrame = mediaInfo.current?.endFrame ? Math.round((mediaInfo.current?.endFrame || 0) / projectFps * clipFps) : undefined;

        iteratorRef.current = await getVideoIterator(selectedSrc, { mediaInfo: mediaInfo.current || undefined, fps: clipFps, startIndex: startIdx, endIndex: targetEndFrame });

        try {
            for await (const wc of iteratorRef.current as AsyncIterable<WrappedCanvas | null>) {
                if (myToken !== drawTokenRef.current) break;
                if (!useControlsStore.getState().isPlaying) break;
                if (!wc) continue;

                

                // Determine the decoded sample's frame index in native fps
                const ts: number | undefined = (wc as any)?.timestamp;

                // Use floor with a tiny epsilon to avoid boundary flip-flop
                let sampleIdx = Number.isFinite(ts as number)
                    ? Math.floor(((ts as number) * clipFps) + 1e-4)
                    : (lastRenderedFrameRef.current + 1);

                // Compute current timeline-local frame mapped to native fps (clip space)
                const computeLocalFocusMedia = () => {
                    const store = useControlsStore.getState();
                    // Base timeline-local frames relative to clip start (no give-start applied)
                    const baseLocal = Math.max(0, ((store.focusFrame || 0) - startFrame));
                    // When using preprocessor src, align to its own frame space by subtracting its start offset.
                    // Otherwise, include trimStart to match the main clip's reference frame.
                    const localProjectFrames = isUsingPreprocessorSrc
                        ? Math.max(0, baseLocal - Math.max(0, frameOffset))
                        : Math.max(0, baseLocal + (trimStart || 0));
                    const speedAdjusted = Math.max(0, localProjectFrames * Math.max(0.1, speed));
                    // Map from project fps to native fps using floor to reduce jitter
                    const actualFrameIdx = Math.floor(((speedAdjusted / projectFps) * clipFps) + 1e-4);
                    return actualFrameIdx + Math.round(((mediaInfo.current?.startFrame || 0) / projectFps) * clipFps);
                };

                // Skip stale frames that are behind the timeline by more than 1 frame
                let localFocus = computeLocalFocusMedia();
                if (sampleIdx < localFocus - 1) {
                    lastRenderedFrameRef.current = sampleIdx;
                    continue;
                }

                // If we're ahead of the timeline, wait until the timeline catches up (sync to rAF)
                while (sampleIdx > (localFocus = computeLocalFocusMedia())) {
                    if (myToken !== drawTokenRef.current) break;
                    if (!useControlsStore.getState().isPlaying) break;
                    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
                }

                if (myToken !== drawTokenRef.current)  break;
                if (!useControlsStore.getState().isPlaying) break;

                const storeState = useControlsStore.getState();
                const focusFrameForMask = storeState.focusFrame || 0;
                const maskFrameBase = Math.max(0, Math.floor(focusFrameForMask - startFrame + (trimStart || 0)));
                const maskFrame = clip ? Math.max(0, Math.floor(getLocalFrame(focusFrameForMask, clip))) : maskFrameBase;

                drawWrappedCanvas(wc as WrappedCanvas, maskFrame);
                lastRenderedFrameRef.current = sampleIdx;
            }
        } catch (e) {
            // swallow
        }
    }, [mediaInfo, fps, selectedSrc,  displayWidth, displayHeight, currentFrame, drawWrappedCanvas, speed, startFrame, frameOffset, trimStart, clip]);


    // Start/stop iterator based on play state. Avoid depending on callbacks to prevent restarting every frame.
    useEffect(() => {
        if (isPlaying) {
            void startRendering();
        } else {
            void seekAndDraw();
        }
        return () => {
            drawTokenRef.current++;
            // @ts-ignore
            iteratorRef.current?.return?.();
        };
    }, [isPlaying, selectedSrc, mediaInfo, displayWidth, displayHeight, fps, speed, frameOffset, applicators.length]);

    // If video is paused, reapply filters and applicators when they change
    useEffect(() => {
        if (!isPlaying && canvasRef.current && imageRef.current) {
            // If we have an original frame cached, use it for fast reapplication
            if (originalFrameRef.current) {
                let canvas = canvasRef.current;
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    const workingCanvas = ensureProcessingCanvas(canvas.width, canvas.height);
                    const workingCtx = workingCanvas.getContext('2d');
                    if (!workingCtx) return;
                    
                    // Start with the original unfiltered frame
                    workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
                    workingCtx.drawImage(originalFrameRef.current, 0, 0);

                    // Apply masks before filters so masked pixels feed the rest of the pipeline
                    const maskedCanvas = applyMask(workingCanvas, maskFrameForCurrentFocus);
                    if (maskedCanvas !== workingCanvas) {
                        workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
                        workingCtx.drawImage(maskedCanvas, 0, 0, workingCanvas.width, workingCanvas.height);
                    }
                    
                    // Apply filters to the clean frame
                    applyFilters(workingCanvas, filterParamsRef.current);
                    
                    // Apply applicators (filter clips from layers above)
                    let processedCanvas = workingCanvas;
                    for (const applicator of applicatorsRef.current) {
                        const result = applicator.apply(processedCanvas);
                        if (result !== processedCanvas) {
                            workingCtx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
                            workingCtx.drawImage(result, 0, 0, workingCanvas.width, workingCanvas.height);
                            processedCanvas = workingCanvas;
                        }
                    }
                    
                    // Always draw final result back to display canvas
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(processedCanvas, 0, 0, canvas.width, canvas.height);
                    
                    imageRef.current.getLayer()?.batchDraw();
                }
            } else {
                // If no cached frame exists, decode the current frame
                // Force re-decode even if we already rendered this frame index
                lastRenderedFrameRef.current = -1;
                void seekAndDraw();
            }
        }
    }, [clip?.brightness, clip?.contrast, clip?.hue, clip?.saturation, clip?.blur, clip?.sharpness, clip?.noise, clip?.vignette, isPlaying, applyFilters, applicators, applicators.length, applyMask, maskFrameForCurrentFocus, seekAndDraw, ensureProcessingCanvas]);

    // Ensure any CLUTs needed by filter applicators are preloaded before drawing
    useEffect(() => {
        let cancelled = false;
        const maybePreload = async () => {
            const preloadTasks: Promise<void>[] = [];
            for (const app of applicatorsRef.current) {
                const maybeEnsure = (app as any)?.ensureResources as (() => Promise<void>) | undefined;
                if (typeof maybeEnsure === 'function') {
                    preloadTasks.push(maybeEnsure());
                }
            }
            if (preloadTasks.length) {
                try {
                    await Promise.all(preloadTasks);
                } catch {}
            }
            if (cancelled) return;
            // After resources are ready, force redraw immediately
            if (canvasRef.current) {
                lastRenderedFrameRef.current = -1;
                if (!isPlaying) {
                    void seekAndDrawRef.current();
                }
                imageRef.current?.getLayer()?.batchDraw?.();
            }
        };
        void maybePreload();
        return () => { cancelled = true };
    }, [applicators, applicators.length, isPlaying]);

    // Use ref to store the latest seekAndDraw to avoid throttle recreation
    const seekAndDrawRef = useRef(seekAndDraw);
    seekAndDrawRef.current = seekAndDraw;

    // Create a stable throttled version for smoother scrubbing
    const throttledSeekAndDraw = useMemo(() => {
        return _.throttle(() => {
            void seekAndDrawRef.current();
        }, 16, { leading: true, trailing: true }); // ~60fps throttle
    }, []);

    // If playback starts, cancel any pending throttled seek callbacks to avoid cancelling live decode
    useEffect(() => {
        if (isPlaying) {
            throttledSeekAndDraw.cancel();
        }
    }, [isPlaying, throttledSeekAndDraw]);

    // While paused, redraw on scrubs/jumps with throttling
    useEffect(() => {
        if (isPlaying) return;
        throttledSeekAndDraw();
    }, [currentFrame, isPlaying, throttledSeekAndDraw]);

    // Cleanup throttled function on unmount
    useEffect(() => {
        return () => {
            throttledSeekAndDraw.cancel();
        };
    }, [throttledSeekAndDraw]);

    // While playing, do not restart iterator on every focusFrame tick; decoding loop drives rendering.

    const handleDragMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        updateGuidesAndMaybeSnap({ snap: true });
        const node = imageRef.current;
        if (node) {
            setClipTransform(clipId, { x: node.x(), y: node.y() });
        } else {
            setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
        }
    }, [setClipTransform, clipId, updateGuidesAndMaybeSnap]);

    const handleDragStart = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        e.target.getStage()!.container().style.cursor = 'grab';
        addClipSelection(clipId);
        const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
        setIsInteracting(true);
        updateGuidesAndMaybeSnap({ snap: true });
    }, [clipId, addClipSelection, updateGuidesAndMaybeSnap]);

    const handleDragEnd = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        e.target.getStage()!.container().style.cursor = 'default';
        const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
        setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
        setIsInteracting(false);
        setGuides({ vCenter: false, hCenter: false, v25: false, v75: false, h25: false, h75: false, left: false, right: false, top: false, bottom: false });
    }, [setClipTransform, clipId]);

    const handleClick = useCallback(() => {
        if (isFullscreen) return;
        clearSelection();
        addClipSelection(clipId);
    }, [addClipSelection, clipId, isFullscreen]);

    useEffect(() => {
        const transformer = transformerRef.current;
        if (!transformer) return;
        const bumpSuppress = () => {
            const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
            suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 300);
        };
        const onTransformStart = () => {
            bumpSuppress();
            setIsTransforming(true);
            const active = (transformer as any)?.getActiveAnchor?.();
            const rotating = typeof active === 'string' && active.includes('rotater');
            setIsRotating(!!rotating);
            setIsInteracting(true);
            if (!rotating) {
                updateGuidesAndMaybeSnap({ snap: false });
            } else {
                setGuides({ vCenter: false, hCenter: false, v25: false, v75: false, h25: false, h75: false, left: false, right: false, top: false, bottom: false });
            }
        };
        const persistTransform = () => {
            const node = imageRef.current;
            if (!node) return;
            const newWidth = node.width() * node.scaleX();
            const newHeight = node.height() * node.scaleY();
            setClipTransform(clipId, {
                x: node.x(),
                y: node.y(),
                width: newWidth,
                height: newHeight,
                scaleX: 1,
                scaleY: 1,
                rotation: node.rotation(),
            });
            node.width(newWidth);
            node.height(newHeight);
            node.scaleX(1);
            node.scaleY(1);
        };
        const onTransform = () => {
            bumpSuppress();
            if (!isRotating) {
                updateGuidesAndMaybeSnap({ snap: false });
            }
            persistTransform();
        };
        const onTransformEnd = () => {
            bumpSuppress();
            setIsTransforming(false);
            setIsInteracting(false);
            setIsRotating(false);
            setGuides({ vCenter: false, hCenter: false, v25: false, v75: false, h25: false, h75: false, left: false, right: false, top: false, bottom: false });
            persistTransform();
        };
        transformer.on('transformstart', onTransformStart);
        transformer.on('transform', onTransform);
        transformer.on('transformend', onTransformEnd);
        return () => {
            transformer.off('transformstart', onTransformStart);
            transformer.off('transform', onTransform);
            transformer.off('transformend', onTransformEnd);
        };
    }, [transformerRef.current, updateGuidesAndMaybeSnap, setClipTransform, clipId, isRotating]);

    useEffect(() => {
        const handleWindowClick = (e: MouseEvent) => {
            if (!isSelected) return;
            const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
            if (now < suppressUntilRef.current) return;
            const stage = imageRef.current?.getStage();
            const container = stage?.container();
            // check that node is inside container
            const node = e.target;
            if (!container?.contains(node as Node)) return;
            if (!stage || !container || !imageRef.current) return;
            const containerRect = container.getBoundingClientRect();
            const pointerX = e.clientX - containerRect.left;
            const pointerY = e.clientY - containerRect.top;
            const imgRect = imageRef.current.getClientRect({ skipShadow: true, skipStroke: true });
            const insideImage = pointerX >= imgRect.x && pointerX <= imgRect.x + imgRect.width && pointerY >= imgRect.y && pointerY <= imgRect.y + imgRect.height;
            
            if (!insideImage) {
                removeClipSelection(clipId);
            }
        };
        window.addEventListener('click', handleWindowClick);
        return () => {
            window.removeEventListener('click', handleWindowClick);
        };
    }, [clipId, isSelected, removeClipSelection]);
    

  return (
    <React.Fragment>
    <Group ref={groupRef} clipX={0} clipY={0} clipWidth={rectWidth} clipHeight={rectHeight}>
      <Image 
      draggable={tool === 'pointer' && !isTransforming } 
      ref={imageRef}  
      image={canvasRef.current || undefined}
       x={clipTransform?.x ?? offsetX} 
       y={clipTransform?.y ?? offsetY} 
       width={clipTransform?.width ?? displayWidth} 
       height={clipTransform?.height ?? displayHeight} 
       scaleX={clipTransform?.scaleX ?? 1}
       scaleY={clipTransform?.scaleY ?? 1}
       rotation={clipTransform?.rotation ?? 0}
       cornerRadius={clipTransform?.cornerRadius ?? 0}
       opacity={(clipTransform?.opacity ?? 100) / 100}
       onDragMove={handleDragMove} 
       onDragStart={handleDragStart} 
       onDragEnd={handleDragEnd} 
       onClick={handleClick} 
       />
      {tool === 'pointer' && isSelected && isInteracting && !isRotating && !isFullscreen && (
        <React.Fragment>
          {guides.vCenter && <Line listening={false} points={[rectWidth/2, 0, rectWidth/2, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.v25 && <Line listening={false} points={[rectWidth*0.25, 0, rectWidth*0.25, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.v75 && <Line listening={false} points={[rectWidth*0.75, 0, rectWidth*0.75, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.hCenter && <Line listening={false} points={[0, rectHeight/2, rectWidth, rectHeight/2]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.h25 && <Line listening={false} points={[0, rectHeight*0.25, rectWidth, rectHeight*0.25]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.h75 && <Line listening={false} points={[0, rectHeight*0.75, rectWidth, rectHeight*0.75]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.left && <Line listening={false} points={[0, 0, 0, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.right && <Line listening={false} points={[rectWidth, 0, rectWidth, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.top && <Line listening={false} points={[0, 0, rectWidth, 0]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.bottom && <Line listening={false} points={[0, rectHeight, rectWidth, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
        </React.Fragment>
      )}
    </Group>
    <Transformer 
        borderStroke='#AE81CE'
        anchorCornerRadius={8} 
        anchorStroke='#E3E3E3' 
        anchorStrokeWidth={1}
        borderStrokeWidth={2}
        visible={tool === 'pointer' && isSelected && !isFullscreen && overlap}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]} 
        boundBoxFunc={transformerBoundBoxFunc as any}
        ref={(node) => {
            transformerRef.current = node;
            if (node && imageRef.current) {
                node.nodes([imageRef.current]);
                if (typeof (node as any).forceUpdate === 'function') {
                    (node as any).forceUpdate();
                }
                node.getLayer()?.batchDraw?.();
            }
        }} 
        enabledAnchors={['top-left', 'bottom-right', 'top-right', 'bottom-left']} />
    </React.Fragment>
  )
}

export default VideoPreview
