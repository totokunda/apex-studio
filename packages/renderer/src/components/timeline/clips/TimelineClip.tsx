import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useClipStore, getClipWidth, getClipX, isValidTimelineForClip, getTimelineTypeForClip } from "@/lib/clip";
import { generateTimelineSamples } from "@/lib/media/timeline";
import { getNearestCachedCanvasSamples } from "@/lib/media/canvas";
import { useControlsStore } from "@/lib/control";
import { Image, Group, Rect, Text } from 'react-konva';
import Konva from 'konva';
import { MediaInfo, ShapeClipProps, TimelineProps } from "@/lib/types";
import { v4 as uuidv4 } from 'uuid';
import { getMediaInfoCached } from "@/lib/media/utils";

const THUMBNAIL_TILE_SIZE = 36;

const TimelineClip: React.FC<TimelineProps & {clipId: string, clipType: 'video' | 'audio' | 'image' | 'shape',  scrollY: number}> = ({timelineWidth = 0, timelineY = 0, timelineHeight = 64, timelinePadding = 24, clipId,  timelineId, clipType, scrollY}) => {
    // Select only what we need to avoid unnecessary rerenders
    const timelineDuration = useControlsStore((s) => s.timelineDuration);
    const selectedClipIds = useControlsStore((s) => s.selectedClipIds);
    const toggleClipSelection = useControlsStore((s) => s.toggleClipSelection);
    const zoomLevel = useControlsStore((s) => s.zoomLevel);
    const resizeClip = useClipStore((s) => s.resizeClip);
    const moveClipToEnd = useClipStore((s) => s.moveClipToEnd);
    const updateClip = useClipStore((s) => s.updateClip);
    const setGhostX = useClipStore((s) => s.setGhostX);
    const setGhostTimelineId = useClipStore((s) => s.setGhostTimelineId);
    const setGhostStartEndFrame = useClipStore((s) => s.setGhostStartEndFrame);
    const setGhostInStage = useClipStore((s) => s.setGhostInStage);
    const setDraggingClipId = useClipStore((s) => s.setDraggingClipId);
    const getClipsForTimeline = useClipStore((s) => s.getClipsForTimeline);
    const getTimelineById = useClipStore((s) => s.getTimelineById);
    const setHoveredTimelineId = useClipStore((s) => s.setHoveredTimelineId);
    const setSnapGuideX = useClipStore((s) => s.setSnapGuideX);
    const addTimeline = useClipStore((s) => s.addTimeline);
    // Subscribe directly to this clip's data
    const currentClip = useClipStore((s) => s.clips.find((c) => c.clipId === clipId && (timelineId ? c.timelineId === timelineId : true)));
    const cornerRadius = useMemo(() => {
        return currentClip?.type === 'shape' ? 4 : 8;
    }, [currentClip?.type]);
    
    const currentStartFrame = currentClip?.startFrame ?? 0;
    const currentEndFrame = currentClip?.endFrame ?? 0;

    const clipWidth = useMemo(() => Math.max(getClipWidth(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), 3), [currentStartFrame, currentEndFrame, timelineWidth, timelineDuration]);
    const clipX = useMemo(() => getClipX(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), [currentStartFrame, currentEndFrame, timelineWidth, timelineDuration, timelineId]);
    const clipRef = useRef<Konva.Rect>(null);
    const [resizeSide, setResizeSide] = useState<'left' | 'right' | null>(null);
    const [imageCanvas] = useState<HTMLCanvasElement>(() => document.createElement('canvas'));
    const mediaInfoRef = useRef<MediaInfo | undefined>(getMediaInfoCached(currentClip?.src!));
    const dragInitialWindowRef = useRef<[number, number] | null>(null);
    const thumbnailClipWidth = useRef<number>(0);
    const maxTimelineWidth = useMemo(() => (timelineWidth), [timelineWidth, timelinePadding]);
    const groupRef = useRef<Konva.Group>(null);
    const exactVideoUpdateTimerRef = useRef<number | null>(null);
    const exactVideoUpdateSeqRef = useRef(0);
    const lastExactRequestKeyRef = useRef<string | null>(null);
    const textRef = useRef<Konva.Text>(null);
    const [textWidth, setTextWidth] = useState(0);
    const [isDragging, setIsDragging] = useState(false);
    const rectRefLeft = useRef<Konva.Rect>(null);
    const rectRefRight = useRef<Konva.Rect>(null);
    
    // (moved) image positioning is computed after clipPosition is defined

    useEffect(() => {
        imageCanvas.width = Math.min(clipWidth, maxTimelineWidth);
        imageCanvas.height = timelineHeight;
    }, [zoomLevel, clipWidth, timelineHeight, timelineWidth, timelinePadding, maxTimelineWidth, clipType, imageCanvas]);

    useEffect(() => {
        return () => {
            if (exactVideoUpdateTimerRef.current != null) {
                window.clearTimeout(exactVideoUpdateTimerRef.current);
                exactVideoUpdateTimerRef.current = null;
            }
        };
    }, []);


    const restoreWindowIfChanged = useCallback((anchorStartFrame: number) => {
        try {
            const initial = dragInitialWindowRef.current;
            if (!initial) return;
            const [initStart, initEnd] = initial;
            const controls = useControlsStore.getState();
            const [curStart, curEnd] = controls.timelineDuration;
            if (initStart === curStart && initEnd === curEnd) return;
            const originalWindowLen = Math.max(1, (initEnd - initStart));
            const desiredStart = anchorStartFrame;
            const desiredEnd = desiredStart + originalWindowLen;
            if (controls.totalTimelineFrames < desiredEnd) {
                controls.incrementTotalTimelineFrames(desiredEnd - controls.totalTimelineFrames);
            }
            controls.setTimelineDuration(desiredStart, desiredEnd);
            if ((controls as any).setFocusFrame) {
                (controls as any).setFocusFrame(desiredStart);
            }
        } finally {
            dragInitialWindowRef.current = null;
        }
    }, []);

    // Use global selection state instead of local state
    const currentClipId = clipId;
    const isSelected = selectedClipIds.includes(currentClipId);

    const [clipPosition, setClipPosition] = useState<{x:number, y:number}>({
        x: clipX + timelinePadding,
        y: timelineY - timelineHeight
    });
    const [tempClipPosition, setTempClipPosition] = useState<{x:number, y:number}>({
        x: clipX + timelinePadding,
        y: timelineY - timelineHeight
    });
    

    // Width used for the thumbnail image we render inside the clip group.
    const imageWidth = useMemo(() => Math.min(clipWidth, maxTimelineWidth), [clipWidth, maxTimelineWidth]);

    // Compute image x so that the image stays centered over the portion of the
    // group that is currently visible inside the stage viewport. This allows us
    // to "virtually" pan across long clips without rendering an infinitely wide image.
    const overHang = useMemo(() => {
        let overhang = 0;
        const positionX = clipPosition.x == 24 || clipWidth <= imageWidth? 0:  (-clipPosition.x);
        
        if (clipWidth - positionX <= timelineWidth && positionX > 0) {
            overhang = timelineWidth - (clipWidth - positionX);
        }
        return overhang;
    }, [clipPosition.x, clipWidth, timelineWidth]);

    const imageX = useMemo(() => {
        let overhang = 0;
        // Default behavior for clips that fit within timeline or are at the start
        const positionX = clipPosition.x == 24 || clipWidth <= imageWidth? 0:  (-clipPosition.x);
        if (clipWidth - positionX <= timelineWidth && positionX > 0) {
            overhang = timelineWidth - (clipWidth - positionX);
        }

        const imageX = positionX - overhang;        
        return Math.max(0, imageX);
    }, [clipPosition.x, clipWidth, timelinePadding, timelineWidth, imageWidth]);


    useEffect(() => {
        thumbnailClipWidth.current = Math.max(getClipWidth(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), 3);
    }, [zoomLevel, timelineWidth, clipType, timelineDuration, currentClip?.framesToGiveStart, currentClip?.framesToGiveEnd, currentClip?.startFrame, currentClip?.endFrame]);

    useEffect(() => {
        setClipPosition({x: clipX + timelinePadding, y: timelineY - timelineHeight});
    }, [timelinePadding, timelineY, timelineId]);

    useEffect(() => {
        if (textRef.current) {
            setTextWidth(textRef.current.width());
        }
    }, [(currentClip as ShapeClipProps)?.shapeType]);

    

	useEffect(() => {
		if (!currentClip) return;

        const generateTimelineThumbnailAudio = async () => {
            if (clipType !== 'audio') return;
    
            const width = mediaInfoRef.current?.stats.audio?.averagePacketRate ?? 1;
            const height = timelineHeight;
            
            // Get the speed factor - higher speed means more audio content in same duration
            const speed = Math.max(0.1, Math.min(5, Number((currentClip as any)?.speed ?? 1)));
            
            // Calculate the visible clip duration (what's shown, accounting for trimming)
            const visibleClipDurationFrames = currentEndFrame - currentStartFrame;
            
            // The waveform canvas width
            const tClipWidth = imageCanvas.width;
            
            // Calculate which portion of the source audio to extract
            // Account for framesToGiveStart (trim from beginning)
            const trimStartFrames = (currentClip.framesToGiveStart ?? 0);
            const audioStartFrame = Math.round(trimStartFrames * speed);
            
            // Calculate end frame: start + visible duration, accounting for speed
            const audioEndFrame = Math.round(audioStartFrame + (visibleClipDurationFrames * speed));

            const samples = await generateTimelineSamples(
                currentClipId,
                currentClip?.src!,
                [0],
                width,
                height,
                tClipWidth,
                {
                    mediaInfo: mediaInfoRef.current,
                    startFrame: audioStartFrame,
                    endFrame: audioEndFrame,
                    volume: (currentClip as any)?.volume,
                    fadeIn: (currentClip as any)?.fadeIn,
                    fadeOut: (currentClip as any)?.fadeOut,
                }
            );

  
            if (samples?.[0]?.canvas) {
                const inputCanvas = samples?.[0]?.canvas as HTMLCanvasElement;

                const ctx = imageCanvas.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                    
                    // Truncate the waveform to fit the canvas without compression
                    const sourceWidth = Math.min(inputCanvas.width, imageCanvas.width);
                    const sourceHeight = Math.min(inputCanvas.height, imageCanvas.height);

                    // When resizing from left, we want to show the end portion of the waveform
                    let sourceX = 0;
                    if (resizeSide === 'left' && inputCanvas.width > imageCanvas.width) {
                        sourceX = inputCanvas.width - imageCanvas.width;
                    }

                    // Draw without scaling - truncate only
                    ctx.drawImage(
                        inputCanvas,
                        sourceX, 0, sourceWidth, sourceHeight, // source rectangle (truncate)
                        0, 0, sourceWidth, sourceHeight        // destination rectangle (no scale)
                    );
                }

            }
            moveClipToEnd(currentClipId);
        }

        const generateTimelineThumbnailImage = async () => {
            if (clipType !== 'image') return;
            const tClipWidth = Math.min(thumbnailClipWidth.current, maxTimelineWidth);
            const width = mediaInfoRef.current?.image?.width ?? 1;
            const height = mediaInfoRef.current?.image?.height ?? 1;
            const ratio = width / height;
            let thumbnailWidth = Math.max(timelineHeight * ratio, THUMBNAIL_TILE_SIZE);
    
            const samples = await generateTimelineSamples(
                currentClipId,
                currentClip?.src!,
                [0],
                thumbnailWidth,
                timelineHeight,
                tClipWidth,
                {
                    mediaInfo: mediaInfoRef.current,
                }
            );
    
            if (samples?.[0]?.canvas) {
                const inputCanvas = samples?.[0]?.canvas as HTMLCanvasElement;
    
                const ctx = imageCanvas.getContext('2d');
                if (ctx) {
                    
                    const targetWidth = Math.max(1, imageCanvas.width);
                    const targetHeight = Math.max(1, imageCanvas.height);
                    ctx.clearRect(0, 0, targetWidth, targetHeight);


                    // Determine tile dimensions from the input canvas/image
                    const tileWidth = Math.max(1, (inputCanvas as any).width || (inputCanvas as any).naturalWidth || 1);
                    const tileHeight = Math.max(1, (inputCanvas as any).height || (inputCanvas as any).naturalHeight || 1);
                    const sourceHeight = Math.min(tileHeight, targetHeight);

                    // When resizing from the left, offset the tiling pattern so new tiles appear from the left
                    let startX = 0;
                    if (resizeSide === 'left') {
                        // Calculate offset so the pattern appears anchored to the right
                        const remainder = targetWidth % tileWidth;
                        startX = remainder > 0 ? -(tileWidth - remainder) : 0;
                    }
                    
                    // Repeat the inputCanvas horizontally until we fill the target width
                    let x = startX;
                    while (x < targetWidth) {
                        const remaining = targetWidth - x;
                        const drawWidth = Math.min(tileWidth, remaining);
                        
                        // Only draw if the tile is visible (x + drawWidth > 0)
                        if (x + drawWidth > 0) {
                            ctx.drawImage(
                                inputCanvas,
                                x, 0, drawWidth, sourceHeight
                            );
                        }
                        x += drawWidth;
                    }
                }
            }
            clipRef.current?.getLayer()?.batchDraw();
            moveClipToEnd(currentClipId);
        }

        const generateTimelineThumbnailVideo = async () => {
            if (clipType !== 'video') return;
            let tClipWidth = Math.min(thumbnailClipWidth.current, maxTimelineWidth);
            const width = mediaInfoRef.current?.video?.codedWidth ?? 1;
            const height = mediaInfoRef.current?.video?.codedHeight ?? 1;
            const ratio = width / height;
            const thumbnailWidth = Math.max(timelineHeight * ratio, THUMBNAIL_TILE_SIZE);
            
            const mediaStartFrame = mediaInfoRef.current?.startFrame ?? 0;
            const mediaEndFrame = mediaInfoRef.current?.endFrame;
            const speed = Math.max(0.1, Math.min(5, Number((currentClip as any)?.speed ?? 1)));
            
            // Calculate frame indices based on timeline duration and available columns
            let numColumns = Math.ceil((tClipWidth - overHang) / thumbnailWidth);

            let numColumnsAlt:number | undefined = undefined;
            if (currentClip.framesToGiveStart || currentClip.framesToGiveEnd) {
                const realStartFrame = currentStartFrame - (currentClip.framesToGiveStart ?? 0);
                const realEndFrame = currentEndFrame - (currentClip.framesToGiveEnd ?? 0);
                tClipWidth = Math.max(getClipWidth(realStartFrame, realEndFrame, timelineWidth, timelineDuration), 3);
                const tempNumColumns = Math.ceil((tClipWidth - overHang) / thumbnailWidth);
                numColumnsAlt = numColumns;
                numColumns = tempNumColumns;
            }
            const timelineShift = currentStartFrame - (currentClip.framesToGiveStart ?? 0);
            const realStartFrame = timelineShift;
            const realEndFrame = currentEndFrame - (currentClip.framesToGiveEnd ?? 0);
            let timelineStartFrame = Math.max(timelineDuration[0], realStartFrame);
            let timelineEndFrame = Math.min(timelineDuration[1], realEndFrame);
            const timelineSpan = timelineEndFrame - timelineStartFrame;
            
            let frameIndices: number[];
            let startFrame = realStartFrame;
            if (timelineStartFrame > realStartFrame) {
                startFrame = timelineStartFrame;
            }
            if (timelineSpan >= numColumns && numColumns > 1) {
                // When timeline duration is large enough, space frames evenly
                frameIndices = Array.from({ length: numColumns }, (_, i) => {
                    const progress = i / (numColumns - 1);
                    const frameIndex = Math.round(startFrame + progress * timelineSpan);
                    return frameIndex;
                });
                frameIndices[frameIndices.length - 1] = frameIndices[frameIndices.length - 1] - 1;
            } else if (numColumns > 1) {
                // When timeline duration is less than numColumns, duplicate frames
                frameIndices = Array.from({ length: numColumns }, (_, i) => {
                    const frameIndex = Math.floor(i / Math.ceil(numColumns / (timelineSpan + 1)));
                    const clampedIndex = Math.min(frameIndex, timelineSpan);
                    return timelineStartFrame + clampedIndex;
                });
            } else {
                // Single column case
                frameIndices = [timelineStartFrame];
            }

            frameIndices = frameIndices.filter((frameIndex) => isNaN(frameIndex) === false && isFinite(frameIndex));
            // Map timeline frames to source frames considering speed, in-clip offset, and split bounds
            {
                frameIndices = frameIndices.map((frameIndex) => {
                    const local = Math.max(0, frameIndex - timelineShift);
                    let sourceFrame = Math.floor(local * speed) + mediaStartFrame;
                    if (mediaEndFrame !== undefined) {
                        sourceFrame = Math.min(sourceFrame, mediaEndFrame);
                    }
                    return Math.max(mediaStartFrame, sourceFrame);
                });
            }

            if (numColumnsAlt && frameIndices.length !== numColumnsAlt) {
                // Trim indices to match the original column count, removing from
                // left/right based on framesToGiveStart (left) and abs(framesToGiveEnd) (right)
                if (frameIndices.length > numColumnsAlt) {
                    const surplus = frameIndices.length - numColumnsAlt;
                    const giveStart = Math.max(0, currentClip?.framesToGiveStart ?? 0);
                    const giveEnd = Math.max(0, -(currentClip?.framesToGiveEnd ?? 0));
                    const totalGive = giveStart + giveEnd;
                    let leftRemove = 0;
                    let rightRemove = 0;
                    if (totalGive > 0) {
                        leftRemove = Math.floor((surplus * giveStart) / totalGive);
                        rightRemove = surplus - leftRemove;
                    } else {
                        leftRemove = Math.floor(surplus / 2);
                        rightRemove = surplus - leftRemove;
                    }
                    const start = Math.min(Math.max(0, leftRemove), frameIndices.length);
                    const end = Math.max(start, frameIndices.length - Math.max(0, rightRemove));
                    frameIndices = frameIndices.slice(start, end);
                    // In case of rounding, ensure exact length
                    if (frameIndices.length > numColumnsAlt) {
                        frameIndices = frameIndices.slice(0, numColumnsAlt);
                    }
                }
            }

            if (frameIndices.length === 0) {
                return;
            }

            // 1) Immediate draw using nearest cached frames (synchronous)

            const nearest = getNearestCachedCanvasSamples(
                    currentClip?.src!,
                    frameIndices,
                    thumbnailWidth,
                    timelineHeight,
                    { mediaInfo: mediaInfoRef.current }
                );
                const ctx = imageCanvas.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                    let x = overHang;
                    const targetWidth = Math.max(1, imageCanvas.width);
                    const targetHeight = Math.max(1, imageCanvas.height);

                    // When resizing from the left for video, truncate from the left by
                    // skipping the overflow width from the left side of the tile sequence.
                    let skipRemaining = 0;
                    if (resizeSide === 'left' && clipType === 'video') {
                        let totalTileWidth = 0;
                        for (let i = 0; i < nearest.length; i++) {
                            const sample = nearest[i];
                            if (!sample) { continue; }
                            const anyCanvas = (sample.canvas as any);
                            const tileWidth = Math.max(1, anyCanvas.width || anyCanvas.naturalWidth || 1);
                            totalTileWidth += tileWidth;
                        }
                        const drawableWidth = Math.max(0, targetWidth - x);
                        skipRemaining = Math.max(0, totalTileWidth - drawableWidth);
                    }

                    for (let i = 0; i < nearest.length && x < targetWidth; i++) {
                        const sample = nearest[i];
                        if (!sample) { continue; }
                        const inputCanvas = sample.canvas as HTMLCanvasElement;
                        const anyCanvas = inputCanvas as any;
                        const tileWidth = Math.max(1, anyCanvas.width || anyCanvas.naturalWidth || 1);
                        const tileHeight = Math.max(1, anyCanvas.height || anyCanvas.naturalHeight || 1);
                        const sourceHeight = Math.min(tileHeight, targetHeight);

                        // Apply left-side truncation when needed
                        let srcX = 0;
                        let availableSrcWidth = tileWidth;
                        if (skipRemaining > 0) {
                            const consume = Math.min(skipRemaining, tileWidth);
                            srcX = consume;
                            availableSrcWidth = tileWidth - consume;
                            skipRemaining -= consume;
                            if (availableSrcWidth <= 0) {
                                continue; // this tile is fully truncated away
                            }
                        }

                        const remaining = targetWidth - x;
                        if (remaining <= 0) break;
                        const drawWidth = Math.min(availableSrcWidth, remaining);
                        if (drawWidth <= 0) break;
                        ctx.drawImage(
                            inputCanvas,
                            srcX, 0, drawWidth, sourceHeight,
                            x, 0, drawWidth, sourceHeight
                        );
                        x += drawWidth;
                    }
                }
                clipRef.current?.getLayer()?.batchDraw();
            
            // 2) Debounced fetch of exact frames and redraw when available
            if (exactVideoUpdateTimerRef.current != null) {
                window.clearTimeout(exactVideoUpdateTimerRef.current);
                exactVideoUpdateTimerRef.current = null;
            }
            const DEBOUNCE_MS = 100;
            const requestKey = `${currentClipId}|${timelineStartFrame}-${timelineEndFrame}|${thumbnailWidth}x${timelineHeight}|${overHang}|${frameIndices.join(',')}`;
            exactVideoUpdateTimerRef.current = window.setTimeout(async () => {
                const mySeq = ++exactVideoUpdateSeqRef.current;
                try {
                    if (lastExactRequestKeyRef.current === requestKey) {
                        return;
                    }
                    const exactSamples = await generateTimelineSamples(
                        currentClipId,
                        currentClip?.src!,
                        frameIndices,
                        thumbnailWidth,
                        timelineHeight,
                        tClipWidth,
                        {
                            volume: (currentClip as any)?.volume,
                            fadeIn: (currentClip as any)?.fadeIn,
                            fadeOut: (currentClip as any)?.fadeOut,
                        }
                    );
                    if (mySeq !== exactVideoUpdateSeqRef.current) {
                        return;
                    }
                    const ctx2 = imageCanvas.getContext('2d');
                    if (ctx2 && exactSamples) {
                        ctx2.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                        let x2 = overHang;
                        const targetWidth2 = Math.max(1, imageCanvas.width);
                        const targetHeight2 = Math.max(1, imageCanvas.height);

                        // Calculate left-side skip when resizing from left for video
                        let skipRemaining2 = 0;
                        if (resizeSide === 'left' && clipType === 'video') {
                            let totalTileWidth2 = 0;
                            for (let i = 0; i < exactSamples.length; i++) {
                                const sample = exactSamples[i];
                                const anyCanvas = (sample.canvas as any);
                                const tileWidth = Math.max(1, anyCanvas.width || anyCanvas.naturalWidth || 1);
                                totalTileWidth2 += tileWidth;
                            }
                            const drawableWidth2 = Math.max(0, targetWidth2 - x2);
                            skipRemaining2 = Math.max(0, totalTileWidth2 - drawableWidth2);
                        }

                        for (let i = 0; i < exactSamples.length && x2 < targetWidth2; i++) {
                            const sample = exactSamples[i];
                            const inputCanvas = sample.canvas as HTMLCanvasElement;
                            const anyCanvas = inputCanvas as any;
                            const tileWidth = Math.max(1, anyCanvas.width || anyCanvas.naturalWidth || 1);
                            const tileHeight = Math.max(1, anyCanvas.height || anyCanvas.naturalHeight || 1);
                            const sourceHeight = Math.min(tileHeight, targetHeight2);

                            // Apply left-side truncation when needed
                            let srcX2 = 0;
                            let availableSrcWidth2 = tileWidth;
                            if (skipRemaining2 > 0) {
                                const consume2 = Math.min(skipRemaining2, tileWidth);
                                srcX2 = consume2;
                                availableSrcWidth2 = tileWidth - consume2;
                                skipRemaining2 -= consume2;
                                if (availableSrcWidth2 <= 0) {
                                    continue; // this tile is fully truncated away
                                }
                            }

                            const remaining2 = targetWidth2 - x2;
                            if (remaining2 <= 0) break;
                            const drawWidth2 = Math.min(availableSrcWidth2, remaining2);
                            if (drawWidth2 <= 0) break;
                            ctx2.drawImage(
                                inputCanvas,
                                srcX2, 0, drawWidth2, sourceHeight,
                                x2, 0, drawWidth2, sourceHeight
                            );
                            x2 += drawWidth2;
                        }
                    }
                } finally {
                    if (mySeq === exactVideoUpdateSeqRef.current) {
                        clipRef.current?.getLayer()?.batchDraw();
                        moveClipToEnd(currentClipId);
                        lastExactRequestKeyRef.current = requestKey;
                    }
                }
            }, DEBOUNCE_MS);
        }

        const generateTimelineThumbnailShape = async () => {
            if (clipType !== 'shape') return;
            // make canvas 
            const ctx = imageCanvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                ctx.fillStyle = '#894c30';
                ctx.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
            }
            clipRef.current?.getLayer()?.batchDraw();
        }

        if (clipType === 'audio') {
            generateTimelineThumbnailAudio()
        } else if (clipType === 'image') {
            generateTimelineThumbnailImage()
        } else if (clipType === 'video') {
            generateTimelineThumbnailVideo()
        } else if (clipType === 'shape') {
            generateTimelineThumbnailShape()
        }

    
    }, [zoomLevel, clipWidth, clipType, currentClip, mediaInfoRef.current, resizeSide, thumbnailClipWidth,  maxTimelineWidth, timelineDuration, overHang, resizeSide]);
    
    const calculateFrameFromX = useCallback((xPosition: number) => {
        // Remove padding to get actual timeline position
        const timelineX = xPosition - timelinePadding;
        // Calculate the frame based on the position within the visible timeline
        const [startFrame, endFrame] = timelineDuration;
        const framePosition = (timelineX / timelineWidth) * (endFrame - startFrame) + startFrame;
        return Math.round(framePosition);
    }, [timelinePadding, timelineWidth, timelineDuration]);

    const handleDragMove = useCallback((e:Konva.KonvaEventObject<MouseEvent>) => {
        const halfStroke = isSelected ? 1.5 : 0;
        setClipPosition({x: e.target.x() - halfStroke, y: e.target.y() - halfStroke});
        const stage = e.target.getStage();
        const container = stage?.container();
        let pointerX: number | null = null;
        let pointerY: number | null = null;
        if (container) {
            const rect = container.getBoundingClientRect();
            pointerX = e.evt.clientX - rect.left ;
            pointerY = e.evt.clientY - rect.top;
        }


        if (pointerX == null || pointerY == null) {
            return;
        }

        // Notify editor to auto-scroll if near top/bottom edges
        try {
            window.dispatchEvent(new CustomEvent('timeline-editor-autoscroll', { detail: { y: pointerY } }));
        } catch {}

        // We only need the pointer position for proximity checks

        const children = stage?.children[0]?.children || [];
        const timelineState = useClipStore.getState();
        const timelineMap = new Map(timelineState.timelines.map((t) => [t.timelineId, t]));

        // Determine proximity to dashed line(s) vs the actual timeline track center
        let nearestDashedId: string | null = null;
        let minDashedDistance = Infinity;
        let nearestTimelineForGhost: string | null = null;
        let minTimelineDistance = Infinity;

        for (const child of children) {
          const childId = child.id();
          if (!childId?.startsWith('dashed-')) continue;

          const timelineKey = childId.replace('dashed-', '').replace('-top', '');
          const timelineData = timelineMap.get(timelineKey);
          const timelineFullHeight = timelineData?.timelineHeight ?? (timelineHeight + 16);
          const timelineBottom = (timelineData?.timelineY ?? 0) + timelineFullHeight + 24;
          const trackHeight = Math.max(1, timelineFullHeight - 16);
          const timelineTop = timelineBottom - trackHeight;
          const timelineCenterY = timelineTop + trackHeight / 2;

          const rect = child.getClientRect();
          const rectLeft = rect.x;
          const rectRight = rect.x + rect.width;
          const pointerInX = pointerX >= rectLeft && pointerX <= rectRight;
          if (!pointerInX) continue;

          // Dashed line is visually a horizontal line within this group's bounds.
          // Use the vertical center of the group as the line Y and check ±15px.
          const lineY = rect.y + (rect.height / 2);
          const dashedDistance = Math.abs(pointerY - lineY);
          if (dashedDistance < minDashedDistance) {
            minDashedDistance = dashedDistance;
            nearestDashedId = childId;
          }

          // Also compute distance to the actual timeline track center for comparison
          const timelineDistance = Math.abs((pointerY + scrollY) - timelineCenterY);
          if (timelineDistance < minTimelineDistance) {
            minTimelineDistance = timelineDistance;
            nearestTimelineForGhost = timelineKey;
          }
        }

        const dashedWins = !!nearestDashedId && (minDashedDistance <= 16) && (minDashedDistance < minTimelineDistance);
        setHoveredTimelineId(dashedWins ? nearestDashedId : null);

		
		
		if (clipRef.current) {
			clipRef.current.x(e.target.x());
			clipRef.current.y(e.target.y());
		}

        // If user is within ±15px of a dashed line and closer to it than the timeline,
        // don't render the ghost overlay (we are "in between" timelines)
        if (dashedWins) {
            setSnapGuideX(null);
            setGhostTimelineId(null);
            setGhostInStage(false);
            return;
        }

		// Update ghost overlay with validated position while dragging
		const rectLeft = Math.round(e.target.x() - timelinePadding);
		const [visibleStartFrame, visibleEndFrame] = timelineDuration;
		const clipLen = Math.max(1, (currentEndFrame - currentStartFrame));
		const ghostWidthPx = getClipWidth(0, clipLen, timelineWidth, [visibleStartFrame, visibleEndFrame]);
		const desiredLeft = rectLeft;

        // Determine target timeline for ghost purely by vertical proximity to timeline centers,
        // independent of layer scroll or dashed bounds
        {
            const timelinesArr = timelineState.timelines || [];
            let bestId: string | null = null;
            let bestDist = Infinity;
            for (const t of timelinesArr) {
                const fullH = (t.timelineHeight ?? (timelineHeight + 16));
                // Center Y in content coordinates: top is y + 40, track height is fullH - 16
                const centerY = (t.timelineY ?? 0) + 40 + (Math.max(1, fullH - 16) / 2);
                const d = Math.abs((pointerY + scrollY) - centerY);
                if (d < bestDist) { bestDist = d; bestId = t.timelineId!; }
            }
            nearestTimelineForGhost = bestId || nearestTimelineForGhost;
        }
        const targetTimelineId = (nearestTimelineForGhost || timelineId!);
        const targetTimeline = getTimelineById(targetTimelineId);
        if (!isValidTimelineForClip(targetTimeline!, currentClip!)) return;
        if (!targetTimeline) return;
        
        // Build occupied intervals on the target timeline excluding this clip (visible-window pixels)
        const otherClips = getClipsForTimeline(targetTimelineId).filter(c => c.clipId !== currentClipId);
        let maxRight = 0;
        const occupied = otherClips
            .map((c) => {
                const sx = getClipX(c.startFrame || 0, c.endFrame || 0, timelineWidth, [visibleStartFrame, visibleEndFrame]);
                const sw = getClipWidth(c.startFrame || 0, c.endFrame || 0, timelineWidth, [visibleStartFrame, visibleEndFrame]);
                const lo = Math.max(0, sx);
                const hi = Math.max(0, sx + sw);
                maxRight = Math.max(maxRight, hi);
                return hi > lo ? [lo, hi] as [number, number] : null;
            })
            .filter(Boolean) as [number, number][];
        occupied.sort((a, b) => a[0] - b[0]);
        const merged: [number, number][] = [];
        for (const [lo, hi] of occupied) {
            if (merged.length === 0) merged.push([lo, hi]);
            else {
                const last = merged[merged.length - 1];
                if (lo <= last[1]) last[1] = Math.max(last[1], hi);
                else merged.push([lo, hi]);
            }
        }
        const gaps: [number, number][] = [];
        let prev = 0;
        for (const [lo, hi] of merged) {
            if (lo > prev) gaps.push([prev, lo]);
            prev = Math.max(prev, hi);
        }
        // add the last gap
        if (prev < timelineWidth) gaps.push([prev, timelineWidth]);

        const validGaps = gaps.filter(([lo, hi]) => hi - lo >= ghostWidthPx);
        const pointerCenter = desiredLeft;
        let chosenGap: [number, number] | null = null;

        for (const gap of validGaps) {
            if (pointerCenter >= gap[0] && pointerCenter <= gap[1]) { chosenGap = gap; break; }
        }
        if (!chosenGap && validGaps.length > 0) {
            chosenGap = validGaps.reduce((best, gap) => {
                const gc = (gap[0] + gap[1]) / 2;
                const bc = (best[0] + best[1]) / 2;
                return Math.abs(pointerCenter - gc) < Math.abs(pointerCenter - bc) ? gap : best;
            });
        }

        let validatedLeft = desiredLeft;

        if (chosenGap) {
            const [gLo, gHi] = chosenGap;
            validatedLeft = Math.min(Math.max(desiredLeft, gLo), gHi - ghostWidthPx);
        } else {
            // Ensure validated Left 
            validatedLeft = Math.max(validatedLeft, maxRight);
        }

		validatedLeft = Math.max(0, validatedLeft);

        // Cross-timeline edge snapping against other timelines' clip edges
        const SNAP_THRESHOLD_PX = 6;
        let appliedSnap = false;
        let snapStageX: number | null = null;
        if (chosenGap) {
            const [gLo, gHi] = chosenGap;
            const allTimelines = useClipStore.getState().timelines || [];
            const [sStart, sEnd] = useControlsStore.getState().timelineDuration;
            const edgeCandidates:number[] = [];
            for (const t of allTimelines) {
                if (!t?.timelineId) continue;
                if (t.timelineId === targetTimelineId) continue;
                const tClips = getClipsForTimeline(t.timelineId);
                for (const c of tClips) {
                    const sx = getClipX(c.startFrame || 0, c.endFrame || 0, timelineWidth, [sStart, sEnd]);
                    const sw = getClipWidth(c.startFrame || 0, c.endFrame || 0, timelineWidth, [sStart, sEnd]);
                    const lo = Math.max(0, Math.min(timelineWidth, sx));
                    const hi = Math.max(0, Math.min(timelineWidth, sx + sw));
                    if (hi > lo) {
                        edgeCandidates.push(lo, hi);
                    }
                }
            }
            if (edgeCandidates.length > 0) {
                const leftEdge = validatedLeft;
                const rightEdge = validatedLeft + ghostWidthPx;
                let bestDist = Infinity;
                let bestEdge: number | null = null;
                let bestSide: 'left' | 'right' | null = null;
                for (const edge of edgeCandidates) {
                    const dL = Math.abs(edge - leftEdge);
                    const dR = Math.abs(edge - rightEdge);
                    if (dL < bestDist) { bestDist = dL; bestEdge = edge; bestSide = 'left'; }
                    if (dR < bestDist) { bestDist = dR; bestEdge = edge; bestSide = 'right'; }
                }
                if (bestEdge != null && bestDist <= SNAP_THRESHOLD_PX) {
                    let snappedLeft = bestSide === 'left' ? bestEdge : (bestEdge - ghostWidthPx);
                    // keep within chosen gap
                    if (snappedLeft < gLo) snappedLeft = gLo;
                    if (snappedLeft + ghostWidthPx > gHi) snappedLeft = gHi - ghostWidthPx;
                    const finalLeft = snappedLeft;
                    const finalRight = snappedLeft + ghostWidthPx;
                    const finalDist = bestSide === 'left' ? Math.abs(bestEdge - finalLeft) : Math.abs(bestEdge - finalRight);
                    if (finalDist <= SNAP_THRESHOLD_PX) {
                        validatedLeft = finalLeft;
                        appliedSnap = true;
                        snapStageX = timelinePadding + bestEdge;
                    }
                }
            }
        }

        setGhostTimelineId(targetTimelineId);
        setGhostInStage(true);
        setGhostX(Math.round(validatedLeft));
        setSnapGuideX(appliedSnap && snapStageX != null ? Math.round(snapStageX) : null);
        setGhostStartEndFrame(0, clipLen);
		
    }, [clipRef, clipWidth, timelineHeight, isSelected, timelinePadding, timelineDuration, currentEndFrame, currentStartFrame, timelineWidth, getClipsForTimeline, timelineId, currentClipId, setGhostTimelineId, setGhostInStage, setGhostX, setGhostStartEndFrame, setHoveredTimelineId, scrollY]);

    const handleDragEnd = useCallback((_e: Konva.KonvaEventObject<MouseEvent>) => {
        setIsDragging(false);
        rectRefLeft.current?.moveToTop();
        rectRefRight.current?.moveToTop();
        // Compute validated frames from ghost state
        const [tStart, tEnd] = timelineDuration;
        const stageWidth = timelineWidth;
        const visibleDuration = tEnd - tStart;
        const clipLen = Math.max(1, (currentEndFrame - currentStartFrame));

        // Use ghost state target timeline and position, but if hovered dashed line exists,
        // create a new timeline at that location and drop onto it (similar to DnD flow)
        const state = useClipStore.getState();
        const hoveredId = state.hoveredTimelineId;
        let dropTimelineId = state.ghostTimelineId || timelineId!;
        let gX = state.ghostX;

        if (hoveredId) {
            const timelines = state.timelines;
            const hoveredKey = hoveredId.replace('dashed-', '');
            const hoveredIdx = timelines.findIndex((t) => t.timelineId === hoveredKey);
            const hoveredTimeline = hoveredIdx !== -1 ? timelines[hoveredIdx] : null;
            const newTimelineId = uuidv4();
            // get the timeline this clip is on
            const timeline = state.getTimelineById(timelineId!);
            const newTimeline = {
                type: getTimelineTypeForClip(currentClip!),
                timelineId: newTimelineId,
                timelineWidth: stageWidth,
                timelineY: (hoveredTimeline?.timelineY ?? 0) + 64,
                timelineHeight: timeline?.timelineHeight ?? 64,
            };
            // check if idx is the same as the timelineId
            const currentIdx = state.timelines.findIndex((t) => t.timelineId === timelineId);
            
            // Check if trying to create a new timeline in the same position with only one clip
            const clipsOnCurrentTimeline = getClipsForTimeline(timelineId!);
            const isOnlyClipOnTimeline = clipsOnCurrentTimeline.length === 1;
            
            // Check if creating timeline at same position or adjacent position would result in no-op
            // hoveredIdx === currentIdx: inserting at same position (pushes current down, then deletes)
            // hoveredIdx === currentIdx + 1: inserting right below (after deletion, ends up at same position)
            const wouldBeNoOp = isOnlyClipOnTimeline && (hoveredIdx === currentIdx || hoveredIdx === currentIdx - 1);
            
            if (wouldBeNoOp) {
                // Snap back to original position - treat as no-op
                setClipPosition({x: clipX + timelinePadding, y: timelineY - timelineHeight});
                setHoveredTimelineId(null);
                setGhostInStage(false);
                setGhostTimelineId(null);
                setGhostStartEndFrame(0, 0);
                setGhostX(0);
                setDraggingClipId(null);
                setSnapGuideX(null);
                return;
            }
            
            addTimeline(newTimeline, hoveredIdx);
            dropTimelineId = newTimelineId;
            // Align with DnD behavior: when inserting a new timeline via dashed hover, start from left
            gX = 0;
        }
        
        setHoveredTimelineId(null);
        let startFrame = Math.round(tStart + (gX / stageWidth) * visibleDuration);
        let newTEnd = tEnd;
        let endFrame = startFrame + clipLen;

        // Validate against overlaps on target timeline (in frame units), excluding this clip if present
        const existing = getClipsForTimeline(dropTimelineId)
            .filter(c => c.clipId !== currentClipId)
            .map(c => ({ lo: c.startFrame || 0, hi: c.endFrame || 0 }))
            .filter(iv => iv.hi > iv.lo)
            .sort((a, b) => a.lo - b.lo);
        const merged: {lo:number, hi:number}[] = [];
        for (const iv of existing) {
            if (merged.length === 0) merged.push({ ...iv });
            else {
                const last = merged[merged.length - 1];
                if (iv.lo <= last.hi) last.hi = Math.max(last.hi, iv.hi);
                else merged.push({ ...iv });
            }
        }

        const overlapsExisting = existing.some(iv => (startFrame + clipLen) > iv.lo && startFrame < iv.hi);

        if (!overlapsExisting) {
            // If no overlap, allow placement even if it's outside original window
            const startFrameClamped = Math.max(0, startFrame);
            const endFrameClamped = startFrameClamped + clipLen;

            if (dropTimelineId === timelineId) {
                setClipPosition({x: gX + timelinePadding, y: timelineY - timelineHeight});
            }
    
            updateClip(clipId, { timelineId: dropTimelineId, startFrame: startFrameClamped, endFrame: endFrameClamped });

            // Conditionally restore window back to original length anchored at new clip start
            restoreWindowIfChanged(startFrameClamped);

        // Clear ghost state
            setGhostInStage(false);
            setGhostTimelineId(null);
            setGhostStartEndFrame(0, 0);
            setGhostX(0);
            setDraggingClipId(null);
            setSnapGuideX(null);
            return;
        }

        // Clamp into visible window for gap-based placement
        startFrame = Math.max(tStart, Math.min(newTEnd - clipLen, startFrame));

        const gaps: {lo:number, hi:number}[] = [];
        let prev = tStart;
        for (const iv of merged) {
            const lo = Math.max(tStart, iv.lo);
            const hi = Math.min(tEnd, iv.hi);
            if (lo > prev) gaps.push({ lo: prev, hi: lo });
            prev = Math.max(prev, hi);
        }
        if (prev < tEnd) gaps.push({ lo: prev, hi: tEnd });
        const validGaps = gaps.filter(g => g.hi - g.lo >= clipLen);
        let chosen = validGaps.find(g => startFrame >= g.lo && (startFrame + clipLen) <= g.hi) || null;
        if (!chosen && validGaps.length > 0) {
            const desiredCenter = startFrame + clipLen / 2;
            chosen = validGaps.reduce((best, g) => {
                const gCenter = (g.lo + g.hi) / 2;
                const bCenter = (best.lo + best.hi) / 2;
                return Math.abs(desiredCenter - gCenter) < Math.abs(desiredCenter - bCenter) ? g : best;
            });
        }

        if (chosen) {
            startFrame = Math.min(Math.max(startFrame, chosen.lo), chosen.hi - clipLen);
            endFrame = startFrame + clipLen;
        }

        if (dropTimelineId === timelineId) {
            setClipPosition({x: gX + timelinePadding, y: timelineY - timelineHeight});
        }
        updateClip(clipId, { timelineId: dropTimelineId, startFrame, endFrame });

        // Conditionally restore window back to original length anchored at new clip start
        restoreWindowIfChanged(startFrame);

        // Clear ghost state
        setGhostInStage(false);
        setGhostTimelineId(null);
        setGhostStartEndFrame(0, 0);
        setGhostX(0);
        setDraggingClipId(null);
        setSnapGuideX(null);
    }, [timelineWidth, timelineDuration, currentEndFrame, currentStartFrame, getClipsForTimeline, timelineId, currentClipId, updateClip, setGhostInStage, setGhostTimelineId, setGhostStartEndFrame, setGhostX, setDraggingClipId, timelinePadding, clipId, restoreWindowIfChanged]);


    useEffect(() => {
        setClipPosition({x: clipX + timelinePadding, y: timelineY - timelineHeight});
    }, [clipX, timelinePadding, timelineY, timelineHeight]);

    const handleClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        const isShiftClick = e.evt?.shiftKey || false;
        toggleClipSelection(currentClipId, isShiftClick);
        
        // Move to end for better visual layering if this clip is now selected
        if (currentClipId && (isShiftClick || !isSelected)) {
            moveClipToEnd(currentClipId);
        }
    }, [isSelected, currentClipId, toggleClipSelection, moveClipToEnd]);

    // Handle resizing via global mouse move/up while a handle is being dragged
    useEffect(() => {
        if (!resizeSide) return;
        const stage = clipRef.current?.getStage();
        if (!stage) return;
 
        const handleMouseMove = (e: MouseEvent) => {
            stage.container().style.cursor = 'col-resize';
            const rect = stage.container().getBoundingClientRect();
            const stageX = e.clientX - rect.left;
            const newFrame = calculateFrameFromX(stageX);

            if (resizeSide === 'right') {
                let targetFrame = newFrame;
                // Cross-timeline edge snapping when resizing right
                const [tStart, tEnd] = useControlsStore.getState().timelineDuration;
                const stageWidth = timelineWidth;
                const pointerEdgeInnerX = Math.max(0, Math.min(stageWidth, stageX - timelinePadding));
                const allTimelines = useClipStore.getState().timelines || [];
                const existingEdges:number[] = [];
                for (const t of allTimelines) {
                    if (!t?.timelineId) continue;
                    const tClips = getClipsForTimeline(t.timelineId).filter(c => c.clipId !== clipId);
                    for (const c of tClips) {
                        const sx = getClipX(c.startFrame || 0, c.endFrame || 0, stageWidth, [tStart, tEnd]);
                        const sw = getClipWidth(c.startFrame || 0, c.endFrame || 0, stageWidth, [tStart, tEnd]);
                        const lo = Math.max(0, Math.min(stageWidth, sx));
                        const hi = Math.max(0, Math.min(stageWidth, sx + sw));
                        if (hi > lo) { existingEdges.push(lo, hi); }
                    }
                }
                
                const SNAP_THRESHOLD_PX = 6;
                let best: {edge:number, dist:number} | null = null;
                for (const edge of existingEdges) {
                    const dist = Math.abs(edge - pointerEdgeInnerX);
                    if (!best || dist < best.dist) best = { edge, dist };
                }
                if (best && best.dist <= SNAP_THRESHOLD_PX) {
                    const snappedFrame = Math.round(tStart + (best.edge / Math.max(1, stageWidth)) * (tEnd - tStart));
                    targetFrame = Math.max((currentStartFrame || 0) + 1, snappedFrame);
                    setSnapGuideX(Math.round(timelinePadding + best.edge));
                } else {
                    setSnapGuideX(null);
                }
                if (targetFrame !== currentEndFrame) {
                    // Use the new contiguous resize method - local state will update via useEffect
                    resizeClip(clipId, 'right', targetFrame);
                }
            } else if (resizeSide === 'left') {
               
                let targetFrame = newFrame;
                // Cross-timeline edge snapping when resizing left
                const [tStart, tEnd] = useControlsStore.getState().timelineDuration;
                const stageWidth = timelineWidth;
                const pointerEdgeInnerX = Math.max(0, Math.min(stageWidth, stageX - timelinePadding));
                const allTimelines = useClipStore.getState().timelines || [];
                const existingEdges:number[] = [];
                for (const t of allTimelines) {
                    if (!t?.timelineId) continue;
                    const tClips = getClipsForTimeline(t.timelineId).filter(c => c.clipId !== clipId);
                    for (const c of tClips) {
                        const sx = getClipX(c.startFrame || 0, c.endFrame || 0, stageWidth, [tStart, tEnd]);
                        const sw = getClipWidth(c.startFrame || 0, c.endFrame || 0, stageWidth, [tStart, tEnd]);
                        const lo = Math.max(0, Math.min(stageWidth, sx));
                        const hi = Math.max(0, Math.min(stageWidth, sx + sw));
                        if (hi > lo) { existingEdges.push(lo, hi); }
                    }
                }
                const SNAP_THRESHOLD_PX = 6;
                let best: {edge:number, dist:number} | null = null;
                for (const edge of existingEdges) {
                    const dist = Math.abs(edge - pointerEdgeInnerX);
                    if (!best || dist < best.dist) best = { edge, dist };
                }
                if (best && best.dist <= SNAP_THRESHOLD_PX) {
                    const snappedFrame = Math.round(tStart + (best.edge / Math.max(1, stageWidth)) * (tEnd - tStart));
                    targetFrame = Math.min((currentEndFrame || 0) - 1, snappedFrame);
                    setSnapGuideX(Math.round(timelinePadding + best.edge));
                } else {
                    setSnapGuideX(null);
                }
                if (targetFrame !== currentStartFrame) {
                    // Use the new contiguous resize method - local state will update via useEffect
                    resizeClip(clipId, 'left', targetFrame);
                }
            }
        };

        const handleMouseUp = () => {
            setResizeSide(null);
            stage.container().style.cursor = 'default';
            setSnapGuideX(null);
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [resizeSide, currentStartFrame, currentEndFrame, calculateFrameFromX, clipId, resizeClip]);

    const handleDragStart = useCallback((e:Konva.KonvaEventObject<MouseEvent>) => {
        groupRef.current?.moveToTop();
        rectRefLeft.current?.moveToBottom();
        rectRefRight.current?.moveToBottom();
        setIsDragging(true);
        setTempClipPosition({x: clipPosition.x, y: clipPosition.y});
        // If this clip isn't already selected, select it (without shift behavior during drag)
        if (!isSelected) {
            toggleClipSelection(currentClipId, false);
        }
        if (currentClipId) {
            moveClipToEnd(currentClipId);
        }
        // Capture initial window at the start of drag
        dragInitialWindowRef.current = [...useControlsStore.getState().timelineDuration] as [number, number];
        
        // Initialize ghost overlay for this clip
        const clipLen = Math.max(1, (currentEndFrame - currentStartFrame));
        setDraggingClipId(currentClipId);
        setGhostTimelineId(timelineId!);
        setGhostStartEndFrame(0, clipLen);
        setGhostInStage(true);

        const stage = e.target.getStage();
        const pos = stage?.getPointerPosition();
        if (pos) {
            const [visibleStartFrame, visibleEndFrame] = useControlsStore.getState().timelineDuration;
            const ghostWidthPx = getClipWidth(0, clipLen, timelineWidth, [visibleStartFrame, visibleEndFrame]);
            const pointerLocalX = pos.x - timelinePadding;
            const desiredLeft = pointerLocalX;
            let validated = Math.max(0, Math.min(timelineWidth - ghostWidthPx, desiredLeft));

            setGhostX(Math.round(validated));
        }
    }, [currentClipId, moveClipToEnd, isSelected, currentEndFrame, currentStartFrame, setDraggingClipId, setGhostTimelineId, setGhostStartEndFrame, setGhostInStage, setGhostX, timelineId, timelineWidth, timelinePadding]);

    const handleMouseOver = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        moveClipToEnd(currentClipId);
        e.target.getStage()!.container().style.cursor = 'pointer';
    }, [isSelected]);
    const handleMouseLeave = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        e.target.getStage()!.container().style.cursor = 'default';
    }, [isSelected]);

    return (
        <>
        {isSelected && (
                <Rect
                    ref={clipRef}
                    x={clipPosition.x}
                    y={clipPosition.y}
                    width={clipWidth}
                    fill={'transparent'}
                    height={timelineHeight}
                    cornerRadius={cornerRadius}
                    stroke={'#AE81CE'}
                    strokeWidth={3}
                />
            )}
            <Group  
                ref={groupRef}
                onClick={handleClick} 
                draggable={resizeSide === null} 
                onDragEnd={handleDragEnd} 
                onDragMove={handleDragMove} 
                onDragStart={handleDragStart}
                x={isDragging ? tempClipPosition.x : clipPosition.x } 
                y={isDragging ? tempClipPosition.y : clipPosition.y} 
                width={clipWidth}
                height={timelineHeight}
                clipX={0}
                clipY={0}
                clipWidth={clipWidth}
                clipHeight={timelineHeight}
                onMouseOver={handleMouseOver}
                onMouseLeave={handleMouseLeave}
                >
                <Image 
                    x={imageX}
                    y={0}
                    image={imageCanvas} 
                    width={imageWidth}
                    height={timelineHeight}
                    cornerRadius={clipPosition.x == 24 || clipWidth <= imageWidth || overHang > 0 || imageX === 0 ? cornerRadius: 0}
                    fill={'#E3E3E3'} 
                />
                {clipType === 'shape' && (currentClip  as ShapeClipProps)?.shapeType && (
                    <Group>
                    <Rect
                        x={8 - 4}
                        y={timelineHeight / 2}
                        width={textWidth + 8}
                        height={14}
                        cornerRadius={2}
                        fill="rgba(255, 255, 255, 0.2)"
                        offsetY={7.5}
                    />
                    <Text
                        ref={textRef}
                        x={8}
                        y={timelineHeight / 2}
                        text={((currentClip as ShapeClipProps)?.shapeType?.charAt(0).toUpperCase() ?? '')  + ((currentClip as ShapeClipProps)?.shapeType?.slice(1 ) ?? '')}
                        fontSize={10}
                        fontFamily="Poppins"
                        fill="white"
                        align="left"
                        verticalAlign="middle"
                        offsetY={5}
                    />
                    </Group>
                )}
            </Group>            
            <Rect
                ref={rectRefRight}
                x={clipPosition.x  + clipWidth - 5}
                y={clipPosition.y + timelineHeight / 2 - 7}
                width={5}
                height={14}
                cornerRadius={[2, 0, 0, 2]}
                stroke={isSelected ? '#AE81CE' : 'transparent'}
                strokeWidth={2.5}
                fill={isSelected ? 'white': 'transparent'}
                onMouseOver={(e) => {
                    if (isSelected) {
                        e.target.getStage()!.container().style.cursor = 'col-resize';
                    }
                }}
                onMouseDown={(e) => {
                    e.cancelBubble = true;
                    if (!isSelected) {
                        toggleClipSelection(currentClipId, false);
                    }
                    if (currentClipId) {
                        moveClipToEnd(currentClipId);
                    }
                    setResizeSide('right');
                    e.target.getStage()!.container().style.cursor = 'col-resize';
                }}
                onMouseLeave={(e) => {
                    if (isSelected) {
                        e.target.getStage()!.container().style.cursor = 'default';
                    }
                }}

                />
                
            <Rect
                ref={rectRefLeft}
                x={clipPosition.x + (isDragging ? 2 : 0)}
                y={clipPosition.y + timelineHeight / 2 - 7}
                width={5}
                height={14}
                cornerRadius={[0, 2, 2, 0]}
                stroke={isSelected ? '#AE81CE' : 'transparent'}
                strokeWidth={2.5}
                fill={isSelected ? 'white': 'transparent'}
                onMouseOver={(e) => {
                    if (isSelected) {
                        e.target.getStage()!.container().style.cursor = 'col-resize';
                    }
                }}
                onMouseDown={(e) => {
                    e.cancelBubble = true;
                    if (!isSelected) {
                        toggleClipSelection(currentClipId, false);
                    }
                    if (currentClipId) {
                        moveClipToEnd(currentClipId);
                    }
                    setResizeSide('left');
                    e.target.getStage()!.container().style.cursor = 'col-resize';
                }}
                onMouseLeave={(e) => {
                    if (isSelected) {
                        e.target.getStage()!.container().style.cursor = 'default';
                    }
                }}

                />
            
            </>
    )
}

export default TimelineClip;
