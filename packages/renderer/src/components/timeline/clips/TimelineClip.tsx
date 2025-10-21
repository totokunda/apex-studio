import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useClipStore, getClipWidth, getClipX, isValidTimelineForClip, getTimelineTypeForClip } from "@/lib/clip";
import { generatePosterCanvas, generateTimelineSamples } from "@/lib/media/timeline";
import { getNearestCachedCanvasSamples } from "@/lib/media/canvas";
import { useControlsStore } from "@/lib/control";
import { Image, Group, Rect, Text, Line } from 'react-konva';
import Konva from 'konva';
import { MediaInfo, ShapeClipProps, TextClipProps, TimelineProps, VideoClipProps, ImageClipProps, ClipType, FilterClipProps, MaskClipProps, PreprocessorClipType,  GroupClipProps } from "@/lib/types";
import { v4 as uuidv4 } from 'uuid';
import { generateAudioWaveformCanvas, getMediaInfoCached } from "@/lib/media/utils";
import { useWebGLFilters } from "@/components/preview/webgl-filters";
import {useWebGLMask} from "@/components/preview/mask/useWebGLMask"
import { PreprocessorClip } from "./PreprocessorClip";
import MaskKeyframes from "./MaskKeyframes";
import { useViewportStore } from "@/lib/viewport";
import { useContextMenuStore, ContextMenuItem } from '@/lib/context-menu';
import { renderToStaticMarkup } from 'react-dom/server';
import { RxText as RxTextIcon } from 'react-icons/rx';
import { MdOutlineDraw as MdOutlineDrawIcon, MdMovie as MdMovieIcon, MdImage as MdImageIcon, MdAudiotrack as MdAudiotrackIcon } from 'react-icons/md';
import { LuShapes as LuShapeIcon } from "react-icons/lu";
import { MdPhotoFilter as MdFilterIcon } from "react-icons/md";
const THUMBNAIL_TILE_SIZE = 36;

const TimelineClip: React.FC<TimelineProps & {clipId: string, clipType: ClipType,  scrollY: number}> = ({timelineWidth = 0, timelineY = 0, timelineHeight = 54, timelinePadding = 24, clipId,  timelineId, clipType, scrollY}) => {
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
    const setSelectedPreprocessorId = useClipStore((s) => s.setSelectedPreprocessorId);
    const setIsDraggingGlobal = useClipStore((s) => s.setIsDragging);
    const focusFrame = useControlsStore((s) => s.focusFrame);
    const tool = useViewportStore((s) => s.tool);
    const getClipById = useClipStore((s) => s.getClipById);
    const [groupedCanvases, setGroupedCanvases] = useState<HTMLCanvasElement[]>([]);
    const [ groupCounts, setGroupCounts] = useState<{video:number,image:number,audio:number,text:number,draw:number,filter:number,shape:number}>({ video: 0, image: 0, audio: 0, text: 0, draw: 0, filter: 0, shape: 0 });
    // Subscribe directly to this clip's data
    const currentClip = useClipStore((s) => s.clips.find((c) => c.clipId === clipId && (timelineId ? c.timelineId === timelineId : true)));

    const { applyMask } = useWebGLMask({
        focusFrame: focusFrame,   
        masks: (currentClip as PreprocessorClipType & {masks: MaskClipProps[]})?.masks || [],
        disabled: tool === 'mask' || (currentClip?.type !== 'video' && currentClip?.type !== 'image') ,
        clip: currentClip,
    });
    const cornerRadius = useMemo(() => {
        return 1;
    }, [currentClip?.type]);
    
    // Check if clip has preprocessors (only for video/image clips)
    const hasPreprocessors = useMemo(() => {
        if (currentClip?.type !== 'video' && currentClip?.type !== 'image') return false;
        const preprocessors = (currentClip as VideoClipProps | ImageClipProps)?.preprocessors;
        return preprocessors && preprocessors.length > 0;
    }, [currentClip]);

    // Total height including preprocessor bar if needed
    const totalClipHeight = useMemo(() => {
        return timelineHeight;
    }, [hasPreprocessors, timelineHeight]);
    
    const currentStartFrame = currentClip?.startFrame ?? 0;
    const currentEndFrame = currentClip?.endFrame ?? 0;

    const clipWidth = useMemo(() => Math.max(getClipWidth(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), 3), [currentStartFrame, currentEndFrame, timelineWidth, timelineDuration]);
    const clipX = useMemo(() => getClipX(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), [currentStartFrame, currentEndFrame, timelineWidth, timelineDuration, timelineId]);
    const clipRef = useRef<Konva.Line>(null);
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
    // global context menu used instead of local state
    const { applyFilters } = useWebGLFilters();
    const [forceRerenderCounter, setForceRerenderCounter] = useState(0);
    
    // Sizing for stacked canvases inside group clips
    const groupCardHeight = useMemo(() => Math.max(1, timelineHeight - 24), [timelineHeight]);
    const groupCardWidth = useMemo(() => Math.max(1, Math.min(clipWidth - 24, Math.round((timelineHeight - 24) * 1.35))), [timelineHeight, clipWidth]);
    
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

    const showMaskKeyframes = useMemo(() => {
        if (!isSelected) return false;
        if (tool !== 'mask') return false;
        if (!currentClip || currentClip.type !== 'video') return false;
        if (isDragging) return false; // hide while dragging current clip
        const masks = (currentClip as VideoClipProps).masks ?? [];
        return masks.length > 0;
    }, [currentClip, isSelected, tool, isDragging]);

    const [clipPosition, setClipPosition] = useState<{x:number, y:number}>({
        x: clipX + timelinePadding,
        y: timelineY - totalClipHeight
    });
    const [tempClipPosition, setTempClipPosition] = useState<{x:number, y:number}>({
        x: clipX + timelinePadding,
        y: timelineY - totalClipHeight
    });
    const fixedYRef = useRef(timelineY - totalClipHeight);
    
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
        const newY = timelineY - totalClipHeight;
        setClipPosition({x: clipX + timelinePadding, y: newY});
        fixedYRef.current = newY;
    }, [timelinePadding, timelineY, timelineId, clipX, totalClipHeight]);

    useEffect(() => {
        if (textRef.current) {
            setTextWidth(textRef.current.width());
        }
    }, [(currentClip as ShapeClipProps)?.shapeType]);

	useEffect(() => {
		if (!currentClip) return;

        const generateTimelineThumbnailAudio = async () => {
            if (clipType !== 'audio') return;
            const speed = Math.max(0.1, Math.min(5, Number((currentClip as any)?.speed ?? 1)));
            
            const width = mediaInfoRef.current?.stats.audio?.averagePacketRate ?? 1;
            const height = timelineHeight;
            const timelineShift = currentStartFrame - (currentClip.framesToGiveStart ?? 0);
            const visibleStartFrame = Math.max(currentStartFrame, timelineDuration[0]);
            const visibleEndFrame = Math.min(currentEndFrame, timelineDuration[1]) * speed;
            const duration = (timelineDuration[1] - timelineDuration[0]);

            const pixelsPerFrame = (timelineWidth / duration);
            const positionOffsetStart = Math.round(Math.max(0, (currentStartFrame - timelineDuration[0]) * pixelsPerFrame))
            const tClipWidth = Math.round((pixelsPerFrame * (visibleEndFrame - visibleStartFrame)) + (positionOffsetStart === 0 ? timelinePadding : 0)) / speed;

            const samples = await generateTimelineSamples(
                currentClipId,
                currentClip?.src!,
                [0],
                width,
                height,
                tClipWidth,
                {
                    mediaInfo: mediaInfoRef.current,
                    startFrame: visibleStartFrame - timelineShift,
                    endFrame: visibleEndFrame - timelineShift,
                    volume: (currentClip as any)?.volume,
                    fadeIn: (currentClip as any)?.fadeIn,
                    fadeOut: (currentClip as any)?.fadeOut,
                }
            );

            if (samples?.[0]?.canvas) {
                const inputCanvas = samples?.[0]?.canvas as HTMLCanvasElement;
                let offset = Math.max(0, imageCanvas.width - tClipWidth - positionOffsetStart) 
                const ctx = imageCanvas.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                    ctx.drawImage(inputCanvas, offset, 0);
                }
                // Ensure Konva layer updates immediately after drawing audio waveform
                groupRef.current?.getLayer()?.batchDraw();
            }

            //moveClipToEnd(currentClipId);
            
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
                const canvasToTile = applyMask(inputCanvas);
                const ctx = imageCanvas.getContext('2d');
                if (ctx) {

                    const targetWidth = Math.max(1, imageCanvas.width);
                    const targetHeight = Math.max(1, imageCanvas.height);
                    ctx.clearRect(0, 0, targetWidth, targetHeight);

                    // Determine tile dimensions from the input canvas/image
                    const tileWidth = Math.max(1, (canvasToTile as any).width || (canvasToTile as any).naturalWidth || 1);
                    const tileHeight = Math.max(1, (canvasToTile as any).height || (canvasToTile as any).naturalHeight || 1);
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
                                canvasToTile,
                                x, 0, drawWidth, sourceHeight
                            );
                        }
                        x += drawWidth;
                    }
                    
                    // Apply WebGL filters to image thumbnails
                    const imgClip = currentClip as ImageClipProps;
                    applyFilters(imageCanvas, {
                        brightness: imgClip?.brightness,
                        contrast: imgClip?.contrast,
                        hue: imgClip?.hue,
                        saturation: imgClip?.saturation,
                        blur: imgClip?.blur,
                        sharpness: imgClip?.sharpness,
                        noise: imgClip?.noise,
                        vignette: imgClip?.vignette
                    });
                    
                }
            }
            groupRef.current?.getLayer()?.batchDraw();
            moveClipToEnd(currentClipId);
        }

        const generateTimelineThumbnailVideo = async () => {
            if (clipType !== 'video') return;
            let tClipWidth = Math.min(thumbnailClipWidth.current, maxTimelineWidth);
            const width = mediaInfoRef.current?.video?.displayWidth ?? 1;
            const height = mediaInfoRef.current?.video?.displayHeight ?? 1;
            const ratio = width / height;
            const thumbnailWidth = Math.max(timelineHeight * ratio, THUMBNAIL_TILE_SIZE);
            
            
            const speed = Math.max(0.1, Math.min(5, Number((currentClip as any)?.speed ?? 1)));
            
            // Calculate frame indices based on timeline duration and available columns
            let numColumns = Math.ceil((tClipWidth - overHang) / thumbnailWidth) + 1;

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
            
            // Map timeline frames to source frames considering speed, in-clip offset, split bounds, and framerate conversion
            
            const projectFps = useControlsStore.getState().fps || 30;
            const clipFps = mediaInfoRef.current?.stats.video?.averagePacketRate || projectFps;
            const fpsAdjustment = projectFps / clipFps;
            
            frameIndices = frameIndices.map((frameIndex) => {
                const local = Math.max(0, frameIndex - timelineShift);
                const speedAdjusted = local * speed;
                // Map from project fps space to native clip fps space
                const nativeFpsFrame = Math.round((speedAdjusted / projectFps) * clipFps);
                const mediaStartFrame = Math.round((mediaInfoRef.current?.startFrame ?? 0) / projectFps * clipFps)
                const mediaEndFrame = Math.round(( mediaInfoRef.current?.endFrame ?? 0) / projectFps * clipFps)
                let sourceFrame = nativeFpsFrame + mediaStartFrame;
                if (mediaEndFrame !== 0) {
                    sourceFrame = Math.min(sourceFrame, mediaEndFrame);
                }
                return Math.max(mediaStartFrame, sourceFrame);
            });
            
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
                
                // Track if we have any cached samples
                const hasCachedSamples = nearest.some(sample => sample !== null && sample !== undefined);
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
                        const canvasToTile = applyMask(inputCanvas, Math.round(frameIndices[i] * fpsAdjustment));
                        const anyCanvas = canvasToTile as any;
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
                            canvasToTile,
                            srcX, 0, drawWidth, sourceHeight,
                            x, 0, drawWidth, sourceHeight
                        );
                        x += drawWidth;
                    }
                    
                    // Apply WebGL filters to video thumbnails
                    const vidClip = currentClip as VideoClipProps;
                    applyFilters(imageCanvas, {
                        brightness: vidClip?.brightness,
                        contrast: vidClip?.contrast,
                        hue: vidClip?.hue,
                        saturation: vidClip?.saturation,
                        blur: vidClip?.blur,
                        sharpness: vidClip?.sharpness,
                        noise: vidClip?.noise,
                        vignette: vidClip?.vignette
                    });
                }
                groupRef.current?.getLayer()?.batchDraw();
            
            // 2) Debounced fetch of exact frames and redraw when available
            if (exactVideoUpdateTimerRef.current != null) {
                window.clearTimeout(exactVideoUpdateTimerRef.current);
                exactVideoUpdateTimerRef.current = null;
            }
            const DEBOUNCE_MS = hasCachedSamples ? 100 : 0;
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
                            const canvasToTile = applyMask(inputCanvas, Math.round(frameIndices[i] * fpsAdjustment));
                            const anyCanvas = canvasToTile as any;
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
                                canvasToTile,
                                srcX2, 0, drawWidth2, sourceHeight,
                                x2, 0, drawWidth2, sourceHeight
                            );
                            x2 += drawWidth2;
                        }
                        
                        // Apply WebGL filters to exact video thumbnails
                        const vidClip = currentClip as VideoClipProps;
                        applyFilters(imageCanvas, {
                            brightness: vidClip?.brightness,
                            contrast: vidClip?.contrast,
                            hue: vidClip?.hue,
                            saturation: vidClip?.saturation,
                            blur: vidClip?.blur,
                            sharpness: vidClip?.sharpness,
                            noise: vidClip?.noise,
                            vignette: vidClip?.vignette
                        });
                    }
                } finally {
                    if (mySeq === exactVideoUpdateSeqRef.current) {
                        groupRef.current?.getLayer()?.batchDraw();
                        
                        lastExactRequestKeyRef.current = requestKey;
                        
                        // Force rerender if there were no cached samples initially
                        if (!hasCachedSamples) {
                            setForceRerenderCounter(prev => prev + 1);
                        }
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
            groupRef.current?.getLayer()?.batchDraw();
        }

        const generateTimelineThumbnailText = async () => {
            if (clipType !== 'text') return;
            // make canvas 
            const ctx = imageCanvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                ctx.fillStyle = '#E3E3E3';
                ctx.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
            }
            groupRef.current?.getLayer()?.batchDraw();
        }
        const generateTimelineThumbnailFilter = async () => {
            if (clipType !== 'filter') return;
            // make canvas 
            const ctx = imageCanvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                ctx.fillStyle = '#02ace6';
                ctx.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
            }
            groupRef.current?.getLayer()?.batchDraw();
        }

        const generateTimelineThumbnailDrawing = async () => {
            if (clipType !== 'draw') return;
            // make canvas 
            const ctx = imageCanvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                ctx.fillStyle = '#9B59B6';
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
        } else if (clipType === 'text') {
            generateTimelineThumbnailText()
        } else if (clipType === 'filter') {
            generateTimelineThumbnailFilter()
        } else if (clipType === 'draw') {
            generateTimelineThumbnailDrawing()
        } 
    
    }, [zoomLevel, clipWidth, clipType, currentClip, applyMask, tool, mediaInfoRef.current, resizeSide, thumbnailClipWidth,  maxTimelineWidth, timelineDuration, overHang, resizeSide, forceRerenderCounter]);
    
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
        // For preprocessor clips, only update X position to prevent vertical drift
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
        if (prev < timelineWidth) gaps.push([prev, Infinity]);

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
		
    }, [clipRef, clipWidth, timelineHeight, isSelected, timelinePadding, timelineDuration, currentEndFrame, currentStartFrame, timelineWidth, getClipsForTimeline, timelineId, currentClipId, setGhostTimelineId, setGhostInStage, setGhostX, setGhostStartEndFrame, setHoveredTimelineId, scrollY, clipType, currentClip, setSnapGuideX]);

    const handleDragEnd = useCallback((_e: Konva.KonvaEventObject<MouseEvent>) => {
        rectRefLeft.current?.moveToTop();
        rectRefRight.current?.moveToTop();
        setIsDragging(false);
        setIsDraggingGlobal(false);
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
                timelineY: (hoveredTimeline?.timelineY ?? 0) + 54,
                timelineHeight: timeline?.timelineHeight ?? 54,
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
                setClipPosition({x: clipX + timelinePadding, y: timelineY - totalClipHeight});
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
            // When creating a new timeline via dashed hover, place based on current pointer X
            const stage = _e.target.getStage();
            const pos = stage?.getPointerPosition();
            if (pos) {
                const innerX = Math.max(0, Math.min(stageWidth, pos.x - timelinePadding));
                gX = Math.round(innerX);
            } else {
                gX = 0;
            }
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
                setClipPosition({x: gX + timelinePadding, y: timelineY - totalClipHeight});
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
            setClipPosition({x: gX + timelinePadding, y: timelineY - totalClipHeight});
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
    }, [timelineWidth, timelineDuration, currentEndFrame, currentStartFrame, getClipsForTimeline, timelineId, currentClipId, updateClip, setGhostInStage, setGhostTimelineId, setGhostStartEndFrame, setGhostX, setDraggingClipId, timelinePadding, clipId, restoreWindowIfChanged, clipType, currentClip, clipX, timelineY, totalClipHeight, setSnapGuideX, setHoveredTimelineId, addTimeline, getTimelineById, setIsDraggingGlobal]);

    useEffect(() => {
        const newY = timelineY - totalClipHeight;
        setClipPosition({x: clipX + timelinePadding, y: newY});
        fixedYRef.current = newY;
    }, [clipX, timelinePadding, timelineY, totalClipHeight]);

    const handleClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        const isShiftClick = e.evt?.shiftKey || false;
        toggleClipSelection(currentClipId, isShiftClick);
        setSelectedPreprocessorId(null);
    }, [isSelected, currentClipId, toggleClipSelection, moveClipToEnd]);

    const handleContextMenu = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        e.evt.preventDefault();
        const stage = e.target.getStage();
        const container = stage?.container();
        if (!container) return;
        // Select this clip if it's not already part of the selection
        const sel = useControlsStore.getState().selectedClipIds || [];
        if (!sel.includes(currentClipId)) {
            useControlsStore.getState().setSelectedClipIds([currentClipId]);
        }
        // Use global context menu store
        const controls = useControlsStore.getState();
        const clipsState = useClipStore.getState();
        const clip = clipsState.getClipById(currentClipId);
        const isVideo = clip?.type === 'video';
        const isSeparated = (() => {
            if (!clip || clip.type !== 'video') return false;
            try {
                const url = new URL(clip.src);
                if ((url.hash || '').replace('#','') === 'video') return true;
                const audioURL = new URL(clip.src); audioURL.hash = 'audio';
                return (clipsState.clips || []).some(c => c.type === 'audio' && c.src === audioURL.toString());
            } catch { return false; }
        })();
        const targetIds = (controls.selectedClipIds || []).includes(currentClipId) ? controls.selectedClipIds : [currentClipId];
        const aiCommands: ContextMenuItem[] = [];
        /**AI Commands, we will have to implement with our AI system. */
        if (isVideo && clip.masks && clip.masks.length === 0 && clip.preprocessors && clip.preprocessors.length === 0) {
            aiCommands.push({ id: 'extend', label: 'Extend', action: 'extend', });
            aiCommands.push({ id: 'stabilize', label: 'Stabilize', action: 'stabilize', });
            aiCommands.push({ id: 'editVideo', label: 'Edit Video', action: 'editVideo', });
        } else if (clip?.type === 'image' && clip.masks && clip.masks.length === 0 && clip.preprocessors && clip.preprocessors.length === 0) {
            aiCommands.push({ id: 'animate', label: 'Animate', action: 'animate', });
            aiCommands.push({ id: 'editImage', label: 'Edit Image', action: 'editImage', });
        }
        if ((clip?.type === 'video' || clip?.type === 'image') && clip.masks && clip.masks.length > 0) {
            aiCommands.push({ id: 'inpaint', label: 'Inpaint', action: 'inpaint', });
            aiCommands.push({ id: 'outpaint', label: 'Outpaint', action: 'outpaint', });
        }
        if ((clip?.type === 'video' || clip?.type === 'image') && clip.preprocessors && clip.preprocessors.length > 0) {
            aiCommands.push({ id: 'control', label: 'Use as Control', action: 'control', });
        }

        const isGroup = clip?.type === 'group';

        const otherCommands: ContextMenuItem[] = [
            { id: 'export', label: 'Export…', action: 'export' },
        ];
        // check if any of the selected clips are groups if so, we cannot group it 
        const isAnyGroup = (controls.selectedClipIds || []).some((clipId) => {
            const clip = clipsState.getClipById(clipId);
            return clip?.type === 'group';
        });
        if (isGroup) {
            otherCommands.push({ id: 'ungroup', label: 'Ungroup', action: 'ungroup', shortcut: '⌘⇧G' });
        } else if (!isAnyGroup) {
            otherCommands.push({ id: 'group', label: 'Group…', action: 'group', disabled: (controls.selectedClipIds || []).length < 2, shortcut: '⌘G' });
        }

        useContextMenuStore.getState().openMenu({
            position: { x: e.evt.clientX, y: e.evt.clientY },
            target: { type: 'clip', clipIds: targetIds, primaryClipId: currentClipId, isVideo: !!isVideo },
            groups: [
                {
                    id: 'edit',
                    items: [
                        { id: 'copy', label: 'Copy', action: 'copy', shortcut: '⌘C' },
                        { id: 'cut', label: 'Cut', action: 'cut', shortcut: '⌘X' },
                        { id: 'paste', label: 'Paste', action: 'paste', shortcut: '⌘V' },
                        { id: 'delete', label: 'Delete', action: 'delete', shortcut: 'Del' },
                    ],
                },
                {
                    id: 'ai',
                    label: 'AI',
                    items: [
                        ...aiCommands,
                    ],
                },
                {
                    id: 'clip-actions',
                    items: [
                        { id: 'split', label: 'Split at Playhead', action: 'split', disabled: isGroup },
                        { id: 'separate', label: 'Detach Audio', action: 'separateAudio', disabled: !isVideo || isSeparated },
                    ],
                },
                {
                    id: 'other',
                    items: [
                        ...otherCommands,
                    ],
                },
            ],
        });
    }, [currentClipId]);

    useEffect(() => {
        if (isSelected) {
            rectRefLeft.current?.moveToTop();
            rectRefRight.current?.moveToTop();
        }
    }, [isSelected]);

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
                
                // Check for preprocessor boundaries - prevent resizing past preprocessors
                if (currentClip && (currentClip.type === 'video')) {
                    const preprocessors = (currentClip as VideoClipProps | ImageClipProps).preprocessors || [];
                    if (preprocessors.length > 0) {
                        // Find the rightmost preprocessor end position (in absolute frames)
                        const rightmostPreprocessorEnd = Math.max(
                            ...preprocessors.map(p => (currentStartFrame || 0) + (p.endFrame ?? 0))
                        );
                        // Limit resize to not go below the rightmost preprocessor end
                        targetFrame = Math.max(targetFrame, rightmostPreprocessorEnd);
                    }
                }
                
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
                
                // Check for preprocessor boundaries - prevent resizing past preprocessors
                if (currentClip && (currentClip.type === 'video')) {
                    const preprocessors = (currentClip as VideoClipProps | ImageClipProps).preprocessors || [];
                    if (preprocessors.length > 0) {
                        // Find the leftmost preprocessor start position (in absolute frames)
                        const leftmostPreprocessorStart = Math.min(
                            ...preprocessors.map(p => (currentStartFrame || 0) + (p.startFrame ?? 0))
                        );
                        // Limit resize to not go above the leftmost preprocessor start
                        targetFrame = Math.min(targetFrame, leftmostPreprocessorStart);
                    }
                }
                
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
                
                // Prevent resizing below frame 0
                targetFrame = Math.max(0, targetFrame);
                
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
    }, [resizeSide, currentStartFrame, currentEndFrame, calculateFrameFromX, clipId, resizeClip, clipType, currentClip, updateClip, timelinePadding, timelineWidth, getClipsForTimeline, setSnapGuideX]);

    const handleDragStart = useCallback((e:Konva.KonvaEventObject<MouseEvent>) => {
        groupRef.current?.moveToTop();
        
        setSelectedPreprocessorId(null);
       
        setIsDragging(true);
        setIsDraggingGlobal(true);
        setTempClipPosition({x: clipPosition.x, y: clipPosition.y});
        // Store fixed Y position at drag start for preprocessor clips
        fixedYRef.current = clipPosition.y;

        rectRefLeft.current?.moveToBottom();
        rectRefRight.current?.moveToBottom();
        // If this clip isn't already selected, select it (without shift behavior during drag)
        if (!isSelected) {
            toggleClipSelection(currentClipId, false);
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


    }, [currentClipId, moveClipToEnd, isSelected, currentEndFrame, currentStartFrame, setDraggingClipId, setGhostTimelineId, setGhostStartEndFrame, setGhostInStage, setGhostX, timelineId, timelineWidth, timelinePadding, clipPosition, toggleClipSelection, setIsDraggingGlobal]);

    const handleMouseOver = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        //moveClipToEnd(currentClipId);
        e.target.getStage()!.container().style.cursor = 'pointer';
    }, [isSelected]);
    const handleMouseLeave = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        e.target.getStage()!.container().style.cursor = 'default';
    }, [isSelected]);

    useEffect(() => {
        (async () => {
        if (clipType === 'group' ) {
            // get the children of the group
            const childIds = (currentClip as GroupClipProps).children.flat();
            const children = childIds.map((childId) => getClipById(childId));
            // Compute per-type counts for badge row
            const counts = { video: 0, image: 0, audio: 0, text: 0, draw: 0, filter: 0, shape: 0 } as {video:number,image:number,audio:number,text:number,draw:number,filter:number,shape:number};
            for (const ch of children) {
                if (!ch) continue;
                if (ch.type === 'video') counts.video++;
                else if (ch.type === 'image') counts.image++;
                else if (ch.type === 'audio') counts.audio++;
                else if (ch.type === 'text') counts.text++;
                else if (ch.type === 'draw') counts.draw++;
                else if (ch.type === 'filter') counts.filter++;
                else if (ch.type === 'shape') counts.shape++;
            }
            setGroupCounts(counts);
            const canvases = await Promise.all(children.reverse().slice(0, 3).map(async (child) => {
                if (child?.type === 'video' || child?.type === 'image' && child?.src) {
                    const mediaInfo = getMediaInfoCached(child.src);
                    if (!mediaInfo) return null;
                    const masks = (child as VideoClipProps | ImageClipProps).masks || [];
                    const preprocessors = (child as VideoClipProps | ImageClipProps).preprocessors || [];
                    const poster = await generatePosterCanvas(child.src, undefined, undefined, { mediaInfo, masks, preprocessors });
                    if (!poster) return null;
                    return poster;
                } else if (child?.type === 'audio' && child?.src) {
                    const mediaInfo = getMediaInfoCached(child.src);
                    if (!mediaInfo) return null;
                    const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
                    const cssWidth = 64;
                    const cssHeight = Math.round(cssWidth * 9 / 16);
                    const width = cssWidth * dpr;
                    const height = cssHeight * dpr;
                    // make the height and width small like max and use that ratio to scale the width and height
                    const waveform = await generateAudioWaveformCanvas(child.src, width, height, { color: '#7791C4', mediaInfo: mediaInfo });
                    if (!waveform) return null;
                    return waveform;
                      } else if (child?.type === 'text' || child?.type === 'draw' || child?.type === 'filter' || child?.type === 'shape' ) {
                    const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
                    const cssWidth = timelineWidth || 240;
                    const cssHeight = Math.round(cssWidth * 9 / 16);
                    const width = cssWidth * dpr;
                    const height = cssHeight * dpr;
                    const canvas = document.createElement('canvas');
                    canvas.width = Math.max(1, width);
                    canvas.height = Math.max(1, height);
                    const ctx = canvas.getContext('2d');
                    if (!ctx) return null;
                    // Set background color based on clip type (matching timeline thumbnails)
                    let bg = '#E3E3E3';
                    if (child.type === 'draw') bg = '#9B59B6';
                    if (child.type === 'filter') bg = '#00BFFF';
                    if (child.type === 'shape') bg = '#894c30';
                    ctx.fillStyle = bg;
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    // Prepare the icon SVG
                    const iconSize = Math.floor(Math.min(canvas.width, canvas.height) * 0.35);
                    let iconSvg = '';
                    if (child.type === 'text') {
                        iconSvg = renderToStaticMarkup(React.createElement(RxTextIcon, { size: iconSize, color: '#222124' }));
                    } else if (child.type === 'draw') {
                        iconSvg = renderToStaticMarkup(React.createElement(MdOutlineDrawIcon, { size: iconSize, color: '#FFFFFF' }));
                    } else if (child.type === 'filter') {
                        iconSvg = renderToStaticMarkup(React.createElement(MdFilterIcon, { size: iconSize, color: '#FFFFFF' }));
                    } else if (child.type === 'shape') {
                        iconSvg = renderToStaticMarkup(React.createElement(LuShapeIcon, { size: iconSize, color: '#FFFFFF' }));
                    }
                    if (iconSvg) {
                        const img = new (window as any).Image() as HTMLImageElement;
                        img.crossOrigin = 'anonymous';
                        // Ensure an SVG wrapper if not present
                        const svgWrapped = iconSvg.startsWith('<svg') ? iconSvg : `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${iconSize} ${iconSize}">${iconSvg}</svg>`;
                        img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgWrapped)}`;
                        await new Promise<void>((resolve) => {
                            img.onload = () => {
                                const x = Math.floor((canvas.width - iconSize) / 2);
                                const y = Math.floor((canvas.height - iconSize) / 2);
                                ctx.drawImage(img, x, y, iconSize, iconSize);
                                resolve();
                            };
                            img.onerror = () => resolve();
                        });
                    }
                    return canvas;
                }
                return null;
            }));
            setGroupedCanvases(canvases.filter((c) => c !== null) as HTMLCanvasElement[]);
        }
    })();
    }, [currentClip, getClipById, clipType]);

    return (
        <>
            <Group  
                ref={groupRef}
                onClick={handleClick} 
                draggable={resizeSide === null} 
                onDragEnd={handleDragEnd} 
                onDragMove={handleDragMove} 
                onDragStart={handleDragStart}
                onContextMenu={handleContextMenu}
                x={isDragging ? tempClipPosition.x : clipPosition.x } 
                y={(isDragging ? tempClipPosition.y : clipPosition.y)} 
                width={clipWidth}
                height={timelineHeight}
                clipX={0}
                clipY={0}
                clipWidth={clipWidth}
                clipHeight={timelineHeight}
                onMouseOver={handleMouseOver}
                onMouseLeave={handleMouseLeave}
                
                >
                {clipType === 'group' ? (
                    <Group>
                    <Rect
                        x={0}
                        y={0}
                        width={clipWidth}
                        height={timelineHeight}
                        cornerRadius={cornerRadius}
                        fillLinearGradientStartPoint={{ x: 0, y: 0 }}
                        fillLinearGradientEndPoint={{ x: clipWidth, y: 0 }}
                        fillLinearGradientColorStops={[0, '#AE81CE', 1, '#6A5ACD']}
                        opacity={0.9}
                    />
                    {/* Stacked preview cards */}
                    {groupedCanvases && groupedCanvases.length > 0 && (
                        <Group>
                            {(() => {
                                const configs = [
                                    { rotation: 16, dx: 12, dy: 6, opacity: 0.75, scale: 0.8 }, // back
                                    { rotation: 8, dx: 6, dy: 3, opacity: 0.9, scale: 0.90 },   // middle
                                    { rotation: 0, dx: 0, dy: 0, opacity: 1, scale: 1.0 },      // front
                                ];
                                const count = Math.min(3, groupedCanvases.length);
                                const startIdx = configs.length - count;
                                const used = configs.slice(startIdx);
                                const baseX = 10 + (groupCardWidth / 2);
                                const baseY = timelineHeight / 2;
                                // Render back-to-front with object-fit: cover
                                return used.map((cfg, i) => {
                                    const canvas = groupedCanvases[i];
                                    const iw = Math.max(1, (canvas as any).width || (canvas as any).naturalWidth || 1);
                                    const ih = Math.max(1, (canvas as any).height || (canvas as any).naturalHeight || 1);
                                    const cardW = Math.max(1, Math.round(groupCardWidth * (cfg as any).scale));
                                    const cardH = Math.max(1, Math.round(groupCardHeight * (cfg as any).scale));
                                    const targetRatio = Math.max(0.0001, cardW / Math.max(1, cardH));
                                    const sourceRatio = iw / ih;
                                    let cropX = 0, cropY = 0, cropW = iw, cropH = ih;
                                    if (sourceRatio > targetRatio) {
                                        // source is wider: crop left/right
                                        cropW = Math.max(1, Math.round(ih * targetRatio));
                                        cropX = Math.max(0, Math.round((iw - cropW) / 2));
                                        cropY = 0; cropH = ih;
                                    } else {
                                        // source is taller: crop top/bottom
                                        cropH = Math.max(1, Math.round(iw / targetRatio));
                                        cropY = Math.max(0, Math.round((ih - cropH) / 2));
                                        cropX = 0; cropW = iw;
                                    }
                                    return (
                                        <Image
                                            key={`group-card-${i}`}
                                            image={canvas}
                                            x={baseX + cfg.dx}
                                            y={baseY + cfg.dy}
                                            width={cardW}
                                            height={cardH}
                                            fill={'#1A2138'}
                                            crop={{ x: cropX, y: cropY, width: cropW, height: cropH }}
                                            offsetX={cardW / 2}
                                            offsetY={cardH / 2}
                                            rotation={cfg.rotation}
                                            opacity={cfg.opacity}
                                            cornerRadius={1}
                                            shadowColor={'#000000'}
                                            shadowBlur={8}
                                            shadowOpacity={0.18}
                                        />
                                    );
                                });
                            })()}
                                <Text
                                    x={groupCardWidth + 28}
                                    y={16}
                                    text={'Group'}
                                    fontSize={10.5}
                                    fontFamily="Poppins"
                                    fontStyle="500"
                                    fill="white"
                                    align="left"
                                />
                            
                            {(() => {
                                const items: { Icon: any; count:number }[] = [
                                    { Icon: MdMovieIcon, count: groupCounts.video },
                                    { Icon: MdImageIcon, count: groupCounts.image },
                                    { Icon: MdAudiotrackIcon, count: groupCounts.audio },
                                    { Icon: RxTextIcon, count: groupCounts.text },
                                    { Icon: MdOutlineDrawIcon, count: groupCounts.draw },
                                    { Icon: MdFilterIcon, count: groupCounts.filter },
                                    { Icon: LuShapeIcon, count: groupCounts.shape },
                                ].filter(i => i.count > 0);

                                const startX = groupCardWidth + 28;
                                const startY = 16 + 18; // below label
                                let curX = startX;
                                return items.map((it, idx) => {
                                    const Ico = it.Icon;
                                    const group = (
                                        <Group key={`gstat-${idx}`}>
                                            {/* icon */}
                                            <Image
                                                x={curX}
                                                y={startY - 1}
                                                width={12}
                                                height={12}
                                                image={(() => {
                                                    const svg = renderToStaticMarkup(React.createElement(Ico, { size: 12, color: '#FFFFFF' }));
                                                    const img = new (window as any).Image();
                                                    img.crossOrigin = 'anonymous';
                                                    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
                                                    return img as any;
                                                })()}
                                                opacity={0.85}
                                            />
                                            {/* count */}
                                            <Text x={curX + 16} y={startY - 1} text={`${it.count}`} fontSize={11} fontFamily="Poppins" fill="rgba(255,255,255,0.82)" />
                                        </Group>
                                    );
                                    curX += 28; // spacing between icon+count pairs
                                    return group;
                                });
                            })()}
                        </Group>
                    )}
                    
                    </Group>
                ) : (
                    <Image 
                        x={imageX}
                        y={0}
                        image={imageCanvas} 
                        width={imageWidth}
                        height={timelineHeight}
                        cornerRadius={cornerRadius}
                        fill={clipType === 'audio' ? '#1A2138' : '#FFFFFF'} 
                    />
                )}
                {clipType === 'shape' && (currentClip  as ShapeClipProps)?.shapeType && (
                    <Group>
                    <Rect
                        x={12 - 4}
                        y={timelineHeight / 2}
                        width={textWidth + 8}
                        height={14}
                        cornerRadius={2}
                        fill="rgba(255, 255, 255, 0.0)"
                        offsetY={7.5}
                    />
                    <Text
                        ref={textRef}
                        x={12}
                        y={timelineHeight / 2}
                        text={((currentClip as ShapeClipProps)?.shapeType?.charAt(0).toUpperCase() ?? '')  + ((currentClip as ShapeClipProps)?.shapeType?.slice(1 ) ?? '')}
                        fontSize={9.5}
                        fontFamily="Poppins"
                        fontStyle="500"
                        fill="white"
                        align="left"
                        verticalAlign="middle"
                        offsetY={5}
                    />
                    </Group>
                )}
                {clipType === 'text' && (currentClip  as TextClipProps)?.text && (
                    <Group>
                    <Rect
                        x={12 - 4}
                        y={timelineHeight / 2}
                        width={textWidth + 8}
                        height={14}
                        cornerRadius={2}
                        fill="rgba(0, 0, 0, 0.0)"
                        offsetY={7.5}
                    />
                    <Text
                        ref={textRef}
                        x={12}
                        y={timelineHeight / 2}
                        text={((currentClip as TextClipProps)?.text?.replace('\n', ' ') ?? '')}
                        fontSize={10}
                        fontFamily={(currentClip as TextClipProps)?.fontFamily ?? 'Poppins'}
                        fontStyle="500"
                        fill="#151517"
                        align="left"
                        verticalAlign="middle"
                        offsetY={5}
                    />
                    </Group>
                )}
                {clipType === 'filter' && (currentClip  as FilterClipProps)?.name && (
                    <Group>
                    <Rect
                        x={12 - 4}
                        y={timelineHeight / 2}
                        width={textWidth + 8}
                        height={14}
                        cornerRadius={2}
                        fill="rgba(0, 0, 0, 0.0)"
                        offsetY={7.5}
                    />
                    <Text
                        ref={textRef}
                        x={12}
                        y={timelineHeight / 2}
                        text={((currentClip as FilterClipProps)?.name ?? '')}
                        fontSize={9.5}
                        fontFamily={'Poppins'}
                        fontStyle="500"
                        fill="#ffffff"
                        align="left"
                        verticalAlign="middle"
                        offsetY={5}
                    />
                    </Group>
                )}
                {clipType === 'draw' && (
                    <Group>
                    <Rect
                        x={12 - 4}
                        y={timelineHeight / 2}
                        width={textWidth + 8}
                        height={14}
                        cornerRadius={2}
                        fill="rgba(255, 255, 255, 0.0)"
                        offsetY={7.5}
                    />
                    <Text
                        ref={textRef}
                        x={12}
                        y={timelineHeight / 2}
                        text="Drawing"
                        fontSize={9.5}
                        fontFamily="Poppins"
                        fontStyle="500"
                        fill="white"
                        align="left"
                        verticalAlign="middle"
                        offsetY={5}
                    />
                    </Group>
                )}
     
            </Group>            
            {/* Per-clip menu component retained (optional); global menu now handles rendering */}
            {hasPreprocessors && (currentClip?.type === 'video' || currentClip?.type === 'image') && (
                <>
                    {(currentClip as VideoClipProps | ImageClipProps).preprocessors.map((preprocessor) => {
                        return <PreprocessorClip 
                        key={preprocessor.id} 
                        preprocessor={preprocessor} 
                        currentStartFrame={currentStartFrame} 
                        currentEndFrame={currentEndFrame} 
                        timelineWidth={timelineWidth} 
                        clipPosition={clipPosition} 
                        timelineHeight={timelineHeight} 
                        isDragging={isDragging}
                        clipId={currentClipId} 
                        cornerRadius={cornerRadius} 
                        timelinePadding={timelinePadding} />
                    })}
                </>
            )}
            {showMaskKeyframes && currentClip?.type === 'video' && (
                <MaskKeyframes
                    clip={currentClip as VideoClipProps}
                    clipPosition={clipPosition}
                    clipWidth={clipWidth}
                    timelineHeight={timelineHeight}
                    isDragging={isDragging}
                    currentStartFrame={currentStartFrame}
                    currentEndFrame={currentEndFrame}
                />
            )}
            <Rect  
                ref={rectRefRight}
                x={isDragging ? clipPosition.x + clipWidth - 1: clipPosition.x + clipWidth - 2.5}
                y={isDragging ? clipPosition.y + 1.5 : clipPosition.y}
                width={3}
                visible={isSelected && clipType !== 'group'}
                height={timelineHeight}
                cornerRadius={[0, cornerRadius, cornerRadius, 0]}
                fill={isSelected ? '#FFFFFF': 'transparent'}
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
                        //moveClipToEnd(currentClipId);
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
                x={isDragging ? clipPosition.x + 1.5 : clipPosition.x }
                y={isDragging ? clipPosition.y + 1.5 : clipPosition.y}
                width={3}
                visible={isSelected && clipType !== 'group'}
                height={timelineHeight}
                cornerRadius={[cornerRadius, 0, 0, cornerRadius]}
                fill={isSelected ? '#FFFFFF': 'transparent'}
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
                {isSelected && (
                <Line
                    ref={clipRef} 
                    x={clipPosition.x}
                    y={clipPosition.y}
                    points={[
                        6, 0,
                        clipWidth - 6, 0,
                        clipWidth, 0,
                        clipWidth, 6,
                        clipWidth, timelineHeight - 6,
                        clipWidth, timelineHeight,
                        clipWidth - 6, timelineHeight,
                        6, timelineHeight,
                        0, timelineHeight,
                        0, timelineHeight - 6,
                        0, 6,
                        0, 0
                    ]}
                    stroke={'#FFFFFF'}
                    strokeWidth={2.0}
                    lineCap='round'
                    lineJoin='round'
                    listening={false}
                    bezier={false}
                    closed
                />
            )}
            </>
    )
}

export default TimelineClip;
