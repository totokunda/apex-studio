import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useClipStore, getCorrectedClip, getClipWidth, getClipX } from "@/lib/clip";
import { generateTimelineThumbnail, useMediaCache } from "@/lib/media";
import { useControlsStore } from "@/lib/control";
import { Image, Rect, Group } from 'react-konva';
import Konva from 'konva';
import { MediaInfo, TimelineProps } from "@/lib/types";

const THUMBNAIL_TILE_SIZE = 36;

const VideoTimeline: React.FC<TimelineProps & {clipId: string}> = ({timelineWidth = 0, timelineY = 0, timelineHeight = 64, timelinePadding = 24, clipId,  timelineId}) => {
    const controlStore = useControlsStore();
    const {clips, resizeClip, clipDuration, moveClipToEnd, updateClip, getClipById} = useClipStore();
    const {getMedia} = useMediaCache();
    
    // Find the current clip data from the store
    const currentClip = getClipById(clipId, timelineId);
    const currentStartFrame = currentClip?.startFrame ?? 0;
    const currentEndFrame = currentClip?.endFrame ?? 0;

    const clipWidth = useMemo(() => Math.max(getClipWidth(currentStartFrame, currentEndFrame, timelineWidth, controlStore.timelineDuration), 3), [currentStartFrame, currentEndFrame, timelineWidth, controlStore.timelineDuration]);
    const clipX = useMemo(() => getClipX(currentStartFrame, currentEndFrame, timelineWidth, controlStore.timelineDuration), [currentStartFrame, currentEndFrame, timelineWidth, controlStore.timelineDuration]);
    const [isDragging, setIsDragging] = useState(false);
    const clipRef = useRef<Konva.Image>(null);
    const [resizeSide, setResizeSide] = useState<'left' | 'right' | null>(null);
    const [isMouseOver, setIsMouseOver] = useState(false);
    const thumnailCanvasRef = useRef<HTMLCanvasElement | undefined>(undefined);
    const mediaInfoRef = useRef<MediaInfo | undefined>(undefined);

    // Use global selection state instead of local state
    const currentClipId = clipId;
    const isSelected = controlStore.selectedClipIds.includes(currentClipId);

    const minMaxX = useMemo(() => {
        return {
            min: timelinePadding,
            max: timelineWidth - clipWidth - timelinePadding
        }
    }, [timelinePadding, timelineWidth, clipWidth]);

    const [clipPosition, setClipPosition] = useState<{x:number, y:number}>({
        x: clipX + timelinePadding,
        y: timelineY - timelineHeight
    });

    useEffect(() => {
        const path = currentClip?.src!.split('/').pop()!;
        if (!currentClip?.src) return;
        if (!mediaInfoRef.current) {
            mediaInfoRef.current = getMedia(path);
        }
        
        (async () => {
            const maxClipWidth = Math.min(clipWidth, timelineWidth - timelinePadding * 2);

            let thumbnailWidth = THUMBNAIL_TILE_SIZE;
            const numColumns = Math.max(1, Math.floor(maxClipWidth /  thumbnailWidth));
            thumbnailWidth = maxClipWidth / numColumns;
            // find the visible frame start and end
            const [startFrame, endFrame] = controlStore.timelineDuration;
            const visibleStartFrame = Math.max(currentStartFrame, startFrame);
            const visibleEndFrame = Math.min(currentEndFrame, endFrame);
            if (visibleStartFrame >= visibleEndFrame) return;

            const step = Math.max(1, Math.floor((visibleEndFrame - visibleStartFrame) / numColumns));
            let frameIndices = [];
            for (let i = 0; i < numColumns; i++) {
                const frame = Math.floor(visibleStartFrame + i * step);
                if (frame >= visibleEndFrame) break;
                frameIndices.push(frame - currentStartFrame);
            }
            if (frameIndices.length < numColumns) {
                // repeat frame indices to fill the numColumns 
                const originalIndices = [...frameIndices];
                frameIndices = [];
                const framesPerColumn = Math.ceil(numColumns / originalIndices.length);

                for (let i = 0; i < originalIndices.length; i++) {
                    const frame = originalIndices[i];
                    const repeatCount = i === originalIndices.length - 1 
                        ? numColumns - frameIndices.length  // fill remaining columns with last frame
                        : framesPerColumn;
                    
                    for (let j = 0; j < repeatCount && frameIndices.length < numColumns; j++) {
                        frameIndices.push(frame);
                    }
                }
            }

            thumbnailWidth = Math.round(thumbnailWidth);
            timelineHeight = Math.round(timelineHeight);

            try {
            const thumbnail = await generateTimelineThumbnail(path, frameIndices, thumbnailWidth, timelineHeight, clipWidth, {canvas: thumnailCanvasRef.current, mediaInfo: mediaInfoRef.current});
            if (!thumbnail) return;
            thumnailCanvasRef.current = thumbnail as HTMLCanvasElement;
            moveClipToEnd(currentClipId);
            } catch (e) {
                thumnailCanvasRef.current = undefined;
            }

            clipRef.current?.getLayer()?.batchDraw?.();
        })();
    }, [currentClip?.src, clipWidth, controlStore.timelineDuration, timelineWidth, timelinePadding, timelineHeight, clipRef, mediaInfoRef]);
    
    const calculateFrameFromX = useCallback((xPosition: number) => {
        // Remove padding to get actual timeline position
        const timelineX = xPosition - timelinePadding;
        // Calculate the frame based on the position within the visible timeline
        const [startFrame, endFrame] = controlStore.timelineDuration;
        const framePosition = (timelineX / timelineWidth) * (endFrame - startFrame) + startFrame;
        return Math.round(framePosition);
    }, [timelinePadding, timelineWidth, controlStore.timelineDuration]);

    const handleDragMove = useCallback((e:Konva.KonvaEventObject<MouseEvent>) => {
        const halfStroke = isSelected ? 1.5 : 0;
        setClipPosition({x: e.target.x() - halfStroke, y: e.target.y() - halfStroke});
        if (clipRef.current) {
            clipRef.current.x(e.target.x());
            clipRef.current.y(e.target.y());
        }
    }, [clipRef, isSelected]);

    const handleDragEnd = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        setIsDragging(false);
        // make sure clip position is valid on the timeline (use outer bounds, not inner image x)
        const halfStroke = isSelected ? 1.5 : 0;
        const outerX = e.target.x() - halfStroke;
        const constrainedX = Math.max(minMaxX.min, Math.min(minMaxX.max, outerX));
        
        // Calculate the nearest frame based on the final position
        const nearestStartFrame = calculateFrameFromX(constrainedX);
        const nearestEndFrame = nearestStartFrame + (currentEndFrame - currentStartFrame);
        const resolvedClip = getCorrectedClip(clipId, clips);

        if (resolvedClip) {
            const newClipX = getClipX(resolvedClip.startFrame!, resolvedClip.endFrame!, timelineWidth, controlStore.timelineDuration);
            setClipPosition({x: newClipX + timelinePadding, y: timelineY - timelineHeight})
        }
        
        updateClip(clipId, {
            startFrame: nearestStartFrame,
            endFrame: nearestEndFrame,
        });
        
    }, [clips, clipRef, isDragging, minMaxX, calculateFrameFromX, timelineWidth, controlStore.timelineDuration, timelinePadding, timelineY, timelineHeight, clipDuration]);


    useEffect(() => {
        setClipPosition({x: clipX + timelinePadding, y: timelineY - timelineHeight});
    }, [clipWidth, clipX]);

    const handleClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        const isShiftClick = e.evt?.shiftKey || false;
        controlStore.toggleClipSelection(currentClipId, isShiftClick);
        
        // Move to end for better visual layering if this clip is now selected
        if (currentClipId && (isShiftClick || !isSelected)) {
            moveClipToEnd(currentClipId);
        }
    }, [isSelected, currentClipId, controlStore, moveClipToEnd]);

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
                
                if (newFrame !== currentEndFrame) {
                    // Use the new contiguous resize method - local state will update via useEffect
                    resizeClip(clipId, 'right', newFrame);
                }
            } else if (resizeSide === 'left') {
               
                if (newFrame !== currentStartFrame) {
                    // Use the new contiguous resize method - local state will update via useEffect
                    resizeClip(clipId, 'left', newFrame);
                }
            }
        };

        const handleMouseUp = () => {
            setResizeSide(null);
            stage.container().style.cursor = 'default';
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [resizeSide, currentStartFrame, currentEndFrame, calculateFrameFromX, clipId, resizeClip]);

    const handleDragStart = useCallback((_e:Konva.KonvaEventObject<MouseEvent>) => {
        // If this clip isn't already selected, select it (without shift behavior during drag)
        if (!isSelected) {
            controlStore.toggleClipSelection(currentClipId, false);
        }
        if (currentClipId) {
            moveClipToEnd(currentClipId);
        }
        setIsDragging(true);
    }, [currentClipId, controlStore, moveClipToEnd, isSelected]);

    const handleMouseOver = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        setIsMouseOver(true);
        moveClipToEnd(currentClipId);
        e.target.getStage()!.container().style.cursor = 'pointer';
    }, [isSelected]);
    const handleMouseLeave = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        setIsMouseOver(false);
        e.target.getStage()!.container().style.cursor = 'default';
    }, [isSelected]);

    // Adjust image bounds to account for stroke so outer size remains constant

    return (
            <Group onClick={handleClick}>
                <Image 
                image={thumnailCanvasRef.current} 
                draggable={resizeSide === null} 
                onDragEnd={handleDragEnd} 
                onDragMove={handleDragMove} 
                onDragStart={handleDragStart}
                ref={clipRef} 
                x={clipPosition.x} 
                y={clipPosition.y} 
                width={clipWidth} 
                height={timelineHeight}
                cornerRadius={8}
                stroke={isSelected ? '#AE81CE' : isMouseOver ? 'rgba(227,227,227,1)' : undefined} 
                strokeWidth={isSelected ? 4 : undefined} 
                fill={'#E3E3E3'} 
                onMouseOver={handleMouseOver}
                onMouseLeave={handleMouseLeave}
                />
                <Rect
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
                            controlStore.toggleClipSelection(currentClipId, false);
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
                    x={clipPosition.x }
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
                            controlStore.toggleClipSelection(currentClipId, false);
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
            </Group>
    )
}

export default VideoTimeline;