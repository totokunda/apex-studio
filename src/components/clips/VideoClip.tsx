import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useClipStore, getCorrectedClip, getClipWidth, getClipX, getClipImage } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { Image, Rect, Group } from 'react-konva';
import Konva from 'konva';
import { VideoClipProps } from "@/lib/types";


const VideoClip: React.FC<VideoClipProps> = ({timelineWidth = 0, timelineY = 0, timelineHeight = 72, clipPadding = 24, src, clipId}) => {
    const controlStore = useControlsStore();
    const {clips, resizeClip, clipDuration, moveClipToEnd, updateClip} = useClipStore();
    
    // Find the current clip data from the store
    const currentClip = clips.find(c => c.clipId === clipId);
    const currentStartFrame = currentClip?.startFrame ?? 0;
    const currentEndFrame = currentClip?.endFrame ?? 0;

    const clipWidth = useMemo(() => getClipWidth(currentStartFrame, currentEndFrame, timelineWidth, controlStore.timelineDuration), [currentStartFrame, currentEndFrame, timelineWidth, controlStore.timelineDuration]);
    const clipX = useMemo(() => getClipX(currentStartFrame, currentEndFrame, timelineWidth, controlStore.timelineDuration), [currentStartFrame, currentEndFrame, timelineWidth, controlStore.timelineDuration]);
    const [isDragging, setIsDragging] = useState(false);
    const clipRef = useRef<Konva.Image>(null);
    const [clipImage, setClipImage] = useState<CanvasImageSource | undefined>(undefined);
    const [resizeSide, setResizeSide] = useState<'left' | 'right' | null>(null);

    // Use global selection state instead of local state
    const currentClipId = clipId;
    const isSelected = controlStore.selectedClipIds.includes(currentClipId);

    const minMaxX = useMemo(() => {
        return {
            min: clipPadding,
            max: timelineWidth - clipWidth - clipPadding
        }
    }, [clipPadding, timelineWidth, clipWidth]);

    const [clipPosition, setClipPosition] = useState<{x:number, y:number}>({
        x: clipX + clipPadding,
        y: timelineY - timelineHeight
    });

    const calculateFrameFromX = useCallback((xPosition: number) => {
        // Remove padding to get actual timeline position
        const timelineX = xPosition - clipPadding;
        // Calculate the frame based on the position within the visible timeline
        const [startFrame, endFrame] = controlStore.timelineDuration;
        const framePosition = (timelineX / timelineWidth) * (endFrame - startFrame) + startFrame;
        return Math.round(framePosition);
    }, [clipPadding, timelineWidth, controlStore.timelineDuration]);

    const handleDragMove = useCallback((e:Konva.KonvaEventObject<MouseEvent>) => {
        setClipPosition({x: e.target.x(), y: e.target.y()});
        if (clipRef.current) {
            clipRef.current.x(e.target.x());
            clipRef.current.y(e.target.y());
        }
    }, [clipRef]);

    const handleDragEnd = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        setIsDragging(false);
        // make sure clip position is valid on the timeline
        const constrainedX = Math.max(minMaxX.min, Math.min(minMaxX.max, e.target.x()));
        
        // Calculate the nearest frame based on the final position
        const nearestStartFrame = calculateFrameFromX(constrainedX);
        const nearestEndFrame = nearestStartFrame + (currentEndFrame - currentStartFrame);
        const resolvedClip = getCorrectedClip(clipId, clips);
        if (resolvedClip) {
            const newClipX = getClipX(resolvedClip.startFrame!, resolvedClip.endFrame!, timelineWidth, controlStore.timelineDuration);
            setClipPosition({x: newClipX + clipPadding, y: timelineY - timelineHeight})
        }
        
        updateClip(clipId, {
            startFrame: nearestStartFrame,
            endFrame: nearestEndFrame,
        });
        
    }, [clips, clipRef, isDragging, minMaxX, calculateFrameFromX, timelineWidth, controlStore.timelineDuration, clipPadding, timelineY, timelineHeight, clipDuration]);

    useEffect(() => {
        let isCancelled = false;
        getClipImage(src).then((canvas) => {
            if (isCancelled) return;
            setClipImage(canvas ?? undefined);
        });
        return () => {
            isCancelled = true;
        }
    }, [src]);

    useEffect(() => {
        setClipPosition({x: clipX + clipPadding, y: timelineY - timelineHeight});
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

    return (
            <Group onClick={handleClick}>
                
                <Image image={clipImage} 
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
                stroke={isSelected ? '#ADD8E6' : 'transparent'} 
                strokeWidth={4} 
                fill={'#E3E3E3'} 
                />
                <Rect
                    x={clipPosition.x  + clipWidth - 2}
                    y={clipPosition.y + timelineHeight / 2 - 10}
                    width={1.5}
                    height={20}
                    cornerRadius={[3, 0, 0, 3]}
                    stroke={isSelected ? '#ADD8E6' : 'transparent'}
                    strokeWidth={3}
                    fill={isSelected ? '#ADD8E6': 'transparent'}
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
                    y={clipPosition.y + timelineHeight / 2 - 10}
                    width={1.5}
                    height={20}
                    cornerRadius={[0, 3, 3, 0]}
                    stroke={isSelected ? '#ADD8E6' : 'transparent'}
                    strokeWidth={3}
                    fill={isSelected ? '#ADD8E6': 'transparent'}
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

export default VideoClip;