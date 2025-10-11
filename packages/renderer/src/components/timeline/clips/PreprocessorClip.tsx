import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {  PreprocessorClipProps } from '@/lib/types'
import { getClipWidth, useClipStore } from '@/lib/clip'
import { Rect} from 'react-konva'
import { Text } from 'react-konva'
import { Group } from 'react-konva'
import { Html } from 'react-konva-utils'
import { FaPlay } from 'react-icons/fa6'
import { LuTrash } from 'react-icons/lu'
import { useControlsStore } from '@/lib/control'
import Konva from 'konva'
import { calculateFrameFromX as calcFrameFromX, getOtherPreprocessors as getOthers, detectCollisions as detectColls, findGapAfterBlock as findGap } from '@/lib/preprocessorHelpers'

interface PropsPreprocessorClip {
    preprocessor: PreprocessorClipProps
    currentStartFrame: number
    currentEndFrame: number
    timelineWidth: number
    clipPosition: {x: number, y: number}
    timelineHeight: number
    isDragging: boolean
    clipId: string
    cornerRadius: number
    timelinePadding: number
}

export const PreprocessorClip:React.FC<PropsPreprocessorClip> = ({preprocessor, currentStartFrame, currentEndFrame, timelineWidth,  clipPosition, timelineHeight,  cornerRadius, clipId, timelinePadding, isDragging}) => {
    const {timelineDuration, setSelectedClipIds} = useControlsStore()
    const removePreprocessorFromClip = useClipStore((s) => s.removePreprocessorFromClip);
    const getPreprocessorsForClip = useClipStore((s) => s.getPreprocessorsForClip);
    // Preprocessor frames are stored relative to the parent clip (0 = clip start)
    const preprocessorStartFrame = preprocessor.startFrame ?? 0;
    const preprocessorEndFrame = preprocessor.endFrame ?? (currentEndFrame - currentStartFrame);
    const textRef = useRef<Konva.Text>(null);
    
    // Calculate parent clip dimensions
    const clipDuration = currentEndFrame - currentStartFrame;
    const clipWidth = Math.max(getClipWidth(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), 3);
    
    // Calculate preprocessor position and size as proportion of parent clip
    const preprocessorDuration = useMemo(() => preprocessorEndFrame - preprocessorStartFrame, [preprocessorEndFrame, preprocessorStartFrame]);
    const preprocessorX = useMemo(() => (preprocessorStartFrame / clipDuration) * clipWidth, [preprocessorStartFrame, clipDuration, clipWidth]);
    const preprocessorWidth = useMemo(() => Math.max((preprocessorDuration / clipDuration) * clipWidth, 3), [preprocessorDuration, clipDuration, clipWidth]);
    const selectedPreprocessorId = useClipStore((s) => s.selectedPreprocessorId);
    const setSelectedPreprocessorId = useClipStore((s) => s.setSelectedPreprocessorId);
    const updatePreprocessor = useClipStore((s) => s.updatePreprocessor);
    const [resizingPreprocessor, setResizingPreprocessor] = useState<{id: string, side: 'left' | 'right'} | null>(null);
    const preprocessorRef = useRef<Konva.Group>(null);
    const prevClipBounds = useRef({ startFrame: currentStartFrame, endFrame: currentEndFrame });
    const [isAltPressed, setIsAltPressed] = useState(false);
    const previousMouseX = useRef<number | null>(null);

    useEffect(() => {
        if (isDragging) {
                preprocessorRef.current?.moveToTop();
                setSelectedPreprocessorId(null);
        }
    }, [isDragging, preprocessor.id, setSelectedPreprocessorId]);

    // Track Alt key state globally
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Alt') {
                setIsAltPressed(true);
            }
            if (e.key === 'Delete' && selectedPreprocessorId === preprocessor.id) {
                removePreprocessorFromClip(clipId, preprocessor.id);
            }
        };
        
        const handleKeyUp = (e: KeyboardEvent) => {
            if (e.key === 'Alt') {
                setIsAltPressed(false);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);
        
        // Handle case where window loses focus while Alt is pressed
        const handleBlur = () => {
            setIsAltPressed(false);
        };
        window.addEventListener('blur', handleBlur);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
            window.removeEventListener('blur', handleBlur);
        };
    }, [selectedPreprocessorId, preprocessor.id]);

    // Adjust preprocessor when parent clip is resized
    useEffect(() => {
        const prevStart = prevClipBounds.current.startFrame;
        const prevEnd = prevClipBounds.current.endFrame;
        const clipDurationNow = currentEndFrame - currentStartFrame;
        
        // Detect if clip was resized or moved
        const startDelta = currentStartFrame - prevStart;
        const endDelta = currentEndFrame - prevEnd;
        
        // If no change, just update ref and return
        if (startDelta === 0 && endDelta === 0) {
            prevClipBounds.current = { startFrame: currentStartFrame, endFrame: currentEndFrame };
            return;
        }

        // If both start and end changed by the same amount, it's a drag/move, not a resize
        // In this case, preprocessor maintains its relative position (no update needed)
        if (startDelta === endDelta) {
            prevClipBounds.current = { startFrame: currentStartFrame, endFrame: currentEndFrame };
            return;
        }

        let needsUpdate = false;
        let newStartFrame = preprocessorStartFrame;
        let newEndFrame = preprocessorEndFrame;
        const preprocessorDuration = preprocessorEndFrame - preprocessorStartFrame;

        // Clip was resized (not just moved)
        // If clip start moved (left resize), shift preprocessor to maintain absolute position
        if (startDelta !== 0) {
            // Shift preprocessor in opposite direction to maintain absolute position
            newStartFrame = preprocessorStartFrame - startDelta;
            newEndFrame = preprocessorEndFrame - startDelta;
            
            // If shifted too far left, clamp to start and shrink if needed
            if (newStartFrame < 0) {
                newStartFrame = 0;
                newEndFrame = Math.min(preprocessorDuration, clipDurationNow);
            }
            
            needsUpdate = true;
        }

        // If clip end moved (right resize), only adjust if preprocessor exceeds bounds
        if (endDelta !== 0 && newEndFrame > clipDurationNow) {
            // First try to shift left to keep full duration
            const overflow = newEndFrame - clipDurationNow;
            if (newStartFrame >= overflow) {
                newStartFrame -= overflow;
                newEndFrame = clipDurationNow;
            } else {
                // Not enough space to shift, shrink from the end
                newEndFrame = clipDurationNow;
            }
            needsUpdate = true;
        }

        // Final bounds check - ensure preprocessor stays within clip
        if (newStartFrame < 0) {
            newStartFrame = 0;
            needsUpdate = true;
        }
        
        if (newEndFrame > clipDurationNow) {
            newEndFrame = clipDurationNow;
            needsUpdate = true;
        }

        // If start is beyond clip duration, move it back
        if (newStartFrame >= clipDurationNow) {
            newStartFrame = Math.max(0, clipDurationNow - 1);
            needsUpdate = true;
        }

        // Ensure end is after start
        if (newEndFrame <= newStartFrame) {
            newEndFrame = Math.min(newStartFrame + 1, clipDurationNow);
            needsUpdate = true;
        }

        if (needsUpdate) {
            updatePreprocessor(clipId, preprocessor.id, { 
                startFrame: newStartFrame, 
                endFrame: newEndFrame 
            });
        }

        // Update previous bounds
        prevClipBounds.current = { startFrame: currentStartFrame, endFrame: currentEndFrame };
    }, [currentStartFrame, currentEndFrame, preprocessorStartFrame, preprocessorEndFrame, clipId, preprocessor.id, updatePreprocessor]);

    const calculateFrameFromX = useCallback((xPosition: number) => {
        return calcFrameFromX(xPosition, timelinePadding, timelineWidth, timelineDuration);
    }, [timelinePadding, timelineWidth, timelineDuration]);

    const getOtherPreprocessors = useCallback(() => {
        return getOthers(getPreprocessorsForClip(clipId), preprocessor.id);
    }, [getPreprocessorsForClip, clipId, preprocessor.id]);

    const detectCollisions = useCallback((targetStart: number, targetEnd: number) => {
        const others = getOtherPreprocessors();
        const clipDurationNow = currentEndFrame - currentStartFrame;
        return detectColls(targetStart, targetEnd, others, clipDurationNow);
    }, [getOtherPreprocessors, currentEndFrame, currentStartFrame]);

    const findGapAfterBlock = useCallback((
        collidingPreprocessors: PreprocessorClipProps[],
        direction: 'left' | 'right',
        preprocessorDuration: number
    ): number | null => {
        const clipDurationNow = currentEndFrame - currentStartFrame;
        return findGap(collidingPreprocessors, direction, preprocessorDuration, clipDurationNow);
    }, [currentEndFrame, currentStartFrame]);

    const handleDragStart = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
            // Only allow preprocessor drag if Alt key is held OR dragging from header
            e.target.getStage()!.container().style.cursor = 'grab';
            setSelectedPreprocessorId(preprocessor.id);
            e.target.moveToTop();
            // Reset mouse tracking for clean drag direction detection
            previousMouseX.current = null;
        }, [preprocessor.id, setSelectedPreprocessorId]);
    
    const handleDragMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        const stage = e.target.getStage();
        if (!stage) return;

        // Get actual mouse position from the stage pointer, not the element's position
        const pointerPosition = stage.getPointerPosition();
        if (!pointerPosition) return;
        
        const mouseX = pointerPosition.x;
        
        // Determine drag direction by comparing with previous position
        let dragDirection: 'left' | 'right' | null = null;
        if (previousMouseX.current !== null) {
            if (mouseX > previousMouseX.current) {
                dragDirection = 'right';
            } else if (mouseX < previousMouseX.current) {
                dragDirection = 'left';
            }
        }
        previousMouseX.current = mouseX;

        // Calculate desired position based on actual mouse position
        let absoluteStartFrame = calculateFrameFromX(mouseX);
        const preprocessorDuration = preprocessorEndFrame - preprocessorStartFrame;
        let absoluteEndFrame = absoluteStartFrame + preprocessorDuration;

        // Enforce bounds within the parent clip
        if (absoluteStartFrame < currentStartFrame) {
            absoluteStartFrame = currentStartFrame;
            absoluteEndFrame = absoluteStartFrame + preprocessorDuration;
        }
        
        if (absoluteEndFrame > currentEndFrame) {
            absoluteEndFrame = currentEndFrame;
            absoluteStartFrame = absoluteEndFrame - preprocessorDuration;
        }

        // Convert to relative frames (relative to clip start)
        let relativeStartFrame = absoluteStartFrame - currentStartFrame;
        let relativeEndFrame = absoluteEndFrame - currentStartFrame;

        // Check for collisions
        const collisions = detectCollisions(relativeStartFrame, relativeEndFrame);
        
        if (collisions.length > 0 && dragDirection) {
            // We have a collision - check if we can snap around it
            const snapPosition = findGapAfterBlock(collisions, dragDirection, preprocessorDuration);
            
            if (snapPosition !== null) {
                // Check if mouse has reached the snap position before applying it
                if (dragDirection === 'right' && relativeStartFrame >= snapPosition) {
                    // Mouse is at or past the snap position - apply the snap
                    relativeStartFrame = snapPosition;
                    relativeEndFrame = snapPosition + preprocessorDuration;
                } else if (dragDirection === 'left' && relativeStartFrame <= snapPosition) {
                    // Mouse is at or past the snap position - apply the snap
                    relativeStartFrame = snapPosition;
                    relativeEndFrame = snapPosition + preprocessorDuration;
                } else {
                    // Mouse hasn't reached snap position yet - move as close as possible to the blocking preprocessor
                    const sorted = [...collisions].sort((a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0));
                    if (dragDirection === 'right') {
                        // Find the leftmost edge of the blocking preprocessor(s)
                        const blockStart = sorted[0].startFrame ?? 0;
                        // Move as close as possible without overlapping
                        relativeStartFrame = Math.min(relativeStartFrame, blockStart - preprocessorDuration);
                        relativeEndFrame = relativeStartFrame + preprocessorDuration;
                    } else {
                        // Find the rightmost edge of the blocking preprocessor(s)
                        let blockEnd = sorted[0].endFrame ?? (currentEndFrame - currentStartFrame);
                        for (const p of sorted) {
                            const pEnd = p.endFrame ?? (currentEndFrame - currentStartFrame);
                            if (pEnd > blockEnd) blockEnd = pEnd;
                        }
                        // Move as close as possible without overlapping
                        relativeStartFrame = Math.max(relativeStartFrame, blockEnd);
                        relativeEndFrame = relativeStartFrame + preprocessorDuration;
                    }
                }
            } else {
                // No gap available - move as close as possible to the blocking preprocessor
                const sorted = [...collisions].sort((a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0));
                if (dragDirection === 'right') {
                    // Find the leftmost edge of the blocking preprocessor(s)
                    const blockStart = sorted[0].startFrame ?? 0;
                    // Move as close as possible without overlapping
                    relativeStartFrame = Math.min(relativeStartFrame, blockStart - preprocessorDuration);
                    relativeEndFrame = relativeStartFrame + preprocessorDuration;
                } else {
                    // Find the rightmost edge of the blocking preprocessor(s)
                    let blockEnd = sorted[0].endFrame ?? (currentEndFrame - currentStartFrame);
                    for (const p of sorted) {
                        const pEnd = p.endFrame ?? (currentEndFrame - currentStartFrame);
                        if (pEnd > blockEnd) blockEnd = pEnd;
                    }
                    // Move as close as possible without overlapping
                    relativeStartFrame = Math.max(relativeStartFrame, blockEnd);
                    relativeEndFrame = relativeStartFrame + preprocessorDuration;
                }
            }
        }

        // Final bounds check - ensure preprocessor stays within parent clip
        const clipDurationNow = currentEndFrame - currentStartFrame;
        relativeStartFrame = Math.max(0, Math.min(relativeStartFrame, clipDurationNow - preprocessorDuration));
        relativeEndFrame = Math.max(preprocessorDuration, Math.min(relativeEndFrame, clipDurationNow));
        
        // Ensure start is before end
        if (relativeStartFrame >= relativeEndFrame) {
            relativeStartFrame = Math.max(0, relativeEndFrame - preprocessorDuration);
        }
        if (relativeEndFrame <= relativeStartFrame) {
            relativeEndFrame = Math.min(clipDurationNow, relativeStartFrame + preprocessorDuration);
        }

        // Final collision check - if there's still a collision, don't update
        const finalCollisionCheck = detectCollisions(relativeStartFrame, relativeEndFrame);
        if (finalCollisionCheck.length > 0) {
            // Collision detected - revert to current position and return
            const [timelineStartFrame, timelineEndFrame] = timelineDuration;
            const absoluteCurrentStart = preprocessorStartFrame + currentStartFrame;
            const relativeToTimeline = (absoluteCurrentStart - timelineStartFrame) / (timelineEndFrame - timelineStartFrame);
            const currentX = timelinePadding + (relativeToTimeline * timelineWidth);
            e.target.x(currentX);
            return;
        }

        // Calculate the visual x position based on the collision-adjusted frames
        const [timelineStartFrame, timelineEndFrame] = timelineDuration;
        const absoluteAdjustedStart = relativeStartFrame + currentStartFrame;
        const relativeToTimeline = (absoluteAdjustedStart - timelineStartFrame) / (timelineEndFrame - timelineStartFrame);
        const adjustedX = timelinePadding + (relativeToTimeline * timelineWidth);
        
        // Update the visual position immediately to avoid jitter
        e.target.x(adjustedX);

        updatePreprocessor(clipId, preprocessor.id, { startFrame: relativeStartFrame, endFrame: relativeEndFrame });
    }, [calculateFrameFromX, preprocessorEndFrame, preprocessorStartFrame, currentStartFrame, currentEndFrame, clipId, preprocessor.id, updatePreprocessor,  timelineDuration, timelinePadding, timelineWidth, detectCollisions, findGapAfterBlock]);

    const handleDragEnd = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        e.target.getStage()!.container().style.cursor = 'default';
    }, []);

    const dragBoundFunc = useCallback((pos: {x: number, y: number}) => {
        // Just constrain Y, let handleDragMove handle X collisions
        return {
            x: pos.x,
            y: clipPosition.y
        };
    }, [clipPosition.y]);
 
    useEffect(() => {
        if (!resizingPreprocessor) return;
        const stage = preprocessorRef.current?.getStage();
        if (!stage) return;

        const handleMouseMove = (e: MouseEvent) => {
            stage.container().style.cursor = 'col-resize';
            const rect = stage.container().getBoundingClientRect();
            const stageX = e.clientX - rect.left;
            
            // Calculate absolute timeline frame from mouse position
            const absoluteFrame = calculateFrameFromX(stageX);
            
            // Convert to relative frame within the clip
            const relativeFrame = absoluteFrame - currentStartFrame;
            const clipDurationNow = currentEndFrame - currentStartFrame;

            if (resizingPreprocessor.side === 'right') {
                // Resizing right edge
                const minRelativeFrame = preprocessorStartFrame + 1;
                let maxRelativeFrame = clipDurationNow;
                
                // Check for collision with other preprocessors
                const otherPreprocessors = getOtherPreprocessors();
                for (const other of otherPreprocessors) {
                    const otherStart = other.startFrame ?? 0;
                    
                    // If other preprocessor is to the right and would collide
                    if (otherStart > preprocessorStartFrame && otherStart < maxRelativeFrame) {
                        maxRelativeFrame = otherStart;
                    }
                }
                
                const targetRelativeFrame = Math.max(minRelativeFrame, Math.min(maxRelativeFrame, relativeFrame));
                updatePreprocessor(clipId, preprocessor.id, { endFrame: targetRelativeFrame });
            } else if (resizingPreprocessor.side === 'left') {
                // Resizing left edge
                let minRelativeFrame = 0;
                const maxRelativeFrame = preprocessorEndFrame - 1;
                
                // Check for collision with other preprocessors
                const otherPreprocessors = getOtherPreprocessors();
                for (const other of otherPreprocessors) {
                    const otherEnd = other.endFrame ?? clipDurationNow;
                    
                    // If other preprocessor is to the left and would collide
                    if (otherEnd < preprocessorEndFrame && otherEnd > minRelativeFrame) {
                        minRelativeFrame = otherEnd;
                    }
                }
                
                const targetRelativeFrame = Math.max(minRelativeFrame, Math.min(maxRelativeFrame, relativeFrame));
                updatePreprocessor(clipId, preprocessor.id, { startFrame: targetRelativeFrame });
            }
        };

        const handleMouseUp = () => {
            setResizingPreprocessor(null);
            stage.container().style.cursor = 'default';
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [resizingPreprocessor, currentStartFrame, currentEndFrame, calculateFrameFromX, clipId, updatePreprocessor, preprocessor.id, preprocessorStartFrame, preprocessorEndFrame, getOtherPreprocessors]);

    // Preprocessor is interactive only when Alt is pressed OR when hovering/interacting with header
    const isListening = !isAltPressed;

    return (
        <Group
            id={preprocessor.id}
            x={clipPosition.x + preprocessorX}
            y={clipPosition.y}
            clipX={0}
            clipY={0}
            draggable 
            listening={isListening}
            key={preprocessor.id} 
            onDragStart={handleDragStart} 
            onDragMove={handleDragMove} onDragEnd={handleDragEnd} dragBoundFunc={dragBoundFunc}
            clipWidth={preprocessorWidth}
            ref={preprocessorRef}
            clipFunc={(ctx) => {
                ctx.rect(-4, -4, preprocessorWidth + 6, timelineHeight + 4);
            }}
            
            >
        <React.Fragment key={preprocessor.id}>
            <Rect
                x={0}
                y={0}
                width={preprocessorWidth}
                onClick={() => {
                    // deselect all other clips
                    setSelectedPreprocessorId(preprocessor.id);
                    setSelectedClipIds([]);
                }}
                height={timelineHeight}
                fill={selectedPreprocessorId === preprocessor.id ? 'rgba(0, 200, 180, 0.75)' : 'rgba(0, 200, 180, 0.4)'}
                stroke={'rgba(255, 255, 255, 1)'}
                strokeWidth={selectedPreprocessorId === preprocessor.id ? 2 :0 }
                cornerRadius={cornerRadius}
                onMouseOver={(e) => {
                    e.target.getStage()!.container().style.cursor = 'grab';
                }}
                onMouseLeave={(e) => {
                    if (!resizingPreprocessor) {
                        e.target.getStage()!.container().style.cursor = 'default';
                    }
                }}
            />
            <Rect  
                x={0}
                y={0}
                width={preprocessorWidth}
                height={18}
                fill={'rgba(0, 90, 80, 1)'}
                strokeWidth={1}
                cornerRadius={cornerRadius}
                listening={true}
                onMouseOver={(e) => {
                    e.target.getStage()!.container().style.cursor = 'grab';
                }}
                onMouseLeave={(e) => {
                    if (!resizingPreprocessor) {
                        e.target.getStage()!.container().style.cursor = 'default';
                    }
                }}
             /> 
            <Text
                x={4}
                y={4}
                text={preprocessor.preprocessor.name}
                fontSize={9}
                fontFamily={'Poppins'}
                listening={true}
                fill={'white'}
                ref={textRef}
            />
            {/* Only show icons if there's enough space (at least 35px wide) */}
            {preprocessorWidth >= (textRef.current?.width() ?? 0) + 32 && (
                <Group 
                    x={Math.max(preprocessorWidth - 28, 4)} 
                    y={4}
                    listening={true}
                >
                <Html >
                    <div className="flex items-center gap-x-1" style={{ overflow: 'hidden', maxWidth: '24px' }}>
                    <FaPlay size={10} fill={'white'} className="cursor-pointer" />
                    <LuTrash onClick={() => removePreprocessorFromClip(clipId, preprocessor.id)} size={10}  className="cursor-pointer text-white hover:fill-white transition-colors duration-200" />
                    </div>
                </Html>
                </Group>
            )}
            {/* Left drag handle for preprocessor */}
            <Rect
                x={-1.5}
                y={0}
                width={1.5}
                height={timelineHeight}
                visible={selectedPreprocessorId === preprocessor.id}
                cornerRadius={[cornerRadius, 0, 0, cornerRadius]}
                fill={'white'}
                stroke={'white'}
                onMouseOver={(e) => {
                    e.target.getStage()!.container().style.cursor = 'col-resize';
                }}
                onMouseDown={(e) => {
                    e.cancelBubble = true;
                    e.evt.stopPropagation();
                    setResizingPreprocessor({id: preprocessor.id, side: 'left'});
                    e.target.getStage()!.container().style.cursor = 'col-resize';
                }}
                onMouseLeave={(e) => {
                    e.target.getStage()!.container().style.cursor = 'default';
                }}
            />
            {/* Right drag handle for preprocessor */}
            <Rect
                x={preprocessorWidth}
                y={0}
                width={1.5}
                height={timelineHeight}
                visible={selectedPreprocessorId === preprocessor.id}
                cornerRadius={[0, cornerRadius, cornerRadius, 0]}
                fill={'white'}
                stroke={'white'}
                onMouseOver={(e) => {
                    e.target.getStage()!.container().style.cursor = 'col-resize';
                }}
                onMouseDown={(e) => {
                    e.cancelBubble = true;
                    e.evt.stopPropagation();
                    setResizingPreprocessor({id: preprocessor.id, side: 'right'});
                    e.target.getStage()!.container().style.cursor = 'col-resize';
                }}
                onMouseLeave={(e) => {
                    e.target.getStage()!.container().style.cursor = 'default';
                }}
            />
        </React.Fragment>
        </Group>
    );
}