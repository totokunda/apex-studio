import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useClipStore, getClipWidth, getClipX } from "@/lib/clip";
import { generateTimelineThumbnail } from "@/lib/media/timeline";
import { useMediaCache } from "@/lib/media/cache";
import { useControlsStore } from "@/lib/control";
import { Image, Rect, Group } from 'react-konva';
import Konva from 'konva';
import { MediaInfo, TimelineProps } from "@/lib/types";
import { v4 as uuidv4 } from 'uuid';

const THUMBNAIL_TILE_SIZE = 36;

const TimelineClip: React.FC<TimelineProps & {clipId: string, type: 'video' | 'audio' | 'image', scrollY: number}> = ({timelineWidth = 0, timelineY = 0, timelineHeight = 64, timelinePadding = 24, clipId,  timelineId, type, scrollY}) => {
    // Select only what we need to avoid unnecessary rerenders
    const timelineDuration = useControlsStore((s) => s.timelineDuration);
    const selectedClipIds = useControlsStore((s) => s.selectedClipIds);
    const toggleClipSelection = useControlsStore((s) => s.toggleClipSelection);

    const resizeClip = useClipStore((s) => s.resizeClip);
    const moveClipToEnd = useClipStore((s) => s.moveClipToEnd);
    const updateClip = useClipStore((s) => s.updateClip);
    const setGhostX = useClipStore((s) => s.setGhostX);
    const setGhostTimelineId = useClipStore((s) => s.setGhostTimelineId);
    const setGhostStartEndFrame = useClipStore((s) => s.setGhostStartEndFrame);
    const setGhostInStage = useClipStore((s) => s.setGhostInStage);
    const setDraggingClipId = useClipStore((s) => s.setDraggingClipId);
    const getClipsForTimeline = useClipStore((s) => s.getClipsForTimeline);
    const setHoveredTimelineId = useClipStore((s) => s.setHoveredTimelineId);
    const setSnapGuideX = useClipStore((s) => s.setSnapGuideX);
    const addTimeline = useClipStore((s) => s.addTimeline);
    // Subscribe directly to this clip's data
    const currentClip = useClipStore((s) => s.clips.find((c) => c.clipId === clipId && (timelineId ? c.timelineId === timelineId : true)));
    const {getMedia} = useMediaCache();
    
    const currentStartFrame = currentClip?.startFrame ?? 0;
    const currentEndFrame = currentClip?.endFrame ?? 0;

    const clipWidth = useMemo(() => Math.max(getClipWidth(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), 3), [currentStartFrame, currentEndFrame, timelineWidth, timelineDuration, timelineId]);
    const clipX = useMemo(() => getClipX(currentStartFrame, currentEndFrame, timelineWidth, timelineDuration), [currentStartFrame, currentEndFrame, timelineWidth, timelineDuration, timelineId]);
    
    
    const clipRef = useRef<Konva.Image>(null);
    const [resizeSide, setResizeSide] = useState<'left' | 'right' | null>(null);
    const [isMouseOver, setIsMouseOver] = useState(false);
    const thumnailCanvasRef = useRef<HTMLCanvasElement | undefined>(undefined);
    const mediaInfoRef = useRef<MediaInfo | undefined>(undefined);
    const dragInitialWindowRef = useRef<[number, number] | null>(null);

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

    useEffect(() => {
        setClipPosition({x: clipX + timelinePadding, y: timelineY - timelineHeight});
    }, [timelinePadding, timelineY, timelineId]);

    useEffect(() => {
        const path = currentClip?.src!;
        if (!currentClip?.src) return;
        let cancelled = false;
        const timeoutId = window.setTimeout(() => {
            if (cancelled) return;
            if (!mediaInfoRef.current) {
                mediaInfoRef.current = getMedia(path);
            }
            
            (async () => {
                const maxClipWidth = Math.min(clipWidth, timelineWidth - timelinePadding * 2);
                let width:number = 1;
                let height:number = 1;
            
                if (type === 'video') { 
                    width = mediaInfoRef.current?.video?.codedWidth ?? 1;
                    height = mediaInfoRef.current?.video?.codedHeight ?? 1
                } else if (type === 'image') {
                    width = mediaInfoRef.current?.image?.width ?? 1;
                    height = mediaInfoRef.current?.image?.height ?? 1
                } else if (type === 'audio') {
                    // Will change of course
                    width = mediaInfoRef.current?.stats.audio?.averagePacketRate ?? 1;
                    height = timelineHeight;
                } 
                const ratio = width / height;
                let thumbnailWidth = Math.max(timelineHeight * ratio, THUMBNAIL_TILE_SIZE);
                const numColumns = Math.max(1, Math.floor(maxClipWidth /  thumbnailWidth));
                thumbnailWidth = maxClipWidth / numColumns;
                // find the visible frame start and end
                const [startFrame, endFrame] = timelineDuration;
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
                    const clipLocalStart = currentStartFrame
                    const clipLocalEnd = currentEndFrame;
                    const id = currentClip.clipId;
                    const thumbnail = await generateTimelineThumbnail(
                        id,
                         path,
                        frameIndices,
                        thumbnailWidth,
                        timelineHeight,
                        clipWidth,
                        {
                            canvas: thumnailCanvasRef.current,
                            mediaInfo: mediaInfoRef.current,
                            startFrame: clipLocalStart,
                            endFrame: clipLocalEnd,
                        }
                    );
                    if (cancelled) return;
                    if (!thumbnail) return;
                    thumnailCanvasRef.current = thumbnail as HTMLCanvasElement;
                    moveClipToEnd(currentClipId);
                } catch (e) {
                    if (cancelled) return;
                    thumnailCanvasRef.current = undefined;
                }

                if (!cancelled) {
                    clipRef.current?.getLayer()?.batchDraw?.();
                }
            })();
        }, 150);

        return () => {
            cancelled = true;
            clearTimeout(timeoutId);
        };
    }, [currentClip?.src, clipWidth, timelineDuration, timelineWidth, timelinePadding, timelineHeight, clipRef, mediaInfoRef]);
    
    const calculateFrameFromX = useCallback((xPosition: number) => {
        // Remove padding to get actual timeline position
        const timelineX = xPosition - timelinePadding;
        // Calculate the frame based on the position within the visible timeline
        const [startFrame, endFrame] = timelineDuration;
        const framePosition = (timelineX / timelineWidth) * (endFrame - startFrame) + startFrame;
        return Math.round(framePosition);
    }, [timelinePadding, timelineWidth, timelineDuration]);

    const handleDragMove = useCallback((e:Konva.KonvaEventObject<MouseEvent>) => {

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

		const halfStroke = isSelected ? 1.5 : 0;
		setClipPosition({x: e.target.x() - halfStroke, y: e.target.y() - halfStroke});
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

        // Determine target timeline for ghost as the nearest timeline by vertical proximity
        const targetTimelineId = (nearestTimelineForGhost || timelineId!);
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

        // Show ghost on the closest timeline

        //console.log(targetTimelineId, validatedLeft);

        setGhostTimelineId(targetTimelineId);
        setGhostInStage(true);
        setGhostX(Math.round(validatedLeft));
        setSnapGuideX(appliedSnap && snapStageX != null ? Math.round(snapStageX) : null);
        setGhostStartEndFrame(0, clipLen);
		
		
	}, [clipRef, clipWidth, timelineHeight, isSelected, timelinePadding, timelineDuration, currentEndFrame, currentStartFrame, timelineWidth, getClipsForTimeline, timelineId, currentClipId, setGhostTimelineId, setGhostInStage, setGhostX, setGhostStartEndFrame, setHoveredTimelineId]);

    const handleDragEnd = useCallback((_e: Konva.KonvaEventObject<MouseEvent>) => {
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
            const newTimeline = {
                timelineId: newTimelineId,
                timelineWidth: stageWidth,
                timelineY: (hoveredTimeline?.timelineY ?? 0) + 64,
                timelineHeight: 64,
            };
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
        setIsMouseOver(true);
        moveClipToEnd(currentClipId);
        e.target.getStage()!.container().style.cursor = 'pointer';
    }, [isSelected]);
    const handleMouseLeave = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        setIsMouseOver(false);
        e.target.getStage()!.container().style.cursor = 'default';
    }, [isSelected]);

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
                fill={type === 'audio' ? 'transparent' : '#E3E3E3'} 
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
            </Group>
    )
}

export default TimelineClip;
