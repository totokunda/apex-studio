
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Layer, Group, Rect, Line as KonvaLine, Ellipse as KonvaEllipse, RegularPolygon, Star as KonvaStar, Circle } from 'react-konva';
import { useViewportStore } from '@/lib/viewport';
import { useClipStore } from '@/lib/clip';
import { KonvaEventObject } from 'konva/lib/Node';
import { BASE_LONG_SIDE, DEFAULT_FPS } from '@/lib/settings'; 
 
import _ from 'lodash';
import VideoPreview from './clips/VideoPreview';
import AudioPreview from './clips/AudioPreview';
import ImagePreview from './clips/ImagePreview';
import ShapePreview from './clips/ShapePreview';
import TextPreview from './clips/TextPreview';
import DrawingPreview from './clips/DrawingPreview';
import MaskPreview from './clips/MaskPreview';
import { useControlsStore } from '@/lib/control';
import { AnyClipProps, PolygonClipProps, ShapeClipProps, TextClipProps, DrawingClipProps, DrawingLine } from '@/lib/types';
import { v4 as uuidv4 } from 'uuid';
import { SlSizeFullscreen } from 'react-icons/sl';
import FullscreenPreview from './FullscreenPreview';
import { getApplicatorsForClip } from '@/lib/applicator-utils';
import { useWebGLHaldClut } from './webgl-filters';
import { useDrawingStore } from '@/lib/drawing';
import { useMaskStore } from '@/lib/mask';
import { erasePolylineByEraser, transformEraserToLocal, mergeConnectedLines } from '@/lib/eraser-utils';

interface PreviewProps {

}

const Preview:React.FC<PreviewProps> = () => {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<any>(null);
  const layerRef = useRef<any>(null);
  const prevRectRef = useRef<{ w: number; h: number } | null>(null);
  const scale = useViewportStore((s) => s.scale);
  const position = useViewportStore((s) => s.position);
  const zoomAtScreenPoint = useViewportStore((s) => s.zoomAtScreenPoint);
  const panBy = useViewportStore((s) => s.panBy);
  const tool = useViewportStore((s) => s.tool);
  const shape = useViewportStore((s) => s.shape);
  const setViewportSize = useViewportStore((s) => s.setViewportSize);
  const setContentBounds = useViewportStore((s) => s.setContentBounds);
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const {clips, clipWithinFrame, timelines, addClip, addTimeline} = useClipStore();
  // Note: we use imperative store access in the recenter effect to avoid rerender loops
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const totalTimelineFrames = useControlsStore((s) => s.totalTimelineFrames);
  // Shape creation state
  const [isDrawingShape, setIsDrawingShape] = useState(false);
  const [isDrawingText, setIsDrawingText] = useState(false);
  const [textStart, setTextStart] = useState<{ x: number; y: number } | null>(null);
  const [textCurrent, setTextCurrent] = useState<{ x: number; y: number } | null>(null);
  const [shapeStart, setShapeStart] = useState<{ x: number; y: number } | null>(null);
  const [shapeCurrent, setShapeCurrent] = useState<{ x: number; y: number } | null>(null);
  const isFullscreen = useControlsStore((s) => s.isFullscreen);
  const setIsFullscreen = useControlsStore((s) => s.setIsFullscreen);
  const haldClutInstance = useWebGLHaldClut();
  const [clutsLoaded, setClutsLoaded] = useState(0);
  const {tool: drawingTool, color: drawingColor, opacity: drawingOpacity, smoothing: drawingSmoothing, getCurrentSize, setCurrentLineId} = useDrawingStore();
  const isDrawingRef = useRef(false);
  const eraserPointsRef = useRef<Array<{ x: number; y: number }>>([]);
  const lastEraserPosRef = useRef<{ x: number; y: number } | null>(null);
  const [cursorPreview, setCursorPreview] = useState<{ x: number; y: number } | null>(null);
  
  // Mask tool state
  const maskTool = useMaskStore((s) => s.tool);
  const maskShape = useMaskStore((s) => s.shape);
  const maskBrushSize = useMaskStore((s) => s.brushSize);
  const featherAmount = useMaskStore((s) => s.featherAmount);
  const touchLabel = useMaskStore((s) => s.touchLabel);
  const touchDrawMode = useMaskStore((s) => s.touchDrawMode);
  const [isDrawingLasso, setIsDrawingLasso] = useState(false);
  const [lassoPoints, setLassoPoints] = useState<number[]>([]);
  const [isDrawingMaskRect, setIsDrawingMaskRect] = useState(false);
  const [maskRectStart, setMaskRectStart] = useState<{ x: number; y: number } | null>(null);
  const [maskRectCurrent, setMaskRectCurrent] = useState<{ x: number; y: number } | null>(null);
  const [isDrawingMaskBrush, setIsDrawingMaskBrush] = useState(false);
  const [maskBrushPoints, setMaskBrushPoints] = useState<number[]>([]);
  const [maskCursorPreview, setMaskCursorPreview] = useState<{ x: number; y: number } | null>(null);
  const [isDrawingTouchLasso, setIsDrawingTouchLasso] = useState(false);
  const [touchLassoPoints, setTouchLassoPoints] = useState<number[]>([]);
  
  // Memoize applicator factory configuration
  const applicatorConfig = useMemo(() => ({
    haldClutInstance,
    // Add more shared resources here as needed
  }), [haldClutInstance]);
  
  // Get applicators for a specific clip using the factory
  const getClipApplicators = useCallback((clipId: string) => {
    return getApplicatorsForClip(clipId, applicatorConfig);
  }, [applicatorConfig]);
  
  // Preload all resources for applicator clips
  useEffect(() => {
    if (!haldClutInstance) return;
    
    // Preload filter CLUTs
    const filterClips = clips.filter(c => c.type === 'filter');
    
    if (filterClips.length === 0) {
      setClutsLoaded(0);
      return;
    }
    
    let loadedCount = 0;
    const loadPromises = filterClips.map(async (clip: any) => {
      const filterPath = clip.fullPath || clip.smallPath;
      if (filterPath) {
        try {
          await haldClutInstance.preloadClut(filterPath);
          loadedCount++;
          setClutsLoaded(loadedCount);
        } catch (e) {
          console.warn('Failed to preload CLUT:', filterPath, e);
        }
      }
    });
    
    // Add preloading for other applicator types here as they are added
    // e.g., mask textures, processor models, etc.
    
    Promise.all(loadPromises).catch(console.error);
  }, [clips, haldClutInstance]);
  

  const sortClips = useCallback((clips: AnyClipProps[]) => {
    // Sort clips by timeline index - earlier timelines render later (on top)
    const sortedClips = clips.slice().sort((a, b) => {
      const indexA = timelines.findIndex(t => t.timelineId === a.timelineId);
      const indexB = timelines.findIndex(t => t.timelineId === b.timelineId);
      
      // Earlier timeline index renders later (higher z-index)
      return indexB - indexA;
    });
    
    return sortedClips;
  }, [timelines])

  const filterClips = useCallback((clips: AnyClipProps[], audio:boolean = false) => {
    const filteredClips = clips.filter((clip) => {
      const timeline = timelines.find((t) => t.timelineId === clip.timelineId);
      
      if (audio) {
        if (timeline?.muted) {
          return false;
        }
      }
      if (timeline?.hidden) {
        return false;
      }
      return true;
    });

    return filteredClips;
  }, [timelines])
  
  // Find the topmost rendered clip at the current focus frame
  const getTopmostClipAtFrame = useCallback(() => {
    const visibleClips = sortClips(filterClips(clips)).filter((clip) => {
      const clipAtFrame = clipWithinFrame(clip, focusFrame);
      // Only consider clips that can have masks (video and image)
      return clipAtFrame && (clip.type === 'video' || clip.type === 'image');
    });
    
    // Return the first clip (topmost due to sortClips)
    return visibleClips.length > 0 ? visibleClips[0] : null;
  }, [clips, timelines, focusFrame, sortClips, filterClips, clipWithinFrame]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    
    // Throttle resize updates to improve performance
    let timeoutId: NodeJS.Timeout;
    const observer = new ResizeObserver((entries) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        const entry = entries[0];
        if (!entry) return;
        const { width, height } = entry.contentRect;
        setSize({ width, height });
        setViewportSize({ width, height });
      }, 16); // ~60fps throttle
    });
    
    observer.observe(el);
    return () => {
      clearTimeout(timeoutId);
      observer.disconnect();
    };
  }, [containerRef.current?.offsetWidth, containerRef.current?.offsetHeight]);

  // Force size update when exiting fullscreen
  useEffect(() => {
    if (!isFullscreen && containerRef.current) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        if (containerRef.current) {
          const el = containerRef.current;
          const rect = el.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) {
            setSize({ width: rect.width, height: rect.height });
            setViewportSize({ width: rect.width, height: rect.height });
          }
        }
      });
    }
  }, [isFullscreen, setViewportSize]);

  // Compute rect dimensions based on aspect ratio
  const { rectWidth, rectHeight } = useMemo(() => {
    const ratio = aspectRatio.width / aspectRatio.height;
    const baseShortSide = BASE_LONG_SIDE; // keep short side constant
    if (!Number.isFinite(ratio) || ratio <= 0) {
      return { rectWidth: 0, rectHeight: 0 };
    }
    // Use a fixed height for consistency; scale width by aspect ratio.
    // This keeps portrait sizes from becoming excessively tall.
    return { rectWidth: baseShortSide * ratio, rectHeight: baseShortSide };
  }, [aspectRatio.width, aspectRatio.height]);


  // Center the rect initially and whenever aspect ratio or viewport changes
  useEffect(() => {
    if (!stageRef.current || isFullscreen) return;
    const rectBounds = { x: 0, y: 0, width: rectWidth, height: rectHeight };
    setContentBounds(rectBounds);
    const rectWorld = { x: rectBounds.x + rectBounds.width / 2, y: rectBounds.y + rectBounds.height / 2 };
    useViewportStore.getState().centerOnWorldPoint(rectWorld, { width: size.width, height: size.height });
  }, [size.width, size.height, rectWidth, rectHeight, isFullscreen]);

  // Recenter all clips when rect size changes (but not on initial render)
  useEffect(() => {
    if (!rectWidth || !rectHeight || isFullscreen) return;
    const prev = prevRectRef.current;
    if (!prev || prev.w === 0 || prev.h === 0) {
      prevRectRef.current = { w: rectWidth, h: rectHeight };
      return; // skip initial mount
    }
    if (prev.w === rectWidth && prev.h === rectHeight) return; // no change

    const { clips: currentClips } = useClipStore.getState();
    currentClips.forEach((clip) => {
      const t = useClipStore.getState().getClipTransform(clip.clipId);
      const w = t?.width;
      const h = t?.height;
      if (!w || !h) return;
      const cx = Math.max(0, (rectWidth - w) / 2);
      const cy = Math.max(0, (rectHeight - h) / 2);
      const dx = (t?.x ?? 0) - cx;
      const dy = (t?.y ?? 0) - cy;
      if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
        useClipStore.getState().setClipTransform(clip.clipId, { x: cx, y: cy });
      }
    });

    prevRectRef.current = { w: rectWidth, h: rectHeight };
  }, [rectWidth, rectHeight, isFullscreen]);


  const handleWheel = useCallback((e: any) => {
    e.evt.preventDefault();
    const isZoom = e.evt.ctrlKey || e.evt.metaKey; // ctrl (win) or cmd (mac)
    const { offsetX, offsetY, deltaY, deltaX } = e.evt;
    if (isZoom) {
      const zoomIntensity = 0.0015; // sensitivity
      const nextScale = scale * (1 - deltaY * zoomIntensity);
      zoomAtScreenPoint(nextScale, { x: offsetX, y: offsetY });
    } else {
      // pan with wheel: shift to horizontal when shiftKey is pressed
      const dx = e.evt.shiftKey ? deltaY : deltaX;
      const dy = e.evt.shiftKey ? 0 : deltaY;
      panBy(dx, dy);
    }
  }, [scale, zoomAtScreenPoint, panBy]);

  const isPanning = useRef(false);
  const lastPointer = useRef<{ x: number; y: number } | null>(null);
  const currentDrawingClipId = useRef<string | null>(null);

  const onMouseDown = useCallback((e: any) => {
    if (tool === 'hand') {
      isPanning.current = true;
      lastPointer.current = { x: e.evt.offsetX, y: e.evt.offsetY };
      return;
    } 
    
    if (tool === 'mask' && maskTool === 'lasso') {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      setIsDrawingLasso(true);
      setLassoPoints([worldX, worldY]);
      return;
    }
    
    if (tool === 'mask' && maskTool === 'shape') {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      setIsDrawingMaskRect(true);
      setMaskRectStart({ x: worldX, y: worldY });
      setMaskRectCurrent({ x: worldX, y: worldY });
      return;
    }
    
    if (tool === 'mask' && maskTool === 'draw') {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      setIsDrawingMaskBrush(true);
      setMaskBrushPoints([worldX, worldY]);
      return;
    }
    
    if (tool === 'mask' && maskTool === 'touch') {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      // Handle draw mode: start lasso drawing
      if (touchDrawMode === 'draw') {
        setIsDrawingTouchLasso(true);
        setTouchLassoPoints([worldX, worldY]);
        return;
      }
      
      // Handle point mode: add touch points
      // Find the topmost clip at the current frame
      const targetClip = getTopmostClipAtFrame();
      
      if (targetClip) {
        const { updateClip } = useClipStore.getState();
        
        // Determine mask duration based on clip type
        let maskStartFrame = focusFrame;
        let maskEndFrame = focusFrame;
        
        if (targetClip.type === 'image') {
          maskStartFrame = targetClip.startFrame ?? focusFrame;
          maskEndFrame = targetClip.endFrame ?? focusFrame;
        } else if (targetClip.type === 'video') {
          maskStartFrame = focusFrame;
          maskEndFrame = focusFrame;
        }
        
        // Get existing masks
        const currentMasks = (targetClip as any).masks || [];
        
        // Check if there's already a touch mask at this frame for this clip
        let existingTouchMask = currentMasks.find((mask: any) => {
          const maskStart = mask.startFrame ?? 0;
          const maskEnd = mask.endFrame ?? 0;
          return mask.tool === 'touch' && focusFrame >= maskStart && focusFrame <= maskEnd;
        });
        
        if (existingTouchMask) {
          // Add point to existing mask
          const keyframes = existingTouchMask.keyframes instanceof Map 
            ? existingTouchMask.keyframes 
            : (existingTouchMask.keyframes as Record<number, any>);
          
          const currentFrameData = keyframes instanceof Map 
            ? keyframes.get(focusFrame) 
            : keyframes[focusFrame];
          
          const touchPoints = currentFrameData?.touchPoints || [];
          const lassoStrokes = currentFrameData?.lassoStrokes || [];
          
          // Add new point with label
          const updatedTouchPoints = [...touchPoints, { x: worldX, y: worldY, label: touchLabel }];
          
          // Update keyframes
          if (keyframes instanceof Map) {
            keyframes.set(focusFrame, { touchPoints: updatedTouchPoints, lassoStrokes });
            existingTouchMask.keyframes = keyframes;
          } else {
            existingTouchMask.keyframes = {
              ...keyframes,
              [focusFrame]: { touchPoints: updatedTouchPoints, lassoStrokes },
            };
          }
          
          existingTouchMask.lastModified = Date.now();
          
          updateClip(targetClip.clipId, {
            masks: currentMasks,
          });
        } else {
          // Create new touch mask
          const newMask = {
            id: uuidv4(),
            clipId: targetClip.clipId,
            tool: 'touch' as const,
            startFrame: maskStartFrame,
            endFrame: maskEndFrame,
            featherAmount: featherAmount,
            keyframes: {
              [focusFrame]: {
                touchPoints: [{ x: worldX, y: worldY, label: touchLabel }],
                lassoStrokes: [],
              },
            },
            isTracked: false,
            createdAt: Date.now(),
            lastModified: Date.now(),
            opacity: 100,
          };
          
          currentMasks.push(newMask);
          
          updateClip(targetClip.clipId, {
            masks: currentMasks,
          });
        }
      }
      
      return;
    }
    
    if (tool === 'shape') {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      setIsDrawingShape(true);
      setShapeStart({ x: worldX, y: worldY });
      setShapeCurrent({ x: worldX, y: worldY });
      return;
    } 

    if (tool === 'text') {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      setIsDrawingText(true);
      setTextStart({ x: worldX, y: worldY });
      setTextCurrent({ x: worldX, y: worldY });
      return;
    }

    if (tool === 'draw') {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      isDrawingRef.current = true;
      
      // Check if there's already a drawing clip at the current frame
      const existingDrawingClip = clips.find(
        (c) => c.type === 'draw' && clipWithinFrame(c, focusFrame)
      ) as DrawingClipProps | undefined;
      
      // Handle eraser tool separately - no line creation, just collect points
      if (drawingTool === 'eraser') {
        if (!existingDrawingClip) {
          // Eraser needs existing clip to work with
          isDrawingRef.current = false;
          currentDrawingClipId.current = null;
          return;
        }
        
        currentDrawingClipId.current = existingDrawingClip.clipId;
        eraserPointsRef.current = [{ x: worldX, y: worldY }];
        lastEraserPosRef.current = { x: worldX, y: worldY };
        return;
      }
      
      const lineId = uuidv4();
      setCurrentLineId(lineId);
      
      if (existingDrawingClip) {
        currentDrawingClipId.current = existingDrawingClip.clipId;
        
        // Add new line to existing clip
        const newLine: DrawingLine = {
          lineId,
          tool: drawingTool,
          points: [worldX, worldY],
          stroke: drawingTool === 'highlighter' ? '#FFFF00' : drawingColor,
          strokeWidth: getCurrentSize(),
          opacity: drawingTool === 'highlighter' ? 50 : drawingOpacity,
          smoothing: drawingSmoothing,
          transform: {
            x: 0,
            y: 0,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
            opacity: 100,
          },
        };
        
        // Append new line to the end so it renders on top of older lines
        const updatedLines = [...(existingDrawingClip.lines || []), newLine];
        useClipStore.getState().updateClip(existingDrawingClip.clipId, { lines: updatedLines });
      } else {
        
        // Create new drawing clip
        let drawingTimeline = timelines.find((t) => t.type === 'draw');
        
        if (!drawingTimeline) {
          drawingTimeline = {
            timelineId: uuidv4(),
            type: 'draw' as const,
            timelineY: 0,
            timelineHeight: 40,
            timelineWidth: 0,
            timelinePadding: 24,
            muted: false,
            hidden: false,
          };
          addTimeline(drawingTimeline, -1);
        }
        
        const clipDuration = 3 * DEFAULT_FPS;
        let proposedEndFrame = Math.min(focusFrame + clipDuration, totalTimelineFrames - 1);
        
        // Check for existing clips on the drawing timeline that would collide
        const clipsOnDrawingTimeline = clips.filter((c) => c.timelineId === drawingTimeline!.timelineId);
        for (const existingClip of clipsOnDrawingTimeline) {
          const existingStart = existingClip.startFrame ?? 0;
          // If an existing clip starts after our current frame but before our proposed end frame
          if (existingStart > focusFrame && existingStart < proposedEndFrame) {
            // Adjust our end frame to stop at the existing clip's start
            proposedEndFrame = existingStart;
          }
        }
        
        const newClipId = uuidv4();
        const newClip: DrawingClipProps = {
          src: null,
          clipId: newClipId,
          type: 'draw' as const,
          timelineId: drawingTimeline.timelineId,
          startFrame: focusFrame,
          endFrame: proposedEndFrame,
          framesToGiveEnd: -Infinity,
          framesToGiveStart: Infinity,
          lines: [
            {
              lineId,
              tool: drawingTool,
              points: [worldX, worldY],
              stroke: drawingTool === 'highlighter' ? '#FFFF00' : drawingColor,
              strokeWidth: getCurrentSize(),
              opacity: drawingTool === 'highlighter' ? 50 : drawingOpacity,
              smoothing: drawingSmoothing,
              transform: {
                x: 0,
                y: 0,
                scaleX: 1,
                scaleY: 1,
                rotation: 0,
                opacity: 100,
              },
            },
          ],
        };
        
        currentDrawingClipId.current = newClipId;
        addClip(newClip);
      }
      
      return;
    }

  }, [tool, position, scale, clips, clipWithinFrame, focusFrame, drawingTool, drawingColor, drawingOpacity, getCurrentSize, setCurrentLineId, timelines, addTimeline, addClip, totalTimelineFrames, maskTool, touchLabel, touchDrawMode, featherAmount, getTopmostClipAtFrame]);

  const onMouseMove = useCallback((e: any) => {
    // Handle lasso drawing
    if (tool === 'mask' && maskTool === 'lasso' && isDrawingLasso) {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      // Add point to lasso path
      setLassoPoints((prev) => [...prev, worldX, worldY]);
      return;
    }
    
    // Handle shape mask drawing
    if (tool === 'mask' && maskTool === 'shape' && isDrawingMaskRect && maskRectStart) {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      setMaskRectCurrent({ x: worldX, y: worldY });
      return;
    }
    
    // Handle brush mask drawing
    if (tool === 'mask' && maskTool === 'draw' && isDrawingMaskBrush) {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      // Add point to brush path only if it's far enough from the last point
      setMaskBrushPoints((prev) => {
        if (prev.length < 2) return [...prev, worldX, worldY];
        
        const lastX = prev[prev.length - 2];
        const lastY = prev[prev.length - 1];
        const dx = worldX - lastX;
        const dy = worldY - lastY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Only add point if it's at least 1 pixel away from the last point
        if (distance >= 1) {
          return [...prev, worldX, worldY];
        }
        return prev;
      });
      return;
    }
    
    // Handle touch lasso drawing
    if (tool === 'mask' && maskTool === 'touch' && touchDrawMode === 'draw' && isDrawingTouchLasso) {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      // Add point to touch lasso path
      setTouchLassoPoints((prev) => [...prev, worldX, worldY]);
      return;
    }
    
    // Update cursor preview for mask draw tool
    if (tool === 'mask' && maskTool === 'draw') {
      const stage = e.target.getStage();
      if (stage) {
        const pointerPos = stage.getPointerPosition();
        if (pointerPos) {
          const worldX = (pointerPos.x - position.x) / scale;
          const worldY = (pointerPos.y - position.y) / scale;
          setMaskCursorPreview({ x: worldX, y: worldY });
        }
      }
    } else {
      setMaskCursorPreview(null);
    }
    
    // Update cursor preview for draw tool
    if (tool === 'draw') {
      const stage = e.target.getStage();
      if (stage) {
        const pointerPos = stage.getPointerPosition();
        if (pointerPos) {
          const worldX = (pointerPos.x - position.x) / scale;
          const worldY = (pointerPos.y - position.y) / scale;
          setCursorPreview({ x: worldX, y: worldY });
        }
      }
    } else {
      setCursorPreview(null);
    }
    
    if (tool === 'hand' && isPanning.current) {
      const current = { x: e.evt.offsetX, y: e.evt.offsetY };
      const last = lastPointer.current || current;
      const dx = current.x - last.x;
      const dy = current.y - last.y;
      panBy(-dx, -dy);
      lastPointer.current = current;
      return;
    }
    
    if (tool === 'shape' && isDrawingShape && shapeStart) {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      setShapeCurrent({ x: worldX, y: worldY });
    } 
    if (tool === 'text' && isDrawingText && textStart) {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;

      setTextCurrent({ x: worldX, y: worldY });
    }

    if (tool === 'draw' && isDrawingRef.current) {
      const stage = e.target.getStage();
      if (!stage) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      // Convert screen coordinates to world coordinates
      const worldX = (pointerPos.x - position.x) / scale;
      const worldY = (pointerPos.y - position.y) / scale;
      
      // Find the drawing clip that we're currently drawing on
      const drawingClipId = currentDrawingClipId.current;
      if (!drawingClipId) return;
      
      const drawingClip = useClipStore.getState().getClipById(drawingClipId) as DrawingClipProps | undefined;
      if (!drawingClip || !drawingClip.lines) return;
      
      // Handle eraser tool - apply real-time erasing
      if (drawingTool === 'eraser') {
        // Interpolate points between last position and current position to avoid gaps
        const lastPos = lastEraserPosRef.current;
        if (lastPos) {
          const dx = worldX - lastPos.x;
          const dy = worldY - lastPos.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          
          // Add interpolated points if distance is significant (more than 2 pixels)
          if (dist > 2) {
            const steps = Math.ceil(dist / 2); // Sample every ~2 pixels
            for (let i = 1; i <= steps; i++) {
              const t = i / steps;
              eraserPointsRef.current.push({
                x: lastPos.x + dx * t,
                y: lastPos.y + dy * t,
              });
            }
          } else {
            eraserPointsRef.current.push({ x: worldX, y: worldY });
          }
        } else {
          eraserPointsRef.current.push({ x: worldX, y: worldY });
        }
        
        lastEraserPosRef.current = { x: worldX, y: worldY };
        
        const eraserBaseRadius = getCurrentSize() / 2;
        const worldEraserPts = eraserPointsRef.current;
        
        // Process all brush/highlighter lines
        const updatedLines: DrawingLine[] = [];
        
        for (const line of drawingClip.lines) {
          // Only erase brush and highlighter lines
          if (line.tool !== 'brush' && line.tool !== 'highlighter') {
            updatedLines.push(line);
            continue;
          }
          
          // Add the line's stroke width to the eraser radius for better coverage
          const effectiveEraserRadius = eraserBaseRadius + (line.strokeWidth / 2);
          
          // Transform eraser points from world space to line's local space
          const { localPts, localRadius } = transformEraserToLocal(
            worldEraserPts,
            line.transform,
            effectiveEraserRadius
          );
          
          // Apply erasing algorithm in local space
          const splitPolylines = erasePolylineByEraser(line.points, localPts, localRadius);
          
          // Create new lines for each split segment (points are already in local space)
          for (const polyline of splitPolylines) {
            // Filter out segments with less than 2 points (less than 4 values in flat array)
            if (polyline.length >= 4) {
              updatedLines.push({
                ...line,
                lineId: uuidv4(), // New ID for split segment
                points: polyline,
              });
            }
          }
        }
        
        // Apply lightweight merging to reduce fragmentation (single pass)
        const mergedLines = mergeConnectedLines(updatedLines, false);
        
        useClipStore.getState().updateClip(drawingClip.clipId, { lines: mergedLines });
        return;
      }
      
      // Find the current line being drawn (brush/highlighter)
      const currentLine = drawingClip.lines[drawingClip.lines.length - 1];
      if (!currentLine) return;
      
      // Add point to the current line
      const updatedLines = [...drawingClip.lines];
      updatedLines[updatedLines.length - 1] = {
        ...currentLine,
        points: [...currentLine.points, worldX, worldY],
      };
      
      useClipStore.getState().updateClip(drawingClip.clipId, { lines: updatedLines });
    }
  }, [panBy, tool, isDrawingShape, shapeStart, position, scale, isDrawingText, textStart, drawingTool, getCurrentSize, cursorPreview, maskTool, isDrawingLasso, isDrawingMaskRect, maskRectStart, isDrawingMaskBrush, touchDrawMode, isDrawingTouchLasso]);

  const onMouseUp = useCallback((e: any) => {
    if (tool === 'hand') {
      isPanning.current = false;
      lastPointer.current = null;
      return;
    }
    
    if (tool === 'mask' && maskTool === 'lasso' && isDrawingLasso && lassoPoints.length >= 6) {
      // Need at least 3 points (6 values) to create a valid lasso
      setIsDrawingLasso(false);
      
      // Find the topmost clip at the current frame
      const targetClip = getTopmostClipAtFrame();
      
      if (targetClip) {
        const { updateClip } = useClipStore.getState();
        
        // Determine mask duration based on clip type
        let maskStartFrame = focusFrame;
        let maskEndFrame = focusFrame;
        
        if (targetClip.type === 'image') {
          // For images, mask covers the entire clip duration
          maskStartFrame = targetClip.startFrame ?? focusFrame;
          maskEndFrame = targetClip.endFrame ?? focusFrame;
        } else if (targetClip.type === 'video') {
          // For videos, mask is only 1 frame long
          maskStartFrame = focusFrame;
          maskEndFrame = focusFrame;
        }
        
        // Create the mask
        const newMask = {
          id: uuidv4(),
          clipId: targetClip.clipId,
          tool: 'lasso' as const,
          startFrame: maskStartFrame,
          endFrame: maskEndFrame,
          featherAmount: featherAmount,
          keyframes: {
            [focusFrame]: {
              lassoPoints: [...lassoPoints],
            },
          },
          isTracked: false,
          createdAt: Date.now(),
          lastModified: Date.now(),
          opacity: 100,
        };
        
        // Get existing masks and check for containment
        const currentMasks = (targetClip as any).masks || [];
        let masksToKeep = [...currentMasks];
        
        // Helper function to calculate polygon area (shoelace formula)
        const calculateArea = (points: number[]) => {
          let area = 0;
          const n = points.length / 2;
          for (let i = 0; i < n; i++) {
            const j = (i + 1) % n;
            const xi = points[i * 2];
            const yi = points[i * 2 + 1];
            const xj = points[j * 2];
            const yj = points[j * 2 + 1];
            area += xi * yj - xj * yi;
          }
          return Math.abs(area / 2);
        };
        
        // Helper function to check if a point is inside a polygon
        const isPointInPolygon = (x: number, y: number, polygonPoints: number[]) => {
          let inside = false;
          const n = polygonPoints.length / 2;
          for (let i = 0, j = n - 1; i < n; j = i++) {
            const xi = polygonPoints[i * 2];
            const yi = polygonPoints[i * 2 + 1];
            const xj = polygonPoints[j * 2];
            const yj = polygonPoints[j * 2 + 1];
            
            const intersect = ((yi > y) !== (yj > y)) && 
                            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
          }
          return inside;
        };
        
        // Helper function to check if one polygon is completely inside another
        const isPolygonInsidePolygon = (innerPoints: number[], outerPoints: number[]) => {
          // Check if all points of inner polygon are inside outer polygon
          const n = innerPoints.length / 2;
          for (let i = 0; i < n; i++) {
            const x = innerPoints[i * 2];
            const y = innerPoints[i * 2 + 1];
            if (!isPointInPolygon(x, y, outerPoints)) {
              return false;
            }
          }
          return true;
        };
        
        // Helper function to check if two line segments intersect
        const doSegmentsIntersect = (
          p1x: number, p1y: number, p2x: number, p2y: number,
          p3x: number, p3y: number, p4x: number, p4y: number
        ) => {
          const denom = (p4y - p3y) * (p2x - p1x) - (p4x - p3x) * (p2y - p1y);
          if (Math.abs(denom) < 1e-10) return false; // Parallel or coincident
          
          const ua = ((p4x - p3x) * (p1y - p3y) - (p4y - p3y) * (p1x - p3x)) / denom;
          const ub = ((p2x - p1x) * (p1y - p3y) - (p2y - p1y) * (p1x - p3x)) / denom;
          
          return ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1;
        };
        
        // Helper function to check if two polygons intersect
        const doPolygonsIntersect = (points1: number[], points2: number[]) => {
          const n1 = points1.length / 2;
          const n2 = points2.length / 2;
          
          // Check if any edges intersect
          for (let i = 0; i < n1; i++) {
            const j = (i + 1) % n1;
            const p1x = points1[i * 2];
            const p1y = points1[i * 2 + 1];
            const p2x = points1[j * 2];
            const p2y = points1[j * 2 + 1];
            
            for (let k = 0; k < n2; k++) {
              const l = (k + 1) % n2;
              const p3x = points2[k * 2];
              const p3y = points2[k * 2 + 1];
              const p4x = points2[l * 2];
              const p4y = points2[l * 2 + 1];
              
              if (doSegmentsIntersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)) {
                return true;
              }
            }
          }
          
          // Check if one polygon is inside the other
          if (isPointInPolygon(points1[0], points1[1], points2)) return true;
          if (isPointInPolygon(points2[0], points2[1], points1)) return true;
          
          return false;
        };
        
        // Helper function to compute convex hull (Graham scan algorithm)
        const computeConvexHull = (points: number[]) => {
          const pts: Array<{x: number, y: number}> = [];
          for (let i = 0; i < points.length; i += 2) {
            pts.push({ x: points[i], y: points[i + 1] });
          }
          
          if (pts.length < 3) return points;
          
          // Find the point with lowest y (and leftmost if tie)
          let minIdx = 0;
          for (let i = 1; i < pts.length; i++) {
            if (pts[i].y < pts[minIdx].y || (pts[i].y === pts[minIdx].y && pts[i].x < pts[minIdx].x)) {
              minIdx = i;
            }
          }
          
          // Swap to put lowest point first
          [pts[0], pts[minIdx]] = [pts[minIdx], pts[0]];
          const pivot = pts[0];
          
          // Sort by polar angle
          const sorted = [pivot, ...pts.slice(1).sort((a, b) => {
            const angleA = Math.atan2(a.y - pivot.y, a.x - pivot.x);
            const angleB = Math.atan2(b.y - pivot.y, b.x - pivot.x);
            if (Math.abs(angleA - angleB) < 1e-10) {
              const distA = (a.x - pivot.x) ** 2 + (a.y - pivot.y) ** 2;
              const distB = (b.x - pivot.x) ** 2 + (b.y - pivot.y) ** 2;
              return distA - distB;
            }
            return angleA - angleB;
          })];
          
          // Build hull
          const hull = [sorted[0], sorted[1]];
          
          const ccw = (p1: {x: number, y: number}, p2: {x: number, y: number}, p3: {x: number, y: number}) => {
            return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
          };
          
          for (let i = 2; i < sorted.length; i++) {
            while (hull.length > 1 && ccw(hull[hull.length - 2], hull[hull.length - 1], sorted[i]) <= 0) {
              hull.pop();
            }
            hull.push(sorted[i]);
          }
          
          // Convert back to flat array
          const result: number[] = [];
          for (const pt of hull) {
            result.push(pt.x, pt.y);
          }
          return result;
        };
        
        // Helper function to merge two polygons (using convex hull)
        const mergePolygons = (points1: number[], points2: number[]) => {
          // Combine all points
          const allPoints = [...points1, ...points2];
          // Return convex hull of all points
          return computeConvexHull(allPoints);
        };
        
        const newMaskArea = calculateArea(lassoPoints);
        
        // Track masks that intersect with the new mask for merging
        const intersectingMasks: any[] = [];
        let mergedPoints = [...lassoPoints];
        
        // Check each existing mask at this frame for containment and intersection
        masksToKeep = currentMasks.filter((existingMask: any) => {
          // Only check masks at the same frame
          const maskStart = existingMask.startFrame ?? 0;
          const maskEnd = existingMask.endFrame ?? 0;
          
          if (focusFrame < maskStart || focusFrame > maskEnd) {
            return true; // Keep masks not at this frame
          }
          
          if (existingMask.tool !== 'lasso') {
            return true; // Keep non-lasso masks for now
          }
          
          // Get the lasso points for this mask at this frame
          const keyframes = existingMask.keyframes instanceof Map 
            ? existingMask.keyframes 
            : (existingMask.keyframes as Record<number, any>);
          
          const keyframeNumbers = keyframes instanceof Map
            ? Array.from(keyframes.keys()).sort((a, b) => a - b)
            : Object.keys(keyframes).map(Number).sort((a, b) => a - b);
          
          const activeKeyframe = keyframeNumbers.filter((k: number) => k <= focusFrame).pop();
          
          if (activeKeyframe === undefined) {
            return true; // Keep if no keyframe data
          }
          
          const maskData = keyframes instanceof Map 
            ? keyframes.get(activeKeyframe) 
            : keyframes[activeKeyframe];
          
          if (!maskData?.lassoPoints) {
            return true; // Keep if no lasso points
          }
          
          const existingPoints = maskData.lassoPoints;
          const existingArea = calculateArea(existingPoints);
          
          // Check containment
          const newInsideExisting = isPolygonInsidePolygon(mergedPoints, existingPoints);
          const existingInsideNew = isPolygonInsidePolygon(existingPoints, mergedPoints);
          
          if (newInsideExisting || existingInsideNew) {
            // One mask is inside the other - keep the larger one
            if (existingArea > newMaskArea) {
              return true; // Keep existing, discard new
            } else {
              return false; // Discard existing, keep new
            }
          }
          
          // Check intersection
          if (doPolygonsIntersect(mergedPoints, existingPoints)) {
            // Masks intersect - merge them
            intersectingMasks.push(existingMask);
            mergedPoints = mergePolygons(mergedPoints, existingPoints);
            return false; // Remove this mask, it will be merged into new mask
          }
          
          return true; // Keep if no containment or intersection
        });
        
        // Update the new mask with merged points if any masks were merged
        if (intersectingMasks.length > 0) {
          newMask.keyframes[focusFrame].lassoPoints = mergedPoints;
        }
        
        // Add the new mask (which may now be a merged mask)
        masksToKeep.push(newMask);
        
        updateClip(targetClip.clipId, {
          masks: masksToKeep,
        });
      }
      
      // Clear lasso points immediately since mask rendering will handle display
      setLassoPoints([]);
      
      return;
    }
    
    // If lasso was being drawn but not enough points, just clear
    if (tool === 'mask' && maskTool === 'lasso' && isDrawingLasso) {
      setIsDrawingLasso(false);
      setLassoPoints([]);
      return;
    }
    
    // Handle touch lasso drawing finish
    if (tool === 'mask' && maskTool === 'touch' && touchDrawMode === 'draw' && isDrawingTouchLasso) {
      setIsDrawingTouchLasso(false);
      
      // Need at least 3 points (6 values) for a valid lasso
      if (touchLassoPoints.length < 6) {
        setTouchLassoPoints([]);
        return;
      }
      
      // Find the topmost clip at the current frame
      const targetClip = getTopmostClipAtFrame();
      
      if (targetClip) {
        const { updateClip } = useClipStore.getState();
        
        // Determine mask duration based on clip type
        let maskStartFrame = focusFrame;
        let maskEndFrame = focusFrame;
        
        if (targetClip.type === 'image') {
          maskStartFrame = targetClip.startFrame ?? focusFrame;
          maskEndFrame = targetClip.endFrame ?? focusFrame;
        } else if (targetClip.type === 'video') {
          maskStartFrame = focusFrame;
          maskEndFrame = focusFrame;
        }
        
        // Get existing masks
        const currentMasks = (targetClip as any).masks || [];
        
        // Check if there's already a touch mask at this frame for this clip
        let existingTouchMask = currentMasks.find((mask: any) => {
          const maskStart = mask.startFrame ?? 0;
          const maskEnd = mask.endFrame ?? 0;
          return mask.tool === 'touch' && focusFrame >= maskStart && focusFrame <= maskEnd;
        });
        
        if (existingTouchMask) {
          // Add lasso stroke to existing mask
          const keyframes = existingTouchMask.keyframes instanceof Map 
            ? existingTouchMask.keyframes 
            : (existingTouchMask.keyframes as Record<number, any>);
          
          const currentFrameData = keyframes instanceof Map 
            ? keyframes.get(focusFrame) 
            : keyframes[focusFrame];
          
          const touchPoints = currentFrameData?.touchPoints || [];
          const lassoStrokes = currentFrameData?.lassoStrokes || [];
          
          // Add new lasso stroke
          const updatedLassoStrokes = [...lassoStrokes, [...touchLassoPoints]];
          
          // Update keyframes
          if (keyframes instanceof Map) {
            keyframes.set(focusFrame, { touchPoints, lassoStrokes: updatedLassoStrokes });
            existingTouchMask.keyframes = keyframes;
          } else {
            existingTouchMask.keyframes = {
              ...keyframes,
              [focusFrame]: { touchPoints, lassoStrokes: updatedLassoStrokes },
            };
          }
          
          existingTouchMask.lastModified = Date.now();
          
          updateClip(targetClip.clipId, {
            masks: currentMasks,
          });
        } else {
          // Create new touch mask with lasso stroke
          const newMask = {
            id: uuidv4(),
            clipId: targetClip.clipId,
            tool: 'touch' as const,
            startFrame: maskStartFrame,
            endFrame: maskEndFrame,
            featherAmount: featherAmount,
            keyframes: {
              [focusFrame]: {
                touchPoints: [],
                lassoStrokes: [[...touchLassoPoints]],
              },
            },
            isTracked: false,
            createdAt: Date.now(),
            lastModified: Date.now(),
            opacity: 100,
          };
          
          currentMasks.push(newMask);
          
          updateClip(targetClip.clipId, {
            masks: currentMasks,
          });
        }
      }
      
      setTouchLassoPoints([]);
      return;
    }
    
    // Handle shape mask creation
    if (tool === 'mask' && maskTool === 'shape' && isDrawingMaskRect && maskRectStart && maskRectCurrent) {
      setIsDrawingMaskRect(false);
      
      const width = Math.abs(maskRectCurrent.x - maskRectStart.x);
      const height = Math.abs(maskRectCurrent.y - maskRectStart.y);
      
      // Only create mask if shape has meaningful size
      if (width > 5 && height > 5) {
        const targetClip = getTopmostClipAtFrame();
        
        if (targetClip) {
          const { updateClip } = useClipStore.getState();
          
          // Determine mask duration based on clip type
          let maskStartFrame = focusFrame;
          let maskEndFrame = focusFrame;
          
          if (targetClip.type === 'image') {
            maskStartFrame = targetClip.startFrame ?? focusFrame;
            maskEndFrame = targetClip.endFrame ?? focusFrame;
          } else if (targetClip.type === 'video') {
            maskStartFrame = focusFrame;
            maskEndFrame = focusFrame;
          }
          
          const x = Math.min(maskRectStart.x, maskRectCurrent.x);
          const y = Math.min(maskRectStart.y, maskRectCurrent.y);
          
          // Create the mask
          const newMask = {
            id: uuidv4(),
            clipId: targetClip.clipId,
            tool: 'shape' as const,
            startFrame: maskStartFrame,
            endFrame: maskEndFrame,
            featherAmount: featherAmount,
            keyframes: {
              [focusFrame]: {
                shapeBounds: {
                  x,
                  y,
                  width,
                  height,
                  rotation: 0,
                  shapeType: maskShape,
                },
              },
            },
            isTracked: false,
            createdAt: Date.now(),
            lastModified: Date.now(),
            opacity: 100,
          };
          
          // Get existing masks and check for containment
          const currentMasks = (targetClip as any).masks || [];
          
          // Helper function to check if one rectangle is inside another (axis-aligned)
          const isRectInsideRect = (
            inner: { x: number; y: number; width: number; height: number },
            outer: { x: number; y: number; width: number; height: number }
          ) => {
            // Check if all corners of inner rectangle are inside outer rectangle
            const innerRight = inner.x + inner.width;
            const innerBottom = inner.y + inner.height;
            const outerRight = outer.x + outer.width;
            const outerBottom = outer.y + outer.height;
            
            return (
              inner.x >= outer.x &&
              inner.y >= outer.y &&
              innerRight <= outerRight &&
              innerBottom <= outerBottom
            );
          };
          
          const newMaskArea = width * height;
          const newRectBounds = { x, y, width, height };
          
          // Filter out masks that are consumed by the new one
          const masksToKeep = currentMasks.filter((existingMask: any) => {
            // Only check masks at the same frame
            const maskStart = existingMask.startFrame ?? 0;
            const maskEnd = existingMask.endFrame ?? 0;
            
            if (focusFrame < maskStart || focusFrame > maskEnd) {
              return true; // Keep masks not at this frame
            }
            
            if (existingMask.tool !== 'shape' && existingMask.tool !== 'rectangle') {
              return true; // Keep non-shape masks
            }
            
            // Get the rectangle bounds for this mask at this frame
            const keyframes = existingMask.keyframes instanceof Map 
              ? existingMask.keyframes 
              : (existingMask.keyframes as Record<number, any>);
            
            const keyframeNumbers = keyframes instanceof Map
              ? Array.from(keyframes.keys()).sort((a, b) => a - b)
              : Object.keys(keyframes).map(Number).sort((a, b) => a - b);
            
            const activeKeyframe = keyframeNumbers.filter((k: number) => k <= focusFrame).pop();
            
            if (activeKeyframe === undefined) {
              return true; // Keep if no keyframe data
            }
            
            const maskData = keyframes instanceof Map 
              ? keyframes.get(activeKeyframe) 
              : keyframes[activeKeyframe];
            
            const existingBounds = maskData?.shapeBounds || maskData?.rectangleBounds;
            if (!existingBounds) {
              return true; // Keep if no bounds
            }
            
            const existingRectBounds = existingBounds;
            const existingArea = existingRectBounds.width * existingRectBounds.height;
            
            // Check containment
            const newInsideExisting = isRectInsideRect(newRectBounds, existingRectBounds);
            const existingInsideNew = isRectInsideRect(existingRectBounds, newRectBounds);
            
            if (newInsideExisting || existingInsideNew) {
              // One mask is inside the other - keep the larger one
              return existingArea > newMaskArea;
            }
            
            return true; // Keep if no containment
          });
          
          // Only add the new mask if it wasn't consumed by a larger mask
          const shouldAddNewMask = !currentMasks.some((existingMask: any) => {
            const maskStart = existingMask.startFrame ?? 0;
            const maskEnd = existingMask.endFrame ?? 0;
            
            if (focusFrame < maskStart || focusFrame > maskEnd || (existingMask.tool !== 'shape' && existingMask.tool !== 'rectangle')) {
              return false;
            }
            
            const keyframes = existingMask.keyframes instanceof Map 
              ? existingMask.keyframes 
              : (existingMask.keyframes as Record<number, any>);
            
            const keyframeNumbers = keyframes instanceof Map
              ? Array.from(keyframes.keys()).sort((a, b) => a - b)
              : Object.keys(keyframes).map(Number).sort((a, b) => a - b);
            
            const activeKeyframe = keyframeNumbers.filter((k: number) => k <= focusFrame).pop();
            
            if (activeKeyframe === undefined) return false;
            
            const maskData = keyframes instanceof Map 
              ? keyframes.get(activeKeyframe) 
              : keyframes[activeKeyframe];
            
            const existingBounds = maskData?.shapeBounds || maskData?.rectangleBounds;
            if (!existingBounds) return false;
            
            const existingRectBounds = existingBounds;
            const existingArea = existingRectBounds.width * existingRectBounds.height;
            
            const newInsideExisting = isRectInsideRect(newRectBounds, existingRectBounds);
            
            // New mask is consumed by existing larger mask
            return newInsideExisting && existingArea > newMaskArea;
          });
          
          if (shouldAddNewMask) {
            masksToKeep.push(newMask);
          }
          
          updateClip(targetClip.clipId, {
            masks: masksToKeep,
          });
        }
      }
      
      // Reset rectangle drawing state
      setMaskRectStart(null);
      setMaskRectCurrent(null);
      return;
    }
    
    // Handle brush mask creation
    if (tool === 'mask' && maskTool === 'draw' && isDrawingMaskBrush && maskBrushPoints.length >= 4) {
      setIsDrawingMaskBrush(false);
      
      // Need at least 2 points (4 values) to create a valid brush stroke
      const targetClip = getTopmostClipAtFrame();
      
      if (targetClip) {
        const { updateClip } = useClipStore.getState();
        
        // Determine mask duration based on clip type
        let maskStartFrame = focusFrame;
        let maskEndFrame = focusFrame;
        
        if (targetClip.type === 'image') {
          // For images, mask covers the entire clip duration
          maskStartFrame = targetClip.startFrame ?? focusFrame;
          maskEndFrame = targetClip.endFrame ?? focusFrame;
        } else if (targetClip.type === 'video') {
          // For videos, mask is only 1 frame long
          maskStartFrame = focusFrame;
          maskEndFrame = focusFrame;
        }
        
        // Create the mask
        const newMask = {
          id: uuidv4(),
          clipId: targetClip.clipId,
          tool: 'draw' as const,
          startFrame: maskStartFrame,
          endFrame: maskEndFrame,
          featherAmount: featherAmount,
          brushSize: maskBrushSize,
          keyframes: {
            [focusFrame]: {
              drawStrokes: [{
                points: [...maskBrushPoints],
                strokeWidth: maskBrushSize,
              }],
            },
          },
          isTracked: false,
          createdAt: Date.now(),
          lastModified: Date.now(),
          opacity: 100,
        };
        
        // Get existing masks and add the new one
        const currentMasks = (targetClip as any).masks || [];
        currentMasks.push(newMask);
        
        updateClip(targetClip.clipId, {
          masks: currentMasks,
        });
      }
      
      // Clear brush points immediately
      setMaskBrushPoints([]);
      
      return;
    }
    
    // If brush mask was being drawn but not enough points, just clear
    if (tool === 'mask' && maskTool === 'draw' && isDrawingMaskBrush) {
      setIsDrawingMaskBrush(false);
      setMaskBrushPoints([]);
      return;
    }

    if (tool === 'draw' && isDrawingRef.current) {
      const clipId = currentDrawingClipId.current;
      isDrawingRef.current = false;
      currentDrawingClipId.current = null;
      
      // Clean up eraser state and perform final aggressive merge
      if (drawingTool === 'eraser' && clipId) {
        eraserPointsRef.current = [];
        lastEraserPosRef.current = null;
        
        // Perform aggressive merging on mouse up to consolidate all fragments
        const drawingClip = useClipStore.getState().getClipById(clipId) as DrawingClipProps | undefined;
        if (drawingClip && drawingClip.lines) {
          const fullyMergedLines = mergeConnectedLines(drawingClip.lines, true);
          useClipStore.getState().updateClip(clipId, { lines: fullyMergedLines });
        }
      } else {
        setCurrentLineId(null);
      }
      return;
    }

    if (tool === 'text' && isDrawingText && textStart && textCurrent) {
      const stage = e.target.getStage();
      if (!stage) {
        setIsDrawingText( false);
        setTextStart(null);
        setTextCurrent(null);
        return;
      }

      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) {
        setIsDrawingText(false);
        setTextStart(null);
        setTextCurrent(null);
        return;
      }

      setIsDrawingText(false);
      setTextStart(null);
      setTextCurrent(null);

      const worldEndX = (pointerPos.x - position.x) / scale;
      const worldEndY = (pointerPos.y - position.y) / scale;
      
      // Calculate shape dimensions using start and current end position
      const x = Math.min(textStart.x, worldEndX);
      const y = Math.min(textStart.y, worldEndY);
      const width = Math.abs(worldEndX - textStart.x);
      const height = Math.abs(worldEndY - textStart.y);

      if (width < 5 || height < 5) {
        setIsDrawingText(false);
        setTextStart(null);
        setTextCurrent(null);
        return;
      }

      let textTimeline = { timelineId: uuidv4(), type: 'text' as const, timelineY: 0, timelineHeight: 40, timelineWidth: 0, muted: false, hidden: false };
      addTimeline(textTimeline, -1);    

      const clipDuration = 3 * DEFAULT_FPS;
      const newClip: TextClipProps = {
        src: null,
        clipId: uuidv4(),
        type: 'text' as const,
        timelineId: textTimeline.timelineId,
        startFrame: focusFrame,
        endFrame: Math.min(focusFrame + clipDuration, totalTimelineFrames - 1),
        framesToGiveEnd: -Infinity,
        framesToGiveStart: Infinity,
        text: 'Default Text',
        fontSize: 32,
        fontWeight: 400,
        fontStyle: 'normal',
        fontFamily: 'Arial',
        color: '#ffffff',
        colorOpacity: 100,
        textAlign: 'left',
        verticalAlign: 'top',
        textTransform: 'none',
        textDecoration: 'none',
        strokeEnabled: false,
        stroke: '#000000',
        strokeWidth: 2,
        strokeOpacity: 100,
        shadowEnabled: false,
        shadowColor: '#000000',
        shadowOpacity: 75,
        shadowBlur: 4,
        shadowOffsetX: 2,
        shadowOffsetY: 2,
        shadowOffsetLocked: true,
        transform: {
          x: x,
          y: y,
          width: width,
          height: height,
          scaleX: 1,
          scaleY: 1,
          rotation: 0,
          cornerRadius: 0,
          opacity: 100,
        }
      };

      addClip(newClip);

    }
    
    if (tool === 'shape' && isDrawingShape && shapeStart && shapeCurrent) {
      // Get current pointer position to ensure accuracy
      const stage = e.target.getStage();
      if (!stage) {
        setIsDrawingShape(false);
        setShapeStart(null);
        setShapeCurrent(null);
        return;
      }

      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) {
        setIsDrawingShape(false);
        setShapeStart(null);
        setShapeCurrent(null);
        return;
      }
      
      // Convert screen coordinates to world coordinates
      const worldEndX = (pointerPos.x - position.x) / scale;
      const worldEndY = (pointerPos.y - position.y) / scale;
      
      // Calculate shape dimensions using start and current end position
      const x = Math.min(shapeStart.x, worldEndX);
      const y = Math.min(shapeStart.y, worldEndY);
      const width = Math.abs(worldEndX - shapeStart.x);
      const height = Math.abs(worldEndY - shapeStart.y);
      
      // Only create shape if it has meaningful size
      if (width > 5 && height > 5) {
        // Find or create shape timeline
        let shapeTimeline = { timelineId: uuidv4(), type: 'shape' as const, timelineY: 0, timelineHeight: 40, timelineWidth: 0, muted: false, hidden: false };
       
        addTimeline(shapeTimeline, -1);

        // Create shape clip with 3-second duration (72 frames at 24fps)
        const clipDuration = 3 * DEFAULT_FPS;
        const newClip: ShapeClipProps = {
          src: null,
          clipId: uuidv4(),
          type: 'shape' as const,
          timelineId: shapeTimeline.timelineId,
          startFrame: focusFrame,
          endFrame: Math.min(focusFrame + clipDuration, totalTimelineFrames - 1),
          framesToGiveEnd: -Infinity,
          framesToGiveStart: Infinity,
          shapeType: shape,
          fill: '#E3E3E3',
          stroke: '#E3E3E3',
          strokeWidth: 1,
          transform: {
            x,
            y,
            width,
            height,
            scaleX: 1,
            scaleY: 1,
            rotation: 0, 
            cornerRadius: 0,
            opacity: 100,
          },
        };

        if (shape === 'polygon') {
          (newClip as PolygonClipProps).sides = 3;
        }
        
        addClip(newClip);
      }
      
      // Reset shape creation state
      setIsDrawingShape(false);
      setShapeStart(null);
      setShapeCurrent(null);
    }
  }, [tool, isDrawingShape, shapeStart, shapeCurrent, shape, position, scale, addClip, addTimeline, isDrawingText, textStart, textCurrent, setCurrentLineId, maskTool, isDrawingLasso, lassoPoints, featherAmount, focusFrame, getTopmostClipAtFrame, isDrawingMaskRect, maskRectStart, maskRectCurrent, isDrawingMaskBrush, maskBrushPoints, maskBrushSize, maskShape, touchDrawMode, isDrawingTouchLasso, touchLassoPoints]);

  const onMouseLeave = useCallback((e:KonvaEventObject<MouseEvent>) => {
     // set pointer to default 
     const  container = e.target?.getStage()?.container();
     if (container) {
         container.style.cursor = 'default';
     }
     // Hide cursor preview when leaving canvas
     setCursorPreview(null);
     
     // Clean up any in-progress drawing
     if (isDrawingTouchLasso) {
       setIsDrawingTouchLasso(false);
       setTouchLassoPoints([]);
     }
     if (isDrawingLasso) {
       setIsDrawingLasso(false);
       setLassoPoints([]);
     }
     if (isDrawingMaskBrush) {
       setIsDrawingMaskBrush(false);
       setMaskBrushPoints([]);
     }
  }, [isDrawingTouchLasso, isDrawingLasso, isDrawingMaskBrush]);

  const onMouseEnter = useCallback((e:KonvaEventObject<MouseEvent>) => {
    const container = e.target?.getStage()?.container();
    if (container) {
      
      if (tool === 'hand') {
        container.style.cursor = 'grab';
      } else if (tool === 'shape' || tool === 'text') {
        container.style.cursor = 'crosshair';
      } else if (tool === 'draw') {
        // Hide cursor and use preview circle instead for accurate positioning
        container.style.cursor = 'none';
      } else if (tool === 'mask' && maskTool === 'draw') {
        // Hide cursor and use preview circle for mask brush tool
        container.style.cursor = 'none';
      } else if (tool === 'mask' && (maskTool === 'lasso' || maskTool === 'shape' || maskTool === 'touch')) {
        container.style.cursor = 'crosshair';
      } else {
        container.style.cursor = 'default';
      }
    }
  }, [tool, drawingTool, maskTool]);

  useEffect(() => {
    const container = stageRef.current?.container();
    if (container) {
      if (tool === 'hand') {
        container.style.cursor = 'grab';
      } else if (tool === 'shape' || tool === 'text') {
        container.style.cursor = 'crosshair';
      } else if (tool === 'draw') {
        // Hide cursor and use preview circle instead for accurate positioning
        container.style.cursor = 'none';
      } else if (tool === 'mask' && maskTool === 'draw') {
        // Hide cursor and use preview circle for mask brush tool
        container.style.cursor = 'none';
      } else if (tool === 'mask' && (maskTool === 'lasso' || maskTool === 'shape' || maskTool === 'touch')) {
        container.style.cursor = 'crosshair';
      } else {
        container.style.cursor = 'default';
      }
    }
  }, [tool, drawingTool, maskTool]);
  
  // Render preview shape while drawing
  const renderDrawingShape = useCallback(() => {
    if (!isDrawingShape || !shapeStart || !shapeCurrent) return null;
    
    const x = Math.min(shapeStart.x, shapeCurrent.x);
    const y = Math.min(shapeStart.y, shapeCurrent.y);
    const width = Math.abs(shapeCurrent.x - shapeStart.x);
    const height = Math.abs(shapeCurrent.y - shapeStart.y);
    
    const sharedProps = {
      stroke: '#3b82f6',
      strokeOpacity: 100,
      strokeWidth: 2,
      fill: '#3b82f644',
      fillOpacity: 100,
      dash: [5, 5],
    };
    
    switch (shape) {
      case 'rectangle':
        return <Rect {...sharedProps} x={x} y={y} width={width} height={height} />;
      case 'ellipse':
        return <KonvaEllipse {...sharedProps} x={x + width / 2} y={y + height / 2} radiusX={width / 2} radiusY={height / 2} />;
      case 'polygon':
        return <RegularPolygon {...sharedProps} x={x + width / 2} y={y + height / 2} sides={3} radius={Math.min(width, height) / 2} />;
      case 'line':
        return <KonvaLine {...sharedProps} points={[shapeStart.x, shapeStart.y, shapeCurrent.x, shapeCurrent.y]} />;
      case 'star':
        return <KonvaStar {...sharedProps} x={x + width / 2} y={y + height / 2} numPoints={5} innerRadius={Math.min(width, height) / 4} outerRadius={Math.min(width, height) / 2} />;
      default:
        return <Rect {...sharedProps} x={x} y={y} width={width} height={height} />;
    }
  }, [isDrawingShape, shapeStart, shapeCurrent, shape]);


  const renderDrawingText = useCallback(() => {
    
    if (!isDrawingText || !textStart || !textCurrent) return null;
    
    const x = Math.min(textStart.x, textCurrent.x);
    const y = Math.min(textStart.y, textCurrent.y);
    const width = Math.abs(textCurrent.x - textStart.x);
    const height = Math.abs(textCurrent.y - textStart.y);

    const sharedProps = {
      stroke: '#3b82f6',
      strokeOpacity: 100,
      strokeWidth: 2,
      fill: '#3b82f644',
      fillOpacity: 100,
      dash: [5, 5],
    };


    return <Rect {...sharedProps} x={x} y={y} width={width} height={height} />;
  }, [isDrawingText, textStart, textCurrent]);

  const renderCursorPreview = useCallback(() => {
    if (!cursorPreview || tool !== 'draw') return null;
    
    const brushSize = getCurrentSize();
    const radius = brushSize / 2;
    
    // Get the color and opacity based on the tool
    let previewColor = drawingColor;
    let previewOpacity = drawingOpacity / 100;
    
    if (drawingTool === 'highlighter') {
      previewColor = '#FFFF00';
      // Fixed 50% opacity for highlighter
      previewOpacity = 0.5;
    } else if (drawingTool === 'eraser') {
      previewColor = '#FFFFFF';
      previewOpacity = 0.5; // Fixed opacity for eraser preview
    }
    
    return (
      <>
        {/* Outer circle showing brush size */}
        <Circle
          x={cursorPreview.x}
          y={cursorPreview.y}
          radius={radius}
          stroke={previewColor}
          strokeWidth={1}
          fill={previewColor}
          opacity={previewOpacity}
          listening={false}
        />
        {/* Center dot showing exact drawing point */}
        <Circle
          x={cursorPreview.x}
          y={cursorPreview.y}
          radius={1}
          fill={drawingTool === 'eraser' ? '#000000' : previewColor}
          opacity={1}
          listening={false}
        />
      </>
    );
  }, [cursorPreview, tool, drawingTool, drawingColor, drawingOpacity, getCurrentSize]);

  const renderMaskCursorPreview = useCallback(() => {
    // Hide cursor preview while actively drawing to avoid overlap
    if (!maskCursorPreview || tool !== 'mask' || maskTool !== 'draw' || isDrawingMaskBrush) return null;
    
    const radius = maskBrushSize / 2;
    
    return (
      <>
        {/* Outer circle showing brush size */}
        <Circle
          x={maskCursorPreview.x}
          y={maskCursorPreview.y}
          radius={radius}
          stroke="rgb(0, 127, 245)"
          strokeWidth={1}
          fill="rgb(0, 127, 245)"
          opacity={0.4}
          listening={false}
        />
        {/* Center dot showing exact drawing point */}
        <Circle
          x={maskCursorPreview.x}
          y={maskCursorPreview.y}
          radius={1}
          fill="rgb(0, 127, 245)"
          opacity={1}
          listening={false}
        />
      </>
    );
  }, [maskCursorPreview, maskBrushSize, tool, maskTool, isDrawingMaskBrush]);

  const renderMaskBrushDrawing = useCallback(() => {
    // Need at least 2 points (4 values) to render a line
    if (!isDrawingMaskBrush || maskBrushPoints.length < 4) return null;
    
    return (
      <KonvaLine
        points={maskBrushPoints}
        stroke="rgb(0, 127, 245)"
        strokeWidth={maskBrushSize}
        opacity={0.4}
        lineCap="round"
        lineJoin="round"
        tension={0.5}
        listening={false}
      />
    );
  }, [isDrawingMaskBrush, maskBrushPoints, maskBrushSize]);
  
  const renderLassoPath = useCallback(() => {
    if (lassoPoints.length < 2) return null;
    
    // While drawing, show a glowing line
    if (isDrawingLasso) {
      return (
        <>
          {/* Outer glow - largest, most transparent */}
          <KonvaLine
            points={lassoPoints}
            stroke="rgba(0, 127, 245, 0.2)"
            strokeWidth={6}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Middle glow */}
          <KonvaLine
            points={lassoPoints}
            stroke="rgba(0, 127, 245, 0.4)"
            strokeWidth={4}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Inner glow */}
          <KonvaLine
            points={lassoPoints}
            stroke="rgba(0, 127, 245, 0.6)"
            strokeWidth={2}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Core line - bright and solid */}
          <KonvaLine
            points={lassoPoints}
            stroke="rgb(0, 127, 245)"
            strokeWidth={2}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
        </>
      );
    }
    
    return null;
  }, [lassoPoints, isDrawingLasso]);
  
  const renderTouchLassoPath = useCallback(() => {
    if (touchLassoPoints.length < 2) return null;
    
    // While drawing, show a glowing line
    if (isDrawingTouchLasso) {
      return (
        <>
          {/* Outer glow - largest, most transparent */}
          <KonvaLine
            points={touchLassoPoints}
            stroke="rgba(0, 127, 245, 0.2)"
            strokeWidth={6}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Middle glow */}
          <KonvaLine
            points={touchLassoPoints}
            stroke="rgba(0, 127, 245, 0.4)"
            strokeWidth={4}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Inner glow */}
          <KonvaLine
            points={touchLassoPoints}
            stroke="rgba(0, 127, 245, 0.6)"
            strokeWidth={2}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          {/* Core line - bright and solid */}
          <KonvaLine
            points={touchLassoPoints}
            stroke="rgb(0, 127, 245)"
            strokeWidth={2}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
        </>
      );
    }
    
    return null;
  }, [touchLassoPoints, isDrawingTouchLasso]);
  
  const renderMaskRectPreview = useCallback(() => {
    if (!isDrawingMaskRect || !maskRectStart || !maskRectCurrent) return null;
    
    const x = Math.min(maskRectStart.x, maskRectCurrent.x);
    const y = Math.min(maskRectStart.y, maskRectCurrent.y);
    const width = Math.abs(maskRectCurrent.x - maskRectStart.x);
    const height = Math.abs(maskRectCurrent.y - maskRectStart.y);
    
    const sharedProps = {
      stroke: 'rgb(0, 127, 245)',
      strokeWidth: 1.5,
      fill: 'rgba(0, 127, 245, 0.2)',
      dash: [5, 5],
      listening: false,
    };
    
    switch (maskShape) {
      case 'rectangle':
        return <Rect {...sharedProps} x={x} y={y} width={width} height={height} />;
      case 'ellipse':
        return <KonvaEllipse {...sharedProps} x={x + width / 2} y={y + height / 2} radiusX={width / 2} radiusY={height / 2} />;
      case 'polygon':
        return <RegularPolygon {...sharedProps} x={x + width / 2} y={y + height / 2} sides={3} radius={Math.min(width, height) / 2} />;
      case 'star':
        return <KonvaStar {...sharedProps} x={x + width / 2} y={y + height / 2} numPoints={5} innerRadius={Math.min(width, height) / 4} outerRadius={Math.min(width, height) / 2} />;
      default:
        return <Rect {...sharedProps} x={x} y={y} width={width} height={height} />;
    }
  }, [isDrawingMaskRect, maskRectStart, maskRectCurrent, maskShape]);

  return (
    <>
    {isFullscreen ? (
      <FullscreenPreview onExit={() => setIsFullscreen(false)} />
    ) : (
      <div className='w-full h-full relative' ref={containerRef}>
        <Stage
          ref={stageRef}
          width={size.width}
          height={size.height}
          className='bg-brand-background'
          onWheel={handleWheel}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseLeave}
          onMouseEnter={onMouseEnter}
        >
          <Layer
             ref={layerRef}
             width={size.width}
             height={size.height}
          >
           <Group x={position.x} y={position.y} scaleX={scale} scaleY={scale} width={rectWidth} height={rectHeight} >
            <Rect x={0} y={0}  width={rectWidth} height={rectHeight} fill={'#000000'} />
               {sortClips(filterClips(clips)).map((clip) => {
                const clipAtFrame = clipWithinFrame(clip, focusFrame);
                if (!clipAtFrame) return null;
                
                // Get applicators for clips that support effects (video, image, etc.)
                const applicators = getClipApplicators(clip.clipId);

                 if (clipAtFrame) {
                   switch (clip.type) {
                    case 'video':
                      return <VideoPreview key={clip.clipId} {...clip} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators}  />
                    case 'image':
                      return <ImagePreview key={clip.clipId} {...clip} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} />
                    case 'shape':
                      return <ShapePreview key={clip.clipId} {...clip} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators} />
                    case 'text':
                      return <TextPreview key={clip.clipId} {...clip} rectWidth={rectWidth} rectHeight={rectHeight} applicators={applicators}  />
                    case 'draw':
                      return <DrawingPreview key={clip.clipId} {...clip} rectWidth={rectWidth} rectHeight={rectHeight} />
                    default:
                      // Applicator clips (filter, mask, processor, etc.) don't render visually
                      return null
                   }
                 } else {
                  return null;
                 }
               })}
               {renderDrawingShape()}
               {renderDrawingText()}
               <MaskPreview clips={clips} sortClips={sortClips} filterClips={filterClips} rectWidth={rectWidth} rectHeight={rectHeight} />
               {renderLassoPath()}
               {renderTouchLassoPath()}
               {renderMaskRectPreview()}
               {renderMaskBrushDrawing()}
               {renderCursorPreview()}
               {renderMaskCursorPreview()}
           </Group>
          </Layer>
        </Stage>
        {/* Fullscreen button */}
        <button
          onClick={() => setIsFullscreen(true)}
          className="absolute bottom-4 right-4 p-2 bg-brand cursor-pointer hover:bg-brand-background-dark hover:text-blue-400 text-white rounded-md transition-colors"
        >
          <SlSizeFullscreen className="h-3 w-3" />
        </button>
      </div>
    )}
    {/* Mount non-visual audio previews OUTSIDE Konva tree so effects run */}
    {<>
        {sortClips(filterClips(clips, true)).map((clip) => {
          const clipAtFrame = clipWithinFrame(clip, focusFrame);
          if (!clipAtFrame) return null;
          if (clip.type === 'audio' || clip.type === 'video') {
            return <AudioPreview key={`audio-${clip.clipId}`} {...(clip as any)} />
          }
          return null;
        })}
      </>
    }
    </>
  )
}

export default Preview;