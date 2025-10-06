
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Layer, Group, Rect, Line as KonvaLine, Ellipse as KonvaEllipse, RegularPolygon, Star as KonvaStar } from 'react-konva';
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
import { useControlsStore } from '@/lib/control';
import { AnyClipProps, PolygonClipProps, ShapeClipProps, TextClipProps } from '@/lib/types';
import { v4 as uuidv4 } from 'uuid';
import { SlSizeFullscreen } from 'react-icons/sl';
import FullscreenPreview from './FullscreenPreview';
import { getApplicatorsForClip } from '@/lib/applicator-utils';
import { useWebGLHaldClut } from './webgl-filters';

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

  const onMouseDown = useCallback((e: any) => {
    if (tool === 'hand') {
      isPanning.current = true;
      lastPointer.current = { x: e.evt.offsetX, y: e.evt.offsetY };
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
    }

  }, [tool, position, scale]);

  const onMouseMove = useCallback((e: any) => {
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
  }, [panBy, tool, isDrawingShape, shapeStart, position, scale, isDrawingText, textStart]);

  const onMouseUp = useCallback((e: any) => {
    if (tool === 'hand') {
      isPanning.current = false;
      lastPointer.current = null;
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
  }, [tool, isDrawingShape, shapeStart, shapeCurrent, shape, position, scale, addClip, addTimeline, isDrawingText, textStart, textCurrent]);

  const onMouseLeave = useCallback((e:KonvaEventObject<MouseEvent>) => {
     // set pointer to default 
     const  container = e.target?.getStage()?.container();
     if (container) {
         container.style.cursor = 'default';
     }
  }, []);

  const onMouseEnter = useCallback((e:KonvaEventObject<MouseEvent>) => {
    const container = e.target?.getStage()?.container();
    if (container) {
      if (tool === 'hand') {
        container.style.cursor = 'grab';
      } else if (tool === 'shape' || tool === 'text') {
        container.style.cursor = 'crosshair';
      } else {
        container.style.cursor = 'default';
      }
    }
  }, [tool]);

  useEffect(() => {
    const container = stageRef.current?.container();
    if (container) {
      if (tool === 'hand') {
        container.style.cursor = 'grab';
      } else if (tool === 'shape' || tool === 'text') {
        container.style.cursor = 'crosshair';
      } else {
        container.style.cursor = 'default';
      }
    }
  }, [tool]);
  
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