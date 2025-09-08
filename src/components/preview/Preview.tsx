
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Layer, Group, Rect } from 'react-konva';
import { useViewportStore } from '@/lib/viewport';
import { useClipStore } from '@/lib/clip';
import { KonvaEventObject } from 'konva/lib/Node';
import { BASE_LONG_SIDE } from '@/lib/settings'; 
import _ from 'lodash';
import VideoPreview from './clips/VideoPreview';
import ImagePreview from './clips/ImagePreview';
import { useControlsStore } from '@/lib/control';

interface PreviewProps {

}

const Preview:React.FC<PreviewProps> = () => {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<any>(null);
  const layerRef = useRef<any>(null);
  const scale = useViewportStore((s) => s.scale);
  const position = useViewportStore((s) => s.position);
  const zoomAtScreenPoint = useViewportStore((s) => s.zoomAtScreenPoint);
  const panBy = useViewportStore((s) => s.panBy);
  const tool = useViewportStore((s) => s.tool);
  const setViewportSize = useViewportStore((s) => s.setViewportSize);
  const setContentBounds = useViewportStore((s) => s.setContentBounds);
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const clips = useClipStore((s) => s.clips);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const getClipAtFrame = useClipStore((s) => s.getClipAtFrame);

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

  // Compute rect dimensions based on aspect ratio
  const { rectWidth, rectHeight } = useMemo(() => {
    const baseLongSide = BASE_LONG_SIDE; // world units
    const ratio = aspectRatio.width / aspectRatio.height;
    if (ratio >= 1) {
      return { rectWidth: baseLongSide, rectHeight: baseLongSide / ratio };
    }
    return { rectWidth: baseLongSide * ratio, rectHeight: baseLongSide };
  }, [aspectRatio.width, aspectRatio.height]);


  // Center the rect initially and whenever aspect ratio or viewport changes
  useEffect(() => {
    if (!stageRef.current) return;
    const rectBounds = { x: 0, y: 0, width: rectWidth, height: rectHeight };
    setContentBounds(rectBounds);
    const rectWorld = { x: rectBounds.x + rectBounds.width / 2, y: rectBounds.y + rectBounds.height / 2 };
    useViewportStore.getState().centerOnWorldPoint(rectWorld, { width: size.width, height: size.height });
  }, [size.width, size.height, rectWidth, rectHeight]);


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
    if (tool !== 'hand') return;
    isPanning.current = true;
    lastPointer.current = { x: e.evt.offsetX, y: e.evt.offsetY };
  }, [tool]);

  const onMouseMove = useCallback((e: any) => {
    if (!isPanning.current || tool !== 'hand') return;
    const current = { x: e.evt.offsetX, y: e.evt.offsetY };
    const last = lastPointer.current || current;
    const dx = current.x - last.x;
    const dy = current.y - last.y;
    panBy(-dx, -dy);
    lastPointer.current = current;
  }, [panBy, tool]);

  const onMouseUp = useCallback(() => {
    isPanning.current = false;
    lastPointer.current = null;
  }, []);

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
      container.style.cursor = tool === 'hand' ? 'grab' : 'default';
    }
  }, [tool]);

  useEffect(() => {
    const container = stageRef.current?.container();
    if (container) {
      container.style.cursor = tool === 'hand' ? 'grab' : 'default';
    }
  }, [tool]);

  return (
    <div className='w-full h-[calc(100%-48px)] mt-12' ref={containerRef}>
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
          <Rect x={0} y={0} height={rectHeight} width={rectWidth} fill={'#000000'} />
             {clips.map((clip) => {
              const clipAtFrame = getClipAtFrame(focusFrame);
              if (!clipAtFrame) return null;
               if (clipAtFrame[0].clipId === clip.clipId) {
                 switch (clip.type) {
                  case 'video':
                    return <VideoPreview key={clip.clipId} {...clip} rectWidth={rectWidth} rectHeight={rectHeight} />
                  case 'image':
                    return <ImagePreview key={clip.clipId} {...clip} rectWidth={rectWidth} rectHeight={rectHeight} />
                 }
               } else {
                return null;
               }
             })}
         </Group>
        </Layer>
      </Stage>

    </div>
  )
}

export default Preview;