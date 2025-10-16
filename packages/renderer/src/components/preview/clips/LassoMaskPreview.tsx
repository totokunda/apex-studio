import React, { useCallback, useRef } from 'react';
import { Group, Line as KonvaLine, Rect as KonvaRect, Transformer } from 'react-konva';
import { useClipStore } from '@/lib/clip';
import { useViewportStore } from '@/lib/viewport';
import { useControlsStore } from '@/lib/control';
import { MaskClipProps } from '@/lib/types';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';

interface LassoMaskPreviewProps {
  mask: MaskClipProps;
  points: number[];
  animationOffset: number;
  rectWidth: number;
  rectHeight: number;
}

const LassoMaskPreview: React.FC<LassoMaskPreviewProps> = ({
  mask,
  points,
  animationOffset,
  rectWidth,
  rectHeight,
}) => {
  const tool = useViewportStore((s) => s.tool);
  const isFullscreen = useControlsStore((s) => s.isFullscreen);
  const selectedMaskId = useControlsStore((s) => s.selectedMaskId);
  const setSelectedMaskId = useControlsStore((s) => s.setSelectedMaskId);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  
  const groupRef = useRef<Konva.Group>(null);
  const rectRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const suppressUntilRef = useRef(0);
  
  const isSelected = selectedMaskId === mask.id;

  // Calculate bounding box for the lasso
  const bounds = React.useMemo(() => {
    if (points.length < 2) return { x: 0, y: 0, width: 100, height: 100 };
    
    let minX = points[0];
    let maxX = points[0];
    let minY = points[1];
    let maxY = points[1];
    
    for (let i = 0; i < points.length; i += 2) {
      minX = Math.min(minX, points[i]);
      maxX = Math.max(maxX, points[i]);
      minY = Math.min(minY, points[i + 1]);
      maxY = Math.max(maxY, points[i + 1]);
    }
    
    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }, [points]);
  

  const handleClick = useCallback((e: KonvaEventObject<MouseEvent>) => {
    if (tool !== 'pointer') return;
    e.cancelBubble = true;
    setSelectedMaskId(mask.id);
  }, [tool, mask.id, setSelectedMaskId]);
  
  const handleDragEnd = useCallback(() => {
    const group = groupRef.current;
    if (!group) return;

    const dx = group.x();
    const dy = group.y();
    
    // Only update if there was actual movement
    if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) {
      group.position({ x: 0, y: 0 });
      return;
    }
    
    // Update mask position by shifting all points
    const keyframes = mask.keyframes instanceof Map 
      ? mask.keyframes 
      : (mask.keyframes as Record<number, any>);
    
    const keyframeNumbers = keyframes instanceof Map
      ? Array.from(keyframes.keys()).sort((a, b) => a - b)
      : Object.keys(keyframes).map(Number).sort((a, b) => a - b);
    
    const activeKeyframe = keyframeNumbers.filter(k => k <= focusFrame).pop();
    
    if (activeKeyframe !== undefined) {
      const maskData = keyframes instanceof Map 
        ? keyframes.get(activeKeyframe) 
        : keyframes[activeKeyframe];
      
      if (maskData?.lassoPoints) {
        const newPoints = [];
        for (let i = 0; i < maskData.lassoPoints.length; i += 2) {
          newPoints.push(maskData.lassoPoints[i] + dx);
          newPoints.push(maskData.lassoPoints[i + 1] + dy);
        }
        
        const updatedKeyframes = keyframes instanceof Map
          ? new Map(keyframes)
          : { ...keyframes };
        
        if (updatedKeyframes instanceof Map) {
          updatedKeyframes.set(activeKeyframe, {
            ...maskData,
            lassoPoints: newPoints,
          });
        } else {
          updatedKeyframes[activeKeyframe] = {
            ...maskData,
            lassoPoints: newPoints,
          };
        }
        
        const { clips } = useClipStore.getState();
        const targetClip = clips.find(c => c.clipId === mask.clipId);
        
        if (targetClip && mask.clipId) {
          const currentMasks = (targetClip as any).masks || [];
          const updatedMasks = currentMasks.map((m: MaskClipProps) =>
            m.id === mask.id ? { ...m, keyframes: updatedKeyframes } : m
          );
          
          useClipStore.getState().updateClip(mask.clipId, { masks: updatedMasks });
        }
      }
    }
    
    // Reset group position
    group.position({ x: 0, y: 0 });
    
    // Add suppression period after drag
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    suppressUntilRef.current = now + 250;
  }, [mask, focusFrame]);

  React.useEffect(() => {
    if (!isSelected || !tool || tool !== 'pointer') return;
    const tr = transformerRef.current;
    const rect = rectRef.current;
    if (!tr || !rect) return;
    
    tr.nodes([rect]);
    tr.getLayer()?.batchDraw();
  }, [isSelected, tool]);
  
  // Deselect when clicking outside
  React.useEffect(() => {
    if (!isSelected) return;
    
    const handleClickOutside = (e: MouseEvent) => {
      // Check suppression period
      const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      if (now < suppressUntilRef.current) return;
      
      const stage = groupRef.current?.getStage();
      if (!stage) return;
      
      const container = stage.container();
      if (!container.contains(e.target as Node)) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      const rect = rectRef.current;
      if (!rect) return;
      
      const rectPos = rect.getClientRect();
      const isInside = 
        pointerPos.x >= rectPos.x && 
        pointerPos.x <= rectPos.x + rectPos.width &&
        pointerPos.y >= rectPos.y && 
        pointerPos.y <= rectPos.y + rectPos.height;
      
      if (!isInside) {
        setSelectedMaskId(null);
      }
    };
    
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  }, [isSelected, setSelectedMaskId]);

  return (
    <Group
      ref={groupRef}
      draggable={tool === 'pointer' && isSelected}
      onDragEnd={handleDragEnd}
      clipX={0}
      clipY={0}
      clipWidth={rectWidth}
      clipHeight={rectHeight}
    >
      {/* Invisible rect for selection and dragging */}
      <KonvaRect
        ref={rectRef}
        x={bounds.x}
        y={bounds.y}
        width={bounds.width}
        height={bounds.height}
        fill="transparent"
        onClick={handleClick}
        onTap={handleClick}
      />
      
      {/* White stripe background */}
      <KonvaLine
        points={points}
        stroke="#ffffff"
        strokeWidth={1}
        lineCap="round"
        lineJoin="round"
        closed={true}
        listening={false}
        fill="rgba(0, 127, 245, 0.4)"
      />
      
      {/* Black stripe foreground with animation */}
      <KonvaLine
        points={points}
        stroke="#000000"
        strokeWidth={1}
        dash={[4.5, 4.5]}
        dashOffset={-animationOffset}
        lineCap="round"
        lineJoin="round"
        closed={true}
        listening={false}
      />
      
      {/* Transformer for visual feedback (drag only, no anchors) */}
      {tool === 'pointer' && isSelected && !isFullscreen && (
        <Transformer
          ref={transformerRef}
          borderStroke="#AE81CE"
          borderStrokeWidth={2}
          enabledAnchors={[]}
          rotateEnabled={false}
          resizeEnabled={false}
        />
      )}
    </Group>
  );
};

export default LassoMaskPreview;

