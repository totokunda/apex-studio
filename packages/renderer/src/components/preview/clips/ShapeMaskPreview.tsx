import React, { useCallback, useRef } from 'react';
import { Group, Rect as KonvaRect, Ellipse as KonvaEllipse, Star as KonvaStar, Transformer, RegularPolygon } from 'react-konva';
import { useClipStore } from '@/lib/clip';
import { useViewportStore } from '@/lib/viewport';
import { useControlsStore } from '@/lib/control';
import { MaskClipProps, MaskShapeTool } from '@/lib/types';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';

interface ShapeMaskPreviewProps {
  mask: MaskClipProps;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  animationOffset: number;
  rectWidth: number;
  rectHeight: number;
  shapeType: MaskShapeTool;
  scaleX?: number;
  scaleY?: number;
}

const ShapeMaskPreview: React.FC<ShapeMaskPreviewProps> = ({
  mask,
  x,
  y,
  width,
  height,
  rotation,
  animationOffset,
  rectWidth,
  rectHeight,
  shapeType,
  scaleX = 1,
  scaleY = 1,
}) => {
  const tool = useViewportStore((s) => s.tool);
  const isFullscreen = useControlsStore((s) => s.isFullscreen);
  const selectedMaskId = useControlsStore((s) => s.selectedMaskId);
  const setSelectedMaskId = useControlsStore((s) => s.setSelectedMaskId);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  
  const shapeWhiteRef = useRef<any>(null);
  const shapeBlackRef = useRef<any>(null);
  const groupRef = useRef<Konva.Group>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const suppressUntilRef = useRef(0);
  
  const isSelected = selectedMaskId === mask.id;

  const handleClick = useCallback((e: KonvaEventObject<MouseEvent>) => {
    if (tool !== 'pointer') return;
    e.cancelBubble = true;
    setSelectedMaskId(mask.id);
  }, [tool, mask.id, setSelectedMaskId]);
  
  const updateMaskBounds = useCallback(() => {
    const shape = shapeWhiteRef.current;
    if (!shape) return;

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
      
      if (maskData?.shapeBounds || maskData?.rectangleBounds) {
        const updatedKeyframes = keyframes instanceof Map
          ? new Map(keyframes)
          : { ...keyframes };
        
        // For centered shapes (ellipse, polygon, star), convert from center position to top-left
        const isCentered = shapeType === 'ellipse' || shapeType === 'polygon' || shapeType === 'star';
        const shapeX = shape.x();
        const shapeY = shape.y();
        const shapeScaleX = shape.scaleX();
        const shapeScaleY = shape.scaleY();
        
        // For rectangle and ellipse: update width/height, reset scale to 1
        // For polygon and star: keep scale on the node, don't update width/height
        const useWidthHeight = shapeType === 'rectangle' || shapeType === 'ellipse';
        
        let actualWidth: number;
        let actualHeight: number;
        let newScaleX = 1;
        let newScaleY = 1;
        
        if (useWidthHeight) {
          if (shapeType === 'ellipse') {
            actualWidth = shape.radiusX() * 2 * shapeScaleX;
            actualHeight = shape.radiusY() * 2 * shapeScaleY;
          } else {
            // Rectangle
            actualWidth = shape.width() * shapeScaleX;
            actualHeight = shape.height() * shapeScaleY;
          }
        } else {
          // For polygon and star, keep the base width/height and track scale
          const currentBounds = maskData.shapeBounds || maskData.rectangleBounds;
          actualWidth = currentBounds?.width || width;
          actualHeight = currentBounds?.height || height;
          newScaleX = shapeScaleX;
          newScaleY = shapeScaleY;
          
          // Calculate actual dimensions for position calculation
          const scaledWidth = actualWidth * shapeScaleX;
          const scaledHeight = actualHeight * shapeScaleY;
          const saveX = isCentered ? shapeX - scaledWidth / 2 : shapeX;
          const saveY = isCentered ? shapeY - scaledHeight / 2 : shapeY;
          
          const newBounds = {
            x: saveX,
            y: saveY,
            width: actualWidth,
            height: actualHeight,
            rotation: shape.rotation(),
            shapeType,
            scaleX: newScaleX,
            scaleY: newScaleY,
          };
          
          if (updatedKeyframes instanceof Map) {
            updatedKeyframes.set(activeKeyframe, {
              ...maskData,
              shapeBounds: newBounds,
            });
          } else {
            updatedKeyframes[activeKeyframe] = {
              ...maskData,
              shapeBounds: newBounds,
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
          
          // Don't reset scale for polygon/star
          syncBlackShape();
          return;
        }
        
        const saveX = isCentered ? shapeX - actualWidth / 2 : shapeX;
        const saveY = isCentered ? shapeY - actualHeight / 2 : shapeY;
        
        const newBounds = {
          x: saveX,
          y: saveY,
          width: actualWidth,
          height: actualHeight,
          rotation: shape.rotation(),
          shapeType,
        };
        
        if (updatedKeyframes instanceof Map) {
          updatedKeyframes.set(activeKeyframe, {
            ...maskData,
            shapeBounds: newBounds,
          });
        } else {
          updatedKeyframes[activeKeyframe] = {
            ...maskData,
            shapeBounds: newBounds,
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

    // Reset scale to 1 for rectangle and ellipse (already handled for polygon/star above)
    if (shapeType === 'rectangle') {
      shape.scaleX(1);
      shape.scaleY(1);
    } else if (shapeType === 'ellipse') {
      shape.scaleX(1);
      shape.scaleY(1);
    }
    
    // Sync black shape with white shape
    syncBlackShape();
  }, [mask, focusFrame, width, height, shapeType]);
  
  const syncBlackShape = useCallback(() => {
    const whiteShape = shapeWhiteRef.current;
    const blackShape = shapeBlackRef.current;
    
    if (whiteShape && blackShape) {
      blackShape.x(whiteShape.x());
      blackShape.y(whiteShape.y());
      blackShape.rotation(whiteShape.rotation());
      
      if (shapeType === 'rectangle') {
        blackShape.width(whiteShape.width() * whiteShape.scaleX());
        blackShape.height(whiteShape.height() * whiteShape.scaleY());
        blackShape.scaleX(1);
        blackShape.scaleY(1);
      } else if (shapeType === 'ellipse') {
        blackShape.radiusX(whiteShape.radiusX() * whiteShape.scaleX());
        blackShape.radiusY(whiteShape.radiusY() * whiteShape.scaleY());
        blackShape.scaleX(1);
        blackShape.scaleY(1);
      } else {
        // For polygon and star, use scale
        blackShape.scaleX(whiteShape.scaleX());
        blackShape.scaleY(whiteShape.scaleY());
      }
    }
  }, [shapeType]);
  
  const handleTransform = useCallback(() => {
    syncBlackShape();
  }, [syncBlackShape]);
  
  const handleDragMove = useCallback(() => {
    syncBlackShape();
  }, [syncBlackShape]);
  
  const handleTransformEnd = useCallback(() => {
    updateMaskBounds();
    
    // Add suppression period after transform
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    suppressUntilRef.current = now + 250;
  }, [updateMaskBounds]);
  
  const handleDragEnd = useCallback(() => {
    updateMaskBounds();
    
    // Add suppression period after drag
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    suppressUntilRef.current = now + 250;
  }, [updateMaskBounds]);

  React.useEffect(() => {
    if (!isSelected || !tool || tool !== 'pointer') return;
    const tr = transformerRef.current;
    const whiteShape = shapeWhiteRef.current;
    if (!tr || !whiteShape) return;
    
    // Sync black shape with white shape initially
    syncBlackShape();
    
    tr.nodes([whiteShape]);
    tr.getLayer()?.batchDraw();
  }, [isSelected, tool, syncBlackShape]);
  
  // Deselect when clicking outside
  React.useEffect(() => {
    if (!isSelected) return;
    
    const handleClickOutside = (e: MouseEvent) => {
      // Check suppression period
      const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      if (now < suppressUntilRef.current) return;
      
      const shape = shapeWhiteRef.current;
      if (!shape) return;
      
      const stage = shape.getStage();
      if (!stage) return;
      
      const container = stage.container();
      if (!container.contains(e.target as Node)) return;
      
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) return;
      
      const shapePos = shape.getClientRect();
      const isInside = 
        pointerPos.x >= shapePos.x && 
        pointerPos.x <= shapePos.x + shapePos.width &&
        pointerPos.y >= shapePos.y && 
        pointerPos.y <= shapePos.y + shapePos.height;
      
      if (!isInside) {
        setSelectedMaskId(null);
      }
    };
    
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  }, [isSelected, setSelectedMaskId]);

  const renderShape = useCallback((isWhite: boolean) => {
    const baseProps = {
      ref: isWhite ? shapeWhiteRef : shapeBlackRef,
      stroke: isWhite ? "#ffffff" : "#000000",
      strokeWidth: 1,
      fill: isWhite ? "rgba(0, 127, 245, 0.4)" : undefined,
      draggable: tool === 'pointer' && isSelected && isWhite,
      onClick: isWhite ? handleClick : undefined,
      onTap: isWhite ? handleClick : undefined,
      onTransform: isWhite ? handleTransform : undefined,
      onTransformEnd: isWhite ? handleTransformEnd : undefined,
      onDragMove: isWhite ? handleDragMove : undefined,
      onDragEnd: isWhite ? handleDragEnd : undefined,
      listening: isWhite,
      rotation,
      dash: !isWhite ? [4.5, 4.5] : undefined,
      dashOffset: !isWhite ? -animationOffset : undefined,
    };

    // Props for shapes positioned from top-left corner
    const cornerProps = {
      ...baseProps,
      x,
      y,
    };

    // Props for shapes positioned from center
    // For polygon and star, apply scale; for others, dimensions are already scaled
    const actualWidth = (shapeType === 'polygon' || shapeType === 'star') ? width * scaleX : width;
    const actualHeight = (shapeType === 'polygon' || shapeType === 'star') ? height * scaleY : height;
    
    const centerProps = {
      ...baseProps,
      x: x + actualWidth / 2,
      y: y + actualHeight / 2,
      scaleX: (shapeType === 'polygon' || shapeType === 'star') ? scaleX : 1,
      scaleY: (shapeType === 'polygon' || shapeType === 'star') ? scaleY : 1,
    };
    
    switch (shapeType) {
      case 'rectangle':
        return <KonvaRect {...cornerProps} width={width} height={height} />;
      
      case 'ellipse':
        return <KonvaEllipse {...centerProps} radiusX={width / 2} radiusY={height / 2} />;
      
      case 'polygon':
        return <RegularPolygon {...centerProps} sides={3} radius={Math.min(width, height) / 2} />;
      
      case 'star':
        return <KonvaStar {...centerProps} numPoints={5} innerRadius={Math.min(width, height) / 4} outerRadius={Math.min(width, height) / 2} />;
      
      default:
        return <KonvaRect {...cornerProps} width={width} height={height} />;
    }
  }, [x, y, width, height, rotation, animationOffset, shapeType, scaleX, scaleY, tool, isSelected, handleClick, handleTransform, handleTransformEnd, handleDragMove, handleDragEnd]);

  return (
    <Group ref={groupRef} clipX={0} clipY={0} clipWidth={rectWidth} clipHeight={rectHeight}>
      {/* White shape background */}
      {renderShape(true)}
      
      {/* Black shape foreground with animation */}
      {renderShape(false)}
      
      {/* Transformer */}
      {tool === 'pointer' && isSelected && !isFullscreen && (
        <Transformer
          ref={transformerRef}
          borderStroke="#AE81CE"
          anchorCornerRadius={8}
          anchorStroke="#E3E3E3"
          anchorStrokeWidth={1}
          borderStrokeWidth={2}
          rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]}
          enabledAnchors={['top-left', 'top-center', 'top-right', 'bottom-left', 'bottom-right', 'middle-left', 'middle-right', 'bottom-center']}
        />
      )}
    </Group>
  );
};

export default ShapeMaskPreview;

