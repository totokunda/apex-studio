import React, { useMemo, useEffect, useState, useRef } from 'react';
import { Group, Circle, Line } from 'react-konva';
import { useMask, useMaskStore } from '@/lib/mask';
import { PreprocessorClipType } from '@/lib/types';
import { useControlsStore } from '@/lib/control';
import { useClipStore } from '@/lib/clip';
import { getMediaInfoCached } from '@/lib/media/utils';

interface TouchMaskPreviewProps {
  clip: PreprocessorClipType;
  touchPoints?: Array<{ x: number; y: number; label: 'positive' | 'negative' }>;
  lassoStrokes?: Array<number[]>; // Array of lasso paths (each path is [x1, y1, x2, y2, ...])
  rectWidth: number;
  rectHeight: number;
  animationOffset: number;
}

const TouchMaskPreview: React.FC<TouchMaskPreviewProps> = ({ clip, touchPoints, lassoStrokes, rectWidth, rectHeight, animationOffset }) => {
  
  if ((!touchPoints || touchPoints.length === 0) && (!lassoStrokes || lassoStrokes.length === 0)) return null;

  const focusFrame = useControlsStore((s) => s.focusFrame);
  const startFrame = clip.startFrame ?? 0;
  const framesToGiveStart = clip.framesToGiveStart ?? 0;
  const getClipTransform = useClipStore((s) => s.getClipTransform);

  const touchDrawMode = useMaskStore((s) => s.touchDrawMode);

  const [renderedTouchPoints, setRenderedTouchPoints] = useState<Array<{ x: number; y: number; label: 'positive' | 'negative' }>>([]);
  const [contours, setContours] = useState<Array<Array<number>>>([]);
  const lastStoredDataRef = useRef<string>('');
  
  // Load existing stored contours and data on mount or when frame changes
  useEffect(() => {
    try {
      const masks = (clip as any).masks || [];
      const currentMask = masks.find((m: any) => m.tool === 'touch');
      
      if (currentMask && currentMask.keyframes) {
        const keyframes = currentMask.keyframes instanceof Map 
          ? currentMask.keyframes 
          : (currentMask.keyframes as Record<number, any>);
        
        if (!keyframes) return;
        
        const keyframeNumbers = keyframes instanceof Map
          ? Array.from(keyframes.keys()).sort((a, b) => a - b)
          : Object.keys(keyframes).map(Number).sort((a, b) => a - b);
        
        const activeKeyframe = keyframeNumbers.filter((k: number) => k <= focusFrame).pop();
        
        if (activeKeyframe !== undefined) {
          const maskData = keyframes instanceof Map 
            ? keyframes.get(activeKeyframe) 
            : keyframes[activeKeyframe];
          
          if (maskData?.contours) {
            setContours(maskData.contours);
            if (maskData?.touchPoints) {
              setRenderedTouchPoints(maskData.touchPoints);
            }
            // Note: lassoStrokes are passed as props, not loaded from storage
            // They're stored for persistence but rendered from parent component
          }
        }
      }
    } catch (error) {
      console.error('Error loading touch mask contours:', error);
    }
  }, [clip, focusFrame]);
  
  // Get media info and clip transform
  const mediaInfo = useMemo(() => getMediaInfoCached(clip.src) || null, [clip.src]);
  const clipTransform = useMemo(() => getClipTransform(clip.clipId || ''), [clip.clipId, getClipTransform]);
  
  const points = useMemo<Array<{ x: number; y: number }>>(() => {
    if (!touchPoints) return [];
    return touchPoints.map((point) => ({x: point.x, y: point.y}));
  }, [touchPoints]);
  
  const pointLabels = useMemo<Array<number>>(() => {
    if (!touchPoints) return [];
    return touchPoints.map((point) => point.label === 'positive' ? 1 : 0);
  }, [touchPoints]);
  
  // Calculate bounding box from the most recent lasso stroke only
  const lassoBox = useMemo(() => {
    if (!lassoStrokes || lassoStrokes.length === 0) return undefined;
    
    // Use only the most recent lasso stroke
    const latestStroke = lassoStrokes[lassoStrokes.length - 1];
    if (!latestStroke || latestStroke.length < 2) return undefined;
    
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    
    for (let i = 0; i < latestStroke.length; i += 2) {
      minX = Math.min(minX, latestStroke[i]);
      maxX = Math.max(maxX, latestStroke[i]);
      minY = Math.min(minY, latestStroke[i + 1]);
      maxY = Math.max(maxY, latestStroke[i + 1]);
    }
    
    if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minY) || !isFinite(maxY)) {
      return undefined;
    }
    
    return {
      x1: minX,
      y1: minY,
      x2: maxX,
      y2: maxY
    };
  }, [lassoStrokes]);
  
  const currentFrame = useMemo(() => {
    if (clip.type === 'image') {
      return 0;
    }
    return focusFrame - startFrame + (framesToGiveStart || 0);
  }, [focusFrame, startFrame, framesToGiveStart, clip.type]);

  const { data, loading } = useMask({
    inputPath: clip.src,
    tool: 'touch',
    points: touchDrawMode === 'point' && points.length > 0 ? points : undefined, // Only send points if we have touch points
    pointLabels: touchDrawMode === 'point' && points.length > 0 ? pointLabels : undefined,
    box: touchDrawMode === 'draw' && lassoBox ? lassoBox : undefined, // Only send box if we have lasso strokes and no touch points
    frameNumber: currentFrame,
    displayWidth: rectWidth,
    displayHeight: rectHeight,
    mediaInfo: mediaInfo || undefined,
    clipTransform: clipTransform || undefined,
    enabled: ((touchPoints && touchPoints.length > 0) || (lassoStrokes && lassoStrokes.length > 0)) && !!mediaInfo,
  });

  // Update rendered touch points and contours when mask data is received and store in mask keyframes
  useEffect(() => {
    if (data?.contours && data.contours.length > 0 && !loading) {
      setContours(data.contours);
      setRenderedTouchPoints(touchPoints || []);
      
      // Check if we've already stored this exact data
      const dataKey = `${clip.clipId}-${focusFrame}-${JSON.stringify(data.contours)}-${JSON.stringify(touchPoints)}-${JSON.stringify(lassoStrokes)}`;
      if (lastStoredDataRef.current === dataKey) {
        return; // Already stored, skip
      }
      
      try {
        // Store contours, touch points, and lasso strokes in mask keyframes
        const masks = (clip as any).masks || [];
        const currentMask = masks.find((m: any) => m.tool === 'touch');
        
        if (currentMask && currentMask.keyframes) {
          const keyframes = currentMask.keyframes instanceof Map 
            ? currentMask.keyframes 
            : (currentMask.keyframes as Record<number, any>);
          
          if (!keyframes) return;
          
          const keyframeNumbers = keyframes instanceof Map
            ? Array.from(keyframes.keys()).sort((a, b) => a - b)
            : Object.keys(keyframes).map(Number).sort((a, b) => a - b);
          
          const activeKeyframe = keyframeNumbers.filter((k: number) => k <= focusFrame).pop();
          
          if (activeKeyframe !== undefined) {
            const maskData = keyframes instanceof Map 
              ? keyframes.get(activeKeyframe) 
              : keyframes[activeKeyframe];
            
            // Only update if any data has actually changed
            const existingContours = maskData?.contours;
            const existingTouchPoints = maskData?.touchPoints;
            const existingLassoStrokes = maskData?.lassoStrokes;
            
            const contoursChanged = !existingContours || 
              JSON.stringify(existingContours) !== JSON.stringify(data.contours);
            const touchPointsChanged = !existingTouchPoints ||
              JSON.stringify(existingTouchPoints) !== JSON.stringify(touchPoints);
            const lassoStrokesChanged = !existingLassoStrokes ||
              JSON.stringify(existingLassoStrokes) !== JSON.stringify(lassoStrokes);
            
            if (contoursChanged || touchPointsChanged || lassoStrokesChanged) {
              const updatedKeyframes = keyframes instanceof Map
                ? new Map(keyframes)
                : { ...keyframes };
              
              if (updatedKeyframes instanceof Map) {
                updatedKeyframes.set(activeKeyframe, {
                  ...maskData,
                  contours: data.contours,
                  touchPoints: touchPoints || [],
                  lassoStrokes: lassoStrokes || [],
                });
              } else {
                updatedKeyframes[activeKeyframe] = {
                  ...maskData,
                  contours: data.contours,
                  touchPoints: touchPoints || [],
                  lassoStrokes: lassoStrokes || [],
                };
              }
              
              const { clips } = useClipStore.getState();
              const targetClip = clips.find(c => c.clipId === clip.clipId);
              
              if (targetClip && clip.clipId) {
                const currentMasks = (targetClip as any).masks || [];
                const updatedMasks = currentMasks.map((m: any) =>
                  m.id === currentMask.id ? { ...m, keyframes: updatedKeyframes } : m
                );
                
                useClipStore.getState().updateClip(clip.clipId, { masks: updatedMasks });
                
                // Mark this data as stored
                lastStoredDataRef.current = dataKey;
              }
            }
          }
        }
      } catch (error) {
        console.error('Error storing touch mask contours:', error);
      }
    }
  }, [data, loading, touchPoints, lassoStrokes, clip.clipId, focusFrame]);

  // Calculate non-rendered touch points (points without contours yet)
  const nonRenderedTouchPoints = useMemo(() => {
    if (renderedTouchPoints.length === 0) return touchPoints || [];
    if (contours.length > 0) return [];
    return touchPoints || [];
  }, [touchPoints, renderedTouchPoints, contours]);
  return (
    <Group clipX={0} clipY={0} clipWidth={rectWidth} clipHeight={rectHeight}>
      {/* Render contours with zebra stripes (generated from lasso strokes and touch points) */}
      {contours.map((contour, index) => (
        <Group key={`contour-${index}`}>
          {/* White stripe background */}
          <Line
            points={contour}
            stroke="#ffffff"
            strokeWidth={1}
            lineCap="round"
            lineJoin="round"
            closed={true}
            listening={false}
            fill="rgba(0, 127, 245, 0.4)"
          />
          
          {/* Black stripe foreground with animation */}
          <Line
            points={contour}
            stroke="#000000"
            strokeWidth={1}
            dash={[4.5, 4.5]}
            dashOffset={-animationOffset}
            lineCap="round"
            lineJoin="round"
            closed={true}
            listening={false}
          />
        </Group>
      ))}
      
      {/* Render rendered touch points (points that have contours) */}
      {contours.length > 0 && touchDrawMode === 'point' && renderedTouchPoints.map((point, index) => {
        const isPositive = point.label === 'positive';
        const color = isPositive ? '#3b82f6' : '#ef4444'; // blue-500 : red-500
        const radius = 8;
        const iconSize = 6;
        
        return (
          <Group key={`rendered-touch-point-${index}`} x={point.x} y={point.y}>
            {/* Outer glow */}
            <Circle
              radius={radius + 4}
              fill={color}
              opacity={0.2}
              listening={false}
            />
            {/* Main circle */}
            <Circle
              radius={radius}
              fill={color}
              opacity={0.9}
              listening={false}
            />
            {/* Icon - Plus or X */}
            {isPositive ? (
              // Plus icon (horizontal and vertical lines)
              <>
                <Line
                  points={[-iconSize / 2, 0, iconSize / 2, 0]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
                <Line
                  points={[0, -iconSize / 2, 0, iconSize / 2]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
              </>
            ) : (
              // X icon (diagonal lines)
              <>
                <Line
                  points={[-iconSize / 2, -iconSize / 2, iconSize / 2, iconSize / 2]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
                <Line
                  points={[iconSize / 2, -iconSize / 2, -iconSize / 2, iconSize / 2]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
              </>
            )}
          </Group>
        );
      })}
      
      {/* Render non-rendered touch points (points without contours yet) */}
      {nonRenderedTouchPoints.map((point, index) => {
        const isPositive = point.label === 'positive';
        const color = isPositive ? '#3b82f6' : '#ef4444'; // blue-500 : red-500
        const radius = 8;
        const iconSize = 6;
        
        return (
          <Group key={`non-rendered-touch-point-${index}`} x={point.x} y={point.y}>
            {/* Outer glow */}
            <Circle
              radius={radius + 4}
              fill={color}
              opacity={0.2}
              listening={false}
            />
            {/* Main circle */}
            <Circle
              radius={radius}
              fill={color}
              opacity={0.9}
              listening={false}
            />
            {/* Icon - Plus or X */}
            {isPositive ? (
              // Plus icon (horizontal and vertical lines)
              <>
                <Line
                  points={[-iconSize / 2, 0, iconSize / 2, 0]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
                <Line
                  points={[0, -iconSize / 2, 0, iconSize / 2]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
              </>
            ) : (
              // X icon (diagonal lines)
              <>
                <Line
                  points={[-iconSize / 2, -iconSize / 2, iconSize / 2, iconSize / 2]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
                <Line
                  points={[iconSize / 2, -iconSize / 2, -iconSize / 2, iconSize / 2]}
                  stroke="#ffffff"
                  strokeWidth={2}
                  lineCap="round"
                  listening={false}
                />
              </>
            )}
          </Group>
        );
      })}
    </Group>
  );
};

export default TouchMaskPreview;

