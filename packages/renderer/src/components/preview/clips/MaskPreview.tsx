import React, { useEffect, useRef, useState } from 'react';
import { Group } from 'react-konva';
import { getLocalFrame, useClipStore } from '@/lib/clip';
import { useControlsStore } from '@/lib/control';
import { AnyClipProps, MaskClipProps, MaskData, MaskShapeTool, PreprocessorClipType } from '@/lib/types';
import LassoMaskPreview from './LassoMaskPreview';
import ShapeMaskPreview from './ShapeMaskPreview';
import DrawMaskPreview from './DrawMaskPreview';
import TouchMaskPreview from './TouchMaskPreview';
import Konva from 'konva';
import { useViewportStore } from '@/lib/viewport';
import { useMaskStore } from '@/lib/mask';

interface MaskPreviewProps {
  clips: AnyClipProps[];
  sortClips: (clips: AnyClipProps[]) => AnyClipProps[];
  filterClips: (clips: AnyClipProps[], audio?: boolean) => AnyClipProps[];
  rectWidth: number;
  rectHeight: number;
}

interface MaskRenderData {
  mask: MaskClipProps;
  maskData: MaskData;
  activeKeyframe: number;
}

const MaskPreview: React.FC<MaskPreviewProps> = ({ clips, sortClips, filterClips, rectWidth, rectHeight }) => {
  const { clipWithinFrame } = useClipStore();
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const [animationOffset, setAnimationOffset] = useState(0);
  const [masksToRender, setMasksToRender] = useState<MaskRenderData[]>([]);
  const [hasMasks, setHasMasks] = useState(false);
  const tool = useViewportStore((s) => s.tool);
  const setIsOverMask = useMaskStore((s) => s.setIsOverMask);
  const isOverMask = useMaskStore((s) => s.isOverMask);
  const updateClip = useClipStore((s) => s.updateClip);

  const ref = useRef<Konva.Group>(null);
  const prevMaskIdsRef = useRef<Set<string>>(new Set());
  // get the clip for the mask 
  

  // Compute which masks should be visible
  useEffect(() => {
    const masksData: MaskRenderData[] = [];
    let foundMasks = false;

    sortClips(filterClips(clips)).forEach((clip) => {
      if ((clip.type !== 'video' && clip.type !== 'image')) return;

      // Only render masks when the parent clip is active at the current focus frame
      const clipIsActive = clipWithinFrame(clip, focusFrame);
      if (!clipIsActive) return;

      const masks = (clip as any).masks || [];

      masks.forEach((mask: MaskClipProps) => {
          // Handle both Map and Record types for keyframes
          const keyframes = mask.keyframes instanceof Map 
            ? mask.keyframes 
            : (mask.keyframes as Record<number, any>);

          const keyframeNumbers = keyframes instanceof Map
            ? Array.from(keyframes.keys()).map(Number).sort((a, b) => a - b)
            : Object.keys(keyframes).map(Number).sort((a, b) => a - b);

          if (keyframeNumbers.length === 0) return;

          // Calculate local frame relative to this clip (aligns with how mask keyframes are stored)
          const startFrame = clip.startFrame ?? 0;
          const framesToGiveStart = isFinite(clip.framesToGiveStart ?? 0) ? (clip.framesToGiveStart ?? 0) : 0;
          const realStartFrame = startFrame + framesToGiveStart;
          const localFrame = focusFrame - realStartFrame;

          // Helper: nearest keyframe selection (same behavior as shape.ts)
          const nearestKeyframe = (frame: number) => {
            if (frame < keyframeNumbers[0]) return keyframeNumbers[0];
            const atOrBefore = keyframeNumbers.filter((k) => k <= frame).pop();
            return atOrBefore ?? keyframeNumbers[keyframeNumbers.length - 1];
          };

          // Prefer localFrame for video; for image it should render for the entire clip duration
          // Fallback to focusFrame if keys were stored globally
          const candidateLocal = nearestKeyframe(localFrame);
          const candidateGlobal = nearestKeyframe(focusFrame);
          const activeKeyframe = clip.type === 'video' ? (candidateLocal ?? candidateGlobal) : 0;


          if (activeKeyframe !== undefined) {
            const maskData = keyframes instanceof Map 
              ? keyframes.get(activeKeyframe) 
              : keyframes[activeKeyframe];

            if (maskData) {
              masksData.push({
                mask,
                maskData,
                activeKeyframe,
              });
              foundMasks = true;
            }
          }
      });
    });

    setMasksToRender(masksData);
    setHasMasks(foundMasks);
    
  }, [clips, focusFrame, sortClips, filterClips, clipWithinFrame]);

  // Animate zebra stripe effect
  useEffect(() => {
    if (!hasMasks) {
      setAnimationOffset(0);
      return;
    }

    let animationFrameId: number;
    const animate = () => {
      setAnimationOffset(prev => (prev + 0.5) % 20);
      animationFrameId = requestAnimationFrame(animate);
    };

    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, [hasMasks]);

  // Reset hover state when masks are deleted while hovering
  useEffect(() => {
    // Create a set of currently rendered mask IDs
    const currentMaskIds = new Set(masksToRender.map(({ mask }) => mask.id));
    
    // If isOverMask is true but no masks are being rendered, reset the hover state
    if (isOverMask && currentMaskIds.size === 0) {
      setIsOverMask(false);
    }
    
    // If a mask was removed while being hovered (comparing with previous render)
    // Check if any previously rendered mask is now missing
    if (isOverMask && prevMaskIdsRef.current.size > 0) {
      const hadMaskRemoved = Array.from(prevMaskIdsRef.current).some(
        (prevId) => !currentMaskIds.has(prevId)
      );
      
      // If a mask was removed and we're in mask tool mode, reset hover state
      // This ensures cursor returns to crosshair
      if (hadMaskRemoved && tool === 'mask') {
        setIsOverMask(false);
      }
    }
    
    // Update the ref for next render
    prevMaskIdsRef.current = currentMaskIds;
  }, [masksToRender, isOverMask, setIsOverMask, tool]);

  if (tool !== 'mask') return null;

  return (
    <Group ref={ref}>
      {masksToRender.map(({ mask, maskData, activeKeyframe }) => {

        const clip = clips.find((c) => c.clipId === mask.clipId);
        if (!clip) return null;
        const localFrame = getLocalFrame(focusFrame, clip);
        if (mask.tool === 'lasso' && maskData?.lassoPoints && maskData.lassoPoints.length >= 6) {
          // Close the path
          const closedPoints = [...maskData.lassoPoints, maskData.lassoPoints[0], maskData.lassoPoints[1]];
          
          return (
            <LassoMaskPreview
              key={`mask-${mask.id}-${activeKeyframe}`}
              mask={mask}
              points={closedPoints}
              animationOffset={animationOffset}
              rectWidth={rectWidth}
              rectHeight={rectHeight}
            />
          );
        } else if ((mask.tool === 'shape' || (mask.tool as any) === 'rectangle') && (maskData?.shapeBounds)) {
          // Support both new shapeBounds and legacy rectangleBounds (for backward compatibility)
          const bounds = maskData.shapeBounds
          const shapeType: MaskShapeTool = bounds?.shapeType || 'rectangle';
          const clipTransform = clip?.transform
          
          return (
            <Group key={`mask-group-${mask.id}-${activeKeyframe}`} scaleX={clipTransform?.scaleX ?? 1} scaleY={clipTransform?.scaleY ?? 1} rotation={clipTransform?.rotation ?? 0}>
            <ShapeMaskPreview
              key={`mask-${mask.id}-${activeKeyframe}`}
              mask={mask}
              x={bounds.x}
              y={bounds.y}
              width={bounds.width}
              height={bounds.height}
              rotation={bounds.rotation ?? 0}
              shapeType={shapeType}
              scaleX={bounds.scaleX ?? 1}
              scaleY={bounds.scaleY ?? 1}
              animationOffset={animationOffset}
              rectWidth={rectWidth}
              rectHeight={rectHeight}
            />
            </Group>
          );
        } else if (mask.tool === 'draw' && maskData?.drawStrokes) {
          return (
            <DrawMaskPreview
              key={`mask-${mask.id}-${activeKeyframe}`}
              mask={mask}
              drawStrokes={maskData.drawStrokes}
              animationOffset={animationOffset}
              rectWidth={rectWidth}
              rectHeight={rectHeight}
            />
          );
          } else if (mask.tool === 'touch' && (maskData?.touchPoints)) {

            const activeDataKeyPoints = localFrame === activeKeyframe || clip.type === 'image' ? maskData.touchPoints : [];

            const handleDeletePoints = (pointsToDelete: Array<{ x: number; y: number; label: 1 | 0 }>) => {
            const updatedTouchPoints = (maskData.touchPoints || []).filter((point: { x: number; y: number; label: 1 | 0 }) => {
              return !pointsToDelete.some(p => p.x === point.x && p.y === point.y && p.label === point.label);
            });

            const targetClip = clips.find(c => c.clipId === mask.clipId);
            
            if (targetClip && mask.clipId) {
              const currentMasks = (targetClip as PreprocessorClipType).masks || [];
              
              // If no points remain and no lasso strokes, remove the mask entirely
              if (updatedTouchPoints.length === 0 ) {
                const updatedMasks = currentMasks.filter((m) => m.id !== mask.id);
                updateClip(mask.clipId, { masks: updatedMasks });
                // Also clear the selected mask if this was selected
                const { setSelectedMaskId, selectedMaskId } = useControlsStore.getState();
                if (selectedMaskId === mask.id) {
                  setSelectedMaskId(null);
                }
              } else {
                // Update the mask keyframes
                const keyframes = mask.keyframes instanceof Map ? mask.keyframes : (mask.keyframes as Record<number, any>);
                const updatedKeyframes = keyframes instanceof Map ? new Map(keyframes) : { ...keyframes };

                if (updatedKeyframes instanceof Map) {
                  updatedKeyframes.set(activeKeyframe, { ...maskData, touchPoints: updatedTouchPoints });
                } else {
                  updatedKeyframes[activeKeyframe] = { ...maskData, touchPoints: updatedTouchPoints };
                }

                const updatedMasks = currentMasks.map((m: any) =>
                  m.id === mask.id ? { ...m, keyframes: updatedKeyframes } : m
                );
                updateClip(mask.clipId, { masks: updatedMasks });
              }
            }
          };

          return (
            <TouchMaskPreview     
              clip={clip as PreprocessorClipType}
              key={`mask-${mask.id}-${activeKeyframe}`}
              touchPoints={activeDataKeyPoints}
              animationOffset={animationOffset}
              rectWidth={rectWidth}
              rectHeight={rectHeight}
              onDeletePoints={handleDeletePoints}
            />
          );
        }
        return null;
      })}
    </Group>
  );
};

export default MaskPreview;

