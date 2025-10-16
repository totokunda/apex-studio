import React, { useEffect, useState } from 'react';
import { Group } from 'react-konva';
import { useClipStore } from '@/lib/clip';
import { useControlsStore } from '@/lib/control';
import { AnyClipProps, MaskClipProps, MaskShapeTool, PreprocessorClipType } from '@/lib/types';
import LassoMaskPreview from './LassoMaskPreview';
import ShapeMaskPreview from './ShapeMaskPreview';
import DrawMaskPreview from './DrawMaskPreview';
import TouchMaskPreview from './TouchMaskPreview';

interface MaskPreviewProps {
  clips: AnyClipProps[];
  sortClips: (clips: AnyClipProps[]) => AnyClipProps[];
  filterClips: (clips: AnyClipProps[], audio?: boolean) => AnyClipProps[];
  rectWidth: number;
  rectHeight: number;
}

interface MaskRenderData {
  mask: MaskClipProps;
  maskData: any;
  activeKeyframe: number;
}

const MaskPreview: React.FC<MaskPreviewProps> = ({ clips, sortClips, filterClips, rectWidth, rectHeight }) => {
  const { clipWithinFrame } = useClipStore();
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const [animationOffset, setAnimationOffset] = useState(0);
  const [masksToRender, setMasksToRender] = useState<MaskRenderData[]>([]);
  const [hasMasks, setHasMasks] = useState(false);
  
  // get the clip for the mask 
  

  // Compute which masks should be visible
  useEffect(() => {
    const masksData: MaskRenderData[] = [];
    let foundMasks = false;

    sortClips(filterClips(clips)).forEach((clip) => {
      const clipAtFrame = clipWithinFrame(clip, focusFrame);
      if (!clipAtFrame || (clip.type !== 'video' && clip.type !== 'image')) return;

      const masks = (clip as any).masks || [];

      masks.forEach((mask: MaskClipProps) => {
        const maskStart = mask.startFrame ?? 0;
        const maskEnd = mask.endFrame ?? 0;

        if (focusFrame >= maskStart && focusFrame <= maskEnd) {
          foundMasks = true;

          // Handle both Map and Record types for keyframes
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

            if (maskData) {
              masksData.push({
                mask,
                maskData,
                activeKeyframe,
              });
            }
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

  return (
    <Group>
      {masksToRender.map(({ mask, maskData, activeKeyframe }) => {
        const clip = clips.find((c) => c.clipId === mask.clipId);
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
        } else if ((mask.tool === 'shape' || (mask.tool as any) === 'rectangle') && (maskData?.shapeBounds || maskData?.rectangleBounds)) {
          // Support both new shapeBounds and legacy rectangleBounds (for backward compatibility)
          const bounds = maskData.shapeBounds || maskData.rectangleBounds;
          const shapeType: MaskShapeTool = bounds?.shapeType || 'rectangle';
          
          return (
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
        } else if (mask.tool === 'touch' && (maskData?.touchPoints || maskData?.lassoStrokes)) {
          return (
            <TouchMaskPreview     
              clip={clip as PreprocessorClipType}
              key={`mask-${mask.id}-${activeKeyframe}`}
              touchPoints={maskData.touchPoints}
              lassoStrokes={maskData.lassoStrokes}
              animationOffset={animationOffset}
              rectWidth={rectWidth}
              rectHeight={rectHeight}
            />
          );
        }
        
        return null;
      })}
    </Group>
  );
};

export default MaskPreview;

