import { useClipStore } from '@/lib/clip';
import { AnyClipProps } from '@/lib/types';
import { useState } from 'react';
import React from 'react'
import { IoRefreshOutline, IoLockClosed, IoLockOpen } from 'react-icons/io5';
import Input from './Input';

interface LayoutPropertiesProps {
  clipId: string
}

const LayoutProperties: React.FC<LayoutPropertiesProps> = ({ clipId }) => {
    const clip = useClipStore((s) => s.getClipById(clipId)) as AnyClipProps;
    const setClipTransform = useClipStore((s) => s.setClipTransform);

    const [spinning, setSpinning] = useState(false);
    const [scaleLocked, setScaleLocked] = useState(true);

    const handleReset = () => {
      if (!clip?.transform) return;
      
      setClipTransform(clipId, { 
        width: clip.transform.width,
        height: clip.transform.height,
        scaleX: 1, 
        scaleY: 1 
      });
      
      setSpinning(true);
      setTimeout(() => {
        setSpinning(false);
      }, 500);
    };

    const handleWidthChange = (value: string) => {
      if (!clip?.transform) return;
      const numValue = Number(value);
      if (isNaN(numValue) || !isFinite(numValue) || numValue <= 0) return;
      
      setClipTransform(clipId, { width: numValue });
    };

    const handleHeightChange = (value: string) => {
      if (!clip?.transform) return;
      const numValue = Number(value);
      if (isNaN(numValue) || !isFinite(numValue) || numValue <= 0) return;
      
      setClipTransform(clipId, { height: numValue });
    };

    const handleScaleXChange = (value: string) => {
      if (!clip?.transform) return;
      const numValue = Number(value);
      if (isNaN(numValue) || !isFinite(numValue)) return;
      
      if (scaleLocked) {
        setClipTransform(clipId, { scaleX: numValue, scaleY: numValue });
      } else {
        setClipTransform(clipId, { scaleX: numValue });
      }
    };

    const handleScaleYChange = (value: string) => {
      if (!clip?.transform) return;
      const numValue = Number(value);
      if (isNaN(numValue) || !isFinite(numValue)) return;
      
      if (scaleLocked) {
        setClipTransform(clipId, { scaleX: numValue, scaleY: numValue });
      } else {
        setClipTransform(clipId, { scaleY: numValue });
      }
    };

  return (
    <div className="flex flex-col gap-y-2">
      <div className="p-4 flex flex-col gap-y-4 px-5">
        <div className="flex flex-row items-center justify-between">
          <h4 className="text-brand-light text-[12px] font-medium text-start">Layout</h4>
          <span
            onClick={handleReset}
            className="text-brand-light text-sm cursor-pointer"
          >
            <IoRefreshOutline
              className={spinning ? "animate-spin duration-500" : ""}
              onAnimationEnd={() => setSpinning(false)}
            />
          </span>
        </div>
        
        <div className="flex flex-col gap-y-3">
          <div className="flex flex-row gap-x-2">
            <Input 
              label="Size" 
              value={clip?.transform?.width.toFixed(0).toString() ?? '0'} 
              onChange={handleWidthChange} 
              startLogo="W" 
            />
            <Input 
              emptyLabel 
              value={clip?.transform?.height.toFixed(0).toString() ?? '0'} 
              onChange={handleHeightChange} 
              startLogo="H" 
            />
          </div>
          
          <div className="flex flex-col ">
            <div className="flex flex-row items-center justify-between">
              <span className="text-brand-light text-[10px] text-start">Scale</span>
              <button
                onClick={() => setScaleLocked(!scaleLocked)}
                className="text-brand-light/70 hover:text-brand-light transition-colors cursor-pointer"
              >
                {scaleLocked ? (
                  <IoLockClosed className="h-3 w-3" />
                ) : (
                  <IoLockOpen className="h-3 w-3" />
                )}
              </button>
            </div>
            <div className="flex flex-row gap-x-2">
              <Input 
                value={clip?.transform?.scaleX.toFixed(2).toString() ?? '1.00'} 
                onChange={handleScaleXChange} 
                startLogo="X"
                canStep
                step={0.1}
                min={0.1}
              />
              <Input 
                value={clip?.transform?.scaleY.toFixed(2).toString() ?? '1.00'} 
                onChange={handleScaleYChange} 
                startLogo="Y"
                canStep
                step={0.1}
                min={0.1}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LayoutProperties

