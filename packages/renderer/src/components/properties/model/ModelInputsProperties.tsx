import React from 'react'
import { useClipStore } from '@/lib/clip';
import { ModelClipProps } from '@/lib/types';

interface ModelInputsPropertiesProps {
  clipId: string;
}

export const ModelInputsProperties: React.FC<ModelInputsPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;
  return (
    <div className="flex flex-col gap-y-2">
      <div className="flex flex-row items-center justify-between">
        <span className="text-brand-light text-[10px]">Inputs</span>
      </div>
    </div>
  )
}