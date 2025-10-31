import React from 'react'
import { useClipStore } from '@/lib/clip';
import { ModelClipProps } from '@/lib/types';

interface ModelGenerationPropertiesProps {
  clipId: string;
}

export const ModelGenerationProperties: React.FC<ModelGenerationPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;
  return (
    <div className="flex flex-col gap-y-2">
      <div className="flex flex-row items-center justify-between">
        <span className="text-brand-light text-[10px]">Generations</span>
      </div>
    </div>
  )
}