import React from 'react'
import { useClipStore } from '@/lib/clip';
import { ModelClipProps } from '@/lib/types';

interface ModelHistoryPropertiesProps {
  clipId: string;
}

export const ModelHistoryProperties: React.FC<ModelHistoryPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;
  return (
    <div className="flex flex-col gap-y-2">
      <div className="flex flex-row items-center justify-between">
        <span className="text-brand-light text-[10px]">History</span>
      </div>
    </div>
  )
}