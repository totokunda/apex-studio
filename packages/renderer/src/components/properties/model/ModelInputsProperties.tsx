import React, { useEffect } from 'react'
import { useClipStore } from '@/lib/clip';
import { ModelClipProps } from '@/lib/types';
import { ModelInputsPanel } from './ModelInputsPanel';

interface ModelInputsPropertiesProps {
  clipId: string;
}

export const ModelInputsProperties: React.FC<ModelInputsPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;

  return (
    <div className="flex flex-col gap-y-2 ">
        <div className="text-brand-light text-[10px] flex flex-col gap-y-2.5 p-2.5">
          {clip.manifest?.spec?.ui?.panels?.map((panel) => {
            return (
              <ModelInputsPanel key={panel.name} panel={panel} inputs={clip.manifest?.spec?.ui?.inputs || []} clipId={clipId} />
            )
          })}
        </div>
    </div>
  )
}