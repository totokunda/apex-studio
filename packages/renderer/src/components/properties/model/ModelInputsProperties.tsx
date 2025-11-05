import React from 'react'
import { useClipStore } from '@/lib/clip';
import { ModelClipProps } from '@/lib/types';
import { ModelInputsPanel } from './ModelInputsPanel';
import { UIPanel } from '@/lib/manifest/api';

interface ModelInputsPropertiesProps {
  clipId: string;
  panelSize: number;
}

export const ModelInputsProperties: React.FC<ModelInputsPropertiesProps> = ({ clipId, panelSize }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;

  return (
    <div className="flex flex-col gap-y-2">
        <div className="text-brand-light text-[10px] flex flex-col divide-y divide-brand-light/5">
          {(() => {
            const basePanels = (clip.manifest?.spec?.ui?.panels || []) as UIPanel[];
            const components = clip.manifest?.spec?.components || [];
            const schedulerOptions = components
              .filter((c: any) => String(c?.type) === 'scheduler')
              .flatMap((c: any) => Array.isArray(c?.scheduler_options) ? c.scheduler_options : []);
            const hasSchedulerOptions = schedulerOptions && schedulerOptions.length > 0;
            const alreadyHasSchedulerPanel = basePanels.some((p) => String(p?.name || '').toLowerCase() === 'scheduler');
            const panelsToRender = hasSchedulerOptions && !alreadyHasSchedulerPanel
              ? [...basePanels, { name: 'scheduler', label: 'Scheduler', collapsible: true, default_open: false, layout: { flow: 'column', rows: [] } } as UIPanel]
              : basePanels;
            return panelsToRender.map((panel) => {
            return (
              <ModelInputsPanel key={panel.name} panel={panel} inputs={clip.manifest?.spec?.ui?.inputs || []} clipId={clipId} panelSize={panelSize} />
            )
            })
          })()}
        </div>
    </div>
  )
}