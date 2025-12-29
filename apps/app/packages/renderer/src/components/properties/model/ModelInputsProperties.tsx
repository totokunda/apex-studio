import React from "react";
import { useClipStore } from "@/lib/clip";
import { ModelClipProps } from "@/lib/types";
import { ModelInputsPanel } from "./ModelInputsPanel";
import { UIPanel } from "@/lib/manifest/api";
import { InputControlsProvider } from "@/lib/inputControl";

interface ModelInputsPropertiesProps {
  clipId: string;
  panelSize: number;
}

export const ModelInputsProperties: React.FC<ModelInputsPropertiesProps> = ({
  clipId,
  panelSize,
}) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as ModelClipProps;

  return (
    <InputControlsProvider clipId={clipId}>
    <div className="flex flex-col gap-y-2">
      <div className="text-brand-light text-[10px] flex flex-col divide-y divide-brand-light/5">
        {(() => {
          const basePanels = (clip.manifest?.spec?.ui?.panels ||
            []) as UIPanel[];
          const components = clip.manifest?.spec?.components || [];
          const schedulerOptions = components
            .filter((c: any) => String(c?.type) === "scheduler")
            .flatMap((c: any) =>
              Array.isArray(c?.scheduler_options) ? c.scheduler_options : [],
            );
          const hasSchedulerOptions =
            schedulerOptions && schedulerOptions.length > 0;
          const alreadyHasSchedulerPanel = basePanels.some(
            (p) => String(p?.name || "").toLowerCase() === "scheduler",
          );
          let panelsToRender =
            hasSchedulerOptions && !alreadyHasSchedulerPanel
              ? [
                  ...basePanels,
                  {
                    name: "scheduler",
                    label: "Scheduler",
                    collapsible: true,
                    default_open: false,
                    layout: { flow: "column", rows: [] },
                  } as UIPanel,
                ]
              : basePanels;

          // Append Attention panel at the very end if options exist and panel not present
          const attentionOptions = (clip.manifest?.spec
            ?.attention_types_detail || []) as any[];
          const hasAttentionOptions =
            Array.isArray(attentionOptions) && attentionOptions.length > 0;
          const alreadyHasAttentionPanel = panelsToRender.some(
            (p) => String(p?.name || "").toLowerCase() === "attention",
          );
          if (hasAttentionOptions && !alreadyHasAttentionPanel) {
            panelsToRender = [
              ...panelsToRender,
              {
                name: "attention",
                label: "Attention",
                collapsible: true,
                default_open: false,
                layout: { flow: "column", rows: [] },
              } as UIPanel,
            ];
          }
          return panelsToRender.map((panel) => {
            return (
              <ModelInputsPanel
                key={panel.name}
                panel={panel}
                inputs={clip.manifest?.spec?.ui?.inputs || []}
                clipId={clipId}
                panelSize={panelSize}
              />
            );
          });
        })()}
      </div>
    </div>
    </InputControlsProvider>
  );
};
