import React, { useRef, useEffect, useState } from "react";
import { ResizablePanel } from "../ui/resizable";
import EmptyPropertiesPanel from "../properties/EmptyPropertiesPanel";
import ClipPropertiesPanel from "./ClipPropertiesPanel";
import { useControlsStore } from "@/lib/control";
import { ImperativePanelHandle } from "react-resizable-panels";
import { useClipStore } from "@/lib/clip";

interface PropertiesPanelProps {
  order?: number;
  defaultSize?: number;
  minSize?: number;
}

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({
  order,
  defaultSize = 25,
  minSize = 25,
}) => {
  const { selectedClipIds } = useControlsStore();
  const { selectedPreprocessorId } = useClipStore();
  const [panelSize, setPanelSize] = useState(0);
  const panelRef = useRef<ImperativePanelHandle>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = entry.contentRect.width;
        setPanelSize(width);
      }
    });

    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <ResizablePanel
      ref={panelRef}
      minSize={minSize}
      defaultSize={defaultSize}
      order={order}
      className="bg-brand flex flex-col rounded-lg shadow border-brand-light/10 min-w-0"
    >
      <div
        ref={containerRef}
        className="bg-brand-background rounded-lg h-full overflow-hidden min-w-0"
      >
        {selectedClipIds.length > 0 || selectedPreprocessorId ? (
          <ClipPropertiesPanel panelSize={panelSize} />
        ) : (
          <EmptyPropertiesPanel />
        )}
      </div>
    </ResizablePanel>
  );
};

export default PropertiesPanel;
