import React from "react";
import Preview from "../preview/Preview";
import { ResizablePanel } from "../ui/resizable";
import FloatingBar from "../preview/clips/FloatingBar";

interface PreviewPanelProps {
  order?: number;
  defaultSize?: number;
  minSize?: number;
}

const PreviewPanel: React.FC<PreviewPanelProps> = ({
  order,
  defaultSize = 70,
  minSize = 40,
}) => {
  return (
    <ResizablePanel
      defaultSize={defaultSize}
      minSize={minSize}
      order={order}
      className="flex flex-col w-full bg-brand relative flex-1 overflow-hidden rounded-lg"
    >
      <FloatingBar />
      <Preview />
    </ResizablePanel>
  );
};

export default PreviewPanel;
