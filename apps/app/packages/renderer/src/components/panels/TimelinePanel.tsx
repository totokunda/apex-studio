import React from "react";
import { ResizablePanel } from "../ui/resizable";
import Controls from "../controls/Controls";
import TimelineEditor from "../timeline/TimelineEditor";

const TimelinePanel: React.FC = () => {
  return (
    <ResizablePanel
      defaultSize={30}
      minSize={30}
      className="bg-brand flex flex-col  rounded-lg shadow w-full border-brand-light/10 flex-shrink-0"
    >
      <Controls />
      <div className="flex flex-col gap-y-2.5 justify-center h-full overflow-auto">
        <TimelineEditor />
      </div>
    </ResizablePanel>
  );
};

export default TimelinePanel;
