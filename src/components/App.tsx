import React, { useEffect, useRef } from "react";
import "../styles/index.css";
import TimelineEditor from "./timeline/TimelineEditor";
import { GrMultimedia } from "react-icons/gr";

import Controls from "./controls/Controls";
import TopBar from "./bars/TopBar";
import Preview from "./preview/Preview";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable"

const App:React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  return (
    <main ref={containerRef} className="w-full text-center font-poppins bg-brand-background h-screen flex flex-col">
      <TopBar title="Apex Studio" />
      <ResizablePanelGroup direction="vertical">
      <ResizablePanel defaultSize={70} minSize={40} className="flex flex-col w-full bg-brand  flex-1 overflow-hidden">
        <Preview />
      </ResizablePanel>
      <ResizableHandle className="bg-brand-light/10" />
      <ResizablePanel defaultSize={30} minSize={30} className="bg-brand flex flex-col border-t-0 border shadow w-full border-brand-light/10 flex-shrink-0 overflow-x-hidden">
      <Controls />
      <div className="flex flex-col gap-y-2.5 justify-center h-full overflow-auto">
      <TimelineEditor icon={<GrMultimedia className="h-4 w-4" />} name="models and content" />
      </div>
      </ResizablePanel>
      </ResizablePanelGroup>
    </main>
  );
}

export default App;
