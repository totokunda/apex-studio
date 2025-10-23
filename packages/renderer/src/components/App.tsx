import React, { useRef, useEffect, useState } from "react";
import "../styles/index.css";

import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable"
import { Toaster } from "@/components/ui/sonner"
import MediaModelPanel from "./panels/MediaModelPanel";
import PreviewPanel from "./panels/PreviewPanel";
import TimelinePanel from "./panels/TimelinePanel";
import PropertiesPanel from "./panels/PropertiesPanel";
import { useLayoutConfigStore } from "@/lib/layout-config";
import Topbar from "./bars/Topbar";
import { DndContext, DragEndEvent, DragOverlay } from "@dnd-kit/core";
import { MediaItem, MediaThumb } from "@/components/media/Item";
import { cn } from "@/lib/utils";
import { useClipStore } from "@/lib/clip";
import { Preprocessor } from "@/lib/preprocessor/api";
import { PreprocessorItem } from "./menus/PreprocessorMenu";
import { ModelItem } from "./menus/ModelMenu";
import type { ManifestInfo } from "@/lib/manifest";
import GlobalContextMenu from "@/components/GlobalContextMenu";

type ManifestInfoWithType = ManifestInfo & {
  type: 'model';
};

const App:React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const layout = useLayoutConfigStore((s) => s.layout);
  const {ghostInStage, clips} = useClipStore();

  const [activeDragItem, setActiveDragItem] = useState<MediaItem | Preprocessor | ManifestInfoWithType | null>(null);

  // Disable scrolling while dragging
  useEffect(() => {
    if (!activeDragItem) return;
    const preventDefault = (e: Event) => { e.preventDefault(); };
    const preventKeyScroll = (e: KeyboardEvent) => {
      const blockedKeys = ['ArrowUp','ArrowDown','PageUp','PageDown','Home','End',' ','Spacebar'];
      if (blockedKeys.includes(e.key) || e.code === 'Space') {
        e.preventDefault();
      }
    };
    // Best-effort: also hide body scrollbars and disable touch panning/overscroll
    const prevOverflow = document.body.style.overflow;
    const prevTouchAction = (document.body.style as any).touchAction;
    const prevOverscroll = (document.body.style as any).overscrollBehavior;
    document.body.style.overflow = 'hidden';
    (document.body.style as any).touchAction = 'none';
    (document.body.style as any).overscrollBehavior = 'none';
    document.addEventListener('wheel', preventDefault, { passive: false, capture: true });
    document.addEventListener('touchmove', preventDefault, { passive: false, capture: true });
    window.addEventListener('keydown', preventKeyScroll, { capture: true });
    return () => {
      document.body.style.overflow = prevOverflow;
      (document.body.style as any).touchAction = prevTouchAction;
      (document.body.style as any).overscrollBehavior = prevOverscroll;
      document.removeEventListener('wheel', preventDefault as EventListener, true);
      document.removeEventListener('touchmove', preventDefault as EventListener, true);
      window.removeEventListener('keydown', preventKeyScroll as EventListener, true);
    };
  }, [activeDragItem]);
  
  const handleDragEnd = (_e: DragEndEvent) => {
    setActiveDragItem(null);
  }

  return (
    <DndContext
      autoScroll={false}
      onDragStart={(event) => {
        const data = event?.active?.data?.current as unknown as MediaItem | Preprocessor | ManifestInfo | undefined;
        if (!data) return;
        // Accept media, preprocessor, or model items
        // MediaItem has ClipType in data.type; preprocessors use 'preprocessor'; models use 'model'
        if ((data as any).name) {
          setActiveDragItem(data as any);
        }
      }}
      onDragEnd={handleDragEnd}
      onDragCancel={() => setActiveDragItem(null)}
    >  
    <main ref={containerRef} className="w-full text-center font-poppins bg-black h-screen flex flex-col">
      <Topbar />
      <Toaster />  
      <div className="flex h-full w-full p-3">
        {layout === 'default' && (
          <ResizablePanelGroup direction="vertical" className="flex-1 gap-0.5 overflow-hidden">
            <ResizablePanel defaultSize={70} minSize={30} maxSize={70}>
            <ResizablePanelGroup direction="horizontal" className="gap-0.5 overflow-hidden">
            <MediaModelPanel order={1} />
            <ResizableHandle className="bg-transparent" />
                <PreviewPanel order={3} />
              <ResizableHandle className="bg-transparent" />
              <PropertiesPanel order={4} />
            </ResizablePanelGroup>
            </ResizablePanel>
            <ResizableHandle className="bg-transparent" />
            <TimelinePanel />
          </ResizablePanelGroup>
        )}
        {layout === 'media' && (
          <ResizablePanelGroup direction="horizontal" className="flex-1 gap-0.5">
            <MediaModelPanel order={undefined} defaultSize={30} minSize={20} maxSize={50} />
            <ResizableHandle className="bg-transparent" />
            <ResizablePanel defaultSize={70} minSize={30}>
              <ResizablePanelGroup direction="vertical" className="gap-0.5">
                <ResizablePanel defaultSize={60} minSize={40}>
                  <ResizablePanelGroup direction="horizontal" className="gap-0.5">
                    <PreviewPanel order={2} defaultSize={55} minSize={30} />
                    <ResizableHandle className="bg-transparent" />
                    <PropertiesPanel order={2} defaultSize={45} minSize={30} />
                  </ResizablePanelGroup>
                </ResizablePanel>
                <ResizableHandle className="bg-transparent" />
                <TimelinePanel />
              </ResizablePanelGroup>
            </ResizablePanel>
          </ResizablePanelGroup>
        )}
        {layout === 'properties' && (
          <ResizablePanelGroup direction="horizontal" className="flex-1 gap-0.5">
            <ResizablePanel defaultSize={65} minSize={40}>
              <ResizablePanelGroup direction="vertical" className="gap-0.5">
                <ResizablePanel defaultSize={70} minSize={50}>
                  <ResizablePanelGroup direction="horizontal" className="gap-0.5">
                    <MediaModelPanel order={1} defaultSize={35} minSize={25} />
                    <ResizableHandle className="bg-transparent" />
                    <PreviewPanel order={1} defaultSize={40} minSize={30} />
                  </ResizablePanelGroup>
                </ResizablePanel>
                <ResizableHandle className="bg-transparent" />
                <TimelinePanel />
              </ResizablePanelGroup>
            </ResizablePanel>
            <ResizableHandle className="bg-transparent" />
            <PropertiesPanel order={2} defaultSize={35} minSize={20} />
          </ResizablePanelGroup>
        )}
      </div>
    </main>
    <GlobalContextMenu />
    <DragOverlay dropAnimation={null}>
      {activeDragItem ? (
        <>
        {(activeDragItem as any)?.type !== 'preprocessor' && (activeDragItem as any)?.type !== 'model' && (
        <div className={cn("w-44 aspect-video rounded-md overflow-hidden bg-brand opacity-100", {
          "opacity-0": ghostInStage && clips.length > 0,
        })}>
          <MediaThumb item={activeDragItem as MediaItem} isDragging={true} />
        </div>
        )}
        {(activeDragItem as any)?.type === 'model' && (
          <div className={cn("opacity-100", { "opacity-0": ghostInStage && clips.length > 0 })}>
            {(() => {
              const m = activeDragItem as ManifestInfo;
 
              return <ModelItem manifest={m} isDragging={true} />
            })()}
          </div>
        )}
        {activeDragItem?.type === 'preprocessor' && (
          <div className={cn("overflow-hidden opacity-100", {
            "opacity-0": ghostInStage && clips.length > 0,
          })}>
          <PreprocessorItem preprocessor={activeDragItem as Preprocessor} isDragging={true} />
        </div>
        )}
        </>
      ) : null}
    </DragOverlay>
    </DndContext>
  );
}

export default App;
