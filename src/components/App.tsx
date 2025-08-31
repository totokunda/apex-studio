import React, { useEffect, useRef, useState } from "react";
import "../styles/index.css";
import TimelineEditor from "./timeline/TimelineEditor";
import { GrMultimedia } from "react-icons/gr";

import Controls from "./controls/Controls";
import TopBar from "./bars/TopBar";
import Video from "./Video";
import { useLayoutStore } from "../lib/layout";

const App:React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const bottomHeight = useLayoutStore((s) => s.bottomHeight);
  const setBottomHeight = useLayoutStore((s) => s.setBottomHeight);
  const setContainerRect = useLayoutStore((s) => s.setContainerRect);
  const setConstraints = useLayoutStore((s) => s.setConstraints);
  const isDragging = useLayoutStore((s) => s.isDragging);
  const startDragging = useLayoutStore((s) => s.startDragging);
  const stopDragging = useLayoutStore((s) => s.stopDragging);
  const dragTo = useLayoutStore((s) => s.dragTo);
  const handleHeight = useLayoutStore((s) => s.handleHeight);

  useEffect(() => {
    // sync constraints
    setConstraints({ minBottom: 240, minTop: 360, handleHeight: handleHeight || 1 });
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) setContainerRect({ top: rect.top, height: rect.height });
    // initialize bottom height ~2/5
    const total = rect?.height ?? window.innerHeight;
    const initial = Math.round((total - (handleHeight || 1)) * 0.4);
    setBottomHeight(initial);
  }, []);

  useEffect(() => {
    const onResize = () => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect) setContainerRect({ top: rect.top, height: rect.height });
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    if (!isDragging) return;
    let rafId: number | null = null;
    let pendingY: number | null = null;
    const onMove = (e: MouseEvent) => {
      pendingY = e.clientY;
      if (rafId === null) {
        rafId = requestAnimationFrame(() => {
          if (pendingY !== null) dragTo(pendingY);
          rafId = null;
        });
      }
    };
    const onUp = () => stopDragging();
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    const prevUserSelect = document.body.style.userSelect;
    const prevCursor = document.body.style.cursor;
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'row-resize';
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.userSelect = prevUserSelect;
      document.body.style.cursor = prevCursor;
    };
  }, [isDragging]);

  const onHandleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    startDragging();
  };

  return (
    <main ref={containerRef} className="w-full text-center font-poppins bg-brand-background h-screen flex flex-col">
      <TopBar title="Apex Studio" />
      <div className="flex flex-col w-full bg-brand  flex-1 overflow-hidden">
        <Video />
      </div>
      <div
        onMouseDown={onHandleMouseDown}
        className="relative w-full flex-shrink-0"
        style={{ height: `${handleHeight}px`, cursor: 'row-resize' }}
      >
        <div className="absolute inset-0 bg-brand-light/10" />
      </div>
      <div className="bg-brand flex flex-col border-t-0 border shadow w-full border-brand-light/10 flex-shrink-0 overflow-x-hidden" style={{ height: `${bottomHeight}px`, transition: isDragging ? 'none' : 'height 120ms ease-out' }}>
      <Controls />
      <div className="flex flex-col gap-y-2.5 justify-center h-full overflow-auto">
      <TimelineEditor icon={<GrMultimedia className="h-4 w-4" />} name="models and content" />
      </div>
      </div>
    </main>
  );
}

export default App;
