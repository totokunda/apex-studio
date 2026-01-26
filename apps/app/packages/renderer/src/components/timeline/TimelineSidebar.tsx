import React, { useCallback, useEffect, useRef, useState } from "react";
import { useClipStore } from "@/lib/clip";
import { GoFileMedia } from "react-icons/go";
import {
  LuAudioWaveform,
  LuBox,
  LuPaintbrush,
  LuShapes,
  LuType,
} from "react-icons/lu";
import { IoVolumeMute } from "react-icons/io5";
import { IoVolumeHigh } from "react-icons/io5";
import { IoEye } from "react-icons/io5";
import { IoEyeOff } from "react-icons/io5";
import { TimelineProps } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useControlsStore } from "@/lib/control";
import { MdPhotoFilter } from "react-icons/md";

interface TimelineSidebarProps {
  clampedScroll: number; // for the scroll position
}

const TimelineSidebarItem: React.FC<
  TimelineProps & {
    isCollapsed: boolean;
    clampedScroll: number;
  }
> = ({
  isCollapsed,
  type,
  timelineHeight,
  timelineY,
  timelineId,
  clampedScroll,
}) => {
  const {
    muteTimeline,
    unmuteTimeline,
    hideTimeline,
    unhideTimeline,
    isTimelineMuted,
    isTimelineHidden,
  } = useClipStore();
  const timelineMuted = isTimelineMuted(timelineId);
  const timelineHidden = isTimelineHidden(timelineId);

  return (
    <>
      <div
        style={{
          height: (timelineHeight ?? 0) - 8,
          top: (timelineY ?? 0) + 32 - clampedScroll,
        }}
        className={cn(
          "bg-brand-light/5  w-full absolute flex items-center justify-center",
        )}
      >
        {isCollapsed && (
          <>
            {type === "media" && (
              <GoFileMedia className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "audio" && (
              <LuAudioWaveform className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "shape" && (
              <LuShapes className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "text" && (
              <LuType className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "filter" && (
              <MdPhotoFilter className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "draw" && (
              <LuPaintbrush className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "model" && (
              <LuBox className="w-3 h-3 text-brand-light/50" />
            )}
          </>
        )}
        {!isCollapsed && (
          <div className="w-full h-full flex items-center gap-x-3 px-3">
            {type === "media" && (
              <GoFileMedia className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "audio" && (
              <LuAudioWaveform className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "shape" && (
              <LuShapes className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "text" && (
              <LuType className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "filter" && (
              <MdPhotoFilter className="w-3 h-3 text-brand-light/50" />
            )}
            {type === "draw" && (
              <LuPaintbrush className="w-3 h-3 text-brand-light/50" />
            )}

            {type !== "shape" &&
              type !== "text" &&
              type !== "filter" &&
              type !== "draw" && (
                <button
                  className="cursor-pointer"
                  onClick={() => {
                    if (timelineMuted) {
                      unmuteTimeline(timelineId);
                    } else {
                      muteTimeline(timelineId);
                    }
                  }}
                >
                  {timelineMuted ? (
                    <IoVolumeMute className="w-3 h-3 text-brand-light hover:text-brand-lighter" />
                  ) : (
                    <IoVolumeHigh className="w-3 h-3 text-brand-light hover:text-brand-lighter" />
                  )}
                </button>
              )}
            {type !== "audio" && (
              <button
                className="cursor-pointer"
                onClick={() => {
                  if (timelineHidden) {
                    unhideTimeline(timelineId);
                  } else {
                    hideTimeline(timelineId);
                  }
                }}
              >
                {timelineHidden ? (
                  <IoEyeOff className="w-3 h-3 text-brand-light hover:text-brand-lighter" />
                ) : (
                  <IoEye className="w-3 h-3 text-brand-light hover:text-brand-lighter" />
                )}
              </button>
            )}
          </div>
        )}
      </div>
    </>
  );
};

const TimelineSidebar: React.FC<TimelineSidebarProps> = ({ clampedScroll }) => {
  const collapsedWidth = 40; // ~ w-14
  const expandedWidth = 84; // ~ w-24
  const [width, setWidth] = useState<number>(collapsedWidth);
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const startXRef = useRef<number>(0);
  const isDraggingRef = useRef<boolean>(false);
  const { timelines } = useClipStore();
  const { isFullscreen } = useControlsStore();

  const endDrag = useCallback(() => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", endDrag);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const dx = e.clientX - startXRef.current;
      const THRESHOLD = 4;
      if (Math.abs(dx) < THRESHOLD) return;
      if (isOpen) {
        // Only allow collapse on initial left drag
        if (dx < 0) {
          setWidth(collapsedWidth);
          setIsOpen(false);
        }
      } else {
        // Only allow expand on initial right drag
        if (dx > 0) {
          setWidth(expandedWidth);
          setIsOpen(true);
        }
      }
      endDrag();
    },
    [collapsedWidth, expandedWidth, isOpen, endDrag],
  );

  const startDrag = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      isDraggingRef.current = true;
      startXRef.current = e.clientX;
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", endDrag);
    },
    [onMouseMove, endDrag],
  );

  useEffect(() => {
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", endDrag);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [onMouseMove, endDrag]);

  return (
    <div
      className=" h-full border-brand-light/10 transition-all duration-200 relative shrink-0 bg-brand"
      style={{ width }}
    >
      {/* Right rail: drag to open or close */}
      <button
        aria-label="Drag to resize timeline sidebar"
        title="Drag to resize"
        className="absolute top-0 hover:bg-brand-light/50 right-0 h-full transition-all duration-100 w-px bg-brand-light/10 z-50 flex items-center justify-center group outline-none"
        style={{
          cursor: width === collapsedWidth ? "e-resize" : "w-resize",
          display: isFullscreen ? "none" : "block",
        }}
        onMouseDown={(e) => startDrag(e)}
      />
      <div className="relative h-full">
        {timelines.map((timeline) => (
          <TimelineSidebarItem
            key={timeline.timelineId}
            isCollapsed={!isOpen}
            {...timeline}
            clampedScroll={clampedScroll}
          />
        ))}
      </div>
    </div>
  );
};

export default TimelineSidebar;
