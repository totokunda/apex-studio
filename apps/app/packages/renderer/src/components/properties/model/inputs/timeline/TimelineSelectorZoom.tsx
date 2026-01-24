import { useInputControlsStore } from "@/lib/inputControl";
import { useState, useRef, useEffect, useMemo } from "react";
import { FiMinusCircle, FiPlusCircle } from "react-icons/fi";
import { cn } from "@/lib/utils";
import { ZoomLevel } from "@/lib/types";
import { MIN_DURATION } from "@/lib/settings";

interface TimelineSelectorZoomProps {
  hasClip?: boolean;
  inputId?: string;
  mode?: "frame" | "range";
}

const minZoomLevel = 1;
const maxZoomLevel = 10;

const TimelineSelectorZoom: React.FC<TimelineSelectorZoomProps> = ({
  hasClip = true,
  inputId,
}) => {
  const {
    zoomLevelByInputId,
    setZoomLevel,
    setTimelineDuration,
    setFocusFrame,
    setFocusAnchorRatio,
    focusFrameByInputId,
    totalTimelineFramesByInputId,
  } = useInputControlsStore();
  const zoomLevel = zoomLevelByInputId[inputId ?? ""] ?? 1;
  const focusFrame = focusFrameByInputId[inputId ?? ""] ?? 0;
  const totalTimelineFrames = totalTimelineFramesByInputId[inputId ?? ""] ?? 1;
  const [isDragging, setIsDragging] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);

  const hasClips = useMemo(() => !!hasClip, [hasClip]);

  useEffect(() => {
    if (!hasClips) {
      setZoomLevel(1, inputId ?? "");

      setFocusFrame(0, inputId ?? "");
    }
  }, [hasClips, inputId, setZoomLevel, setFocusFrame]);

  const setZoom = (level: number) => {
    const clampedLevel = Math.max(
      minZoomLevel,
      Math.min(maxZoomLevel, Math.round(level)),
    );
    const maxDuration = Math.max(1, totalTimelineFrames);
    const minDuration = Math.max(1, Math.min(MIN_DURATION, maxDuration));
    const steps = Math.max(1, maxZoomLevel - minZoomLevel);
    const ratio = minDuration / maxDuration;
    const levelIndex = clampedLevel - minZoomLevel;

    const durations: number[] = new Array(steps + 1).fill(0).map((_, i) => {
      const ti = i / steps;
      const d = Math.round(maxDuration * Math.pow(ratio, ti));
      return Math.max(minDuration, Math.min(maxDuration, d));
    });

    const targetDuration = durations[levelIndex];

    // Always keep the focus frame visible when zooming,
    // centering it in the window whenever possible.
    const clampedFocus = Math.max(
      0,
      Math.min(totalTimelineFrames - 1, focusFrame),
    );
    const anchorFocusFrame = clampedFocus;
    const anchorRatio = 0.5;

    let newStart = Math.round(anchorFocusFrame - anchorRatio * targetDuration);
    newStart = Math.max(
      0,
      Math.min(newStart, Math.max(0, totalTimelineFrames - targetDuration)),
    );
    const newEnd = newStart + targetDuration;

    const newAnchor =
      targetDuration > 0 ? (anchorFocusFrame - newStart) / targetDuration : 0.5;
    const finalAnchor = Math.max(0, Math.min(1, newAnchor));

    setFocusAnchorRatio(finalAnchor, inputId ?? "");
    setTimelineDuration(newStart, newEnd, inputId ?? "");
    setZoomLevel(clampedLevel as ZoomLevel, inputId ?? "");
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    updateZoomFromMouse(e);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    updateZoomFromMouse(e);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const updateZoomFromMouse = (e: React.MouseEvent | MouseEvent) => {
    if (!barRef.current) return;
    const rect = barRef.current.getBoundingClientRect();
    const x = (e as MouseEvent).clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const newZoom =
      minZoomLevel + Math.round(percentage * (maxZoomLevel - minZoomLevel));
    setZoom(newZoom);
  };

  const getCirclePosition = () => {
    const progress = (zoomLevel - minZoomLevel) / (maxZoomLevel - minZoomLevel);
    return progress * 100;
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging]);

  return (
    <div className="flex items-center gap-x-2 justify-end transition-opacity duration-300 w-3/5">
      <FiMinusCircle
        onClick={() => {
          if (zoomLevel === 1 || !hasClips) return;
          setZoom(zoomLevel - 1);
        }}
        className={cn("text-brand-light/70 h-3.5 w-3.5 duration-300", {
          "opacity-60": zoomLevel === 1 || !hasClips,
          "opacity-100": zoomLevel > 1 && hasClips,
          "cursor-not-allowed": zoomLevel === 1 || !hasClips,
          "cursor-pointer": zoomLevel > 1 && hasClips,
        })}
      />
      <div
        ref={barRef}
        onMouseDown={(e) => hasClips && handleMouseDown(e)}
        onMouseEnter={() => hasClips && setIsHovering(true)}
        onMouseLeave={() => hasClips && setIsHovering(false)}
        className={cn(
          "h-1 w-11/12 max-w-48 rounded-full bg-brand-light/10 transform-gpu cursor-pointer relative",
          {
            "cursor-grabbing": isDragging,
          },
        )}
      >
        <div
          className={cn(
            "h-1 rounded-full bg-brand-light/70 transition-all duration-300 transform-gpu pointer-events-none",
            {
              "w-0": zoomLevel === 1 || !hasClips,
              "w-1/9": zoomLevel === 2,
              "w-2/9": zoomLevel === 3,
              "w-3/9": zoomLevel === 4,
              "w-4/9": zoomLevel === 5,
              "w-5/9": zoomLevel === 6,
              "w-6/9": zoomLevel === 7,
              "w-7/9": zoomLevel === 8,
              "w-8/9": zoomLevel === 9,
              "w-9/9": zoomLevel === 10,
              "transition-none": isDragging,
            },
          )}
        ></div>
        <div
          className={cn(
            "absolute top-1/2 w-3 h-3 bg-brand-light/90 rounded-full transform -translate-y-1/2 -translate-x-1/2 transition-all duration-200 pointer-events-none",
            {
              "opacity-100 scale-100": isHovering || isDragging,
              "opacity-0 scale-75": !isHovering && !isDragging,
              "bg-brand-light": isDragging,
              "transition-none": isDragging,
            },
          )}
          style={{
            left: `${getCirclePosition()}%`,
          }}
        />
      </div>
      <FiPlusCircle
        onClick={() => {
          if (zoomLevel >= 10 || !hasClips) return;
          setZoom(zoomLevel + 1);
        }}
        className={cn("text-brand-light/70 h-3.5 w-3.5 duration-300", {
          "opacity-100": zoomLevel < 10 && hasClips,
          "opacity-60": zoomLevel >= 10 || !hasClips,
          "cursor-not-allowed": zoomLevel >= 10 || !hasClips,
          "cursor-pointer": zoomLevel < 10 && hasClips,
        })}
      />
    </div>
  );
};

export default TimelineSelectorZoom;
