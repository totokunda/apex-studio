import { FiMinusCircle, FiPlusCircle } from "react-icons/fi";
import React, { useState, useRef, useEffect, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  BackButton,
  RewindBackward,
  RewindForward,
  SplitButton,
  TrashButton,
  PlayPauseButton,
  SeparateButton,
  CropButton,
} from "./Buttons";
import { useControlsStore } from "@/lib/control";
import { useClipStore } from "@/lib/clip";
import { ZoomLevel } from "@/lib/types";
import { MIN_DURATION } from "@/lib/settings";
import { useDrawingStore } from "@/lib/drawing";
import { useMaskStore } from "@/lib/mask";
import { useViewportStore } from "@/lib/viewport";

const TimelineZoom = () => {
  const {
    zoomLevel,
    setZoomLevel,
    setTimelineDuration,
    focusFrame,
    setFocusFrame,
    focusAnchorRatio,
    totalTimelineFrames,
    setFocusAnchorRatio,
    minZoomLevel,
    maxZoomLevel,
  } = useControlsStore();
  const [isDragging, setIsDragging] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  const clips = useClipStore((state) => state.clips);
  const hasClips = useMemo(() => clips.length > 0, [clips]);

  useEffect(() => {
    if (!hasClips) {
      setZoomLevel(1);
      setFocusFrame(0);
    }
  }, [hasClips]);

  const setZoom = (level: number) => {
    // Clamp to valid integer zoom step
    const clampedLevel = Math.max(
      minZoomLevel,
      Math.min(maxZoomLevel, Math.round(level)),
    );

    // Use dynamic baseline where zoomLevel 1 spans the entire timeline baseline
    // and max zoom spans exactly MIN_DURATION (5 frames)
    const maxDuration = Math.max(1, totalTimelineFrames);
    const minDuration = Math.max(1, Math.min(MIN_DURATION, maxDuration));
    const steps = Math.max(1, maxZoomLevel - minZoomLevel);
    const ratio = minDuration / maxDuration;
    const levelIndex = clampedLevel - minZoomLevel; // 0..steps

    // Build deterministic duration table once per call (cheap) to avoid rounding drift
    const durations: number[] = new Array(steps + 1).fill(0).map((_, i) => {
      const ti = i / steps; // 0..1
      const d = Math.round(maxDuration * Math.pow(ratio, ti));
      return Math.max(minDuration, Math.min(maxDuration, d));
    });

    const targetDuration = durations[levelIndex];

    // Anchor focusFrame; clamp within timeline bounds while keeping exact width
    let newStart = Math.round(focusFrame - focusAnchorRatio * targetDuration);
    newStart = Math.max(
      0,
      Math.min(newStart, Math.max(0, totalTimelineFrames - targetDuration)),
    );
    const newEnd = newStart + targetDuration;

    // Keep anchor ratio consistent with the final clamped window
    const newAnchor =
      targetDuration > 0 ? (focusFrame - newStart) / targetDuration : 0.5;
    setFocusAnchorRatio(Math.max(0, Math.min(1, newAnchor)));

    setTimelineDuration(newStart, newEnd);
    setZoomLevel(clampedLevel as ZoomLevel);
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
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const newZoom =
      minZoomLevel + Math.round(percentage * (maxZoomLevel - minZoomLevel));
    setZoom(newZoom);
  };

  // Calculate circle position based on zoom level
  const getCirclePosition = () => {
    const progress = (zoomLevel - minZoomLevel) / (maxZoomLevel - minZoomLevel);
    return progress * 100; // percentage
  };

  // Add global event listeners for mouse move and up
  React.useEffect(() => {
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
    <div className="flex items-center  gap-x-2 justify-center transition-opacity duration-300">
      <FiMinusCircle
        onClick={() => {
          if (zoomLevel === 1 || !hasClips) return;
          setZoom(zoomLevel - 1);
        }}
        className={cn("text-brand-light/70 h-4 w-4 duration-300", {
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
          "h-1 w-24 rounded-full bg-brand-light/10 transform-gpu cursor-pointer relative",
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

        {/* Draggable Circle */}
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
        className={cn("text-brand-light/70 h-4 w-4 duration-300", {
          "opacity-100": zoomLevel < 10 && hasClips,
          "opacity-60": zoomLevel >= 10 || !hasClips,
          "cursor-not-allowed": zoomLevel >= 10 || !hasClips,
          "cursor-pointer": zoomLevel < 10 && hasClips,
        })}
      />
    </div>
  );
};

interface TimeControlProps {}

const TimeControl: React.FC<TimeControlProps> = () => {
  const { clipDuration } = useClipStore();
  const { focusFrame, fps } = useControlsStore();

  const formatTime = (frames: number) => {
    if (
      frames === 0 ||
      frames === undefined ||
      frames === null ||
      isNaN(frames) ||
      frames === Infinity ||
      frames === -Infinity
    )
      return "00:00.00";
    // convert frames to seconds
    const totalSeconds = frames / fps;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const remainingSeconds = totalSeconds % 60;

    if (hours > 0) {
      return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${remainingSeconds.toFixed(2).padStart(5, "0")}`;
    }
    return `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toFixed(2).padStart(5, "0")}`;
  };

  return (
    <div className="flex items-center gap-x-2 w-full">
      <span className="text-brand-light/60 text-xs">
        <span className="w-17 inline-block">{formatTime(focusFrame)}</span>/
        <span className="w-16 inline-block">{formatTime(clipDuration)}</span>
      </span>
    </div>
  );
};

const Controls = () => {
  const { setIsOverMask, setIsMaskDragging } = useMaskStore();
  const tool = useViewportStore((s) => s.tool);
  const selectedMaskId = useControlsStore((s) => s.selectedMaskId);
  const selectedLineId = useDrawingStore((s) => s.selectedLineId);
  const selectedIds = useControlsStore((s) => s.selectedClipIds);
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          (target as HTMLElement).isContentEditable)
      ) {
        return;
      }

      // If the user has selected normal (non-input) text in the UI, prefer native
      // browser copy/cut behavior over timeline shortcut handling.
      const hasTextSelection = (() => {
        try {
          const sel = window.getSelection?.();
          if (!sel) return false;
          if (sel.isCollapsed) return false;
          // Ignore empty/whitespace-only selections.
          return sel.toString().trim().length > 0;
        } catch {
          return false;
        }
      })();

      const isMod = e.metaKey || e.ctrlKey;
      const controls = useControlsStore.getState();
      const clipsStore = useClipStore.getState();
      const drawingStore = useDrawingStore.getState();
      const selectedIds = controls.selectedClipIds || [];
      const selectedLineId = drawingStore.selectedLineId;
      const selectedMaskId = controls.selectedMaskId;

      // Delete selected clips, lines, or masks
      if (e.key === "Delete" || e.key === "Backspace") {
        // Delete selected mask if it exists
        if (selectedMaskId && tool === "mask") {
          e.preventDefault();
          // Find the clip that has this mask
          const clipWithMask = clipsStore.clips.find((clip) => {
            if (clip.type !== "video" && clip.type !== "image") return false;
            const masks = (clip as any).masks || [];
            return masks.some((m: any) => m.id === selectedMaskId);
          });

          if (clipWithMask) {
            const currentMasks = (clipWithMask as any).masks || [];
            const updatedMasks = currentMasks.filter(
              (m: any) => m.id !== selectedMaskId,
            );
            clipsStore.updateClip(clipWithMask.clipId, { masks: updatedMasks });
          }

          controls.setSelectedMaskId(null);
          setIsOverMask(false);
          setIsMaskDragging(false);
          return;
        }

        // delete the selected line if it exists
        if (selectedLineId && selectedIds.length > 0) {
          e.preventDefault();
          // get the clip with the drawing type
          selectedIds.forEach((id) => {
            const clip = clipsStore.getClipById(id);
            if (clip?.type === "draw") {
              clipsStore.updateClip(id, {
                lines: clip.lines.filter(
                  (line) => line.lineId !== selectedLineId,
                ),
              });
            }
          });
          // update the clip to remove the line
          drawingStore.setSelectedLineId(null);
          return;
        }
        if (selectedIds.length > 0) {
          e.preventDefault();
          selectedIds.forEach((id) => clipsStore.removeClip(id));
          controls.clearSelection();
        }
        return;
      }

      // Copy
      if (isMod && e.key.toLowerCase() === "c") {
        if (hasTextSelection) return;
        if (selectedIds.length > 0) {
          e.preventDefault();
          clipsStore.copyClips(selectedIds);
        }
        return;
      }

      // Cut
      if (isMod && e.key.toLowerCase() === "x") {
        if (hasTextSelection) return;
        if (selectedIds.length > 0) {
          e.preventDefault();
          clipsStore.cutClips(selectedIds);
          controls.clearSelection();
        }
        return;
      }

      // Paste at current focus frame
      if (isMod && e.key.toLowerCase() === "v") {
        e.preventDefault();
        const { focusFrame } = controls;
        clipsStore.pasteClips(focusFrame);
        return;
      }

      // Convert selected model clips to media (Cmd/Ctrl + Shift + M)
      if (isMod && e.shiftKey && e.key.toLowerCase() === "m") {
        e.preventDefault();
        const selected = controls.selectedClipIds || [];
        if (selected.length > 0) {
          selected.forEach((id) => {
            try {
              clipsStore.convertToMedia(id);
            } catch {}
          });
        }
        return;
      }

      // Group (Cmd/Ctrl + G) and Ungroup (Cmd/Ctrl + Shift + G)
      if (isMod && e.key.toLowerCase() === "g") {
        e.preventDefault();
        if (e.shiftKey) {
          // Ungroup: if a single selected group, ungroup it; else if children of same group selected, ungroup that group
          const selected = controls.selectedClipIds || [];
          if (selected.length === 1) {
            const clip = clipsStore.getClipById(selected[0]);
            if (clip && clip.type === "group") {
              clipsStore.ungroupClips(clip.clipId);
            } else if (clip && (clip as any).groupId) {
              clipsStore.ungroupClips((clip as any).groupId as string);
            }
          } else if (selected.length > 1) {
            // If multiple selected and they share a group, ungroup that
            const first = clipsStore.getClipById(selected[0]);
            const gid = (first as any)?.groupId;
            if (
              gid &&
              selected.every(
                (id) => (clipsStore.getClipById(id) as any)?.groupId === gid,
              )
            ) {
              clipsStore.ungroupClips(gid);
            }
          }
        } else {
          // Group: need at least 2 selected non-group items; ignore if any is a group
          const selected = (controls.selectedClipIds || []).filter((id) => {
            const c = clipsStore.getClipById(id);
            return c && c.type !== "group";
          });
          if (selected.length >= 2) {
            clipsStore.groupClips(selected);
          }
        }
        return;
      }

      // Zoom shortcuts (viewport)
      // Cmd/Ctrl + 0 => Zoom to Fit
      // Cmd/Ctrl + 5 => 50%
      // Cmd/Ctrl + 1 => 100%
      // Cmd/Ctrl + 2 => 200%
      const viewport = useViewportStore.getState();
      const key = e.key.toLowerCase();
      if (isMod && (key === "0" || key === "1" || key === "2" || key === "5")) {
        e.preventDefault();
        if (key === "0") {
          viewport.zoomToFit();
        } else if (key === "5") {
          viewport.setScalePercent(50);
        } else if (key === "1") {
          viewport.setScalePercent(100);
        } else if (key === "2") {
          viewport.setScalePercent(200);
        }
        return;
      }

      // In-mode selections using number keys (no modifiers)
      // Shape mode: 1=Rectangle, 2=Ellipse, 3=Polygon, 4=Line, 5=Star
      if (!isMod && viewport.tool === "shape") {
        if (key === "1") {
          e.preventDefault();
          viewport.setShape("rectangle");
          return;
        }
        if (key === "2") {
          e.preventDefault();
          viewport.setShape("ellipse");
          return;
        }
        if (key === "3") {
          e.preventDefault();
          viewport.setShape("polygon");
          return;
        }
        if (key === "4") {
          e.preventDefault();
          viewport.setShape("line");
          return;
        }
        if (key === "5") {
          e.preventDefault();
          viewport.setShape("star");
          return;
        }
      }

      // Mask mode: 1=Lasso, 2=Shape, 3=Draw, 4=Touch
      if (!isMod && viewport.tool === "mask") {
        const maskStore = useMaskStore.getState();
        if (key === "1") {
          e.preventDefault();
          maskStore.setTool("lasso");
          return;
        }
        if (key === "2") {
          e.preventDefault();
          maskStore.setTool("shape");
          return;
        }
        if (key === "3") {
          e.preventDefault();
          maskStore.setTool("draw");
          return;
        }
        if (key === "4") {
          e.preventDefault();
          maskStore.setTool("touch");
          return;
        }
      }

      // Draw mode: 1=Brush, 2=Highlighter, 3=Eraser
      if (!isMod && viewport.tool === "draw") {
        if (key === "1") {
          e.preventDefault();
          drawingStore.setTool("brush");
          return;
        }
        if (key === "2") {
          e.preventDefault();
          drawingStore.setTool("highlighter");
          return;
        }
        if (key === "3") {
          e.preventDefault();
          drawingStore.setTool("eraser");
          return;
        }
      }

      // Tool switching (letters, no modifiers)
      if (!isMod) {
        if (key === "v") {
          e.preventDefault();
          viewport.setTool("pointer");
          return;
        }
        if (key === "h") {
          e.preventDefault();
          viewport.setTool("hand");
          return;
        }
        if (key === "d") {
          e.preventDefault();
          viewport.setTool("draw");
          return;
        }
        if (key === "t") {
          e.preventDefault();
          viewport.setTool("text");
          return;
        }
        if (key === "m") {
          e.preventDefault();
          viewport.setTool("mask");
          return;
        }
        if (key === "s") {
          e.preventDefault();
          viewport.setTool("shape");
          return;
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [tool, selectedMaskId, selectedLineId, selectedIds]);
  return (
    <div className="relative  flex items-center ">
      <div className="flex items-center w-full bg-brand-background/30 justify-between border-b border-brand-light/5 px-5 py-1">
        <div className="flex items-center gap-x-2">
          <SplitButton />
          <TrashButton />
          <SeparateButton />
          <CropButton />
        </div>
        <div className=" flex items-center justify-center gap-x-4 absolute left-1/2 -translate-x-1/2">
          <BackButton />
          <div className="flex items-center gap-x-2">
            <RewindBackward />
            <RewindForward />
          </div>
          <PlayPauseButton />
          <TimeControl />
        </div>
        <div className="flex items-center gap-x-2">
          <div className="ml-3">
            <TimelineZoom />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Controls;
