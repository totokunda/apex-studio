import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Stage, Layer, Group, Rect } from "react-konva";
import { useViewportStore } from "@/lib/viewport";
import { useClipStore } from "@/lib/clip";
import { BASE_LONG_SIDE } from "@/lib/settings";
import VideoPreview from "./clips/VideoPreview";
import ImagePreview from "./clips/ImagePreview";
import ShapePreview from "./clips/ShapePreview";
import TextPreview from "./clips/TextPreview";
import { useControlsStore } from "@/lib/control";
import { AnyClipProps } from "@/lib/types";
import { SlSizeActual } from "react-icons/sl";
import { FaCirclePause, FaCirclePlay } from "react-icons/fa6";
import { getApplicatorsForClip } from "@/lib/applicator-utils";
import { useWebGLHaldClut } from "./webgl-filters";
import DrawingPreview from "./clips/DrawingPreview";
import DynamicModelPreview from "./clips/DynamicModelPreview";

interface FullscreenPreviewProps {
  onExit: () => void;
}

const FullscreenPreview: React.FC<FullscreenPreviewProps> = ({ onExit }) => {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<any>(null);
  const [showControls, setShowControls] = useState(true);
  const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const { clips, clipWithinFrame, timelines, clipDuration } = useClipStore();
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const { play, pause, isPlaying, setFocusFrame, fps } = useControlsStore();
  const haldClutInstance = useWebGLHaldClut();

  // Memoize applicator factory configuration
  const applicatorConfig = useMemo(
    () => ({
      haldClutInstance,
      // Add more shared resources here as needed
    }),
    [haldClutInstance],
  );

  // Get applicators for a specific clip using the factory
  const getClipApplicators = useCallback(
    (clipId: string) => {
      return getApplicatorsForClip(clipId, applicatorConfig);
    },
    [applicatorConfig, focusFrame, timelines, clips],
  );

  // Preload all resources for applicator clips
  useEffect(() => {
    if (!haldClutInstance) return;

    // Preload filter CLUTs
    const filterClips = clips.filter((c) => c.type === "filter");
    const loadPromises = filterClips.map(async (clip: any) => {
      const filterPath = clip.fullPath || clip.smallPath;
      if (filterPath) {
        try {
          await haldClutInstance.preloadClut(filterPath);
        } catch (e) {
          console.warn("Failed to preload CLUT:", filterPath, e);
        }
      }
    });

    // Add preloading for other applicator types here as they are added
    // e.g., mask textures, processor models, etc.

    Promise.all(loadPromises).catch(console.error);
  }, [clips, haldClutInstance]);

  const formatTime = useCallback(
    (frames: number) => {
      if (
        frames === 0 ||
        frames === undefined ||
        frames === null ||
        isNaN(frames) ||
        frames === Infinity ||
        frames === -Infinity
      )
        return "00:00.00";
      const totalSeconds = frames / fps;
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const remainingSeconds = totalSeconds % 60;

      if (hours > 0) {
        return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${remainingSeconds.toFixed(2).padStart(5, "0")}`;
      }
      return `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toFixed(2).padStart(5, "0")}`;
    },
    [fps],
  );

  const sortClips = useCallback(
    (clips: AnyClipProps[]) => {
      // Treat each group as a single sortable unit; then expand children in defined order
      type GroupUnit = {
        kind: "group";
        id: string;
        y: number;
        start: number;
        children: AnyClipProps[];
      };
      type SingleUnit = {
        kind: "single";
        y: number;
        start: number;
        clip: AnyClipProps;
      };

      const groups = clips.filter((c) => c.type === "group") as AnyClipProps[];
      const childrenSet = new Set<string>(
        groups.flatMap((g) => {
          const nested = ((g as any).children as string[][] | undefined) ?? [];
          return nested.flat();
        }),
      );

      // Build group units
      const groupUnits: GroupUnit[] = groups.map((g) => {
        const y =
          timelines.find((t) => t.timelineId === g.timelineId)?.timelineY ?? 0;
        const start = g.startFrame ?? 0;
        const nested = ((g as any).children as string[][] | undefined) ?? [];
        const childIdsFlat = nested.flat();
        const children = childIdsFlat
          .map((id) => clips.find((c) => c.clipId === id))
          .filter(Boolean) as AnyClipProps[];
        return { kind: "group", id: g.clipId, y, start, children };
      });

      // Build single units for non-group, non-child clips
      const singleUnits: SingleUnit[] = clips
        .filter((c) => c.type !== "group" && !childrenSet.has(c.clipId))
        .map((c) => {
          const y =
            timelines.find((t) => t.timelineId === c.timelineId)?.timelineY ??
            0;
          const start = c.startFrame ?? 0;
          return { kind: "single", y, start, clip: c };
        });

      // Sort units: lower on screen first (higher y), then earlier start
      const units = [...groupUnits, ...singleUnits].sort((a, b) => {
        if (a.y !== b.y) return b.y - a.y;
        return a.start - b.start;
      });

      // Flatten units back to clip list; for groups, expand children in their defined order
      const result: AnyClipProps[] = [];
      for (const u of units) {
        if (u.kind === "single") {
          result.push(u.clip);
        } else {
          // Ensure children are ordered as in group's children list
          result.push(...u.children.reverse());
        }
      }

      return result;
    },
    [timelines, clips],
  );

  const filterClips = useCallback(
    (clips: AnyClipProps[], audio: boolean = false) => {
      const filteredClips = clips.filter((clip) => {
        const timeline = timelines.find(
          (t) => t.timelineId === clip.timelineId,
        );
        if (audio && timeline?.muted) return false;
        if (timeline?.hidden) return false;
        return true;
      });
      return filteredClips;
    },
    [timelines],
  );

  // Compute rect dimensions based on aspect ratio
  const { rectWidth, rectHeight } = useMemo(() => {
    const ratio = aspectRatio.width / aspectRatio.height;
    const baseShortSide = BASE_LONG_SIDE;
    if (!Number.isFinite(ratio) || ratio <= 0) {
      return { rectWidth: 0, rectHeight: 0 };
    }
    return { rectWidth: baseShortSide * ratio, rectHeight: baseShortSide };
  }, [aspectRatio.width, aspectRatio.height]);

  // Center the content in fullscreen
  const { scale, position } = useMemo(() => {
    if (!size.width || !size.height || !rectWidth || !rectHeight) {
      return { scale: 1, position: { x: 0, y: 0 } };
    }

    // Calculate scale to fit content in viewport
    const scaleX = size.width / rectWidth;
    const scaleY = size.height / rectHeight;
    const scale = Math.min(scaleX, scaleY);

    // Center the scaled content
    const scaledWidth = rectWidth * scale;
    const scaledHeight = rectHeight * scale;
    const x = (size.width - scaledWidth) / 2;
    const y = (size.height - scaledHeight) / 2;

    return { scale, position: { x, y } };
  }, [size.width, size.height, rectWidth, rectHeight]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Auto-hide controls logic
  const resetHideTimer = useCallback(() => {
    setShowControls(true);
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
    }
    hideTimeoutRef.current = setTimeout(() => {
      setShowControls(false);
    }, 3000);
  }, []);

  const handleMouseMove = useCallback(() => {
    resetHideTimer();
  }, [resetHideTimer]);

  useEffect(() => {
    resetHideTimer();
    return () => {
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
      }
    };
  }, [resetHideTimer]);

  // Scrubber logic
  const [isDragging, setIsDragging] = useState(false);
  const [dragProgress, setDragProgress] = useState<number | null>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);

  const progress =
    isDragging && dragProgress !== null
      ? dragProgress
      : clipDuration > 0
        ? focusFrame / clipDuration
        : 0;

  const handleScrubberMove = useCallback(
    (clientX: number) => {
      if (!progressBarRef.current) return;
      const rect = progressBarRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const newProgress = Math.max(0, Math.min(x / rect.width, 1));

      // Update progress immediately for smooth visual feedback
      requestAnimationFrame(() => {
        setDragProgress(newProgress);
      });

      const newFrame = Math.round(newProgress * clipDuration);
      setFocusFrame(Math.min(newFrame, clipDuration));
    },
    [clipDuration, setFocusFrame],
  );

  const handleScrubberMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      handleScrubberMove(e.clientX);
    },
    [isDragging, handleScrubberMove],
  );

  const handleScrubberMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragProgress(null);
  }, []);

  const handleProgressBarMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
      if (isPlaying) pause();
      handleScrubberMove(e.clientX);
    },
    [handleScrubberMove, isPlaying, pause],
  );

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleScrubberMouseMove);
      document.addEventListener("mouseup", handleScrubberMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleScrubberMouseMove);
        document.removeEventListener("mouseup", handleScrubberMouseUp);
      };
    }
  }, [isDragging, handleScrubberMouseMove, handleScrubberMouseUp]);

  // Handle escape key to exit fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onExit();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onExit]);

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 z-9999 bg-black"
      onMouseMove={handleMouseMove}
    >
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        className="bg-black"
      >
        <Layer width={size.width} height={size.height}>
          <Group x={position.x} y={position.y} scaleX={scale} scaleY={scale}>
            <Rect
              x={0}
              y={0}
              width={rectWidth}
              height={rectHeight}
              fill={"#000000"}
            />
            {sortClips(filterClips(clips)).map((clip) => {
              const clipAtFrameNoOverlap = clipWithinFrame(clip, focusFrame);

              // Get applicators for clips that support effects (video, image, etc.)
              const applicators = getClipApplicators(clip.clipId);

              switch (clip.type) {
                case "video":
                  return (
                    <VideoPreview
                      key={clip.clipId}
                      {...clip}
                      rectWidth={rectWidth}
                      rectHeight={rectHeight}
                      applicators={applicators}
                      overlap={clipAtFrameNoOverlap}
                      hidden={!clipAtFrameNoOverlap}
                    />
                  );
                case "image":
                  return (
                    <ImagePreview
                      key={clip.clipId}
                      {...clip}
                      rectWidth={rectWidth}
                      rectHeight={rectHeight}
                      applicators={applicators}
                      overlap={clipAtFrameNoOverlap}
                    />
                  );
                case "model":
                  return (
                    <DynamicModelPreview
                      key={clip.clipId}
                      clip={clip as any}
                      rectWidth={rectWidth}
                      rectHeight={rectHeight}
                      applicators={applicators}
                      overlap={clipAtFrameNoOverlap}
                    />
                  );
                case "shape":
                  return (
                    <ShapePreview
                      key={clip.clipId}
                      {...clip}
                      rectWidth={rectWidth}
                      rectHeight={rectHeight}
                      applicators={applicators}
                    />
                  );
                case "text":
                  return (
                    <TextPreview
                      key={clip.clipId}
                      {...clip}
                      rectWidth={rectWidth}
                      rectHeight={rectHeight}
                      applicators={applicators}
                    />
                  );
                case "draw":
                  return (
                    <DrawingPreview
                      key={clip.clipId}
                      {...clip}
                      rectWidth={rectWidth}
                      rectHeight={rectHeight}
                      applicators={applicators}
                    />
                  );
                default:
                  // Applicator clips (filter, mask, processor, etc.) don't render visually
                  return null;
              }
            })}
          </Group>
        </Layer>
      </Stage>

      {/* Floating control bar */}
      <div
        className={`absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/80 to-transparent transition-opacity duration-300 ${
          showControls ? "opacity-100" : "opacity-0"
        }`}
      >
        <div className="px-8 pb-6 pt-12">
          {/* Progress bar */}
          <div
            ref={progressBarRef}
            className="relative w-full h-1 bg-white/20 rounded-full cursor-pointer mb-4 group"
            onMouseDown={handleProgressBarMouseDown}
          >
            <div
              className={`absolute h-full bg-blue-500 rounded-full pointer-events-none ${
                isDragging ? "" : "transition-all"
              }`}
              style={{ width: `${progress * 100}%` }}
            />
            <div
              className={`absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-lg transition-opacity ${
                isDragging
                  ? "opacity-100 cursor-grabbing"
                  : "opacity-0 group-hover:opacity-100 cursor-grab"
              }`}
              style={{ left: `calc(${progress * 100}% - 6px)` }}
            />
          </div>

          {/* Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Play/Pause button */}
              <button
                onClick={() => (isPlaying ? pause() : play())}
                className="text-white hover:text-blue-400 transition-colors cursor-pointer"
              >
                {isPlaying ? (
                  <FaCirclePause className="h-8 w-8" />
                ) : (
                  <FaCirclePlay className="h-8 w-8" />
                )}
              </button>

              {/* Time display */}
              <div className="text-white text-sm">
                {formatTime(focusFrame)} / {formatTime(clipDuration)}
              </div>
            </div>

            {/* Exit fullscreen button */}
            <button
              onClick={onExit}
              className="text-white hover:text-blue-400 transition-colors cursor-pointer"
            >
              <SlSizeActual className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FullscreenPreview;
