import React, { useMemo } from "react";
import { useAssetControlsStore } from "@/lib/assetControl";
import { getZoomLevelConfig } from "@/lib/zoom";
import { Line, Rect, Text } from "react-konva";
import { useInputControlsStore } from "@/lib/inputControl";

interface TimelineMomentsProps {
  stageWidth: number;
  startPadding: number;
  maxScroll: number;
  thumbY: () => number;
  mode?: "asset" | "input";
  topLine?: boolean;
  inputId?: string;
}

interface TickMark {
  x: number;
  text: string;
  type: "major" | "minor";
  format: "frame" | "second";
}

const getMajorZoomConfigFormat = (
  zoomConfig: {
    majorTickFormat: "frame" | "second";
    minorTickFormat: "frame" | "second";
  },
  fps: number,
  frameInterval: number,
) => {
  if (zoomConfig.majorTickFormat == "second") {
    return "second";
  } else if (
    zoomConfig.majorTickFormat == "frame" &&
    frameInterval % fps == 0
  ) {
    return "second";
  } else {
    return "frame";
  }
};

const TimelineMoments: React.FC<TimelineMomentsProps> = React.memo(
  ({
    stageWidth,
    startPadding,
    maxScroll,
    thumbY,
    mode = "asset",
    topLine = false,
    inputId,
  }) => {
    const assetTimelineDuration = useAssetControlsStore(
      (s) => s.timelineDuration,
    );
    const assetFps = useAssetControlsStore((s) => s.fps);
    const assetZoomLevel = useAssetControlsStore((s) => s.zoomLevel);
    const assetMaxZoomLevel = useAssetControlsStore((s) => s.maxZoomLevel);
    const assetMinZoomLevel = useAssetControlsStore((s) => s.minZoomLevel);
    const inputTimelineDuration = useInputControlsStore((s) =>
      s.getTimelineDuration(inputId),
    );
    const inputFps = useInputControlsStore((s) => s.getFps(inputId));
    const inputZoomLevel = useInputControlsStore((s) =>
      s.getZoomLevel(inputId),
    );
    const inputMaxZoomLevel = useInputControlsStore((s) => s.maxZoomLevel);
    const inputMinZoomLevel = useInputControlsStore((s) => s.minZoomLevel);

    const timelineDuration =
      mode === "asset" ? assetTimelineDuration : inputTimelineDuration;
    const fps = mode === "asset" ? assetFps : inputFps;
    const zoomLevel = mode === "asset" ? assetZoomLevel : inputZoomLevel;
    const maxZoomLevel =
      mode === "asset" ? assetMaxZoomLevel : inputMaxZoomLevel;
    const minZoomLevel =
      mode === "asset" ? assetMinZoomLevel : inputMinZoomLevel;
    const [startFrame, endFrame] = timelineDuration;

    // Convert duration to milliseconds if needed for consistent calculations
    const tickMark: TickMark[] = useMemo(() => {
      let ticks: TickMark[] = [];
      const zoomConfig = getZoomLevelConfig(
        zoomLevel,
        timelineDuration,
        fps,
        maxZoomLevel,
        minZoomLevel,
      );
      const majorTickInterval =
        zoomConfig.majorTickInterval *
        (zoomConfig.majorTickFormat === "second" ? fps : 1);
      const minorTickInterval =
        zoomConfig.minorTickInterval *
        (zoomConfig.minorTickFormat === "second" ? fps : 1);
      const [startFrame, endFrame] = timelineDuration;

      for (let i = startFrame; i <= endFrame; i += majorTickInterval) {
        ticks.push({
          x: Math.round(i),
          text: Math.round(i).toString(),
          type: "major",
          format: getMajorZoomConfigFormat(zoomConfig, fps, majorTickInterval),
        });
      }

      for (let i = startFrame; i <= endFrame; i += minorTickInterval) {
        ticks.push({
          x: Math.round(i),
          text: Math.round(i).toString(),
          type: "minor",
          format: zoomConfig.minorTickFormat,
        });
      }
      ticks.sort((a, b) => a.x - b.x);
      // remove duplicates where x is the same and minor is true
      ticks = ticks.filter(
        (tick, index, self) =>
          index === 0 || tick.x !== self[index - 1].x || tick.type !== "minor",
      );

      return ticks;
    }, [timelineDuration, zoomLevel, fps, maxZoomLevel, minZoomLevel]);

    // Helper function to format tick labels
    const formatTickLabel = (
      value: number,
      format: "frame" | "second",
    ): string => {
      if (format === "frame") {
        return `${value}f`;
      } else {
        // Convert frames to seconds
        const seconds = value / fps;
        const totalSeconds = Math.floor(seconds);
        const minutes = Math.floor(totalSeconds / 60);
        const remainingSeconds = totalSeconds % 60;
        return `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
      }
    };

    return (
      <>
        <Rect
          x={0}
          y={0}
          width={stageWidth}
          height={28}
          fill={maxScroll > 0 && thumbY() > 24 ? "#222124" : undefined}
          listening={false}
        />
        {tickMark.map((tick, index) => {
          // Calculate x position based on timeline progress
          const progress = (tick.x - startFrame) / (endFrame - startFrame);
          const xPosition = progress * stageWidth;

          // Only render ticks that are visible on screen
          if (xPosition < -50 || xPosition > stageWidth + 50) {
            return null;
          }

          const isMajor = tick.type === "major";
          const tickHeight = isMajor ? 16 : 4;
          const tickY = 0;

          return (
            <React.Fragment key={`${tick.x}-${tick.type}-${index}`}>
              {/* Tick line */}
              <Line
                points={[
                  xPosition + startPadding,
                  tickY,
                  xPosition + startPadding,
                  tickY + tickHeight,
                ]}
                stroke={"rgba(227, 227, 227, 0.5)"}
                strokeWidth={1}
                listening={false}
              />

              {/* Label for major ticks only */}
              {isMajor && (
                <Text
                  x={xPosition + 4 + startPadding}
                  y={tickY + tickHeight - 8}
                  text={formatTickLabel(tick.x, tick.format)}
                  fontSize={7.5}
                  fill="rgba(227, 227, 227, 0.7)"
                  fontFamily="Poppins, system-ui, sans-serif"
                  listening={false}
                />
              )}
            </React.Fragment>
          );
        })}
        {topLine && (
          <Line
            points={[startPadding, 0, stageWidth, 0]}
            x={0}
            y={0}
            width={stageWidth}
            height={28}
            stroke={"rgba(150, 150, 150, 1)"}
            strokeWidth={2}
            listening={false}
          />
        )}
      </>
    );
  },
);

export default TimelineMoments;
