import InputTimelineClip from "./InputTimelineClip";
import { AnyClipProps } from "@/lib/types";
import { Stage, Layer, Group } from "react-konva";
import TimelineMoments from "./InputTimelineMoments";
import TimelineSelectorZoom from "./TimelineSelectorZoom";
import { useEffect } from "react";
import { useInputControlsStore } from "@/lib/inputControl";

export interface TimelineSelectorProps {
  clip: AnyClipProps;
  height: number;
  width: number;
  mode: "frame" | "range";
  inputId: string;
  // Optional max duration (in frames) for this input's selectable range
  maxDuration?: number;
}

const TimelineSelector: React.FC<TimelineSelectorProps> = ({
  clip,
  width,
  height,
  mode,
  inputId,
  maxDuration,
}) => {
  const setTotalTimelineFrames = useInputControlsStore(
    (s) => s.setTotalTimelineFrames,
  );
  const setTimelineDuration = useInputControlsStore(
    (s) => s.setTimelineDuration,
  );
  const setZoomLevel = useInputControlsStore((s) => s.setZoomLevel);

  useEffect(() => {
    if (!clip) return;
    const start = Math.max(0, Math.round((clip as any)?.startFrame ?? 0));
    const endRaw = Math.max(
      start + 1,
      Math.round((clip as any)?.endFrame ?? start + 1),
    );
    const span = Math.max(1, endRaw - start);
    // Ensure max zoom out equals the clip's actual span
    setTotalTimelineFrames(span, inputId);
    // Keep the input timeline local to the clip [0, span]
    setTimelineDuration(0, span, inputId);
    setZoomLevel(1 as any, inputId);
  }, [clip, inputId]);

  if (width < 3 || height < 3) return null;

  return (
    <div className="relative w-full flex flex-col gap-y-3 mb-4 ">
      <div className="z-10 flex flex-row gap-x-2 w-full items-center justify-between pr-1.5">
        <div className="text-[10px] font-medium text-brand-light">
          {mode === "frame" ? "Select Frame" : "Select Frame Range"}
        </div>
        <TimelineSelectorZoom hasClip={!!clip} inputId={inputId} mode={mode} />
      </div>
      <Stage width={width + 4} height={height + 30} className=" shadow">
        <Layer>
          <Group x={2}>
            <InputTimelineClip
              inputId={inputId}
              timelineId="timeline-selector"
              clip={clip}
              cornerRadius={3}
              timelineWidth={width}
              timelineHeight={height}
              timelineY={height + 28}
              timelinePadding={0}
              muted={false}
              mode={mode}
              hidden={false}
              type="media"
              maxDuration={maxDuration}
            />
          </Group>
        </Layer>
        <Layer listening={false}>
          <TimelineMoments
            stageWidth={width}
            startPadding={0.5}
            maxScroll={0}
            thumbY={() => {
              return 0;
            }}
            mode="input"
            inputId={inputId}
          />
        </Layer>
      </Stage>
    </div>
  );
};

export default TimelineSelector;
