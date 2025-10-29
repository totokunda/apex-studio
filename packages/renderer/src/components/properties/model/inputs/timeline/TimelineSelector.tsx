import InputTimelineClip from "./InputTimelineClip"
import { AnyClipProps } from "@/lib/types";
import {Stage, Layer, Group} from "react-konva";
import TimelineMoments from "./InputTimelineMoments";
import TimelineSelectorZoom from "./TimelineSelectorZoom";
import { useEffect } from "react";
import { useInputControlsStore } from "@/lib/inputControl";

interface TimelineSelectorProps {
    clip: AnyClipProps;
    height: number;
    width: number;
    mode: 'frame' | 'range';
}

const TimelineSelector: React.FC<TimelineSelectorProps> = ({ clip, width, height, mode }) => {
    const setTotalTimelineFrames = useInputControlsStore((s) => s.setTotalTimelineFrames);
    const setTimelineDuration = useInputControlsStore((s) => s.setTimelineDuration);
    const setZoomLevel = useInputControlsStore((s) => s.setZoomLevel);

    useEffect(() => {
        if (!clip) return;
        const start = Math.max(0, Math.round((clip as any)?.startFrame ?? 0));
        const endRaw = Math.max(start + 1, Math.round((clip as any)?.endFrame ?? (start + 1)));
        const span = Math.max(1, endRaw - start);
        // Ensure max zoom out equals the clip's actual span
        setTotalTimelineFrames(span);
        setTimelineDuration(start, start + span);
        setZoomLevel(1 as any);
    }, [clip, setTotalTimelineFrames, setTimelineDuration, setZoomLevel]);

    return (
        <div className="relative w-full h-full flex flex-col gap-y-3 mb-4 mt-4 ">
            <div className="z-10 flex flex-row gap-x-2 w-full items-center justify-between ">
                <div className="text-[10px] font-medium text-brand-light">
                    {mode === 'frame' ? 'Select Frame' : 'Selct FrameRange'}
                </div>
                <TimelineSelectorZoom hasClip={!!clip} />
            </div>
            <Stage width={width} height={height + 24} className=" shadow">
                <Layer>
                    <Group>
                        <InputTimelineClip 
                            timelineId="timeline-selector"
                            clip={clip}
                            cornerRadius={3}
                            timelineWidth={width}
                            timelineHeight={height}
                            timelineY={height + 24}
                            timelinePadding={0}
                            muted={false}
                            mode={mode}
                            hidden={false}
                            type='media'
                        />
                    </Group>
                </Layer>
                <Layer listening={false}>
                <TimelineMoments stageWidth={width}  startPadding={0.5} maxScroll={0} thumbY={() => {
                    return 0;
                }} mode="input" />
                </Layer>
            </Stage>
        </div>
    );
}

export default TimelineSelector;