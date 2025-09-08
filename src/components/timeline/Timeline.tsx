import React, { useMemo } from "react";
import { Rect,  Line } from "react-konva";
import VideoTimeline from "./clips/VideoTimeline";
import { TimelineProps } from "@/lib/types";
import { useClipStore } from "@/lib/clip";


    
const Timeline:React.FC<TimelineProps> = ({timelineWidth, timelineY, timelineHeight = 72, timelinePadding = 24, timelineId}) => {

    const timelineYBottom = useMemo(() => {
        return timelineHeight + 12;
    }, [timelineHeight]);

    const {getClipsForTimeline} = useClipStore();
    const clips = getClipsForTimeline(timelineId);

    return (
        <React.Fragment>
            <Rect x={timelinePadding} y={timelineY! + 8} cornerRadius={8} width={timelineWidth} height={timelineHeight}/>
            <Line x={timelinePadding} y={timelineYBottom} points={[0, 0, timelineWidth!, 0]} stroke={'rgba(255,255,255,0.1)'} strokeWidth={1} />
            {clips.map((clip) => (
                <VideoTimeline 
                    key={clip.clipId} 
                    timelineId={timelineId}
                    clipId={clip.clipId}
                    timelinePadding={timelinePadding}
                    timelineWidth={timelineWidth} 
                    timelineY={timelineYBottom} 
                    timelineHeight={timelineHeight - 16} 
                />
            ))}
        </React.Fragment>
    )
}

export default Timeline;