import React, { useMemo } from "react";
import { Rect,  Line } from "react-konva";
import VideoClip from "../clips/VideoClip";
import { VideoClipProps } from "@/lib/types";
import { useClipStore } from "@/lib/clip";

export interface TimelineProps {
    stageWidth:number;
    timelineNumber:number;
    timelineHeight?:number;
    timelinePadding?:number;
}
    
const Timeline:React.FC<TimelineProps> = ({stageWidth, timelineNumber, timelineHeight = 72, timelinePadding = 24}) => {

    const timelineY = useMemo(() => {
        return timelineNumber * timelineHeight;
    }, [timelineNumber]);

    const timelineYBottom = useMemo(() => {
        return timelineHeight + 12;
    }, [timelineHeight]);

    const {clips} = useClipStore();

    return (
        <React.Fragment>
            <Rect x={timelinePadding} y={timelineY + 8} cornerRadius={8} width={stageWidth} height={timelineHeight}/>
            <Line x={timelinePadding} y={timelineYBottom} points={[0, 0, stageWidth, 0]} stroke={'rgba(255,255,255,0.1)'} strokeWidth={1} />
            {clips.map((clip) => (
                <VideoClip 
                    key={clip.clipId} 
                    clipId={clip.clipId}
                    src={(clip as VideoClipProps).src} 
                    timelineWidth={stageWidth} 
                    timelineY={timelineYBottom} 
                    timelineHeight={timelineHeight - 16} 
                    startFrame={clip.startFrame} 
                    endFrame={clip.endFrame} 
                />
            ))}
        </React.Fragment>
    )
}

export default Timeline;