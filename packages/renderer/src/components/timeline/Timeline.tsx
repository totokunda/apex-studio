import React, {useMemo, useEffect } from "react";
import { Rect,  Line,  Group } from "react-konva";
import { TimelineProps } from "@/lib/types";
import { useClipStore, getTimelineX } from "@/lib/clip";
import TimelineClip from "./clips/TimelineClip";
import GhostTimeline from "./clips/GhostTimeline";
import { useControlsStore } from "@/lib/control";


const Timeline:React.FC<TimelineProps & {index: number, scrollY: number}> = ({timelineWidth, timelineY, timelineHeight = 72, timelinePadding = 24, timelineId, index, scrollY, type, muted, hidden}) => {
    const {hoveredTimelineId, removeTimeline, getTimelineById} = useClipStore();
    const {setFocusFrame, setZoomLevel} = useControlsStore();

    const timelineYBottom = useMemo(() => {
        return timelineHeight + (timelineY ?? 0) + 24;
    }, [timelineHeight, timelineY, timelineId]);

    // Default height for the under-timeline group
    const underGroupHeight = 14;

    // Compute exact edges of the visible timeline rect
    const timelineTopY = (timelineY ?? 0) + 40;
    const timelineBottomY = timelineTopY + (timelineHeight - 16);

    // Desired consistent gap between the timeline and hover lines (in px)
    const hoverGap = underGroupHeight / 2; // keep existing visual gap

    // Position groups so the line inside (drawn at underGroupHeight/2) sits exactly at the desired gap
    const topDashGroupY = timelineTopY - hoverGap - underGroupHeight / 2;
    const bottomDashGroupY = timelineBottomY + hoverGap - underGroupHeight / 2;

    const {getClipsForTimeline, timelines} = useClipStore();
    const {timelineDuration} = useControlsStore();
    const clips = getClipsForTimeline(timelineId);
    
    useEffect(() => {
        if (clips.length === 0 && getTimelineById(timelineId)) {
            // delete the timeline
            removeTimeline(timelineId);
            if (timelines.length === 1) {
                setFocusFrame(0);
                setZoomLevel(1);
            }
        }
    },[timelineId, clips]);

    const timelineX = useMemo(() => getTimelineX(timelineWidth!, timelinePadding, timelineDuration), [timelineWidth, timelinePadding, timelineDuration]);

    return (
        <>
           {
            index === 0 && (
                <Group id={`dashed-${timelineId}-top`} name={'timeline-dashed'} height={underGroupHeight} x={timelinePadding} y={topDashGroupY}>
                    <Line 
                        points={[0, underGroupHeight / 2, timelineWidth!, underGroupHeight / 2]} 
                        stroke={'rgba(174, 129, 206, 0.75)'} 
                        strokeWidth={hoveredTimelineId === `dashed-${timelineId}-top` ? 1.2 : 0} 
                    />
                </Group>
            )
           }
            <Rect
            id={timelineId} x={timelineX} y={timelineY! + 40} cornerRadius={8} width={timelineWidth! - (timelineX) + 8} height={timelineHeight - 16} fill={'rgba(11, 11, 13, 0.25)'}/>
            
            {clips.map((clip) => (
                    <TimelineClip 
                            key={clip.clipId} 
                            muted={muted}
                            hidden={hidden}
                            timelineId={timelineId}
                            clipId={clip.clipId}
                            timelinePadding={timelinePadding}
                            timelineWidth={timelineWidth} 
                            timelineY={timelineYBottom} 
                            timelineHeight={timelineHeight - 16} 
                            clipType={clip.type}
                            type={type}
                            scrollY={scrollY}
                        />
                    )
                )}
            <GhostTimeline timelineId={timelineId} timelineY={timelineYBottom} timelineHeight={timelineHeight - 16} timelinePadding={timelinePadding} timelineWidth={timelineWidth} type={type} muted={muted} hidden={hidden} />
            <Group id={`dashed-${timelineId}`} name={'timeline-dashed'} height={underGroupHeight} x={timelinePadding} y={bottomDashGroupY}>
                <Line 
                    points={[0, underGroupHeight / 2, timelineWidth!, underGroupHeight / 2]} 
                    stroke={'rgba(174, 129, 206, 0.75)'} 
                    strokeWidth={hoveredTimelineId === `dashed-${timelineId}` ? 1.2 : 0} 
                />
            </Group>
            {(hidden || (muted && type === 'audio')) && (
                <Rect
                id={`hidden-${timelineId}`} x={timelineX} y={timelineY! + 40} cornerRadius={8} width={timelineWidth! - timelineX + 8} height={timelineHeight - 16} fill={'rgba(11, 11, 13, 0.60)'}/>
            )}
        </>
    )
}

export default Timeline;