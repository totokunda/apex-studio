
export interface ClipProps {
    timelineWidth?: number;
    timelineY?: number;
    timelineHeight?:number;
    startFrame?:number;
    endFrame?:number;
    framesToGiveEnd?:number;
    framesToGiveStart?:number;
    clipPadding?:number;
    clipId: string;
}

export type VideoClipProps = ClipProps & {
    src: string;
}

export type ZoomLevel = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;
