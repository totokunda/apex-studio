import { PacketStats, InputVideoTrack, InputAudioTrack, MetadataTags, InputFormat } from "mediabunny";

export type ClipType = 'video' | 'image';

export interface ClipProps {

    // May be less relevant when adding more timelines, might just store these separately
    timelineId?: string;
    timelineWidth?: number;
    timelineY?: number;
    timelineHeight?:number;
    

    startFrame?:number;
    endFrame?:number;
    framesToGiveEnd?:number;
    framesToGiveStart?:number;
    clipPadding?:number;
    clipId: string;
    width?: number;
    height?: number;
    type: ClipType;
    
}

export interface TimelineProps {
    timelineId: string;
    timelineWidth?: number;
    timelineY?: number;
    timelineHeight?:number;
    timelinePadding?:number;
}


export type VideoClipProps = ClipProps & {
    src: string;
    type: 'video';
}

export type ImageClipProps = ClipProps & {
    src: string;
    type: 'image';
}

export type AnyClipProps = VideoClipProps | ImageClipProps;

export type ZoomLevel = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;

export type MediaInfo = {
    path: string;
    video: InputVideoTrack | null;
    audio: InputAudioTrack | null;
    stats: {
        video: PacketStats | undefined;
        audio: PacketStats | undefined;
    };
    duration: number | undefined;
    metadata: MetadataTags | undefined;
    mimeType: string | undefined;
    format: InputFormat | undefined;
}

export type FrameBatch = {
    path: string;
    start_frame: number;
    end_frame: number;
    width: number;
    height: number;
    pixel_format: 'rgba8';
    frames: number[][] | Uint8Array[];
}


export interface Point {
    x: number;
    y: number;
  }