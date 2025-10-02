import { PacketStats, InputVideoTrack, InputAudioTrack, MetadataTags, InputFormat } from "mediabunny";

export type ClipType = 'video' | 'image' | 'audio' | 'model' | 'processor' | 'mask' | 'text' | 'lora' | 'shape' | 'draw'
export type TimelineType = 'media' | 'audio' | 'model' | 'processor' | 'mask' | 'text' | 'lora' | 'shape' | 'draw'
export type ViewTool = 'pointer' | 'hand' | 'mask' | 'draw' | 'shape'| 'text'
export type ShapeTool = 'rectangle' | 'ellipse' | 'polygon' | 'line' | 'star'

export interface ClipTransform {
    x: number;
    y: number;
    width: number;
    height: number;
    scaleX: number;
    scaleY: number;
    rotation: number;
    // Shape-specific properties
   
}

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
    // Persisted transform for preview canvas (position/size/scale/rotation)
    transform?: ClipTransform;
    type: ClipType;
    
}

export interface TimelineProps {
    type: TimelineType;
    timelineId: string;
    timelineWidth?: number;
    timelineY?: number;
    timelineHeight?:number;
    timelinePadding?:number;
    muted: boolean;
    hidden: boolean;
}


export type VideoClipProps = ClipProps & {
    src: string;
    type: 'video';
    volume?: number;
    fadeIn?: number;
    fadeOut?: number;
    speed?: number;
}

export type ImageClipProps = ClipProps & {
    src: string;
    type: 'image';
}

export type AudioClipProps = ClipProps & {
    src: string;
    type: 'audio';
    volume?: number;
    fadeIn?: number;
    fadeOut?: number;
    speed?: number;
}

export type ShapeClipProps = ClipProps & {
    src: null | undefined;
    type: 'shape';
    shapeType?: ShapeTool;
    fill?: string;
    stroke?: string;
    strokeWidth?: number;
}

export type AnyClipProps = VideoClipProps | ImageClipProps | AudioClipProps | ShapeClipProps;

export type ZoomLevel = number;

export type InputImageTrack = {
    width: number;
    height: number;
    mime?: string;
    size?: number;           // bytes (if Blob/File provided or fetched)
    animated?: boolean;      // GIF/WebP/AVIF (if ImageDecoder path)
    orientation?: 1|2|3|4|5|6|7|8; // EXIF tag 0x0112 (JPEG only here)
};

export type MediaInfo = {
    path: string;
    video: InputVideoTrack | null;
    audio: InputAudioTrack & { sampleSize?: number } | null;
    image: InputImageTrack | null;
    stats: {
        video: PacketStats | undefined;
        audio: PacketStats | undefined;
    };
    duration: number | undefined;
    metadata: MetadataTags | undefined;
    mimeType: string | undefined;
    format: InputFormat | undefined;
    startFrame?: number;
    endFrame?: number;
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

export type SidebarSection = 'media' | 'models' | 'effects' | 'loras' | 'templates';
