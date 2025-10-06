import { PacketStats, InputVideoTrack, InputAudioTrack, MetadataTags, InputFormat } from "mediabunny";

export type ClipType = 'video' | 'image' | 'audio' | 'model' | 'processor' | 'mask' | 'text' | 'lora' | 'shape' | 'draw' | 'filter'
export type TimelineType = 'media' | 'audio' | 'model' | 'processor' | 'mask' | 'text' | 'lora' | 'shape' | 'draw' | 'filter'
export type ViewTool = 'pointer' | 'hand' | 'mask' | 'draw' | 'shape'| 'text'
export type ShapeTool = 'rectangle' | 'ellipse' | 'polygon' | 'line' | 'star'

export interface MediaAdjustments {
    // Color Correction
    brightness?: number; // isFilter    
    contrast?: number; // isFilter
    exposure?: number;
    hue?: number; // isFilter
    saturation?: number; // isFilter
    // Effects
    sharpness?: number;
    noise?: number; // isFilter
    blur?: number; // isFilter
    vignette?: number;
}

export interface ClipTransform {
    x: number;
    y: number;
    width: number;
    height: number;
    scaleX: number;
    scaleY: number;
    rotation: number;
    cornerRadius: number;
    opacity: number;
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


export type VideoClipProps = ClipProps & MediaAdjustments & {
    src: string;
    type: 'video';
    volume?: number;
    fadeIn?: number;
    fadeOut?: number;
    speed?: number;
}

export type ImageClipProps = ClipProps & MediaAdjustments & {
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
    fillOpacity?: number;
    stroke?: string;
    strokeOpacity?: number;
    strokeWidth?: number;
}

export type PolygonClipProps = ShapeClipProps & {
    shapeType: 'polygon';
    sides?: number;
}

export type TextClipProps = ClipProps & {
    src: null | undefined;
    type: 'text';
    text?: string;
    fontSize?: number;
    fontWeight?: number;
    fontStyle?: 'normal' | 'italic';
    fontFamily?: string;
    color?: string;
    colorOpacity?: number;
    textAlign?: 'left' | 'center' | 'right';
    verticalAlign?: 'top' | 'middle' | 'bottom';
    textTransform?: 'none' | 'uppercase' | 'lowercase' | 'capitalize';
    textDecoration?: 'none' | 'underline' | 'overline' | 'line-through';
    // Stroke properties
    strokeEnabled?: boolean;
    stroke?: string;
    strokeWidth?: number;
    strokeOpacity?: number;
    // Shadow properties
    shadowEnabled?: boolean;
    shadowColor?: string;
    shadowOpacity?: number;
    shadowBlur?: number;
    shadowOffsetX?: number;
    shadowOffsetY?: number;
    shadowOffsetLocked?: boolean;
    // Background properties
    backgroundEnabled?: boolean;
    backgroundColor?: string;
    backgroundOpacity?: number;
    backgroundCornerRadius?: number;
}

export type FilterClipProps = ClipProps & {
    src: null | undefined;
    name?: string;
    type: 'filter';
    smallPath?: string;
    fullPath?: string;
    category?: string;
    examplePath?: string;
    exampleAssetUrl?: string;
}

export type AnyClipProps = VideoClipProps | ImageClipProps | AudioClipProps | ShapeClipProps | PolygonClipProps | TextClipProps | FilterClipProps;

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

export type SidebarSection = 'media' | 'models' | 'filters' | 'loras' | 'templates';


export interface Filter {
    id: string;
    name: string;
    smallPath: string;
    fullPath: string;
    category: string;
    examplePath: string;
    exampleAssetUrl: string;
}

export type FilterWithType = Filter & {
    type: 'filter';
}