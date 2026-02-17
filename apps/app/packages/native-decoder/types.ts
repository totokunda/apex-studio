// ─── Shared types for IPC between renderer and main process ───

export type FileInfo = {
    format: string;
    duration: number;
    bitrate: number;
    nb_streams: number;
    hw_accelerated: boolean;
    decode_backend: "videotoolbox_direct" | "ffmpeg_hwaccel" | "ffmpeg_software";
    video?: {
      width: number;
      height: number;
      codec: string;
      pixel_format: string;
      fps: number;
      stream_index: number;
    };
    audio?: {
      codec: string;
      sample_rate: number;
      channels: number;
      stream_index: number;
    };
  };
  
  // ─── IPC Channel names ───
  
  export const IPC = {
    LOAD: "native-decoder:load",
    DECODE_FRAME: "native-decoder:decodeFrame",
    DECODE_NEXT: "native-decoder:decodeNextFrame",
    DESTROY: "native-decoder:destroy",
    DESTROY_ALL: "native-decoder:destroyAll",
  } as const;
  
  // ─── IPC Payloads ───
  
  export type LoadRequest = { decoderId: string; filePath: string };
  export type LoadResponse = { info: FileInfo; width: number; height: number };
  
  export type DecodeFrameRequest = {
    decoderId: string;
    timestamp: number;
    keyframeOnly?: boolean;
  };
  export type DecodeFrameResponse = { timestamp: number } | null;
  
  export type DecodeNextFrameRequest = {
    decoderId: string;
    startTime?: number;
    endTime?: number;
  };
  export type DecodeNextFrameResponse = { timestamp: number } | null;
