import { CanvasSink, AudioBufferSink } from "mediabunny";

export type VideoDecoderKey = string; // `${path}#video`
export type CanvasDecoderKey = string; // `${path}#canvas@${width}x${height}`
export type AudioDecoderKey = string; // `${path}#audio`

export interface VideoDecoderContext {
  sink: CanvasSink;
  inFlight: Set<number>;
  lastAccessTs: number;
  frameRate: number;
}

export interface CanvasDecoderContext {
  sink: CanvasSink;
  inFlight: Set<number>;
  lastAccessTs: number;
  frameRate: number;
  width: number;
  height: number;
}

export interface AudioDecoderContext {
  sink: AudioBufferSink;
  inFlight: Set<number>;
  lastAccessTs: number;
  sampleRate: number;
}
