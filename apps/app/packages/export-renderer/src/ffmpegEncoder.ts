import type { FrameEncoder } from "./exporter";
import {
  exportVideoOpen,
  exportVideoAppend,
  exportVideoClose,
  exportVideoAbort,
} from "@app/preload";

export type FfmpegEncoderOptionsNoFilename = {
  format?: "mp4" | "mov" | "mkv" | "webm";
  codec?: "h264" | "hevc" | "prores" | "vp9" | "av1";
  preset?:
    | "ultrafast"
    | "superfast"
    | "veryfast"
    | "faster"
    | "fast"
    | "medium"
    | "slow"
    | "slower"
    | "veryslow";
  crf?: number;
  bitrate?: string; // e.g. '8M'
  resolution?: { width: number; height: number };
  alpha?: boolean; // preserve transparency
};

export type FfmpegEncoderOptions = {
  filename: string;
} & FfmpegEncoderOptionsNoFilename;

export class FfmpegFrameEncoder implements FrameEncoder {
  private options: FfmpegEncoderOptions;
  private sessionId: string | null = null;

  constructor(options: FfmpegEncoderOptions) {
    this.options = options;
  }

  async start(opts: {
    width: number;
    height: number;
    fps: number;
    audioPath?: string;
  }): Promise<void> {
    const { filename, format, codec, preset, crf, bitrate, resolution, alpha } =
      this.options;
    const { sessionId } = await exportVideoOpen({
      filename,
      format,
      codec,
      preset,
      crf,
      bitrate,
      resolution,
      alpha,
      width: opts.width,
      height: opts.height,
      fps: opts.fps,
      audioPath: opts.audioPath,
    });
    this.sessionId = sessionId;
  }

  async addFrame(buffer: Uint8Array): Promise<void> {
    if (!this.sessionId)
      throw new Error("FfmpegFrameEncoder: start() not called");
    await exportVideoAppend(this.sessionId, buffer);
  }

  async finalize(): Promise<string> {
    if (!this.sessionId)
      throw new Error("FfmpegFrameEncoder: start() not called");
    try {
      const abs = await exportVideoClose(this.sessionId);
      this.sessionId = null;
      return abs;
    } catch (err) {
      const sid = this.sessionId;
      try {
        if (sid) await exportVideoAbort(sid);
      } catch {}
      this.sessionId = null;
      throw err;
    }
  }
}
