// ─── Shared types for renderer ↔ decode worker_thread ───

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

// ═══════════════════════════════════════════════════════════════════════════
//  Ring buffer layout for lock-free playback
//
//  During playback, the worker decodes ahead into a ring buffer of N slots.
//  Each slot holds one RGBA frame. The renderer consumes slots at display
//  rate. Coordination uses Atomics on a small control SharedArrayBuffer.
//
//  This eliminates per-frame postMessage round-trips during playback.
//  The worker just writes frames as fast as it can (up to the buffer limit),
//  and the renderer reads them at vsync rate.
// ═══════════════════════════════════════════════════════════════════════════

/** Number of frame slots in the ring buffer. 3 is enough for smooth playback
 *  without excessive memory usage. At 4K RGBA that's 3 × ~33MB = ~100MB. */
export const RING_SLOTS = 3;

/**
 * Layout of the Int32Array control buffer (backed by SharedArrayBuffer).
 *
 * [0] writeIndex  — next slot the worker will write to (mod RING_SLOTS)
 * [1] readIndex   — next slot the renderer will read from (mod RING_SLOTS)
 * [2] state       — PlaybackState enum
 * [3] slotReady0  — 1 if slot 0 has a decoded frame, 0 if consumed/empty
 * [4] slotReady1  — 1 if slot 1 has a decoded frame
 * [5] slotReady2  — 1 if slot 2 has a decoded frame
 */
export const CTRL_WRITE_IDX = 0;
export const CTRL_READ_IDX = 1;
export const CTRL_STATE = 2;
export const CTRL_SLOT_READY_BASE = 3;
export const CTRL_BUFFER_SIZE = CTRL_SLOT_READY_BASE + RING_SLOTS; // 6

export const enum PlaybackState {
  IDLE = 0,
  PLAYING = 1,
  PAUSED = 2,
  SEEKING = 3,
  EOS = 4,
}

/**
 * Per-slot timestamp array layout.
 * Float64Array so we get full double precision for PTS values.
 * timestamps[slotIndex] = timestamp in seconds of the frame in that slot.
 */
export const TIMESTAMP_BUFFER_SIZE = RING_SLOTS; // 3 Float64s

// ═══════════════════════════════════════════════════════════════════════════
//  Message types (only used for commands + one-shot decodes, NOT per-frame)
// ═══════════════════════════════════════════════════════════════════════════

// ─── Renderer → Worker ───

export type LoadMsg = {
  type: "load";
  id: number;
  decoderId: string;
  filePath: string;
};

export type DecodeFrameMsg = {
  type: "decodeFrame";
  id: number;
  decoderId: string;
  timestamp: number;
  keyframeOnly?: boolean;
};

export type PlayMsg = {
  type: "play";
  decoderId: string;
  startTime: number;
  endTime: number;
};

export type PauseMsg = {
  type: "pause";
  decoderId: string;
};

export type DestroyMsg = {
  type: "destroy";
  id: number;
  decoderId: string;
};

export type WorkerRequest = LoadMsg | DecodeFrameMsg | PlayMsg | PauseMsg | DestroyMsg;

// ─── Worker → Renderer ───

export type LoadedReply = {
  type: "loaded";
  id: number;
  decoderId: string;
  info: FileInfo;
  width: number;
  height: number;
};

export type PlaybackFrameReply = {
  type: "playbackFrame";
  decoderId: string;
  timestamp: number;
  frameData: ArrayBuffer;
};

export type PlaybackEosReply = {
  type: "playbackEos";
  decoderId: string;
};

export type FrameReply = {
  type: "frame";
  id: number;
  decoderId: string;
  timestamp: number;
  frameData: ArrayBuffer;
};

export type ErrorReply = {
  type: "error";
  id: number;
  decoderId: string;
  message: string;
};

export type DestroyedReply = {
  type: "destroyed";
  id: number;
  decoderId: string;
};

export type WorkerReply =
  | LoadedReply
  | FrameReply
  | PlaybackFrameReply
  | PlaybackEosReply
  | ErrorReply
  | DestroyedReply;
