/**
 * Shared message protocol for the demux + decode worker farm.
 *
 * This file is intentionally runtime-agnostic and should be imported by:
 * - renderer coordinator/orchestrator code
 * - demux worker
 * - decode worker(s)
 */

export type SourceId = string;
export type LogicalClipId = string;
export type SessionId = string;
export type JobId = number;
export type RequestId = number;

export type DecodePriority = "realtime" | "interactive" | "background";

export type SourceDescriptor = {
  sourceId: SourceId;
  assetId: string;
  path: string;
  folderUuid?: string;
  userDataPath?: string;
  formatStr?: string;
};

export type SerializedEncodedChunk = {
  type: EncodedVideoChunkType;
  timestamp: number; // microseconds
  duration?: number; // microseconds
  data: ArrayBuffer;
};

export type DemuxPacketEnvelope = {
  sourceId: SourceId;
  requestId: RequestId;
  jobId: JobId;
  timestampSec: number;
  durationSec: number;
  isKey: boolean;
  chunk: SerializedEncodedChunk;
  alphaChunk?: SerializedEncodedChunk;
};

export type DemuxWorkerRequest =
  | {
      type: "registerSource";
      requestId: RequestId;
      source: SourceDescriptor;
    }
  | {
      type: "disposeSource";
      requestId: RequestId;
      sourceId: SourceId;
    }
  | {
      type: "getKeyPacketAt";
      requestId: RequestId;
      sourceId: SourceId;
      timestampSec: number;
    }
  | {
      type: "streamPackets";
      requestId: RequestId;
      sourceId: SourceId;
      jobId: JobId;
      startTimeSec: number;
      endTimeSec: number;
      startAtKeyframe: boolean;
      maxPackets?: number;
      priority?: DecodePriority;
    }
  | {
      type: "cancelJob";
      requestId: RequestId;
      sourceId: SourceId;
      jobId: JobId;
    };

export type DemuxWorkerResponse =
  | {
      type: "sourceReady";
      requestId: RequestId;
      sourceId: SourceId;
    }
  | {
      type: "keyPacket";
      requestId: RequestId;
      sourceId: SourceId;
      packet: DemuxPacketEnvelope;
    }
  | {
      type: "packets";
      requestId: RequestId;
      sourceId: SourceId;
      jobId: JobId;
      packets: DemuxPacketEnvelope[];
    }
  | {
      type: "streamDone";
      requestId: RequestId;
      sourceId: SourceId;
      jobId: JobId;
    }
  | {
      type: "jobCancelled";
      requestId: RequestId;
      sourceId: SourceId;
      jobId: JobId;
    }
  | {
      type: "error";
      requestId: RequestId;
      sourceId?: SourceId;
      jobId?: JobId;
      error: string;
    }
  | {
      type: "debug";
      event: string;
      sourceId?: SourceId;
      requestId?: RequestId;
      payload?: unknown;
    };

export type DecodeSessionDescriptor = {
  sessionId: SessionId;
  logicalClipId: LogicalClipId;
  sourceId: SourceId;
  decoderConfig: VideoDecoderConfig;
  priority?: DecodePriority;
};

export type DecodeWorkerRequest =
  | {
      type: "createSession";
      requestId: RequestId;
      session: DecodeSessionDescriptor;
    }
  | {
      type: "resetSession";
      requestId: RequestId;
      sessionId: SessionId;
    }
  | {
      type: "flushSession";
      requestId: RequestId;
      sessionId: SessionId;
    }
  | {
      type: "decodeChunk";
      requestId: RequestId;
      sessionId: SessionId;
      sourceId: SourceId;
      jobId: JobId;
      packet: DemuxPacketEnvelope;
    }
  | {
      type: "cancelJob";
      requestId: RequestId;
      sessionId: SessionId;
      jobId: JobId;
    }
  | {
      type: "disposeSession";
      requestId: RequestId;
      sessionId: SessionId;
    };

export type DecodeWorkerResponse =
  | {
      type: "sessionReady";
      requestId: RequestId;
      sessionId: SessionId;
      logicalClipId: LogicalClipId;
      sourceId: SourceId;
    }
  | {
      type: "resetDone";
      requestId: RequestId;
      sessionId: SessionId;
      sourceId: SourceId;
    }
  | {
      type: "flushDone";
      requestId: RequestId;
      sessionId: SessionId;
      sourceId: SourceId;
    }
  | {
      type: "frame";
      requestId: RequestId;
      sessionId: SessionId;
      logicalClipId: LogicalClipId;
      sourceId: SourceId;
      jobId: JobId;
      frame: VideoFrame;
      timestampSec: number;
      durationSec: number;
    }
  | {
      type: "decodeDone";
      requestId: RequestId;
      sessionId: SessionId;
      sourceId: SourceId;
      jobId: JobId;
    }
  | {
      type: "jobCancelled";
      requestId: RequestId;
      sessionId: SessionId;
      sourceId: SourceId;
      jobId: JobId;
    }
  | {
      type: "error";
      requestId: RequestId;
      sessionId?: SessionId;
      sourceId?: SourceId;
      jobId?: JobId;
      error: string;
    }
  | {
      type: "debug";
      event: string;
      sessionId?: SessionId;
      sourceId?: SourceId;
      requestId?: RequestId;
      payload?: unknown;
    };
