import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Mirror native/include/media/control_contract.h
export const MEDIA_NATIVE_CONTROL_ABI_VERSION = 1;

export enum MediaNativeClipKind {
  Unknown = 0,
  Video = 1,
  Image = 2,
  Model = 3,
  Shape = 4,
  Text = 5,
  Drawing = 6,
  Audio = 7,
}

export enum MediaNativeMaskKind {
  Unknown = 0,
  Shape = 1,
  Lasso = 2,
  Touch = 3,
}

export enum MediaNativeCommandType {
  UpsertClip = 1,
  RemoveClip = 2,
  SetPlayState = 3,
  SetPlayhead = 4,
  SetHoleRect = 5,
  SetViewport = 6,
  ResetGraph = 7,
  Shutdown = 8,
}

export interface MediaNativeRectI {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface MediaNativeNormalizedCrop {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface MediaNativeClipTransform {
  x: number;
  y: number;
  width: number;
  height: number;
  scale_x: number;
  scale_y: number;
  rotation_deg: number;
  opacity: number;
  corner_radius: number;
  visible: boolean;
  has_crop: boolean;
  crop: MediaNativeNormalizedCrop;
}

export interface MediaNativeFilterParams {
  brightness: number;
  contrast: number;
  hue: number;
  saturation: number;
  blur: number;
  sharpness: number;
  noise: number;
  vignette: number;
  scan_lines: number;
  chromatic_aberration: number;
  interlace: number;
  pixelate: number;
  jitter: number;
  color_tint_color_hex: string;
  color_tint_intensity: number;
}

export interface MediaNativeLutParams {
  enabled: boolean;
  lut_path: string;
  intensity: number;
}

export interface MediaNativeMaskParams {
  mask_id: string;
  kind: MediaNativeMaskKind;
  enabled: boolean;
  inverted: boolean;
  feather: number;
  payload_json: string;
}

export interface MediaNativeTimelineRange {
  start_frame: number;
  end_frame: number;
  trim_start: number;
  trim_end: number;
  speed: number;
}

export interface MediaNativeUpsertClipCommand {
  clip_id: string;
  clip_kind: MediaNativeClipKind;
  asset_id: string;
  media_path: string;
  timeline: MediaNativeTimelineRange;
  transform: MediaNativeClipTransform;
  filters: MediaNativeFilterParams;
  luts: MediaNativeLutParams[];
  masks: MediaNativeMaskParams[];
  z_index: number;
}

export interface MediaNativeRemoveClipCommand {
  clip_id: string;
}

export interface MediaNativeSetPlayStateCommand {
  is_playing: boolean;
}

export interface MediaNativeSetPlayheadCommand {
  focus_frame: number;
  fps: number;
  accurate_seek: boolean;
}

export interface MediaNativeSetHoleRectCommand {
  rect: MediaNativeRectI;
  visible: boolean;
}

export interface MediaNativeSetViewportCommand {
  width: number;
  height: number;
  scale: number;
  stage_x: number;
  stage_y: number;
}

export type MediaNativeResetGraphCommand = Record<string, never>;
export type MediaNativeShutdownCommand = Record<string, never>;

export type MediaNativeCommandPayload =
  | MediaNativeUpsertClipCommand
  | MediaNativeRemoveClipCommand
  | MediaNativeSetPlayStateCommand
  | MediaNativeSetPlayheadCommand
  | MediaNativeSetHoleRectCommand
  | MediaNativeSetViewportCommand
  | MediaNativeResetGraphCommand
  | MediaNativeShutdownCommand;

export interface MediaNativeCommandEnvelope {
  abi_version: number;
  sequence: number;
  type: MediaNativeCommandType;
  payload: MediaNativeCommandPayload;
}

export interface MediaNativeEngineConfig {
  width?: number;
  height?: number;
  fps?: number;
}

export interface MediaNativeDecodedVideoFrame {
  clipId: string;
  mediaPath: string;
  focusFrame: number;
  sourceFrame: number;
  pts: number;
  width: number;
  height: number;
  pixelFormat: number;
  zIndex: number;
  hasNativeFrame: boolean;
}

export interface MediaNativeDecodedAudioChunk {
  clipId: string;
  mediaPath: string;
  focusFrame: number;
  sourceFrame: number;
  pts: number;
  sampleRate: number;
  channels: number;
  sampleFormat: number;
  nbSamples: number;
  planar: boolean;
  zIndex: number;
  data: Buffer;
}

export interface MediaNativeMixedAudioBlock {
  focusFrame: number;
  sampleRate: number;
  channels: number;
  nbSamples: number;
  data: Buffer;
}

export interface MediaNativeStats {
  exists: boolean;
  id?: number;
  width?: number;
  height?: number;
  fps?: number;
  hasSurface?: boolean;
  lastCommand?: string;
  lastSequence?: number;
  decodeSchedulerTicks?: number;
  decodeSchedulerLastClockFrame?: number;
  clipCount?: number;
  activeClipCount?: number;
  pendingDecodeCount?: number;
  totalDecodeRequestsEnqueued?: number;
  videoExecutorSubmittedRequests?: number;
  videoExecutorProcessedRequests?: number;
  videoExecutorSucceededRequests?: number;
  videoExecutorFailedRequests?: number;
  videoExecutorDroppedRequests?: number;
  videoExecutorQueueDepth?: number;
  imageExecutorSubmittedRequests?: number;
  imageExecutorProcessedRequests?: number;
  imageExecutorCacheHits?: number;
  imageExecutorFailedRequests?: number;
  imageExecutorDroppedRequests?: number;
  imageExecutorQueueDepth?: number;
  imageExecutorCachedImages?: number;
  audioExecutorSubmittedRequests?: number;
  audioExecutorProcessedRequests?: number;
  audioExecutorSucceededRequests?: number;
  audioExecutorFailedRequests?: number;
  audioExecutorDroppedRequests?: number;
  audioExecutorQueueDepth?: number;
  audioMixerSubmittedChunks?: number;
  audioMixerMixedBlocksEnqueued?: number;
  audioMixerMixedBlocksDrained?: number;
  audioMixerDroppedChunks?: number;
  audioMixerInputQueueDepth?: number;
  mixedAudioBlocksDrained?: number;
  audioSinkStarted?: boolean;
  audioSinkPlaying?: boolean;
  audioSinkSampleRate?: number;
  audioSinkChannels?: number;
  audioSinkQueuedSamples?: number;
  audioSinkSubmittedBlocks?: number;
  audioSinkSubmittedSamples?: number;
  audioSinkConsumedSamples?: number;
  audioSinkDroppedSamples?: number;
  audioClockSeconds?: number;
  videoPresenterQueuedFrames?: number;
  videoPresenterPresentedFrames?: number;
  videoPresenterDroppedFrames?: number;
  videoPresenterHasPresentedFrame?: boolean;
  videoPresenterLastFocusFrame?: number;
  videoPresenterLastSourceFrame?: number;
  videoPresenterLastPts?: number;
  videoPresenterLastWidth?: number;
  videoPresenterLastHeight?: number;
  videoPresenterLastPixelFormat?: number;
  videoPresenterLastZIndex?: number;
  videoPresenterTargetFocusFrame?: number;
  videoPresenterAudioClockSeconds?: number;
  holeRendererAttached?: boolean;
  holeRendererVisible?: boolean;
  holeRendererPresentedFrames?: number;
  holeRendererFailedFrames?: number;
  holeRendererWidth?: number;
  holeRendererHeight?: number;
  decodedVideoQueueDepth?: number;
  decodedVideoFramesEnqueued?: number;
  decodedVideoFramesDropped?: number;
  decodedVideoFramesDrained?: number;
  decodedAudioQueueDepth?: number;
  decodedAudioChunksEnqueued?: number;
  decodedAudioChunksDropped?: number;
  decodedAudioChunksDrained?: number;
  demuxWorkerSubmittedRequests?: number;
  demuxWorkerProcessedRequests?: number;
  demuxWorkerSucceededRequests?: number;
  demuxWorkerFailedRequests?: number;
  demuxWorkerDroppedRequests?: number;
  demuxWorkerQueueDepth?: number;
  isPlaying?: boolean;
  focusFrame?: number;
  holeVisible?: boolean;
}

type NativeAddon = {
  createEngine: (config?: MediaNativeEngineConfig) => number;
  destroyEngine: (engineId: number) => boolean;
  attachSurface: (engineId: number, nativeHandle: Buffer) => boolean;
  submitCommand: (
    engineId: number,
    command: MediaNativeCommandEnvelope,
  ) => boolean;
  drainDecodedVideoFrames: (engineId: number) => MediaNativeDecodedVideoFrame[];
  drainDecodedAudioChunks: (engineId: number) => MediaNativeDecodedAudioChunk[];
  drainMixedAudioBlocks: (engineId: number) => MediaNativeMixedAudioBlock[];
  getStats: (engineId: number) => MediaNativeStats;
};

let cachedAddon: NativeAddon | null = null;

function loadAddon(): NativeAddon {
  let addonPath = join(__dirname, "..", "build", "Release", "addon.node");
  addonPath = addonPath.replace("main", "media-native");
  return createRequire(import.meta.url)(addonPath) as NativeAddon;
}

function getAddon(): NativeAddon {
  if (!cachedAddon) {
    cachedAddon = loadAddon();
  }
  return cachedAddon;
}

export class MediaNativeEngine {
  readonly id: number;
  #destroyed = false;
  #sequence = 0;

  constructor(config?: MediaNativeEngineConfig) {
    this.id = getAddon().createEngine(config);
  }

  attachSurface(nativeHandle: Buffer): boolean {
    if (this.#destroyed) return false;
    return getAddon().attachSurface(this.id, nativeHandle);
  }

  submitEnvelope(command: MediaNativeCommandEnvelope): boolean {
    if (this.#destroyed) return false;
    return getAddon().submitCommand(this.id, command);
  }

  submit(type: MediaNativeCommandType, payload: MediaNativeCommandPayload): boolean {
    const command: MediaNativeCommandEnvelope = {
      abi_version: MEDIA_NATIVE_CONTROL_ABI_VERSION,
      sequence: ++this.#sequence,
      type,
      payload,
    };
    return this.submitEnvelope(command);
  }

  drainDecodedVideoFrames(): MediaNativeDecodedVideoFrame[] {
    if (this.#destroyed) return [];
    return getAddon().drainDecodedVideoFrames(this.id);
  }

  drainDecodedAudioChunks(): MediaNativeDecodedAudioChunk[] {
    if (this.#destroyed) return [];
    return getAddon().drainDecodedAudioChunks(this.id);
  }

  drainMixedAudioBlocks(): MediaNativeMixedAudioBlock[] {
    if (this.#destroyed) return [];
    return getAddon().drainMixedAudioBlocks(this.id);
  }

  stats(): MediaNativeStats {
    if (this.#destroyed) {
      return { exists: false };
    }
    return getAddon().getStats(this.id);
  }

  destroy(): boolean {
    if (this.#destroyed) return true;
    this.#destroyed = true;
    return getAddon().destroyEngine(this.id);
  }
}

export function createMediaNativeEngine(config?: MediaNativeEngineConfig) {
  return new MediaNativeEngine(config);
}
