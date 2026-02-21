import { ipcRenderer } from "electron";

export interface PreviewHoleRect {
  left: number;
  top: number;
  width: number;
  height: number;
  visible: boolean;
}

export interface MediaNativePreviewPlayheadPayload {
  focusFrame: number;
  fps: number;
  accurateSeek?: boolean;
}

export interface MediaNativePreviewPlayStatePayload {
  isPlaying: boolean;
}

export interface MediaNativePreviewStatsResponse {
  ok: boolean;
  skipped?: string;
  error?: string;
  stats?: Record<string, unknown>;
}

export interface MediaNativePreviewClipTransformPayload {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  scaleX?: number;
  scaleY?: number;
  rotation?: number;
  opacity?: number;
  cornerRadius?: number;
  crop?: {
    x?: number;
    y?: number;
    width?: number;
    height?: number;
  };
}

export interface MediaNativePreviewClipSyncPayload {
  clipId: string;
  clipType: string;
  assetId?: string;
  mediaPath?: string;
  timelineId?: string;
  startFrame?: number;
  endFrame?: number;
  trimStart?: number;
  trimEnd?: number;
  speed?: number;
  hidden?: boolean;
  transform?: MediaNativePreviewClipTransformPayload;
  adjustments?: {
    brightness?: number;
    contrast?: number;
    hue?: number;
    saturation?: number;
    blur?: number;
    sharpness?: number;
    noise?: number;
    vignette?: number;
    scanLines?: number;
    chromaticAberration?: number;
    interlace?: number;
    pixelate?: number;
    jitter?: number;
    colorTintColor?: string;
    colorTintIntensity?: number;
  };
  zIndex?: number;
}

export async function startPreviewHoleVideo(videoPath: string) {
  return await ipcRenderer.invoke("media-native-preview:start", { videoPath });
}

export async function stopPreviewHoleVideo() {
  return await ipcRenderer.invoke("media-native-preview:stop");
}

export async function setPreviewHoleRect(rect: PreviewHoleRect) {
  return await ipcRenderer.invoke("media-native-preview:set-rect", rect);
}

export async function upsertMediaNativePreviewClip(
  clip: MediaNativePreviewClipSyncPayload,
) {
  return await ipcRenderer.invoke("media-native-preview:upsert-clip", clip);
}

export async function removeMediaNativePreviewClip(clipId: string) {
  return await ipcRenderer.invoke("media-native-preview:remove-clip", {
    clipId,
  });
}

export async function setMediaNativePreviewPlayhead(
  payload: MediaNativePreviewPlayheadPayload,
) {
  return await ipcRenderer.invoke("media-native-preview:set-playhead", payload);
}

export async function setMediaNativePreviewPlayState(
  payload: MediaNativePreviewPlayStatePayload,
) {
  return await ipcRenderer.invoke("media-native-preview:set-play-state", payload);
}

export async function getMediaNativePreviewStats(): Promise<MediaNativePreviewStatsResponse> {
  return await ipcRenderer.invoke("media-native-preview:stats");
}
