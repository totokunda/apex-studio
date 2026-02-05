import { VideoDecoderKey } from "./types";
import { VideoDecoderContext } from "./types";
import { AudioDecoderKey } from "./types";
import { AudioDecoderContext } from "./types";
import { CanvasDecoderKey } from "./types";
import { CanvasDecoderContext } from "./types";
import { DECODER_STALE_MS } from "../settings";
import { MediaInfo } from "../types";
import { MediaCache } from "./cache";
import { IMAGE_EXTS } from "../settings";
import { readImageMetadataFast } from "./image";
import {
  Input,
  ALL_FORMATS,
  BlobSource,
  UrlSource,
  AudioBufferSink,
} from "mediabunny";
import {
  getLowercaseExtension,
  readFileBuffer,
  fileURLToPath,
  resolveOriginalPath,
  pathToFileURL,
} from "@app/preload";
import { useControlsStore } from "../control";
import { useClipStore } from "../clip";
import { useProjectsStore } from "../projects";

export function nowMs(): number {
  return typeof performance !== "undefined" && performance.now
    ? performance.now()
    : Date.now();
}

export const videoDecoders = new Map<VideoDecoderKey, VideoDecoderContext>();
export const audioDecoders = new Map<AudioDecoderKey, AudioDecoderContext>();
export const canvasDecoders = new Map<CanvasDecoderKey, CanvasDecoderContext>();

export const isUUID = (str: string): boolean => {
  return /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(str);
};

export const getMediaInfoCached = (
  path: string | undefined,
): MediaInfo | undefined => {
  const mediaCache = MediaCache.getState();
  if (path && mediaCache.isMediaCached(path)) {
    return mediaCache.getMedia(path)!;
  }

  // check if path is a uuid 
  if (path && isUUID(path)) {
    // get from assets 
    const asset = useClipStore.getState().getAssetById(path);
    return mediaCache.getMedia(asset?.path ?? "")!;
  }

  if (path && path.startsWith("app://user-data/")) {
    const localPath = convertUserDataPathToLocalPath(path);
    if (mediaCache.isMediaCached(localPath)) {
      return mediaCache.getMedia(localPath)!;
    }
  }
  return undefined;
};

const getUrlWithoutHashSuffix = (path: string): string => {
  const pathUrl = new URL(path);
  pathUrl.hash = "";
  return pathUrl.toString();
};

const getUrlWithoutSearchParams = (path: string): string => {
  const pathUrl = new URL(path);
  pathUrl.search = "";
  return pathUrl.toString();
};

export const convertUserDataPath = (path: string): string => {
  // Convert a local file URL/path into a Windows-safe app://user-data URL.
  // Important: do NOT embed backslashes in the URL; normalize to forward slashes and encode.
  if (path.startsWith("app://")) return path;
  const fsPath = (() => {
    try {
      // Accept file:// URLs
      if (path.startsWith("file://")) return fileURLToPath(path);
    } catch {
      // ignore
    }
    // Accept raw filesystem paths too (best-effort)
    return path;
  })();
  const posix = String(fsPath).replace(/\\/g, "/");
  const pathPart = posix.startsWith("/") ? posix : `/${posix}`;
  return `app://user-data${encodeURI(pathPart)}`;
};

export const convertUserDataPathToLocalPath = (path: string): string => {
  // Best-effort conversion from app://user-data/... to file://... for cache-key compatibility.
  // If the app:// URL encodes an absolute filesystem path (common for media), this will work on Windows too.
  if (!path.startsWith("app://")) return path;
  try {
    const u = new URL(path);
    const decodedPathname = (() => {
      try {
        return decodeURIComponent(u.pathname);
      } catch {
        return u.pathname;
      }
    })();

    // Windows drive letter paths come through as "/C:/Users/..."
    const driveMatch = decodedPathname.match(/^\/([a-zA-Z]):\/(.*)$/);
    if (driveMatch) {
      const drive = driveMatch[1]!;
      const rest = driveMatch[2]!.replace(/\//g, "\\");
      const fsPath = `${drive}:\\${rest}`;
      return pathToFileURL(fsPath);
    }

    // UNC paths may come through as "//server/share/..."
    if (decodedPathname.startsWith("//")) {
      const rest = decodedPathname.slice(2).replace(/\//g, "\\");
      const fsPath = `\\\\${rest}`;
      return pathToFileURL(fsPath);
    }

    // POSIX absolute paths (mac/linux) - keep as-is.
    return pathToFileURL(decodedPathname);
  } catch {
    return path;
  }
};

export const convertApexCachePath = (path: string): string => {
  if (path.startsWith("app://apex-cache/")) {
    return path;
  }
  if (path.startsWith("app://")) return path;
  const fsPath = (() => {
    try {
      if (path.startsWith("file://")) return fileURLToPath(path);
    } catch {}
    return path;
  })();
  const posix = String(fsPath).replace(/\\/g, "/");
  const pathPart = posix.startsWith("/") ? posix : `/${posix}`;
  return `app://apex-cache${encodeURI(pathPart)}`;
};

export const getMediaInfo = async (
  path: string,
  options?: {
    fullStats?: boolean;
    sourceDir?: "user-data" | "apex-cache";
    useOriginal?: boolean;
    folderUuid?: string;
  },
): Promise<MediaInfo> => {
  const fsPathToAppUrl = (
    host: "user-data" | "apex-cache",
    fsPath: string,
    folderUuid?: string,
  ): string => {
    const posix = String(fsPath).replace(/\\/g, "/");
    const pathPart = posix.startsWith("/") ? posix : `/${posix}`;
    const u = new URL(`app://${host}${encodeURI(pathPart)}`);
    if (folderUuid && host === "apex-cache") {
      u.searchParams.set("folderUuid", folderUuid);
    }
    return u.toString();
  };

  if (options?.useOriginal) {
    try {
      const fsPath = fileURLToPath(path);
      const originalFsPath = await resolveOriginalPath(fsPath);
      if (originalFsPath !== fsPath) {
        path = pathToFileURL(originalFsPath);
      }
    } catch (e) {
      // ignore path parsing errors
    }
  }

  const folderUuid = options?.folderUuid || useProjectsStore.getState().getActiveProject()?.folderUuid || undefined;

  const pathUrl = new URL(path);
  const startFrame = pathUrl.searchParams.get("startFrame")
    ? Number(pathUrl.searchParams.get("startFrame"))
    : undefined;
  const endFrame = pathUrl.searchParams.get("endFrame")
    ? Number(pathUrl.searchParams.get("endFrame"))
    : undefined;
  const quickLoad = !options?.fullStats;

  // check if the metadata is cached
  const mediaCache = MediaCache.getState();
  if (mediaCache.isMediaCached(path)) {
    return mediaCache.getMedia(path)!;
  }

  // Check if path has #audio or #video suffix
  const hashIndex = pathUrl.hash.indexOf("#");
  const hasHashSuffix = hashIndex !== -1;
  const originalPath = hasHashSuffix ? getUrlWithoutHashSuffix(path) : path;
  const hashSuffix = hasHashSuffix
    ? pathUrl.hash.substring(hashIndex + 1)
    : null;
  const isAudioOnly = hashSuffix === "audio";
  const isVideoOnly = hashSuffix === "video";

  // If we have a hash suffix, check if the original path is cached
  let baseMediaInfo: MediaInfo | undefined = undefined;
  if (hasHashSuffix && mediaCache.isMediaCached(originalPath)) {
    baseMediaInfo = mediaCache.getMedia(originalPath)!;

    // Create a filtered version based on hash suffix
    const filteredMediaInfo: MediaInfo = {
      ...baseMediaInfo,
      path,
      video: isAudioOnly ? null : baseMediaInfo.video,
      audio: isVideoOnly ? null : baseMediaInfo.audio,
      stats: {
        video: isAudioOnly ? undefined : baseMediaInfo.stats.video,
        audio: isVideoOnly ? undefined : baseMediaInfo.stats.audio,
      },
      startFrame: startFrame,
      endFrame: endFrame,
    };

    mediaCache.setMedia(path, filteredMediaInfo);
    return filteredMediaInfo;
  }

  const ext = getLowercaseExtension(hasHashSuffix ? originalPath : path);
  // check if file is an image. then modify

  if (IMAGE_EXTS.includes(ext)) {
    // Route image reads through app protocol to handle remote cache mirroring
    let imageReadUrl = path;
    let fsPathForImage: string | null = null;
    const primarySourceDir = options?.sourceDir ?? "user-data";
    const secondarySourceDir =
      primarySourceDir === "user-data" ? "apex-cache" : "user-data";
    try {
      let url: URL | null = null;
      if (path.startsWith("app://")) {
        url = new URL(path);
        if (!url.searchParams.get("folderUuid")) {
          url.searchParams.set("folderUuid", folderUuid ?? "");
        }
      } else {
        fsPathForImage = fileURLToPath(hasHashSuffix ? originalPath : path);
        url = new URL(fsPathToAppUrl(primarySourceDir, fsPathForImage, folderUuid));
      }

      imageReadUrl = url.toString();
    } catch {}

    // get height and width from the file
    let metadata;
    try {
      metadata = await readImageMetadataFast(imageReadUrl);
    } catch (e) {
      // If reading via primary sourceDir fails (e.g., 404), try other sourceDir as backup
      if (fsPathForImage) {
        const fallbackUrl = new URL(
          fsPathToAppUrl(secondarySourceDir, fsPathForImage, folderUuid),
        );
        metadata = await readImageMetadataFast(fallbackUrl.toString());
      } else {
        throw e;
      }
    }

    const mediaInfo = {
      path,
      video: null,
      audio: null,
      image: metadata,
      stats: {
        video: undefined,
        audio: undefined,
      },
      duration: undefined,
      metadata: undefined,
      mimeType: metadata.mime,
      format: undefined,
      videoDecoderConfig: undefined,
    };
    mediaCache.setMedia(path, mediaInfo);
    return mediaInfo;
  }

  if (startFrame || endFrame) {
    const pathWithoutSearchParams = getUrlWithoutSearchParams(path);
    // check if the metadata is cached
    const mediaCache = MediaCache.getState();
    if (mediaCache.isMediaCached(pathWithoutSearchParams)) {
      const mediaInfo = mediaCache.getMedia(pathWithoutSearchParams)!;
      const newMediaInfo: MediaInfo = {
        ...mediaInfo,
        path,
        startFrame: startFrame,
        endFrame: endFrame,
      };
      mediaCache.setMedia(path, newMediaInfo);
      return newMediaInfo;
    }
  }

  // Prefer streaming via UrlSource, but fall back to BlobSource if the server truncates streams or range reads fail
  let input: Input | null = null;
  let filePath: string | null = null;
  const primarySourceDir = options?.sourceDir ?? "user-data";
  const secondarySourceDir =
    primarySourceDir === "user-data" ? "apex-cache" : "user-data";
  try {
    filePath = fileURLToPath(hasHashSuffix ? originalPath : path);
    const url = new URL(fsPathToAppUrl(primarySourceDir, filePath, folderUuid));
    
    input = new Input({ formats: ALL_FORMATS, source: new UrlSource(url) });
  } catch (e) {
    try {
      if (!filePath) throw e;
      const url = new URL(fsPathToAppUrl(secondarySourceDir, filePath, folderUuid));
      input = new Input({ formats: ALL_FORMATS, source: new UrlSource(url) });
    } catch (e) {
      input = null;
    }
  }

  // If UrlSource creation failed for some reason, or if later reads fail, we'll fallback below
  async function gatherInfo(inp: Input, quickLoad = true) {
    
    const videoTrack = await inp.getPrimaryVideoTrack();
    const audioTrack = await inp.getPrimaryAudioTrack();
    // For quick load, only scan first ~100 packets for fast frame rate estimate
    // For full load (when clip is on timeline), scan all packets for accuracy
    const targetPacketCount = quickLoad ? 100 : Infinity;
    const packetStats = await videoTrack?.computePacketStats(targetPacketCount);
    const audioPacketStats =
      await audioTrack?.computePacketStats(targetPacketCount);

    const duration = await inp.computeDuration();
    const metadata = await inp.getMetadataTags();
    const mimeType = await inp.getMimeType();
    const format = await inp.getFormat();

    const videoDecoderConfig = await videoTrack?.getDecoderConfig();

    if (audioTrack) {
      const sink = new AudioBufferSink(audioTrack);
      const sample = await sink.getBuffer(0);
      const sampleSize = sample?.buffer.length || 2048;
      (audioTrack as any).sampleSize = sampleSize;
    }
    return {
      videoTrack,
      audioTrack,
      packetStats,
      audioPacketStats,
      duration,
      metadata,
      mimeType,
      format,
      videoDecoderConfig,
    };
  }

  let infoBundle: any | null = null;
  // Attempt with primary sourceDir
  try {
    if (!input) throw new Error("UrlSource init failed");
    infoBundle = await gatherInfo(input, quickLoad);
  } catch (e) {

      // Try secondary sourceDir via UrlSource
    try {
      if (!filePath) throw new Error("Missing filePath");
      const fallbackUrl = new URL(
        fsPathToAppUrl(secondarySourceDir, filePath, folderUuid),
      );
      const input2 = new Input({
        formats: ALL_FORMATS,
        source: new UrlSource(fallbackUrl),
      });
      infoBundle = await gatherInfo(input2, quickLoad);
      input = input2; // use the working input as originalInput
    } catch (e2) {
      // Fallback: fetch the entire resource as a Blob and use BlobSource to avoid streaming issues
      try {
        if (!filePath) throw new Error("Missing filePath");
        // Try primary app protocol first
        const primaryUrl = new URL(
          fsPathToAppUrl(primarySourceDir, filePath, folderUuid),
        );
        const bufferPrimary = await readFileBuffer(primaryUrl.toString());
        const blobP = new Blob([bufferPrimary as unknown as ArrayBuffer]);
        const blobInputP = new Input({
          formats: ALL_FORMATS,
          source: new BlobSource(blobP),
        });
        infoBundle = await gatherInfo(blobInputP, quickLoad);
        } catch (e3) {

        try {
          if (!filePath) throw new Error("Missing filePath");
          // Try secondary app protocol next
          const secondaryUrl = new URL(
            fsPathToAppUrl(secondarySourceDir, filePath, folderUuid),
          );
          const bufferSecondary = await readFileBuffer(secondaryUrl.toString());
          const blobS = new Blob([bufferSecondary as unknown as ArrayBuffer]);
          const blobInputS = new Input({
            formats: ALL_FORMATS,
            source: new BlobSource(blobS),
          });
          infoBundle = await gatherInfo(blobInputS, quickLoad);
        } catch (e4) {

          // Last resort: original path provided
          try {
            const buffer = await readFileBuffer(
              hasHashSuffix ? originalPath : path,
            );
            const blob = new Blob([buffer as unknown as ArrayBuffer]);
            const blobInput = new Input({
              formats: ALL_FORMATS,
              source: new BlobSource(blob),
            });
            infoBundle = await gatherInfo(blobInput, quickLoad);
          } catch (fallbackErr) {
            throw fallbackErr;
          }
        }
      }
    }
  }

  let {
    videoTrack,
    audioTrack,
    packetStats,
    audioPacketStats,
    duration,
    metadata,
    mimeType,
    format,
    videoDecoderConfig,
  } = infoBundle;
  if (audioTrack && !(await audioTrack.canDecode())) {
    audioTrack = null;
    console.warn("Audio track cannot be decoded", path);
  } else if (videoTrack && !(await videoTrack.canDecode())) {
    videoTrack = null;
    console.warn("Video track cannot be decoded", path);
  }

  const mediaInfo: MediaInfo = {
    path,
    video: isAudioOnly ? null : videoTrack,
    audio: isVideoOnly ? null : audioTrack,
    image: null,
    stats: {
      video: isAudioOnly ? undefined : packetStats,
      audio: isVideoOnly ? undefined : audioPacketStats,
    },
    duration,
    metadata,
    mimeType,
    format,
    startFrame: startFrame,
    endFrame: endFrame,
    originalInput: input ?? undefined,
    videoDecoderConfig,
  };

  mediaCache.setMedia(path, mediaInfo);

  // If we fetched with a hash suffix but the base wasn't cached, also cache the full version
  if (hasHashSuffix && !baseMediaInfo) {
    const fullMediaInfo: MediaInfo = {
      ...mediaInfo,
      path: originalPath,
      video: videoTrack,
      audio: audioTrack,
      stats: {
        video: packetStats,
        audio: audioPacketStats,
      },
      startFrame: startFrame,
      endFrame: endFrame,
    };
    mediaCache.setMedia(originalPath, fullMediaInfo);
  }

  // If this was a quick load and we have tracks with estimates, optionally compute full stats in background
  if (quickLoad && !options?.fullStats && (videoTrack || audioTrack)) {
    void (async () => {
      try {
        if (input && (videoTrack || audioTrack)) {
          const fullVideoStats = videoTrack
            ? await videoTrack.computePacketStats()
            : undefined;
          const fullAudioStats = audioTrack
            ? await audioTrack.computePacketStats()
            : undefined;

          // Update cache with full stats
          const cached = mediaCache.getMedia(path);
          if (cached) {
            cached.stats.video = fullVideoStats;
            cached.stats.audio = fullAudioStats;
            mediaCache.setMedia(path, cached);
          }
        }
      } catch {}
    })();
  }

  return mediaInfo;
};


export const getMediaInfoFromUrl = async (url: string, headers?: Record<string, string>): Promise<MediaInfo | undefined> => {
  // check if cached 
  const mediaCache = MediaCache.getState();
  if (mediaCache.isMediaCached(url)) {
    return mediaCache.getMedia(url)!;
  }

  // check if url is a valid url
  if (!url.startsWith("http")) {
    return undefined;
  }


  let extension = url.split(".").pop() || "";
  // check if image 
  if (IMAGE_EXTS.includes(extension)) {
    // fetch the image
    let realURL = new URL(url);
    const response = await fetch(realURL, { headers });
    const blob = await response.blob();
    const metadata = await readImageMetadataFast(blob);
    const mediaInfo = {
      path: url,
      video: null,
      audio: null,
      image: metadata,
      stats: {
        video: undefined,
        audio: undefined,
      },
      duration: undefined,
      metadata: undefined,
      mimeType: metadata.mime,
      format: undefined,
      videoDecoderConfig: undefined,
    };
    mediaCache.setMedia(url, mediaInfo);
    return mediaInfo;
  }
  // now do regular media info
  let input = new Input({ formats: ALL_FORMATS, source: new UrlSource(url, { requestInit: { headers } }) });
  async function gatherInfo(inp: Input, quickLoad = true) {

    const videoTrack = await inp.getPrimaryVideoTrack();
    const audioTrack = await inp.getPrimaryAudioTrack();
    // For quick load, only scan first ~100 packets for fast frame rate estimate
    // For full load (when clip is on timeline), scan all packets for accuracy
    const targetPacketCount = quickLoad ? 100 : Infinity;
    const packetStats = await videoTrack?.computePacketStats(targetPacketCount);
    const audioPacketStats =
      await audioTrack?.computePacketStats(targetPacketCount);

    const duration = await inp.computeDuration();
    const metadata = await inp.getMetadataTags();
    const mimeType = await inp.getMimeType();
    const format = await inp.getFormat();

    const videoDecoderConfig = await videoTrack?.getDecoderConfig();

    if (audioTrack) {
      const sink = new AudioBufferSink(audioTrack);
      const sample = await sink.getBuffer(0);
      const sampleSize = sample?.buffer.length || 2048;
      (audioTrack as any).sampleSize = sampleSize;
    }
    return {
      videoTrack,
      audioTrack,
      packetStats,
      audioPacketStats,
      duration,
      metadata,
      mimeType,
      format,
      videoDecoderConfig,
    };
  }
  let infoBundle = await gatherInfo(input);

  let {
    videoTrack,
    audioTrack,
    packetStats,
    audioPacketStats,
    duration,
    metadata,
    mimeType,
    format,
    videoDecoderConfig,
  } = infoBundle;
  if (audioTrack && !(await audioTrack.canDecode())) {
    audioTrack = null;
    console.warn("Audio track cannot be decoded", url);
  } else if (videoTrack && !(await videoTrack.canDecode())) {
    videoTrack = null;
    console.warn("Video track cannot be decoded", url);
  }

  const mediaInfo: MediaInfo = {
    path: url,
    video: videoTrack,
    audio: audioTrack,
    image: null,
    stats: {
      video: packetStats,
      audio: audioPacketStats,
    },
    duration,
    metadata: metadata,
    mimeType: mimeType,
    format: format,
    videoDecoderConfig: videoDecoderConfig ?? undefined,
  };

  mediaCache.setMedia(url, mediaInfo);
  return mediaInfo;
   
}

// Helper to upgrade media info to full stats (call when clip added to timeline)
export const ensureFullMediaStats = async (path: string): Promise<void> => {
  const mediaCache = MediaCache.getState();
  const cached = mediaCache.getMedia(path);
  if (!cached) return;

  // Check if we already have full stats (packetCount > 100 indicates full scan)
  const hasFullVideoStats =
    cached.stats.video?.packetCount && cached.stats.video.packetCount > 100;
  const hasFullAudioStats =
    cached.stats.audio?.packetCount && cached.stats.audio.packetCount > 100;

  if (hasFullVideoStats && hasFullAudioStats) return;

  try {
    if (!hasFullVideoStats && cached.video) {
      const fullStats = await cached.video.computePacketStats();
      cached.stats.video = fullStats;
    }
    if (!hasFullAudioStats && cached.audio) {
      const fullStats = await cached.audio.computePacketStats();
      cached.stats.audio = fullStats;
    }
    mediaCache.setMedia(path, cached);
  } catch (e) {
    console.warn("Failed to compute full stats:", e);
  }
};

export function pruneStaleDecoders() {
  const threshold = nowMs() - DECODER_STALE_MS;
  for (const [key, ctx] of videoDecoders) {
    if (ctx.lastAccessTs < threshold) {
      try {
        (ctx.sink as any)?.close?.();
      } catch {}
      videoDecoders.delete(key);
    }
  }
  for (const [key, ctx] of audioDecoders) {
    if (ctx.lastAccessTs < threshold) {
      try {
        (ctx.sink as any)?.close?.();
      } catch {}
      audioDecoders.delete(key);
    }
  }
  for (const [key, ctx] of canvasDecoders) {
    if (ctx.lastAccessTs < threshold) {
      try {
        (ctx.sink as any)?.close?.();
      } catch {}
      canvasDecoders.delete(key);
    }
  }
}

/**
 * For safekeeping
 */
export const generateAudioWaveformCanvas = async (
  path: string,
  width: number,
  height: number,
  options?: {
    samples?: number;
    color?: string;
    mediaInfo?: MediaInfo;
    startFrame?: number;
    endFrame?: number;
    sampleWidth?: number;
    sampleGap?: number;
  },
): Promise<HTMLCanvasElement | null> => {
  let ac: AudioContext | null = null;
  try {
    const targetW = Math.max(1, Math.floor(width));
    const targetH = Math.max(1, Math.floor(height));
    const color = options?.color || "#A477C4";
    // We render the full waveform at 1px per frame; 'samples' is unused here.
    const rawStart = options?.startFrame;
    const rawEnd = options?.endFrame;
    const startDefined =
      Number.isFinite(rawStart as number) && (rawStart as number) > 0;
    const endDefined =
      Number.isFinite(rawEnd as number) && (rawEnd as number) > 0;
    const mediaCache = MediaCache.getState();
    const fpsFromStore = Math.max(
      1,
      Math.round(useControlsStore.getState().fps || 24),
    );
    const dataKey = `waveform_data:${path}#fps${fpsFromStore}`;
    const fullKey = `waveform_full:${path}#h${targetH}#fps${fpsFromStore}#c${color}`;

    // Helper: mix color with white for lighter gradient
    const mixColorWithWhite = (hex: string, t: number): string => {
      const h = hex.replace("#", "");
      const r = parseInt(h.substring(0, 2), 16);
      const g = parseInt(h.substring(2, 4), 16);
      const b = parseInt(h.substring(4, 6), 16);
      const lerp = (a: number, b: number, t: number) =>
        Math.round(a + (b - a) * t);
      const rr = Math.max(0, Math.min(255, lerp(r, 255, t)))
        .toString(16)
        .padStart(2, "0");
      const gg = Math.max(0, Math.min(255, lerp(g, 255, t)))
        .toString(16)
        .padStart(2, "0");
      const bb = Math.max(0, Math.min(255, lerp(b, 255, t)))
        .toString(16)
        .padStart(2, "0");
      return `#${rr}${gg}${bb}`;
    };

    // Helper: draw rounded rectangle
    const fillRoundRect = (
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      w: number,
      h: number,
      r: number,
    ) => {
      const radius = Math.max(0, Math.min(r, Math.min(w, h) / 2));
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + w - radius, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
      ctx.lineTo(x + w, y + h - radius);
      ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
      ctx.lineTo(x + radius, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
      ctx.fill();
    };

    // Helper: render target canvas as grouped bottom bars from amplitude array
    const renderBars = (
      amplitudes: Float32Array,
      segStart: number,
      segEnd: number,
    ): HTMLCanvasElement => {
      const canvas = document.createElement("canvas");
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return canvas;
      ctx.clearRect(0, 0, targetW, targetH);
      ctx.imageSmoothingEnabled = false;

      const segLen = Math.max(1, segEnd - segStart);

      // Layout: compute bar count and per-bar widths so we fill targetW exactly
      const minBarWidth = options?.sampleWidth ?? (targetW >= 240 ? 2 : 1);
      const baseGap = options?.sampleGap ?? 1;
      let barCount = Math.max(
        1,
        Math.floor((targetW + baseGap) / (minBarWidth + baseGap)),
      );
      // Safety fallback
      if (barCount < 1) barCount = 1;
      const gapUsed = barCount > 1 ? baseGap : 0;
      const baseTotal = minBarWidth * barCount + gapUsed * (barCount - 1);
      let leftover = Math.max(0, targetW - baseTotal);
      const perBarWidths: number[] = new Array(barCount).fill(minBarWidth);
      for (let i = 0; i < barCount && leftover > 0; i++, leftover--)
        perBarWidths[i] += 1;
      // Precompute x positions and centers for accurate envelope drawing
      const barX: number[] = new Array(barCount).fill(0);
      const barCenterX: number[] = new Array(barCount).fill(0);
      {
        let accX = 0;
        for (let i = 0; i < barCount; i++) {
          barX[i] = accX;
          barCenterX[i] = Math.min(
            targetW - 1,
            accX + Math.floor(perBarWidths[i] / 2),
          );
          accX += perBarWidths[i] + gapUsed;
        }
      }
      const framesPerBar = Math.max(1, Math.ceil(segLen / barCount));

      // Visuals
      const topColor = mixColorWithWhite(color, 0.5);
      const bottomColor = mixColorWithWhite(color, 0.15);
      const gradient = ctx.createLinearGradient(0, 0, 0, targetH);
      gradient.addColorStop(0, topColor);
      gradient.addColorStop(1, bottomColor);
      const capRadius = Math.min(2, Math.floor(Math.max(...perBarWidths) / 2));
      const bottomPad = 3; // leave a thin baseline
      const usableHeight = Math.max(1, targetH - bottomPad);

      // Optional subtle baseline
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = mixColorWithWhite(color, 0.85);
      ctx.fillRect(0, targetH - 1, targetW, 1);
      ctx.globalAlpha = 1.0;

      // Build bars
      ctx.fillStyle = gradient;
      const loudnessCurve = (v: number) =>
        Math.pow(Math.max(0, Math.min(1, v)), 0.7); // slight expansion of low volumes
      for (let i = 0; i < barCount; i++) {
        const f0 = segStart + i * framesPerBar;
        const f1 = Math.min(segEnd, f0 + framesPerBar);
        if (f0 >= segEnd) break;
        let peak = 0;
        for (let f = f0; f < f1; f++) {
          const v = amplitudes[f] || 0;
          if (v > peak) peak = v;
        }
        const amp = loudnessCurve(peak);
        const h = Math.max(1, Math.round(amp * usableHeight));
        const y = targetH - h;
        fillRoundRect(ctx, barX[i], y, perBarWidths[i], h, capRadius);
      }

      // Optional soft envelope line for a cleaner look
      try {
        ctx.lineWidth = 1;
        ctx.lineJoin = "round";
        ctx.strokeStyle = mixColorWithWhite(color, 0.6);
        ctx.globalAlpha = 0.6;
        const step = Math.max(
          1,
          Math.floor(barCount / Math.max(16, Math.floor(targetW / 24))),
        );
        ctx.beginPath();
        let first = true;
        for (let i = 0; i < barCount; i += step) {
          const f0 = segStart + i * framesPerBar;
          const f1 = Math.min(segEnd, f0 + framesPerBar);
          let maxV = 0;
          for (let f = f0; f < f1; f++)
            maxV = Math.max(maxV, amplitudes[f] || 0);
          const amp = loudnessCurve(maxV);
          const h = Math.max(1, Math.round(amp * usableHeight));
          const cx = barCenterX[Math.min(i, barCenterX.length - 1)];
          const cy = targetH - h;
          if (first) {
            ctx.moveTo(cx, cy);
            first = false;
          } else {
            ctx.lineTo(cx, cy);
          }
        }
        ctx.stroke();
      } catch {}

      ctx.globalAlpha = 1.0;
      return canvas;
    };

    // If we have cached amplitude data, render bars from it immediately
    const tryData = mediaCache.getThumbnailCache(dataKey);
    if (tryData) {
      const dataCanvas = tryData as HTMLCanvasElement;
      const totalFrames = Math.max(1, dataCanvas.width);
      const startFrame = startDefined
        ? Math.max(0, Math.floor(rawStart as number))
        : 0;
      const endFrame = endDefined
        ? Math.max(1, Math.floor(rawEnd as number))
        : totalFrames;
      const segStart = Math.min(Math.max(0, startFrame), totalFrames - 1);
      const segEnd = Math.max(segStart + 1, Math.min(totalFrames, endFrame));

      const dctx = dataCanvas.getContext("2d");
      if (!dctx) return null;
      const img = dctx.getImageData(0, 0, totalFrames, 1).data;
      const amps = new Float32Array(totalFrames);
      for (let i = 0; i < totalFrames; i++) {
        const r = img[i * 4];
        amps[i] = Math.max(0, Math.min(1, r / 255));
      }
      return renderBars(amps, segStart, segEnd);
    }

    const resp = await readFileBuffer(path);
    const arr = resp.buffer as ArrayBuffer;

    // Decode audio
    const AudioContextCtor =
      (window as any).AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextCtor) return null;
    ac = new AudioContextCtor();
    const audioBuffer: AudioBuffer = await new Promise((resolve, reject) => {
      // Some browsers require callback-style decode
      try {
        // Safari requires a copy of the ArrayBuffer sometimes
        const copy = arr.slice(0);
        (ac as AudioContext).decodeAudioData(copy, resolve, reject);
      } catch (e) {
        reject(e);
      }
    });

    const channels = audioBuffer.numberOfChannels;
    const totalLength = audioBuffer.length; // PCM samples per channel
    const sr = Math.max(1, audioBuffer.sampleRate || 1);
    const fpsLocal = fpsFromStore;
    const perFrameDecoded = sr / fpsLocal;
    const totalFrames = Math.max(1, Math.floor((totalLength * fpsLocal) / sr));

    // Build full-frame peaks (1 px per frame)
    const sOf = (f: number) => Math.round(f * perFrameDecoded);
    const framePeaks = new Float32Array(totalFrames);
    for (let f = 0; f < totalFrames; f++) {
      const s0 = Math.max(0, Math.min(totalLength - 1, sOf(f)));
      const s1 = Math.max(s0 + 1, Math.min(totalLength, sOf(f + 1)));
      let min = 1.0;
      let max = -1.0;
      const windowLen = s1 - s0;
      const innerStride = Math.max(1, Math.floor(windowLen / 64));
      for (let j = s0; j < s1; j += innerStride) {
        let sum = 0;
        for (let c = 0; c < channels; c++)
          sum += audioBuffer.getChannelData(c)[j] || 0;
        const v = sum / channels;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      framePeaks[f] = Math.max(Math.abs(min), Math.abs(max));
    }

    // Light symmetric smoothing
    const smoothed = new Float32Array(totalFrames);
    const halfK = Math.max(1, Math.min(3, Math.floor(totalFrames / 200)));
    for (let i = 0; i < totalFrames; i++) {
      let acc = 0;
      let cnt = 0;
      for (let k = -halfK; k <= halfK; k++) {
        const idx = i + k;
        if (idx >= 0 && idx < totalFrames) {
          acc += framePeaks[idx];
          cnt++;
        }
      }
      smoothed[i] = cnt > 0 ? acc / cnt : framePeaks[i];
    }

    // Persist amplitude data as a compact 1xN canvas for fast future renders
    try {
      const dataCanvas = document.createElement("canvas");
      dataCanvas.width = totalFrames;
      dataCanvas.height = 1;
      const dctx = dataCanvas.getContext("2d");
      if (dctx) {
        const img = dctx.createImageData(totalFrames, 1);
        for (let i = 0; i < totalFrames; i++) {
          const v = Math.max(0, Math.min(1, smoothed[i] || 0));
          const r = Math.round(v * 255);
          img.data[i * 4 + 0] = r;
          img.data[i * 4 + 1] = r;
          img.data[i * 4 + 2] = r;
          img.data[i * 4 + 3] = 255;
        }
        dctx.putImageData(img, 0, 0);
        mediaCache.setThumbnailCache(dataKey, dataCanvas);
      }
    } catch {}

    // Also cache a minimal full canvas placeholder (optional, may be unused)
    try {
      const fullCanvas = document.createElement("canvas");
      fullCanvas.width = totalFrames;
      fullCanvas.height = Math.max(1, Math.min(targetH, 64));
      const fctx = fullCanvas.getContext("2d");
      if (fctx) {
        fctx.clearRect(0, 0, fullCanvas.width, fullCanvas.height);
        fctx.imageSmoothingEnabled = false;
        fctx.fillStyle = "#00000000";
        fctx.fillRect(0, 0, fullCanvas.width, fullCanvas.height);
        fctx.fillStyle = mixColorWithWhite(color, 0.6);
        const baseY = fullCanvas.height - 2;
        for (let x = 0; x < totalFrames; x++) {
          const v = smoothed[x] || 0;
          const h = Math.max(1, Math.round(v * (fullCanvas.height - 2)));
          fctx.fillRect(x, baseY - h + 1, 1, h);
        }
        mediaCache.setThumbnailCache(fullKey, fullCanvas);
      }
    } catch {}

    // Render grouped bars for the requested target
    {
      const startFrame = startDefined
        ? Math.max(0, Math.floor(rawStart as number))
        : 0;
      const endFrame = endDefined
        ? Math.max(1, Math.floor(rawEnd as number))
        : totalFrames;
      const segStart = Math.min(Math.max(0, startFrame), totalFrames - 1);
      const segEnd = Math.max(segStart + 1, Math.min(totalFrames, endFrame));
      return renderBars(smoothed, segStart, segEnd);
    }
  } catch (e) {
    console.warn("[media] Failed to generate audio waveform canvas", {
      path,
      error: e,
    });
    return null;
  } finally {
    if (ac) {
      try {
        await (ac as AudioContext).close();
      } catch {}
    }
  }
};
