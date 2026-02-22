import { generateTimelineSamples } from "@/lib/media/timeline";
import { MediaInfo, ImageClipProps } from "@/lib/types";
import { useClipStore } from "@/lib/clip";
import { CompositorShader } from "@/components/preview/webgl-filters";
const THUMBNAIL_TILE_SIZE = 36;

let imageThumbnailCompositor: CompositorShader | null = null;

const getImageThumbnailCompositor = (): CompositorShader => {
  if (!imageThumbnailCompositor) {
    imageThumbnailCompositor = new CompositorShader();
  }
  return imageThumbnailCompositor;
};

const getImageFilterState = (clip: ImageClipProps) => ({
  brightness: clip?.brightness,
  contrast: clip?.contrast,
  hue: clip?.hue,
  saturation: clip?.saturation,
  blur: clip?.blur,
  sharpness: clip?.sharpness,
  noise: clip?.noise,
  vignette: clip?.vignette,
  colorTintColor: clip?.colorTintColor,
  colorTintIntensity: clip?.colorTintIntensity,
  scanLines: clip?.scanLines,
  chromaticAberration: clip?.chromaticAberration,
  interlace: clip?.interlace,
  pixelate: clip?.pixelate,
  jitter: clip?.jitter,
});

const applyImageThumbnailEffects = (
  sourceCanvas: HTMLCanvasElement,
  clip: ImageClipProps,
  maskFrame: number,
  applyMask: (
    canvas: HTMLCanvasElement,
    frameIndex?: number,
  ) => HTMLCanvasElement,
  applyFilters: (canvas: HTMLCanvasElement, filters: any) => void,
): HTMLCanvasElement => {
  const width = Math.max(1, sourceCanvas.width || 1);
  const height = Math.max(1, sourceCanvas.height || 1);
  const filters = getImageFilterState(clip);

  const working = document.createElement("canvas");
  working.width = width;
  working.height = height;
  const workingCtx = working.getContext("2d");
  if (!workingCtx) return sourceCanvas;
  workingCtx.drawImage(sourceCanvas, 0, 0, width, height);

  try {
    const compositor = getImageThumbnailCompositor();
    const effectsResult = compositor.apply(working, {
      filterParams: filters,
      // Effects first; mask is applied after compositor below.
      masks: [],
    });

    if (effectsResult && effectsResult !== working) {
      workingCtx.clearRect(0, 0, width, height);
      workingCtx.drawImage(effectsResult, 0, 0, width, height);
    }

    const masked = applyMask(working, maskFrame);
    if (!masked || masked === working) return working;

    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = Math.max(1, (masked as any).width || width);
    finalCanvas.height = Math.max(1, (masked as any).height || height);
    const finalCtx = finalCanvas.getContext("2d");
    if (!finalCtx) return masked;
    finalCtx.drawImage(masked, 0, 0, finalCanvas.width, finalCanvas.height);
    return finalCanvas;
  } catch {
    // Fallback order must also be effects first, mask second.
    applyFilters(working, filters);
    const masked = applyMask(working, maskFrame);
    if (!masked || masked === working) return working;

    const fallback = document.createElement("canvas");
    fallback.width = Math.max(1, (masked as any).width || width);
    fallback.height = Math.max(1, (masked as any).height || height);
    const fallbackCtx = fallback.getContext("2d");
    if (!fallbackCtx) return masked;
    fallbackCtx.drawImage(masked, 0, 0, fallback.width, fallback.height);
    return fallback;
  }
};

export const generateTimelineThumbnailImage = async (
  clipType: string,
  currentClip: ImageClipProps,
  currentClipId: string,
  mediaInfoRef: MediaInfo | null,
  imageCanvas: HTMLCanvasElement,
  timelineHeight: number,
  thumbnailClipWidth: number,
  maxTimelineWidth: number,
  applyMask: (
    canvas: HTMLCanvasElement,
    frameIndex?: number,
  ) => HTMLCanvasElement,
  applyFilters: (canvas: HTMLCanvasElement, filters: any) => void,
  groupRef: any,
  moveClipToEnd: (clipId: string) => void,
  resizeSide: "left" | "right" | null,
) => {
  if (clipType !== "image") return;
  const tClipWidth = Math.min(thumbnailClipWidth, maxTimelineWidth);
  let width = mediaInfoRef?.image?.width ?? 0;
  let height = mediaInfoRef?.image?.height ?? 0;
  let ratio = width / height;
  let thumbnailWidth = timelineHeight * ratio;

  thumbnailWidth = Math.max(thumbnailWidth, THUMBNAIL_TILE_SIZE);

  const getAssetById = useClipStore.getState().getAssetById;
  const asset = getAssetById(currentClip.assetId);
  if (!asset) return;

  const samples = await generateTimelineSamples(
    currentClipId,
    asset.path,
    [0],
    thumbnailWidth,
    timelineHeight,
    tClipWidth,
    {
      mediaInfo: mediaInfoRef ?? undefined,
    },
  );

  if (samples?.[0]?.canvas) {
    const inputCanvas = samples?.[0]?.canvas as HTMLCanvasElement;
    const maskFrame = Math.max(0, Math.round(currentClip?.startFrame ?? 0));
    const canvasToTile = applyImageThumbnailEffects(
      inputCanvas,
      currentClip,
      maskFrame,
      applyMask,
      applyFilters,
    );
    const ctx = imageCanvas.getContext("2d");
    if (ctx) {
      const targetWidth = Math.max(1, imageCanvas.width);
      const targetHeight = Math.max(1, imageCanvas.height);
      ctx.clearRect(0, 0, targetWidth, targetHeight);

      // Determine tile dimensions from the input canvas/image
      const tileWidth = Math.max(
        1,
        (canvasToTile as any).width || (canvasToTile as any).naturalWidth || 1,
      );
      const tileHeight = Math.max(
        1,
        (canvasToTile as any).height ||
          (canvasToTile as any).naturalHeight ||
          1,
      );
      const sourceHeight = Math.min(tileHeight, targetHeight);

      // When resizing from the left, offset the tiling pattern so new tiles appear from the left
      let startX = 0;
      if (resizeSide === "left") {
        // Calculate offset so the pattern appears anchored to the right
        const remainder = targetWidth % tileWidth;
        startX = remainder > 0 ? -(tileWidth - remainder) : 0;
      }

      // Repeat the inputCanvas horizontally until we fill the target width
      let x = startX;
      while (x < targetWidth) {
        const remaining = targetWidth - x;
        const drawWidth = Math.min(tileWidth, remaining);

        // Only draw if the tile is visible (x + drawWidth > 0)
        if (x + drawWidth > 0) {
          ctx.drawImage(canvasToTile, x, 0, drawWidth, sourceHeight);
        }
        x += drawWidth;
      }

    }
  }
  groupRef.current?.getLayer()?.batchDraw();
  moveClipToEnd(currentClipId);
};
