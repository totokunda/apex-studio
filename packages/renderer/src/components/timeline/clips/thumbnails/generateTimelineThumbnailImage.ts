import { generateTimelineSamples } from "@/lib/media/timeline";
import { MediaInfo, ImageClipProps } from "@/lib/types";
import { useClipStore } from "@/lib/clip";
const THUMBNAIL_TILE_SIZE = 36;

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
    const canvasToTile = applyMask(inputCanvas);
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

      // Apply WebGL filters to image thumbnails
      const imgClip = currentClip as ImageClipProps;
      applyFilters(imageCanvas, {
        brightness: imgClip?.brightness,
        contrast: imgClip?.contrast,
        hue: imgClip?.hue,
        saturation: imgClip?.saturation,
        blur: imgClip?.blur,
        sharpness: imgClip?.sharpness,
        noise: imgClip?.noise,
        vignette: imgClip?.vignette,
      });
    }
  }
  groupRef.current?.getLayer()?.batchDraw();
  moveClipToEnd(currentClipId);
};
