import { generateTimelineSamples } from "@/lib/media/timeline";
import { getNearestCachedCanvasSamples } from "@/lib/media/canvas";
import { useControlsStore } from "@/lib/control";
import { MediaInfo, VideoClipProps } from "@/lib/types";
import { getClipWidth } from "@/lib/clip";
import { useClipStore } from "@/lib/clip";
const THUMBNAIL_TILE_SIZE = 36;

export const generateTimelineThumbnailVideo = async (
  clipType: string,
  currentClip: VideoClipProps,
  currentClipId: string,
  mediaInfoRef: MediaInfo | null,
  imageCanvas: HTMLCanvasElement,
  timelineHeight: number,
  thumbnailClipWidth: number,
  maxTimelineWidth: number,
  timelineWidth: number,
  timelineDuration: [number, number],
  currentStartFrame: number,
  currentEndFrame: number,
  overHang: number,
  applyMask: (
    canvas: HTMLCanvasElement,
    frameIndex?: number,
  ) => HTMLCanvasElement,
  applyFilters: (canvas: HTMLCanvasElement, filters: any) => void,
  groupRef: any,
  resizeSide: "left" | "right" | null,
  exactVideoUpdateTimerRef: React.MutableRefObject<number | null>,
  exactVideoUpdateSeqRef: React.MutableRefObject<number>,
  lastExactRequestKeyRef: React.MutableRefObject<string | null>,
  setForceRerenderCounter: React.Dispatch<React.SetStateAction<number>>,
) => {
  if (clipType !== "video") return;
  let tClipWidth = Math.min(thumbnailClipWidth, maxTimelineWidth);
  const width = mediaInfoRef?.video?.displayWidth ?? 1;
  const height = mediaInfoRef?.video?.displayHeight ?? 1;
  const ratio = width / height;
  const thumbnailWidth = Math.max(timelineHeight * ratio, THUMBNAIL_TILE_SIZE);

  const speed = Math.max(
    0.1,
    Math.min(5, Number((currentClip as any)?.speed ?? 1)),
  );

  // Calculate frame indices based on timeline duration and available columns
  let numColumns = Math.ceil((tClipWidth - overHang) / thumbnailWidth) + 1;

  let numColumnsAlt: number | undefined = undefined;
  if (currentClip.trimStart || currentClip.trimEnd) {
    const realStartFrame = currentStartFrame - (currentClip.trimStart ?? 0);
    const realEndFrame = currentEndFrame - (currentClip.trimEnd ?? 0);
    tClipWidth = Math.max(
      getClipWidth(
        realStartFrame,
        realEndFrame,
        timelineWidth,
        timelineDuration,
      ),
      3,
    );
    const tempNumColumns = Math.ceil((tClipWidth - overHang) / thumbnailWidth);
    numColumnsAlt = numColumns;
    numColumns = tempNumColumns;
  }
  const timelineShift = currentStartFrame - (currentClip.trimStart ?? 0);
  const realStartFrame = timelineShift;
  const realEndFrame = currentEndFrame - (currentClip.trimEnd ?? 0);
  let timelineStartFrame = Math.max(timelineDuration[0], realStartFrame);
  let timelineEndFrame = Math.min(timelineDuration[1], realEndFrame);
  const timelineSpan = timelineEndFrame - timelineStartFrame;

  let frameIndices: number[];
  let startFrame = realStartFrame;
  if (timelineStartFrame > realStartFrame) {
    startFrame = timelineStartFrame;
  }
  if (timelineSpan >= numColumns && numColumns > 1) {
    // When timeline duration is large enough, space frames evenly
    frameIndices = Array.from({ length: numColumns }, (_, i) => {
      const progress = i / (numColumns - 1);
      const frameIndex = Math.round(startFrame + progress * timelineSpan);
      return frameIndex;
    });
    frameIndices[frameIndices.length - 1] =
      frameIndices[frameIndices.length - 1] - 1;
  } else if (numColumns > 1) {
    // When timeline duration is less than numColumns, duplicate frames
    frameIndices = Array.from({ length: numColumns }, (_, i) => {
      const frameIndex = Math.floor(
        i / Math.ceil(numColumns / (timelineSpan + 1)),
      );
      const clampedIndex = Math.min(frameIndex, timelineSpan);
      return timelineStartFrame + clampedIndex;
    });
  } else {
    // Single column case
    frameIndices = [timelineStartFrame];
  }

  frameIndices = frameIndices.filter(
    (frameIndex) => isNaN(frameIndex) === false && isFinite(frameIndex),
  );

  // Map timeline frames to source frames considering speed, in-clip offset, split bounds, and framerate conversion

  const projectFps = useControlsStore.getState().fps || 30;
  const clipFps = mediaInfoRef?.stats.video?.averagePacketRate || projectFps;
  const fpsAdjustment = projectFps / clipFps;

  frameIndices = frameIndices.map((frameIndex) => {
    const local = Math.max(0, frameIndex - timelineShift);
    const speedAdjusted = local * speed;
    // Map from project fps space to native clip fps space
    const nativeFpsFrame = Math.round((speedAdjusted / projectFps) * clipFps);
    const mediaStartFrame = Math.round(
      ((mediaInfoRef?.startFrame ?? 0) / projectFps) * clipFps,
    );
    const mediaEndFrame = Math.round(
      ((mediaInfoRef?.endFrame ?? 0) / projectFps) * clipFps,
    );
    let sourceFrame = nativeFpsFrame + mediaStartFrame;
    if (mediaEndFrame !== 0) {
      sourceFrame = Math.min(sourceFrame, mediaEndFrame);
    }
    return Math.max(mediaStartFrame, sourceFrame);
  });


  if (numColumnsAlt && frameIndices.length !== numColumnsAlt) {
    // Trim indices to match the original column count, removing from
    // left/right based on trimStart (left) and abs(trimEnd) (right)
    if (frameIndices.length > numColumnsAlt) {
      const surplus = frameIndices.length - numColumnsAlt;
      const giveStart = Math.max(0, currentClip?.trimStart ?? 0);
      const giveEnd = Math.max(0, -(currentClip?.trimEnd ?? 0));
      const totalGive = giveStart + giveEnd;
      let leftRemove = 0;
      let rightRemove = 0;
      if (totalGive > 0) {
        leftRemove = Math.floor((surplus * giveStart) / totalGive);
        rightRemove = surplus - leftRemove;
      } else {
        leftRemove = Math.floor(surplus / 2);
        rightRemove = surplus - leftRemove;
      }
      const start = Math.min(Math.max(0, leftRemove), frameIndices.length);
      const end = Math.max(
        start,
        frameIndices.length - Math.max(0, rightRemove),
      );
      frameIndices = frameIndices.slice(start, end);
      // In case of rounding, ensure exact length
      if (frameIndices.length > numColumnsAlt) {
        frameIndices = frameIndices.slice(0, numColumnsAlt);
      }
    }
  }

  if (frameIndices.length === 0) {
    return;
  }

  // 1) Immediate draw using nearest cached frames (synchronous)
  const getAssetById = useClipStore.getState().getAssetById;
  const asset = getAssetById(currentClip.assetId);
  if (!asset) return;

  

 

  const nearest = getNearestCachedCanvasSamples(
    asset.path,
    frameIndices,
    thumbnailWidth,
    timelineHeight,
    { mediaInfo: mediaInfoRef ?? undefined },
  );

  // Track if we have any cached samples
  const hasCachedSamples = nearest.some(
    (sample) => sample !== null && sample !== undefined,
  );
  const ctx = imageCanvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    let x = overHang;
    const targetWidth = Math.max(1, imageCanvas.width);
    const targetHeight = Math.max(1, imageCanvas.height);

    // When resizing from the left for video, truncate from the left by
    // skipping the overflow width from the left side of the tile sequence.
    let skipRemaining = 0;
    if (resizeSide === "left" && clipType === "video") {
      let totalTileWidth = 0;
      for (let i = 0; i < nearest.length; i++) {
        const sample = nearest[i];
        if (!sample) {
          continue;
        }
        const anyCanvas = sample.canvas as any;
        const tileWidth = Math.max(
          1,
          anyCanvas.width || anyCanvas.naturalWidth || 1,
        );
        totalTileWidth += tileWidth;
      }
      const drawableWidth = Math.max(0, targetWidth - x);
      skipRemaining = Math.max(0, totalTileWidth - drawableWidth);
    }

    for (let i = 0; i < nearest.length && x < targetWidth; i++) {
      const sample = nearest[i];
      if (!sample) {
        continue;
      }
      const inputCanvas = sample.canvas as HTMLCanvasElement;
      const canvasToTile = applyMask(
        inputCanvas,
        Math.round(frameIndices[i] * fpsAdjustment),
      );
      const anyCanvas = canvasToTile as any;
      const tileWidth = Math.max(
        1,
        anyCanvas.width || anyCanvas.naturalWidth || 1,
      );
      const tileHeight = Math.max(
        1,
        anyCanvas.height || anyCanvas.naturalHeight || 1,
      );
      const sourceHeight = Math.min(tileHeight, targetHeight);

      // Apply left-side truncation when needed
      let srcX = 0;
      let availableSrcWidth = tileWidth;
      if (skipRemaining > 0) {
        const consume = Math.min(skipRemaining, tileWidth);
        srcX = consume;
        availableSrcWidth = tileWidth - consume;
        skipRemaining -= consume;
        if (availableSrcWidth <= 0) {
          continue; // this tile is fully truncated away
        }
      }

      const remaining = targetWidth - x;
      if (remaining <= 0) break;
      const drawWidth = Math.min(availableSrcWidth, remaining);
      if (drawWidth <= 0) break;
      ctx.drawImage(
        canvasToTile,
        srcX,
        0,
        drawWidth,
        sourceHeight,
        x,
        0,
        drawWidth,
        sourceHeight,
      );
      x += drawWidth;
    }

    // Apply WebGL filters to video thumbnails
    const vidClip = currentClip as VideoClipProps;
    applyFilters(imageCanvas, {
      brightness: vidClip?.brightness,
      contrast: vidClip?.contrast,
      hue: vidClip?.hue,
      saturation: vidClip?.saturation,
      blur: vidClip?.blur,
      sharpness: vidClip?.sharpness,
      noise: vidClip?.noise,
      vignette: vidClip?.vignette,
    });
  }
  groupRef.current?.getLayer()?.batchDraw();

  // 2) Debounced fetch of exact frames and redraw when available
  if (exactVideoUpdateTimerRef.current != null) {
    window.clearTimeout(exactVideoUpdateTimerRef.current);
    exactVideoUpdateTimerRef.current = null;
  }
  const DEBOUNCE_MS = hasCachedSamples ? 100 : 0;
  const requestKey = `${currentClipId}|${timelineStartFrame}-${timelineEndFrame}|${thumbnailWidth}x${timelineHeight}|${overHang}|${frameIndices.join(",")}`;
  exactVideoUpdateTimerRef.current = window.setTimeout(async () => {
    const mySeq = ++exactVideoUpdateSeqRef.current;
    try {
      if (lastExactRequestKeyRef.current === requestKey) {
        return;
      }
      const exactSamples = await generateTimelineSamples(
        currentClipId,
        asset.path,
        frameIndices,
        thumbnailWidth,
        timelineHeight,
        tClipWidth,
        {
          volume: (currentClip as any)?.volume,
          fadeIn: (currentClip as any)?.fadeIn,
          fadeOut: (currentClip as any)?.fadeOut,
        },
      );

      if (mySeq !== exactVideoUpdateSeqRef.current) {
        return;
      }
      const ctx2 = imageCanvas.getContext("2d");

      if (ctx2 && exactSamples) {
        ctx2.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
        let x2 = overHang;
        const targetWidth2 = Math.max(1, imageCanvas.width);
        const targetHeight2 = Math.max(1, imageCanvas.height);

        // Calculate left-side skip when resizing from left for video
        let skipRemaining2 = 0;
        if (resizeSide === "left" && clipType === "video") {
          let totalTileWidth2 = 0;
          for (let i = 0; i < exactSamples.length; i++) {
            const sample = exactSamples[i];
            const anyCanvas = sample.canvas as any;
            const tileWidth = Math.max(
              1,
              anyCanvas.width || anyCanvas.naturalWidth || 1,
            );
            totalTileWidth2 += tileWidth;
          }
          const drawableWidth2 = Math.max(0, targetWidth2 - x2);
          skipRemaining2 = Math.max(0, totalTileWidth2 - drawableWidth2);
        }

        for (let i = 0; i < exactSamples.length && x2 < targetWidth2; i++) {
          const sample = exactSamples[i];
          const inputCanvas = sample.canvas as HTMLCanvasElement;
          const canvasToTile = applyMask(
            inputCanvas,
            Math.round(frameIndices[i] * fpsAdjustment),
          );
          const anyCanvas = canvasToTile as any;
          const tileWidth = Math.max(
            1,
            anyCanvas.width || anyCanvas.naturalWidth || 1,
          );
          const tileHeight = Math.max(
            1,
            anyCanvas.height || anyCanvas.naturalHeight || 1,
          );
          const sourceHeight = Math.min(tileHeight, targetHeight2);

          // Apply left-side truncation when needed
          let srcX2 = 0;
          let availableSrcWidth2 = tileWidth;
          if (skipRemaining2 > 0) {
            const consume2 = Math.min(skipRemaining2, tileWidth);
            srcX2 = consume2;
            availableSrcWidth2 = tileWidth - consume2;
            skipRemaining2 -= consume2;
            if (availableSrcWidth2 <= 0) {
              continue; // this tile is fully truncated away
            }
          }

          const remaining2 = targetWidth2 - x2;
          if (remaining2 <= 0) break;
          const drawWidth2 = Math.min(availableSrcWidth2, remaining2);
          if (drawWidth2 <= 0) break;
          ctx2.drawImage(
            canvasToTile,
            srcX2,
            0,
            drawWidth2,
            sourceHeight,
            x2,
            0,
            drawWidth2,
            sourceHeight,
          );
          x2 += drawWidth2;
        }

        // Apply WebGL filters to exact video thumbnails
        const vidClip = currentClip as VideoClipProps;
        applyFilters(imageCanvas, {
          brightness: vidClip?.brightness,
          contrast: vidClip?.contrast,
          hue: vidClip?.hue,
          saturation: vidClip?.saturation,
          blur: vidClip?.blur,
          sharpness: vidClip?.sharpness,
          noise: vidClip?.noise,
          vignette: vidClip?.vignette,
        });
      }
    } finally {
      if (mySeq === exactVideoUpdateSeqRef.current) {
        groupRef.current?.getLayer()?.batchDraw();

        lastExactRequestKeyRef.current = requestKey;

        // Force rerender if there were no cached samples initially
        if (!hasCachedSamples) {
          setForceRerenderCounter((prev) => prev + 1);
        }
      }
    }
  }, DEBOUNCE_MS);
};
