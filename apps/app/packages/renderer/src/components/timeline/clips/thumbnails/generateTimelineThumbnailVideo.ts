import {
  fetchCanvasSamplesStream,
  getNearestCachedCanvasSamples,
} from "@/lib/media/canvas";
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
  noShift: boolean = false,
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
  const timelineShift = noShift ? 0 : currentStartFrame - (currentClip.trimStart ?? 0);
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
    try {
      frameIndices = Array.from({ length: numColumns }, (_, i) => {
        const frameIndex = Math.floor(
          i / Math.ceil(numColumns / (timelineSpan + 1)),
        );
        const clampedIndex = Math.min(frameIndex, timelineSpan);
        return timelineStartFrame + clampedIndex;
      });
    } catch (e) {
      frameIndices = [timelineStartFrame];
    }
   
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
    let x = overHang;
    const targetWidth = Math.max(1, imageCanvas.width);
    const targetHeight = Math.max(1, imageCanvas.height);
    const expectedTileWidth = Math.max(1, Math.floor(thumbnailWidth));
    let lastCanvasToTile: HTMLCanvasElement | null = null;

    // Always paint a background so we never show transparent/white gaps.
    // We intentionally "overfill" the entire canvas; the Konva clip will cut it.
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.fillStyle = "#0B0B0D";
    ctx.fillRect(0, 0, targetWidth, targetHeight);

    // When resizing from the left for video, truncate from the left by
    // skipping the overflow width from the left side of the tile sequence.
    let skipRemaining = 0;
    if (resizeSide === "left" && clipType === "video") {
      let totalTileWidth = 0;
      for (let i = 0; i < nearest.length; i++) {
        const sample = nearest[i];
        if (!sample) {
          totalTileWidth += expectedTileWidth;
          continue;
        }
        const anyCanvas = sample.canvas as any;
        totalTileWidth += Math.max(
          1,
          anyCanvas.width || anyCanvas.naturalWidth || 1,
        );
      }
      const drawableWidth = Math.max(0, targetWidth - x);
      skipRemaining = Math.max(0, totalTileWidth - drawableWidth);
    }

    for (let i = 0; i < nearest.length && x < targetWidth; i++) {
      const sample = nearest[i];
      const inputCanvas = (sample?.canvas as HTMLCanvasElement | undefined) ?? null;
      const canvasToTile = inputCanvas
        ? applyMask(inputCanvas, Math.round(frameIndices[i] * fpsAdjustment))
        : null;
      if (canvasToTile) lastCanvasToTile = canvasToTile;
      const anyCanvas = (canvasToTile as any) ?? null;
      const tileWidth = Math.max(
        1,
        anyCanvas?.width || anyCanvas?.naturalWidth || expectedTileWidth,
      );
      const tileHeight = Math.max(
        1,
        anyCanvas?.height || anyCanvas?.naturalHeight || targetHeight,
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
      if (canvasToTile) {
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
      }
      x += drawWidth;
    }

    // If the nearest-cache pass didn't fully cover the width (e.g. cache misses),
    // repeat the last available tile so we never show the underlying fill while
    // the exact stream is still decoding.
    if (lastCanvasToTile && x < targetWidth) {
      const anyCanvas = lastCanvasToTile as any;
      const tileWidth = Math.max(
        1,
        anyCanvas.width || anyCanvas.naturalWidth || expectedTileWidth,
      );
      const tileHeight = Math.max(
        1,
        anyCanvas.height || anyCanvas.naturalHeight || targetHeight,
      );
      const sourceHeight = Math.min(tileHeight, targetHeight);
      let guard = 0;
      while (x < targetWidth && guard++ < 2048) {
        const remaining = targetWidth - x;
        const drawWidth = Math.min(tileWidth, remaining);
        if (drawWidth <= 0) break;
        ctx.drawImage(
          lastCanvasToTile,
          0,
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
    }

    // Snapshot the cached (unfiltered) render so the streaming pass can use it
    // as a base. This prevents a white/empty flash when we start the async stream.
    const seedCanvas = document.createElement("canvas");
    seedCanvas.width = targetWidth;
    seedCanvas.height = targetHeight;
    const seedCtx = seedCanvas.getContext("2d");
    if (seedCtx) {
      seedCtx.clearRect(0, 0, targetWidth, targetHeight);
      seedCtx.drawImage(imageCanvas, 0, 0);
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

        // Stream exact frames sequentially: decode/draw tile-by-tile as data arrives.
        const decoded: (HTMLCanvasElement | null)[] = new Array(
          frameIndices.length,
        ).fill(null);

        const workingCanvas = document.createElement("canvas");
        workingCanvas.width = Math.max(1, imageCanvas.width);
        workingCanvas.height = Math.max(1, imageCanvas.height);
        const wctx = workingCanvas.getContext("2d");
        if (!wctx) return;

        // Seed with the cached render so missing frames don’t show as white.
        wctx.clearRect(0, 0, workingCanvas.width, workingCanvas.height);
        if (seedCtx) {
          // If we managed to create a seed snapshot, use it.
          wctx.drawImage(seedCanvas, 0, 0);
        }

        const ctx2 = imageCanvas.getContext("2d");
        if (!ctx2) return;

        const targetWidth2 = Math.max(1, imageCanvas.width);
        const targetHeight2 = Math.max(1, imageCanvas.height);

        const vidClip = currentClip as VideoClipProps;
        const filters = {
          brightness: vidClip?.brightness,
          contrast: vidClip?.contrast,
          hue: vidClip?.hue,
          saturation: vidClip?.saturation,
          blur: vidClip?.blur,
          sharpness: vidClip?.sharpness,
          noise: vidClip?.noise,
          vignette: vidClip?.vignette,
        };

        const present = () => {
          if (mySeq !== exactVideoUpdateSeqRef.current) return;
          ctx2.clearRect(0, 0, targetWidth2, targetHeight2);
          ctx2.drawImage(workingCanvas, 0, 0);
          // Apply filters on each present, but throttle calls to keep perf OK.
          applyFilters(imageCanvas, filters);
          groupRef.current?.getLayer()?.batchDraw();
        };

        // Approximate left-side truncation during streaming using expected tile widths
        let x2 = overHang;
        let skipRemaining2 = 0;
        if (resizeSide === "left" && clipType === "video") {
          const expectedTileWidth = Math.max(1, Math.floor(thumbnailWidth));
          const totalTileWidth2 = expectedTileWidth * frameIndices.length;
          const drawableWidth2 = Math.max(0, targetWidth2 - x2);
          skipRemaining2 = Math.max(0, totalTileWidth2 - drawableWidth2);
        }

        let lastPresentAt = 0;
        for await (const { pos, frameIndex: fi, sample } of fetchCanvasSamplesStream(
          asset.path,
          frameIndices,
          thumbnailWidth,
          timelineHeight,
          { mediaInfo: mediaInfoRef ?? undefined },
        )) {
          if (mySeq !== exactVideoUpdateSeqRef.current) return;
          if (x2 >= targetWidth2) break;

          // If the stream yields null (decode fail / missing), fall back to the
          // nearest cached sample for this position so we never blank a tile.
          const fallbackWrapped = nearest[pos] ?? null;
          const inputCanvas =
            ((sample?.canvas as HTMLCanvasElement | undefined) ??
              (fallbackWrapped?.canvas as HTMLCanvasElement | undefined)) ??
            null;
          decoded[pos] = inputCanvas;

          // If we don't have a tile yet, still advance layout to keep the stream moving.
          let canvasToTile: HTMLCanvasElement | null = null;
          let tileWidth = Math.max(1, Math.floor(thumbnailWidth));
          let tileHeight = targetHeight2;
          if (inputCanvas) {
            canvasToTile = applyMask(inputCanvas, Math.round(fi * fpsAdjustment));
            const anyCanvas = canvasToTile as any;
            tileWidth = Math.max(1, anyCanvas.width || anyCanvas.naturalWidth || 1);
            tileHeight = Math.max(
              1,
              anyCanvas.height || anyCanvas.naturalHeight || 1,
            );
          }

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
              continue; // fully truncated away
            }
          }

          const remaining2 = targetWidth2 - x2;
          if (remaining2 <= 0) break;
          const drawWidth2 = Math.min(availableSrcWidth2, remaining2);
          if (drawWidth2 <= 0) break;

          if (canvasToTile) {
            wctx.drawImage(
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
          }
          x2 += drawWidth2;

          const now = performance.now();
          if (now - lastPresentAt > 140) {
            lastPresentAt = now;
            present();
          }
        }

        if (mySeq !== exactVideoUpdateSeqRef.current) return;

        // Final pass: redraw precisely (accurate left-truncation) on top of the
        // cached seed so any missing/failed samples still show *something*.
        wctx.clearRect(0, 0, targetWidth2, targetHeight2);
        // Seed with cached render to prevent blank tiles if decode failed.
        wctx.drawImage(seedCanvas, 0, 0);
        let xFinal = overHang;
        let skipFinal = 0;
        if (resizeSide === "left" && clipType === "video") {
          let total = 0;
          for (let i = 0; i < frameIndices.length; i++) {
            const fi = frameIndices[i]!;
            const c = decoded[i];
            if (c) {
              const masked = applyMask(c, Math.round(fi * fpsAdjustment));
              const anyCanvas = masked as any;
              total += Math.max(1, anyCanvas.width || anyCanvas.naturalWidth || 1);
            } else {
              total += Math.max(1, Math.floor(thumbnailWidth));
            }
          }
          const drawable = Math.max(0, targetWidth2 - xFinal);
          skipFinal = Math.max(0, total - drawable);
        }

        for (let i = 0; i < frameIndices.length && xFinal < targetWidth2; i++) {
          const fi = frameIndices[i]!;
          // Prefer decoded sample; fall back to nearest cached if missing.
          const c = decoded[i] ?? ((nearest[i]?.canvas as HTMLCanvasElement | undefined) ?? null);
          let tileCanvas: HTMLCanvasElement | null = null;
          let tileW = Math.max(1, Math.floor(thumbnailWidth));
          let tileH = targetHeight2;
          if (c) {
            tileCanvas = applyMask(c, Math.round(fi * fpsAdjustment));
            const anyCanvas = tileCanvas as any;
            tileW = Math.max(1, anyCanvas.width || anyCanvas.naturalWidth || 1);
            tileH = Math.max(1, anyCanvas.height || anyCanvas.naturalHeight || 1);
          }
          const sourceH = Math.min(tileH, targetHeight2);

          let srcX = 0;
          let availW = tileW;
          if (skipFinal > 0) {
            const consume = Math.min(skipFinal, tileW);
            srcX = consume;
            availW = tileW - consume;
            skipFinal -= consume;
            if (availW <= 0) {
              continue;
            }
          }
          const remaining = targetWidth2 - xFinal;
          if (remaining <= 0) break;
          const drawW = Math.min(availW, remaining);
          if (drawW <= 0) break;
          if (tileCanvas) {
            wctx.drawImage(
              tileCanvas,
              srcX,
              0,
              drawW,
              sourceH,
              xFinal,
              0,
              drawW,
              sourceH,
            );
          }
          xFinal += drawW;
        }

        // If we still didn't fill the width (e.g. stream yielded nulls), repeat the
        // last decoded tile to guarantee full coverage.
        if (xFinal < targetWidth2) {
          let last: HTMLCanvasElement | null = null;
          for (let i = decoded.length - 1; i >= 0; i--) {
            const c = decoded[i];
            if (c) {
              const fi = frameIndices[i]!;
              last = applyMask(c, Math.round(fi * fpsAdjustment));
              break;
            }
          }
          if (last) {
            const anyCanvas = last as any;
            const tileW = Math.max(
              1,
              anyCanvas.width || anyCanvas.naturalWidth || Math.floor(thumbnailWidth) || 1,
            );
            const tileH = Math.max(
              1,
              anyCanvas.height || anyCanvas.naturalHeight || targetHeight2,
            );
            const sourceH = Math.min(tileH, targetHeight2);
            let guard = 0;
            while (xFinal < targetWidth2 && guard++ < 2048) {
              const remaining = targetWidth2 - xFinal;
              const drawW = Math.min(tileW, remaining);
              if (drawW <= 0) break;
              wctx.drawImage(last, 0, 0, drawW, sourceH, xFinal, 0, drawW, sourceH);
              xFinal += drawW;
            }
          }
        }

        present();
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
  }
  groupRef.current?.getLayer()?.batchDraw();
};
