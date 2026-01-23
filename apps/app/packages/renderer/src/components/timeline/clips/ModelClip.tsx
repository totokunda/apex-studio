import React, { useEffect, useMemo, useRef, useState } from "react";
import { Group, Image, Text, Arc, Circle, Rect } from "react-konva";
import RotatingCube from "@/components/common/RotatingCube";
import { renderToStaticMarkup } from "react-dom/server";
import { ModelClipProps } from "@/lib/types";
import {
  FaRegFileImage as FaRegFileImageIcon,
  FaRegFileVideo as FaRegFileVideoIcon,
  FaRegFileAudio as FaRegFileAudioIcon,
} from "react-icons/fa6";
import { TbMask as TbMaskIcon, TbFileTextSpark } from "react-icons/tb";
import {
  RiImageAiLine as RiImageAiLineIcon,
  RiVideoAiLine as RiVideoAiLineIcon,
} from "react-icons/ri";
import { LuImages as LuImagesIcon } from "react-icons/lu";
import { BiSolidVideos as BiSolidVideosIcon } from "react-icons/bi";
import {
  useEngineJob,
} from "@/lib/engine/api";
import { useClipStore } from "@/lib/clip";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import {
  generateTimelineThumbnailImage,
  generateTimelineThumbnailVideo,
} from "./thumbnails";
import { sanitizeCornerRadius } from "@/lib/konva/sanitizeCornerRadius";

type Props = {
  clipWidth: number;
  timelineHeight: number;
  cornerRadius?: number;
  currentClip: ModelClipProps | null | undefined;
  modelUiCounts: Record<string, number> | null;
  modelNameRef: any;
  modelNameWidth: number;
  clipPosition: { x: number; y: number };
  timelineWidth: number;
  imageWidth: number;
  resizeSide: "left" | "right" | null;
  zoomLevel: number;
  clipType: "video" | "image";
  tool: "move" | "resize" | null;
  thumbnailClipWidth: number;
  maxTimelineWidth: number;
  timelineDuration: [number, number];
  clipId: string;
};

const ModelClip: React.FC<Props> = ({
  clipWidth,
  timelineHeight,
  cornerRadius = 1,
  clipPosition,
  timelineWidth,
  imageWidth,
  currentClip,
  modelUiCounts,
  modelNameRef,
  modelNameWidth,
  resizeSide,
  zoomLevel,
  clipType,
  tool,
  thumbnailClipWidth,
  maxTimelineWidth,
  timelineDuration,
  clipId,
}) => {
  const safeCornerRadius = useMemo(
    () => sanitizeCornerRadius(cornerRadius, clipWidth, timelineHeight) as number,
    [cornerRadius, clipWidth, timelineHeight],
  );

  const updateClip = useClipStore((s) => s.updateClip);
  const getClipById = useClipStore((s) => s.getClipById);
  const getAssetById = useClipStore((s) => s.getAssetById);
  const isRunState =
    currentClip?.modelStatus === "running" ||
    currentClip?.modelStatus === "pending";
  const clip = getClipById(clipId) as ModelClipProps | undefined;
  const { progress } = useEngineJob(
    clip?.activeJobId ?? null,
    isRunState,
  );
  const imageCanvas = useRef<HTMLCanvasElement>(
    document.createElement("canvas"),
  );
  const fallbackCanvas = useRef<HTMLCanvasElement>(
    document.createElement("canvas"),
  );
  const displayCanvasRef = useRef<HTMLCanvasElement>(imageCanvas.current);
  const mediaInfoRef = useRef<any | null>(null);
  const groupRef = useRef<any>(null);
  const exactVideoUpdateTimerRef = useRef<number | null>(null);
  const exactVideoUpdateSeqRef = useRef(0);
  const lastExactRequestKeyRef = useRef<string | null>(null);
  const finalSrcSetRef = useRef(false);
  const [, setForceRerenderCounter] = useState(0);

  // Allow re-runs: when a new run starts, clear final-result guard and transient state
  useEffect(() => {
    const status = currentClip?.modelStatus;
    if (status === "pending" || status === "running") {
      finalSrcSetRef.current = false;
      // Reset exact request state to ensure fresh thumbnail generation for this run
      if (exactVideoUpdateTimerRef.current != null) {
        try {
          window.clearTimeout(exactVideoUpdateTimerRef.current);
        } catch {}
        exactVideoUpdateTimerRef.current = null;
      }
      lastExactRequestKeyRef.current = null;
      exactVideoUpdateSeqRef.current++;
    }
  }, [currentClip?.modelStatus]);

  const progressValue = useMemo(
    () => Math.max(0, Math.min(100, Math.floor(progress || 0))),
    [progress],
  );
  const showProgress = isRunState && progressValue >= 0;
  const [spin, setSpin] = useState(0);

  useEffect(() => {
    let raf: number | null = null;
    if (showProgress) {
      let last = performance.now();
      const step = (now: number) => {
        const dt = now - last;
        last = now;
        setSpin((prev) => (prev + dt * 0.18) % 360);
        raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    }
    return () => {
      if (raf != null) cancelAnimationFrame(raf);
    };
  }, [showProgress]);

  // Keep the preview canvas sized to the clip area
  useEffect(() => {
    // IMPORTANT: cap the internal canvas to the visible timeline width.
    // At high zoom `clipWidth` can become extremely large and exceed canvas limits
    // or hit internal tiling guards, resulting in "missing" thumbnails.
    imageCanvas.current.width = Math.max(1, imageWidth);
    imageCanvas.current.height = Math.max(1, timelineHeight);
    // Keep fallback canvas in sync with size
    fallbackCanvas.current.width = Math.max(1, imageWidth);
    fallbackCanvas.current.height = Math.max(1, timelineHeight);
  }, [imageWidth, timelineHeight]);

  // Reset video thumbnail request state when src changes to avoid stale requestKey short-circuiting
  useEffect(() => {
    if (!(currentClip as ModelClipProps)?.assetId) return;
    // Snapshot current canvas into fallback and display it during transition
    try {
      const ctx = fallbackCanvas.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(
          0,
          0,
          fallbackCanvas.current.width,
          fallbackCanvas.current.height,
        );
        ctx.drawImage(imageCanvas.current, 0, 0);
        displayCanvasRef.current = fallbackCanvas.current;
      }
    } catch {}
    if (exactVideoUpdateTimerRef.current != null) {
      window.clearTimeout(exactVideoUpdateTimerRef.current);
      exactVideoUpdateTimerRef.current = null;
    }
    lastExactRequestKeyRef.current = null;
    exactVideoUpdateSeqRef.current++;
    setForceRerenderCounter((v) => v + 1);
    try {
      groupRef.current?.getLayer()?.batchDraw();
    } catch {}
  }, [(currentClip as ModelClipProps)?.assetId]);

  const overHang = useMemo(() => {
    let overhang = 0;
    const positionX =
      clipPosition.x == 24 || clipWidth <= imageWidth ? 0 : -clipPosition.x;

    if (clipWidth - positionX <= timelineWidth && positionX > 0) {
      overhang = timelineWidth - (clipWidth - positionX);
    }
    return overhang;
  }, [clipPosition.x, clipWidth, imageWidth, timelineWidth]);

  const imageX = useMemo(() => {
    let overhang = 0;
    // Default behavior for clips that fit within timeline or are at the start
    const positionX =
      clipPosition.x == 24 || clipWidth <= imageWidth ? 0 : -clipPosition.x;
    if (clipWidth - positionX <= timelineWidth && positionX > 0) {
      overhang = timelineWidth - (clipWidth - positionX);
    }
    const x = positionX - overhang;
    return Math.max(0, x);
  }, [clipPosition.x, clipWidth, imageWidth, timelineWidth]);

  // Generate / update the timeline thumbnail when preview or source changes
  useEffect(() => {
    const asset = getAssetById(currentClip?.assetId ?? "");

    (async () => {
      try {
        if (!asset) return;
        mediaInfoRef.current = getMediaInfoCached(asset?.path);
        if (!mediaInfoRef.current) {
          mediaInfoRef.current = await getMediaInfo(asset?.path ?? currentClip?.previewPath ?? "", {
            sourceDir: "apex-cache",
          });
        }
        
        const clip = getClipById(clipId) as ModelClipProps | undefined;
        if (!clip) return;
        const isVideo = !!mediaInfoRef.current?.video;
        const isImage = !!mediaInfoRef.current?.image;

        // Treat model video like a video clip: initialize trims to 0 on first update
        if (
          isVideo &&
          (!isFinite(clip.trimStart ?? 0) || !isFinite(clip.trimEnd ?? 0))
        ) {
          try {
            const ts = (clip as any)?.trimStart;
            const te = (clip as any)?.trimEnd;
            const needsInit =
              !Number.isFinite(ts) ||
              !Number.isFinite(te) ||
              ts === Infinity ||
              te === -Infinity;
            if (needsInit) {
              updateClip(clipId, { trimStart: 0, trimEnd: 0 });
            }
          } catch {}
        }
        const noopMask = (canvas: HTMLCanvasElement) => canvas;
        const noopFilters = () => {};

        if (isImage) {
          await generateTimelineThumbnailImage(
            "image",
            clip as any,
            clipId,
            mediaInfoRef.current,
            imageCanvas.current,
            timelineHeight,
            thumbnailClipWidth,
            maxTimelineWidth,
            noopMask,
            noopFilters,
            groupRef,
            () => {},
            null,
          );
          // Switch back to live canvas now that it's drawn
          displayCanvasRef.current = imageCanvas.current;
          setForceRerenderCounter((v) => v + 1);
        } else if (isVideo) {
          
          const startFrame = clip?.startFrame ?? 0;
          const endFrame = clip?.endFrame ?? startFrame + 1;
          await generateTimelineThumbnailVideo(
            "video",
            clip as any,
            clipId,
            mediaInfoRef.current,
            imageCanvas.current,
            timelineHeight,
            thumbnailClipWidth,
            // Clamp thumbnail generation to the visible timeline width, just like
            // regular TimelineClips. This avoids requesting thousands of columns
            // and prevents blank canvases at max zoom.
            maxTimelineWidth,
            timelineWidth,
            timelineDuration,
            startFrame,
            endFrame,
            overHang,
            noopMask,
            noopFilters,
            groupRef,
            resizeSide,
            exactVideoUpdateTimerRef,
            exactVideoUpdateSeqRef,
            lastExactRequestKeyRef,
            setForceRerenderCounter,
          );
          // Switch back to live canvas after video tiles drawn
          displayCanvasRef.current = imageCanvas.current;
          setForceRerenderCounter((v) => v + 1);
        }
        try {
          groupRef.current?.getLayer()?.batchDraw();
        } catch (e) {
          console.error(e);
        }
      } catch (e) {
        console.error(e);
      }
    })();
  }, [
    (currentClip as ModelClipProps)?.assetId,
    (currentClip as ModelClipProps)?.previewPath,
    clipWidth,
    timelineHeight,
    clipId,
    getClipById,
    updateClip,
    zoomLevel,
    clipType,
    tool,
    thumbnailClipWidth,
    maxTimelineWidth,
    timelineDuration[0],
    timelineDuration[1],
    overHang,
    resizeSide,
    setForceRerenderCounter,
  ]);

  return (
    <>
      <Group ref={groupRef} width={clipWidth} height={timelineHeight}>
        <Rect
          x={0}
          y={0}
          width={clipWidth}
          height={timelineHeight}
          cornerRadius={safeCornerRadius}
          fillLinearGradientStartPoint={{ x: 0, y: 0 }}
          fillLinearGradientEndPoint={{ x: 0, y: timelineHeight }}
          fillLinearGradientColorStops={[
            0,
            "#6F56C6",
            0.08,
            "#6A50C0",
            0.5,
            "#5A40B2",
            1,
            "#4A329E",
          ]}
          shadowColor={"#000000"}
          fill={isRunState ? "#0B0B0D" : undefined}
          shadowBlur={8}
          shadowOffsetY={2}
          shadowOpacity={0.22}
        />
        <Image
          x={imageX}
          y={0}
          image={displayCanvasRef.current}
          width={imageWidth}
          height={timelineHeight}
          cornerRadius={safeCornerRadius}
        />

        {(() => {
          const size = Math.max(
            10,
            Math.min(18, Math.floor(timelineHeight * 0.55)),
          );
          const cx = Math.floor(size / 2) + 4;
          const cy = timelineHeight - 14;
          return (
            <>
              {showProgress && !mediaInfoRef.current ? (
                <Group listening={false}>
                  <Circle
                    x={cx}
                    y={cy}
                    radius={8}
                    stroke={"#9CA3AF"}
                    strokeWidth={1}
                    opacity={0.7}
                  />
                  <Arc
                    x={cx}
                    y={cy}
                    innerRadius={4}
                    outerRadius={6}
                    angle={(progressValue / 100) * 360}
                    rotation={-90 + spin}
                    stroke={"#D1D5DB"}
                    strokeWidth={2}
                    fillEnabled={false}
                    opacity={1}
                  />
                </Group>
              ) : (
                <RotatingCube
                  baseColors={[
                    "#ffffff",
                    "#6247AA",
                    "#6247AA",
                    "#6247AA",
                    "#6247AA",
                    "#ffffff",
                  ]}
                  x={cx}
                  y={cy}
                  size={8}
                  opacity={1}
                  stroke="#ffffff"
                  strokeWidth={1}
                  phaseKey={String(clipId)}
                  listening={false}
                />
              )}
              <Text
                ref={modelNameRef}
                x={size + 7}
                y={timelineHeight - 19}
                text={currentClip?.manifest?.metadata?.name ?? ""}
                fontSize={10}
                fontFamily="Poppins"
                fontStyle="500"
                fill="white"
                align="left"
              />
              {(() => {
                const counts = modelUiCounts || {};
                const ordered: { Icon: any; count: number }[] = [
                  { Icon: FaRegFileImageIcon, count: counts["image"] || 0 },
                  { Icon: FaRegFileVideoIcon, count: counts["video"] || 0 },
                  { Icon: FaRegFileAudioIcon, count: counts["audio"] || 0 },
                  { Icon: TbFileTextSpark, count: counts["text"] || 0 },
                  {
                    Icon: TbMaskIcon,
                    count:
                      (counts["image+mask"] || 0) + (counts["video+mask"] || 0),
                  },
                  {
                    Icon: RiImageAiLineIcon,
                    count: counts["image+preprocessor"] || 0,
                  },
                  {
                    Icon: RiVideoAiLineIcon,
                    count: counts["video+preprocessor"] || 0,
                  },
                  { Icon: LuImagesIcon, count: counts["image_list"] || 0 },
                  { Icon: BiSolidVideosIcon, count: counts["video_list"] || 0 },
                ].filter((i) => i.count > 0);
                if (ordered.length === 0) return null;
                const iconSlotWidth = 28;
                const totalIconsWidth = ordered.length * iconSlotWidth;
                const rightPadding = 0;
                const modelName = currentClip?.manifest?.metadata?.name ?? "";
                if (modelName && modelNameWidth === 0) return null;
                const leftOccupied = size + 7 + modelNameWidth + 6;
                const availableRightWidth = Math.max(
                  0,
                  clipWidth - leftOccupied,
                );
                if (availableRightWidth < totalIconsWidth) return null;
                const startX = Math.max(
                  6,
                  clipWidth - totalIconsWidth - rightPadding,
                );
                const startY = timelineHeight - 19;
                let curX = startX;
                return ordered.map((it, idx) => {
                  const Ico = it.Icon;
                  const group = (
                    <Group key={`mstat-${idx}`}>
                      <Image
                        x={curX}
                        y={startY - 1}
                        width={12}
                        height={12}
                        image={(() => {
                          const svg = renderToStaticMarkup(
                            React.createElement(Ico, {
                              size: 11,
                              color: "#FFFFFF",
                            }),
                          );
                          const img = new (window as any).Image();
                          (img as any).crossOrigin = "anonymous";
                          img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
                          return img as any;
                        })()}
                        opacity={1}
                      />
                      <Text
                        x={curX + 16}
                        y={startY - 1}
                        text={`${it.count}`}
                        fontSize={11}
                        fontStyle="500"
                        fontFamily="Poppins"
                        fill="rgba(255,255,255,0.82)"
                      />
                    </Group>
                  );
                  curX += iconSlotWidth;
                  return group;
                });
              })()}
            </>
          );
        })()}
      </Group>
    </>
  );
};

export default ModelClip;
