import React, { useEffect, useMemo, useRef, useState } from "react";
import { Group, Image, Text, Arc, Circle } from "react-konva";
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
  useJobProgress,
  useEngineJobActions,
} from "@/lib/engine/api";
import { useClipStore } from "@/lib/clip";
import { getMediaInfo } from "@/lib/media/utils";
import { pathToFileURLString } from "@app/preload";
import {
  generateTimelineThumbnailImage,
  generateTimelineThumbnailVideo,
} from "./thumbnails";
import { useControlsStore } from "@/lib/control";

type Props = {
  clipWidth: number;
  timelineHeight: number;
  cornerRadius?: number;
  currentClip: ModelClipProps | null | undefined;
  modelUiCounts: Record<string, number> | null;
  modelNameRef: any;
  modelNameWidth: number;
  clipId: string;
};

const ModelClip: React.FC<Props> = ({
  clipWidth,
  timelineHeight,
  cornerRadius = 1,
  currentClip,
  modelUiCounts,
  modelNameRef,
  modelNameWidth,
  clipId,
}) => {
  const updateClip = useClipStore((s) => s.updateClip);
  const getClipById = useClipStore((s) => s.getClipById);
  const isRunState =
    currentClip?.modelStatus === "running" ||
    currentClip?.modelStatus === "pending";
  const clip = getClipById(clipId) as ModelClipProps | undefined;
  const { progress, isProcessing, isComplete, isFailed } = useEngineJob(
    clip?.activeJobId ?? null,
    isRunState,
  );
  const job = useJobProgress(clip?.activeJobId ?? null);
  const { fetchJobResult } = useEngineJobActions();
  const targetFramesRef = useRef<number | null>(null);
  const initialStartRef = useRef<number | null>(null);
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
  const { fps } = useControlsStore();
  const addAsset = useClipStore((s) => s.addAsset);
  const getAssetById = useClipStore((s) => s.getAssetById);

  useEffect(() => {
    const clip = getClipById(clipId) as ModelClipProps | undefined;
    if (!clip) return;
    const start = clip.startFrame ?? 0;
    const end = clip.endFrame ?? start + 1;
    if (clip.modelStatus === "running" || clip.modelStatus === "pending") {
      if (targetFramesRef.current == null) {
        targetFramesRef.current = Math.max(1, end - start);
        initialStartRef.current = start;
      }
      const target = targetFramesRef.current || Math.max(1, end - start);
      const pct = Math.max(0, Math.min(100, Math.floor(progress || 0)));
      const grown = Math.max(1, Math.round((target * pct) / 100));
      const desiredEnd = (initialStartRef.current ?? start) + grown;
      if (desiredEnd > (clip.endFrame ?? 0)) {
        updateClip(clipId, { endFrame: desiredEnd });
      }
    } else if (clip.modelStatus === "complete") {
      if (targetFramesRef.current != null && initialStartRef.current != null) {
        const desiredEnd = initialStartRef.current + targetFramesRef.current;
        updateClip(clipId, { endFrame: desiredEnd });
      }
      targetFramesRef.current = null;
      initialStartRef.current = null;
    }
  }, [progress, isProcessing, isComplete, clipId, getClipById, updateClip]);

  // Reflect engine lifecycle into internal clip status
  useEffect(() => {
    if (!currentClip) return;
    if (isProcessing && currentClip.modelStatus !== "running") {
      updateClip(clipId, { modelStatus: "running" });
    }
    if (isComplete && currentClip.modelStatus !== "complete") {
      updateClip(clipId, { modelStatus: "complete" });
    }
    if (isFailed && currentClip.modelStatus !== "failed") {
      updateClip(clipId, { modelStatus: "failed" });
    }
  }, [clipId, currentClip, isProcessing, isComplete, isFailed, updateClip]);

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
    imageCanvas.current.width = Math.max(1, clipWidth);
    imageCanvas.current.height = Math.max(1, timelineHeight);
    // Keep fallback canvas in sync with size
    fallbackCanvas.current.width = Math.max(1, clipWidth);
    fallbackCanvas.current.height = Math.max(1, timelineHeight);
  }, [clipWidth, timelineHeight]);

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

  // Listen for preview frames and update src + thumbnail
  useEffect(() => {
    const updates = job?.updates || [];
    if (!currentClip || updates.length === 0) return;
    const last = updates[updates.length - 1];
    const meta = (last as any)?.metadata || {};
    const previewPath: string | undefined = meta?.preview_path;
    if (!previewPath) return;

    const fileUrl = pathToFileURLString(previewPath);
    if ((currentClip as ModelClipProps)?.previewPath !== fileUrl) {
      updateClip(clipId, { previewPath: fileUrl });
    }

    (async () => {
      try {
        mediaInfoRef.current = await getMediaInfo(fileUrl, {
          sourceDir: "apex-cache",
        });
        
        const clip = getClipById(clipId) as ModelClipProps | undefined;
        if (!clip) return;
        const isVideo = !!mediaInfoRef.current?.video;
        const isImage = !!mediaInfoRef.current?.image;

        // Treat model video like a video clip: initialize trims to 0 on first update
        if (isVideo) {
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
            clipWidth,
            clipWidth,
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
          // Adjust clip duration to match actual video duration (in project frames), factoring speed
          try {
            const projectFps = Math.max(1, Number(fps || 1));
            const durationSeconds = Math.max(
              0,
              Number(mediaInfoRef.current?.duration || 0),
            );
            let durationFrames = Math.max(
              1,
              Math.floor(durationSeconds * projectFps),
            );
            const speed = Math.max(
              0.1,
              Math.min(5, Number((clip as any)?.speed || 1)),
            );
            durationFrames = Math.max(1, Math.round(durationFrames / speed));
            const start = Math.max(0, clip?.startFrame ?? 0);
            const desiredEndFrame = start + durationFrames;
            if (desiredEndFrame !== (clip?.endFrame ?? start + 1)) {
              updateClip(clipId, { endFrame: desiredEndFrame });
            }
          } catch {}
          const timelineWidth = clipWidth;
          const startFrame = clip?.startFrame ?? 0;
          const endFrame = clip?.endFrame ?? startFrame + 1;
          const timelineDuration: [number, number] = [0, Math.max(1, endFrame)];
          await generateTimelineThumbnailVideo(
            "video",
            clip as any,
            clipId,
            mediaInfoRef.current,
            imageCanvas.current,
            timelineHeight,
            clipWidth,
            clipWidth,
            timelineWidth,
            timelineDuration,
            startFrame,
            endFrame,
            0,
            noopMask,
            noopFilters,
            groupRef,
            null,
            exactVideoUpdateTimerRef,
            exactVideoUpdateSeqRef,
            lastExactRequestKeyRef,
            setForceRerenderCounter,
          );
          // Switch back to live canvas after video tiles drawn
          displayCanvasRef.current = imageCanvas.current;
          setForceRerenderCounter((v) => v + 1);
        }
        // Update matching generation entry with preview src and running status
        try {
          const current = getClipById(clipId) as ModelClipProps | undefined;
          const gens = current?.generations ?? [];
          const activeId = current?.activeJobId ?? "";
          const idx = gens.findIndex((g: any) => g?.jobId === activeId);
          if (idx >= 0) {
            const g = gens[idx] as any;
            const asset = getAssetById(g.assetId);
            if ((asset?.path !== previewPath || g.modelStatus !== "running" ) && asset?.id) {
              const updated = gens.map((it, i: number) =>
                i === idx
                  ? { ...g, assetId: asset.id, modelStatus: "running" }
                  : it,
              );
              updateClip(clipId, { generations: updated } as any);
            }
          }
        } catch {}
        try {
          groupRef.current?.getLayer()?.batchDraw();
        } catch {}
      } catch {}
    })();
  }, [
    job?.updates,
    clipWidth,
    timelineHeight,
    currentClip,
    clipId,
    getClipById,
  ]);

  // On completion, ensure we fetch final result and set src to result_path once
  useEffect(() => {
    if (!clipId) return;
    if (!isComplete) return;
    // Kick a fetch to ensure result is populated
    if (!clip?.activeJobId) return;
    void fetchJobResult(clip.activeJobId);
  }, [isComplete, clipId, fetchJobResult, clip?.activeJobId]);

  useEffect(() => {
    if (!isComplete) return;
    const resultPath = (job?.result as any)?.result_path as string | undefined;
    if (!resultPath) return;
    const fileUrl = pathToFileURLString(resultPath);
    // now we add this to asset store
    const asset = addAsset({ path: fileUrl });
    if (!finalSrcSetRef.current || (currentClip as ModelClipProps)?.assetId !== asset.id) {
      finalSrcSetRef.current = true;
      const prevGenerations = Array.isArray(clip?.generations)
        ? (clip?.generations as any[])
        : [];
      const activeId = clip?.activeJobId ?? "";
      const idx = prevGenerations.findIndex((g: any) => g?.jobId === activeId);
      let nextGenerations: any[] = prevGenerations;
      if (idx >= 0) {
        const g = prevGenerations[idx] as any;
        nextGenerations = prevGenerations.map((it: any, i: number) =>
          i === idx ? { ...g, src: resultPath, modelStatus: "complete" } : it,
        );
      } else {
        nextGenerations = [
          ...prevGenerations,
          {
            jobId: activeId,
            modelStatus: "complete",
            src: resultPath,
            createdAt: Date.now(),
          },
        ];
      }
      updateClip(clipId, {
        assetId: asset.id,
        modelStatus: "complete",
        generations: nextGenerations,
        activeJobId: undefined,
      } as any);
    }
  }, [
    isComplete,
    job?.result,
    (currentClip as ModelClipProps)?.assetId,
    clipId,
    updateClip,
    clip?.generations,
  ]);

  // When clip src changes (outside of job updates), regenerate the timeline thumbnail
  useEffect(() => {
    if (!(currentClip as ModelClipProps)?.assetId) return;

    (async () => {
      try {
        mediaInfoRef.current = await getMediaInfo(currentClip?.assetId!, {
          sourceDir: "apex-cache",
        });
        const clip = getClipById(clipId) as ModelClipProps | undefined;
        if (!clip) return;
        const isVideo = !!mediaInfoRef.current?.video;
        const isImage = !!mediaInfoRef.current?.image;
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
            clipWidth,
            clipWidth,
            noopMask,
            noopFilters,
            groupRef,
            () => {},
            null,
          );
          displayCanvasRef.current = imageCanvas.current;
          setForceRerenderCounter((v) => v + 1);
        } else if (isVideo) {
          // Adjust clip duration to match actual video duration (in project frames), factoring speed
          try {
            const projectFps = Math.max(1, Number(fps || 1));
            const durationSeconds = Math.max(
              0,
              Number(mediaInfoRef.current?.duration || 0),
            );
            let durationFrames = Math.max(
              1,
              Math.floor(durationSeconds * projectFps),
            );
            const speed = Math.max(
              0.1,
              Math.min(5, Number((clip as any)?.speed || 1)),
            );
            durationFrames = Math.max(1, Math.round(durationFrames / speed));
            const start = Math.max(0, clip?.startFrame ?? 0);
            const desiredEndFrame = start + durationFrames;
            if (desiredEndFrame !== (clip?.endFrame ?? start + 1)) {
              updateClip(clipId, { endFrame: desiredEndFrame });
            }
          } catch {}
          const timelineWidth = clipWidth;
          const startFrame = clip?.startFrame ?? 0;
          const endFrame = clip?.endFrame ?? startFrame + 1;
          const timelineDuration: [number, number] = [0, Math.max(1, endFrame)];
          await generateTimelineThumbnailVideo(
            "video",
            clip as any,
            clipId,
            mediaInfoRef.current,
            imageCanvas.current,
            timelineHeight,
            clipWidth,
            clipWidth,
            timelineWidth,
            timelineDuration,
            startFrame,
            endFrame,
            0,
            noopMask,
            noopFilters,
            groupRef,
            null,
            exactVideoUpdateTimerRef,
            exactVideoUpdateSeqRef,
            lastExactRequestKeyRef,
            setForceRerenderCounter,
          );
          displayCanvasRef.current = imageCanvas.current;
          setForceRerenderCounter((v) => v + 1);
        }
        try {
          groupRef.current?.getLayer()?.batchDraw();
        } catch {}
      } catch {}
    })();
  }, [
    currentClip?.assetId,
    clipWidth,
    timelineHeight,
    fps,
    clipId,
    getClipById,
    updateClip,
  ]);

  return (
    <>
      <Group ref={groupRef} width={clipWidth} height={timelineHeight}>
        <Image
          x={0}
          y={0}
          image={displayCanvasRef.current}
          width={clipWidth}
          height={timelineHeight}
          cornerRadius={cornerRadius}
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
