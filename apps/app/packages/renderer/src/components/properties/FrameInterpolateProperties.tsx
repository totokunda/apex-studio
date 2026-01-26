import React, { useEffect, useMemo, useState } from "react";
import { useClipStore } from "@/lib/clip";
import { VideoClipProps } from "@/lib/types";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import PropertiesSlider from "./PropertiesSlider";
import { LuInfo, LuPlay } from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  runPostprocessor,
  getPostprocessorStatus,
  cancelPostprocessor,
} from "@/lib/postprocessor/api";
import { usePostprocessorJob } from "@/lib/postprocessor/hooks";
import { useControlsStore } from "@/lib/control";
import { pathToFileURLString } from "@app/preload";
import { getPreviewPath } from "@app/preload";
import { exportClip } from "@app/export-renderer";
import { v4 as uuidv4 } from "uuid";
import { cn } from "@/lib/utils";
import { TbCancel } from "react-icons/tb";
import { prepareExportClipsForValue } from "@/lib/prepareExportClips";

interface FrameInterpolatePropertiesProps {
  clipId: string;
}

const FrameInterpolateProperties: React.FC<FrameInterpolatePropertiesProps> = ({
  clipId,
}) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as VideoClipProps;
  const getAssetById = useClipStore((s) => s.getAssetById);
  const asset = useMemo(() => getAssetById(clip.assetId), [clip.assetId]);
  if (!asset) return null;
  const sourceFps = useMemo(() => {
    if (!asset.path) return 24;
    const info = getMediaInfoCached(asset.path);
    const fps = info?.stats?.video?.averagePacketRate;
    return typeof fps === "number" && isFinite(fps) && fps > 0 ? fps : 24;
  }, [asset.path]);

  const [jobId, setJobId] = useState<string | null>(null);
  const getClipsForGroup = useClipStore((s) => s.getClipsForGroup);
  const getClipsByType = useClipStore((s) => s.getClipsByType);
  const getClipPositionScore = useClipStore((s) => s.getClipPositionScore);
  const timelines = useClipStore((s) => s.timelines);
  const { progress, isComplete, isFailed, error } = usePostprocessorJob(jobId);
  const updateClip = useClipStore((s) => s.updateClip);
  const fps = useControlsStore((s) => s.fps);
  const [multiplier, setMultiplier] = useState<number>(2);
  const targetFps = useMemo(
    () => Math.round(sourceFps * multiplier),
    [sourceFps, multiplier],
  );
  const originalAssetId = clip.assetIdHistory?.[clip.assetIdHistory?.length - 1];
  const hasInterpolated = Boolean(
    originalAssetId && clip.assetId !== originalAssetId,
  );
  const [scale, setScale] = useState<number>(1);
  const addAsset = useClipStore((s) => s.addAsset);
  const [isPreparing, setIsPreparing] = useState<boolean>(false);
  const allowedScales = useMemo(() => [0.25, 0.5, 1.0, 2.0, 4.0], []);
  const snapScale = (value: number) => {
    const clamped = Math.min(
      4.0,
      Math.max(0.25, Number.isFinite(value) ? value : 1.0),
    );
    let nearest = allowedScales[0];
    for (const s of allowedScales) {
      if (Math.abs(s - clamped) < Math.abs(nearest - clamped)) nearest = s;
    }
    return nearest;
  };

  const handleRunPostProcessor = async () => {
    if (!asset.path || isPreparing) return;
    setIsPreparing(true);
    try {
      // Prefer original source if present (pre-interpolation), else current clip src
      const baseInputPath = originalAssetId ? (getAssetById(originalAssetId)?.path || asset.path) : asset.path;
      const mediaInfo =
        getMediaInfoCached(baseInputPath) || getMediaInfoCached(asset.path);

      let inputPathForProcessing = baseInputPath;

      const start =
        typeof mediaInfo?.startFrame === "number"
          ? mediaInfo.startFrame
          : undefined;
      const end =
        typeof mediaInfo?.endFrame === "number"
          ? mediaInfo.endFrame
          : undefined;

      // If media has an explicit frame range, export that range to a temporary video and use it
      if (typeof start === "number" && typeof end === "number" && end > start) {
        // Determine export resolution
        const exportWidth =
          (clip as any)?.originalTransform?.width ??
          (clip as any)?.transform?.width ??
          0;
        const exportHeight =
          (clip as any)?.originalTransform?.height ??
          (clip as any)?.transform?.height ??
          0;

        const filename = await getPreviewPath(
          `${clip.clipId}_frame_interp_src_${start}_${end}_${uuidv4()}`,
        );
        const clipToExport = { ...clip } as any;
        clipToExport.speed = 1;
        // Map project frame indices to source frame indices for accurate export at source FPS
        const startSource = Math.max(
          0,
          Math.floor(
            ((Number(start) || 0) * Math.max(1, sourceFps)) / Math.max(1, fps),
          ),
        );
        const endSourceRaw = Math.max(
          0,
          Math.floor(
            ((Number(end) || 0) * Math.max(1, sourceFps)) / Math.max(1, fps),
          ),
        );
        const endSource = Math.max(startSource + 1, endSourceRaw);
        clipToExport.startFrame = startSource;
        clipToExport.endFrame = endSource;
        const prepared = prepareExportClipsForValue(clipToExport, {
          getClipsForGroup,
          getClipsByType,
          getClipPositionScore,
          timelines,
          getAssetById,
          aspectRatio: { width: exportWidth || 0, height: exportHeight || 0 },
        });

        const { exportClips } = prepared;
        const result = await exportClip({
          mode: "video",
          width: exportWidth || 0,
          height: exportHeight || 0,
          range: { start: startSource, end: endSource },
          clip: exportClips[0],
          fps: sourceFps,
          includeAudio: true,
          filename,
          encoderOptions: {
            format: "webm",
            codec: "vp9",
            preset: "ultrafast",
            crf: 23,
            bitrate: "1000k",
            resolution: { width: exportWidth || 0, height: exportHeight || 0 },
            alpha: true,
          },
        });
        if (typeof result === "string") {
          inputPathForProcessing = result;
        }
      }

      const res = await runPostprocessor({
        method: "frame-interpolate",
        input_path: inputPathForProcessing,
        target_fps: targetFps,
        scale,
      });
      if (res.success && res.data?.job_id) {
        setJobId(res.data.job_id);
      }
    } catch {
    } finally {
      setIsPreparing(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!clip || !isComplete || !jobId) return;
      try {
        // Poll a few times in case result propagation lags behind completion
        for (let attempt = 0; attempt < 10 && !cancelled; attempt++) {
          const status = await getPostprocessorStatus(jobId);
          const resultPath =
            (status?.data as any)?.result?.result_path ||
            (status?.data as any)?.result_path;
          if (status.success && typeof resultPath === "string" && resultPath) {
            const fileUrl = pathToFileURLString(resultPath);
            // we need to fetch the media info to get the fps (warm cache)
            await getMediaInfo(fileUrl, {
              sourceDir: "apex-cache",
            });
            let newAsset = addAsset({ path: fileUrl });
            if (asset.id !== newAsset.id) {
              updateClip(clip.clipId, {
                assetId: newAsset.id,
                assetIdHistory: [...(clip.assetIdHistory || []), originalAssetId || asset.id],
              });
            }
            break;
          }
          await new Promise((r) => setTimeout(r, attempt < 3 ? 300 : 800));
        }
      } catch {}
    })();
    return () => {
      cancelled = true;
    };
  }, [isComplete, jobId, clip?.clipId]);

  return (
    <div className="flex flex-col gap-y-3 p-4 justify-start">
      <div className="text-brand-light text-[12px] font-medium text-left">
        <div className="flex items-center w-full min-w-0">
          <span>Frame Interpolation</span>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="ml-1 p-1 rounded bg-transparent hover:bg-brand-light/10 transition-colors"
                aria-label="What is frame interpolation?"
              >
                <LuInfo className="w-3.5 h-3.5 text-brand-light" />
              </button>
            </TooltipTrigger>
            <TooltipContent
              sideOffset={4}
              align="center"
              side="bottom"
              hideArrow
              className="max-w-xs font-medium backdrop-blur-sm text-[10.5px] rounded-[6px] font-poppins bg-brand/70 border border-brand-light/10"
            >
              Frame interpolation generates intermediate frames to increase FPS
              for smooth motion. Audio is not changed.
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      <div className="flex flex-col gap-y-3">
        <div
          className={cn(
            "flex items-center justify-between w-full min-w-0",
            !hasInterpolated && "hidden",
          )}
        >
          <label className="text-brand-light text-[10.5px] font-medium">
            {hasInterpolated ? "Interpolated FPS" : "Source FPS"}
          </label>
          <div className="rounded py-1 px-3 bg-brand-light/10 text-brand-light text-[10.5px] font-medium border border-brand-light/10">
            {sourceFps.toFixed(2)} fps
          </div>
        </div>
        {!hasInterpolated && (
          <div className="flex flex-col gap-y-2 w-full min-w-0">
            <div className="flex items-center w-full min-w-0 gap-x-8 justify-between">
              <label className="text-brand-light text-[10.5px] font-medium whitespace-nowrap">
                Target FPS
              </label>
              <div className="flex items-center w-full">
                <div className="flex items-center justify-between min-w-0 w-full gap-x-6">
                  <Select
                    value={String(multiplier)}
                    onValueChange={(v) => setMultiplier(parseInt(v, 10))}
                  >
                    <SelectTrigger
                      className="h-7! w-full rounded-l rounded-r-none text-[11px] bg-brand-light/10 border text-brand-light border-brand-light/10 font-medium"
                      size="sm"
                    >
                      <SelectValue placeholder="2x" />
                    </SelectTrigger>
                    <SelectContent className="dark font-poppins">
                      {Array.from({ length: 15 }, (_, i) => i + 2).map((m) => (
                        <SelectItem
                          className="text-[11px]"
                          key={m}
                          value={String(m)}
                        >
                          {m}x
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="rounded-r flex items-center justify-center w-24 px-3 h-7 border-l-0 bg-brand-light/10 text-brand-light text-[10.5px] font-medium border border-brand-light/10">
                  {targetFps.toFixed(0)} fps
                </div>
              </div>
            </div>
          </div>
        )}

        {!hasInterpolated && (
          <div className="flex flex-col gap-y-1 w-full min-w-0">
            <div className="flex items-center w-full min-w-0 text-brand-light text-[10.5px] font-medium">
              <span>Scale</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    className="ml-1 p-1 rounded bg-transparent hover:bg-brand-light/10 transition-colors"
                    aria-label="What is scale?"
                  >
                    <LuInfo className="w-3.5 h-3.5 text-brand-light" />
                  </button>
                </TooltipTrigger>
                <TooltipContent
                  sideOffset={4}
                  align="center"
                  side="bottom"
                  hideArrow
                  className="max-w-xs font-medium backdrop-blur-sm text-[10.5px] rounded-[6px] font-poppins bg-brand/70 border border-brand-light/10"
                >
                  Sets resolution scaling. Lower values downscale and are
                  recommended for higher-resolution sources to save VRAM; higher
                  values upscale low-res sources and cost more.
                </TooltipContent>
              </Tooltip>
            </div>
            <PropertiesSlider
              label=""
              labelClass="hidden"
              value={scale}
              onChange={(v) => setScale(snapScale(v as number))}
              min={0.25}
              max={4}
              step={0.25}
              toFixed={2}
            />
          </div>
        )}
        {jobId && !isComplete && !isFailed && (
          <div className="mt-1">
            <div className="h-1.5 w-full rounded bg-brand-light/10 overflow-hidden">
              <div
                className="h-full bg-brand-light transition-all duration-150"
                style={{
                  width: `${Math.max(0, Math.min(100, Math.floor(progress || 0)))}%`,
                }}
              />
            </div>
            <div className="flex items-center justify-between text-brand-light/90 text-[10.5px] mt-1.5 font-medium">
              <span>
                {error ? "Failed" : isComplete ? "Complete" : "Processing..."}
              </span>
              <span>
                {Math.max(0, Math.min(100, Math.floor(progress || 0)))}%
              </span>
            </div>
          </div>
        )}

        {jobId && !isComplete && !isFailed ? (
          <button
            className="w-full mt-2 py-2 px-6 text-[11px] flex items-center justify-center gap-x-1 transition-all duration-200 shadow font-medium rounded-[6px] bg-red-500/50 hover:bg-red-500/60 border border-red-500/30  text-brand-lighter hover:opacity-90"
            onClick={async () => {
              try {
                await cancelPostprocessor(jobId);
              } catch {}
              // Reset UI state immediately after attempting cancel
              setJobId(null);
            }}
          >
            <TbCancel className="w-3.5 h-3.5" />
            Cancel
          </button>
        ) : hasInterpolated ? (
          <button
            className="w-full mt-2 py-2 border border-brand-light/10 px-6 rounded-[6px] font-medium text-[11px] flex items-center justify-center gap-x-2 transition-all duration-200 shadow bg-brand-light/10 text-brand-lighter hover:opacity-90"
            onClick={async () => {
              if (!clip?.clipId || !originalAssetId) return;
              const originalAsset = getAssetById(originalAssetId);
              if (!originalAsset) return;
              try {
                await getMediaInfo(originalAsset.path, { sourceDir: "apex-cache" });
              } catch {}
              updateClip(clip.clipId, { assetId: clip.assetIdHistory[clip.assetIdHistory.length - 1] } as any);
              setJobId(null);
            }}
          >
            Revert to Original
          </button>
        ) : (
          <button
            className="w-full mt-2 py-2 border border-brand-light/10 px-6 rounded-[6px] font-medium text-[11px] flex items-center justify-center gap-x-1 transition-all duration-200 shadow bg-brand text-brand-lighter hover:opacity-90 disabled:opacity-60 disabled:cursor-not-allowed"
            disabled={!asset.path || hasInterpolated || isPreparing}
            onClick={handleRunPostProcessor}
          >
            <LuPlay className="w-3.5 h-3.5" />
            {isPreparing ? "Preparing..." : "Interpolate"}
          </button>
        )}
      </div>
    </div>
  );
};

export default FrameInterpolateProperties;
