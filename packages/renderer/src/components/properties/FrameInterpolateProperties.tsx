import React, { useEffect, useMemo, useState } from 'react'
import { useClipStore } from '@/lib/clip';
import { VideoClipProps } from '@/lib/types';
import { getMediaInfo, getMediaInfoCached } from '@/lib/media/utils';
import PropertiesSlider from './PropertiesSlider';
import { LuInfo, LuPlay, LuX } from 'react-icons/lu';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { runPostprocessor, getPostprocessorStatus, cancelPostprocessor } from '@/lib/postprocessor/api';
import { usePostprocessorJob } from '@/lib/postprocessor/hooks';
import { useControlsStore } from '@/lib/control';
import { pathToFileURLString } from '@app/preload';
import { getPreviewPath } from '@app/preload';
import { exportClip } from '@app/export-renderer';

interface FrameInterpolatePropertiesProps {
  clipId: string;
}

const FrameInterpolateProperties: React.FC<FrameInterpolatePropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as VideoClipProps
  const sourceFps = useMemo(() => {
    if (!clip?.src) return 24;
    const info = getMediaInfoCached(clip.src);
    const fps = info?.stats?.video?.averagePacketRate;
    return (typeof fps === 'number' && isFinite(fps) && fps > 0) ? fps : 24;
  }, [clip?.src]);

  const [jobId, setJobId] = useState<string | null>(null);
  const { progress, isComplete, isFailed, error } = usePostprocessorJob(jobId);
  const updateClip = useClipStore((s) => s.updateClip);
  const fps = useControlsStore((s) => s.fps);
  const [targetFps, setTargetFps] = useState<number>(sourceFps * 2);
  const originalSrc = (clip as any)?.originalSrc as string | undefined;
  const hasInterpolated = Boolean(originalSrc && clip?.src && clip.src !== originalSrc);
  const [scale, setScale] = useState<number>(1);
  const [isPreparing, setIsPreparing] = useState<boolean>(false);
  const allowedScales = useMemo(() => [0.25, 0.5, 1.0, 2.0, 4.0], []);
  const snapScale = (value: number) => {
    const clamped = Math.min(4.0, Math.max(0.25, Number.isFinite(value) ? value : 1.0));
    let nearest = allowedScales[0];
    for (const s of allowedScales) {
      if (Math.abs(s - clamped) < Math.abs(nearest - clamped)) nearest = s;
    }
    return nearest;
  };

  const handleRunPostProcessor = async () => {
    if (!clip?.src || isPreparing) return;
    setIsPreparing(true);
    try {
      // Prefer original source if present (pre-interpolation), else current clip src
      const baseInputPath = (clip as any).originalSrc ?? clip.src;
      const mediaInfo = getMediaInfoCached(baseInputPath) || getMediaInfoCached(clip.src);

      let inputPathForProcessing = baseInputPath;

      const start = typeof mediaInfo?.startFrame === 'number' ? mediaInfo.startFrame : undefined;
      const end = typeof mediaInfo?.endFrame === 'number' ? mediaInfo.endFrame : undefined;

      // If media has an explicit frame range, export that range to a temporary video and use it
      if (typeof start === 'number' && typeof end === 'number' && end > start) {
        // Determine export resolution
        const exportWidth = (clip as any)?.originalTransform?.width
          ?? (clip as any)?.transform?.width
          ?? 0;
        const exportHeight = (clip as any)?.originalTransform?.height
          ?? (clip as any)?.transform?.height
          ?? 0;

        const filename = await getPreviewPath(`${clip.clipId}_frame_interp_src_${start}_${end}`);
        const result = await exportClip({
          mode: 'video',
          width: exportWidth || 0,
          height: exportHeight || 0,
          range: { start, end },
          clip: (clip as any),
          fps: fps,
          filename,
          encoderOptions: {
            format: 'webm',
            codec: 'vp9',
            preset: 'ultrafast',
            crf: 23,
            bitrate: '1000k',
            resolution: { width: exportWidth || 0, height: exportHeight || 0 },
            alpha: true,
          },
        });
        if (typeof result === 'string') {
          inputPathForProcessing = result;
        }
      }

      const res = await runPostprocessor({
        method: 'frame-interpolate',
        input_path: inputPathForProcessing,
        target_fps: targetFps,
        scale,
      });
      if (res.success && res.data?.job_id) {
        setJobId(res.data.job_id);
      }
    } catch {} finally {
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
          const resultPath = (status?.data as any)?.result?.result_path || (status?.data as any)?.result_path;
          if (status.success && typeof resultPath === 'string' && resultPath) {
            const fileUrl = pathToFileURLString(resultPath);
            // we need to fetch the media info to get the fps
            const mediaInfo = await getMediaInfo(fileUrl, {
              sourceDir: 'apex-cache',
            });
            console.log(mediaInfo.stats.video?.averagePacketRate);
            if (clip.src !== fileUrl) {
              updateClip(clip.clipId, { src: fileUrl, originalSrc: (clip as any).originalSrc ?? clip.src } as any);
            }
            break;
          }
          await new Promise((r) => setTimeout(r, attempt < 3 ? 300 : 800));
        }
      } catch {}
    })();
    return () => { cancelled = true; };
  }, [isComplete, jobId, clip?.clipId]);

  return (
    <div className='flex flex-col gap-y-3 p-4 justify-start'>
        <div className="text-brand-light text-[12px] font-medium text-left">
            <div className="flex items-center w-full min-w-0">
                <span>Frame Interpolation</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button className="ml-1 p-1 rounded bg-transparent hover:bg-brand-light/10 transition-colors" aria-label="What is frame interpolation?">
                      <LuInfo className="w-3.5 h-3.5 text-brand-light" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent sideOffset={4} align="center" side="bottom" hideArrow className="max-w-xs font-medium backdrop-blur-sm text-[10.5px] rounded-[6px] font-poppins bg-brand/70 border border-brand-light/10">
                    Frame interpolation generates intermediate frames to increase FPS for smooth motion. Audio is not changed.
                  </TooltipContent>
                </Tooltip>
            </div>
        </div>
        <div className='flex flex-col gap-y-3'>
          <div className="flex items-center justify-between w-full min-w-0">
            <label className="text-brand-light text-[10.5px] font-medium">{hasInterpolated ? 'Interpolated FPS' : 'Source FPS'}</label>
            <div className="rounded-full py-1 px-3 bg-brand-light/10 text-brand-light text-[10.5px] font-medium border border-brand-light/10">
              {sourceFps.toFixed(2)} fps
            </div>
          </div>
          {!hasInterpolated && (
            <PropertiesSlider
              label="Target FPS"
              labelClass='mb-1'
              value={targetFps}
              onChange={(v) => setTargetFps(Math.min(240, Math.max(sourceFps, v)))}
              suffix=" fps"
              min={sourceFps + 1}
              max={fps*10}
              step={1}
              toFixed={0}
            />
          )}

          {!hasInterpolated && (
            <div className="flex flex-col gap-y-1 w-full min-w-0">
              <div className="flex items-center w-full min-w-0 text-brand-light text-[10.5px] font-medium">
                <span>Scale</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button className="ml-1 p-1 rounded bg-transparent hover:bg-brand-light/10 transition-colors" aria-label="What is scale?">
                      <LuInfo className="w-3.5 h-3.5 text-brand-light" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent sideOffset={4} align="center" side="bottom" hideArrow className="max-w-xs font-medium backdrop-blur-sm text-[10.5px] rounded-[6px] font-poppins bg-brand/70 border border-brand-light/10">
                    Sets resolution scaling. Lower values downscale and are recommended for higher-resolution sources to save VRAM; higher values upscale low-res sources and cost more.
                  </TooltipContent>
                </Tooltip>
              </div>
              <PropertiesSlider
                label=""
                labelClass='hidden'
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
                <div className="h-full bg-brand-light transition-all duration-150" style={{ width: `${Math.max(0, Math.min(100, Math.floor(progress || 0)))}%` }} />
              </div>
              <div className="flex items-center justify-between text-brand-light/90 text-[10.5px] mt-1.5 font-medium">
                <span>{error ? 'Failed' : (isComplete ? 'Complete' : 'Processing...')}</span>
                <span>{Math.max(0, Math.min(100, Math.floor(progress || 0)))}%</span>
              </div>
            </div>
          )}

          {jobId && !isComplete && !isFailed ? (
            <button
              className="w-full mt-2 py-2 border border-brand-light/10 px-6 rounded-[6px] font-medium text-[11px] flex items-center justify-center gap-x-1 transition-all duration-200 shadow bg-red-500 text-brand-lighter hover:opacity-90"
              onClick={async () => {
                try {
                  await cancelPostprocessor(jobId);
                } catch {}
                // Reset UI state immediately after attempting cancel
                setJobId(null);
              }}
            >
              <LuX className="w-3.5 h-3.5" />
              Cancel
            </button>
          ) : hasInterpolated ? (
            <button
              className="w-full mt-2 py-2 border border-brand-light/10 px-6 rounded-[6px] font-medium text-[11px] flex items-center justify-center gap-x-2 transition-all duration-200 shadow bg-brand-light/10 text-brand-lighter hover:opacity-90"
              onClick={async () => {
                if (!clip?.clipId || !originalSrc) return;
                try {
                  await getMediaInfo(originalSrc, { sourceDir: 'apex-cache' });
                } catch {}
                updateClip(clip.clipId, { src: originalSrc } as any);
                setJobId(null);
              }}
            >
              Revert to Original
            </button>
          ) : (
            <button
              className="w-full mt-2 py-2 border border-brand-light/10 px-6 rounded-[6px] font-medium text-[11px] flex items-center justify-center gap-x-1 transition-all duration-200 shadow bg-brand text-brand-lighter hover:opacity-90 disabled:opacity-60 disabled:cursor-not-allowed"
              disabled={!clip?.src || hasInterpolated || isPreparing}
              onClick={handleRunPostProcessor}
            >
              <LuPlay className="w-3.5 h-3.5" />
              {isPreparing ? 'Preparing...' : 'Interpolate'}
            </button>
          )}
        </div>
    </div>
  )
}

export default FrameInterpolateProperties;