import * as React from "react";
import { pathToFileURLString } from "@app/preload";
import { useClipStore } from "@/lib/clip";
import { getMediaInfoCached } from "@/lib/media/utils";
import { BASE_LONG_SIDE } from "@/lib/settings";
import { useViewportStore } from "@/lib/viewport";
import type {
  ClipTransform,
  GenerationModelClipProps,
  ModelClipProps,
} from "@/lib/types";
import {
  connectJobWebSocket,
  disconnectJobWebSocket,
  subscribeToJobErrors,
  subscribeToJobStatus,
  subscribeToJobUpdates,
} from "@/lib/engine/api";

type JobLike = {
  job_id?: string;
  status?: string;
  category?: string;
  message?: string;
  error?: string;
  latest?: {
    progress?: number | null;
    message?: string | null;
    status?: string;
    metadata?: Record<string, any> | null;
  } | null;
};

export function useEngineJobClipSync<TJob extends JobLike>(params: {
  jobsById: Record<string, TJob>;
  setJobsById: React.Dispatch<React.SetStateAction<Record<string, TJob>>>;
  updateClip: (clipId: string, patch: Partial<ModelClipProps>) => void;
  addAssetAsync: (
    asset: string | { path: string } | Record<string, any>,
    sourceDir?: "user-data" | "apex-cache",
  ) => Promise<any>;
  fps: number;
}) {
  const { jobsById, setJobsById, updateClip, addAssetAsync, fps } = params;
  const subscribedRef = React.useRef<Map<string, () => void>>(new Map());

  // Sync jobs to clip store + manage WS subscriptions for active engine jobs
  React.useEffect(() => {
    const clips = useClipStore.getState().clips;
    const getAssetById = useClipStore.getState().getAssetById;
    const activeJobs = Object.values(jobsById);

    // Keep engine job websockets connected until the result has actually been
    // applied to the clip (preview_path + assetId). This avoids a laggy "done"
    // state where the engine is complete but the timeline hasn't received a
    // usable result yet.
    const activeEngineJobIds = new Set(
      activeJobs
        .filter((j) => {
          const s = (j.status || "").toLowerCase();
          const isTerminal = [
            "complete",
            "completed",
            "cancelled",
            "canceled",
            "error",
            "failed",
          ].includes(s);

          if (j.category !== "engine" || !j.job_id) return false;
          if (!isTerminal) return true;

          // Special-case: "complete" still considered active until the clip has
          // the result applied.
          if (s === "complete" || s === "completed") {
            const meta = (j.latest?.metadata || {}) as any;
            const previewPath = meta.preview_path;
            const fileUrl = previewPath ? pathToFileURLString(previewPath) : undefined;
            const clip = clips.find(
              (c) => c.type === "model" && (c as ModelClipProps).activeJobId === j.job_id,
            ) as ModelClipProps | undefined;
            const hasAppliedResult = !!(
              clip &&
              fileUrl &&
              clip.previewPath === fileUrl &&
              clip.assetId
            );
            return !hasAppliedResult;
          }

          return false;
        })
        .map((j) => j.job_id as string),
    );

    // Connect to new jobs
    activeEngineJobIds.forEach((jobId) => {
      if (!subscribedRef.current.has(jobId)) {
        const setup = async () => {
          try {
            await connectJobWebSocket(jobId);

            const unsubUpdate = subscribeToJobUpdates(jobId, (data) => {
              setJobsById((prev) => {
                const job = prev[jobId];
                if (!job) return prev;

                const now = Date.now();
                const latest = (job.latest || {}) as any;
                const newLatest = {
                  ...latest,
                  progress:
                    typeof (data as any).progress === "number"
                      ? (data as any).progress
                      : latest.progress,
                  message: (data as any).message || (data as any).step || latest.message,
                  status: (data as any).status || latest.status,
                  metadata: { ...latest.metadata, ...(((data as any).metadata as any) || {}) },
                };

                return {
                  ...prev,
                  [jobId]: {
                    ...(job as any),
                    status: (data as any).status || (job as any).status,
                    progress:
                      typeof (data as any).progress === "number"
                        ? (data as any).progress
                        : (job as any).progress,
                    message: (data as any).message || (job as any).message,
                    latest: newLatest,
                    updatedAt: now,
                  },
                };
              });
            });

            const unsubStatus = subscribeToJobStatus(jobId, (data) => {
              setJobsById((prev) => {
                const job = prev[jobId];
                if (!job) return prev;
                return {
                  ...prev,
                  [jobId]: {
                    ...(job as any),
                    status: (data as any).status || (job as any).status,
                    updatedAt: Date.now(),
                  },
                };
              });
            });

            const unsubError = subscribeToJobErrors(jobId, (data) => {
              setJobsById((prev) => {
                const job = prev[jobId];
                if (!job) return prev;
                return {
                  ...prev,
                  [jobId]: {
                    ...(job as any),
                    status: "failed",
                    error:
                      (data as any).error || (data as any).message || "Unknown error",
                    updatedAt: Date.now(),
                  },
                };
              });
            });

            const cleanup = () => {
              unsubUpdate();
              unsubStatus();
              unsubError();
              disconnectJobWebSocket(jobId).catch(() => {});
            };

            subscribedRef.current.set(jobId, cleanup);
          } catch (err) {
            console.error(`Failed to connect to websocket for job ${jobId}`, err);
          }
        };
        setup();
      }
    });

    // Cleanup finished jobs
    for (const [jobId, cleanup] of subscribedRef.current.entries()) {
      if (!activeEngineJobIds.has(jobId)) {
        cleanup();
        subscribedRef.current.delete(jobId);
      }
    }

    activeJobs.forEach(async (job: any) => {
      if (!job.job_id) return;

      const clip = clips.find(
        (c) => c.type === "model" && (c as ModelClipProps).activeJobId === job.job_id,
      ) as ModelClipProps | undefined;
      if (!clip) return;

      const status = (job.status || "").toLowerCase();
      let newStatus: "pending" | "running" | "complete" | "failed" | undefined;

      if (status === "queued") newStatus = "pending";
      else if (status === "running" || status === "processing" || status === "preview")
        newStatus = "running";
      else if (status === "complete" || status === "completed") newStatus = "complete";
      else if (status === "error" || status === "failed") newStatus = "failed";

      const meta = job.latest?.metadata || {};
      const previewPath = meta.preview_path;
      const fileUrl = previewPath ? pathToFileURLString(previewPath) : undefined;

      const patch: Partial<ModelClipProps> = {};
      let needsUpdate = false;

      // Gate "complete" until we have a usable timeline result (preview+asset).
      // This prevents flicker where the job is "done" but nothing is rendered yet.
      let shouldFinalizeComplete = false;

      let resolvedAssetId: string | undefined;
      if (fileUrl && (clip.previewPath !== fileUrl || !clip.assetId)) {
        try {
          patch.previewPath = fileUrl;
          needsUpdate = true;
          const asset = await addAssetAsync({ path: fileUrl }, "apex-cache");
          resolvedAssetId = asset?.id;
          if (resolvedAssetId) {
            patch.assetId = resolvedAssetId;
            // Preserve any existing trims; attaching a result asset should not
            // implicitly "untrim" the user's clip.
          }
          const mediaInfo = asset?.path ? getMediaInfoCached(asset.path) : undefined;
          if (mediaInfo) {
            const duration = mediaInfo.duration;
            
            if (typeof duration === "number" && duration > 0) {
              const newEndFrame = clip.startFrame + duration * fps;
              if (clip.endFrame !== newEndFrame) {
                patch.endFrame = Math.round(newEndFrame);
                needsUpdate = true;
              }
            }
          }
        } catch (e) {
          // If we fail to attach the preview asset, do not finalize completion yet.
          console.warn("Failed to attach engine preview asset to model clip", e);
        }
      }

      const effectivePreviewPath = patch.previewPath ?? clip.previewPath;
      const effectiveAssetId = patch.assetId ?? clip.assetId;
      const hasUsableResult = !!(fileUrl && effectivePreviewPath === fileUrl && effectiveAssetId);

      if (newStatus === "complete") {
        if (hasUsableResult) {
          shouldFinalizeComplete = true;
        } else {
          // Keep the clip "running" until the result is applied.
          newStatus = "running";
        }
      }

      // Never set modelStatus to undefined via this sync.
      if (newStatus && clip.modelStatus !== newStatus) {
        patch.modelStatus = newStatus;
        needsUpdate = true;
      }

      if (shouldFinalizeComplete) {
        // Mark the job as done for the clip only once we have a usable result.
        patch.activeJobId = undefined;
        needsUpdate = true;
      }

      const gens = clip.generations || [];
      let gen: GenerationModelClipProps | null = null;
      let newGen: GenerationModelClipProps | null = null;
      const genIndex = gens.findIndex((g) => g.jobId === job.job_id);
      let genUpdate = false;

      if (genIndex >= 0) {
        gen = gens[genIndex];
        genUpdate = false;
        newGen = { ...gen };
      }

      if (newStatus && gen && newGen && gen.modelStatus !== newStatus) {
        newGen.modelStatus = newStatus;
        genUpdate = true;
      }

      if (shouldFinalizeComplete && newGen) {
        newGen.modelStatus = "complete";
        newGen.assetId = (patch.assetId ?? clip.assetId ?? "") as any;
        needsUpdate = true;
        genUpdate = true;
        window.dispatchEvent(
          new CustomEvent("generations-menu-reload", {
            detail: { jobId: job.job_id },
          }),
        );
      }

      if (patch.assetId) {
        const hasTransform = !!clip.transform;
        const hasOriginalTransform = !!clip.originalTransform;
        const asset = getAssetById(patch.assetId);

        if (!hasTransform || !hasOriginalTransform || !gen?.transform) {
          let nativeW =
            (asset && typeof asset.width === "number" ? asset.width : 0) ||
            // @ts-ignore - mediaWidth may exist on model clips when set by DynamicModelPreview
            (clip as any).mediaWidth ||
            BASE_LONG_SIDE;
          let nativeH =
            (asset && typeof asset.height === "number" ? asset.height : 0) ||
            // @ts-ignore - mediaHeight may exist on model clips when set by DynamicModelPreview
            (clip as any).mediaHeight ||
            BASE_LONG_SIDE;

          if (!Number.isFinite(nativeW) || nativeW <= 0) nativeW = BASE_LONG_SIDE;
          if (!Number.isFinite(nativeH) || nativeH <= 0) nativeH = BASE_LONG_SIDE;

          const ratio =
            nativeH > 0 && Number.isFinite(nativeW / nativeH) ? nativeW / nativeH : 1;

          const width = BASE_LONG_SIDE * ratio;
          const height = BASE_LONG_SIDE;

          const baseTransform: ClipTransform = {
            x: 0,
            y: 0,
            width,
            height,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
            cornerRadius: 0,
            opacity: 100,
            crop: { x: 0, y: 0, width: 1, height: 1 },
          };
          (patch as any).transform = baseTransform;
          (patch as any).originalTransform = { ...baseTransform };
          (patch as any).mediaWidth = nativeW;
          (patch as any).mediaHeight = nativeH;
          (patch as any).mediaAspectRatio = nativeW / nativeH;
          if (newGen) newGen.transform = baseTransform;
          genUpdate = true;
          needsUpdate = true;

          try {
            const allClips = useClipStore.getState().clips;
            const hasOtherMediaClips = allClips.some((c) => {
              const isMedia =
                c.type === "video" ||
                c.type === "image" ||
                (c.type === "model" && (c as ModelClipProps).assetId);
              return isMedia && c.clipId !== clip.clipId;
            });

            if (!hasOtherMediaClips) {
              const w = Number(nativeW);
              const h = Number(nativeH);
              if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
                const gcd = (a: number, b: number): number =>
                  b === 0 ? a : gcd(b, a % b);
                const g = gcd(Math.round(w), Math.round(h)) || 1;
                const id = `${Math.round(w / g)}:${Math.round(h / g)}`;
                useViewportStore.getState().setAspectRatio({
                  width: Math.round(w),
                  height: Math.round(h),
                  id,
                });
              }
            }
          } catch {}
        }
      }

      if (genUpdate && newGen) {
        const newGens = [...gens];
        newGens[genIndex] = newGen;
        patch.generations = newGens;
        needsUpdate = true;
      }

      if (needsUpdate || genUpdate) {
        updateClip(clip.clipId, patch);
      }
    });
  }, [jobsById, setJobsById, updateClip, addAssetAsync, fps]);

  // On unmount: ensure we disconnect any remaining subscriptions
  React.useEffect(() => {
    return () => {
      for (const cleanup of subscribedRef.current.values()) {
        try {
          cleanup();
        } catch {}
      }
      subscribedRef.current.clear();
    };
  }, []);
}


