import * as React from "react";
import { pathToFileURLString } from "@app/preload";
import { useClipStore } from "@/lib/clip";
import type { VideoClipProps } from "@/lib/types";
import { getPostprocessorStatus } from "@/lib/postprocessor/api";
import { getMediaInfo } from "@/lib/media/utils";

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

const extractResultPath = (job: any): string | undefined => {
  const meta = (job?.latest?.metadata || {}) as any;
  const direct =
    meta?.result_path ||
    meta?.resultPath ||
    job?.result_path ||
    job?.result?.result_path ||
    job?.result?.resultPath;
  return typeof direct === "string" && direct.length > 0 ? direct : undefined;
};

export function usePostprocessorJobClipSync<TJob extends JobLike>(params: {
  jobsById: Record<string, TJob>;
  setJobsById: React.Dispatch<React.SetStateAction<Record<string, TJob>>>;
  updateClip: (clipId: string, patch: Partial<VideoClipProps>) => void;
  addAssetAsync: (
    asset: string | { path: string } | Record<string, any>,
    sourceDir?: "user-data" | "apex-cache",
  ) => Promise<any>;
}) {
  const { jobsById, setJobsById, updateClip, addAssetAsync } = params;
  const finalizedByJobIdRef = React.useRef<Set<string>>(new Set());
  const finalizingByJobIdRef = React.useRef<Set<string>>(new Set());
  const fetchingResultByJobIdRef = React.useRef<Set<string>>(new Set());

  // Ensure active postprocessor jobs exist in jobsById so they appear in JobsMenu immediately.
  React.useEffect(() => {
    const state = useClipStore.getState();
    const clips = state.clips || [];
    const activeJobIds = new Set<string>();

    for (const c of clips) {
      if (!c || c.type !== "video") continue;
      const jobId = (c as VideoClipProps).frameInterpolateJobId;
      if (jobId) activeJobIds.add(jobId);
    }

    setJobsById((prev) => {
      let next: any = prev;
      const now = Date.now();
      for (const jobId of activeJobIds) {
        if (next[jobId]) continue;
        if (next === prev) next = { ...prev };
        next[jobId] = {
          job_id: jobId,
          status: "running",
          category: "postprocessor",
          message: "Postprocessor running…",
          latest: {
            progress: 0,
            status: "running",
            message: "Postprocessor running…",
            metadata: {},
          },
          updatedAt: now,
          progress: 0,
        };
      }
      return next;
    });
  }, [setJobsById]);

  // Apply completed postprocessor results back onto the owning clip.
  React.useEffect(() => {
    const state = useClipStore.getState();
    const clips = state.clips || [];

    // jobId -> clipId
    const map = new Map<string, string>();
    for (const c of clips) {
      if (!c || c.type !== "video") continue;
      const jobId = (c as VideoClipProps).frameInterpolateJobId;
      if (!jobId) continue;
      map.set(jobId, c.clipId);
    }

    const entries = Object.values(jobsById || {}) as any[];
    entries.forEach(async (job) => {
      const jobId = job?.job_id;
      if (!jobId) return;
      const clipId = map.get(jobId);
      if (!clipId) return;

      // Guard against stale async updates: only apply if the clip still points at this job.
      try {
        const live = useClipStore.getState().getClipById(clipId) as
          | VideoClipProps
          | undefined;
        if (!live || live.type !== "video") return;
        if (live.frameInterpolateJobId !== jobId) return;
      } catch {
        return;
      }

      const effectiveStatus = String(job?.latest?.status || job?.status || "")
        .toLowerCase()
        .trim();

      const isCanceled =
        effectiveStatus === "cancelled" || effectiveStatus === "canceled";
      const isFailed = effectiveStatus === "error" || effectiveStatus === "failed";
      const isComplete =
        effectiveStatus === "complete" || effectiveStatus === "completed";

      if (isCanceled || isFailed) {
        updateClip(clipId, { frameInterpolateJobId: undefined });
        return;
      }

      if (!isComplete) return;
      if (finalizedByJobIdRef.current.has(jobId)) return;
      if (finalizingByJobIdRef.current.has(jobId)) return;

      // Resolve result path (WS/poll metadata preferred; fallback to API query).
      let resultPath = extractResultPath(job);

      if (!resultPath && !fetchingResultByJobIdRef.current.has(jobId)) {
        fetchingResultByJobIdRef.current.add(jobId);
        try {
          const res = await getPostprocessorStatus(jobId);
          const raw =
            (res?.data as any)?.result?.result_path ||
            (res?.data as any)?.result_path ||
            (res?.data as any)?.result?.resultPath ||
            (res?.data as any)?.resultPath;
          if (res?.success && typeof raw === "string" && raw) {
            resultPath = raw;
          }
        } catch {
          // no-op
        } finally {
          fetchingResultByJobIdRef.current.delete(jobId);
        }
      }

      if (!resultPath) return;

      try {
        finalizingByJobIdRef.current.add(jobId);
        const fileUrl = pathToFileURLString(resultPath);
        try {
          // Warm media metadata cache so the resulting asset has correct duration/fps.
          await getMediaInfo(fileUrl, { sourceDir: "apex-cache" });
        } catch {
          // best-effort only
        }
        const asset = await addAssetAsync({ path: fileUrl }, "apex-cache");
        const assetId = asset?.id as string | undefined;
        if (!assetId) return;

        // Keep existing history as-is; it already points to the original source.
        try {
          const live = useClipStore.getState().getClipById(clipId) as
            | VideoClipProps
            | undefined;
          const history =
            live && Array.isArray(live.assetIdHistory) && live.assetIdHistory.length > 0
              ? live.assetIdHistory
              : live && typeof live.assetId === "string"
                ? [live.assetId]
                : undefined;
          updateClip(
            clipId,
            {
              assetId,
              ...(history ? { assetIdHistory: history } : {}),
              frameInterpolateJobId: undefined,
            } as any,
          );
        } catch {
          updateClip(clipId, { assetId, frameInterpolateJobId: undefined } as any);
        }

        finalizedByJobIdRef.current.add(jobId);
      } catch (_err) {
        // best effort: leave job id in place so we can retry on next poll tick
      } finally {
        finalizingByJobIdRef.current.delete(jobId);
      }
    });
  }, [jobsById, addAssetAsync, updateClip]);
}

