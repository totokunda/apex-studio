import * as React from "react";
import { v4 as uuidv4 } from "uuid";
import { pathToFileURLString } from "@app/preload";
import { useClipStore, getTimelineHeightForClip } from "@/lib/clip";
import type {
  ImageClipProps,
  MaskClipProps,
  PreprocessorClipProps,
  VideoClipProps,
} from "@/lib/types";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import {
  connectJobWebSocket,
  disconnectJobWebSocket,
  subscribeToJobErrors,
  subscribeToJobStatus,
  subscribeToJobUpdates,
  getPreprocessorResult,
} from "@/lib/preprocessor/api";

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

const normalizePct = (p: any): number | undefined => {
  const n = typeof p === "number" ? p : Number(p);
  if (!Number.isFinite(n)) return undefined;
  const pct = n <= 1 ? n * 100 : n;
  return Math.max(0, Math.min(100, pct));
};

const rangesOverlap = (aStart: number, aEnd: number, bStart: number, bEnd: number) => {
  const as = Math.min(aStart, aEnd);
  const ae = Math.max(aStart, aEnd);
  const bs = Math.min(bStart, bEnd);
  const be = Math.max(bStart, bEnd);
  return Math.max(as, bs) < Math.min(ae, be);
};

const extractResultPath = (job: any): string | undefined => {
  const meta = (job?.latest?.metadata || {}) as any;
  const direct =
    meta?.result_path ||
    meta?.resultPath ||
    meta?.preview_path ||
    meta?.previewPath ||
    job?.result_path ||
    job?.preview_path ||
    job?.result?.result_path ||
    job?.result?.preview_path;
  return typeof direct === "string" && direct.length > 0 ? direct : undefined;
};

const mapToPreprocessorStatus = (
  jobStatus: string | undefined,
): PreprocessorClipProps["status"] | undefined => {
  const s = (jobStatus || "").toLowerCase().trim();
  // Treat any non-terminal "not started yet" state as running for clip UI purposes.
  // This prevents a queued/pending job from briefly looking "idle" and allowing
  // accidental resubmits.
  if (
    s === "queued" ||
    s === "pending" ||
    s === "submitted" ||
    s === "waiting" ||
    s === "queue" ||
    s === "running" ||
    s === "processing" ||
    s === "preview"
  ) {
    return "running";
  }
  if (s === "complete" || s === "completed") return "complete";
  if (s === "error" || s === "failed") return "failed";
  // Treat cancellations as "idle" on the clip.
  if (s === "cancelled" || s === "canceled") return undefined;
  return undefined;
};

async function createNewClipFromResultAsset(params: {
  parentClipId: string;
  preprocessor: PreprocessorClipProps;
  resultAssetId: string;
}) {
  const { parentClipId, preprocessor, resultAssetId } = params;
  const state = useClipStore.getState();
  const parentClip = state.getClipById(parentClipId);
  if (!parentClip) return;
  const parentTimelineId = (parentClip as any).timelineId as string | undefined;
  if (!parentTimelineId) return;

  const asset = state.getAssetById(resultAssetId);
  if (!asset?.path) return;

  // Build absolute frame range aligned to the parent clip timeline position
  const relStart = Math.max(0, Math.round(preprocessor.startFrame ?? 0));
  const relEnd = Math.max(
    relStart + 1,
    Math.round(preprocessor.endFrame ?? ((parentClip.endFrame ?? 0) - (parentClip.startFrame ?? 0))),
  );
  const absStart = Math.max(0, Math.round((parentClip.startFrame ?? 0) + relStart));
  const absEnd = Math.max(absStart + 1, Math.round((parentClip.startFrame ?? 0) + relEnd));

  const chooseTimelineAbove = (
    desiredType: "media",
    startFrame: number,
    endFrame: number,
  ) => {
    const timelines = state.timelines || [];
    const clips = state.clips || [];
    const parentIdx = timelines.findIndex((t) => t.timelineId === parentTimelineId);

    // Find the closest compatible timeline above that has no overlap in the desired range.
    for (let i = parentIdx - 1; i >= 0; i--) {
      const t = timelines[i];
      if (!t || t.type !== desiredType) continue;
      const hasOverlap = clips.some((c) => {
        if (!c || (c as any).hidden) return false;
        if ((c as any).timelineId !== t.timelineId) return false;
        return rangesOverlap(
          startFrame,
          endFrame,
          (c as any).startFrame ?? 0,
          (c as any).endFrame ?? 0,
        );
      });
      if (!hasOverlap) return t.timelineId;
    }

    // Otherwise insert a new compatible timeline directly above the parent.
    const newTimelineId = uuidv4();
    const parentTimeline = timelines.find((t) => t.timelineId === parentTimelineId);
    state.addTimeline(
      {
        timelineId: newTimelineId,
        type: desiredType,
        timelineHeight: getTimelineHeightForClip(desiredType),
        timelineWidth:
          parentTimeline?.timelineWidth ?? timelines[timelines.length - 1]?.timelineWidth ?? 0,
        timelinePadding:
          parentTimeline?.timelinePadding ?? timelines[timelines.length - 1]?.timelinePadding ?? 24,
      },
      parentIdx - 1,
    );
    return newTimelineId;
  };

  let mi = getMediaInfoCached(asset.path);
  if (!mi) {
    mi = await getMediaInfo(asset.path, { sourceDir: "apex-cache" });
  }
  const isVideo = !!mi?.video;

  const timelineId = chooseTimelineAbove("media", absStart, absEnd);
  const width = isVideo ? mi?.video?.displayWidth : mi?.image?.width;
  const height = isVideo ? mi?.video?.displayHeight : mi?.image?.height;
  const newClipId = uuidv4();
  const base = {
    clipId: newClipId,
    timelineId,
    startFrame: absStart,
    endFrame: absEnd,
    trimStart: 0,
    trimEnd: 0,
    mediaWidth: width,
    mediaHeight: height,
    assetId: resultAssetId,
    assetIdHistory: [resultAssetId],
    preprocessors: [] as PreprocessorClipProps[],
    masks: [] as MaskClipProps[],
    transform: (parentClip as any)?.transform ?? undefined,
    originalTransform: (parentClip as any)?.originalTransform ?? undefined,
  };

  const newClip: VideoClipProps | ImageClipProps = isVideo
    ? ({ ...base, type: "video", volume: 1, fadeIn: 0, fadeOut: 0, speed: 1 } as any)
    : ({ ...base, type: "image" } as any);

  // Important ordering: add the clip first so the asset won't be pruned
  // when the preprocessor is removed from the parent clip.
  state.addClip(newClip as any);
  state.removePreprocessorFromClip(parentClipId, preprocessor.id);
}

export function usePreprocessorJobClipSync<TJob extends JobLike>(params: {
  jobsById: Record<string, TJob>;
  setJobsById: React.Dispatch<React.SetStateAction<Record<string, TJob>>>;
  addAssetAsync: (
    asset: string | { path: string } | Record<string, any>,
    sourceDir?: "user-data" | "apex-cache",
  ) => Promise<any>;
}) {
  const { jobsById, setJobsById, addAssetAsync } = params;
  const subscribedRef = React.useRef<Map<string, () => void>>(new Map());
  const finalizedByJobIdRef = React.useRef<Set<string>>(new Set());
  const finalizingByJobIdRef = React.useRef<Set<string>>(new Set());
  const fetchingResultByJobIdRef = React.useRef<Set<string>>(new Set());

  // Manage WS subscriptions for active preprocessor jobs (category "processor"),
  // but only when we can map the jobId back to a preprocessor on a clip.
  React.useEffect(() => {
    const state = useClipStore.getState();
    const clips = state.clips || [];
    const terminalStatuses = new Set([
      "complete",
      "completed",
      "cancelled",
      "canceled",
      "error",
      "failed",
    ]);

    // Build jobId -> (parentClipId, preprocessor) map once per tick.
    const jobMap = new Map<string, { clipId: string; pre: PreprocessorClipProps }>();
    for (const c of clips) {
      if (!c || (c.type !== "video" && c.type !== "image")) continue;
      const parentClipId = c.clipId;
      const pres = (c as VideoClipProps | ImageClipProps).preprocessors || [];
      for (const p of pres) {
        const jobId = p.activeJobId;
        if (!jobId) continue;
        jobMap.set(jobId, { clipId: parentClipId, pre: p });
      }
    }

    const activePreprocessorJobIds = new Set(
      [...jobMap.entries()]
        .filter(([, { pre }]) => {
          // Keep active if the preprocessor is running OR if it completed but
          // the result hasn't been applied yet (no assetId).
          const s = (pre.status || "").toLowerCase();
          const isTerminal = terminalStatuses.has(s);
          if (!isTerminal) return true;
          if ((s === "complete" || s === "completed") && !pre.assetId) return true;
          return false;
        })
        .map(([jobId]) => jobId),
    );

    // Ensure active preprocessor jobs exist in jobsById so websocket updates can
    // be merged immediately (polling may lag by up to POLL_MS).
    setJobsById((prev) => {
      let next: any = prev;
      const now = Date.now();
      for (const jobId of activePreprocessorJobIds) {
        if (next[jobId]) continue;
        if (next === prev) next = { ...prev };
        next[jobId] = {
          job_id: jobId,
          status: "running",
          category: "processor",
          message: "Preprocessor running…",
          latest: { progress: 0, status: "running", message: "Preprocessor running…", metadata: {} },
          updatedAt: now,
          progress: 0,
        };
      }
      return next;
    });

    // Connect to new jobs
    activePreprocessorJobIds.forEach((jobId) => {
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
                  message:
                    (data as any).message || (data as any).step || latest.message,
                  status: (data as any).status || latest.status,
                  metadata: { ...latest.metadata, ...(((data as any).metadata as any) || {}) },
                };
                return {
                  ...prev,
                  [jobId]: {
                    ...(job as any),
                    status: (data as any).status || (job as any).status,
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
                      (data as any).error ||
                      (data as any).message ||
                      "Unknown error",
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
      if (!activePreprocessorJobIds.has(jobId)) {
        cleanup();
        subscribedRef.current.delete(jobId);
      }
    }
  }, [jobsById, setJobsById]);

  // Apply job status/progress/results to clip preprocessors.
  React.useEffect(() => {
    const state = useClipStore.getState();
    const clips = state.clips || [];

    // jobId -> (parentClipId, preprocessorId, preprocessor)
    const map = new Map<
      string,
      { clipId: string; preId: string; pre: PreprocessorClipProps }
    >();
    for (const c of clips) {
      if (!c || (c.type !== "video" && c.type !== "image")) continue;
      const pres = (c as VideoClipProps | ImageClipProps).preprocessors || [];
      for (const p of pres) {
        const jobId = p.activeJobId;
        if (!jobId) continue;
        map.set(jobId, { clipId: c.clipId, preId: p.id, pre: p });
      }
    }

    const entries = Object.values(jobsById || {}) as any[];
    entries.forEach(async (job) => {
      const jobId = job?.job_id;
      if (!jobId) return;
      const mapped = map.get(jobId);
      if (!mapped) return;

      const { clipId, preId, pre } = mapped;

      // Guard against stale async updates: only apply job updates if this preprocessor
      // is STILL associated with this jobId. This prevents late WS/poll ticks from
      // resurrecting progress after a user stops/cancels and we clear activeJobId.
      try {
        const live = useClipStore.getState().getPreprocessorById(preId);
        if (!live || live.activeJobId !== jobId) return;
      } catch {
        return;
      }

      const effectiveStatus = (job.latest?.status || job.status || "") as string;
     
      const sLower = String(effectiveStatus || "").toLowerCase().trim();
      let nextStatus = mapToPreprocessorStatus(sLower);
      const pct = normalizePct(job.latest?.progress ?? (job as any).progress);

      // Gate "complete" until we have a usable timeline result (assetId).
      // This prevents flicker where the job is "done" but the timeline has not
      // received a working output path/asset yet.
      const rawLower = sLower;
      const jobSaysComplete = rawLower === "complete" || rawLower === "completed";
      if (jobSaysComplete && !pre.assetId) {
        nextStatus = "running";
      }

      // Never downgrade a running/queued preprocessor back to "idle" while it still has
      // an activeJobId, unless the job is explicitly terminal/canceled/failed. This closes
      // the gap where intermediate states like "queued" / "pending" / unknown variants
      // could otherwise make the UI look runnable and allow accidental resubmits.
      const isCanceled = rawLower === "cancelled" || rawLower === "canceled";
      const isFailed =
        rawLower === "error" || rawLower === "failed" || nextStatus === "failed";
      const isTerminalComplete = rawLower === "complete" || rawLower === "completed";
      const isTerminal = isCanceled || isFailed || isTerminalComplete;
      if (pre.activeJobId && !isTerminal) {
        nextStatus = "running";
      }

      // Lightweight status/progress sync (idempotent).
      const patch: Partial<PreprocessorClipProps> = {};
      let needsUpdate = false;

      if (pct != null && pct !== pre.progress) {
        patch.progress = pct;
        needsUpdate = true;
      }
      if (nextStatus !== pre.status) {
        patch.status = nextStatus;
        needsUpdate = true;
      }

      // If job was canceled, clear active job id immediately.
      if (isCanceled && pre.activeJobId) {
        patch.activeJobId = undefined;
        patch.progress = 0;
        needsUpdate = true;
      }

      if (needsUpdate) {
        try {
          state.updatePreprocessor(clipId, preId, patch);
        } catch {}
      }

      // Finalization: apply result asset and clear activeJobId once we have a usable result.
      const isComplete =
        sLower === "complete" || sLower === "completed" || nextStatus === "complete";

      if (isFailed) {
        if (pre.activeJobId) {
          try {
            state.updatePreprocessor(clipId, preId, {
              status: "failed",
              activeJobId: undefined,
              progress: 0,
            });
          } catch {}
        }
        return;
      }

      console.log("isComplete", isComplete);

      if (!isComplete) return;
      if (finalizedByJobIdRef.current.has(jobId)) return;
      if (finalizingByJobIdRef.current.has(jobId)) return;

      // Resolve result path (WS metadata preferred; fallback to API query).
      let resultPath = extractResultPath(job);
      
      if (!resultPath && !fetchingResultByJobIdRef.current.has(jobId)) {
        fetchingResultByJobIdRef.current.add(jobId);
        try {
          const res = await getPreprocessorResult(jobId);
          if (res?.success && res.data?.result_path) {
            resultPath = res.data.result_path;
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
        const asset = await addAssetAsync({ path: fileUrl }, "apex-cache");
        const assetId = asset?.id as string | undefined;
        if (!assetId) return;

        // If createNewClip is enabled (default), create a new clip and remove preprocessor.
        if (pre.createNewClip !== false) {
          await createNewClipFromResultAsset({
            parentClipId: clipId,
            preprocessor: pre,
            resultAssetId: assetId,
          });
          finalizedByJobIdRef.current.add(jobId);
          finalizingByJobIdRef.current.delete(jobId);
          return;
        }

        // Otherwise attach the result asset to the preprocessor.
        state.updatePreprocessor(clipId, preId, {
          assetId,
          status: "complete",
          activeJobId: undefined,
          progress: 100,
        });
        finalizedByJobIdRef.current.add(jobId);
        finalizingByJobIdRef.current.delete(jobId);
      } catch (err) {
        finalizingByJobIdRef.current.delete(jobId);
        console.error("Failed to apply preprocessor result to clip", err);
      }
    });
  }, [jobsById, addAssetAsync]);

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


