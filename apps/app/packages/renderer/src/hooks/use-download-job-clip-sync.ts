import { useEffect, useRef } from "react";

import {
  connectUnifiedDownloadWebSocket,
  disconnectUnifiedDownloadWebSocket,
  onUnifiedDownloadUpdate,
  onUnifiedDownloadStatus,
  onUnifiedDownloadError,
} from "@/lib/download/api";
import { useDownloadJobIdStore } from "@/lib/download/job-id-store";
import { useQueryClient } from "@tanstack/react-query";
import { refreshManifestPart } from "@/lib/manifest/queries";

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

export function useDownloadJobClipSync<TJob extends JobLike>(params: {
  jobsById: Record<string, TJob>;
  polledJobs: TJob[];
  setJobsById: React.Dispatch<React.SetStateAction<Record<string, TJob>>>;
}) {
  const { setJobsById, polledJobs } = params;
  const {
    addJobUpdate,
    removeJobUpdates,
    removeSourceByJobId,
    getJobIdToParts,
    removeJobIdToParts,
    getJobIdToManifestId,
    removeJobIdToManifestId,
  } = useDownloadJobIdStore();
  const subscribedRef = useRef<Map<string, () => void>>(new Map());
  const prevJobStatusRef = useRef<Map<string, string>>(new Map());
  const refreshingManifestByJobIdRef = useRef<Set<string>>(new Set());
  const refreshedManifestByJobIdRef = useRef<Set<string>>(new Set());
  const refreshRunIdRef = useRef(0);
  const queryClient = useQueryClient();

  // Sync jobs to clip store + manage WS subscriptions for active download jobs
  useEffect(() => {
    const activeJobs = [...polledJobs];
    const terminalStatuses = new Set([
      "complete",
      "completed",
      "cancelled",
      "canceled",
      "error",
      "failed",
    ]);

    // Cleanup completed/cancelled jobs + refresh manifest parts, if present.
    // Important: we must not refresh on every poll tick; only refresh once when a job
    // transitions into a terminal state, and never run overlapping refreshes per jobId.
    const runId = ++refreshRunIdRef.current;
    void (async () => {
      for (const j of activeJobs) {
        // If a newer effect run started, stop processing additional jobs in this older run.
        // Do NOT abort an in-flight refresh for a jobId (that would risk partial refreshes).
        if (refreshRunIdRef.current !== runId) break;

        const jobId = j.job_id as string | undefined;
        if (!jobId) continue;

        const status = (j.status || "").toLowerCase();
        const isTerminal = terminalStatuses.has(status);

        const prevStatus = prevJobStatusRef.current.get(jobId);
        prevJobStatusRef.current.set(jobId, status);

        if (!isTerminal) continue;

        // Only refresh manifests once per completed job.
        // - Prefer edge-triggered: status becomes terminal.
        // - Also allow a one-time refresh if the app restarts and we see a terminal job
        //   with persisted jobId->parts/manifestId mappings.
        const wasTerminal = prevStatus ? terminalStatuses.has(prevStatus) : false;
        const hasRefreshed = refreshedManifestByJobIdRef.current.has(jobId);
        const shouldConsiderRefresh = !hasRefreshed && (!wasTerminal || !prevStatus);

        const parts = getJobIdToParts(jobId);
        const manifestId = getJobIdToManifestId(jobId);
        const hasRefreshPayload = !!manifestId && !!parts?.length;

        // If we didn't transition (e.g. terminal already), still allow a one-time refresh
        // when we have persisted payload (parts+manifestId) and haven't refreshed yet.
        const allowOneTimeRecoveryRefresh = !hasRefreshed && hasRefreshPayload;

        if (!(shouldConsiderRefresh || allowOneTimeRecoveryRefresh)) continue;
        if (!hasRefreshPayload) continue;
        if (refreshingManifestByJobIdRef.current.has(jobId)) continue;

        refreshingManifestByJobIdRef.current.add(jobId);
        try {
          // Refresh sequentially to avoid stampeding the API server.
          for (let idx = 0; idx < parts.length; idx++) {
            const part = parts[idx];
            await refreshManifestPart(
              manifestId,
              part,
              queryClient,
              idx === parts.length - 1,
            );
          }
        } catch (err) {
          console.error(
            `Failed to refresh manifest parts for job ${jobId} (manifest ${manifestId})`,
            err,
          );
        } finally {
          // Prevent future refresh storms regardless of success; the polling loop would
          // otherwise retry on every tick and can overwhelm the API server.
          refreshedManifestByJobIdRef.current.add(jobId);
          refreshingManifestByJobIdRef.current.delete(jobId);

          // After we refresh (or attempt to), cleanup terminal-job tracking.
          // We run this after refresh so the UI doesn't drop the job before the
          // manifest cache is updated.
          try {
            removeJobUpdates(jobId);
          } catch {}
          try {
            removeSourceByJobId(jobId);
          } catch {}

          try {
            removeJobIdToManifestId(jobId);
          } catch {}
          try {
            removeJobIdToParts(jobId);
          } catch {}
        }
      }
    })();

    const activeDownloadJobIds = new Set(
      activeJobs
        .filter((j) => {
          const s = (j.status || "").toLowerCase();
          const isActive = !terminalStatuses.has(s);
          return isActive && j.category === "download" && j.job_id;
        })
        .map((j) => j.job_id as string),
    );

    // Connect to new jobs
    activeDownloadJobIds.forEach((jobId) => {
      if (!subscribedRef.current.has(jobId)) {
        const setup = async () => {
          try {
            await connectUnifiedDownloadWebSocket(jobId);

            const unsubUpdate = onUnifiedDownloadUpdate(jobId, (data) => {
              addJobUpdate(jobId, data);
            });

            const unsubStatus = onUnifiedDownloadStatus(jobId, (data) => {
              setJobsById((prev) => {
                const job = prev[jobId];
                if (!job) return prev;
                return {
                  ...prev,
                  [jobId]: {
                    ...job,
                    status: (data?.status || job?.status),
                    updatedAt: Date.now(),
                  },
                };
              });
            });

            const unsubError = onUnifiedDownloadError(jobId, (data) => {
              setJobsById((prev) => {
                const job = prev[jobId];
                if (!job) return prev;
                return {
                  ...prev,
                  [jobId]: {
                    ...(job as any),
                    status: "failed",
                    error:
                      (data?.error || "Unknown error"),
                    updatedAt: Date.now(),
                  },
                };
              });
            });

            const cleanup = () => {
              unsubUpdate();
              unsubStatus();
              unsubError();
              disconnectUnifiedDownloadWebSocket(jobId).catch(() => {});
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
      if (!activeDownloadJobIds.has(jobId)) {
        cleanup();
        subscribedRef.current.delete(jobId);
      }
    }


  }, [
    polledJobs,
    setJobsById,
    queryClient,
    addJobUpdate,
    removeJobUpdates,
    removeSourceByJobId,
    getJobIdToParts,
    removeJobIdToParts,
    getJobIdToManifestId,
    removeJobIdToManifestId,
  ]);

  // On unmount: ensure we disconnect any remaining subscriptions
  useEffect(() => {
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


