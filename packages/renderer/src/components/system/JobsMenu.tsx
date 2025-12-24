import React, { useEffect, useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useQuery } from "@tanstack/react-query";
import { ProgressBar } from "@/components/common/ProgressBar";
import { fetchRayJobs, cancelRayJob, RayJobStatus } from "@/lib/jobs/api";
import { LuLoader, LuTrash2 } from "react-icons/lu";
import { GrTasks } from "react-icons/gr";
import { useClipStore } from "@/lib/clip";
import {
  ModelClipProps,
} from "@/lib/types";
import { useControlsStore } from "@/lib/control";
import { useEngineJobClipSync } from "@/hooks/use-engine-job-clip-sync";
import { useDownloadJobClipSync } from "@/hooks/use-download-job-clip-sync";

const POLL_MS = 2000;
const RAY_JOBS_QUERY_KEY = ["rayJobs"] as const;

type TrackedJob = RayJobStatus & {
  // Normalized 0..1 progress and last update time
  progress?: number | null;
  updatedAt: number;
};

const statusLabel = (status: string | undefined): string => {
  if (!status) return "unknown";
  const s = status.toLowerCase();
  if (s === "queued") return "Queued";
  if (s === "running" || s === "processing") return "Running";
  if (s === "complete" || s === "completed") return "Completed";
  if (s === "error" || s === "failed") return "Error";
  if (s === "cancelled" || s === "canceled") return "Cancelled";
  return status;
};

const JobsMenu: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [jobsById, setJobsById] = useState<Record<string, TrackedJob>>({});
  const [busyIds, setBusyIds] = useState<Set<string>>(new Set());
  const { updateClip, addAssetAsync } = useClipStore();
  const { fps } = useControlsStore();
  // Poll aggregated Ray jobs
  const { data: polledJobs = [] } = useQuery<RayJobStatus[]>({
    queryKey: RAY_JOBS_QUERY_KEY,
    queryFn: fetchRayJobs,
    placeholderData: (prev) => prev ?? [],
    retry: false,
    refetchOnWindowFocus: false,
    refetchInterval: POLL_MS,
    refetchIntervalInBackground: true,
  });


  useEffect(() => {
    const now = Date.now();
    setJobsById((prev) => {
      const next: Record<string, TrackedJob> = { ...prev };

      for (const job of polledJobs) {
        const id = job.job_id;
        if (!id) continue;
        const existing = next[id];
        const latest = (job as any).latest ?? existing?.latest ?? null;
        const rawProgress =
          (latest && typeof latest.progress === "number"
            ? latest.progress
            : null) ??
          (typeof (job as any).progress === "number"
            ? (job as any).progress
            : null) ??
          (typeof existing?.progress === "number" ? existing.progress : null);

        next[id] = {
          ...(existing || {}),
          ...job,
          latest,
          progress: rawProgress,
          updatedAt: existing?.updatedAt ?? now,
        };
      }

      // Optionally prune very old completed/cancelled jobs to keep the list tidy
      const cutoffMs = now - 60_000;
      for (const [id, j] of Object.entries(next)) {
        const s = (j.status || "").toLowerCase();
        if (
          (s === "complete" ||
            s === "completed" ||
            s === "cancelled" ||
            s === "canceled" ||
            s === "error") &&
          j.updatedAt < cutoffMs
        ) {
          delete next[id];
        }
      }

      return next;
    });
  }, [polledJobs]);

  useEngineJobClipSync({
    jobsById,
    setJobsById,
    updateClip,
    addAssetAsync,
    fps,
  });

  useDownloadJobClipSync({
    jobsById,
    setJobsById,
    polledJobs: polledJobs.map((j) => ({
      ...j,
      updatedAt: Date.now(),
    })),
  });

  // When a component card cancels a download, it dispatches a `jobs-menu-reload`
  // event with the associated jobId. Mark that job as canceled locally so it
  // disappears from the active jobs list immediately.
  useEffect(() => {
    const handler = (e: any) => {
      try {
        const detail = e?.detail || {};
        const jobId: string | undefined = detail.jobId;
        if (!jobId) return;
        const now = Date.now();
        setJobsById((prev) => {
          const existing = prev[jobId];
          if (!existing) return prev;
          return {
            ...prev,
            [jobId]: {
              ...existing,
              status: "canceled",
              updatedAt: now,
            },
          };
        });
      } catch {
        // no-op
      }
    };

    try {
      window.addEventListener("jobs-menu-reload", handler as EventListener);
    } catch {
      // In non-browser environments this may fail; ignore.
    }

    return () => {
      try {
        window.removeEventListener(
          "jobs-menu-reload",
          handler as EventListener,
        );
      } catch {
        // ignore
      }
    };
  }, []);

  useEffect(() => {
    const handler = (e: any) => {
      try {
        const detail = e?.detail || {};
        const jobId: string | undefined = detail.jobId;
        if (!jobId) return;

        const clips = useClipStore.getState().clips || [];
        for (const c of clips) {
          if (!c || c.type !== "model") continue;
          const modelClip = c as ModelClipProps;
          if (modelClip.activeJobId && modelClip.activeJobId === jobId) {
            updateClip(modelClip.clipId, { modelStatus: undefined } as any);
          }
        }
      } catch {
        // no-op
      }
    };

    try {
      window.addEventListener("clear-job-id", handler as EventListener);
    } catch {
      // ignore
    }

    return () => {
      try {
        window.removeEventListener("clear-job-id", handler as EventListener);
      } catch {
        // ignore
      }
    };
  }, [updateClip]);

  const activeJobs = useMemo(() => {
    const all = [...polledJobs].map((j) => ({
      ...j,
      updatedAt: Date.now(),
    }));
    return all
      .filter((j) => {
        const s = (j.status || "").toLowerCase();
        // Consider any non-terminal job as active (includes queued + running/processing)
        return ![
          "complete",
          "completed",
          "cancelled",
          "canceled",
          "error",
        ].includes(s);
      })
      .sort((a, b) => b.updatedAt - a.updatedAt);
  }, [polledJobs]);

  const activeCount = activeJobs.length;

  const handleCancel = async (jobId: string) => {
    if (!jobId) return;
    setBusyIds((prev) => new Set(prev).add(jobId));
    try {
      await cancelRayJob(jobId);
      
      window.dispatchEvent(new CustomEvent("clear-job-id", { detail: { jobId: jobId } }));
      setJobsById((prev) => {
        const existing = prev[jobId];
        if (!existing) return prev;
        return {
          ...prev,
          [jobId]: {
            ...existing,
            status: "canceled",
          },
        };
      });
    } finally {
      setBusyIds((prev) => {
        const next = new Set(prev);
        next.delete(jobId);
        return next;
      });
    }
  };

  const renderJobRow = (job: TrackedJob) => {
    const showProgress = (() => {
      const s = (job.status || "").toLowerCase();
      return s === "running" || s === "processing";
    })();
    const pct = showProgress
      ? Math.round(
          ((typeof job.progress === "number" ? job.progress : 0) || 0) * 100,
        )
      : 0;
    const msg =
      (job.latest &&
        typeof job.latest.message === "string" &&
        job.latest.message) ||
      job.message ||
      "";
    const isCancelling = busyIds.has(job.job_id);
    const fullJobId = job.job_id || "";
    const shortJobId =
      fullJobId.length > 8 ? `${fullJobId.slice(0, 8)}…` : fullJobId;

    const cancelButton = (
      <button
        type="button"
        className="h-4.5 w-4.5 items-center justify-center rounded-[4px] inline-flex hover:text-red-500 text-brand-light/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={(e) => {
          e.stopPropagation();
          handleCancel(job.job_id);
        }} 
        disabled={isCancelling}
        aria-label="Cancel job"
      >
        {isCancelling ? (
          <LuLoader className="h-3 w-3 animate-spin" />
        ) : (
          <LuTrash2 className="h-3 w-3" />
        )}
      </button>
    );

    return (
      <div
        key={job.job_id}
        className="flex flex-col gap-1 rounded-md border border-brand-light/10 bg-brand-background-dark/60 px-3 py-2.5 relative"
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-col w-full">
            <span
              className="text-[10.5px] font-medium text-brand-light/90 truncate max-w-[240px]"
              title={fullJobId}
            >
              <span className="font-mono tracking-wide bg-brand-light/10 px-1 py-0.5 rounded">
                {shortJobId || "—"}
              </span>
            </span>
            <div className="flex flex-row items-center justify-between gap-1 w-full mt-1">
              <div className="flex flex-col flex-1 min-w-0">
                {msg && (
                  <span className="text-[10px] text-brand-light/60 truncate max-w-[180px]">
                    {msg}
                  </span>
                )}
                {job.category && (
                  <span className="text-[8.5px] text-brand-light/50  tracking-wide uppercase font-medium right-1.5 absolute top-1.5 bg-brand px-1.5 py-0.5 rounded-md">
                    {job.category}
                  </span>
                )}
              </div>
              <span className="text-[10px] text-brand-light/60 whitespace-nowrap ml-1">
                {statusLabel(job.status)}
              </span>
            </div>
          </div>
        </div>
        {showProgress ? (
          <div className="flex flex-col space-y-1.5 mt-1 h-7.5">
            <ProgressBar percent={pct} className="flex-1" />
            <div className="flex flex-row items-center justify-between gap-2">
              <span className="text-[10px] text-brand-light/60">{pct}%</span>
              {cancelButton}
            </div>
          </div>
        ) : (
          <div className="flex flex-row items-center justify-end gap-1">
            {cancelButton}
          </div>
        )}
      </div>
    );
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger className="text-brand-light/90 dark h-[34px] relative flex items-center space-x-2 w-fit px-3 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer">
        <GrTasks className="w-3 h-3" />
        <span className="text-[11px]">Jobs</span>
        {activeCount > 0 && (
          <span className="ml-1 inline-flex items-center justify-center rounded-full bg-brand-light/10 px-1.5 text-[10px] text-brand-light/80 border border-brand-light/20">
            {activeCount}
          </span>
        )}
      </PopoverTrigger>
      <PopoverContent
        align="end"
        className="bg-brand-background/90 backdrop-blur-md border border-brand-light/10 rounded-[8px] p-3 font-poppins w-[360px] max-h-[70vh] overflow-y-auto"
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] uppercase tracking-wide text-brand-light/80 font-medium">
            Running Jobs
          </span>
          <span className="text-[11px] text-brand-light/60">
            {activeCount === 0 ? "Idle" : `${activeCount} active`}
          </span>
        </div>
        {activeCount > 0 ? (
          <div className="flex flex-col gap-2 pr-1">
            {activeJobs.map(renderJobRow)}
          </div>
        ) : (
          <div className="text-[11.5px] text-brand-light/70 py-0.5 font-medium">
            No running jobs.
          </div>
        )}
        <div className="mt-2 text-[10px] text-brand-light/40 flex items-center justify-between">
          <span>Updates every 2s</span>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default JobsMenu;
