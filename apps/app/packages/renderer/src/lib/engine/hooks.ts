import { useEffect, useRef } from "react";
import { useEngineJobStore, JobProgress } from "./store";

export function useEngineJob(jobId: string | null, autoStart = true) {
  const startTracking = useEngineJobStore((s) => s.startTracking);
  const stopTracking = useEngineJobStore((s) => s.stopTracking);
  const clearJob = useEngineJobStore((s) => s.clearJob);
  const job = useEngineJobStore((s) => (jobId ? s.getJob(jobId) : undefined));

  const hasStarted = useRef(false);
  const previousJobId = useRef<string | null>(null);

  useEffect(() => {
    if (!jobId || !autoStart) {
      hasStarted.current = false;
      previousJobId.current = null;
      return;
    }

    if (jobId !== previousJobId.current && previousJobId.current !== null) {
      clearJob(previousJobId.current);
      hasStarted.current = false;
    }

    if (!hasStarted.current) {
      hasStarted.current = true;
      previousJobId.current = jobId;
      startTracking(jobId);
    }

    return () => {
      if (jobId) {
        stopTracking(jobId);
        hasStarted.current = false;
      }
    };
  }, [jobId, autoStart, startTracking, stopTracking, clearJob]);

  return {
    job,
    isProcessing:
      (job?.status === "running" ||
        job?.status === "pending" ||
        job?.status === "queued") &&
      (job?.progress ?? 0) > 0,
    isComplete: job?.status === "complete",
    isFailed: job?.status === "failed",
    progress: job?.progress ?? 0,
    error: job?.error,
    result: job?.result ?? null,
    startTracking: () => jobId && startTracking(jobId),
    stopTracking: () => jobId && stopTracking(jobId),
  };
}

export function useActiveJobs(): JobProgress[] {
  const jobs = useEngineJobStore((s) => s.jobs);
  const activeJobs = useEngineJobStore((s) => s.activeJobs);
  return Array.from(activeJobs)
    .map((jobId) => jobs[jobId])
    .filter(Boolean);
}

export function useJobProgress(jobId: string | null): JobProgress | undefined {
  return useEngineJobStore((s) => (jobId ? s.getJob(jobId) : undefined));
}

export function useEngineJobActions() {
  const startTracking = useEngineJobStore((s) => s.startTracking);
  const stopTracking = useEngineJobStore((s) => s.stopTracking);
  const fetchJobResult = useEngineJobStore((s) => s.fetchJobResult);
  const clearJob = useEngineJobStore((s) => s.clearJob);
  const clearCompletedJobs = useEngineJobStore((s) => s.clearCompletedJobs);

  return {
    startTracking,
    stopTracking,
    fetchJobResult,
    clearJob,
    clearCompletedJobs,
  };
}
