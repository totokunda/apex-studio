import { useEffect, useRef } from "react";
import { usePostprocessorJobStore, JobProgress } from "./store";

export function usePostprocessorJob(jobId: string | null, autoStart = true) {
  const startTracking = usePostprocessorJobStore((s) => s.startTracking);
  const stopTracking = usePostprocessorJobStore((s) => s.stopTracking);
  const clearJob = usePostprocessorJobStore((s) => s.clearJob);
  const job = usePostprocessorJobStore((s) =>
    jobId ? s.getJob(jobId) : undefined,
  );
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
    isProcessing: job?.status === "running" || job?.status === "pending",
    isComplete: job?.status === "complete",
    isFailed: job?.status === "failed",
    progress: job?.progress ?? 0,
    error: job?.error,
    result: job?.result ?? null,
    startTracking: () => jobId && startTracking(jobId),
    stopTracking: () => jobId && stopTracking(jobId),
  };
}

export function usePostprocessorActiveJobs(): JobProgress[] {
  const jobs = usePostprocessorJobStore((s) => s.jobs);
  const activeJobs = usePostprocessorJobStore((s) => s.activeJobs);
  return Array.from(activeJobs)
    .map((id) => jobs[id])
    .filter(Boolean);
}

export function usePostprocessorJobProgress(
  jobId: string | null,
): JobProgress | undefined {
  return usePostprocessorJobStore((s) => (jobId ? s.getJob(jobId) : undefined));
}

export function usePostprocessorJobActions() {
  const startTracking = usePostprocessorJobStore((s) => s.startTracking);
  const stopTracking = usePostprocessorJobStore((s) => s.stopTracking);
  const fetchJobResult = usePostprocessorJobStore((s) => s.fetchJobResult);
  const clearJob = usePostprocessorJobStore((s) => s.clearJob);
  const clearCompletedJobs = usePostprocessorJobStore(
    (s) => s.clearCompletedJobs,
  );
  return {
    startTracking,
    stopTracking,
    fetchJobResult,
    clearJob,
    clearCompletedJobs,
  };
}
