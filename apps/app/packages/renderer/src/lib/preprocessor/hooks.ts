import { useEffect, useRef } from "react";
import { usePreprocessorJobStore, JobProgress } from "./store";

/**
 * Hook to track a preprocessor job and get real-time updates
 * @param jobId - The job ID to track
 * @param autoStart - Whether to automatically start tracking (default: true)
 * @returns Job progress and control functions
 */
export function usePreprocessorJob(jobId: string | null, autoStart = true) {
  const startTracking = usePreprocessorJobStore((s) => s.startTracking);
  const stopTracking = usePreprocessorJobStore((s) => s.stopTracking);
  const clearJob = usePreprocessorJobStore((s) => s.clearJob);
  const job = usePreprocessorJobStore((s) =>
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

    // If jobId changed or we're starting fresh, clear any existing job data
    // This ensures a clean state when resubmitting with different parameters
    if (jobId !== previousJobId.current && previousJobId.current !== null) {
      clearJob(previousJobId.current);
      hasStarted.current = false;
    }

    // Start tracking if not already started
    if (!hasStarted.current) {
      hasStarted.current = true;
      previousJobId.current = jobId;
      startTracking(jobId);
    }

    // Cleanup on unmount
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

/**
 * Hook to get all active jobs
 * @returns Array of active job progress objects
 */
export function useActiveJobs(): JobProgress[] {
  const jobs = usePreprocessorJobStore((s) => s.jobs);
  const activeJobs = usePreprocessorJobStore((s) => s.activeJobs);

  return Array.from(activeJobs)
    .map((jobId) => jobs[jobId])
    .filter(Boolean);
}

/**
 * Hook to get a specific job's progress without auto-tracking
 * @param jobId - The job ID to get progress for
 * @returns Job progress or undefined
 */
export function useJobProgress(jobId: string | null): JobProgress | undefined {
  return usePreprocessorJobStore((s) => (jobId ? s.getJob(jobId) : undefined));
}

/**
 * Hook to get job store actions
 * @returns Job store action functions
 */
export function usePreprocessorJobActions() {
  const startTracking = usePreprocessorJobStore((s) => s.startTracking);
  const stopTracking = usePreprocessorJobStore((s) => s.stopTracking);
  const fetchJobResult = usePreprocessorJobStore((s) => s.fetchJobResult);
  const clearJob = usePreprocessorJobStore((s) => s.clearJob);
  const clearCompletedJobs = usePreprocessorJobStore(
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
