import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import {
  JobResult,
  connectJobWebSocket as connectWS,
  disconnectJobWebSocket as disconnectWS,
  subscribeToJobErrors,
  subscribeToJobStatus,
  subscribeToJobUpdates,
  getPostprocessorStatus,
} from "./api";

export interface JobProgress {
  jobId: string;
  status: "pending" | "running" | "complete" | "failed";
  progress?: number;
  currentStep?: string;
  error?: string;
  result?: JobResult | any;
  lastUpdate?: number;
}

interface PostprocessorJobStore {
  jobs: Record<string, JobProgress>;
  activeJobs: Set<string>;
  startTracking: (jobId: string) => Promise<void>;
  stopTracking: (jobId: string) => Promise<void>;
  updateJobProgress: (jobId: string, update: Partial<JobProgress>) => void;
  fetchJobResult: (jobId: string) => Promise<boolean>;
  getJob: (jobId: string) => JobProgress | undefined;
  clearJob: (jobId: string) => void;
  clearCompletedJobs: () => void;
}

export const usePostprocessorJobStore = create<PostprocessorJobStore>()(
  subscribeWithSelector((set, get) => ({
    jobs: {},
    activeJobs: new Set(),

    startTracking: async (jobId: string) => {
      if (!jobId) return;
      const now = Date.now();
      set((state) => ({
        jobs: {
          ...state.jobs,
          [jobId]: {
            jobId,
            status: "pending",
            lastUpdate: now,
          },
        },
        activeJobs: new Set(state.activeJobs).add(jobId),
      }));

      // Wire websocket listeners
      subscribeToJobUpdates(jobId, (data) => {
        const existing = get().jobs[jobId];
        const now = Date.now();
        get().updateJobProgress(jobId, {
          progress:
            typeof data?.progress === "number"
              ? data.progress <= 1
                ? data.progress * 100
                : data.progress
              : existing?.progress,
          currentStep: (data?.message ||
            data?.step ||
            existing?.currentStep) as any,
          status: (data?.status as any) || existing?.status || "running",
          lastUpdate: now,
        });
      });

      subscribeToJobStatus(jobId, (data) => {
        get().updateJobProgress(jobId, {
          status: (data?.status as any) || "running",
          lastUpdate: Date.now(),
        });
      });

      subscribeToJobErrors(jobId, (data) => {
        get().updateJobProgress(jobId, {
          status: "failed",
          error: data?.error || data?.message || "Unknown error",
          lastUpdate: Date.now(),
        });
      });

      // Connect websocket
      try {
        await connectWS(jobId);
      } catch {}

      // Stash unsubscribers by attaching to job entry (lightweight)
      // We'll just rely on disconnectWS on stopTracking
    },

    stopTracking: async (jobId: string) => {
      if (!jobId) return;
      try {
        await disconnectWS(jobId);
      } catch {}
      set((state) => {
        const active = new Set(state.activeJobs);
        active.delete(jobId);
        return { activeJobs: active };
      });
    },

    updateJobProgress: (jobId: string, update: Partial<JobProgress>) => {
      set((state) => ({
        jobs: {
          ...state.jobs,
          [jobId]: {
            ...state.jobs[jobId],
            jobId,
            ...update,
          },
        },
      }));
    },

    fetchJobResult: async (jobId: string): Promise<boolean> => {
      try {
        const res = await getPostprocessorStatus(jobId);
        if (res.success && res.data) {
          const result = (res.data as any).result ?? res.data;
          get().updateJobProgress(jobId, {
            result,
            lastUpdate: Date.now(),
          });
          return true;
        }
        get().updateJobProgress(jobId, {
          error: res.error || "Failed to fetch result",
          lastUpdate: Date.now(),
        });
        return false;
      } catch (e) {
        get().updateJobProgress(jobId, {
          error: e instanceof Error ? e.message : "Failed to fetch result",
          lastUpdate: Date.now(),
        });
        return false;
      }
    },

    getJob: (jobId: string) => get().jobs[jobId],

    clearJob: (jobId: string) => {
      get().stopTracking(jobId);
      set((state) => {
        const jobs = { ...state.jobs };
        delete jobs[jobId];
        return { jobs };
      });
    },

    clearCompletedJobs: () => {
      const { jobs } = get();
      const done = Object.keys(jobs).filter(
        (id) => jobs[id].status === "complete" || jobs[id].status === "failed",
      );
      done.forEach((id) => get().clearJob(id));
    },
  })),
);
