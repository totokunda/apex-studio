import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { EngineJob, getEngineResult, JobResult } from "./api";

export interface JobProgress {
  jobId: string;
  status:
    | "pending"
    | "running"
    | "complete"
    | "failed"
    | "connected"
    | "queued";
  progress?: number;
  currentStep?: string;
  error?: string;
  result?: JobResult;
  lastUpdate?: number;
  updates?: Array<{
    message?: string;
    progress?: number;
    status?: string;
    time: number;
    metadata?: any;
  }>;
  files?: Record<
    string,
    {
      filename: string;
      progress?: number;
      downloadedBytes?: number;
      totalBytes?: number;
      downloadSpeed?: number;
      lastUpdateTime?: number;
      status?: "downloading" | "completed" | "error";
    }
  >;
}

// Normalize backend statuses to UI-known set
function normalizeStatus(status: any): JobProgress["status"] {
  try {
    const s = String(status || "").toLowerCase();
    if (s === "queued" || s === "pending" || s === "waiting" || s === "unknown")
      return "pending";
    if (
      s === "processing" ||
      s === "running" ||
      s === "in_progress" ||
      s === "in-progress"
    )
      return "running";
    if (
      s === "complete" ||
      s === "completed" ||
      s === "success" ||
      s === "succeeded"
    )
      return "complete";
    if (
      s === "failed" ||
      s === "error" ||
      s === "canceled" ||
      s === "cancelled" ||
      s === "stopped"
    )
      return "failed";
    if (s === "connected") return "connected";
  } catch {}
  return "running";
}

export interface LoraJobProgress extends JobProgress {
  loraId?: string;
  manifestId?: string;
  source?: string;
  verified?: boolean;
  remote_source?: string;
}

interface EngineJobStore {
  jobs: Record<string, JobProgress>;
  activeJobs: Set<string>;
  startTracking: (jobId: string) => Promise<EngineJob>;
  stopTracking: (jobId: string) => Promise<void>;
  updateJobProgress: (jobId: string, update: Partial<JobProgress>) => void;
  fetchJobResult: (jobId: string) => Promise<boolean>;
  getJob: (jobId: string) => JobProgress | undefined;
  clearJob: (jobId: string) => void;
  clearCompletedJobs: () => void;
  _jobInstances: Map<string, EngineJob>;
}

export const useEngineJobStore = create<EngineJobStore>()(
  subscribeWithSelector((set, get) => ({
    jobs: {},
    activeJobs: new Set(),
    _jobInstances: new Map(),

    startTracking: async (jobId: string) => {
      const existing = get()._jobInstances.get(jobId);
      if (existing) return existing;

      const job = new EngineJob(jobId);

      set((state) => {
        const prev = state.jobs[jobId];
        return {
          jobs: {
            ...state.jobs,
            [jobId]: {
              // Preserve previous job data (especially updates) if it exists
              ...(prev || {}),
              jobId,
              status: prev?.status ?? "pending",
              lastUpdate: Date.now(),
              updates: prev?.updates ?? [],
            },
          },
          activeJobs: new Set(state.activeJobs).add(jobId),
        };
      });

      get()._jobInstances.set(jobId, job);

      job.onUpdate(async (data) => {
        const now = Date.now();
        const existing = get().jobs[jobId];
        const normalizedStatus = normalizeStatus(
          (data as any)?.status || existing?.status || "running",
        );
        let nextFiles = { ...(existing?.files || {}) } as NonNullable<
          JobProgress["files"]
        >;
        try {
          const meta = data?.metadata || {};
          const filename = (meta.filename || meta.label || "").toString();
          const downloaded =
            meta.downloaded ?? meta.bytes_downloaded ?? meta.current_bytes;
          const total = meta.total ?? meta.bytes_total ?? meta.total_bytes;
          if (filename) {
            const prev = nextFiles[filename];
            let speed: number | undefined;
            if (
              typeof downloaded === "number" &&
              prev?.downloadedBytes != null &&
              prev?.lastUpdateTime
            ) {
              const dt = (now - prev.lastUpdateTime) / 1000;
              const db = downloaded - (prev.downloadedBytes || 0);
              if (dt > 0 && db > 0) speed = db / dt;
            }
            const pct =
              typeof downloaded === "number" &&
              typeof total === "number" &&
              total > 0
                ? Math.max(
                    0,
                    Math.min(100, Math.floor((downloaded / total) * 100)),
                  )
                : typeof data.progress === "number"
                  ? data.progress <= 1
                    ? data.progress * 100
                    : data.progress
                  : prev?.progress;
            nextFiles[filename] = {
              filename,
              progress: pct ?? prev?.progress,
              downloadedBytes:
                typeof downloaded === "number"
                  ? downloaded
                  : prev?.downloadedBytes,
              totalBytes: typeof total === "number" ? total : prev?.totalBytes,
              downloadSpeed: speed ?? prev?.downloadSpeed,
              lastUpdateTime: now,
              status:
                data?.status === "complete" || data?.status === "completed"
                  ? "completed"
                  : data?.status === "error"
                    ? "error"
                    : "downloading",
            };
          }
        } catch {}

        const normalizedProgress =
          typeof data?.progress === "number"
            ? data.progress <= 1
              ? Math.round(data.progress * 100)
              : Math.round(data.progress)
            : typeof existing?.progress === "number"
              ? Math.round(existing.progress)
              : undefined;
        const stepMsg = (data?.message ||
          data?.step ||
          existing?.currentStep) as any;
        const nextUpdates = [
          ...((existing?.updates as NonNullable<JobProgress["updates"]>) || []),
          {
            message: stepMsg,
            progress: normalizedProgress,
            status: normalizedStatus,
            time: now,
            metadata: data?.metadata,
          },
        ];

        get().updateJobProgress(jobId, {
          progress:
            typeof data?.progress === "number"
              ? data.progress <= 1
                ? data.progress * 100
                : data.progress
              : existing?.progress,
          currentStep: stepMsg as any,
          status: normalizedStatus,
          lastUpdate: now,
          updates: nextUpdates,
          files: nextFiles,
        });

        if (normalizedStatus === "complete") {
          const fetchWithRetry = async (attempt = 0) => {
            const delay = Math.min(500 * Math.pow(1.5, attempt), 30000);
            setTimeout(async () => {
              const success = await get().fetchJobResult(jobId);
              if (!success) fetchWithRetry(attempt + 1);
            }, delay);
          };
          fetchWithRetry();
        }
      });

      job.onStatus(async (data) => {
        get().updateJobProgress(jobId, {
          status: normalizeStatus((data as any)?.status),
          lastUpdate: Date.now(),
        });
      });

      job.onError((data) => {
        get().updateJobProgress(jobId, {
          status: "failed",
          error: data.error || data.message || "Unknown error",
          lastUpdate: Date.now(),
        });
      });

      try {
        await job.connect();
      } catch (error) {
        get().updateJobProgress(jobId, {
          status: "failed",
          error: error instanceof Error ? error.message : "Failed to connect",
          lastUpdate: Date.now(),
        });
      }

      return job;
    },

    stopTracking: async (jobId: string) => {
      const job = get()._jobInstances.get(jobId);
      if (job) {
        await job.disconnect();
        get()._jobInstances.delete(jobId);
        set((state) => {
          const newActiveJobs = new Set(state.activeJobs);
          newActiveJobs.delete(jobId);
          return { activeJobs: newActiveJobs };
        });
      }
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
        const response = await getEngineResult(jobId);
        if (response.success && response.data) {
          get().updateJobProgress(jobId, {
            result: response.data,
            lastUpdate: Date.now(),
          });
          return true;
        } else {
          get().updateJobProgress(jobId, {
            error: response.error || "Failed to fetch result",
            lastUpdate: Date.now(),
          });
          return false;
        }
      } catch (error) {
        get().updateJobProgress(jobId, {
          error:
            error instanceof Error ? error.message : "Failed to fetch result",
          lastUpdate: Date.now(),
        });
        return false;
      }
    },

    getJob: (jobId: string) => get().jobs[jobId],

    clearJob: (jobId: string) => {
      const { stopTracking } = get();
      stopTracking(jobId);
      set((state) => {
        const newJobs = { ...state.jobs };
        delete newJobs[jobId];
        return { jobs: newJobs };
      });
    },

    clearCompletedJobs: () => {
      const { jobs } = get();
      const completed = Object.keys(jobs).filter(
        (id) => jobs[id].status === "complete" || jobs[id].status === "failed",
      );
      completed.forEach((id) => get().clearJob(id));
    },
  })),
);

interface LoraJobStore {
  jobs: Record<string, LoraJobProgress>;
  activeJobs: Set<string>;
  startTracking: (
    jobId: string,
    meta?: { loraId?: string; manifestId?: string; source?: string },
  ) => Promise<EngineJob>;
  stopTracking: (jobId: string) => Promise<void>;
  updateJobProgress: (jobId: string, update: Partial<LoraJobProgress>) => void;
  fetchJobResult: (jobId: string) => Promise<boolean>;
  getJob: (jobId: string) => LoraJobProgress | undefined;
  clearJob: (jobId: string) => void;
  clearCompletedJobs: () => void;
  _jobInstances: Map<string, EngineJob>;
}

export const useLoraJobStore = create<LoraJobStore>()(
  subscribeWithSelector((set, get) => ({
    jobs: {},
    activeJobs: new Set(),
    _jobInstances: new Map(),

    startTracking: async (
      jobId: string,
      meta?: { loraId?: string; manifestId?: string; source?: string },
    ) => {
      const existing = get()._jobInstances.get(jobId);
      if (existing) return existing;

      const job = new EngineJob(jobId);

      set((state) => {
        const prev = state.jobs[jobId];
        return {
          jobs: {
            ...state.jobs,
            [jobId]: {
              ...(prev || {}),
              jobId,
              status: prev?.status ?? "pending",
              lastUpdate: Date.now(),
              updates: prev?.updates ?? [],
              loraId: meta?.loraId ?? prev?.loraId,
              manifestId: meta?.manifestId ?? prev?.manifestId,
              source: meta?.source ?? prev?.source,
            },
          },
          activeJobs: new Set(state.activeJobs).add(jobId),
        };
      });

      get()._jobInstances.set(jobId, job);

      job.onUpdate(async (data) => {
        const now = Date.now();
        const existingJob = get().jobs[jobId];
        const normalizedStatus = normalizeStatus(
          (data as any)?.status || existingJob?.status || "running",
        );

        const meta = (data && (data as any).metadata) || {};
        const loraId =
          (meta.lora_id as string | undefined) ??
          (meta.loraId as string | undefined) ??
          existingJob?.loraId;
        const manifestId =
          (meta.manifest_id as string | undefined) ??
          (meta.manifestId as string | undefined) ??
          existingJob?.manifestId;
        const source =
          (meta.source as string | undefined) ??
          (meta.remote_source as string | undefined) ??
          existingJob?.source;
        const verified =
          typeof meta.verified === "boolean"
            ? meta.verified
            : existingJob?.verified;

        let nextFiles = { ...(existingJob?.files || {}) } as NonNullable<
          JobProgress["files"]
        >;
        try {
          const filename = (meta.filename || meta.label || "").toString();
          const downloaded =
            meta.downloaded ?? meta.bytes_downloaded ?? meta.current_bytes;
          const total = meta.total ?? meta.bytes_total ?? meta.total_bytes;
          if (filename) {
            const prev = nextFiles[filename];
            let speed: number | undefined;
            if (
              typeof downloaded === "number" &&
              prev?.downloadedBytes != null &&
              prev?.lastUpdateTime
            ) {
              const dt = (now - prev.lastUpdateTime) / 1000;
              const db = downloaded - (prev.downloadedBytes || 0);
              if (dt > 0 && db > 0) speed = db / dt;
            }
            const pct =
              typeof downloaded === "number" &&
              typeof total === "number" &&
              total > 0
                ? Math.max(
                    0,
                    Math.min(100, Math.floor((downloaded / total) * 100)),
                  )
                : typeof data.progress === "number"
                  ? data.progress <= 1
                    ? data.progress * 100
                    : data.progress
                  : prev?.progress;
            nextFiles[filename] = {
              filename,
              progress: pct ?? prev?.progress,
              downloadedBytes:
                typeof downloaded === "number"
                  ? downloaded
                  : prev?.downloadedBytes,
              totalBytes: typeof total === "number" ? total : prev?.totalBytes,
              downloadSpeed: speed ?? prev?.downloadSpeed,
              lastUpdateTime: now,
              status:
                data?.status === "complete" || data?.status === "completed"
                  ? "completed"
                  : data?.status === "error"
                    ? "error"
                    : "downloading",
            };
          }
        } catch {}

        const normalizedProgress =
          typeof data?.progress === "number"
            ? data.progress <= 1
              ? Math.round(data.progress * 100)
              : Math.round(data.progress)
            : typeof existingJob?.progress === "number"
              ? Math.round(existingJob.progress)
              : undefined;
        const stepMsg = (data?.message ||
          data?.step ||
          existingJob?.currentStep) as any;
        const nextUpdates = [
          ...((existingJob?.updates as NonNullable<
            JobProgress["updates"]
          >) || []),
          {
            message: stepMsg,
            progress: normalizedProgress,
            status: normalizedStatus,
            time: now,
            metadata: data?.metadata,
          },
        ];

        get().updateJobProgress(jobId, {
          progress:
            typeof data?.progress === "number"
              ? data.progress <= 1
                ? data.progress * 100
                : data.progress
              : existingJob?.progress,
          currentStep: stepMsg as any,
          status: normalizedStatus,
          lastUpdate: now,
          updates: nextUpdates,
          files: nextFiles,
          loraId,
          manifestId,
          source,
          verified,
        });

        if (normalizedStatus === "complete") {
          const fetchWithRetry = async (attempt = 0) => {
            const delay = Math.min(500 * Math.pow(1.5, attempt), 30000);
            setTimeout(async () => {
              const success = await get().fetchJobResult(jobId);
              if (!success) fetchWithRetry(attempt + 1);
            }, delay);
          };
          fetchWithRetry();
        }
      });

      job.onStatus(async (data) => {
        get().updateJobProgress(jobId, {
          status: normalizeStatus((data as any)?.status),
          lastUpdate: Date.now(),
        });
      });

      job.onError((data) => {
        get().updateJobProgress(jobId, {
          status: "failed",
          error: data.error || data.message || "Unknown error",
          lastUpdate: Date.now(),
        });
      });

      try {
        await job.connect();
      } catch (error) {
        get().updateJobProgress(jobId, {
          status: "failed",
          error: error instanceof Error ? error.message : "Failed to connect",
          lastUpdate: Date.now(),
        });
      }

      return job;
    },

    stopTracking: async (jobId: string) => {
      const job = get()._jobInstances.get(jobId);
      if (job) {
        await job.disconnect();
        get()._jobInstances.delete(jobId);
        set((state) => {
          const newActiveJobs = new Set(state.activeJobs);
          newActiveJobs.delete(jobId);
          return { activeJobs: newActiveJobs };
        });
      }
    },

    updateJobProgress: (jobId: string, update: Partial<LoraJobProgress>) => {
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
        const response = await getEngineResult(jobId);
        if (response.success && response.data) {
          get().updateJobProgress(jobId, {
            result: response.data,
            lastUpdate: Date.now(),
          });
          return true;
        } else {
          get().updateJobProgress(jobId, {
            error: response.error || "Failed to fetch result",
            lastUpdate: Date.now(),
          });
          return false;
        }
      } catch (error) {
        get().updateJobProgress(jobId, {
          error:
            error instanceof Error ? error.message : "Failed to fetch result",
          lastUpdate: Date.now(),
        });
        return false;
      }
    },

    getJob: (jobId: string) => get().jobs[jobId],

    clearJob: (jobId: string) => {
      const { stopTracking } = get();
      stopTracking(jobId);
      set((state) => {
        const newJobs = { ...state.jobs };
        delete newJobs[jobId];
        return { jobs: newJobs };
      });
    },

    clearCompletedJobs: () => {
      const { jobs } = get();
      const completed = Object.keys(jobs).filter(
        (id) => jobs[id].status === "complete" || jobs[id].status === "failed",
      );
      completed.forEach((id) => get().clearJob(id));
    },
  })),
);

