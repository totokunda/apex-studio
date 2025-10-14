import { create } from "zustand";
import { PreprocessorJob, getPreprocessorResult, JobResult } from "./api";

export interface JobProgress {
  jobId: string;
  status: 'pending' | 'running' | 'complete' | 'failed';
  progress?: number;
  currentStep?: string;
  error?: string;
  result?: JobResult;
  lastUpdate?: number;
}

interface PreprocessorJobStore {
  jobs: Record<string, JobProgress>;
  activeJobs: Set<string>;
  
  // Job management
  startTracking: (jobId: string) => Promise<PreprocessorJob>;
  stopTracking: (jobId: string) => Promise<void>;
  updateJobProgress: (jobId: string, update: Partial<JobProgress>) => void;
  fetchJobResult: (jobId: string) => Promise<boolean>;
  getJob: (jobId: string) => JobProgress | undefined;
  clearJob: (jobId: string) => void;
  clearCompletedJobs: () => void;
  
  // Internal job instances
  _jobInstances: Map<string, PreprocessorJob>;
}

export const usePreprocessorJobStore = create<PreprocessorJobStore>((set, get) => ({
  jobs: {},
  activeJobs: new Set(),
  _jobInstances: new Map(),

  startTracking: async (jobId: string) => {
    const existing = get()._jobInstances.get(jobId);
    if (existing) {
      return existing;
    }

    const job = new PreprocessorJob(jobId);
    
    // Initialize job state
    set((state) => ({
      jobs: {
        ...state.jobs,
        [jobId]: {
          jobId,
          status: 'pending',
          lastUpdate: Date.now(),
        },
      },
      activeJobs: new Set(state.activeJobs).add(jobId),
    }));

    // Store instance
    get()._jobInstances.set(jobId, job);

    // Setup listeners
    job.onUpdate(async (data) => {
      
      get().updateJobProgress(jobId, {
        progress: data.progress,
        currentStep: data.step,
        status: data.status || 'processing',
        lastUpdate: Date.now(),
      });

      if (data.status === 'complete') {
        // Repeatedly fetch result until retrieved
        const fetchWithRetry = async (attempt = 0) => {
          const delay = Math.min(500 * Math.pow(1.5, attempt), 5000);
          setTimeout(async () => {
            const success = await get().fetchJobResult(jobId);
            if (!success) {
              fetchWithRetry(attempt + 1);
            }
          }, delay);
        };
        fetchWithRetry();
      }
    });


    job.onStatus(async (data) => {
      get().updateJobProgress(jobId, {
        status: data.status,
        lastUpdate: Date.now(),
      });
      // Fetch result when job completes
      
    });

    job.onError((data) => {
      get().updateJobProgress(jobId, {
        status: 'failed',
        error: data.error || data.message || 'Unknown error',
        lastUpdate: Date.now(),
      });
    });

    // Connect to WebSocket
    try {
      await job.connect();
    } catch (error) {
      get().updateJobProgress(jobId, {
        status: 'failed',
        error: error instanceof Error ? error.message : 'Failed to connect',
        lastUpdate: Date.now(),
      });
    }

    return job;
  },

  stopTracking: async (jobId: string) => {
    const job = get()._jobInstances.get(jobId);
    if (job) {
      await job.disconnect();
      await job.cancel();
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
      const response = await getPreprocessorResult(jobId);
      if (response.success && response.data) {
        get().updateJobProgress(jobId, {
          result: response.data,
          lastUpdate: Date.now(),
        });
        return true;
      } else {
        get().updateJobProgress(jobId, {
          error: response.error || 'Failed to fetch result',
          lastUpdate: Date.now(),
        });
        return false;
      }
    } catch (error) {
      get().updateJobProgress(jobId, {
        error: error instanceof Error ? error.message : 'Failed to fetch result',
        lastUpdate: Date.now(),
      });
      return false;
    }
  },

  getJob: (jobId: string) => {
    return get().jobs[jobId];
  },

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
    const completedJobIds = Object.keys(jobs).filter(
      (jobId) => jobs[jobId].status === 'complete' || jobs[jobId].status === 'failed'
    );
    
    completedJobIds.forEach((jobId) => {
      get().clearJob(jobId);
    });
  },
}));

