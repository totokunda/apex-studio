import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { UnifiedDownloadWsUpdate } from "./api";

interface DownloadJobIdStore {
  sourceToJobId: Record<string, string>;
  getSourceToJobId: (source: string) => string | undefined;
  addSourceToJobId: (source: string | string[], jobId: string) => void;
  removeSourceToJobId: (source: string) => void;
  removeSourceByJobId: (jobId: string) => void;
  jobUpdates: Record<string, UnifiedDownloadWsUpdate[]>;
  addJobUpdate: (jobId: string, update: UnifiedDownloadWsUpdate) => void;
  removeJobUpdates: (jobId: string) => void;
  getJobUpdates: (jobId: string | undefined) => UnifiedDownloadWsUpdate[] | undefined;
  jobIdToParts: Record<string, string[]>;
  addJobIdToParts: (jobId: string, parts: string[]) => void;
  getJobIdToParts: (jobId: string) => string[] | undefined;
  removeJobIdToParts: (jobId: string) => void;
  jobIdToManifestId: Record<string, string>;
  addJobIdToManifestId: (jobId: string, manifestId: string) => void;
  getJobIdToManifestId: (jobId: string) => string | undefined;
  removeJobIdToManifestId: (jobId: string) => void;
}

const MAX_UPDATES_PER_JOB = 300;

export const useDownloadJobIdStore = create<DownloadJobIdStore>()(
  persist(
    (set, get) => ({
      sourceToJobId: {},
      getSourceToJobId: (source: string) => get().sourceToJobId[source],

      jobUpdates: {},

      addSourceToJobId: (source: string | string[], jobId: string) =>
        set((state) => ({
          sourceToJobId: {
            ...state.sourceToJobId,
            [typeof source === "string" ? source : source.join(",")]: jobId,
          },
        })),

      removeSourceToJobId: (source: string | string[]) =>
        set((state) => {
          const key = typeof source === "string" ? source : source.join(",");
          if (!(key in state.sourceToJobId)) return state;
          const next = { ...state.sourceToJobId };
          delete next[key];
          return { sourceToJobId: next };
        }),

      removeSourceByJobId: (jobId: string) => {
        const sourceToJobId = get().sourceToJobId;
        const keysToDelete = Object.keys(sourceToJobId).filter(
          (source) => sourceToJobId[source] === jobId,
        );
        if (!keysToDelete.length) return;

        set((state) => {
          const next = { ...state.sourceToJobId };
          for (const k of keysToDelete) delete next[k];
          return { sourceToJobId: next };
        });
      },

      addJobUpdate: (jobId: string, update: UnifiedDownloadWsUpdate) =>
        set((state) => {
          const prev = state.jobUpdates[jobId] ?? [];
          const next =
            prev.length >= MAX_UPDATES_PER_JOB
              ? [...prev.slice(-(MAX_UPDATES_PER_JOB - 1)), update]
              : [...prev, update];
          return { jobUpdates: { ...state.jobUpdates, [jobId]: next } };
        }),

      removeJobUpdates: (jobId: string) => {
        const jobUpdates = get().jobUpdates;
        if (!jobUpdates[jobId]) return;
        const updatedJobUpdates = { ...jobUpdates };
        delete updatedJobUpdates[jobId];
        set({ jobUpdates: updatedJobUpdates });
      },

      getJobUpdates: (jobId: string | undefined) =>
        jobId ? get().jobUpdates[jobId] : undefined,

      jobIdToParts: {},
      addJobIdToParts: (jobId: string, parts: string[]) =>
        set((state) => ({ jobIdToParts: { ...state.jobIdToParts, [jobId]: parts } })),
      getJobIdToParts: (jobId: string) => get().jobIdToParts[jobId],
      removeJobIdToParts: (jobId: string) =>
        set((state) => {
          if (!(jobId in state.jobIdToParts)) return state;
          const next = { ...state.jobIdToParts };
          delete next[jobId];
          return { jobIdToParts: next };
        }),

      jobIdToManifestId: {},
      addJobIdToManifestId: (jobId: string, manifestId: string) =>
        set((state) => ({
          jobIdToManifestId: { ...state.jobIdToManifestId, [jobId]: manifestId },
        })),
      getJobIdToManifestId: (jobId: string) => get().jobIdToManifestId[jobId],
      removeJobIdToManifestId: (jobId: string) =>
        set((state) => {
          if (!(jobId in state.jobIdToManifestId)) return state;
          const next = { ...state.jobIdToManifestId };
          delete next[jobId];
          return { jobIdToManifestId: next };
        }),
    }),
    {
      name: "job-id-store",
      storage: createJSONStorage(() => localStorage),

      // Do not persist streaming WS updates; keep them in-memory only.
      partialize: (state) => ({
        sourceToJobId: state.sourceToJobId,
        jobIdToParts: state.jobIdToParts,
        jobIdToManifestId: state.jobIdToManifestId,
      }),

      // Drop any previously persisted jobUpdates from older versions.
      version: 2,
      migrate: (persistedState: any) => {
        if (persistedState && typeof persistedState === "object") {
          delete persistedState.jobUpdates;
        }
        return persistedState;
      },
    },
  ),
);
