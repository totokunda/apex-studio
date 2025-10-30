import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  downloadComponents,
  cancelComponents,
  getComponentsStatus,
  ComponentsDownloadJob,
} from './api';
import { extractPercent, extractStatus } from './progress';

export type DownloadState = 'pending' | 'downloading' | 'completed' | 'error' | 'canceled';

export type DownloadEntry = {
  jobId: string;
  path: string;
  progress: number;
  status: DownloadState;
  error?: string;
  downloadedBytes?: number;
  totalBytes?: number;
  downloadSpeed?: number; // bytes per second
  lastUpdateTime?: number; // timestamp
  files?: Record<string, FileDownloadEntry>;
};

type FileDownloadEntry = {
  filename: string;
  progress: number;
  status: DownloadState | 'processing';
  downloadedBytes?: number;
  totalBytes?: number;
  downloadSpeed?: number;
  lastUpdateTime?: number;
  error?: string;
};


interface ComponentsDownloadPayload {
  message?: string;
  metadata?: {
    label?: string;
    downloaded?: number;
    total?: number;
    filename?: string;
  };
  status?: string;
  progress?: number;
}

type Unsubscriber = () => void;

type ComponentsDownloadStore = {
  entries: Record<string, DownloadEntry>; // key by path
  jobIndex: Record<string, string[]>; // jobId -> paths
  connectJob: (jobId: string, path: string) => Promise<void>;
  startPath: (path: string, savePath?: string, jobId?: string) => Promise<string>; // returns jobId
  cancelPath: (path: string, onCanceled?: () => void) => Promise<void>;
  clearJob: (jobId: string) => void;
  setEntry: (path: string, entry: Partial<DownloadEntry>) => void;
  removeEntry: (path: string) => void;
};

const unsubRegistry: Record<string, Unsubscriber[]> = {};
const jobsRegistry: Record<string, ComponentsDownloadJob> = {};

export const useComponentsDownloadStore = create<ComponentsDownloadStore>()(persist((set, get) => ({
  entries: {},
  jobIndex: {},

  setEntry: (path, entry) => set((s) => ({
    entries: {
      ...s.entries,
      [path]: { ...(s.entries[path] || { path, jobId: '', progress: 0, status: 'pending' as const }), ...entry },
    },
  })),

  connectJob: async (jobId: string, path: string) => {
    // Create or reuse job instance
    const job = jobsRegistry[jobId] || new ComponentsDownloadJob(jobId);
    jobsRegistry[jobId] = job;

    try {
      await job.connect();
    } catch (e) {
      // Websocket connection failed, will rely on polling
    }

    // Throttle state updates per path to reduce render/persist churn
    const THROTTLE_MS = 150;
    const MIN_PROGRESS_STEP = 0.5; // percent
    let lastEmitTime = 0;
    let lastProgress: number | null = null;
    let pending: { pct: number; payload?: ComponentsDownloadPayload } | null = null;
    let timer: any | null = null;

    const emitNow = (pct: number, payload?: ComponentsDownloadPayload) => {
      const filename = (payload?.metadata?.filename || payload?.metadata?.label || '').toString();
      const downloaded = payload?.metadata?.downloaded ?? (payload as any)?.downloaded ?? (payload as any)?.bytes_downloaded;
      const total = payload?.metadata?.total ?? (payload as any)?.total ?? (payload as any)?.bytes_total;
      
      const now = Date.now();
      const currentEntry = get().entries[path];
      const prevFiles = currentEntry?.files || {};
      
      // Per-file update when filename is present
      let nextFiles = { ...prevFiles } as Record<string, FileDownloadEntry>;
      if (filename) {
        const prev = prevFiles[filename];
        let fileSpeed: number | undefined;
        if (downloaded != null && prev?.downloadedBytes != null && prev?.lastUpdateTime) {
          const bytesDelta = downloaded - (prev.downloadedBytes || 0);
          const timeDelta = (now - prev.lastUpdateTime) / 1000;
          if (timeDelta > 0 && bytesDelta > 0) fileSpeed = bytesDelta / timeDelta;
        }
        nextFiles[filename] = {
          filename,
          progress: pct,
          status: 'downloading',
          downloadedBytes: typeof downloaded === 'number' ? downloaded : prev?.downloadedBytes,
          totalBytes: typeof total === 'number' ? total : prev?.totalBytes,
          downloadSpeed: fileSpeed ?? prev?.downloadSpeed,
          lastUpdateTime: now,
          error: undefined,
        };
      }
      
      // Aggregate overall progress from files if possible
      let aggProgress = pct;
      let aggDownloaded: number | undefined = undefined;
      let aggTotal: number | undefined = undefined;
      let aggSpeed: number | undefined = undefined;
      const fileList = Object.values(nextFiles);
      if (fileList.length > 0) {
        const totalsKnown = fileList.every(f => typeof f.totalBytes === 'number' && f.totalBytes! > 0);
        if (totalsKnown) {
          const sumDownloaded = fileList.reduce((s, f) => s + (f.downloadedBytes || 0), 0);
          const sumTotal = fileList.reduce((s, f) => s + (f.totalBytes || 0), 0);
          aggDownloaded = sumDownloaded;
          aggTotal = sumTotal;
          aggProgress = sumTotal > 0 ? Math.max(0, Math.min(100, Math.floor((sumDownloaded / sumTotal) * 100))) : pct ?? 0;
        } else {
          const avg = fileList.reduce((s, f) => s + (typeof f.progress === 'number' ? f.progress : 0), 0) / fileList.length;
          aggProgress = isFinite(avg) ? avg : pct ?? 0;
        }
        // Sum speeds where available
        const sumSpeed = fileList.reduce((s, f) => s + (f.downloadSpeed || 0), 0);
        aggSpeed = sumSpeed > 0 ? sumSpeed : currentEntry?.downloadSpeed;
      }
      
      get().setEntry(path, {
        progress: typeof aggProgress === 'number' ? aggProgress : pct,
        status: 'downloading',
        downloadedBytes: aggDownloaded ?? downloaded ?? currentEntry?.downloadedBytes,
        totalBytes: aggTotal ?? total ?? currentEntry?.totalBytes,
        downloadSpeed: aggSpeed ?? currentEntry?.downloadSpeed,
        lastUpdateTime: now,
        files: nextFiles,
      });
      lastEmitTime = now;
      lastProgress = pct;
    };

    const flushPending = () => {
      if (pending) {
        const data = pending;
        pending = null;
        emitNow(data.pct, data.payload);
      }
    };

    const schedule = (pct: number, payload?: ComponentsDownloadPayload) => {
      pending = { pct, payload };
      if (timer) return;
      const delay = Math.max(0, THROTTLE_MS - (Date.now() - lastEmitTime));
      timer = setTimeout(() => {
        timer = null;
        flushPending();
      }, delay);
    };

    const applyProgress = (pct: number | null, payload?: ComponentsDownloadPayload) => {
      if (pct == null) return;
      const now = Date.now();
      const shouldEmitByTime = now - lastEmitTime >= THROTTLE_MS;
      const shouldEmitByStep = lastProgress == null || Math.abs(pct - lastProgress) >= MIN_PROGRESS_STEP;
      if (shouldEmitByTime || shouldEmitByStep) {
        emitNow(pct, payload);
      } else {
        schedule(pct, payload);
      }
    };

    const handlePayload = (payload: ComponentsDownloadPayload, finalizeOn?: DownloadState) => {
      const status = extractStatus(payload);
      const filename = (payload?.metadata?.filename || payload?.metadata?.label || '').toString();
      
      if (['canceled', 'cancelled'].includes(status)) {
        // flush any pending progress before final status
        flushPending();
        get().setEntry(path, { status: 'canceled' });
        try { get().clearJob(jobId); } catch {}
        return true;
      }
      
      if (['error', 'failed'].includes(status)) {
        flushPending();
        const currentEntry = get().entries[path];
        if (filename && currentEntry) {
          const files = { ...(currentEntry.files || {}) };
          const prev = files[filename];
          files[filename] = {
            filename,
            progress: prev?.progress ?? 0,
            status: 'error',
            downloadedBytes: prev?.downloadedBytes,
            totalBytes: prev?.totalBytes,
            downloadSpeed: 0,
            lastUpdateTime: Date.now(),
            error: payload?.message || 'Download failed',
          };
          get().setEntry(path, { files });
        } else {
          get().setEntry(path, { status: 'error' });
          try { get().clearJob(jobId); } catch {}
          return true;
        }
      }
      
      if (['completed', 'complete', 'success', 'done', 'finished'].includes(status)) {
        flushPending();
        // If a specific filename is provided, mark that file complete
        if (filename) {
          const currentEntry = get().entries[path];
          const prev = currentEntry?.files?.[filename];
          const total = prev?.totalBytes ?? payload?.metadata?.total;
          const downloaded = prev?.downloadedBytes ?? payload?.metadata?.downloaded ?? total;
          const files = { ...(currentEntry?.files || {}) };
          files[filename] = {
            filename,
            progress: 100,
            status: 'completed',
            downloadedBytes: typeof downloaded === 'number' ? downloaded : prev?.downloadedBytes,
            totalBytes: typeof total === 'number' ? total : prev?.totalBytes,
            downloadSpeed: 0,
            lastUpdateTime: Date.now(),
            error: undefined,
          };
          get().setEntry(path, { files });
        } else {
          // No filename => this is likely the final job-level completion signal
          get().setEntry(path, { status: 'completed', progress: 100 });
          try { get().clearJob(jobId); } catch {}
          return true;
        }
        return false;
      }
      
      if (finalizeOn && status === finalizeOn) {
        // Only used in special flows; we do not mark overall completed here
        return true;
      }
      
      const pct = extractPercent(payload);
      applyProgress(pct, payload);
      return false;
    };

    const unsubs: Unsubscriber[] = [];

    // Register handlers via job (cleanup handled in job.disconnect)
    job.onUpdate((data) => {
      handlePayload(data);
    });
    job.onStatus((data) => {
      handlePayload(data);
    });
    job.onError((_err) => {
      handlePayload({ status: 'error' });
    });

    // Ensure timers are cleaned on job clear
    unsubs.push(() => { try { if (timer) { clearTimeout(timer); timer = null; } } catch {} });
    unsubRegistry[jobId] = unsubs;

    set((s) => ({ jobIndex: { ...s.jobIndex, [jobId]: Array.from(new Set([...(s.jobIndex[jobId] || []), path])) } }));
    

  },

  startPath: async (path: string, savePath?: string, jobId?: string) => {
    const id = jobId || (globalThis.crypto && 'randomUUID' in globalThis.crypto ? globalThis.crypto.randomUUID() : `job_${Date.now()}_${Math.random().toString(36).slice(2,8)}`);
    set((s) => ({ entries: { ...s.entries, [path]: { jobId: id, path, progress: 0, status: 'downloading' } } }));
    await get().connectJob(id, path);
    try {
      await downloadComponents([path], savePath, id);
    } catch (e) {
      set((s) => ({ entries: { ...s.entries, [path]: { ...(s.entries[path] || { jobId: id, path, progress: 0, status: 'error' }), status: 'error', error: (e as any)?.message || 'Failed to start download' } } }));
      try { get().clearJob(id); } catch {}
    }
    return id;
  },

  cancelPath: async (path: string, onCanceled?: () => void) => {
    const entry = get().entries[path];
    if (!entry) return;
    try { await cancelComponents(entry.jobId); } catch {}
    set((s) => ({ entries: { ...s.entries, [path]: { ...(s.entries[path] as DownloadEntry), status: 'canceled' } } }));
    try { get().clearJob(entry.jobId); } catch {}
    
    // Trigger callback after cleanup (e.g., to refetch manifest)
    if (onCanceled) {
      setTimeout(() => {
        try { onCanceled(); } catch {}
      }, 500);
    }
  },

  clearJob: (jobId: string) => {
    // Disconnect job websocket and listeners
    try { jobsRegistry[jobId]?.disconnect(); } catch {}
    const arr = unsubRegistry[jobId] || [];
    arr.forEach((fn) => { try { fn(); } catch {} });
    delete unsubRegistry[jobId];
    delete jobsRegistry[jobId];
  },

  removeEntry: (path: string) => {
    set((s) => {
      const next = { ...s.entries };
      delete next[path];
      return { entries: next };
    });
  },
}), {
  name: 'components-download-store',
  // Persist only lightweight fields to avoid heavy, frequent writes
  partialize: (state) => {
    const minimalEntries: Record<string, Pick<DownloadEntry, 'jobId' | 'path' | 'progress' | 'status' | 'error'>> = {};
    for (const [k, v] of Object.entries(state.entries)) {
      minimalEntries[k] = { jobId: v.jobId, path: v.path, progress: v.progress, status: v.status, error: v.error };
    }
    return { entries: minimalEntries } as Partial<typeof state>;
  },
  onRehydrateStorage: () => async (state) => {
    // After state is rehydrated from storage, check and reconnect any active jobs
    if (!state?.entries) return;
    
    const entries = state.entries;
    // check if it exists yet useComponentsDownloadStore.getState() is defined
    
    
    // Check each entry that was downloading
    for (const [path, entry] of Object.entries(entries)) {
      if (!entry || !entry.jobId) continue;
      if (entry.status !== 'downloading' && entry.status !== 'pending') continue;
      
      try {
        // Check if the job is still active on the backend
        const statusRes = await getComponentsStatus(entry.jobId);
        
        if (statusRes.success && statusRes.data) {
          const { status, latest } = statusRes.data;
          
          // If job is still running, reconnect
          if (status === 'running' || status === 'queued' || status === 'processing') {
            // Update with latest progress if available
            if (latest) {
              const pct = extractPercent(latest);
              if (pct != null) {
                state.setEntry(path, { progress: pct });
              }
            }
            setTimeout(() => {
              try {
                state.connectJob(entry.jobId, path);
              } catch {}
            }, 100);
          } else if (status === 'complete' || status === 'completed') {
            // Job completed while we were away
            state.setEntry(path, { status: 'completed', progress: 100 });
            setTimeout(() => {
              try { state.removeEntry(path); } catch {}
            }, 2000);
          } else if (status === 'error' || status === 'failed') {
            // Job failed
            state.setEntry(path, { status: 'error' });
          } else if (status === 'canceled' || status === 'cancelled') {
            // Job was canceled
            state.setEntry(path, { status: 'canceled' });
          } else {
            // Unknown status, clean up
            state.removeEntry(path);
          }
        } else {
          // Job not found on backend, clean up stale entry
          state.removeEntry(path);
        }
      } catch (e) {
        // Error checking status, clean up stale entry
        state.removeEntry(path);
      }
    }
  }
}));
