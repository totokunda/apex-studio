import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { useManifestStore } from '@/lib/manifest/store';
import type { ManifestComponent, ManifestDocument, ManifestSchedulerOption } from '@/lib/manifest/api';
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
  manifestIds?: string[];
  componentType?: string;
  label?: string;
  kind?: DownloadTargetKind;
  awaitingManifest?: boolean;
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

export type DownloadTargetKind = 'model' | 'config' | 'asset' | 'unknown';

export type StartDownloadOptions = {
  savePath?: string;
  jobId?: string;
  manifestId?: string;
  manifestIds?: string[];
  componentType?: string;
  label?: string;
  kind?: DownloadTargetKind;
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
  pendingDeletions: Record<string, Record<string, true>>;
  connectJob: (jobId: string, path: string) => Promise<void>;
  startPath: (path: string, options?: StartDownloadOptions) => Promise<string>; // returns jobId
  cancelPath: (path: string, onCanceled?: () => void) => Promise<void>;
  clearJob: (jobId: string) => void;
  setEntry: (path: string, entry: Partial<DownloadEntry>) => void;
  removeEntry: (path: string) => void;
  markPendingDeletion: (manifestId: string, path: string) => void;
  clearPendingDeletion: (manifestId: string, path: string) => void;
};

const unsubRegistry: Record<string, Unsubscriber[]> = {};
const jobsRegistry: Record<string, ComponentsDownloadJob> = {};

export const useComponentsDownloadStore = create<ComponentsDownloadStore>()(persist((set, get) => ({
  entries: {},
  jobIndex: {},
  pendingDeletions: {},

  setEntry: (path, entry) => set((s) => ({
    entries: {
      ...s.entries,
      [path]: {
        ...(s.entries[path] || {
          path,
          jobId: '',
          progress: 0,
          status: 'pending' as const,
          manifestIds: [],
          kind: 'unknown',
        }),
        ...entry,
      },
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
      const entrySnapshot = get().entries[path];
      const filename = (payload?.metadata?.filename || payload?.metadata?.label || '').toString();
      
      if (['canceled', 'cancelled'].includes(status)) {
        // flush any pending progress before final status
        flushPending();
        get().setEntry(path, { status: 'canceled', awaitingManifest: false });
        if (entrySnapshot?.manifestIds?.length) {
          requestManifestSync(entrySnapshot.manifestIds, { includeList: true });
        }
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
          get().setEntry(path, { status: 'error', awaitingManifest: false });
          if (entrySnapshot?.manifestIds?.length) {
            requestManifestSync(entrySnapshot.manifestIds, { includeList: false });
          }
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
          const isModelKind = (entrySnapshot?.kind || 'unknown') === 'model';
          get().setEntry(path, { status: 'completed', progress: 100, awaitingManifest: isModelKind });
          if (entrySnapshot?.manifestIds?.length) {
            requestManifestSync(entrySnapshot.manifestIds, { includeList: true });
          }
          if (!isModelKind) {
            setTimeout(() => {
              try {
                const entryNow = get().entries[path];
                if (entryNow && entryNow.jobId === (entrySnapshot?.jobId || jobId)) {
                  get().removeEntry(path);
                }
              } catch {}
            }, 1000);
          }
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

  startPath: async (path: string, options?: StartDownloadOptions) => {
    const {
      savePath,
      jobId,
      manifestId,
      manifestIds,
      componentType,
      label,
      kind,
    } = options || {};
    const id = jobId || (globalThis.crypto && 'randomUUID' in globalThis.crypto ? globalThis.crypto.randomUUID() : `job_${Date.now()}_${Math.random().toString(36).slice(2,8)}`);
    const manifestIdList = (manifestIds && manifestIds.length ? manifestIds : (manifestId ? [manifestId] : [])).filter((v): v is string => !!v);
    const baseEntry: DownloadEntry = {
      jobId: id,
      path,
      progress: 0,
      status: 'downloading',
      manifestIds: manifestIdList,
      componentType,
      label,
      kind: kind || 'unknown',
      awaitingManifest: false,
    };
    manifestIdList.forEach((id) => {
      try { get().clearPendingDeletion(id, path); } catch {}
    });
    set((s) => ({ entries: { ...s.entries, [path]: { ...(s.entries[path] || baseEntry), ...baseEntry } } }));
    await get().connectJob(id, path);
    try {
      await downloadComponents([path], savePath, id);
    } catch (e) {
      set((s) => ({
        entries: {
          ...s.entries,
          [path]: {
            ...(s.entries[path] || baseEntry),
            status: 'error',
            error: (e as any)?.message || 'Failed to start download',
          },
        },
      }));
      try { get().clearJob(id); } catch {}
      if (baseEntry.manifestIds?.length) {
        requestManifestSync(baseEntry.manifestIds, { includeList: true });
      }
    }
    return id;
  },

  cancelPath: async (path: string, onCanceled?: () => void) => {
    const entry = get().entries[path];
    if (!entry) return;
    try { await cancelComponents(entry.jobId); } catch {}
    set((s) => ({ entries: { ...s.entries, [path]: { ...(s.entries[path] as DownloadEntry), status: 'canceled', awaitingManifest: false } } }));
    try { get().clearJob(entry.jobId); } catch {}
    if (entry.manifestIds?.length) {
      requestManifestSync(entry.manifestIds, { includeList: true });
    }
    
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
  markPendingDeletion: (manifestId: string, path: string) => {
    if (!manifestId || !path) return;
    set((s) => {
      const manifestMap = s.pendingDeletions[manifestId] || {};
      if (manifestMap[path]) return s;
      return {
        pendingDeletions: {
          ...s.pendingDeletions,
          [manifestId]: { ...manifestMap, [path]: true },
        },
      };
    });
  },
  clearPendingDeletion: (manifestId: string, path: string) => {
    if (!manifestId || !path) return;
    set((s) => {
      const manifestMap = s.pendingDeletions[manifestId];
      if (!manifestMap || !manifestMap[path]) return s;
      const nextManifestMap = { ...manifestMap };
      delete nextManifestMap[path];
      const nextPending = { ...s.pendingDeletions };
      if (Object.keys(nextManifestMap).length > 0) nextPending[manifestId] = nextManifestMap;
      else delete nextPending[manifestId];
      return { pendingDeletions: nextPending };
    });
  },
}), {
  name: 'components-download-store',
  // Persist only lightweight fields to avoid heavy, frequent writes
  partialize: (state) => {
    const minimalEntries: Record<string, Pick<DownloadEntry, 'jobId' | 'path' | 'progress' | 'status' | 'error' | 'manifestIds' | 'componentType' | 'label' | 'kind' | 'awaitingManifest'>> = {};
    for (const [k, v] of Object.entries(state.entries)) {
      minimalEntries[k] = {
        jobId: v.jobId,
        path: v.path,
        progress: v.progress,
        status: v.status,
        error: v.error,
        manifestIds: v.manifestIds,
        componentType: v.componentType,
        label: v.label,
        kind: v.kind,
        awaitingManifest: v.awaitingManifest,
      };
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
      if (entry.manifestIds?.length && (entry.awaitingManifest || entry.status === 'completed')) {
        requestManifestSync(entry.manifestIds, { includeList: true });
      }
      if (entry.status !== 'downloading' && entry.status !== 'pending') continue;
      const isModelKind = (entry.kind || 'unknown') === 'model';
      
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
            state.setEntry(path, { status: 'completed', progress: 100, awaitingManifest: isModelKind });
            if (entry.manifestIds?.length) {
              requestManifestSync(entry.manifestIds, { includeList: true });
            }
            if (!isModelKind) {
              setTimeout(() => {
                try { state.removeEntry(path); } catch {}
              }, 2000);
            }
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

type ManifestSyncOptions = {
  includeList?: boolean;
  immediate?: boolean;
};

const manifestSyncQueue = new Set<string>();
let manifestListRefreshRequested = false;
let manifestSyncTimer: ReturnType<typeof setTimeout> | null = null;
let manifestSyncInFlight: Promise<void> | null = null;

export function requestManifestSync(manifestIds?: string | string[], options?: ManifestSyncOptions) {
  const ids = (Array.isArray(manifestIds) ? manifestIds : manifestIds ? [manifestIds] : []).filter((id): id is string => !!id);
  ids.forEach((id) => manifestSyncQueue.add(id));
  if (options?.includeList) {
    manifestListRefreshRequested = true;
  }
  const trigger = () => {
    if (manifestSyncTimer) {
      clearTimeout(manifestSyncTimer);
      manifestSyncTimer = null;
    }
    void flushManifestSyncQueue();
  };
  if (options?.immediate) {
    trigger();
    return;
  }
  if (!manifestSyncTimer) {
    manifestSyncTimer = setTimeout(() => {
      manifestSyncTimer = null;
      void flushManifestSyncQueue();
    }, 600);
  }
}

async function flushManifestSyncQueue() {
  if (manifestSyncInFlight) {
    await manifestSyncInFlight;
  }
  if (manifestSyncQueue.size === 0 && !manifestListRefreshRequested) return;

  manifestSyncInFlight = (async () => {
    const { loadManifest, loadManifests } = useManifestStore.getState();
    while (manifestSyncQueue.size > 0 || manifestListRefreshRequested) {
      const ids = Array.from(manifestSyncQueue);
      const refreshList = manifestListRefreshRequested;
      manifestSyncQueue.clear();
      manifestListRefreshRequested = false;
      if (ids.length > 0) {
        await Promise.all(ids.map((id) => loadManifest(id, true).catch(() => {})));
      }
      if (refreshList) {
        try { await loadManifests(true); } catch {}
      }
      ids.forEach((id) => {
        try {
          reconcileManifestDownloads(id);
        } catch {
          // swallow reconciliation errors to keep sync loop alive
        }
      });
    }
  })();

  try {
    await manifestSyncInFlight;
  } finally {
    manifestSyncInFlight = null;
  }
}

type NormalizedPath = { path: string; downloaded?: boolean };

function normalizeModelPaths(component: ManifestComponent | Record<string, any>): NormalizedPath[] {
  const raw = Array.isArray(component?.model_path) ? component.model_path : component?.model_path ? [component.model_path] : [];
  const entries: NormalizedPath[] = [];
  for (const item of raw) {
    if (typeof item === 'string') {
      entries.push({ path: item });
    } else if (item && typeof item === 'object' && item.path) {
      entries.push({ path: item.path, downloaded: typeof item.is_downloaded === 'boolean' ? !!item.is_downloaded : undefined });
    }
  }
  return entries;
}

function normalizeSchedulerConfigs(component: ManifestComponent | Record<string, any>): NormalizedPath[] {
  const opts = Array.isArray(component?.scheduler_options) ? component.scheduler_options : [];
  const entries: NormalizedPath[] = [];
  for (const opt of opts as Array<ManifestSchedulerOption & { is_downloaded?: boolean }>) {
    if (!opt || !opt.config_path) continue;
    entries.push({ path: String(opt.config_path), downloaded: typeof opt.is_downloaded === 'boolean' ? !!opt.is_downloaded : undefined });
  }
  return entries;
}

function buildManifestPathStatus(manifest: ManifestDocument): Map<string, boolean> {
  const map = new Map<string, boolean>();
  const components = Array.isArray(manifest?.spec?.components) ? manifest.spec.components : [];
  for (const component of components as Array<ManifestComponent & { extra_model_paths?: string[]; converted_model_path?: string; gguf_files?: { path?: string }[] }>) {
    const componentDownloaded = !!(component as any)?.is_downloaded;
    const mark = (path: string | undefined, overridden?: boolean) => {
      if (!path) return;
      const flag = typeof overridden === 'boolean' ? overridden : componentDownloaded;
      map.set(String(path), flag);
    };
    normalizeModelPaths(component).forEach((item) => mark(item.path, item.downloaded));
    if (component?.config_path) mark(String(component.config_path));
    normalizeSchedulerConfigs(component).forEach((item) => mark(item.path, item.downloaded));
    if (Array.isArray((component as any)?.extra_model_paths)) {
      for (const extra of (component as any).extra_model_paths) {
        if (typeof extra === 'string') mark(extra);
      }
    }
    if ((component as any)?.converted_model_path) {
      mark(String((component as any).converted_model_path));
    }
    if (Array.isArray((component as any)?.gguf_files)) {
      for (const file of (component as any).gguf_files) {
        if (file?.path) mark(String(file.path));
      }
    }
  }
  return map;
}

function reconcileManifestDownloads(manifestId: string) {
  const manifest = useManifestStore.getState().manifestById[manifestId];
  if (!manifest) return;
  const downloadedMap = buildManifestPathStatus(manifest);
  if (downloadedMap.size === 0) return;
  const store = useComponentsDownloadStore.getState();
  const latestEntries = store.entries;
  for (const [path, entry] of Object.entries(latestEntries)) {
    if (!entry) continue;
    if (!(entry.manifestIds || []).includes(manifestId)) continue;
    const resolved = downloadedMap.get(path);
    if (resolved === true) {
      store.setEntry(path, { progress: 100, status: 'completed', awaitingManifest: false, lastUpdateTime: Date.now() });
      setTimeout(() => {
        const current = useComponentsDownloadStore.getState().entries[path];
        if (!current) return;
        if ((current.manifestIds || []).includes(manifestId)) {
          try {
            useComponentsDownloadStore.getState().removeEntry(path);
          } catch {}
        }
      }, 900);
    }
  }
  const pending = store.pendingDeletions[manifestId];
  if (pending) {
    for (const pendingPath of Object.keys(pending)) {
      const status = downloadedMap.get(pendingPath);
      if (status === false || status == null) {
        try { store.clearPendingDeletion(manifestId, pendingPath); } catch {}
      }
    }
  }
}
