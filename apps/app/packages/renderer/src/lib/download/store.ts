import { create } from "zustand";
import {
  startUnifiedDownload,
  resolveUnifiedDownload,
  resolveUnifiedDownloadBatch,
  getUnifiedDownloadStatus,
  cancelUnifiedDownload,
  deleteDownload,
  onUnifiedDownloadUpdate,
  onUnifiedDownloadStatus,
  onUnifiedDownloadError,
  connectUnifiedDownloadWebSocket,
  disconnectUnifiedDownloadWebSocket,
  type UnifiedJobStatus,
  type UnifiedDownloadWsUpdate,
  type WsConnectionStatus,
} from "./api";

export interface DownloadStore {
  downloadingPaths: Set<string>;
  jobIdToPath: Record<string, string | string[]>;
  wsFilesByPath: Record<
    string,
    Record<
      string,
      {
        filename: string;
        downloadedBytes: number;
        totalBytes: number;
        status?: string;
        progress?: number | null;
        message?: string;
        bucket?: "component" | "lora" | "preprocessor";
        label?: string;
        downloadSpeed?: number;
        lastUpdateTs?: number;
        lastDownloadedBytes?: number;
      }
    >
  >;
  // Overloads to accommodate single path and multiple paths
  subscribeToJob: {
    (
      jobId: string,
      path: string,
      onComplete?: (path: string) => void,
      onError?: (error: unknown, path: string) => void,
    ): Promise<() => void>;
    (
      jobId: string,
      paths: string[],
      onComplete?: (paths: string[]) => void,
      onError?: (error: unknown, paths: string[]) => void,
    ): Promise<() => void>;
  };
  startAndTrackDownload: (
    request: {
      item_type: "component" | "lora" | "preprocessor";
      source: string | string[];
      save_path?: string;
      manifest_id?: string;
      lora_name?: string;
    },
    onComplete?: (path: string) => void,
    onError?: (error: unknown, source: string) => void,
  ) => Promise<string[]>;
  startDownload: (request: {
    item_type: "component" | "lora" | "preprocessor";
    source: string | string[];
    save_path?: string;
    job_id?: string;
    manifest_id?: string;
    lora_name?: string;
  }) => Promise<string>;
  resolveDownload: (request: {
    item_type: "component" | "lora" | "preprocessor";
    source: string | string[];
    save_path?: string;
  }) => Promise<
    | {
        job_id: string;
        exists: boolean;
        running: boolean;
        downloaded: boolean;
        bucket: string;
        save_dir: string;
        source: string | string[];
      }
    | undefined
  >;
  resolveDownloadBatch: (request: {
    item_type: "component" | "lora" | "preprocessor";
    sources: Array<string | string[]>;
    save_path?: string;
  }) => Promise<
    | {
        results: Array<{
          job_id: string;
          exists: boolean;
          running: boolean;
          downloaded: boolean;
          bucket: string;
          save_dir: string;
          source: string | string[];
        }>;
      }
    | undefined
  >;
  getDownloadStatus: (jobId: string) => Promise<UnifiedJobStatus | undefined>;
  cancelDownload: (
    jobId: string,
  ) => Promise<
    { job_id: string; status: string; message?: string } | undefined
  >;
  deleteDownload: (request: {
    path: string;
    item_type?: "component" | "lora" | "preprocessor";
    source?: string | string[];
    save_path?: string;
  }) => Promise<
    | {
        path: string;
        status: string;
        removed_mapping?: boolean;
        unmarked?: boolean;
      }
    | undefined
  >;
  connectDownloadWS: (jobId: string) => Promise<boolean>;
  disconnectDownloadWS: (jobId: string) => Promise<boolean>;
  onDownloadUpdate: (
    jobId: string,
    cb: (data: UnifiedDownloadWsUpdate) => void,
  ) => () => void;
  onDownloadStatus: (
    jobId: string,
    cb: (data: WsConnectionStatus) => void,
  ) => () => void;
  onDownloadError: (
    jobId: string,
    cb: (data: { error: string; raw?: any }) => void,
  ) => () => void;
}

export const useDownloadStore = create<DownloadStore>()((set, get) => ({
  downloadingPaths: new Set<string>(),
  jobIdToPath: {},
  wsFilesByPath: {},
  subscribeToJob: async (
    jobId: string,
    pathOrPaths: string | string[],
    onComplete?: (arg: any) => void,
    onError?: (error: unknown, arg: any) => void,
  ) => {
    const pathsArr: string[] = Array.isArray(pathOrPaths)
      ? pathOrPaths
      : [pathOrPaths];
    const primaryPath: string | undefined = pathsArr[0];
    set((state) => ({
      jobIdToPath: { ...state.jobIdToPath, [jobId]: pathOrPaths },
    }));
    try {
      await connectUnifiedDownloadWebSocket(jobId);
    } catch (error) {
      if (onError) {
        try {
          onError(error, pathOrPaths);
        } catch {
          // ignore secondary errors from onError
        }
      }
    }
    const off = onUnifiedDownloadUpdate(
      jobId,
      (data: UnifiedDownloadWsUpdate) => {
        try {
          const meta = (data && (data as any).metadata) || {};
          const filename =
            typeof meta.filename === "string" ? meta.filename : undefined;
          const downloaded =
            typeof meta.downloaded === "number"
              ? (meta.downloaded as number)
              : undefined;
          const total =
            typeof meta.total === "number" ? (meta.total as number) : undefined;
          const label =
            typeof meta.label === "string" ? (meta.label as string) : undefined;
          const bucket =
            meta.bucket === "component" ||
            meta.bucket === "lora" ||
            meta.bucket === "preprocessor"
              ? meta.bucket
              : undefined;
          const progress =
            typeof (data as any)?.progress === "number"
              ? (data as any).progress
              : null;
          const message =
            typeof (data as any)?.message === "string"
              ? (data as any).message
              : undefined;
          const status = (data as any)?.status as string | undefined;
          const isTerminal =
            !!status &&
            (status === "complete" ||
              status === "completed" ||
              status === "error" ||
              status === "canceled");
          // If we received a terminal status without byte info, still clear local downloading flag

          if (isTerminal) {
            if ((status === "error" || status === "canceled") && onError) {
              try {
                onError({ status, message }, pathOrPaths);
              } catch {
                // ignore secondary errors from onError
              }
            }
            if (onComplete) onComplete(pathOrPaths);
            setTimeout(() => {
              set((state) => {
                const nextDownloading = new Set(state.downloadingPaths);
                for (const p of pathsArr) nextDownloading.delete(p);
                const nextWsFilesByPath = { ...state.wsFilesByPath };
                for (const p of pathsArr) {
                  if (p && nextWsFilesByPath[p]) {
                    delete nextWsFilesByPath[p];
                  }
                }
                const nextJobIdToPath = { ...state.jobIdToPath };
                if (jobId && nextJobIdToPath[jobId]) {
                  delete nextJobIdToPath[jobId];
                }
                return {
                  downloadingPaths: nextDownloading,
                  wsFilesByPath: nextWsFilesByPath,
                  jobIdToPath: nextJobIdToPath,
                };
              });
            }, 750);
          }
          if (
            !isTerminal &&
            primaryPath &&
            filename &&
            (downloaded != null || total != null)
          ) {
            set((state) => {
              const currentForPath = state.wsFilesByPath[primaryPath] || {};
              const prev = currentForPath[filename];
              const now = Date.now();

              const prevDownloaded =
                typeof prev?.downloadedBytes === "number"
                  ? prev.downloadedBytes
                  : (downloaded ?? 0);
              const prevTs =
                typeof prev?.lastUpdateTs === "number"
                  ? prev.lastUpdateTs
                  : now;
              const nextDownloaded = downloaded ?? prevDownloaded;
              const dtSeconds = (now - prevTs) / 1000;

              let downloadSpeed: number | undefined = prev?.downloadSpeed;
              // Only update speed when we have a meaningful time delta and bytes actually increased
              if (
                downloaded != null &&
                dtSeconds > 0.25 &&
                downloaded > prevDownloaded
              ) {
                const instantSpeed = (downloaded - prevDownloaded) / dtSeconds;
                const alpha = 0.3; // smoothing factor for EMA
                downloadSpeed =
                  downloadSpeed != null
                    ? downloadSpeed * (1 - alpha) + instantSpeed * alpha
                    : instantSpeed;
              }

              return {
                wsFilesByPath: {
                  ...state.wsFilesByPath,
                  [primaryPath]: {
                    ...currentForPath,
                    [filename]: {
                      filename,
                      downloadedBytes: nextDownloaded,
                      totalBytes:
                        total ?? currentForPath[filename]?.totalBytes ?? 0,
                      status,
                      progress,
                      message,
                      bucket,
                      label,
                      downloadSpeed,
                      lastUpdateTs: now,
                      lastDownloadedBytes: nextDownloaded,
                    },
                  },
                },
                downloadingPaths: (() => {
                  const next = new Set(state.downloadingPaths);
                  if (primaryPath) next.delete(primaryPath);
                  return next;
                })(),
              };
            });
          }
          if (isTerminal) {
            try {
              off();
            } catch {}
            disconnectUnifiedDownloadWebSocket(jobId).catch(() => {});
          }
        } catch (error) {
          if (onError) {
            try {
              onError(error, pathOrPaths);
            } catch {
              // ignore secondary errors from onError
            }
          }
        }
      },
    );
    return off;
  },
  startAndTrackDownload: async (
    request: {
      item_type: "component" | "lora" | "preprocessor";
      source: string | string[];
      save_path?: string;
      manifest_id?: string;
      lora_name?: string;
    },
    onComplete?: (path: string) => void,
    onError?: (error: unknown, source: string) => void,
  ) => {
    const sources = Array.isArray(request.source)
      ? request.source
      : [request.source];

    set((state) => {
      const next = new Set(state.downloadingPaths);
      for (const s of sources) next.add(s);
      return { downloadingPaths: next };
    });
    const jobIds: string[] = [];
    for (const s of sources) {
      try {
        const jobId = await get().startDownload({
          item_type: request.item_type,
          source: s,
          save_path: request.save_path,
          manifest_id: request.manifest_id,
          lora_name: request.lora_name,
        });
        if (jobId) jobIds.push(jobId);
        try {
          const res = await get().resolveDownload({
            item_type: request.item_type,
            source: s,
            save_path: request.save_path,
          });

          // If the source is already fully downloaded, clear local tracking
          // and skip subscribing to WS updates for this source.
          if (res?.downloaded) {
            set((state) => {
              const nextDownloading = new Set(state.downloadingPaths);
              nextDownloading.delete(s);

              const nextWsFilesByPath = { ...state.wsFilesByPath };
              if (nextWsFilesByPath[s]) {
                delete nextWsFilesByPath[s];
              }

              return {
                downloadingPaths: nextDownloading,
                wsFilesByPath: nextWsFilesByPath,
              };
            });
            continue;
          }
        } catch {}
        if (jobId) {
          await get().subscribeToJob(jobId, s, onComplete, onError);
        }
      } catch (error) {
        set((state) => {
          const next = new Set(state.downloadingPaths);
          next.delete(s);
          return { downloadingPaths: next };
        });
        if (onError) {
          try {
            onError(error, s);
          } catch {
            // Swallow errors from onError handler to avoid breaking the download loop
          }
        }
      }
    }
    return jobIds;
  },
  startDownload: async (request: {
    item_type: "component" | "lora" | "preprocessor";
    source: string | string[];
    save_path?: string;
    job_id?: string;
    manifest_id?: string;
    lora_name?: string;
  }) => {
    const response = await startUnifiedDownload(request);
    if (response.success) {
      return response.data?.job_id || "";
    }
    throw new Error(response.error || "Failed to start download");
  },
  resolveDownload: async (request: {
    item_type: "component" | "lora" | "preprocessor";
    source: string | string[];
    save_path?: string;
  }) => {
    const response = await resolveUnifiedDownload(request);
    if (response.success) {
      return response.data;
    }
    return undefined;
  },
  getDownloadStatus: async (jobId: string) => {
    const response = await getUnifiedDownloadStatus(jobId);
    if (response.success) {
      return response.data;
    }
    return undefined;
  },
  resolveDownloadBatch: async (request: {
    item_type: "component" | "lora" | "preprocessor";
    sources: Array<string | string[]>;
    save_path?: string;
  }) => {
    const response = await resolveUnifiedDownloadBatch(request);
    if (response.success) {
      return response.data;
    }
    return undefined;
  },
  cancelDownload: async (jobId: string) => {
    // Always clear local state for this jobId, regardless of backend response
    try {
      const response = await cancelUnifiedDownload(jobId);
      if (response.success) {
        return response.data;
      }
    } catch {
      // Swallow errors: UI state has already been cleaned up locally
    }
    return undefined;
  },
  deleteDownload: async (request: {
    path: string;
    item_type?: "component" | "lora" | "preprocessor";
    source?: string | string[];
    save_path?: string;
  }) => {
    const response = await deleteDownload(request);
    if (response.success) {
      return response.data;
    }
    return undefined;
  },
  connectDownloadWS: async (jobId: string) => {
    const res = await connectUnifiedDownloadWebSocket(jobId);
    return !!res.success;
  },
  disconnectDownloadWS: async (jobId: string) => {
    const res = await disconnectUnifiedDownloadWebSocket(jobId);
    return !!res.success;
  },
  onDownloadUpdate: (jobId: string, cb: (data: any) => void) => {
    return onUnifiedDownloadUpdate(jobId, cb);
  },
  onDownloadStatus: (jobId: string, cb: (data: any) => void) => {
    return onUnifiedDownloadStatus(jobId, cb);
  },
  onDownloadError: (jobId: string, cb: (data: any) => void) => {
    return onUnifiedDownloadError(jobId, cb);
  },
}));
