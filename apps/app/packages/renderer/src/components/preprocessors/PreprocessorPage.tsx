import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  LuChevronLeft,
  LuImage,
  LuVideo,
  LuTrash,
  LuLoader,
  LuDownload,
} from "react-icons/lu";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  deletePreprocessor as deletePreprocessorApi,
} from "@/lib/preprocessor/api";
import { cn } from "@/lib/utils";

import {
  PREPROCESSOR_QUERY_KEY,
  PREPROCESSORS_LIST_QUERY_KEY,
  usePreprocessorQuery,
} from "@/lib/preprocessor/queries";
import { useQueryClient } from "@tanstack/react-query";
import { ProgressBar } from "@/components/common/ProgressBar";
import { unifiedDownloadWsUpdatesToFiles } from "@/lib/download/ws-updates-to-files";
import { useDownloadJobIdStore } from "@/lib/download/job-id-store";
import { useStartUnifiedDownloadMutation } from "@/lib/download/mutations";
import { useQuery } from "@tanstack/react-query";
import { fetchRayJobs, type RayJobStatus } from "@/lib/jobs/api";
import { cancelRayJob } from "@/lib/jobs/api";

import {
  formatDownloadProgress,
  formatSpeed,
} from "@/lib/components-download/format";

interface PreprocessorPageProps {
  preprocessorId: string;
  onBack?: () => void;
}

const PreprocessorPage: React.FC<PreprocessorPageProps> = ({
  preprocessorId,
  onBack,
}) => {
  const queryClient = useQueryClient();
  const {
    data,
    isLoading: loading,
    isError,
  } = usePreprocessorQuery(preprocessorId, {
    enabled: !!preprocessorId,
    forceFetch: true,
  });

  if (loading) return null;
  if (isError || !data) return null;

  const totalBytes = (data.files ?? []).reduce(
    (acc, f) => acc + (f.size_bytes || 0),
    0,
  );
  const formatSize = (bytes: number): string | null => {
    if (bytes === 0) return null;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024)
      return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(0)} GB`;
  };
  const totalSize = formatSize(totalBytes);

  return (
    <div className="flex flex-col h-full w-full">
      <ScrollArea className="flex-1">
        <div className="p-4 pt-3 pb-28">
          {onBack && (
            <div className="flex items-center gap-x-3">
              <button
                onClick={onBack}
                className="text-brand-light hover:text-brand-light/70 p-1 flex items-center justify-center bg-brand border border-brand-light/10 rounded transition-colors cursor-pointer"
              >
                <LuChevronLeft className="w-3 h-3" />
              </button>
              <span className="text-brand-light/90 text-[11px] font-medium">
                Back
              </span>
            </div>
          )}

          <div className="mt-4 flex flex-col gap-y-4 w-full">
            <div className="flex flex-col gap-y-2 min-w-0">
              <h2 className="text-brand-light text-[16px] font-semibold text-start truncate">
                {data.name}
              </h2>
              <p className="text-brand-light/90 text-[11px] text-start">
                {data.description}
              </p>

              <div className="flex flex-col mt-1 items-start gap-y-0.5">
                <span className="text-brand-light text-[12px] font-medium">
                  {data.category}
                </span>
                {totalSize && (
                  <span className="text-brand-light/80 text-[11px]">
                    {totalSize}
                  </span>
                )}
              </div>

              <div className="flex flex-row items-center gap-x-1.5 mt-2">
                {data.supports_image && (
                  <span className="text-brand-light text-[11px] bg-brand border shadow border-brand-light/10 rounded px-2 py-1 flex items-center gap-x-1.5">
                    <LuImage className="w-3 h-3" />
                    <span className="text-brand-light text-[10px]">Image</span>
                  </span>
                )}
                {data.supports_video && (
                  <span className="text-brand-light text-[11px] bg-brand border shadow border-brand-light/10 rounded px-2 py-1 flex items-center gap-x-1.5">
                    <LuVideo className="w-3 h-3" />
                    <span className="text-brand-light text-[10px]">Video</span>
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* If not downloaded yet, show planned files and a Download action */}
          {!data.is_downloaded && Array.isArray(data.files) && (
            <PreprocessorDownloadSection
              preprocessorId={data.id}
              files={data.files}
              onDownloaded={async () => {
                try {
                  await Promise.all([
                    queryClient.invalidateQueries({
                      queryKey: PREPROCESSORS_LIST_QUERY_KEY,
                    }),
                    queryClient.invalidateQueries({
                      queryKey: PREPROCESSOR_QUERY_KEY(preprocessorId),
                    }),
                  ]);
                } catch {}
              }}
            />
          )}

          {Boolean(data.is_downloaded) &&
            Array.isArray(data.files) &&
            data.files.length > 0 && (
              <PreprocessorFilesSection
                preprocessorId={data.id}
                files={data.files}
                onRefresh={async () => {
                  try {
                    await queryClient.invalidateQueries({
                      queryKey: PREPROCESSOR_QUERY_KEY(preprocessorId),
                    });
                  } catch {}
                }}
              />
            )}
        </div>
      </ScrollArea>
    </div>
  );
};

type DownloadEntry = {
  filename?: string;
  downloadedBytes?: number;
  totalBytes?: number;
  status?: string;
  progress?: number | null;
  message?: string;
  bucket?: string;
  label?: string;
  downloadSpeed?: number;
};
const PreprocessorDownloadSection: React.FC<{
  preprocessorId: string;
  files: { path: string; size_bytes: number; name?: string }[];
  onDownloaded: () => Promise<void>;
}> = ({ preprocessorId, files, onDownloaded }) => {
  const queryClient = useQueryClient();
  const {
    addSourceToJobId,
    getSourceToJobId,
    getJobUpdates,
    removeJobUpdates,
    removeSourceToJobId,
  } = useDownloadJobIdStore();

  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  // Shared polling cache (also used by JobsMenu); when a download is active, ensure it keeps polling.
  const { data: polledJobs = [] } = useQuery<RayJobStatus[]>({
    queryKey: ["rayJobs"],
    queryFn: fetchRayJobs,
    placeholderData: (prev) => prev ?? [],
    retry: false,
    refetchOnWindowFocus: false,
    refetchInterval: starting ? 2000 : 4000,
    refetchIntervalInBackground: true,
  });

  const [localJobId, setLocalJobId] = useState<string | null>(null);
  const mappedJobId = getSourceToJobId(preprocessorId) || null;
  const jobId = mappedJobId || localJobId;
  const jobUpdates = getJobUpdates(jobId ?? undefined) ?? [];

  const { mutateAsync: startDownload } = useStartUnifiedDownloadMutation({
    onSuccess(data, variables) {
      addSourceToJobId(variables.source, data.job_id);
      setLocalJobId(data.job_id);
    },
  });

  const totalBytes = files.reduce((acc, f) => acc + (f.size_bytes || 0), 0);
  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024)
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const terminalStatuses = useMemo(
    () => new Set(["complete", "completed", "cancelled", "canceled", "error", "failed"]),
    [],
  );

  const polledJob = useMemo(() => {
    if (!jobId) return null;
    return (polledJobs || []).find((j) => j.job_id === jobId) ?? null;
  }, [jobId, polledJobs]);

  const polledStatus = (polledJob?.status || "").toLowerCase();
  const isJobActive =
    !!jobId &&
    !!polledJob &&
    polledJob.category === "download" &&
    !terminalStatuses.has(polledStatus);

  const isDownloading = starting || (jobUpdates?.length ?? 0) > 0 || isJobActive;

  // When we observe the job complete (via polling), refresh preprocessor queries.
  useEffect(() => {
    if (!jobId || !polledJob) return;
    if (!terminalStatuses.has(polledStatus)) return;
    
    void (async () => {
      try {
        await Promise.all([
          queryClient.invalidateQueries({ queryKey: PREPROCESSOR_QUERY_KEY(preprocessorId) }),
        ]);
      } catch {}
      try {
        await onDownloaded();
      } catch {}
      setStarting(false);
      setCancelling(false);
    })();
  }, [
    jobId,
    polledJob,
    polledStatus,
    terminalStatuses,
    onDownloaded,
    queryClient,
    preprocessorId,
  ]);

  const wsFilesByName = useMemo(() => {
    const map: Record<string, DownloadEntry> = {};
    const downloadFiles = unifiedDownloadWsUpdatesToFiles(jobUpdates) as any[];
    downloadFiles.forEach((entry) => {
      if (entry.filename) map[entry.filename] = entry;
      if (entry.label && !map[entry.label]) map[entry.label] = entry;
    });
    return map;
  }, [jobUpdates]);

  const findWsFile = useCallback(
    (file: { path: string; size_bytes: number; name?: string }) => {
      const baseName =
        file.name || (file.path || "").split(/[/\\]/).pop() || "";
      const noExt = baseName ? baseName.replace(/\.[^/.]+$/, "") : "";
      const candidates = [baseName, file.path, noExt].filter(
        Boolean,
      ) as string[];
      for (const key of candidates) {
        if (key && wsFilesByName[key]) {
          return wsFilesByName[key];
        }
      }
      return undefined;
    },
    [wsFilesByName],
  );

  const getPercent = useCallback((entry?: DownloadEntry) => {
    if (!entry) return 0;
    if (typeof entry.totalBytes === "number" && entry.totalBytes > 0) {
      return Math.max(
        0,
        Math.min(100, ((entry.downloadedBytes || 0) / entry.totalBytes) * 100),
      );
    }
    if (typeof entry.progress === "number") {
      const pct = entry.progress <= 1 ? entry.progress * 100 : entry.progress;
      return Math.max(0, Math.min(100, pct));
    }
    return 0;
  }, []);

  const handleDownload = async () => {
    if (starting || isDownloading) return;
    setStarting(true);
    try {
      const res = await startDownload({
        item_type: "preprocessor",
        source: preprocessorId,
      });
      setLocalJobId(res.job_id);
    } catch {
      setStarting(false);
    }
  };

  const handleCancel = async () => {
    if (!jobId) return;
    try {
      setCancelling(true);
      await cancelRayJob(jobId);
    } catch {}
    try {
      removeJobUpdates(jobId);
    } catch {}
    try {
      removeSourceToJobId(preprocessorId);
    } catch {}
    setStarting(false);
    setCancelling(false);
  };

  const downloadFiles = unifiedDownloadWsUpdatesToFiles(jobUpdates) as any[];

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-brand-light text-[13px] font-semibold">
          Preprocessor Files
        </h4>
        <div className="text-[10px] text-brand-light/80 font-mono">
          {formatSize(totalBytes)}
        </div>
      </div>
      <div className="space-y-2">
        {files.length === 0 ? (
          <div className="text-brand-light/60 text-[12px]">
            No files listed for this preprocessor.
          </div>
        ) : (
          files.map((f, idx) => {
            const name =
              f.name || (f.path || "").split(/[/\\]/).pop() || f.path;
            const wsFile = findWsFile(f);
            const pct = getPercent(wsFile);
            const fileSizeBytes = (wsFile?.totalBytes ?? f.size_bytes) || 0;
            return (
              <div
                key={`${f.path}-${idx}`}
                className="bg-brand border border-brand-light/10 rounded-md p-3"
              >
                <div className="flex items-start justify-between gap-x-2 w-full">
                  <div className="flex-1 min-w-0">
                    <div className="text-[10px] text-brand-light/90 font-medium break-all text-start">
                      {name}
                    </div>
                    <div className="text-[10px] text-brand-light/60 font-mono break-all text-start">
                      {f.path}
                    </div>
                  </div>
                  <div className="text-[10px] text-brand-light/80 font-mono shrink-0">
                    {formatSize(fileSizeBytes)}
                  </div>
                </div>
                {wsFile && (
                  <div className="mt-2 flex flex-col gap-y-1">
                    <ProgressBar
                      percent={pct}
                      barClassName="bg-brand-light/50"
                    />
                    <div className="flex items-center justify-between text-[10px] text-brand-light/80">
                      {typeof wsFile.downloadedBytes === "number" &&
                      typeof wsFile.totalBytes === "number" &&
                      wsFile.totalBytes > 0 ? (
                        <span>
                          {formatDownloadProgress(
                            wsFile.downloadedBytes,
                            wsFile.totalBytes,
                          )}
                        </span>
                      ) : (
                        <span />
                      )}
                      {wsFile.status === "completed" ||
                      wsFile.status === "complete" ? (
                        <span className="text-brand-light/90">Completed</span>
                      ) : (
                        <span className="text-[9px] text-brand-light/60">
                          {wsFile.downloadSpeed
                            ? formatSpeed(wsFile.downloadSpeed)
                            : ""}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
      <div className="mt-3">
        {jobId && downloadFiles.length > 0 ? (
          <div className="w-full flex items-center gap-x-2">
            <button
              onClick={handleCancel}
              className="w-full text-[10px] text-brand-light/90 font-medium hover:text-brand-light transition-all duration-200 bg-brand hover:bg-brand/70 border border-brand-light/10 rounded-[6px] px-2 py-2"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={handleDownload}
            className="w-full text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-md px-3 py-2 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
            disabled={starting || isDownloading}
          >
            {starting || isDownloading ? (
              <LuLoader className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <LuDownload className="w-3.5 h-3.5" />
            )}
            <span>
              {cancelling
                ? "Cancelling..."
                : starting || isDownloading
                  ? "Downloading..."
                  : "Download Preprocessor"}
            </span>
          </button>
        )}
      </div>
    </div>
  );
};

const PreprocessorFilesSection: React.FC<{
  preprocessorId: string;
  files: { path: string; size_bytes: number; name?: string }[];
  onRefresh: () => Promise<void>;
}> = ({ preprocessorId, files, onRefresh }) => {
  const queryClient = useQueryClient();
  const [deleting, setDeleting] = useState(false);
  const { getSourceToJobId, removeJobUpdates, removeSourceToJobId } =
    useDownloadJobIdStore();
  const activeJobId = getSourceToJobId(preprocessorId) ?? null;

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024)
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const totalBytes = files.reduce((acc, f) => acc + (f.size_bytes || 0), 0);

  const handleDeleteAll = async () => {
    if (deleting) return;
    setDeleting(true);
    try {
      if (activeJobId) {
        try {
          await cancelRayJob(activeJobId);
        } catch {}
        try {
          removeJobUpdates(activeJobId);
        } catch {}
        try {
          removeSourceToJobId(preprocessorId);
        } catch {}
      }
      await deletePreprocessorApi(preprocessorId);
      await onRefresh();
      try {
        await queryClient.invalidateQueries({
          queryKey: PREPROCESSORS_LIST_QUERY_KEY,
        });
      } catch {}
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-brand-light text-[13px] font-semibold">
          Downloaded Files
        </h4>
        <div className="text-[10px] text-brand-light/80 font-mono">
          {formatSize(totalBytes)}
        </div>
      </div>
      <div className="space-y-2">
        {files.map((f, idx) => {
          const name = f.name || (f.path || "").split(/[/\\]/).pop() || f.path;
          return (
            <div
              key={`${f.path}-${idx}`}
              className="bg-brand border border-brand-light/10 rounded-md p-3"
            >
              <div className="flex items-start justify-between gap-x-2 w-full">
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] text-brand-light/90 font-medium break-all text-start">
                    {name}
                  </div>
                  <div className="text-[10px] text-brand-light/60 font-mono break-all text-start">
                    {f.path}
                  </div>
                </div>
                <div className="text-[10px] text-brand-light/80 font-mono shrink-0">
                  {formatSize(f.size_bytes || 0)}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex items-center justify-end mt-3">
        <button
          onClick={handleDeleteAll}
          disabled={deleting}
          className={cn(
            "w-fit text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all",
            {
              "opacity-60 cursor-not-allowed": deleting,
            },
          )}
        >
          {deleting ? (
            <LuLoader className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <LuTrash className="w-3.5 h-3.5" />
          )}
          <span>{deleting ? "Deleting…" : "Delete"}</span>
        </button>
      </div>
    </div>
  );
};

export default PreprocessorPage;
