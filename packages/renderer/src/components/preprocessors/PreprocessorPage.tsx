import React from 'react'
import { useEffect, useRef, useState } from 'react'
import { LuChevronLeft, LuChevronDown, LuChevronRight, LuImage, LuVideo, LuTrash, LuLoader
 } from 'react-icons/lu'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { Preprocessor } from '@/lib/preprocessor/api'
import { getPreprocessor, getPreprocessorStatus } from '@/lib/preprocessor/api'
import { cn } from '@/lib/utils'
import { deletePreprocessor as deletePreprocessorApi, downloadPreprocessor as downloadPreprocessorApi, usePreprocessorJob, useJobProgress } from '@/lib/preprocessor/api'
import { usePreprocessorsListStore } from '@/lib/preprocessor/list-store'
import { usePreprocessorJobStore } from '@/lib/preprocessor/store'
import { cancelPreprocessor } from '@/lib/preprocessor/api'
import { formatDownloadProgress, formatSpeed } from '@/lib/components-download/format'

interface PreprocessorPageProps {
  preprocessorId: string;
  onBack?: () => void;
}

const PreprocessorPage:React.FC<PreprocessorPageProps> = ({ preprocessorId, onBack }) => {
  const [data, setData] = useState<Preprocessor | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await getPreprocessor(preprocessorId);
        if (!cancelled) {
          setData(res.data ?? null);
        }
      } catch (e: any) {
        if (!cancelled) setError(e?.message || 'Failed to load preprocessor');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void run();
    return () => { cancelled = true; };
  }, [preprocessorId]);

  

  if (loading) return null;
  if (error || !data) return null;

  const totalBytes = (data.files ?? []).reduce((acc, f) => acc + (f.size_bytes || 0), 0);
  const formatSize = (bytes: number): string | null => {
    if (bytes === 0) return null;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(0)} GB`;
  };
  const totalSize = formatSize(totalBytes);

  return (
    <div className="flex flex-col h-full w-full">
      <ScrollArea className="flex-1">
        <div className="p-7 pt-3 pb-28">
          {onBack && <div className="flex items-center gap-x-3">
            <button onClick={onBack} className="text-brand-light hover:text-brand-light/70 p-1 flex items-center justify-center bg-brand border border-brand-light/10 rounded transition-colors cursor-pointer">
              <LuChevronLeft className="w-3 h-3" />
            </button>
            <span className="text-brand-light/90 text-[11px] font-medium">Back</span>
          </div>}

          <div className='mt-4 flex flex-row gap-x-4 w-full'>
            <div className="rounded-md overflow-hidden flex items-center w-44 aspect-square justify-start flex-shrink-0">
              <img src={`/preprocessors/${data.id}.png`} alt={data.name} className="h-full object-cover rounded-md" />
            </div>

            <div className="flex flex-col gap-y-2 min-w-0">
              <h2 className="text-brand-light text-[18px] font-semibold text-start truncate">{data.name}</h2>
              <p className="text-brand-light/90 text-[12px] text-start">{data.description}</p>

              <div className='flex flex-col mt-1 items-start gap-y-0.5'>
                <span className="text-brand-light text-[12px] font-medium">{data.category}</span>
                {totalSize && (
                  <span className="text-brand-light/80 text-[11px]">{totalSize}</span>
                )}
              </div>

              <div className='flex flex-row items-center gap-x-1.5 mt-2'>
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

          {/* Only the parameters summary section from PreprocessorInfoPanel */}
          {Array.isArray(data.parameters) && data.parameters.length > 0 && (
            <div className="mt-6">
              <InfoParametersSection parameters={data.parameters as InfoParam[]} />
            </div>
          )}

          {/* If not downloaded yet, show planned files and a Download action */}
          {!data.is_downloaded && Array.isArray(data.files) && (
            <PreprocessorDownloadSection
              preprocessorId={data.id}
              files={data.files}
              onDownloaded={async () => {
                try {
                  const res = await getPreprocessor(preprocessorId);
                  setData(res.data ?? null);
                } catch {}
                try { await usePreprocessorsListStore.getState().load(true); } catch {}
              }}
            />
          )}

          {Boolean(data.is_downloaded) && Array.isArray(data.files) && data.files.length > 0 && (
            <PreprocessorFilesSection
              preprocessorId={data.id}
              files={data.files}
              onRefresh={async () => {
                try {
                  const res = await getPreprocessor(preprocessorId);
                  setData(res.data ?? null);
                } catch {}
              }}
            />
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

type InfoParam = {
  name: string;
  display_name?: string;
  type: string;
  description?: string;
  required?: boolean;
};

type FileEntry = { filename: string; downloadedBytes?: number; totalBytes?: number; progress?: number; downloadSpeed?: number };
const PreprocessorDownloadSection: React.FC<{ preprocessorId: string; files: { path: string; size_bytes: number; name?: string }[]; onDownloaded: () => Promise<void> }> = ({ preprocessorId, files, onDownloaded }) => {
  const [jobId, setJobId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const { isProcessing, isComplete } = usePreprocessorJob(jobId, true);
  // Read-only global job state; do not auto-start tracking to avoid phantom pending
  const globalJob = useJobProgress(preprocessorId);
  const isGlobalProcessing = !!(globalJob && (globalJob.status === 'running' || globalJob.status === 'pending'));
  const isGlobalComplete = !!(globalJob && globalJob.status === 'complete');
  const [fileMap, setFileMap] = useState<Record<string, FileEntry>>({});
  


  const totalBytes = files.reduce((acc, f) => acc + (f.size_bytes || 0), 0);
  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  useEffect(() => {
    if (isComplete && jobId) {
      (async () => {
        try { await onDownloaded(); } catch {}
        setJobId(null);
        setStarting(false);
        setFileMap({});
      })();
    }
  }, [isComplete, jobId, onDownloaded]);

  // If a job for this preprocessor is already running elsewhere, adopt it so progress is shown immediately
  useEffect(() => {
    if (!jobId && isGlobalProcessing && !isGlobalComplete) {
      setJobId(preprocessorId);
    }
  }, [jobId, isGlobalProcessing, isGlobalComplete, preprocessorId]);

  // If a global job completes before we adopted it locally, still refresh UI/state
  useEffect(() => {
    if (!jobId && isGlobalComplete) {
      (async () => {
        try { await onDownloaded(); } catch {}
        setStarting(false);
        setFileMap({});
      })();
    }
  }, [jobId, isGlobalComplete, onDownloaded]);

  // Fallback: probe server for an existing job and adopt it if running/pending
  useEffect(() => {
    let cancelled = false;
    if (jobId || isGlobalProcessing || isGlobalComplete) return;
    (async () => {
      try {
        const res = await getPreprocessorStatus(preprocessorId);
        const st = res?.data?.status;
        if (!cancelled && res.success && (st === 'running' || st === 'pending')) {
          setJobId(preprocessorId);
        }
      } catch {}
    })();
    return () => { cancelled = true; };
  }, [jobId, isGlobalProcessing, isGlobalComplete, preprocessorId]);

  // Build per-file progress from job store
  useEffect(() => {
    if (!jobId) return;
    try {
      // initial sync
      const initial = usePreprocessorJobStore.getState().jobs[jobId];
      if (initial?.files) setFileMap(initial.files as Record<string, FileEntry>);
      // subscribe to only this job's updates
      const unsub = usePreprocessorJobStore.subscribe(
        (s) => s.jobs[jobId],
        (j) => {
          if (j?.files) setFileMap(j.files as Record<string, FileEntry>);
        }
      );
      return () => { try { unsub(); } catch {} };
    } catch {}
  }, [jobId]);

  const handleDownload = async () => {
    // Avoid blocking a new download attempt due to stale processing state
    // Also block if there is an active global job for this preprocessor
    if (starting || (jobId && isProcessing) || isGlobalProcessing) return;
    setStarting(true);
    try {
      const res = await downloadPreprocessorApi(preprocessorId, preprocessorId);
      if (res.success) {
        setJobId(preprocessorId);
      } else {
        setStarting(false);
      }
    } catch {
      setStarting(false);
    }
  };

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-brand-light text-[13px] font-semibold">Preprocessor Files</h4>
        <div className="text-[10px] text-brand-light/80 font-mono">{formatSize(totalBytes)}</div>
      </div>
      <div className="space-y-2">
        {files.length === 0 ? (
          <div className="text-brand-light/60 text-[12px]">No files listed for this preprocessor.</div>
        ) : (
          files.map((f, idx) => {
            const name = f.name || (f.path || '').split(/[/\\]/).pop() || f.path;
            const nameNoExt = name.replace(/\.[^/.]+$/, '');
            const feByName = fileMap[name] || fileMap[nameNoExt] || fileMap[(f.name || '').replace(/\.[^/.]+$/, '')];
            const feBySize = (!feByName && f.size_bytes)
              ? Object.values(fileMap).find((e) => (e?.totalBytes ?? 0) === (f.size_bytes || 0))
              : undefined;
            const fe = (feByName || feBySize || {}) as FileEntry;

            const pct = typeof fe.progress === 'number' ? fe.progress : undefined;
            return (
              <div key={`${f.path}-${idx}`} className="bg-brand border border-brand-light/10 rounded-md p-3">
                <div className="flex items-start justify-between gap-x-2 w-full">
                  <div className="flex-1 min-w-0">
                    <div className="text-[10px] text-brand-light/90 font-medium break-all text-start">{name}</div>
                    <div className="text-[10px] text-brand-light/60 font-mono break-all text-start">{f.path}</div>
                  </div>
                  <div className="text-[10px] text-brand-light/80 font-mono flex-shrink-0">{formatSize((fe.totalBytes ?? f.size_bytes) || 0)}</div>
                </div>
                {pct != null ? (
                  <div className="mt-2">
                    <div className="w-full h-2 bg-brand-background rounded overflow-hidden border border-brand-light/10">
                      <div className="h-full bg-brand transition-all" style={{ width: `${pct}%` }} />
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <div className="text-[10px] text-brand-light/90">
                        {formatDownloadProgress(fe.downloadedBytes ?? 0, fe.totalBytes ?? 0)}
                      </div>
                      {fe.downloadSpeed != null && fe.downloadSpeed > 0 ? (
                        <div className="text-[9px] text-brand-light/50">{formatSpeed(fe.downloadSpeed)}</div>
                      ) : <div />}
                    </div>
                  </div>
                ) : ((starting || (jobId && (isProcessing || !isComplete)) || (isGlobalProcessing && !isGlobalComplete)) && (
                  <div className="mt-2 flex items-center gap-x-2 text-brand-light/70">
                    <LuLoader className="w-3.5 h-3.5 animate-spin" />
                    <span className="text-[10px]">Preparing download…</span>
                  </div>
                ))}
              </div>
            );
          })
        )}
      </div>
      <div className="mt-3">
        {jobId && (isProcessing || !isComplete) ? (
          <div className="w-full flex items-center gap-x-2">
            <button
              onClick={async () => {
                try { if (jobId) await cancelPreprocessor(jobId); } catch {}
                setStarting(false);
                setJobId(null);
                setFileMap({});
              }}
              className="text-[10px] text-brand-light/90 mt-0 font-medium hover:text-brand-light transition-all duration-200 bg-brand hover:bg-brand/70 border border-brand-light/10 rounded-[6px] px-2 py-2"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={handleDownload}
            className="w-full text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-md px-3 py-2 transition-all"
            disabled={starting || !!isGlobalProcessing || (!!jobId && !!isProcessing)}
          >
            <span>{starting ? 'Starting…' : 'Download Preprocessor'}</span>
          </button>
        )}
      </div>
    </div>
  );
};

const InfoParamDescription: React.FC<{
  param: InfoParam;
  isExpanded: boolean;
  isTruncated: boolean;
  onToggle: () => void;
  onTruncationDetected: (isTruncated: boolean) => void;
}> = ({ param, isExpanded, isTruncated, onToggle, onTruncationDetected }) => {
  const descRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (descRef.current && param.description && !isExpanded) {
      const checkTruncation = descRef.current.scrollHeight > descRef.current.clientHeight;
      if (checkTruncation !== isTruncated) {
        onTruncationDetected(checkTruncation);
      }
    }
  }, [param.description, isTruncated, isExpanded, onTruncationDetected]);

  if (!param.description) return null;

  return (
    <div className="flex flex-col gap-y-1">
      <span
        ref={descRef}
        className={`text-brand-light text-[12px] text-start ${!isExpanded ? 'line-clamp-1' : ''}`}
      >
        {param.description}
      </span>
      {isTruncated && (
        <button
          onClick={onToggle}
          className="text-brand-light/50 hover:text-brand-light text-[9px] text-start transition-colors duration-200"
        >
          {isExpanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
};

const PreprocessorFilesSection: React.FC<{ preprocessorId: string; files: { path: string; size_bytes: number; name?: string }[]; onRefresh: () => Promise<void> }> = ({ preprocessorId, files, onRefresh }) => {
  const [deleting, setDeleting] = useState(false);
  const loadPreprocessors = usePreprocessorsListStore((s) => s.load);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const totalBytes = files.reduce((acc, f) => acc + (f.size_bytes || 0), 0);

  const handleDeleteAll = async () => {
    if (deleting) return;
    setDeleting(true);
    try {
      // Cancel any active/preparing download job and clear its state so UI doesn't show stale downloading
      try { await cancelPreprocessor(preprocessorId); } catch {}
      try { usePreprocessorJobStore.getState().clearJob(preprocessorId); } catch {}
      await deletePreprocessorApi(preprocessorId);
      await onRefresh();
      try { await loadPreprocessors(true); } catch {}
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-brand-light text-[13px] font-semibold">Downloaded Files</h4>
        <div className="text-[10px] text-brand-light/80 font-mono">{formatSize(totalBytes)}</div>
      </div>
      <div className="space-y-2">
        {files.map((f, idx) => {
          const name = f.name || (f.path || '').split(/[/\\]/).pop() || f.path;
          return (
            <div key={`${f.path}-${idx}`} className="bg-brand border border-brand-light/10 rounded-md p-3">
              <div className="flex items-start justify-between gap-x-2 w-full">
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] text-brand-light/90 font-medium break-all text-start">{name}</div>
                  <div className="text-[10px] text-brand-light/60 font-mono break-all text-start">{f.path}</div>
                </div>
                <div className="text-[10px] text-brand-light/80 font-mono flex-shrink-0">{formatSize(f.size_bytes || 0)}</div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex items-center justify-end mt-3">
        <button
          onClick={handleDeleteAll}
          disabled={deleting}
          className={cn("w-fit text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all", {
            'opacity-60 cursor-not-allowed': deleting,
          })}
        >
          {deleting ? (
            <LuLoader className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <LuTrash className="w-3.5 h-3.5" />
          )}
          <span>{deleting ? 'Deleting…' : 'Delete'}</span>
        </button>
      </div>
    </div>
  );
};

const InfoParametersSection: React.FC<{ parameters: InfoParam[] }> = ({ parameters }) => {
  const [isParametersExpanded, setIsParametersExpanded] = useState(false);
  const [expandedDescriptions, setExpandedDescriptions] = useState<Record<string, boolean>>({});
  const [truncatedDescriptions, setTruncatedDescriptions] = useState<Record<string, boolean>>({});

  const formatParameterName = (name: string) =>
    name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');

  const formatParameterType = (type: string) => {
    if (type === 'int') return 'Integer';
    if (type === 'float') return 'Float';
    if (type === 'bool') return 'Boolean';
    if (type === 'str') return 'String';
    if (type === 'category') return 'Category';
    return type;
  };

  return (
    <div className="flex flex-col  border-t border-brand-light/10 pt-4 ">
      <button
        onClick={() => setIsParametersExpanded(!isParametersExpanded)}
        className={cn("flex flex-row items-center gap-x-2 px-2 py-3 bg-brand rounded-md border border-b-0 rounded-b-none border-brand-light/10 hover:bg-brand/70 transition-colors duration-200", {
          'rounded-b-none': isParametersExpanded,
          'rounded-b-md': !isParametersExpanded,
          'border-b-0': isParametersExpanded,
          'border-b': !isParametersExpanded,
        })}
      >
        {isParametersExpanded ? (
          <LuChevronDown className="text-brand-light w-3.5 h-3.5" />
        ) : (
          <LuChevronRight className="text-brand-light w-3.5 h-3.5" />
        )}
        <h4 className="text-brand-lighter text-[12px] font-semibold text-start">Inputs</h4>
      </button>

      {isParametersExpanded && (
        <div className="flex flex-col gap-y-3 p-3 border-x border-b border-brand-light/10 rounded-b-md bg-brand transition-colors duration-200">
          {parameters.map((param, index) => (
            <div key={`${param.name}-${index}`} className="flex flex-col gap-y-2 p-3 rounded-lg bg-brand-light/5 border border-brand-light/10 ">
              <div className="flex flex-row items-center gap-x-2 justify-between">
                <span className="text-brand-lighter text-[11px] font-medium">{param.display_name || formatParameterName(param.name)}</span>
                <div className="flex items-center gap-x-2">
                  <span className="text-brand-light/60 text-[10px]">{formatParameterType(param.type)}</span>
                  {param.required && (
                    <span className="text-red-400/80 text-[9px] px-1.5 py-0.5 bg-red-400/10 rounded-full border border-red-400/20">Required</span>
                  )}
                </div>
              </div>
              <InfoParamDescription
                param={param}
                isExpanded={expandedDescriptions[param.name] || false}
                isTruncated={truncatedDescriptions[param.name] || false}
                onToggle={() => setExpandedDescriptions(prev => ({ ...prev, [param.name]: !prev[param.name] }))}
                onTruncationDetected={(isTruncated) => {
                  setTruncatedDescriptions(prev => {
                    if (prev[param.name] === isTruncated) return prev;
                    return { ...prev, [param.name]: isTruncated };
                  });
                }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default PreprocessorPage


