import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { LoraType } from '@/lib/manifest/api';
import { formatDownloadProgress, formatSpeed } from '@/lib/components-download/format';
import { LuChevronDown, LuChevronRight, LuDownload, LuLoader, LuCheck, LuTrash } from 'react-icons/lu';
import { cn } from '@/lib/utils';
import { useDownloadStore } from '@/lib/download/store';
import { useManifestStore } from '@/lib/manifest/store';

const LoRACard: React.FC<{ item: LoraType; manifestId: string }> = ({ item, manifestId }) => {
  const [starting, setStarting] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [pathToJobId, setPathToJobId] = useState<Record<string, string>>({});
  const [, setDownloadedPaths] = useState<Set<string>>(() => new Set());
  const runRef = useRef<(() => Promise<void>) | null>(null);
  const unsubsRef = useRef<Array<() => void>>([]);
  const isUnmountedRef = useRef(false);

  const path = useMemo(() => (typeof item === 'string' ? item : item?.source || ''), [item]);
  const label = useMemo(() => {
    if (typeof item !== 'string') {
      return item?.label || item?.name || (path ? path.split('/').pop() || path : 'LoRA');
    }
    return path ? path.split('/').pop() || path : 'LoRA';
  }, [item, path]);
  const scale = useMemo(() => (typeof item !== 'string' ? item?.scale : undefined), [item]);

  const {
    startAndTrackDownload,
    cancelDownload,
    resolveDownload,
    deleteDownload: deleteDownloadEntry,
    subscribeToJob,
    downloadingPaths,
    wsFilesByPath,
  } = useDownloadStore();
  const { refreshManifestPart } = useManifestStore();

  const manifestDownloaded = typeof item !== 'string' && !!item?.is_downloaded;
  const isDownloaded = manifestDownloaded;
  const wsFilesObj = path ? wsFilesByPath[path] || {} : {};
  const files = Object.values(wsFilesObj) as any[];
  const jobId = path ? pathToJobId[path] : undefined;
  const isActive =
    !!path &&
    (downloadingPaths.has(path) || files.some((f) => f?.status === 'processing' || f?.status === 'pending'));

  const computePercent = useCallback((file: any) => {
    if (!file) return 0;
    if (typeof file.totalBytes === 'number' && file.totalBytes > 0) {
      const pct = ((file.downloadedBytes || 0) / file.totalBytes) * 100;
      return Math.max(0, Math.min(100, pct));
    }
    if (typeof file.progress === 'number') {
      const base = file.progress > 1 ? file.progress : file.progress * 100;
      return Math.max(0, Math.min(100, base));
    }
    return 0;
  }, []);

  const onDownloadComplete = useCallback(
    async (completed: string | string[]) => {
      const list = (Array.isArray(completed) ? completed : [completed]).filter(
        (p): p is string => typeof p === 'string' && !!p,
      );
      if (list.length === 0) return;
      setDownloadedPaths((prev) => {
        const next = new Set(prev);
        list.forEach((p) => next.add(p));
        return next;
      });
      setPathToJobId((prev) => {
        const next = { ...prev };
        list.forEach((p) => {
          if (next[p]) delete next[p];
        });
        return next;
      });
      try {
        await refreshManifestPart(manifestId, 'spec.loras');
      } catch {}
      try {
        window.dispatchEvent(new CustomEvent('lora-card-reload', { detail: { paths: list, manifestId } }));
      } catch {}
      runRef.current?.();
    },
    [manifestId, refreshManifestPart],
  );

  useEffect(() => {
    if (!path) return;
    isUnmountedRef.current = false;
    const run = async () => {
      if (!path) return;
      try {
        const unsubs = unsubsRef.current || [];
        unsubs.forEach((fn) => {
          try {
            fn();
          } catch {}
        });
      } catch {}
      unsubsRef.current = [];
      try {
        const res = await resolveDownload({ item_type: 'lora', source: path });
        if (isUnmountedRef.current) return;
        setDownloadedPaths(() => {
          const next = new Set<string>();
          if (res?.downloaded && path) next.add(path);
          return next;
        });
        if (res?.job_id && path) {
          setPathToJobId({ [path]: res.job_id });
          if (res.running) {
            try {
              const off = await subscribeToJob(res.job_id, path, onDownloadComplete);
              unsubsRef.current.push(() => {
                try {
                  off();
                } catch {}
              });
            } catch {}
          }
        }
      } catch {}
    };
    runRef.current = run;
    run();
    return () => {
      isUnmountedRef.current = true;
      const unsubs = unsubsRef.current || [];
      unsubs.forEach((fn) => {
        try {
          fn();
        } catch {}
      });
      unsubsRef.current = [];
    };
  }, [path, resolveDownload, subscribeToJob, onDownloadComplete]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handler = (event: Event) => {
      try {
        const detail = (event as CustomEvent)?.detail || {};
        const eventManifestId = detail?.manifestId;
        const eventPaths: string[] | undefined = detail?.paths;
        if (eventManifestId && eventManifestId !== manifestId) return;
        if (!path) return;
        if (!eventPaths || eventPaths.includes(path)) {
          runRef.current?.();
        }
      } catch {}
    };
    window.addEventListener('lora-card-reload', handler as EventListener);
    return () => {
      try {
        window.removeEventListener('lora-card-reload', handler as EventListener);
      } catch {}
    };
  }, [manifestId, path]);

  const handleDownload = useCallback(async () => {
    if (!path || isDownloaded) return;
    setStarting(true);
    try {
      const jobIds = await startAndTrackDownload(
        {
          item_type: 'lora',
          source: path,
        },
        onDownloadComplete,
      );
      if (jobIds?.[0]) {
        setPathToJobId((prev) => ({ ...prev, [path]: jobIds[0] }));
      }
      try {
        window.dispatchEvent(new CustomEvent('lora-card-reload', { detail: { paths: [path], manifestId } }));
      } catch {}
    } catch {}
    finally {
      setTimeout(() => setStarting(false), 200);
    }
  }, [path, isDownloaded, startAndTrackDownload, onDownloadComplete, manifestId]);

  const handleCancel = useCallback(async () => {
    if (!path || !jobId) return;
    try {
      await cancelDownload(jobId);
    } catch {}
    setPathToJobId((prev) => {
      const next = { ...prev };
      delete next[path];
      return next;
    });
    try {
      window.dispatchEvent(new CustomEvent('lora-card-reload', { detail: { paths: [path], manifestId } }));
    } catch {}
    runRef.current?.();
  }, [path, jobId, cancelDownload, manifestId]);

  const handleDelete = useCallback(async () => {
    if (!path) return;
    setDeleting(true);
    try {
      await deleteDownloadEntry({ path, item_type: 'lora' });
      setDownloadedPaths((prev) => {
        const next = new Set(prev);
        next.delete(path);
        return next;
      });
      await refreshManifestPart(manifestId, 'spec.loras');
      try {
        window.dispatchEvent(new CustomEvent('lora-card-reload', { detail: { paths: [path], manifestId } }));
      } catch {}
    } catch {}
    finally {
      setDeleting(false);
    }
  }, [path, deleteDownloadEntry, manifestId, refreshManifestPart]);

  return (
    <div className="bg-brand border border-brand-light/10 rounded-md text-start">
      <LoRACardHeader
        isActive={isActive}
        isDownloaded={isDownloaded}
        isExpanded={isExpanded}
        label={label}
        onToggleExpanded={() => setIsExpanded((prev) => !prev)}
        starting={starting}
      />
      {isExpanded && (
        <LoRACardBody
          deleting={deleting}
          downloadingPaths={downloadingPaths}
          files={files}
          handleCancel={handleCancel}
          handleDelete={handleDelete}
          handleDownload={handleDownload}
          isActive={isActive}
          isDownloaded={isDownloaded}
          jobId={jobId}
          path={path}
          scale={scale}
          computePercent={computePercent}
        />
      )}
    </div>
  );
};

interface LoRACardHeaderProps {
  isActive: boolean;
  isDownloaded: boolean;
  isExpanded: boolean;
  label: string;
  onToggleExpanded: () => void;
  starting: boolean;
}

const LoRACardHeader: React.FC<LoRACardHeaderProps> = ({
  isActive,
  isDownloaded,
  isExpanded,
  label,
  onToggleExpanded,
  starting,
}) => {
  return (
    <div
      onClick={onToggleExpanded}
      className="w-full flex items-center justify-between p-3 hover:bg-brand-background/30 transition-colors"
    >
      <div className="flex items-center gap-x-2 justify-between w-full mr-2">
        <div className="flex items-center gap-x-2">
          <div
            className={cn(
              'flex items-center justify-center w-4.5 h-4.5 rounded-full border',
              isDownloaded
                ? 'bg-green-500/20 border-green-500/40'
                : starting
                  ? 'bg-brand-background border-brand-light/20'
                  : 'bg-brand-background border-brand-light/20 hover:bg-brand-background/30 cursor-pointer',
            )}
          >
            {isDownloaded ? (
              <LuCheck className="w-3 h-3 text-green-400" />
            ) : isActive ? (
              <LuLoader className="w-2.5 h-2.5 text-brand-light/60 animate-spin" />
            ) : (
              <LuDownload className="w-2.5 h-2.5 text-brand-light/50" />
            )}
          </div>
          <span className="text-brand-light text-[12px] font-medium">{label}</span>
        </div>
      </div>
      {isExpanded ? (
        <LuChevronDown className="w-4 h-4 text-brand-light/60" />
      ) : (
        <LuChevronRight className="w-4 h-4 text-brand-light/60" />
      )}
    </div>
  );
};

interface LoRACardBodyProps {
  computePercent: (file: any) => number;
  deleting: boolean;
  downloadingPaths: Set<string>;
  files: any[];
  handleCancel: () => Promise<void>;
  handleDelete: () => Promise<void>;
  handleDownload: () => Promise<void>;
  isActive: boolean;
  isDownloaded: boolean;
  jobId?: string;
  path: string;
  scale?: number;
}

const LoRACardBody: React.FC<LoRACardBodyProps> = ({
  computePercent,
  deleting,
  downloadingPaths,
  files,
  handleCancel,
  handleDelete,
  handleDownload,
  isActive,
  isDownloaded,
  jobId,
  path,
  scale,
}) => {
  return (
    <div className="px-4 pb-4">
      <div className="space-y-2 mt-1">
        <div className="bg-brand-background border border-brand-light/10 rounded-md p-3 overflow-hidden w-full">
          <div className="text-brand-light text-[10.5px] font-medium mb-1">Path</div>
          <div className="text-brand-light text-[10px] font-mono break-all">{path || '—'}</div>
          {typeof scale === 'number' && (
            <div className="text-brand-light text-[10.5px] w-full  flex gap-x-1 mt-2.5  py-1.5 pt-2 border-t border-brand-light/10 flex-col">
              <div className="text-brand-light font-medium mb-1">LoRA Specifications</div>
              <div className="flex gap-x-1.5 text-[10px]">
                <span className="text-brand-light/70 font-medium block">Weight Scale</span>{' '}
                <span className="text-brand-light font-mono block">{scale.toFixed(2)}</span>
              </div>
            </div>
          )}

          {!path ? null : isDownloaded ? (
            <DownloadedLoRASection deleting={deleting} handleDelete={handleDelete} />
          ) : isActive ? (
            <ActiveDownloadSection
              computePercent={computePercent}
              downloadingPaths={downloadingPaths}
              files={files}
              handleCancel={handleCancel}
              jobId={jobId}
              path={path}
            />
          ) : (
            <IdleDownloadSection
              downloadingPaths={downloadingPaths}
              handleDownload={handleDownload}
              path={path}
            />
          )}
        </div>
      </div>
    </div>
  );
};

interface DownloadedLoRASectionProps {
  deleting: boolean;
  handleDelete: () => Promise<void>;
}

const DownloadedLoRASection: React.FC<DownloadedLoRASectionProps> = ({ deleting, handleDelete }) => {
  return (
    <div className="flex flex-row items-center justify-between gap-x-2 mt-2">
      <div className="flex items-center justify-start gap-x-1">
        <LuCheck className="w-3 h-3 text-green-400" />
        <span className="text-[11px] font-medium text-brand-light/90">Downloaded</span>
      </div>
      <button
        onClick={handleDelete}
        disabled={deleting}
        className="w-fit text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all"
      >
        {deleting ? (
          <LuLoader className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <LuTrash className="w-3.5 h-3.5" />
        )}
        <span>{deleting ? 'Deleting...' : 'Delete LoRA'}</span>
      </button>
    </div>
  );
};

interface ActiveDownloadSectionProps {
  computePercent: (file: any) => number;
  downloadingPaths: Set<string>;
  files: any[];
  handleCancel: () => Promise<void>;
  jobId?: string;
  path: string;
}

const ActiveDownloadSection: React.FC<ActiveDownloadSectionProps> = ({
  computePercent,
  downloadingPaths,
  files,
  handleCancel,
  jobId,
  path,
}) => {
  return (
    <>
      <div className="w-full mt-3">
        {files.length > 0 ? (
          <div className="flex flex-col gap-y-2">
            {files.map((f: any, idx: number) => (
              <div key={f.filename || f.label || idx} className="flex flex-col gap-y-1">
                <div className="flex items-center justify-between gap-x-2 w-full">
                  <div className="flex-1 min-w-0">
                    <div className="text-[10px] text-brand-light/80 font-mono truncate break-all">
                      {f.filename || f.label || path}
                    </div>
                  </div>
                  <div className="text-[10px] text-brand-light/80 font-mono flex-shrink-0">
                    {computePercent(f).toFixed(1)}%
                  </div>
                </div>
                <div className="w-full h-2 bg-brand-background rounded overflow-hidden border border-brand-light/10">
                  <div
                    className="h-full bg-brand/90 transition-all"
                    style={{
                      width: `${computePercent(f)}%`,
                    }}
                  />
                </div>
                <div className="flex items-center justify-between">
                  {typeof f.downloadedBytes === 'number' && typeof f.totalBytes === 'number' ? (
                    <div className="text-[10px] text-brand-light/90">
                      {formatDownloadProgress(f.downloadedBytes, f.totalBytes)}
                    </div>
                  ) : (
                    <div />
                  )}
                  {f.status === 'completed' || f.status === 'complete' ? (
                    <div className="text-[10px] text-green-400">Completed</div>
                  ) : f.downloadSpeed != null && f.downloadSpeed > 0 ? (
                    <div className="text-[9px] text-brand-light/50">{formatSpeed(f.downloadSpeed)}</div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        ) : null}
        <div className="flex flex-col items-center justify-between mt-2 w-full">
          <button
            onClick={handleCancel}
            disabled={!jobId || downloadingPaths.has(path || '')}
            className="text-[10px] text-brand-light/90 w-full mt-2 font-medium hover:text-brand-light transition-all duration-200 bg-brand hover:bg-brand/70 border border-brand-light/10 rounded-[6px] px-2 py-2 disabled:hover:bg-brand disabled:opacity-90 disabled:cursor-not-allowed"
          >
            {downloadingPaths.has(path || '') ? 'Downloading...' : 'Cancel'}
          </button>
        </div>
      </div>
    </>
  );
};

interface IdleDownloadSectionProps {
  downloadingPaths: Set<string>;
  handleDownload: () => Promise<void>;
  path: string;
}

const IdleDownloadSection: React.FC<IdleDownloadSectionProps> = ({ downloadingPaths, handleDownload, path }) => {
  return (
    <button
      onClick={handleDownload}
      className="w-full mt-3 text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-md px-3 py-2 transition-all"
    >
      {downloadingPaths.has(path || '') ? (
        <LuLoader className="w-3.5 h-3.5 animate-spin" />
      ) : (
        <LuDownload className="w-3.5 h-3.5" />
      )}
      <span>{downloadingPaths.has(path || '') ? 'Downloading...' : 'Download LoRA'}</span>
    </button>
  );
};

export default LoRACard;


