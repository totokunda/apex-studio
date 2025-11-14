import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ManifestComponent, ManifestComponentModelPathItem } from '@/lib/manifest/api';
import { cn } from '@/lib/utils';
import { formatDownloadProgress, formatSpeed, formatBytes } from '@/lib/components-download/format';
import { LuChevronDown, LuChevronRight, LuDownload, LuCheck, LuTrash, LuLoader } from 'react-icons/lu';
import { useDownloadStore } from '@/lib/download/store';
import { ProgressBar } from '@/components/common/ProgressBar';
import { useManifestStore } from '@/lib/manifest/store';

const getComponentTypeLabel = (type: string): string => {
  const labels: Record<string, string> = {
    'transformer': 'Transformer',
    'text_encoder': 'Text Encoder',
    'vae': 'Variational Autoencoder',
    'scheduler': 'Scheduler',
    'helper': 'Helper'
  };
  return labels[type] || type.charAt(0).toUpperCase() + type.slice(1);
};

const formatComponentName = (name: string): string => {
  return name
    .replace(/\./g, ' ')
    .replace(/_/g, ' ')
    .replace(/-/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const ComponentCard: React.FC<{ component: ManifestComponent; manifestId: string, index:number }> = ({ component:originalComponent, manifestId, index }) => {
  const { refreshManifestPart, getLoadedManifest } = useManifestStore();
  const [isExpanded, setIsExpanded] = useState(false);
  const { startAndTrackDownload,  cancelDownload, resolveDownloadBatch, deleteDownload, subscribeToJob, downloadingPaths, wsFilesByPath } = useDownloadStore();
  const [deletingPaths, setDeletingPaths] = useState<Set<string>>(new Set());
  const schedulersConfigDownloading = false;
  const [downloadedPaths, setDownloadedPaths] = useState<Set<string>>(new Set());
  const [pathToJobId, setPathToJobId] = useState<Record<string, string>>({});
  const liveComponent = getLoadedManifest(manifestId)?.spec.components?.[index] as ManifestComponent | undefined;
  const component = liveComponent || originalComponent;
  const [schedulerIsDownloading, setSchedulerIsDownloading] = useState(false);

  const relevantPaths = useMemo(() => {
    const paths: string[] = [];
    const modelPathsRaw = Array.isArray(component.model_path)
      ? component.model_path
      : component.model_path
        ? [{ path: component.model_path }]
        : [];
    for (const it of modelPathsRaw) {
      const p = typeof it === 'string' ? it : (it as any)?.path;
      if (p) paths.push(p);
    }
  const baseConfig = (component as any)?.config_path;
    if (typeof baseConfig === 'string' && baseConfig) paths.push(baseConfig);
    const optionConfigs = Array.isArray((component as any)?.scheduler_options)
      ? (component as any).scheduler_options.map((o: any) => o?.config_path).filter(Boolean)
      : [];
    for (const p of optionConfigs) if (typeof p === 'string') paths.push(p);
    return Array.from(new Set(paths));
  }, [component]);

  const onCompleteDownload = useCallback(async (_: string) => {
    window.dispatchEvent(new CustomEvent('component-card-reload', { detail: { paths: relevantPaths, manifestId: manifestId, componentIndex: index } }));
    await refreshManifestPart(manifestId, `spec.components.${index}`);
    if (component.type === 'scheduler') {
      setSchedulerIsDownloading(s => !s);
    }
  }, [manifestId, index, component.type]);

  const isComponentDownlading = useMemo(() => {
    return relevantPaths.some((p) => downloadingPaths.has(p)) || relevantPaths.some((p) => wsFilesByPath[p] && Object.values(wsFilesByPath[p]).some((v) => v.status === 'processing' || v.status === 'pending')) || 
     schedulerIsDownloading 
  }, [relevantPaths, wsFilesByPath, downloadingPaths]);

  const defaultPaths = useMemo(() => {
    const paths: string[] = [];
    const modelPathsRaw = Array.isArray(component.model_path)
      ? component.model_path
      : component.model_path
        ? [{ path: component.model_path }]
        : [];
    for (const it of modelPathsRaw) {
      const p = typeof it === 'string' ? it : (it as any)?.path;
      if (p && (typeof it === 'string' || (it as any)?.variant?.toLowerCase() === 'default')) paths.push(p);
    }
  const baseConfig = (component as any)?.config_path;
    if (typeof baseConfig === 'string' && baseConfig) paths.push(baseConfig);
    const optionConfigs = Array.isArray((component as any)?.scheduler_options)
      ? (component as any).scheduler_options.map((o: any) => o?.config_path).filter(Boolean)
      : [];
    for (const p of optionConfigs) if (typeof p === 'string') paths.push(p);
    return Array.from(new Set(paths));
  }, [component]);

  const runRef = useRef<(() => Promise<void>) | null>(null);
  const unsubsRef = useRef<Array<() => void>>([]);
  const isUnmountedRef = useRef(false);

  useEffect(() => {
    isUnmountedRef.current = false;
    const run = async () => {
      try {
        // cleanup any previous subscriptions before re-running
        try {
          (unsubsRef.current || []).forEach((fn) => { try { fn(); } catch {} });
        } catch {}
        unsubsRef.current = [];

        const response = await resolveDownloadBatch({
          item_type: 'component',
          sources: relevantPaths,
        });
        if (isUnmountedRef.current) return;
        const results = response?.results || [];

        // Build fresh state snapshots to ensure deletes/downloads are reflected
        const nextDownloadedPaths = new Set<string>();
        const nextPathToJobId: Record<string, string> = {};
        for (let idx = 0; idx < results.length; idx++) {
          const r = results[idx];
          const src = relevantPaths[idx];
          if (!r?.job_id || !src) continue;
          nextPathToJobId[src] = r.job_id;
          if (r.downloaded && src) {
            nextDownloadedPaths.add(src);
          }
        }
        setDownloadedPaths(nextDownloadedPaths);
        setPathToJobId(nextPathToJobId);
        // Subscribe to any running jobs after state is set
        for (let idx = 0; idx < results.length; idx++) {
          const r = results[idx];
          const src = relevantPaths[idx];
          if (r?.job_id && r.running && src) {
            try {
              const off = await subscribeToJob(r.job_id, src, onCompleteDownload);
              unsubsRef.current.push(() => { try { off(); } catch {} });
            } catch {}
          }
        }
      } catch {}
    };
    runRef.current = run;
    run();
    return () => {
      isUnmountedRef.current = true;
      try {
        (unsubsRef.current || []).forEach((fn) => { try { fn(); } catch {} });
      } finally {
        unsubsRef.current = [];
      }
    };
  }, [component, relevantPaths]);

  useEffect(() => {
    const handler = (e: any) => {
      try {
        const detail = e?.detail || {};
        const paths: string[] | undefined = detail?.paths;
        const componentName: string | undefined = detail?.componentName;
        const componentIndex: number | undefined = detail?.componentIndex;
        const compName = (component as any)?.name || (component as any)?.base || '';
        const windowManifestId = detail?.manifestId;
        if (windowManifestId && windowManifestId !== manifestId) return;

        // If no filter provided, reload all. Otherwise, match on paths or component identity.
        if (!paths && !componentName) {
          runRef.current?.();
          return;
        }
        const intersects = Array.isArray(paths) ? paths.some((p) => relevantPaths.includes(p)) : false;
        const matchesComponent = typeof componentName === 'string' && !!componentName && componentName === compName;
        const matchesComponentIndex = typeof componentIndex === 'number' && !!componentIndex && componentIndex === index;
        if (intersects || matchesComponent || matchesComponentIndex) {
          runRef.current?.();
        }
      } catch {}
    };
    try { window.addEventListener('component-card-reload', handler as EventListener); } catch {}
    return () => {
      try { window.removeEventListener('component-card-reload', handler as EventListener); } catch {}
    };
  }, [component, relevantPaths]);

  const modelPaths = Array.isArray(component.model_path) ? component.model_path : component.model_path ? [{ path: component.model_path }] : [];

  const schedulerConfigPaths = useMemo(() => {
    const paths: string[] = [];
    const baseConfig = (component)?.config_path;
    if (typeof baseConfig === 'string' && baseConfig) paths.push(baseConfig);
    if (Array.isArray((component)?.scheduler_options)) {
      for (const opt of (component)?.scheduler_options) {
        const cp = opt?.config_path;
        if (typeof cp === 'string' && cp) paths.push(cp);
      }
    }
    return Array.from(new Set(paths));
  }, [component]);

  const componentFlagDownloaded = !!(component)?.is_downloaded;

  const isConfigOnly = useMemo(() => {
    const hasNoModelPaths = (Array.isArray(modelPaths) ? modelPaths : []).length === 0;
    const baseConfig = (component)?.config_path
    const schedulerOptions = (component)?.scheduler_options;
    const schedulerConfigs = schedulerOptions?.map((option) => option.config_path).filter(Boolean);
    return hasNoModelPaths && (typeof baseConfig === 'string' && !!baseConfig || (schedulerConfigs && schedulerConfigs.length > 0));
  }, [component, modelPaths]);
  const baseConfigPath = useMemo(() => {
    const val = (component)?.config_path;
    return typeof val === 'string' && val ? (val as string) : undefined;
  }, [component]);

  const handleDownload = async (path: string | string[]) => {
    if (!path) return;
    window.dispatchEvent(new CustomEvent('component-card-reload', { detail: { paths: Array.isArray(path) ? path : [path], manifestId: manifestId, componentIndex: index } }));
    const jobIds = await startAndTrackDownload({
      item_type: 'component',
      source: Array.isArray(path) ? path : [path],
    }, onCompleteDownload);
    if (jobIds.length > 0) {
      // add the path to each job id
      let pathList:string[] = Array.isArray(path) ? path : [path];
      const nextPathToJobId: Record<string, string> = {};
      for (let idx = 0; idx < pathList.length; idx++) {
        const jobId = jobIds[idx];
        if (jobId && pathList[idx]) {
          nextPathToJobId[pathList[idx]] = jobId;
        }
      }
      setPathToJobId(nextPathToJobId);
    }
    if (component.type === 'scheduler') {
      setSchedulerIsDownloading(true);
    }
  };

  const handleCancel = async (jobId: string) => {

    await cancelDownload(jobId);
    try {
      window.dispatchEvent(new CustomEvent('component-card-reload', { detail: { componentName: component.name, manifestId: manifestId, componentIndex: index } }));
    } catch {}
  };

  const handleDelete = async (path: string) => {
    setDeletingPaths((s) => new Set([...s, path]));
    await deleteDownload({
      path,
      item_type: 'component',
    });
    try {
      window.dispatchEvent(new CustomEvent('component-card-reload', { detail: { paths: [path], manifestId: manifestId, componentIndex: index } }));
    } catch {}
    await refreshManifestPart(manifestId, `spec.components.${index}`);
    setDeletingPaths((s) => new Set(Array.from(s).filter((p) => p !== path)));
  };

  const handleDownloadAll = async () => {
    await startAndTrackDownload({
      item_type: 'component',
      source: defaultPaths,
    }, onCompleteDownload);
    
    if (component.type === 'scheduler') {
      setSchedulerIsDownloading(true);
    }
  };

  const typeLabel = getComponentTypeLabel(component.type);
  const displayName = component.label || (component.name ? formatComponentName(component.name) : component.base ? formatComponentName(component.base) : typeLabel);
  const componentCarRef = useRef<HTMLDivElement>(null);

  return (
    <div ref={componentCarRef} className="bg-brand border border-brand-light/10 rounded-md text-start">
      {(() => {
        const hasModelPaths = modelPaths.length > 0;
        const hasSchedulerOptions = component.type === 'scheduler' && component.scheduler_options && component.scheduler_options.length > 0;
        const hasExpandableContent = hasModelPaths || hasSchedulerOptions || isConfigOnly;
        return hasExpandableContent ? (
          <div 
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full flex items-center justify-between p-3 hover:bg-brand-background/30 transition-colors"
          >
            <div className="flex items-center gap-x-2 justify-between w-full mr-2">
              <div className="flex items-center gap-x-2">
                <button onClick={async (e) => {
                  if (componentFlagDownloaded || isComponentDownlading) return;
                  e.preventDefault();
                  e.stopPropagation();
                  await handleDownloadAll();
                }} className={cn(
                  "flex items-center justify-center w-4.5 h-4.5 rounded-full border",
                  componentFlagDownloaded
                    ? "bg-green-500/20 border-green-500/40" 
                    : isComponentDownlading
                      ? "bg-brand-background border-brand-light/20"
                    : "bg-brand-background border-brand-light/20 hover:bg-brand-background/30 cursor-pointer"
                )}>
                  {componentFlagDownloaded ? (
                    <LuCheck className="w-3 h-3 text-green-400" />
                  ) : isComponentDownlading ? (
                    <LuLoader className="w-2.5 h-2.5 text-brand-light/60 animate-spin" />
                  ) : (
                    <LuDownload className="w-2.5 h-2.5 text-brand-light/50" />
                  )}
                </button>
                <span className="text-brand-light text-[12px] font-medium">{displayName}</span>
              </div>
              <span className="text-brand-light/60 text-[10px] font-mono bg-brand-background px-2 py-0.5 rounded">{typeLabel}</span>
            </div>
            {isExpanded ? <LuChevronDown className="w-4 h-4 text-brand-light/60" /> : <LuChevronRight className="w-4 h-4 text-brand-light/60" />}
          </div>
        ) : (
          <div className="w-full flex items-center justify-between p-3">
            <div className="flex items-center gap-x-2 justify-between w-full">
              <div className="flex items-center gap-x-2.5">
                <div className={cn(
                  "flex items-center justify-center w-4.5 h-4.5 rounded-full border",
                  componentFlagDownloaded 
                    ? "bg-green-500/20 border-green-500/40" 
                    : "bg-brand-background border-brand-light/20"
                )}>
                  {componentFlagDownloaded ? (
                    <LuCheck className="w-3 h-3 text-green-400" />
                  ) : (
                    <LuDownload className="w-2.5 h-2.5 text-brand-light/50" />
                  )}
                </div>
                <span className="text-brand-light text-[12px] font-medium">{displayName}</span>
              </div>
              <span className="text-brand-light/60 text-[10px] font-mono bg-brand-background px-2 py-0.5 rounded">{typeLabel}</span>
            </div>
          </div>
        );
      })()}

      {isExpanded && (
        <div className="px-4 pb-4">
          {modelPaths.length > 0 && (
            <div className="space-y-2 mt-3">
              {modelPaths.map((item, idx) => {
                const pathItem = typeof item === 'string' ? { path: item } : item as ManifestComponentModelPathItem;
                return (
                  <div key={idx} className="bg-brand-background border border-brand-light/10 rounded-md  p-3 overflow-hidden w-full">
                    {pathItem.variant && (
                      <div className="flex flex-row justify-between items-center mb-2.5">
                        <div className="text-brand-light text-[11px] break-all font-semibold">{pathItem.variant.toLowerCase().includes('default') ? 'Default' : pathItem.variant.toLowerCase().includes('gguf') ? pathItem.variant.replace('GGUF_', '').replace('Q', 'q').toUpperCase() : pathItem.variant}</div>
                        {typeof (pathItem as any).file_size === 'number' && (pathItem as any).file_size > 0 && (
                          <div className="flex-shrink-0 ml-2 text-[10px] text-brand-light/80 font-mono whitespace-nowrap">
                            {formatBytes((pathItem as any).file_size, 1)}
                          </div>
                        )}
                      </div>
                    )}
                    <div className="flex items-start justify-between gap-x-2 ">
                      <div className="flex-1 min-w-0 flex-row items-center gap-x-2">
                        <div className="text-brand-light text-[10.5px] font-medium mb-1">Model Path</div>
                        <div className="text-brand-light text-[10px] font-mono break-all">{pathItem.path}</div>
                      </div>
                    </div>
                    {(pathItem.type || pathItem.precision) && (
                      <div className="flex flex-col items-start  mt-2 justify-start border-t border-brand-light/5  pt-2">
                        <h4 className="text-brand-light text-[10.5px] font-medium mb-1">Model specifications</h4>
                        {pathItem.type && (
                          <div className="text-[10px] flex flex-row items-center gap-x-1 ">
                            <span className="text-brand-light/60 font-medium">Model Type </span>
                            <span className="text-brand-light/80 font-mono">{pathItem.type === 'gguf' ? 'GGUF' : formatComponentName(pathItem.type)}</span>
                          </div>
                        )}
                        {pathItem.precision && (
                          <div className="text-[10px] flex flex-row items-center gap-x-1 ">
                            <span className="text-brand-light/60 font-medium">Precision </span>
                            <span className="text-brand-light/90 font-mono">{pathItem.precision.toUpperCase()}</span>
                          </div>
                        )}
                      </div>
                    )}

                    {pathItem.resource_requirements && (
                      <div className="mt-2 pt-2 border-t border-brand-light/5">
                        <div className="text-brand-light text-[10.5px] font-medium mb-1">Resource Requirements</div>
                        <div className="flex flex-col ">
                          {pathItem.resource_requirements.min_vram_gb && (
                            <div className="text-[10px]">
                              <span className="text-brand-light/60 font-medium">Min VRAM </span>
                              <span className="text-brand-light/90">{pathItem.resource_requirements.min_vram_gb}GB</span>
                            </div>
                          )}
                          {pathItem.resource_requirements.recommended_vram_gb && (
                            <div className="text-[10px]">
                              <span className="text-brand-light/60 font-medium">Recommended VRAM </span>
                              <span className="text-brand-light/90">{pathItem.resource_requirements.recommended_vram_gb}GB</span>
                            </div>
                          )}
                          {pathItem.resource_requirements.compute_capability && (
                            <div className="text-[10px]">
                              <span className="text-brand-light/60">Compute Capability: </span>
                              <span className="text-brand-light/90">{pathItem.resource_requirements.compute_capability}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    {!downloadedPaths.has(pathItem.path) ? (
                        (() => {
                          const wsFilesObj = wsFilesByPath[pathItem.path] || {};
                          const wsFiles = Object.entries(wsFilesObj).map(([filename, v]) => ({
                            filename,
                            downloadedBytes: v.downloadedBytes,
                            totalBytes: v.totalBytes,
                            status: v.status,
                            progress: v.progress,
                            message: v.message,
                            bucket: v.bucket,
                            label: v.label,
                          })) as any[];
                          const isDownloading = wsFiles.length > 0;
                          return isDownloading;
                        })() ? (
                        <div className="w-full mt-3">
                          {(() => {
                            const wsFilesObj = wsFilesByPath[pathItem.path] || {};
                            const wsFiles = Object.entries(wsFilesObj).map(([filename, v]) => ({
                              filename,
                              downloadedBytes: v.downloadedBytes,
                              totalBytes: v.totalBytes,
                              status: v.status,
                              progress: v.progress,
                              message: v.message,
                              bucket: v.bucket,
                              label: v.label,
                            })) as any[];
                            const files = wsFiles.length > 0 ? wsFiles : [] as any[];
                            if (files.length === 0) return null;
                            return (
                              <div className="flex flex-col gap-y-2">
                                {files.map((f: any) => (
                                  <div key={f.filename} className="flex flex-col gap-y-1">
                                    <div className="flex items-center justify-between gap-x-2 w-full">
                                      <div className="flex-1 min-w-0">
                                        <div style={{ maxWidth: `${(componentCarRef.current?.clientWidth || 0) - 120}px` }} className="text-[10px] text-brand-light/80 font-mono truncate break-all">{f.filename}</div>
                                      </div>
                                      <div className="text-[10px] text-brand-light/80 font-mono flex-shrink-0">
                                        {(() => {
                                          const pct = f.totalBytes ? ((f.downloadedBytes || 0) / f.totalBytes) * 100 : (typeof f.progress === 'number' ? f.progress * 100 : 0);
                                          return `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`;
                                        })()}
                                      </div>
                                    </div>
                                    <ProgressBar percent={(() => {
                                      const pct = f.totalBytes ? ((f.downloadedBytes || 0) / f.totalBytes) * 100 : (typeof f.progress === 'number' ? f.progress * 100 : 0);
                                      return Math.max(0, Math.min(100, pct));
                                    })()} />
                                    <div className="flex items-center justify-between">
                                      {typeof f.downloadedBytes === 'number' && typeof f.totalBytes === 'number' ? (
                                        <div className="text-[10px] text-brand-light/90">
                                          {formatDownloadProgress(f.downloadedBytes, f.totalBytes)}
                                        </div>
                                      ) : <div />}
                                      {f.status === 'completed' || f.status === 'complete' ? (
                                        <div className="text-[10px] text-green-400">Completed</div>
                                      ) : (
                                        <div className="text-[9px] text-brand-light/60">
                                          {(f.downloadSpeed != null && f.downloadSpeed > 0 ? formatSpeed(f.downloadSpeed) : '')}
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            );
                          })()}
                          <div className="flex flex-col items-center justify-between mt-2 w-full">
                            <button
                              onClick={() => handleCancel(
                                pathToJobId[pathItem.path]
                              )}
                              className={cn("text-[10px] text-brand-light/90 w-full mt-2 font-medium hover:text-brand-light transition-all duration-200 bg-brand hover:bg-brand/70 border border-brand-light/10 rounded-[6px] px-2 py-2", )}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <button
                          onClick={() => handleDownload(pathItem.path)}
                          className="w-full mt-3 text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-md px-3 py-2 transition-all"
                        >
                          {downloadingPaths.has(pathItem.path) ? (
                            <>
                              <LuLoader className="w-3.5 h-3.5 animate-spin" />
                              <span>Downloading...</span>
                            </>
                          ) : (
                            <>
                              <LuDownload className="w-3.5 h-3.5" />
                              <span>Download Model</span>
                            </>
                          )}
                        </button>
                      )
                    ) : (
                      <div className="flex flex-row items-center justify-between gap-x-2">
                        <div className="text-[11px] font-medium text-brand-light/90 mt-4 mb-1.5 flex items-center justify-start gap-x-1">
                          <LuCheck className="w-3 h-3 text-green-400" />
                          <span>Downloaded</span>
                        </div>
                        <button onClick={() => handleDelete(pathItem.path)} disabled={deletingPaths.has(pathItem.path)} className="w-fit mt-3 text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all">
                          {deletingPaths.has(pathItem.path) ? (
                            <LuLoader className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <LuTrash className="w-3.5 h-3.5" />
                          )}
                          <span>{deletingPaths.has(pathItem.path) ? 'Deleting...' : 'Delete Model'}</span>
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          {isConfigOnly && component.type !== 'scheduler' && (
            <div className="space-y-2 mt-3">
              <div className="bg-brand-background border border-brand-light/10 rounded-md p-3 w-full">
                <div className="text-brand-light text-[10.5px] font-medium mb-1.5">Config Path</div>
                {baseConfigPath && (
                  <div className="text-[10px] text-brand-light/80 font-mono break-all">{baseConfigPath}</div>
                )}
                {!componentFlagDownloaded ? (
                  (() => {
                    const wsFilesObj = wsFilesByPath[baseConfigPath || ''] || {};
                    const wsFiles = Object.entries(wsFilesObj).map(([filename, v]) => ({
                      filename,
                      downloadedBytes: v.downloadedBytes,
                      totalBytes: v.totalBytes,
                      status: v.status,
                      progress: v.progress,
                      message: v.message,
                      bucket: v.bucket,
                      label: v.label,
                    })) as any[];
                    const isDownloading = wsFiles.length > 0;
                    if (isDownloading) {
                      const files = wsFiles.length > 0 ? wsFiles : [] as any[];
                      if (files.length === 0) {
                        return (
                          <div className="w-full mt-3" />
                        );
                      }
                      return (
                        <div className="w-full mt-3">
                          <div className="flex flex-col gap-y-2">
                            {files.map((f: any) => (
                              <div key={f.filename} className="flex flex-col gap-y-1">
                                <div className="flex items-center justify-between gap-x-2 w-full">
                                  <div className="flex-1 min-w-0">
                                    <div style={{ maxWidth: `${(componentCarRef.current?.clientWidth || 0) - 120}px` }} className="text-[10px] text-brand-light/80 font-mono truncate break-all">{f.filename}</div>
                                  </div>
                                  <div className="text-[10px] text-brand-light/80 font-mono flex-shrink-0">
                                    {(() => {
                                      const pct = f.totalBytes ? ((f.downloadedBytes || 0) / f.totalBytes) * 100 : (typeof f.progress === 'number' ? f.progress * 100 : 0);
                                      return `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`;
                                    })()}
                                  </div>
                                </div>
                                <ProgressBar percent={(() => {
                                  const pct = f.totalBytes ? ((f.downloadedBytes || 0) / f.totalBytes) * 100 : (typeof f.progress === 'number' ? f.progress * 100 : 0);
                                  return Math.max(0, Math.min(100, pct));
                                })()} />
                              </div>
                            ))}
                          </div>
                          <div className="flex flex-col items-center justify-between mt-2 w-full">
                            <button
                              onClick={() => baseConfigPath && handleCancel(baseConfigPath)}
                              className="text-[10px] text-brand-light/90 w-full mt-2 font-medium hover:text-brand-light transition-all duration-200 bg-brand hover:bg-brand/70 border border-brand-light/10 rounded-[6px] px-2 py-2"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      );
                    }
                    return (
                      <button
                        onClick={() => baseConfigPath && handleDownload(baseConfigPath)}
                        className="w-full mt-3 text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-md px-3 py-2 transition-all"
                      >
                        {downloadingPaths.has(baseConfigPath || '') ? (
                          <LuLoader className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <LuDownload className="w-3.5 h-3.5" />
                        )}
                        <span>{downloadingPaths.has(baseConfigPath || '') ? 'Downloading...' : 'Download Config'}</span>
                      </button>
                    );
                  })()
                ) : (
                  <div className="flex flex-row items-center justify-between gap-x-2">
                    <div className="text-[11px] font-medium text-brand-light/90 mt-4 mb-1.5 flex items-center justify-start gap-x-1">
                      <LuCheck className="w-3 h-3 text-green-400" />
                      <span>Downloaded</span>
                    </div>
                    <button onClick={() => baseConfigPath && handleDelete(baseConfigPath)} disabled={deletingPaths.has(baseConfigPath || '')} className="w-fit mt-3 text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all">
                      {deletingPaths.has(baseConfigPath || '') ? (
                        <LuLoader className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <LuTrash className="w-3.5 h-3.5" />
                      )}
                      <span>{deletingPaths.has(baseConfigPath || '') ? 'Deleting...' : 'Delete Config'}</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
          {component.type === 'scheduler' && component.scheduler_options && component.scheduler_options.length > 0 && (
            <div className="mt-3">
              <div className="text-brand-light/80 text-[11px] font-medium mb-2">Scheduler Options</div>
              <div className={cn("space-y-0 bg-brand-background border  border-b-0 border-brand-light/10 rounded-t-[6px] divide-y divide-brand-light/10", {
                "rounded-b-[6px] border-b": componentFlagDownloaded
              })}>
                {component.scheduler_options.map((option, idx) => (
                  <div key={idx} className={cn(" border-brand-light/10 border-t-border-x px-3.5 py-2")}>
                    <div className="mb-1">
                      <span className="text-brand-light text-[11px] font-medium">{option.label || option.name}</span>
                    </div>
                    {option.description && (
                      <div className="text-[10px] text-brand-light/70 mt-1 mb-1.5 ">
                        {option.description}
                      </div>
                    )}
                    
                  </div>
                ))}
              </div>
              {schedulerConfigPaths.length > 0 && (
                <div className={cn("bg-brand-background p-3 rounded-b-[6px]", {
                  "p-0": !schedulersConfigDownloading
                })}>
                  {schedulerConfigPaths.map((p) => {
                    const wsFilesObj = wsFilesByPath[p] || {};
                    const wsFiles = Object.entries(wsFilesObj).map(([filename, v]) => ({
                      filename,
                      downloadedBytes: v.downloadedBytes,
                      totalBytes: v.totalBytes,
                      status: v.status,
                      progress: v.progress,
                      message: v.message,
                      bucket: v.bucket,
                      label: v.label,
                    }));
                    const files = wsFiles.length > 0 ? wsFiles : [];
                    if (!(files.length > 0)) return null;
                    return (
                      <div key={p} className="w-full">
                        <div className="text-brand-light text-[10.5px] font-medium mb-2.5">Config Download</div>
                        {files.length > 0 ? (
                          <div className="flex flex-col gap-y-2">
                            {files.map((f: any) => (
                              <div key={f.filename} className="flex flex-col gap-y-1">
                                <div className="flex flex-col justify-start gap-y-2 w-full">
                                  <div className="flex-1 min-w-0">
                                    <div style={{ maxWidth: `${(componentCarRef.current?.clientWidth || 0) - 120}px` }} className="text-[10px] text-brand-light/80 font-mono truncate break-all">{f.filename}</div>
                                  </div>
                                  <div className="text-[10px] text-brand-light/80 font-mono flex items-center gap-x-1">
                                    <LuLoader className="w-3 h-3 text-brand-light/60 animate-spin" />
                                    <span>{typeof f.message === 'string' && f.message ? f.message : 'Preparing...'}</span>
                                  </div>
                                </div>
                                <ProgressBar percent={(() => {
                                  const pct = f.totalBytes ? ((f.downloadedBytes || 0) / f.totalBytes) * 100 : (typeof f.progress === 'number' ? f.progress * 100 : 0);
                                  return Math.max(0, Math.min(100, pct));
                                })()} />
                                <div className="flex items-center justify-between">
                                  {typeof f.downloadedBytes === 'number' && typeof f.totalBytes === 'number' ? (
                                    <div className="text-[10px] text-brand-light/90">
                                      {formatDownloadProgress(f.downloadedBytes, f.totalBytes)}
                                    </div>
                                  ) : <div />}
                                  {f.status === 'completed' || f.status === 'complete' ? (
                                    <div className="text-[10px] text-green-400">Completed</div>
                                  ) : (
                                    <div className="text-[9px] text-brand-light/60">
                                      {(f.downloadSpeed != null && f.downloadSpeed > 0 ? formatSpeed(f.downloadSpeed) : '')}
                                    </div>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="flex flex-col items-start justify-start gap-y-2 w-full">
                            <div className="flex-1 min-w-0">
                              <div style={{ maxWidth: `${(componentCarRef.current?.clientWidth || 0) - 40}px` }} className="text-[10px] text-brand-light/80 font-mono truncate break-all">{p}</div>
                            </div>
                            <div className="text-[10px] text-brand-light/80 font-mono flex items-center gap-x-1 justify-start">
                              <LuLoader className="w-3 h-3 text-brand-light/60 animate-spin" />
                              <span>Preparing...</span>
                            </div>
                          </div>
                        )}
                        <div className="flex flex-col items-center justify-between mt-2 w-full">
                          <button
                            onClick={() => handleCancel(p)}
                            className="text-[10px] text-brand-light/90 w-full mt-2 font-medium hover:text-brand-light transition-all duration-200 bg-brand hover:bg-brand/70 border border-brand-light/10 rounded-[6px] px-2 py-2"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              {!componentFlagDownloaded && (
                <div className="bg-brand-background p-3 rounded-b-[6px] border border-t-0 border-brand-light/10">
                  <button
                    onClick={() => handleDownload(schedulerConfigPaths)}
                    className="w-full text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light  hover:bg-brand/70 border border-brand-light/10 rounded-[6px] bg-brand px-3 py-2 transition-all"
                  >
                    {schedulerConfigPaths.some((p) => downloadingPaths.has(p)) || schedulerIsDownloading ? (
                      <LuLoader className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <LuDownload className="w-3.5 h-3.5" />
                    )}
                    <span>{schedulerConfigPaths.some((p) => downloadingPaths.has(p)) || schedulerIsDownloading ? 'Downloading...' : 'Download Config'}</span>
                  </button>
                </div>
              )}
              {componentFlagDownloaded && schedulerConfigPaths.length > 0 && (
                <div className="bg-brand-background rounded-[6px] mt-2 py-1 border  border-brand-light/10">
                  <div className="space-y-2 divide-y divide-brand-light/10">
                    {schedulerConfigPaths.map((p) => (
                      <div key={p} className="flex flex-col items-start justify-start gap-y-2 w-full py-2 px-3">
                        <div className="flex-1 min-w-0">
                          <p className="text-[10px] text-brand-light font-medium mb-1"> 
                            Config Path
                          </p>
                          <div style={{ maxWidth: `${(componentCarRef.current?.clientWidth || 0) - 60}px` }} className="text-[10px] text-brand-light/80 font-mono truncate break-all">{p}</div>
                        </div>
                        <div className="flex flex-row items-center justify-between gap-x-2 w-full">
                          <div className="text-[11px] font-medium text-brand-light/90 flex items-center justify-start gap-x-1">
                            <LuCheck className="w-3 h-3 text-green-400" />
                            <span>Downloaded</span>
                          </div>
                          <button onClick={() => handleDelete(p)} className="w-fit text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[5px] px-2.5 py-1.5 transition-all">
                            {deletingPaths.has(p) ? (
                              <LuLoader className="w-3 h-3 animate-spin" />
                            ) : (
                              <LuTrash className="w-3 h-3" />
                            )}
                            <span>{deletingPaths.has(p) ? 'Deleting...' : 'Delete'}</span>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ComponentCard;


