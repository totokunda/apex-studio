import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useClipStore } from "@/lib/clip";
import { ModelClipProps } from "@/lib/types";
import { PiCubeDuotone } from "react-icons/pi";
import { LuPlus, LuChevronDown, LuPencil, LuTrash2 } from "react-icons/lu";
import {
  LoraType,
  updateManifestLoraScale,
  updateManifestLoraName,
  deleteManifestLora,
  type ManifestComponent,
} from "@/lib/manifest/api";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import PropertiesSlider from "@/components/properties/PropertiesSlider";
import { LuLoader } from "react-icons/lu";
import { toast } from "sonner";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useStartUnifiedDownloadMutation } from "@/lib/download/mutations";
import { useDownloadJobIdStore } from "@/lib/download/job-id-store";
import {
  connectUnifiedDownloadWebSocket,
  disconnectUnifiedDownloadWebSocket,
  onUnifiedDownloadError,
  onUnifiedDownloadUpdate,
  type UnifiedDownloadWsUpdate,
} from "@/lib/download/api";
import { unifiedDownloadWsUpdatesToFiles } from "@/lib/download/ws-updates-to-files";
import { ProgressBar } from "@/components/common/ProgressBar";
import { formatDownloadProgress, formatSpeed } from "@/lib/download/format";
import { cancelRayJob, fetchRayJobs, type RayJobStatus } from "@/lib/jobs/api";
import {
  refreshManifestPart,
  useManifestQuery,
} from "@/lib/manifest/queries";
import { cn } from "@/lib/utils";

interface LoraPanelProps {
  clipId: string;
  panelSize: number;
}

const isLoraDownloading = (item: any): boolean => {
  if (typeof item === "string") return true;
  if (typeof item !== "object" || item == null) return false;
  return !(item as any).is_downloaded;
};

const POLL_MS = 2000;
const FAST_POLL_MS = 500;
const RAY_JOBS_QUERY_KEY = ["rayJobs"] as const;
const STARTUP_TIMEOUT_MS = 15_000;
const LORA_VERIFICATION_FAILED_MSG = "LoRA verification failed; removed from manifest";

const DownloadProgressSection: React.FC<{
  jobUpdates: UnifiedDownloadWsUpdate[];
  onCancel: () => void;
  width: number;
  title?: string;
}> = ({ jobUpdates, onCancel, width, title }) => {
  const files = useMemo(
    () => unifiedDownloadWsUpdatesToFiles(jobUpdates),
    [jobUpdates],
  );
  if (!files.length) return null;
  // (latest update currently unused; keep progress purely file-based)

  const displayFiles = useMemo(() => {
    // If we collapsed unlabeled backend updates into a single "download" row,
    // show a more meaningful name in the UI.
    if (files.length === 1) {
      const f = files[0] as any;
      const baseName =
        (typeof title === "string" && title.trim()) ||
        (typeof f?.label === "string" && f.label.trim()) ||
        (typeof f?.filename === "string" && f.filename.trim()) ||
        "Download";

      const shouldReplace =
        f?.filename === "download" ||
        f?.filename === "lora" ||
        f?.filename === "component" ||
        f?.filename === "preprocessor" ||
        /^file-\\d+$/i.test(String(f?.filename || ""));

      if (shouldReplace) {
        return [{ ...f, filename: baseName }];
      }
    }
    return files;
  }, [files, title]);


  return (
    <div
      className="mt-2 overflow-hidden"
      style={{ width: `${width}px`, maxWidth: `${width}px` }}
    >
      <div className="flex flex-col gap-y-2">
        {displayFiles.map((f: any) => {
          const pct = f.totalBytes
            ? ((f.downloadedBytes || 0) / f.totalBytes) * 100
            : typeof f.progress === "number"
              ? f.progress * 100
              : 0;
          const clamped = Math.max(0, Math.min(100, pct));

          return (
            <div
              key={f.filename}
              className="flex flex-col gap-y-1 min-w-0"
              style={{ width: `${width}px`, maxWidth: `${width}px` }}
            >
              <div className="flex items-center justify-between gap-x-2">
                <div className="text-[10px] text-brand-light/80 font-mono truncate break-all flex-1 min-w-0 text-start">
                  {f.filename}
                </div>
                <div className="text-[10px] text-brand-light/80 font-mono shrink-0">
                  {clamped.toFixed(1)}%
                </div>
              </div>
              <ProgressBar percent={clamped} />
              <div className="flex items-center justify-between">
                {typeof f.downloadedBytes === "number" &&
                typeof f.totalBytes === "number" ? (
                  <div className="text-[10px] text-brand-light/90">
                    {formatDownloadProgress(f.downloadedBytes, f.totalBytes)}
                  </div>
                ) : (
                  <div />
                )}
                <div className="text-[9px] text-brand-light/60">
                  {f.status === "completed" || f.status === "complete"
                    ? "Completed"
                    : formatSpeed(f.downloadSpeed) || ""}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <button
        type="button"
        onClick={onCancel}
        className="text-[10px] text-brand-light/90 w-full flex items-center justify-center gap-x-1.5 mt-2 font-medium bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded-[6px] px-2 py-2 transition-all"
      >
        <span>Cancel</span>
      </button>
    </div>
  );
};

const LoraDownloadRow: React.FC<{
  item: any;
  index: number;
  manifestId: string;
  width: number;
  waitingForJob?: boolean;
}> = ({ item, index: _index, manifestId: _manifestId, width, waitingForJob }) => {
  
  const path =
    typeof item === "string"
      ? item
      : (item as any).source || (item as any).remote_source;
  const label =
    typeof item === "string"
      ? path
        ? path.split("/").pop() || path
        : "LoRA"
      : (item as any).label ||
        (item as any).name ||
        (path ? path.split("/").pop() || path : "LoRA");
  const isDownloaded = typeof item === "string" ? false : !!(item as any).is_downloaded;
  const [startDownloading, setStartDownloading] = useState(false);

  const {
    getSourceToJobId,
    getJobUpdates,
    removeJobUpdates,
    removeSourceByJobId,
    removeSourceToJobId,
  } = useDownloadJobIdStore();



  const jobId = path ? getSourceToJobId(path) : undefined;
  const jobUpdates = getJobUpdates(jobId);
  const isDownloading = (jobUpdates?.length ?? 0) > 0;
  const lastUpdate = jobUpdates?.[jobUpdates.length - 1];
  const lastMsg =
    (typeof lastUpdate?.message === "string" ? lastUpdate.message : "") ||
    (typeof lastUpdate?.status === "string" ? lastUpdate.status : "");
  const isVerifyingStep = /verif/i.test(lastMsg)
  const verified: boolean | undefined =
    typeof item === "object" && item != null && "verified" in item
      ? !!(item as any).verified
      : undefined;

  useEffect(() => {
    if (jobId && !jobUpdates?.length) setStartDownloading(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (isDownloading || !jobId) setStartDownloading(false);
  }, [isDownloading, jobId]);

  // If we never receive any WS updates for this job, revert back to the idle state after a timeout.
  useEffect(() => {
    if (!jobId) return;
    if (!startDownloading) return;
    if ((jobUpdates?.length ?? 0) > 0) return;

    const t = setTimeout(() => {
      setStartDownloading(false);
      // If the job never actually started, remove stale mapping so it doesn't linger forever.
      try {
        removeJobUpdates(jobId);
      } catch {}
      try {
        removeSourceByJobId(jobId);
      } catch {}
      try {
        if (path) removeSourceToJobId(path);
      } catch {}
    }, STARTUP_TIMEOUT_MS);

    return () => clearTimeout(t);
  }, [
    jobId,
    jobUpdates?.length,
    path,
    removeJobUpdates,
    removeSourceByJobId,
    removeSourceToJobId,
    startDownloading,
  ]);

  if (!path) return null;
  const isWaiting = !!waitingForJob || startDownloading;

  return (
    <div className="w-full bg-brand border border-brand-light/10 rounded-md px-3 py-2.5">
      <div className="flex items-start justify-between gap-x-3">
        <div className="min-w-0 flex-1">
          <div className="text-brand-light text-[11px] font-medium truncate break-all text-start">
            {label}
          </div>
          <div className="text-[10px] text-brand-light/70 font-mono text-start mt-0.5">
            {path}
          </div>
        </div>
        <div className="flex items-center gap-x-1.5 shrink-0">
          {verified !== undefined || isVerifyingStep ? (
            <span
              className={
                verified === true
                  ? "px-2 py-0.5 rounded-full border border-green-500/40 bg-green-500/15 text-[9px] font-medium text-green-300" : "px-2 py-0.5 rounded-full border border-blue-500/40 bg-blue-500/15 text-[9px] font-medium text-blue-300"
              }
            >
              {verified === true
                ? "Verified"
                :"Verifying"}
            </span>
          ) : null}
        </div>
      </div>

      {!isDownloaded && (
        <>
          {isDownloading && (
            <DownloadProgressSection
              jobUpdates={jobUpdates ?? []}
              onCancel={async () => {
                if (!jobId) return;
                try {
                  await cancelRayJob(jobId);
                } catch {}
                try {
                  removeJobUpdates(jobId);
                } catch {}
                try {
                  removeSourceByJobId(jobId);
                } catch {}
                try {
                  window.dispatchEvent(
                    new CustomEvent("jobs-menu-reload", { detail: { jobId } }),
                  );
                } catch {}
              }}
              width={width}
              title={label}
            />
          )}
          {!isDownloading && isWaiting && (
            <div className="mt-2 flex items-center justify-start gap-x-2 text-[10px] text-brand-light/70">
              <LuLoader className="w-3.5 h-3.5 animate-spin text-brand-light/60" />
              <span>Starting download…</span>
            </div>
          )}
        </>
      )}
    </div>
  );
};

const LoraItem: React.FC<{
  item: LoraType;
  manifestId: string;
  index: number;
}> = ({ item, manifestId, index }) => {
  const isObject = typeof item !== "string";
  const remoteSource = useMemo(
    () => (isObject ? (item as any).remote_source || "" : ""),
    [isObject, item],
  );

  const path = useMemo(() => {
    if (!isObject) return item as string;
    const obj = item;
    return obj.source || remoteSource || "";
  }, [isObject, item, remoteSource]);

  const label = useMemo(() => {
    if (isObject) {
      const obj = item as any;
      return (
        obj.label || obj.name || (path ? path.split("/").pop() || path : "LoRA")
      );
    }
    return path ? path.split("/").pop() || path : "LoRA";
  }, [isObject, item, path]);

  const verified = isObject ? !!(item as any).verified : false;
  const initialScale = useMemo(() => {
    if (isObject && typeof (item as any).scale === "number") {
      return Math.max(0, Math.min(1, (item as any).scale as number));
    }
    return 1.0;
  }, [isObject, item]);
  const [scale, setScale] = useState<number>(initialScale);
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isPathOpen, setIsPathOpen] = useState(false);
  const [isEditingName, setIsEditingName] = useState(false);
  const queryClient = useQueryClient();
  const [pendingName, setPendingName] = useState<string>(() => {
    if (isObject) {
      const obj = item as any;
      return obj.label || obj.name || "";
    }
    return "";
  });
  const [isSavingName, setIsSavingName] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [initialEditingName, setInitialEditingName] = useState("");

  const displayPath = useMemo(() => {
    if (remoteSource) return remoteSource;
    return path;
  }, [remoteSource, path]);


  useEffect(() => {
    setScale(initialScale);
  }, [initialScale]);

  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
        saveTimeoutRef.current = null;
      }
    };
  }, []);

  // Local / resolved LoRA – show name, path, scale slider and verified badge
  return (
    <div className="w-full bg-brand shadow-md border border-brand-light/5 rounded-md flex flex-col ">
      <div className="flex items-center justify-between gap-x-2 px-3 bg-brand py-2.5 rounded-t-md">
        <div className="flex items-center gap-x-2 min-w-0 flex-1">
          {isEditingName ? (
            <div className="flex items-center gap-x-1.5 w-full">
              <input
                type="text"
                value={pendingName}
                onChange={(e) => setPendingName(e.target.value)}
                disabled={isSavingName || isDeleting}
                className="flex-1 min-w-0 bg-brand-background border border-brand-light/20 rounded-[4px] px-2 py-1 text-[10px] text-brand-light placeholder:text-brand-light/40 focus:outline-none focus:ring-1 focus:ring-brand-light/40"
                placeholder="LoRA name"
              />
              <button
                type="button"
                disabled={!pendingName.trim() || isSavingName || isDeleting}
                onClick={async () => {
                  if (pendingName.trim() === initialEditingName) {
                    setIsEditingName(false);
                    return;
                  }
                  const name = pendingName.trim();
                  if (!name) return;
                  const manifestIdSafe =
                    manifestId ||
                    (isObject && (item as any)?.metadata?.id) ||
                    "";
                  if (!manifestIdSafe) return;
                  setIsSavingName(true);
                  try {
                    await updateManifestLoraName(manifestIdSafe, index, name);
                    try {
                      await refreshManifestPart(manifestIdSafe, "spec.loras", queryClient);
                    } catch {}
                    setIsEditingName(false);
                  } finally {
                    setIsSavingName(false);
                  }
                }}
                className="px-2 py-0.5 rounded-[4px] text-[10px] font-medium bg-brand-accent-shade text-brand-light border border-brand-light/20 hover:bg-brand-accent-two-shade disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSavingName ? "Saving…" : "Save"}
              </button>
              <button
                type="button"
                disabled={isSavingName || isDeleting}
                onClick={() => {
                  setIsEditingName(false);
                  if (isObject) {
                    const obj = item as any;
                    setPendingName(obj.label || obj.name || "");
                  } else {
                    setPendingName("");
                  }
                }}
                className="px-2 py-1 rounded-[4px] text-[10px] font-medium text-brand-light/70 hover:text-brand-light/90"
              >
                Cancel
              </button>
            </div>
          ) : (
            <>
              <div className=" w-fit">
                <span className="text-brand-light text-[11px] font-medium truncate text-start w-fit">
                  {label}
                </span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setIsEditingName(true);
                  setInitialEditingName(label);
                }}
                className={cn("shrink-0 p-1 rounded-[4px] border border-transparent hover:border-brand-light/30 hover:bg-brand-background/60 text-brand-light/80 hover:text-brand-light transition-colors", (isObject && (item as any)?.required) ? "hidden" : "")} 
              >
                <LuPencil className="w-3 h-3" />
              </button>
            </>
          )}
        </div>
        <div className="flex items-center gap-x-2">
          <span
            className={
              verified
                ? "px-1.5 py-0.5 rounded-full border border-green-500/40 bg-green-500/15 text-[9px] font-medium text-green-300"
                : "px-1.5 py-0.5 rounded-full border border-brand-light/20 bg-brand-background/60 text-[9px] font-medium text-brand-light/70"
            }
          >
            {verified ? "Verified" : "Unverified"}
          </span>
          <button
            type="button"
            disabled={isDeleting || isSavingName || (isObject && (item as any)?.required)}
            onClick={async () => {
              const manifestIdSafe =
                manifestId || (isObject && (item as any)?.metadata?.id) || "";
              if (!manifestIdSafe) return;
              setIsDeleting(true);
              try {
                await deleteManifestLora(manifestIdSafe, index);
                try {
                  await refreshManifestPart(manifestIdSafe, "spec.loras", queryClient);
                } catch {}
              } finally {
                setIsDeleting(false);
              }
            }}
            className={cn("shrink-0 p-1 rounded-[4px] bg-brand-background border border-brand-light/10 text-brand-light/70 hover:text-brand-light/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors", (isObject && (item as any)?.required) ? "hidden" : "")}
          >
            {isDeleting ? (
              <LuLoader className="w-3 h-3 animate-spin" />
            ) : (
              <LuTrash2 className="w-3 h-3" />
            )}
          </button>
        </div>
      </div>

      <div
        className={`flex flex-col gap-y-1 px-3 py-2.5 bg-brand border-t border-brand-light/10`}
      >
        <PropertiesSlider
          inputClass="bg-brand-background"
          buttonClass="bg-brand-background border-brand-light/10 border border-l-0"
          label="Scale"
          value={scale}
          onChange={async (next) => {
            const clamped = Math.max(0, Math.min(3, next));
            setScale(clamped);
            const manifestIdSafe =
              manifestId || (isObject && (item as any)?.metadata?.id) || "";
            if (!manifestIdSafe) return;
            if (saveTimeoutRef.current) {
              clearTimeout(saveTimeoutRef.current);
            }
            saveTimeoutRef.current = setTimeout(async () => {
              try {
                await updateManifestLoraScale(manifestIdSafe, index, clamped);
                try {
                  await refreshManifestPart(manifestIdSafe, "spec.loras", queryClient);
                } catch {}
              } catch {
                // Best-effort; UI already reflects local slider value
              }
            }, 300);
          }}
          min={0}
          max={3}
          step={0.01}
          toFixed={2}
        />
      </div>
      <div className="flex flex-col gap-y-1 border-t border-brand-light/10 bg-brand rounded-b-md">
        <button
          type="button"
          onClick={() => setIsPathOpen((prev) => !prev)}
          className="flex items-center justify-between py-2.5 px-3 text-brand-light text-[10.5px] font-medium text-start"
        >
          <span>LoRA Path</span>
          <LuChevronDown
            className={`w-3 h-3 text-brand-light/80 transition-transform ${
              isPathOpen ? "rotate-180" : "rotate-0"
            }`}
          />
        </button>
        {isPathOpen && (
          <div className="pb-2 px-3 -mt-2">
            <div className="text-[10px] text-brand-light/70 font-mono break-all text-start">
              {displayPath || "—"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const LoraPanel: React.FC<LoraPanelProps> = ({ clipId, panelSize }) => {
  const clip = useClipStore(
    useCallback(
      (s) => s.getClipById(clipId) as ModelClipProps | undefined,
      [clipId],
    ),
  );
  const [isAddingLora, setIsAddingLora] = useState(false);
  const [newLoraName, setNewLoraName] = useState("");
  const [newLoraSource, setNewLoraSource] = useState("");
  const [activeTab, setActiveTab] = useState<"downloads" | "installed">(
    "installed",
  );
  const sourceHelpShort = "Enter a CivitAI ID/URN, URL, or local path.";
  if (!clip || !clip.manifest) return null;
  const { data: manifest } = useManifestQuery(clip.manifest.id);
  const queryClient = useQueryClient();
  const [startingSources, setStartingSources] = useState<
    Array<{ source: string; manifestId: string; startedAt: number }>
  >([]);
  const [finalizingJobIds, setFinalizingJobIds] = useState<Set<string>>(
    () => new Set(),
  );
  const verificationToastShownRef = useRef<Set<string>>(new Set());
  const prevVerifiedInstalledCountRef = useRef<number | null>(null);
  const hadDownloadActivityRef = useRef(false);

  const isWaitingForDownloads =
    startingSources.length > 0 || finalizingJobIds.size > 0;

  // Pull the same polled job list as JobsMenu (shared react-query cache via queryKey).
  // When we are waiting on downloads, poll faster; otherwise back off.
  const { data: polledJobs = [] } = useQuery<RayJobStatus[]>({
    queryKey: RAY_JOBS_QUERY_KEY,
    queryFn: fetchRayJobs,
    placeholderData: (prev) => prev ?? [],
    retry: false,
    refetchOnWindowFocus: false,
    refetchInterval: isWaitingForDownloads ? FAST_POLL_MS : POLL_MS,
    refetchIntervalInBackground: true,
  });

  const sourceToJobId = useDownloadJobIdStore((s) => s.sourceToJobId);
  const jobIdToManifestId = useDownloadJobIdStore((s) => s.jobIdToManifestId);
  const subscribedRef = useRef<Map<string, () => void>>(new Map());
  const {
    addJobUpdate,
    getJobIdToParts,
    getJobIdToManifestId,
    getJobUpdates,
    removeJobIdToParts,
    removeJobIdToManifestId,
    removeJobUpdates,
    removeSourceByJobId,
    removeSourceToJobId,
  } = useDownloadJobIdStore();

  // Toast once if the backend reports a LoRA verification failure.
  useEffect(() => {
    const currentManifestId = manifest?.metadata?.id || clip.manifest.id || "";
    for (const j of polledJobs || []) {
      const jobId = j?.job_id;
      if (!jobId) continue;
      if (j.category !== "download") continue;

      const mappedManifestId = jobIdToManifestId?.[jobId];
      if (currentManifestId && mappedManifestId && mappedManifestId !== currentManifestId) continue;

      const msg =
        (typeof (j as any)?.latest?.message === "string" ? (j as any).latest.message : "") ||
        (typeof j.message === "string" ? j.message : "");

      if (msg === LORA_VERIFICATION_FAILED_MSG) {
        if (!verificationToastShownRef.current.has(jobId)) {
          verificationToastShownRef.current.add(jobId);
          toast.error("LoRA verification failed.");
        }
      }
    }
  }, [clip.manifest.id, jobIdToManifestId, manifest?.metadata?.id, polledJobs]);

  const pendingDownloadSources = useMemo(() => {
    const terminalStatuses = new Set([
      "complete",
      "completed",
      "cancelled",
      "canceled",
      "error",
      "failed",
    ]);

    const activeDownloadJobIds = new Set(
      (polledJobs || [])
        .filter((j) => {
          const s = (j.status || "").toLowerCase();
          const isActive = !terminalStatuses.has(s);
          return isActive && j.category === "download" && j.job_id;
        })
        .map((j) => j.job_id as string),
    );
    const visibleJobIds = new Set<string>([
      ...activeDownloadJobIds,
      ...Array.from(finalizingJobIds),
    ]);

    const currentManifestId = manifest?.metadata?.id || clip.manifest.id;
    if (!currentManifestId) return [];

    // Invert the (source -> jobId) map for active jobIds, filtered by manifestId.
    return Object.keys(sourceToJobId || {}).filter((source) => {
      const jobId = sourceToJobId[source];
      if (!jobId || !visibleJobIds.has(jobId)) return false;
      return jobIdToManifestId?.[jobId] === currentManifestId;
    });
  }, [
    clip.manifest.id,
    finalizingJobIds,
    jobIdToManifestId,
    manifest?.metadata?.id,
    polledJobs,
    sourceToJobId,
  ]);

  // Track whether we have (or recently had) download activity; used to avoid auto-switching on initial load.
  useEffect(() => {
    if (pendingDownloadSources.length > 0 || startingSources.length > 0) {
      hadDownloadActivityRef.current = true;
    }
  }, [pendingDownloadSources.length, startingSources.length]);

  // Subscribe to active download jobs discovered by polling so we get step-by-step updates.
  useEffect(() => {
    const terminalStatuses = new Set([
      "complete",
      "completed",
      "cancelled",
      "canceled",
      "error",
      "failed",
    ]);
    const currentManifestId = manifest?.metadata?.id || clip.manifest.id || "";

    const activeJobIds = new Set(
      (polledJobs || [])
        .filter((j) => {
          const s = (j.status || "").toLowerCase();
          const isActive = !terminalStatuses.has(s);
          return isActive && j.category === "download" && j.job_id;
        })
        .map((j) => j.job_id as string),
    );

    const terminalJobIds = new Set(
      (polledJobs || [])
        .filter((j) => {
          const s = (j.status || "").toLowerCase();
          const isTerminal = terminalStatuses.has(s);
          return isTerminal && j.category === "download" && j.job_id;
        })
        .map((j) => j.job_id as string),
    );

    const cleanupJob = async (jobId: string) => {
      try {
        const parts = getJobIdToParts(jobId);
        const manifestId =
          getJobIdToManifestId(jobId) || currentManifestId || null;
        if (parts?.length && manifestId) {
          await Promise.all(
            parts.map((part) => refreshManifestPart(manifestId, part, queryClient)),
          );
        }
      } catch {}
      try {
        removeJobIdToManifestId(jobId);
      } catch {}
      try {
        removeJobIdToParts(jobId);
      } catch {}
      try {
        removeJobUpdates(jobId);
      } catch {}
      try {
        removeSourceByJobId(jobId);
      } catch {}
    };

    // Connect to new jobs (only those tied to this manifest via the store mapping).
    activeJobIds.forEach((jobId) => {
      const mappedManifestId = jobIdToManifestId?.[jobId];
      if (currentManifestId && mappedManifestId !== currentManifestId) return;
      if (subscribedRef.current.has(jobId)) return;

      const setup = async () => {
        try {
          await connectUnifiedDownloadWebSocket(jobId);
          const unsubUpdate = onUnifiedDownloadUpdate(jobId, async (data) => {
            addJobUpdate(jobId, data);
            const msg = typeof (data as any)?.message === "string" ? (data as any).message : "";
            if (msg === LORA_VERIFICATION_FAILED_MSG) {
              if (!verificationToastShownRef.current.has(jobId)) {
                verificationToastShownRef.current.add(jobId);
                toast.error("LoRA verification failed.");
              }
            }
            const s = (data?.status || "").toLowerCase();
            if (terminalStatuses.has(s)) {
              setFinalizingJobIds((prev) => {
                const next = new Set(prev);
                next.add(jobId);
                return next;
              });
              try {
                await cleanupJob(jobId);
              } finally {
                setFinalizingJobIds((prev) => {
                  const next = new Set(prev);
                  next.delete(jobId);
                  return next;
                });
                subscribedRef.current.get(jobId)?.();
                subscribedRef.current.delete(jobId);
              }
            }
          });

          const unsubError = onUnifiedDownloadError(jobId, async (data) => {
            addJobUpdate(jobId, {
              status: "failed",
              message: data?.error || "Unknown error",
              metadata: {},
            } as any);
            setFinalizingJobIds((prev) => {
              const next = new Set(prev);
              next.add(jobId);
              return next;
            });
            try {
              await cleanupJob(jobId);
            } finally {
              setFinalizingJobIds((prev) => {
                const next = new Set(prev);
                next.delete(jobId);
                return next;
              });
              subscribedRef.current.get(jobId)?.();
              subscribedRef.current.delete(jobId);
            }
          });

          const cleanup = () => {
            unsubUpdate();
            unsubError();
            disconnectUnifiedDownloadWebSocket(jobId).catch(() => {});
          };
          subscribedRef.current.set(jobId, cleanup);
        } catch {
          // ignore; polling will keep list alive and we can retry next tick
        }
      };
      setup();
    });

    // Cleanup finished / irrelevant jobs
    for (const [jobId, cleanup] of subscribedRef.current.entries()) {
      const mappedManifestId = jobIdToManifestId?.[jobId];
      const isFinalizing = finalizingJobIds.has(jobId) || terminalJobIds.has(jobId);
      const stillRelevant =
        (activeJobIds.has(jobId) || isFinalizing) &&
        (!currentManifestId || mappedManifestId === currentManifestId);
      if (!stillRelevant) {
        cleanup();
        subscribedRef.current.delete(jobId);
      }
    }

    // If polling reports terminal status (but WS terminal update didn't arrive), finalize anyway.
    terminalJobIds.forEach((jobId) => {
      const mappedManifestId = jobIdToManifestId?.[jobId];
      if (currentManifestId && mappedManifestId !== currentManifestId) return;
      if (finalizingJobIds.has(jobId)) return;
      if (!subscribedRef.current.has(jobId)) return;
      setFinalizingJobIds((prev) => {
        const next = new Set(prev);
        next.add(jobId);
        return next;
      });
      (async () => {
        try {
          await cleanupJob(jobId);
        } finally {
          setFinalizingJobIds((prev) => {
            const next = new Set(prev);
            next.delete(jobId);
            return next;
          });
          subscribedRef.current.get(jobId)?.();
          subscribedRef.current.delete(jobId);
        }
      })();
    });
  }, [
    polledJobs,
    jobIdToManifestId,
    finalizingJobIds,
    manifest?.metadata?.id,
    clip.manifest.id,
    queryClient,
    addJobUpdate,
    getJobIdToParts,
    getJobIdToManifestId,
    removeJobIdToParts,
    removeJobIdToManifestId,
    removeJobUpdates,
    removeSourceByJobId,
  ]);

  useEffect(() => {
    return () => {
      for (const cleanup of subscribedRef.current.values()) {
        try {
          cleanup();
        } catch {}
      }
      subscribedRef.current.clear();
    };
  }, []);

  const { addSourceToJobId, addJobIdToParts, addJobIdToManifestId } =
    useDownloadJobIdStore();
  const { mutateAsync: startDownload, isPending: isStartingDownload } =
    useStartUnifiedDownloadMutation({
      onSuccess(data, variables) {
        addSourceToJobId(variables.source, data.job_id);
        addJobIdToParts(data.job_id, ["spec.loras"]);
        if (manifest?.metadata?.id) addJobIdToManifestId(data.job_id, manifest.metadata.id);
      },
    });

  const [newLoraComponent, setNewLoraComponent] = useState<string>("");

  const transformerComponents = useMemo(
    () => {
      const components = (manifest?.spec?.components ||
        []) as ManifestComponent[] | undefined;
      if (!Array.isArray(components)) return [] as Array<{ comp: ManifestComponent; index: number }>;
      return components
        .map((comp, index) => ({ comp, index }))
        .filter(
          ({ comp }) =>
            comp &&
            typeof comp === "object" &&
            (comp as any).type === "transformer",
        );
    },
    [manifest?.spec?.components],
  );

  const hasMultipleTransformers = transformerComponents.length > 1;

  const getTransformerComponentKey = useCallback(
    (entry: { comp: ManifestComponent; index: number }) => {
      const compAny = entry.comp as any;
      const key =
        (compAny?.name as string | undefined) ||
        (compAny?.base as string | undefined) ||
        `transformer-${entry.index}`;
      return key;
    },
    [],
  );

  useEffect(() => {
    if (!isAddingLora) {
      setNewLoraComponent("");
      return;
    }
    if (!newLoraComponent && transformerComponents.length > 0) {
      const firstKey = getTransformerComponentKey(transformerComponents[0]);
      setNewLoraComponent(firstKey);
    }
  }, [
    isAddingLora,
    newLoraComponent,
    transformerComponents,
    getTransformerComponentKey,
  ]);

  const handleAddLoraSource = async (
    source: string,
    name: string,
    component?: string,
  ) => {
    if (!source) return;
    if (!name) return;
    const currentManifestId = manifest?.metadata?.id || clip.manifest.id || "";
    if (currentManifestId) {
      setStartingSources((prev) => [
        { source, manifestId: currentManifestId, startedAt: Date.now() },
        ...prev.filter((p) => !(p.source === source && p.manifestId === currentManifestId)),
      ]);
    }

    try {
      await startDownload({
        item_type: "lora",
        source,
        manifest_id: manifest?.metadata?.id || "",
        lora_name: name,
        ...(component ? { component } : {}),
      });
      try {
        await refreshManifestPart(
          manifest?.metadata?.id || clip.manifest.id || "",
          "spec.loras",
          queryClient,
        );
      } catch {}
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : `Failed to start LoRA download: ${source}`,
      );
      setStartingSources((prev) =>
        prev.filter((p) => !(p.source === source && p.manifestId === currentManifestId)),
      );
      setNewLoraName("");
      setNewLoraSource("");
      setNewLoraComponent("");
      setIsAddingLora(false);
      return;
    }
    setIsAddingLora(false);
    setNewLoraName("");
    setNewLoraSource("");
    setNewLoraComponent("");
  };

  const loras = manifest?.spec.loras || [];
  const visibleLoras = useMemo(
    () =>
      loras.filter((lora) =>
        typeof lora !== "string"
      ),
    [loras],
  );
  const hasInstalledLoras = useMemo(
    () => visibleLoras.some((lora) => !isLoraDownloading(lora)),
    [visibleLoras],
  );

  // Auto-switch to "Installed" once a LoRA becomes verified+installed (and only if we're currently on Downloads).
  useEffect(() => {
    const verifiedInstalledCount = (manifest?.spec?.loras || []).filter((l: any) => {
      if (!l || typeof l !== "object") return false;
      const downloaded = !!(l as any).is_downloaded;
      const verified = (l as any).verified === true;
      return downloaded && verified;
    }).length;

    if (prevVerifiedInstalledCountRef.current == null) {
      prevVerifiedInstalledCountRef.current = verifiedInstalledCount;
      return;
    }

    const prev = prevVerifiedInstalledCountRef.current;
    prevVerifiedInstalledCountRef.current = verifiedInstalledCount;

    if (
      activeTab === "downloads" &&
      hadDownloadActivityRef.current &&
      verifiedInstalledCount > prev
    ) {
      setActiveTab("installed");
      // Reset activity so we only auto-switch once per "burst" of downloads.
      hadDownloadActivityRef.current = false;
    }
  }, [activeTab, manifest?.spec?.loras]);


  const handleCancelAdd = () => {
    setIsAddingLora(false);
    setNewLoraName("");
    setNewLoraSource("");
    setNewLoraComponent("");
  };

  const panelWidth = Math.max(0, panelSize - 64);
  // Drop "starting…" entries once the job shows up, the LoRA is downloaded, or after a short timeout.
  useEffect(() => {
    const now = Date.now();
    const currentManifestId = manifest?.metadata?.id || clip.manifest.id || "";
    const lorasList = (manifest?.spec?.loras || []) as any[];
    const jobsList = (polledJobs || []) as RayJobStatus[];
    setStartingSources((prev) =>
      prev.filter((p) => {
        if (currentManifestId && p.manifestId !== currentManifestId) return false;
        // If a job never appears / never emits updates within the startup timeout,
        // treat it as failed-to-start and drop it.
        if (now - p.startedAt > STARTUP_TIMEOUT_MS) {
          const mappedJobId = sourceToJobId?.[p.source];
          if (mappedJobId) {
            const existsInPoll = jobsList.some((j) => j?.job_id === mappedJobId);
            const hasWsUpdates = (getJobUpdates(mappedJobId)?.length ?? 0) > 0;
            if (!existsInPoll && !hasWsUpdates) {
              try {
                removeJobUpdates(mappedJobId);
              } catch {}
              try {
                removeJobIdToManifestId(mappedJobId);
              } catch {}
              try {
                removeJobIdToParts(mappedJobId);
              } catch {}
              try {
                removeSourceByJobId(mappedJobId);
              } catch {}
              try {
                removeSourceToJobId(p.source);
              } catch {}
            }
          }
          return false;
        }
        if (pendingDownloadSources.includes(p.source)) return false;
        const found = lorasList.find((l) => {
          if (!l) return false;
          if (typeof l === "string") return l === p.source;
          return (
            l.source === p.source ||
            l.remote_source === p.source
          );
        });
        if (found && typeof found === "object" && !!(found as any).is_downloaded) return false;
        return true;
      }),
    );
  }, [
    clip.manifest.id,
    getJobUpdates,
    manifest?.metadata?.id,
    manifest?.spec?.loras,
    pendingDownloadSources,
    polledJobs,
    removeJobIdToManifestId,
    removeJobIdToParts,
    removeJobUpdates,
    removeSourceByJobId,
    removeSourceToJobId,
    sourceToJobId,
  ]);

  return (
    <div className="flex flex-col gap-y-3 p-5">
      <div className="flex flex-col mb-1 gap-y-0.5">
        <h4 className="text-brand-light text-[12px] font-medium text-start flex items-center justify-start gap-x-1.5">
          <PiCubeDuotone className="w-4 h-4 text-brand-light" />
          <span>LoRAs</span>
        </h4>
        <p className="text-brand-light/70 text-[10.5px] text-start">
          Add LoRAs to your model to enhance or modify its behavior.
        </p>
      </div>
      <div className="mt-2 flex items-center rounded-[6px]  bg-brand-background/50 ">
        <button
          type="button"
          onClick={() => setActiveTab("downloads")}
          className={`flex-1 text-[11px] font-medium px-2 py-1.5 rounded-l-[6px] transition-colors ${
            activeTab === "downloads"
              ? "bg-brand-accent-shade text-brand-light"
              : "bg-brand-background-light text-brand-light/70 hover:text-brand-light"
          }`}
        >
          Downloads
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("installed")}
          className={`flex-1 text-[11px] font-medium px-2 py-1.5 rounded-r-[6px] transition-colors ${
            activeTab === "installed"
              ? "bg-brand-accent-shade text-brand-light"
              : "bg-brand-background-light text-brand-light/70 hover:text-brand-light"
          }`}
        >
          Installed
        </button>
      </div>

      {activeTab === "downloads" && (
        <>
          {!isAddingLora ? (
            <button
              type="button"
              onClick={() => setIsAddingLora(true)}
              className="w-full text-[11px] font-medium flex items-center justify-center gap-x-1.5  text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/5 rounded-[6px] px-3 py-2 transition-all"
            >
              <LuPlus className="w-4 h-4 text-brand-light" />
              <span>Add LoRA</span>
            </button>
          ) : (
            <div className="w-full bg-brand border border-brand-light/15 rounded-md px-3.5 py-3.5 space-y-2.5 backdrop-blur-md">
              <div className="flex flex-col gap-y-0.5 text-start">
                <label className="text-[11px] text-brand-light/90 font-medium">
                  Name
                </label>
                <p className="text-[10px] text-brand-light/55">
                  Friendly label used in the UI to identify this LoRA.
                </p>
                <input
                  type="text"
                  value={newLoraName}
                  onChange={(e) => setNewLoraName(e.target.value)}
                  className="w-full bg-brand-background border border-brand-light/15 rounded-[5px] mt-1.5 px-2.5 py-2 text-[10px] text-brand-light placeholder:text-brand-light/40 focus:outline-none focus:ring-1 focus:ring-brand-light/40"
                  placeholder="e.g. Anime style"
                />
              </div>
              <div className="flex flex-col gap-y-0.5 text-start">
                <label className="text-[11px] text-brand-light/90 font-medium">
                  Source
                </label>
                <div className="flex items-center justify-between gap-x-2 w-full">
                  <p className="text-[10px] text-brand-light/55 whitespace-nowrap overflow-hidden text-ellipsis">
                    {sourceHelpShort}
                  </p>
                </div>
                <input
                  type="text"
                  value={newLoraSource}
                  onChange={(e) => setNewLoraSource(e.target.value)}
                  className="w-full bg-brand-background border border-brand-light/15 rounded-[5px] px-2.5 py-2 mt-1.5 text-[10px] text-brand-light font-mono placeholder:text-brand-light/40 focus:outline-none focus:ring-1 focus:ring-brand-light/40"
                  placeholder="civitai:123456 or urn:air:model:lora:civitai:12345@56789 or https://... or /Users/you/models/my_lora.safetensors"
                />
              </div>
              {hasMultipleTransformers && transformerComponents.length > 0 && (
                <div className="flex flex-col gap-y-0.5 text-start">
                  <label className="text-[11px] text-brand-light/90 font-medium">
                    Apply to transformer
                  </label>
                  <p className="text-[10px] text-brand-light/55">
                    Choose which transformer this LoRA should be applied to.
                  </p>
                  <Select
                    value={newLoraComponent}
                    onValueChange={(value) => setNewLoraComponent(value)}
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-full h-8! mt-1.5 text-[10.5px] bg-brand-background border border-brand-light/15 rounded-[5px] text-brand-lighter"
                    >
                      <SelectValue placeholder="Select transformer" />
                    </SelectTrigger>
                    <SelectContent className="bg-brand-background text-brand-light font-poppins z-101 dark">
                      {transformerComponents.map((entry) => {
                        const key = getTransformerComponentKey(entry);
                        const compAny = entry.comp as any;
                        const label =
                          (compAny?.label as string | undefined) ||
                          (compAny?.name as string | undefined) ||
                          (compAny?.base as string | undefined) ||
                          key;
                        return (
                          <SelectItem
                            key={key}
                            value={key}
                            className="text-[11px] font-medium"
                          >
                            {label}
                          </SelectItem>
                        );
                      })}
                    </SelectContent>
                  </Select>
                </div>
              )}
              <div className="flex items-center justify-end gap-x-2 pt-1">
                <button
                  type="button"
                  onClick={handleCancelAdd}
                  className="text-[10px] font-medium text-brand-light/70 hover:text-brand-light/90 px-2 py-1 rounded-md transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() =>
                    {
                    handleAddLoraSource(
                      newLoraSource.trim(),
                      newLoraName,
                      newLoraComponent || undefined,
                    );
                    setIsAddingLora(false);
                    }
                  }
                  disabled={
                    !newLoraSource.trim() ||
                    !newLoraName.trim() ||
                    (hasMultipleTransformers &&
                      transformerComponents.length > 1 &&
                      !newLoraComponent.trim()) ||
                    isStartingDownload
                  }
                  className="text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90  bg-brand-accent-shade hover:bg-brand-accent-two-shade border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all"
                >
                  {isStartingDownload ? (
                    <LuLoader className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <LuPlus className="w-3.5 h-3.5" />
                  )}
                  <span>{isStartingDownload ? "Adding..." : "Add LoRA"}</span>
                </button>
              </div>
            </div>
          )}
          {(() => {
            // Pending entries come from active (polled) download jobs, mapped back to their sources,
            // and filtered by matching manifestId.
            const entries = [...pendingDownloadSources, ...startingSources.map((s) => s.source)]
              .filter((s, idx, arr) => arr.indexOf(s) === idx)
              .map((source) => {
                const lorasList = (manifest?.spec?.loras || []) as any[];
                const foundIndex = lorasList.findIndex((l) => {
                  if (!l) return false;
                  if (typeof l === "string") return l === source;
                  return l.source === source || l.remote_source === source;
                });
                return {
                  l: foundIndex >= 0 ? lorasList[foundIndex] : source,
                  index: foundIndex >= 0 ? foundIndex : 0,
                  source,
                  waitingForJob:
                    startingSources.some((s) => s.source === source) &&
                    !pendingDownloadSources.includes(source),
                };
              });
            if (!entries.length && !isAddingLora) {
              return (
                <div className="text-brand-light/90 text-[12px] font-medium flex flex-col justify-center items-center gap-y-2 p-4 w-full h-28 border border-brand-light/10 rounded-md bg-brand">
                  <PiCubeDuotone className="w-6 h-6 text-brand-light/90" />
                  <span>No pending downloads.</span>
                </div>
              );
            }
            return (
              <div className="flex flex-col gap-y-2">
                {entries.map(({ l, index, source, waitingForJob }) => (
                  <LoraDownloadRow
                    key={`${manifest?.metadata?.id || ""}:${source}:${index}`}
                    item={l}
                    index={index}
                    manifestId={manifest?.metadata?.id || ""}
                    width={panelWidth}
                    waitingForJob={waitingForJob}
                  />
                ))}
              </div>
            );
          })()}
        </>
      )}

      {activeTab === "installed" && (
        <>
          {!hasInstalledLoras && (
            <div className="text-brand-light/90 text-[12px] font-medium flex flex-col justify-center items-center gap-y-2 p-4 w-full h-28 border border-brand-light/10 rounded-md bg-brand">
              <PiCubeDuotone className="w-6 h-6 text-brand-light/90" />
              <span>No LoRAs added.</span>
            </div>
          )}

          {hasInstalledLoras && (
            <div className="flex flex-col gap-y-2">
              {loras.map((lora, idx) => {
                if (typeof lora === "string") return null;
                if (isLoraDownloading(lora)) return null; 
                return (
                  <LoraItem
                    key={
                      typeof lora === "string"
                        ? lora
                        : lora.source ||
                          `${manifest?.metadata?.id || ""}-${idx}`
                    }
                    item={lora}
                    manifestId={manifest?.metadata?.id || ""}
                    index={idx}
                  />
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default LoraPanel;
