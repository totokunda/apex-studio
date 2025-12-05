import React, { useCallback } from "react";
import { useClipStore } from "@/lib/clip";
import { ModelClipProps } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";
import { PiCubeDuotone } from "react-icons/pi";
import { LuPlus, LuChevronDown, LuPencil, LuTrash2 } from "react-icons/lu";
import { useManifestStore } from "@/lib/manifest/store";
import {
  LoraType,
  updateManifestLoraScale,
  updateManifestLoraName,
  deleteManifestLora,
} from "@/lib/manifest/api";
import PropertiesSlider from "@/components/properties/PropertiesSlider";
import { LuLoader } from "react-icons/lu";
import { toast } from "sonner";
import { useLoraJobStore, type LoraJobProgress } from "@/lib/engine/api";
import {
  startUnifiedDownload,
  resolveUnifiedDownloadBatch,
} from "@/lib/download/api";

interface LoraPanelProps {
  clipId: string;
  panelSize: number;
}

const isLoraDownloading = (item: LoraType): boolean => {
  if (typeof item !== "object" || item === null) return false;
  const obj = item as any;
  return !(obj.source);
};

const LoraDownloadItem: React.FC<{ job: LoraJobProgress, panelWidth: number }> = ({ job, panelWidth }) => {
  const clearJob = useLoraJobStore((s) => s.clearJob);
  const { refreshManifestPart } = useManifestStore();

  console.log(job);

  const lastUpdate = useMemo(
    () =>
      job.updates && job.updates.length > 0
        ? job.updates[job.updates.length - 1]
        : undefined,
    [job.updates],
  );

  const lastMetadata = (lastUpdate?.metadata as any) || {};

  const label = useMemo(() => {
    const metaLabel =
      lastMetadata.label || lastMetadata.filename || lastMetadata.name;
    if (metaLabel && typeof metaLabel === "string") return metaLabel;
    if (job.loraId) return job.loraId;
    if (job.source) {
      const parts = job.source.split("/");
      return parts[parts.length - 1] || job.source;
    }
    return "LoRA";
  }, [job.loraId, job.source, lastMetadata]);

  const displayPath = useMemo(() => {
    const metaPath =
      lastMetadata.remote_source ||
      lastMetadata.source ||
      lastMetadata.filename;
    return (metaPath as string) || job.source || "—";
  }, [job.source, lastMetadata]);

  const files = useMemo(() => {
    if (!job.files) return [] as any[];
    return Object.values(job.files) as any[];
  }, [job.files]);

  const primaryFile = files[0];

  const percent = useMemo(() => {
    if (typeof job.progress === "number") {
      return Math.max(0, Math.min(100, job.progress));
    }
    if (primaryFile && typeof primaryFile.progress === "number") {
      return Math.max(0, Math.min(100, primaryFile.progress));
    }
    if (typeof lastUpdate?.progress === "number") {
      return Math.max(0, Math.min(100, lastUpdate.progress));
    }
    return 0;
  }, [job.progress, primaryFile, lastUpdate]);

  const latestMessage =
    (lastUpdate?.message as string | undefined) ||
    (typeof job.currentStep === "string" ? job.currentStep : undefined);

  useEffect(() => {
    if (job.status === "complete") {
      const manifestId = job.manifestId;
      (async () => {
        if (manifestId) {
          try {
            await refreshManifestPart(manifestId, "spec.loras");
          } catch {}
        }
        clearJob(job.jobId);
      })();
    }
  }, [job.status, job.manifestId, job.jobId, clearJob, refreshManifestPart]);

  useEffect(() => {
    if (job.status === "failed") {
      const manifestId = job.manifestId;
      (async () => {
        if (manifestId) {
          try {
            await refreshManifestPart(manifestId, "spec.loras");
          } catch {}
        }
        clearJob(job.jobId);
      })();
    }
  }, [job.status, job.manifestId, job.jobId, clearJob, refreshManifestPart]);

  const statusLabel =
    job.status === "complete"
      ? "Completed"
      : job.status === "failed"
        ? "Failed"
        : "Downloading";

  return (
    <div className="w-fit bg-brand border border-brand-light/10 rounded-md px-3 py-2 flex flex-col gap-y-2" style={{ width: panelWidth - 36 }}>
      <div className="flex items-center justify-between gap-x-2 w-full">
        <div className="flex-1 min-w-0">
          <div className="text-brand-light text-[11px] font-medium truncate break-all text-start">
            {label}
          </div>
        </div>
        <div className="flex items-center gap-x-1.5 shrink-0">
          <span className="px-1.5 py-0.5 rounded-full border  border-brand-light/20 bg-brand-background/60 text-[9px] font-medium text-brand-light/80">
            {statusLabel}
          </span>
          <button
            type="button"
            onClick={() => clearJob(job.jobId)}
            className="p-1 rounded-[4px] border border-red-500/20 bg-red-500/5 text-red-300 hover:bg-red-500/15 hover:border-red-500/40 transition-colors"
          >
            <LuTrash2 className="w-3 h-3" />
          </button>
        </div>
      </div>
      <div className="text-[10px] text-brand-light/70 font-mono text-start truncate w-full">
        {displayPath || "—"}
      </div>
      <div className="space-y-1.5">
        <div className="w-full h-2 bg-brand-background rounded overflow-hidden border border-brand-light/10">
          <div
            className="h-full bg-brand-accent-shade transition-all"
            style={{ width: `${percent}%` }}
          />
        </div>
        {percent > 0 && !primaryFile && (
          <div className="flex items-center justify-between text-[10px] text-brand-light/80">
            <span>{percent.toFixed(1)}%</span>
          </div>
        )}
        {primaryFile && (
          <div className="flex items-center justify-between text-[10px] text-brand-light/80">
            <span>{percent.toFixed(1)}%</span>
            {typeof primaryFile.downloadedBytes === "number" &&
            typeof primaryFile.totalBytes === "number" ? (
              <span className="font-mono">
                {(primaryFile.downloadedBytes / (1024 * 1024)).toFixed(1)}
                MB / {(primaryFile.totalBytes / (1024 * 1024)).toFixed(1)}
                MB
              </span>
            ) : null}
          </div>
        )}
        {latestMessage && (
          <div className="text-[10px] text-brand-light/70 font-mono  text-start truncate">
            {latestMessage}
          </div>
        )}
      </div>
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

  const { refreshManifestPart } = useManifestStore();

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
                className="flex-1 min-w-0 bg-brand border border-brand-light/20 rounded-[4px] px-2 py-1 text-[10px] text-brand-light placeholder:text-brand-light/40 focus:outline-none focus:ring-1 focus:ring-brand-light/40"
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
                      await refreshManifestPart(manifestIdSafe, "spec.loras");
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
                className="shrink-0 p-1 rounded-[4px] border border-transparent hover:border-brand-light/30 hover:bg-brand-background/60 text-brand-light/80 hover:text-brand-light transition-colors"
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
            disabled={isDeleting || isSavingName}
            onClick={async () => {
              const manifestIdSafe =
                manifestId || (isObject && (item as any)?.metadata?.id) || "";
              if (!manifestIdSafe) return;
              setIsDeleting(true);
              try {
                await deleteManifestLora(manifestIdSafe, index);
                try {
                  await refreshManifestPart(manifestIdSafe, "spec.loras");
                } catch {}
              } finally {
                setIsDeleting(false);
              }
            }}
            className="shrink-0 p-1 rounded-[4px] border border-red-500/20 bg-red-500/5 text-red-300 hover:bg-red-500/15 hover:border-red-500/40 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
            const clamped = Math.max(0, Math.min(1, next));
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
                  await refreshManifestPart(manifestIdSafe, "spec.loras");
                } catch {}
              } catch {
                // Best-effort; UI already reflects local slider value
              }
            }, 300);
          }}
          min={0}
          max={1}
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
  const { refreshManifestPart, getLoadedManifest } = useManifestStore();
  const [addingLoraJob, setAddingLoraJob] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"downloads" | "installed">(
    "installed",
  );
  const sourceHelpShort = "Enter a CivitAI ID, URL, or local path.";
  if (!clip || !clip.manifest) return null;
  const manifest = getLoadedManifest(clip.manifest.id);
  const isUnmountedRef = useRef(false);
  const unsubsRef = useRef<Array<() => void>>([]);
  const startTracking  = useLoraJobStore((s) => s.startTracking);
  const stopTracking  = useLoraJobStore((s) => s.stopTracking);
  const loraJobs  = useLoraJobStore((s) => s.jobs);

  
  const onCompleteDownload = useCallback(
    async (_path: string) => {
      try {
        await refreshManifestPart(manifest?.metadata?.id || "", "spec.loras");
      } catch {}
    },
    [refreshManifestPart, manifest?.metadata?.id],
  );

  const onErrorDownload = useCallback(
    async (error: unknown, source: string) => {
      await refreshManifestPart(manifest?.metadata?.id || "", "spec.loras");
      toast.error(
        error instanceof Error
          ? error.message
          : `Failed to download LoRA: ${source}`,
      );
      setAddingLoraJob(null);
      setNewLoraName("");
      setNewLoraSource("");
      setIsAddingLora(false);
    },
    [manifest?.metadata?.id, refreshManifestPart],
  );

  useEffect(() => {
    return () => {
      isUnmountedRef.current = true;
      for (const off of unsubsRef.current) {
        try {
          off();
        } catch {}
      }
      unsubsRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (!manifest) return;
    isUnmountedRef.current = false;

    const run = async () => {
      try {
        const loras = manifest.spec.loras || [];
        const response = await resolveUnifiedDownloadBatch({
          item_type: "lora",
          sources: loras.map((lora) =>
            typeof lora === "string" ? lora : lora.source || "",
          ),
        });
        if (isUnmountedRef.current) return;
        const results = response?.data?.results || [];
        const runningJobIds: string[] = [];
        for (let idx = 0; idx < results.length; idx++) {
          const r = results[idx];
          const lora = loras[idx];
          const src = typeof lora === "string" ? lora : lora.source || "";
          if (r?.job_id && r.running && src) {
            runningJobIds.push(r.job_id);
            try {
              await startTracking(r.job_id, {
                manifestId: manifest.metadata?.id || "",
                source: src,
              });
            } catch {}
          }
        }
        unsubsRef.current = runningJobIds.map(
          (jobId) => () => {
            void stopTracking(jobId);
          },
        );
      } catch {}
    };
    run();
  }, [
    manifest?.spec.loras,
    manifest?.metadata?.id,
    resolveUnifiedDownloadBatch,
    startTracking,
    stopTracking,
    onCompleteDownload,
    onErrorDownload,
  ]);

  const handleAddLoraSource = async (source: string, name: string) => {
    if (!source) return;
    if (!name) return;
    setAddingLoraJob(source);
    try {
      const response = await startUnifiedDownload({
        item_type: "lora",
        source,
        manifest_id: manifest?.metadata?.id || "",
        lora_name: name,
      });
      if (!response.success || !response.data?.job_id) {
        throw new Error(response.error || "Failed to start LoRA download");
      }
      const jobId = response.data.job_id;
      try {
        await startTracking(jobId, {
          manifestId: manifest?.metadata?.id || "",
          source,
        });
      } catch {}
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : `Failed to start LoRA download: ${source}`,
      );
      setAddingLoraJob(null);
      setNewLoraName("");
      setNewLoraSource("");
      setIsAddingLora(false);
    } finally {
      const manifestIdSafe = manifest?.metadata?.id || "";
      if (!manifestIdSafe) {
        setIsAddingLora(false);
        setNewLoraName("");
        setNewLoraSource("");
        setAddingLoraJob(null);
        return;
      }

      const targetSource = source;
      const targetName = name;
      const maxAttempts = 30;
      const delayMs = 1000;

      const pollUntilLoraAppears = async () => {
        let attempt = 0;
        while (attempt < maxAttempts && !isUnmountedRef.current) {
          try {
            await refreshManifestPart(manifestIdSafe, "spec.loras");
          } catch {}

          try {
            const { getLoadedManifest } = useManifestStore.getState();
            const updatedManifest = getLoadedManifest(manifestIdSafe) as any;
            const lorasList = updatedManifest?.spec?.loras || [];
            const found =
              Array.isArray(lorasList) &&
              lorasList.some((l: any) => {
                if (typeof l === "string") return l === targetSource;
                const src = l?.source;
                const remote = l?.remote_source;
                const nm = l?.name || l?.label;
                return (
                  src === targetSource ||
                  remote === targetSource ||
                  nm === targetName
                );
              });
            if (found) break;
          } catch {
            // ignore lookup errors and keep polling until attempts exhausted
          }

          attempt += 1;
          if (attempt >= maxAttempts || isUnmountedRef.current) break;
          await new Promise((resolve) => setTimeout(resolve, delayMs));
        }

        if (!isUnmountedRef.current) {
          setIsAddingLora(false);
          setNewLoraName("");
          setNewLoraSource("");
          setAddingLoraJob(null);
        }
      };

      void pollUntilLoraAppears();
    }
  };

  const loras = manifest?.spec.loras || [];
  const visibleLoras = useMemo(
    () =>
      loras.filter((lora) =>
        typeof lora !== "string" ? !lora.required : true,
      ),
    [loras],
  );
  const hasInstalledLoras = useMemo(
    () => visibleLoras.some((lora) => !isLoraDownloading(lora)),
    [visibleLoras],
  );


  const handleCancelAdd = () => {
    setIsAddingLora(false);
    setNewLoraName("");
    setNewLoraSource("");
    setAddingLoraJob(null);
  };

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
                  placeholder="civitai:123456 or https://... or /Users/you/models/my_lora.safetensors"
                />
              </div>
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
                    handleAddLoraSource(newLoraSource.trim(), newLoraName)
                    setIsAddingLora(false);
                    }
                  }
                  disabled={
                    !newLoraSource.trim() ||
                    !newLoraName.trim() ||
                    !!addingLoraJob
                  }
                  className="text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90  bg-brand-accent-shade hover:bg-brand-accent-two-shade border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all"
                >
                  {addingLoraJob ? (
                    <LuLoader className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <LuPlus className="w-3.5 h-3.5" />
                  )}
                  <span>{addingLoraJob ? "Adding..." : "Add LoRA"}</span>
                </button>
              </div>
            </div>
          )}

          {!loraJobs && Object.values(loraJobs).length === 0 && !isAddingLora && (
            <div className="text-brand-light/90 text-[12px] font-medium flex flex-col justify-center items-center gap-y-2 p-4 w-full h-28 border border-brand-light/10 rounded-md bg-brand">
              <PiCubeDuotone className="w-6 h-6 text-brand-light/90" />
              <span>No active downloads.</span>
            </div>
          )}

          {loraJobs && Object.values(loraJobs).length > 0 && (
            <div className="flex flex-col gap-y-1">
              {Object.values(loraJobs).map((job: LoraJobProgress) => {
                if (!job) return null;
                return (
                  <LoraDownloadItem
                    panelWidth={panelSize}
                    key={
                      job.source || job.remote_source || `${manifest?.metadata?.id || ""}-${job.jobId}`
                    }
                    job={job}
                  />
                );
              })}
            </div>
          )}
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
                if (typeof lora !== "string" && lora.required) return null;
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
