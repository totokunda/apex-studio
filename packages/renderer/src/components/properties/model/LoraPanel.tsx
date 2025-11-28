import React, { useCallback } from "react";
import { useClipStore } from "@/lib/clip";
import { ModelClipProps } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";
import { PiCubeDuotone } from "react-icons/pi";
import { LuPlus, LuChevronDown, LuPencil, LuTrash2 } from "react-icons/lu";
import { useDownloadStore } from "@/lib/download/store";
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

interface LoraPanelProps {
  clipId: string;
}

const LoraItem: React.FC<{
  item: LoraType;
  wsFilesByPath: Record<string, any>;
  manifestId: string;
  index: number;
}> = ({ item, wsFilesByPath, manifestId, index }) => {
  const isObject = typeof item !== "string";
  const remoteSource = useMemo(
    () => (isObject ? item.remote_source || "" : ""),
    [isObject, item],
  );

  const path = useMemo(() => {
    if (!isObject) return item as string;
    const obj = item as any;
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

  const files = useMemo(() => {
    if (!remoteSource) return [] as any[];
    const entry = wsFilesByPath?.[remoteSource] || {};
    return Object.values(entry) as any[];
  }, [wsFilesByPath, remoteSource]);

  const primaryFile = files[0];

  const percent = useMemo(() => {
    if (!primaryFile) return 0;
    if (
      typeof primaryFile.totalBytes === "number" &&
      primaryFile.totalBytes > 0
    ) {
      const pct =
        ((primaryFile.downloadedBytes || 0) / primaryFile.totalBytes) * 100;
      return Math.max(0, Math.min(100, pct));
    }
    if (typeof primaryFile.progress === "number") {
      const base =
        primaryFile.progress > 1
          ? primaryFile.progress
          : primaryFile.progress * 100;
      return Math.max(0, Math.min(100, base));
    }
    return 0;
  }, [primaryFile]);

  const isDownloading = useMemo(() => {
    if (!isObject) return false;
    const obj = item;
    return !(obj.source && obj.verified);
  }, [isObject, item]);

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

  if (isDownloading) {
    // Remote LoRA – show download progress
    return (
      <div className="w-full bg-brand border border-brand-light/10 rounded-md px-3 py-2 flex flex-col gap-y-2">
        <div className="flex items-center justify-between gap-x-2">
          <span className="text-brand-light text-[11px] font-medium truncate">
            {label}
          </span>
          <span className="px-1.5 py-0.5 rounded-full border border-brand-light/20 bg-brand-background/60 text-[9px] font-medium text-brand-light/80">
            {isObject
              ? item.source
                ? "Verifying"
                : "Downloading"
              : "Downloading"}
          </span>
        </div>
        <div className="text-[10px] text-brand-light/70 font-mono break-all text-start">
          {displayPath || "—"}
        </div>
        <div className="space-y-1.5">
          <div className="w-full h-2 bg-brand-background rounded overflow-hidden border border-brand-light/10">
            <div
              className="h-full bg-brand-background-light transition-all"
              style={{ width: `${percent.toFixed(1)}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-[10px] text-brand-light/80">
            <span>{percent.toFixed(1)}%</span>
            {primaryFile &&
            typeof primaryFile.downloadedBytes === "number" &&
            typeof primaryFile.totalBytes === "number" ? (
              <span className="font-mono">
                {(primaryFile.downloadedBytes / (1024 * 1024)).toFixed(1)}
                MB / {(primaryFile.totalBytes / (1024 * 1024)).toFixed(1)}
                MB
              </span>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  // Local / resolved LoRA – show name, path, scale slider and verified badge
  return (
    <div className="w-full bg-brand shadow-md border border-brand-light/5 rounded-md flex flex-col ">
      <div className="flex items-center justify-between gap-x-2 px-3 bg-brand py-2.5 rounded-t-md">
        <div className="flex items-center gap-x-2 min-w-0">
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
            <div className="flex items-center gap-x-1.5 min-w-0">
              <span className="text-brand-light text-[11px] font-medium truncate">
                {label}
              </span>
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
            </div>
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

const LoraPanel: React.FC<LoraPanelProps> = ({ clipId }) => {
  const getClipById = useClipStore((s) => s.getClipById);
  const clip = getClipById(clipId) as ModelClipProps;
  const [isAddingLora, setIsAddingLora] = useState(false);
  const [newLoraName, setNewLoraName] = useState("");
  const [newLoraSource, setNewLoraSource] = useState("");
  const [pathToJobId, setPathToJobId] = useState<Record<string, string>>({});
  const { refreshManifestPart, getLoadedManifest } = useManifestStore();
  const [addingLoraJob, setAddingLoraJob] = useState<string | null>(null);
  const sourceHelpShort = "Enter a CivitAI ID, URL, or local path.";
  if (!clip || !clip.manifest) return null;
  const manifest = getLoadedManifest(clip.manifest.id);
  const isUnmountedRef = useRef(false);
  const unsubsRef = useRef<Array<() => void>>([]);
  const {
    startAndTrackDownload,
    subscribeToJob,
    wsFilesByPath,
    resolveDownloadBatch,
  } = useDownloadStore();

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
    if (!manifest) return;
    isUnmountedRef.current = false;
    const run = async () => {
      try {
        const loras = manifest.spec.loras || [];
        const response = await resolveDownloadBatch({
          item_type: "lora",
          sources: loras.map((lora) =>
            typeof lora === "string" ? lora : lora.source || "",
          ),
        });
        if (isUnmountedRef.current) return;
        const results = response?.results || [];
        // Build fresh state snapshots to ensure deletes/downloads are reflected
        const nextPathToJobId: Record<string, string> = {};
        for (let idx = 0; idx < results.length; idx++) {
          const r = results[idx];
          const lora = loras[idx];
          const src = typeof lora === "string" ? lora : lora.source || "";
          if (!r?.job_id || !src) continue;
          nextPathToJobId[src] = r.job_id;
        }
        setPathToJobId(nextPathToJobId);
        // Subscribe to any running jobs after state is set
        for (let idx = 0; idx < results.length; idx++) {
          const r = results[idx];
          const lora = loras[idx];
          const src = typeof lora === "string" ? lora : lora.source || "";
          if (r?.job_id && r.running && src) {
            try {
              const off = await subscribeToJob(
                r.job_id,
                src,
                onCompleteDownload,
                onErrorDownload,
              );
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
    run();
  }, [manifest?.spec.loras, onCompleteDownload]);

  const handleAddLoraSource = async (source: string, name: string) => {
    if (!source) return;
    if (!name) return;
    setAddingLoraJob(source);
    try {
      const jobIds = await startAndTrackDownload(
        {
          item_type: "lora",
          source,
          manifest_id: manifest?.metadata?.id || "",
          lora_name: name,
        },
        onCompleteDownload,
        onErrorDownload,
      );
      if (jobIds.length > 0) {
        // add the path to each job id
        let pathList: string[] = [source];
        const nextPathToJobId: Record<string, string> = {};
        for (let idx = 0; idx < pathList.length; idx++) {
          const jobId = jobIds[idx];
          if (jobId && pathList[idx]) {
            nextPathToJobId[pathList[idx]] = jobId;
          }
        }
        setPathToJobId(nextPathToJobId);
      }
    } catch {
      // no-op; this is an optional integration hook
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
                handleAddLoraSource(newLoraSource.trim(), newLoraName)
              }
              disabled={
                !newLoraSource.trim() || !newLoraName.trim() || !!addingLoraJob
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
      {loras.filter((lora) =>
        typeof lora !== "string" ? !lora.required : true,
      ).length === 0 &&
        !isAddingLora && (
          <div className="text-brand-light/90 text-[12px] font-medium flex flex-col justify-center items-center gap-y-2 p-4 w-full h-28 border border-brand-light/10 rounded-md bg-brand">
            <PiCubeDuotone className="w-6 h-6 text-brand-light/90" />
            <span>No LoRAs added.</span>
          </div>
        )}
      {loras.length > 0 && (
        <div className="flex flex-col gap-y-1">
          {loras
            .filter((lora) =>
              typeof lora !== "string" ? !lora.required : true,
            )
            .map((lora, idx) => (
              <LoraItem
                key={
                  typeof lora === "string"
                    ? lora
                    : lora.source || `${manifest?.metadata?.id || ""}-${idx}`
                }
                item={lora}
                wsFilesByPath={wsFilesByPath}
                manifestId={manifest?.metadata?.id || ""}
                index={idx}
              />
            ))}
        </div>
      )}
    </div>
  );
};

export default LoraPanel;
