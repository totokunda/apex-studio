import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type ManifestComponent,
  type LoraType,
  type ManifestComponentModelPathItem,
  ManifestSchedulerOption,
  validateAndRegisterCustomModelPath,
  deleteCustomModelPath,
  extractAllComponentDownloadingPaths,
} from "@/lib/manifest/api";
import { cn } from "@/lib/utils";
import { deleteDownload } from "@/lib/download/api";
import {
  formatDownloadProgress,
  formatSpeed,
  formatBytes,
} from "@/lib/components-download/format";
import {
  LuChevronDown,
  LuChevronRight,
  LuDownload,
  LuCheck,
  LuTrash,
  LuLoader,
  LuPlus,
  LuFolder,
} from "react-icons/lu";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useStartUnifiedDownloadMutation } from "@/lib/download/mutations";
import { useDownloadJobIdStore } from "@/lib/download/job-id-store";
import { UnifiedDownloadWsUpdate } from "@/lib/download/api";
import { unifiedDownloadWsUpdatesToFiles } from "@/lib/download/ws-updates-to-files";
import { ProgressBar } from "../common/ProgressBar";
import { cancelRayJob } from "@/lib/jobs/api";
import { useQueryClient } from "@tanstack/react-query";
import { refreshManifestPart } from "@/lib/manifest/queries";
import { toast } from "sonner";
import { TbCancel } from "react-icons/tb";
import { pickMediaPaths } from "@app/preload";

const STARTUP_TIMEOUT_MS = 15_000;

const getComponentTypeLabel = (type: string): string => {
  const labels: Record<string, string> = {
    transformer: "Transformer",
    text_encoder: "Text Encoder",
    vae: "Variational Autoencoder",
    scheduler: "Scheduler",
    helper: "Helper",
    extra_model_path: "Extra Model Path",
  };
  return labels[type] || type.charAt(0).toUpperCase() + type.slice(1);
};

const formatComponentName = (name: string): string => {
  return name
    .replace(/\./g, " ")
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};


interface AddModelPathFormProps {
  manifestId: string;
  componentIndex: number;
  existingVariants: string[];
  onAddCustomPath: (variant: string) => void;
}

const AddModelPathForm: React.FC<AddModelPathFormProps> = ({
  manifestId,
  componentIndex,
  existingVariants,
  onAddCustomPath,
}) => {
  const [newModelName, setNewModelName] = useState("");
  const [newModelPath, setNewModelPath] = useState("");
  const [isValidatingModelPath, setIsValidatingModelPath] = useState(false);
  const queryClient = useQueryClient();

  const normalizedExistingVariantNames = useMemo(() => {
    return new Set(
      (existingVariants ?? [])
        .map((v) => v?.trim?.().toLowerCase?.())
        .filter((v): v is string => typeof v === "string" && v.length > 0),
    );
  }, [existingVariants]);
  const normalizedNewModelName = newModelName.trim().toLowerCase();
  const isDuplicateVariantName =
    normalizedNewModelName.length > 0 &&
    normalizedExistingVariantNames.has(normalizedNewModelName);

  return (
    <div className="mt-2.5 w-full bg-brand-background border border-brand-light/15 rounded-md px-3 py-3.5 space-y-2.5">
      <div className="flex flex-col gap-y-1.5">
        <label className="text-[10px] text-brand-light/90 font-medium">
          Model Name
        </label>
        <p className="text-[9.5px] text-brand-light/55">
          A friendly label used in the UI for this model; this can be any text.
        </p>
        <input
          type="text"
          value={newModelName}
          onChange={(e) => setNewModelName(e.target.value)}
          className="w-full bg-brand border border-brand-light/15 rounded-[5px] px-2.5 py-1.5 text-[10px] text-brand-light placeholder:text-brand-light/40 focus:outline-none focus:ring-1 focus:ring-brand-light/40"
          placeholder="e.g. Local GGUF Q4 variant"
        />
        {isDuplicateVariantName && (
          <p className="text-[9.5px] text-red-400">
            A variant with this name already exists.
          </p>
        )}
      </div>
      <div className="flex flex-col gap-y-1.5">
        <label className="text-[10px] text-brand-light/90 font-medium">
          Model Path
        </label>
        <p className="text-[9.5px] text-brand-light/55">
          Local path on this machine. Can be a file or a directory and will be
          checked before use.
        </p>
        <div className="flex items-center gap-x-2">
          <input
            type="text"
            value={newModelPath}
            onChange={(e) => setNewModelPath(e.target.value)}
            className="flex-1 bg-brand border border-brand-light/15 rounded-[5px] px-2.5 py-1.5 text-[10px] text-brand-light font-mono placeholder:text-brand-light/40 focus:outline-none focus:ring-1 focus:ring-brand-light/40"
            placeholder="/Users/you/models/my_model.safetensors"
          />
          <button
            type="button"
            onClick={async () => {
              try {
                const picked = await pickMediaPaths({
                  directory: false,
                  title: "Select model file or folder",
                  filters: [
                    {
                      name: "Model Files",
                      extensions: [
                        "safetensors",
                        "ckpt",
                        "pt",
                        "pth",
                        "bin",
                        "gguf",
                        "ggml",
                        "onnx",
                        "tflite",
                      ],
                    },
                    { name: "All Files", extensions: ["*"] },
                  ],
                  defaultPath:
                    newModelPath && newModelPath.trim().length > 0
                      ? newModelPath.trim()
                      : undefined,
                });
                const selectedPath =
                  Array.isArray(picked) && picked.length > 0 ? picked[0] : null;
                if (selectedPath && typeof selectedPath === "string") {
                  setNewModelPath(selectedPath);
                }
              } catch {
                // Swallow errors; keep existing path untouched
              }
            }}
            className="text-[10px] font-medium flex items-center justify-center gap-x-1 text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[5px] px-2.5 py-1.5 transition-all"
          >
            <LuFolder className="w-3.5 h-3.5" />
            <span>Browse</span>
          </button>
        </div>
      </div>
      <div className="flex items-center justify-end gap-x-2 pt-1">
        
        <button
          type="button"
          disabled={isValidatingModelPath || isDuplicateVariantName}
          onClick={async () => {
            const path = newModelPath.trim();
            const name = newModelName.trim();
            if (!path || isValidatingModelPath) return;
            if (
              name &&
              normalizedExistingVariantNames.has(name.trim().toLowerCase())
            ) {
              toast.error("A variant with this name already exists.");
              return;
            }
            try {
              setIsValidatingModelPath(true);
              const res = await validateAndRegisterCustomModelPath(
                manifestId,
                componentIndex,
                name || undefined,
                path,
              );

              if (!res?.success) {
                toast.error(
                  res && typeof res.error === "string"
                    ? res.error
                    : "Failed to validate model path",
                );
                return;
              }
                
              await Promise.all([
                refreshManifestPart(manifestId, `spec.components.${componentIndex}`, queryClient),
                refreshManifestPart(manifestId, 'downloaded', queryClient, true),
              ]);
              onAddCustomPath(name);  
              setNewModelName("");
              setNewModelPath("");
              toast.success("Model path validated and registered successfully");
          } catch {
            toast.error("Failed to validate model path");
          } finally {
            setIsValidatingModelPath(false);
          }
        }}
          className="text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all"
        >
          {isValidatingModelPath ? (
            <>
              <LuLoader className="w-3.5 h-3.5 animate-spin" />
              <span>Verifying...</span>
            </>
          ) : (
            <>
              <LuPlus className="w-3.5 h-3.5" />
              <span>Add</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

const ComponentCard: React.FC<{
  component: ManifestComponent;
  manifestId: string;
  index: number;
}> = ({ component, manifestId, index }) => {
  
  const typeLabel = getComponentTypeLabel(component.type);
  const displayName =
    component.label ||
    (component.name
      ? formatComponentName(component.name)
      : component.base
        ? formatComponentName(component.base)
        : typeLabel);
  const componentCarRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  
  
  const isScheduler = component.type === "scheduler";
  const isConfigOnly = (!component.model_path?.length && component.config_path);
  const isModelPaths = (component.model_path?.length ?? 0) > 0;
  const isEmpty = !isScheduler && !isConfigOnly && !isModelPaths;
  const isDownloaded = component.is_downloaded || isEmpty;
  const { getSourceToJobId, getJobUpdates } = useDownloadJobIdStore();
  const allDownloadingPaths = useMemo(() => {
    return extractAllComponentDownloadingPaths(component);
  }, [component]);
  const isDownloading = allDownloadingPaths.some((path) => {
      const jobId = getSourceToJobId(path.path);
      return jobId && (getJobUpdates(jobId)?.length ?? 0) > 0;
    });

  return (
    <>
      <Collapsible open={expanded} onOpenChange={setExpanded}
        ref={componentCarRef}
        className="bg-brand border border-brand-light/10 rounded-md text-start"
      >
        <CollapsibleTrigger className="w-full">
          <div className="w-full flex items-center justify-between p-3">
            <div className="flex items-center gap-x-2 justify-between w-fit">
              <div className={cn("bg-brand-background border rounded-full p-0.5", isDownloaded ? "bg-green-600/30 border-green-400/40" : "bg-brand-background border-brand-light/20")}>
              {isDownloaded ? <LuCheck className="text-green-400 w-3 h-3" /> : isDownloading ? <LuLoader className="w-3 h-3 text-brand-light/70 animate-spin" /> : <LuDownload className="text-brand-light/70 w-3 h-3" />}
              </div>
              <h3 className="text-brand-light text-[12px] font-medium">{displayName}</h3>
            </div>
            <div className="flex items-center gap-x-2 animate-in fade-in-0 duration-300">
              <div className="text-brand-light/60 text-[10px] font-mono bg-brand-background px-2 py-0.5 rounded">
                {typeLabel}
              </div>
              {
                !isEmpty && <>
                {expanded ? <LuChevronDown className="text-brand-light w-3 h-3" /> : <LuChevronRight className="text-brand-light w-3 h-3" />}
                </>
              }
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
        {isScheduler && <SchedulerSection component={component} manifestId={manifestId} componentIndex={index} />}
        {isConfigOnly && <ConfigOnlySection component={component} />}
        {isModelPaths && <ModelPathsSection component={component} manifestId={manifestId} componentIndex={index} />}
        </CollapsibleContent>
       
      </ Collapsible>

    </>
  );
};


const SchedulerSection: React.FC<{
  component: ManifestComponent;
  manifestId: string;
  componentIndex: number;
}> = ({ component, manifestId, componentIndex }) => {
  const schedulerOptions = component.scheduler_options;
  const isDownloaded = component.is_downloaded;
  const optionLabels= schedulerOptions?.map((option) => option.label);
  const [selectedOption, setSelectedOption] = useState(optionLabels?.[0] ?? "");

  const isConfigPathDownloaded = (configPath: string) => {
    // Lazy heuristic to check if the config path is downloaded
    // A path is considered downloaded if it looks like a local file path (not a URL)
    // and the component is marked as downloaded
    
    if (!isDownloaded) {
      return false;
    }

    // Exclude URLs (http://, https://, ftp://, etc.)
    // URLs have :// after the scheme, not just : (which would match Windows drive letters)
    if (/^[a-zA-Z][a-zA-Z\d+\-.]*:\/\//.test(configPath)) {
      return false;
    }
    
    // Windows absolute paths (check before Unix paths to avoid false positives):
    // - Drive letter paths (C:\, D:\, etc.)
    // - UNC paths (\\server\share)
    // - Absolute paths starting with \ (relative to current drive root)
    if (/^[A-Za-z]:[\\/]/.test(configPath) || configPath.startsWith("\\\\") || configPath.startsWith("\\")) {
      return true;
    }
    
    // Unix absolute paths (starting with /)
    if (configPath.startsWith("/")) {
      return true;
    }
    
    return false;
  }

  return (
    <div>
      <div className="w-full px-3 pb-4">
        <Tabs className="w-full" defaultValue={optionLabels?.[0]} onValueChange={setSelectedOption}>
          <TabsList className="w-full flex justify-start  dark gap-2 bg-brand flex-wrap mb-0"> {optionLabels?.map((option) => (
            <TabsTrigger key={option} value={option ?? ""} className={cn("text-brand-light/60 text-[10px]  border border-brand-light/10 shadow hover:bg-brand-light/5 hover:text-brand-light px-2 py-1 rounded", selectedOption === option ? "bg-brand-light/10! text-brand-light" : "bg-brand-background/70")}>
              {option}
            </TabsTrigger>
          ))}</TabsList>
            
            <div className=" bg-brand-background border border-brand-light/10 rounded-md p-3 " > {schedulerOptions?.map((option) => (
              <TabsContent className="mt-1" value={option.label ?? ""}>  
            <SchedulerConfigPathItem manifestId={manifestId} componentIndex={componentIndex} key={option.name} option={option} configPath={option.config_path} isConfigPathDownloaded={isConfigPathDownloaded} />
            </TabsContent>
          ))}</div>
        </Tabs>

      </div>
    </div>
  );
};

const SchedulerConfigPathItem: React.FC<{
  configPath: string | undefined;
  option: ManifestSchedulerOption;
  isConfigPathDownloaded: (configPath: string) => boolean | undefined;
  manifestId: string;
  componentIndex: number;

}> = ({ configPath = "", option, isConfigPathDownloaded, manifestId, componentIndex }) => {


  const isDownloaded = isConfigPathDownloaded(configPath);
  const queryClient = useQueryClient();
  const {
    addSourceToJobId,
    getSourceToJobId,
    getJobUpdates,
    removeJobUpdates,
    removeSourceToJobId,
    clearDownloadTracking,
    addJobIdToParts,
    addJobIdToManifestId,
  } = useDownloadJobIdStore();
  const jobId = getSourceToJobId(configPath);
  const jobUpdates = getJobUpdates(jobId);
  const isDownloading = (jobUpdates?.length ?? 0) > 0;
  const [startDownloading, setStartDownloading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (jobId && !jobUpdates?.length) {
      setStartDownloading(true);
    }
  }, []);

  useEffect(() => {
    if (isDownloading || !jobId) {
      setStartDownloading(false);
    }
  }, [isDownloading, jobId]);

  const { mutate: startDownload } = useStartUnifiedDownloadMutation({
    onSuccess(data, variables) {
      addSourceToJobId(variables.source, data.job_id);
      addJobIdToParts(data.job_id, [`spec.components.${componentIndex}`, `downloaded`]);
      addJobIdToManifestId(data.job_id, manifestId);
    },
  });

  const ref = useRef<HTMLDivElement>(null);
  const width = ref.current?.clientWidth ?? 0;

  return (
    <>
      <div key={option.name} className="space-y-0.5" ref={ref}>
        <h5 className="text-brand-light text-[11px] font-medium">{option.name}</h5>
        <p className="text-brand-light text-[10px]">{option.description}</p>
      </div>
      {option.config_path && (
        <div className="text-brand-light text-[10.5px] pt-2 gap-y-0.5 flex w-full flex-col border-t border-brand-light/5 mt-2">
         <span className="text-brand-light font-medium">Config Path</span>
          <div className="text-brand-light/70 text-[9.5px] font-mono break-all w-fit">
            {option.config_path}
          </div>
          {isDownloaded && (
            <div className="text-[10px] text-green-400 flex items-center justify-between gap-x-1 mt-1 ">
              <div className="text-[10px] font-medium text-brand-light/90 flex items-center justify-start gap-x-1">
              <LuCheck className="w-3 h-3 text-green-400" />
              <span>Downloaded</span>
            </div>
            <button
              onClick={async () => {
                setIsDeleting(true);
                await deleteDownload({
                  path: configPath,
                  item_type: "component",
                });
                await Promise.all([
                  refreshManifestPart(manifestId, `spec.components.${componentIndex}`, queryClient),
                  refreshManifestPart(manifestId, `downloaded`, queryClient, true),
                ]);
                setIsDeleting(false);
              }}
              className="w-fit text-[10px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[5px] px-2.5 py-1.5 transition-all"
            >
              {isDeleting ? <LuLoader className="w-3.5 h-3.5 text-brand-light/70 animate-spin" /> : <LuTrash className="w-3 h-3" />}  
              <span>{isDeleting ? "Deleting..." : "Delete"}</span>
            </button>
            </div>
          )}
          {!isDownloaded && (
            <div className="flex flex-col gap-y-2 mt-2">
              {isDownloading ? (
                <DownloadProgressSection
                  jobUpdates={jobUpdates ?? []}
                  width={width}
                  onReset={() => {
                    setStartDownloading(false);
                    clearDownloadTracking({ jobId, source: configPath });
                  }}
                  onCancel={async () => {
                    if (!jobId) return;
                    try {
                      await cancelRayJob(jobId);
                    } catch {}
                    try {
                      removeJobUpdates(jobId);
                    } catch {}
                    try {
                      removeSourceToJobId(configPath);
                    } catch {}
                  }}
                />
              ) :
              (
                <>
                  <button
                    onClick={async () => {
                      setStartDownloading(true);
                      await startDownload({
                        item_type: "component",
                        source: configPath,
                      });
                    }}
                    disabled={startDownloading || isDownloading}
                    className={cn("w-full text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light  hover:bg-brand/70 border border-brand-light/10 rounded-[6px] bg-brand px-3 py-2 transition-all", startDownloading ? "opacity-60 cursor-default hover:bg-brand" : "")}
                  >
                    {startDownloading ? (
                      <LuLoader className="w-3.5 h-3.5 text-brand-light/70 animate-spin" />
                    ) : (
                      <LuDownload className="w-3.5 h-3.5" />
                    )}
                    <span>
                      {startDownloading ? "Downloading..." : "Download Config"}
                    </span>
                  </button>
                  {startDownloading && (
                    <button
                      type="button"
                      onClick={() => {
                        setStartDownloading(false);
                        clearDownloadTracking({ jobId, source: configPath });
                      }}
                      className="w-full text-[10px] text-brand-light/80 hover:text-brand-light font-medium bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded-[6px] px-2 py-2 transition-all"
                    >
                      Reset download state
                    </button>
                  )}
                </>
            )}
            </div>
          )}
        </div>
      )}
      </>
  );
};



// Will add when used 
const ConfigOnlySection: React.FC<{
  component: ManifestComponent;
}> = ({ component: _component }) => {
  return (
    <div>
      <h3 className="text-brand-light text-[12px] font-medium">Config Only</h3>
    </div>
  );
};

const ModelPathsSection: React.FC<{
  component: ManifestComponent;
  manifestId: string;
  componentIndex: number;
}> = ({ component, manifestId, componentIndex }) => {
   const variants = (component.model_path  as ManifestComponentModelPathItem[])?.map((path) => path.variant);
   const defaultVariant = variants?.[0] ?? "";
   const [selectedVariant, setSelectedVariant] = useState(defaultVariant);

   const onRemoveCustomPath = useCallback(() => {
    setSelectedVariant(defaultVariant);
   }, [defaultVariant, setSelectedVariant]);

   const onAddCustomPath = useCallback((variant: string) => {
    setSelectedVariant(variant);
   }, [setSelectedVariant]);

  return (
    <div className="w-full">
      <Tabs defaultValue={defaultVariant} onValueChange={setSelectedVariant} value={selectedVariant}>
        <TabsList className="w-full flex justify-start dark gap-1.5 bg-brand flex-wrap px-3">
         {variants?.map((variant ) => variant && (
          <TabsTrigger 
          className={cn("text-brand-light/60 text-[10px] font-mono border border-brand-light/10 shadow hover:bg-brand-light/5 hover:text-brand-light px-2 py-0.5 rounded",
             variant === selectedVariant ? "bg-brand-light/10! text-brand-light" : "bg-brand-background/70")}
          key={variant}
          value={variant ?? ""}
    
          >{variant.charAt(0).toUpperCase() + variant.slice(1)}</TabsTrigger>
         ))}
         <TabsTrigger 
          className={cn("text-brand-light/60 text-[10px] font-mono border border-brand-light/10 shadow hover:bg-brand-light/5 hover:text-brand-light px-2 py-0.5 rounded flex items-center gap-x-1",
            selectedVariant === "add-variant" ? "bg-brand-light/10! text-brand-light" : "bg-brand-background/70",
             )}
          key="add-variant"
          value="add-variant"
          ><LuPlus className="w-3.5 h-3.5" /> <span>Add Variant</span></TabsTrigger>
        </TabsList>
          {variants?.map((variant) => (
            <TabsContent key={variant} value={variant ?? ""}>
              <div className="px-3 pb-4">
                {(component.model_path  as ManifestComponentModelPathItem[])?.filter((path) => path.variant === variant).map((path) => (
                  <ModelPathItem key={path.path} path={path} variant={variant ?? ""} componentIndex={componentIndex} manifestId={manifestId} onRemoveCustomPath={onRemoveCustomPath} />
                ))}
              </div>
            </TabsContent>
          ))}
          <TabsContent key="add-variant" value="add-variant">
            <div className="px-3 pb-4">
              <AddModelPathForm
                manifestId={manifestId}
                componentIndex={componentIndex}
                existingVariants={(variants ?? []).filter(
                  (v): v is string => typeof v === "string" && v.length > 0,
                )}
                onAddCustomPath={onAddCustomPath}
              />
            </div>
          </TabsContent>
      </Tabs>
    </div>
  );
};

const ModelPathItem: React.FC<{
  path: ManifestComponentModelPathItem;
  variant: string;
  componentIndex: number;
  manifestId: string;
  onRemoveCustomPath: () => void;
}> = ({path, componentIndex, manifestId, onRemoveCustomPath}) => {
  
  const {
    getSourceToJobId,
    addSourceToJobId,
    getJobUpdates,
    removeJobUpdates,
    removeSourceByJobId,
    clearDownloadTracking,
    addJobIdToParts,
    addJobIdToManifestId,
  } = useDownloadJobIdStore();
  const queryClient = useQueryClient();
  const { mutate: startDownload } = useStartUnifiedDownloadMutation({
    onSuccess(data, variables) {
      addSourceToJobId(variables.source, data.job_id);
      addJobIdToParts(data.job_id, [`spec.components.${componentIndex}`, `downloaded`]);
      addJobIdToManifestId(data.job_id, manifestId);
    },
  });

  
  const jobId = getSourceToJobId(path.path);
  const jobUpdates = getJobUpdates(jobId);
  const isDownloading = (jobUpdates?.length ?? 0) > 0;
  const ref = useRef<HTMLDivElement>(null);
  const width = ref.current?.clientWidth ?? 0;
  const [startDownloading, setStartDownloading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);


  const onClearJobId = useCallback(async (e: Event) => {
    const event = e as CustomEvent<{ jobId: string }>;
    // Refresh the manifest query when job ids are cleared.
    const eventJobId = event.detail.jobId;
    if (!eventJobId || eventJobId !== jobId) return;
    setStartDownloading(false);
    clearDownloadTracking({ jobId: eventJobId, source: path.path });
    try {
      await Promise.all([
        refreshManifestPart(
          manifestId,
          `spec.components.${componentIndex}`,
          queryClient,
        ),
        //refreshManifestPart(manifestId, `downloaded`, queryClient, true),
      ]);
    } catch {}
  }, [clearDownloadTracking, componentIndex, jobId, manifestId, path.path, queryClient]);

  useEffect(() => {
    window.addEventListener("clear-job-id", onClearJobId as EventListener);
    return () => {
      window.removeEventListener("clear-job-id", onClearJobId as EventListener);
    };
  }, [onClearJobId]);


  useEffect(() => {
    if (isDownloading || !jobId) {
      setStartDownloading(false);
    }
  }, [isDownloading, jobId]);

  useEffect(() => {
    if (jobId && !jobUpdates?.length) {
      setStartDownloading(true);
    }
  }, []);


  return (
    <div className="w-full bg-brand-background rounded-md p-3" ref={ref}>
     {(path.variant || path.custom) && (
              <div className="flex flex-row justify-between items-center mb-2.5">
                <div className="flex items-center gap-x-1.5">
                  {path.variant && (
                    <div className="text-brand-light text-[11px] break-all font-semibold">
                      {path.variant.toLowerCase().includes("default")
                        ? "Default"
                        : path.variant.toLowerCase().includes("gguf")
                          ? path.variant
                              .replace("GGUF_", "")
                              .replace("Q", "q")
                              .toUpperCase()
                          : path.variant}
                    </div>
                  )}
                  {path.custom && (
                    <span className="inline-flex items-center rounded-full bg-brand-background px-2 py-0.5 text-[9px] font-medium text-brand-light/80 border border-brand-light/20">
                      Custom
                    </span>
                  )}
                </div>
                {typeof (path as any).file_size === "number" &&
                  (path as any).file_size > 0 && (
                    <div className="shrink-0 ml-2 text-[10px] text-brand-light/80 font-mono whitespace-nowrap">
                      {formatBytes((path as any).file_size, 1)}
                    </div>
                  )}
              </div>
            )}
            <div className="flex items-start justify-between gap-x-2 ">
              <div className="flex-1 min-w-0 flex-row items-center gap-x-2">
                <div className="text-brand-light text-[10.5px] font-medium mb-1">
                  Model Path
                </div>
                <div className="text-brand-light text-[10px] font-mono break-all">
                  {path.path}
                </div>
              </div>
            </div>
              {(path.type || path.precision) && (
              <div className="flex flex-col items-start  mt-2 justify-start border-t border-brand-light/5  pt-2">
                <h4 className="text-brand-light text-[10.5px] font-medium mb-1">
                  Model specifications
                </h4>
                {path.type && (
                  <div className="text-[10px] flex flex-row items-center gap-x-1 ">
                    <span className="text-brand-light/60 font-medium">
                      Model Type{" "}
                    </span>
                    <span className="text-brand-light/80 font-mono">
                      {path.type === "gguf"
                        ? "GGUF"
                        : formatComponentName(path.type)}
                    </span>
                  </div>
                )}
                {path.precision && (
                  <div className="text-[10px] flex flex-row items-center gap-x-1 ">
                    <span className="text-brand-light/60 font-medium">
                      Precision{" "}
                    </span>
                    <span className="text-brand-light/90 font-mono">
                      {path.precision.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>
            )}
          {path.is_downloaded && (
            <div className="flex flex-row items-center justify-between gap-x-2">
              <div className="text-[11px] font-medium text-brand-light/90 mt-4 mb-1.5 flex items-center justify-start gap-x-1">
                <LuCheck className="w-3 h-3 text-green-400" />
                <span>Downloaded</span>
              </div> 
              <button
                onClick={async () => {
                  setIsDeleting(true);
                  if (path.custom) {
                    await deleteCustomModelPath(manifestId, componentIndex, path.path);
                    onRemoveCustomPath();
                  } else {
                    await deleteDownload({
                      path: path.path,
                      item_type: "component",
                    });
                  }
                  await Promise.all([
                    refreshManifestPart(manifestId, `spec.components.${componentIndex}`, queryClient),
                    refreshManifestPart(manifestId, `downloaded`, queryClient, true),
                  ]);
                  setIsDeleting(false);
                }}
                className={cn("text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all", isDeleting ? "opacity-60 cursor-default hover:bg-brand" : "")}
              >
                {isDeleting ? <LuLoader className="w-3 h-3 text-brand-light/70 animate-spin" /> : <LuTrash className="w-3.5 h-3.5" />}
                <span>{isDeleting ? "Deleting..." : "Delete"}</span>
              </button>
            </div>
          )}
          {!path.is_downloaded && (
            <div className="flex flex-row items-center justify-between gap-x-2">
              {isDownloading ? (
                <DownloadProgressSection
                  jobUpdates={jobUpdates ?? []}
                  width={width}
                  onReset={() => {
                    setStartDownloading(false);
                    clearDownloadTracking({ jobId, source: path.path });
                  }}
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
                  }}
                />
              ) : (
                <div className="w-full">
                  <button
                    onClick={async () => {
                      setStartDownloading(true);
                      await startDownload({
                        item_type: "component",
                        source: path.path,
                      });
                    }}
                    disabled={startDownloading || isDownloading}
                    className={cn(
                      "w-full mt-3 text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-md px-3 py-2 transition-all",
                      startDownloading ? "opacity-60 cursor-default hover:bg-brand" : "",
                    )}
                  >
                    {startDownloading ? (
                      <LuLoader className="w-3 h-3 text-brand-light/70 animate-spin" />
                    ) : (
                      <LuDownload className="w-3 h-3 text-brand-light/70" />
                    )}
                    <span>
                      {startDownloading ? "Downloading..." : "Download Model"}
                    </span>
                  </button>
                  {startDownloading && (
                    <button
                      type="button"
                      onClick={async () => {
                        setStartDownloading(false);
                        if (!jobId) return;
                        try {
                          await cancelRayJob(jobId);
                        } catch {}
                        try {
                          clearDownloadTracking({ jobId, source: path.path });
                        } catch {}
                      }}
                      className="w-full mt-2 text-[10px] text-brand-light/80 hover:text-brand-light font-medium bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded-[6px] px-2 py-2 transition-all"
                    >
                      Reset download state
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
    </div>
  );
};

const DownloadProgressSection: React.FC<{
  jobUpdates: UnifiedDownloadWsUpdate[];
  onCancel: () => void;
  onReset?: () => void;
  width: number;
}> = ({ jobUpdates, onCancel, width }) => {
  if (!jobUpdates.length) return null;
  const [isCancelling] = useState(false);

  const files = useMemo(
    () => unifiedDownloadWsUpdatesToFiles(jobUpdates),
    [jobUpdates],
  );
  if (!files.length) return null;

  return (
    <div className="w-full mt-3">
      <div className="flex flex-col gap-y-2">
        {files.map((f: any) => (
          <div key={f.filename} className="flex flex-col gap-y-1">
            <div className="flex items-center justify-between gap-x-2 w-full">
              <div className="flex-1 min-w-0">
                <div
                  style={{
                    maxWidth: `${width}px`,
                  }}
                  className="text-[10px] text-brand-light/80 font-mono break-all"
                >
                  {f.filename}
                </div>
              </div>
                            <div className="text-[10px] text-brand-light/80 font-mono shrink-0">
                {(() => {
                  const pct = f.totalBytes
                    ? ((f.downloadedBytes || 0) / f.totalBytes) * 100
                    : typeof f.progress === "number"
                      ? f.progress * 100
                      : 0;
                  return `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`;
                })()}
              </div>
            </div>
            <ProgressBar
              percent={(() => {
                const pct = f.totalBytes
                  ? ((f.downloadedBytes || 0) / f.totalBytes) * 100
                  : typeof f.progress === "number"
                    ? f.progress * 100
                    : 0;
                return Math.max(0, Math.min(100, pct));
              })()}
            />
            <div className="flex items-center justify-between">
              {typeof f.downloadedBytes === "number" &&
              typeof f.totalBytes === "number" ? (
                <div className="text-[10px] text-brand-light/90">
                  {formatDownloadProgress(f.downloadedBytes, f.totalBytes)}
                </div>
              ) : (
                <div />
              )}
              {f.status === "completed" || f.status === "complete" ? (
                <div className="text-[10px] text-brand-light/90">Completed</div>
              ) : (
                <div className="text-[9px] text-brand-light/60">
                  {f.downloadSpeed != null && f.downloadSpeed > 0
                    ? formatSpeed(f.downloadSpeed)
                    : ""}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-x-2 mt-2 w-full">
        <button
          onClick={onCancel}
          className={cn(
            "text-[10px] text-brand-light/90 w-full flex items-center justify-center gap-x-1.5 font-medium hover:text-brand-light transition-all duration-200 bg-brand hover:bg-brand/70 border border-brand-light/10 rounded-[6px] px-2 py-2",
          )}
        >
          {isCancelling ? <LuLoader className="w-3.5 h-3.5 text-brand-light/70 animate-spin" /> : <TbCancel className="w-3.5 h-3.5 text-brand-light/70" />}
          <span>{isCancelling ? "Cancelling..." : "Cancel"}</span>
        </button>
        
      </div>
    </div>
  );
};



export const LoraCard: React.FC<{
  lora: LoraType;
  manifestId: string;
  loraIndex: number;
}> = ({ lora, loraIndex, manifestId }) => {
  const [expanded, setExpanded] = useState(false);
  const isDownloaded = lora.is_downloaded;
  const loraCarRef = useRef<HTMLDivElement>(null);
  return (
    <Collapsible open={expanded} onOpenChange={setExpanded}
    ref={loraCarRef}
    className="bg-brand border border-brand-light/10 rounded-md text-start"
  >
    <CollapsibleTrigger className="w-full">
      <div className="w-full flex items-center justify-between p-3">
        <div className="flex items-center gap-x-2 justify-between w-fit">
          <div className={cn("bg-brand-background border rounded-full p-0.5", isDownloaded ? "bg-green-600/30 border-green-400/40" : "bg-brand-background border-brand-light/20")}>
          {isDownloaded ? <LuCheck className="text-green-400 w-3 h-3" /> : <LuDownload className="text-brand-light/70 w-3 h-3" />}
          </div>
          <h3 className="text-brand-light text-[12px] font-medium">{lora.label}</h3>
        </div>
        <div className="flex items-center gap-x-2 animate-in fade-in-0 duration-300">
          {
             <>
            {expanded ? <LuChevronDown className="text-brand-light w-3 h-3" /> : <LuChevronRight className="text-brand-light w-3 h-3" />}
            </>
          }
        </div>
      </div>
    </CollapsibleTrigger>
    <CollapsibleContent>
      <LoraSection lora={lora} loraIndex={loraIndex} manifestId={manifestId} />
    </CollapsibleContent>
   
  </ Collapsible>
  );
}

const LoraSection: React.FC<{
  lora: LoraType;
  loraIndex: number;
  manifestId: string;
}> = ({ lora, loraIndex, manifestId }) => {
  const path = lora.source || lora.remote_source;
  const scale = lora.scale;
  const isDownloaded = lora.is_downloaded;
  const queryClient = useQueryClient();
  const [isDeleting, setIsDeleting] = useState(false);
  const [startDownloading, setStartDownloading] = useState(false);
  const {
    addSourceToJobId,
    getSourceToJobId,
    getJobUpdates,
    removeJobUpdates,
    removeSourceByJobId,
    clearDownloadTracking,
    addJobIdToParts,
    addJobIdToManifestId,
  } = useDownloadJobIdStore();
  const { mutate: startDownload } = useStartUnifiedDownloadMutation({
    onSuccess(data, variables) {
      addSourceToJobId(variables.source, data.job_id);
      addJobIdToParts(data.job_id, [`spec.loras.${loraIndex}`, `downloaded`]);
      addJobIdToManifestId(data.job_id, manifestId);
    },
  });
  const jobId = getSourceToJobId(path as string);
  const jobUpdates = getJobUpdates(jobId);
  const isDownloading = (jobUpdates?.length ?? 0) > 0;
  const ref = useRef<HTMLDivElement>(null);
  const width = ref.current?.clientWidth ?? 0;


  useEffect(() => {
    if (isDownloading || !jobId) {
      setStartDownloading(false);
    }
  }, [isDownloading, jobId]);

  useEffect(() => {
    if (jobId && !jobUpdates?.length) {
      setStartDownloading(true);
    }
  }, []);

  // If we never receive WS updates for this job, revert to idle so the UI can't get stuck.
  useEffect(() => {
    if (!jobId) return;
    if (!startDownloading) return;
    if ((jobUpdates?.length ?? 0) > 0) return;
    const t = window.setTimeout(() => {
      setStartDownloading(false);
    }, STARTUP_TIMEOUT_MS);
    return () => window.clearTimeout(t);
  }, [jobId, jobUpdates?.length, startDownloading]);


  return (
    <div>
      <div className="px-4 pb-4">
      <div className="space-y-2 mt-1">
        <div
          ref={ref}
          className="bg-brand-background border border-brand-light/10 rounded-md p-3 overflow-hidden w-full"
        >
          <div className="text-brand-light text-[10.5px] font-medium mb-1">
            Path
          </div>
          <div className="text-brand-light text-[10px] font-mono break-all">
            {path || "—"}
          </div>
          {typeof scale === "number" && (
            <div className="text-brand-light text-[10.5px] w-full  flex gap-x-1 mt-2.5  py-1.5 pt-2 border-t border-brand-light/10 flex-col">
              <div className="text-brand-light font-medium mb-1">
                LoRA Specifications
              </div>
              <div className="flex gap-x-1.5 text-[10px]">
                <span className="text-brand-light/70 font-medium block">
                  Weight Scale
                </span>{" "}
                <span className="text-brand-light font-mono block">
                  {scale}
                </span>
              </div>
            </div>
          )}
      {isDownloaded && path && (
      <div className="flex flex-row items-center justify-between gap-x-2">
        <div className="text-[11px] font-medium text-brand-light/90 mt-4 mb-1.5 flex items-center justify-start gap-x-1">
          <LuCheck className="w-3 h-3 text-green-400" />
          <span>Downloaded</span>
        </div>
        <button
          onClick={async () => {
            setIsDeleting(true);
            await deleteDownload({
              path: path as string,
              item_type: "lora",
            }); 
            await Promise.all([
              refreshManifestPart(manifestId, `spec.loras.${loraIndex}`, queryClient),
              refreshManifestPart(manifestId, `downloaded`, queryClient, true),
            ]);
            setIsDeleting(false);
          }}
          className={cn("text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 disabled:opacity-60 disabled:cursor-not-allowed bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-[6px] px-3 py-1.5 transition-all", isDeleting ? "opacity-60 cursor-default hover:bg-brand" : "")}
        >
          {isDeleting ? <LuLoader className="w-3 h-3 text-brand-light/70 animate-spin" /> : <LuTrash className="w-3.5 h-3.5" />}
          <span>{isDeleting ? "Deleting..." : "Delete"}</span>
        </button>
      </div>
    )}
    {!isDownloaded && (
      <div className="flex flex-row items-center justify-between gap-x-2">
        {isDownloading ? (
          <DownloadProgressSection
            jobUpdates={jobUpdates ?? []}
            width={width}
            onReset={() => {
              setStartDownloading(false);
              clearDownloadTracking({ jobId, source: path as string });
            }}
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
            }}
          />
        ) : (
          <div className="w-full">
            <button
              disabled={!path || startDownloading || isDownloading}
              onClick={async () => {
                if (!path) return;
                setStartDownloading(true);
                await startDownload({
                  item_type: "lora",
                  source: path as string,
                  manifest_id: manifestId,
                  lora_name: lora.name,
                });
              }}
              className={cn(
                "w-full mt-3 text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-md px-3 py-2 transition-all disabled:opacity-60 disabled:cursor-not-allowed",
                startDownloading ? "opacity-60 cursor-default hover:bg-brand" : "",
              )}
            >
              {startDownloading ? (
                <LuLoader className="w-3 h-3 text-brand-light/70 animate-spin" />
              ) : (
                <LuDownload className="w-3 h-3 text-brand-light/70" />
              )}
              <span>{startDownloading ? "Downloading..." : "Download Lora"}</span>
            </button>
            {startDownloading && (
              <button
                type="button"
                onClick={() => {
                  setStartDownloading(false);
                  clearDownloadTracking({ jobId, source: path as string });
                }}
                className="w-full mt-2 text-[10px] text-brand-light/80 hover:text-brand-light font-medium bg-brand-background hover:bg-brand-background/70 border border-brand-light/10 rounded-[6px] px-2 py-2 transition-all"
              >
                Reset download state
              </button>
            )}
          </div>
        )}
      </div>
    )}
    </div>
    
    </div>
    
    </div>
    
    </div>
  );
}


export default ComponentCard;
