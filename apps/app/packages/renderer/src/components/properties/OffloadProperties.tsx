import React, { useMemo, useCallback, useRef, useState } from "react";
import { useClipStore } from "@/lib/clip";
import type { ModelClipProps } from "@/lib/types";
import type { ManifestComponent } from "@/lib/manifest/api";
import { cn } from "@/lib/utils";
import Input from "./Input";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Check, ChevronDown } from "lucide-react";
import { LuInfo } from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useManifestQuery } from "@/lib/manifest/queries";

interface OffloadPropertiesProps {
  clipId: string;
}

const formatComponentName = (name: string): string => {
  return String(name || "")
    .replace(/\./g, " ")
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

type OffloadLevel = "leaf" | "block";

const OFFLOAD_LEVELS: Array<{
  value: OffloadLevel;
  name: string;
  description: string;
}> = [
  {
    value: "leaf",
    name: "Leaf level",
    description: "Offload at the leaf/module level for finer granularity.",
  },
  {
    value: "block",
    name: "Block level",
    description:
      "Offload by blocks for fewer transfers; tune the number of blocks.",
  },
];

const OffloadLevelDropdown: React.FC<{
  value: OffloadLevel;
  onChange: (next: OffloadLevel) => void;
}> = ({ value, onChange }) => {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const selected = useMemo(() => {
    return OFFLOAD_LEVELS.find((o) => o.value === value) || OFFLOAD_LEVELS[0];
  }, [value]);

  return (
    <div className="w-full min-w-0">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            ref={triggerRef}
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className={cn(
              "w-full min-w-0 overflow-hidden justify-between rounded-[6px] dark bg-brand-background-light/60 border-0 shadow-none px-3 text-brand-light text-[11px] h-12 hover:bg-brand-background-light/70",
            )}
            onClick={() => setOpen(true)}
          >
            <div className="flex flex-col text-left min-w-0 w-0 flex-1 gap-y-0.5 overflow-hidden">
              <span className="block truncate font-medium">
                {selected?.name || "Select offload level"}
              </span>
              {selected?.description && (
                <span className="block truncate text-[10px] text-brand-light/70 font-normal">
                  {selected.description}
                </span>
              )}
            </div>
            <div className="flex flex-col items-center shrink-0">
       
              <ChevronDown className="h-4q! w-3! opacity-50" />
            </div>
          </Button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          className="p-0 bg-brand-background border-brand-light/10 font-poppins"
          style={{ width: triggerRef.current?.offsetWidth || 320 }}
        >
          <Command className="bg-brand-background">
           <div className="text-brand-light text-[11px] font-medium px-3 py-2 border-b border-brand-light/10">
            Offload level
           </div>
            <CommandList>
              <CommandEmpty className="text-brand-light/40 p-3 text-xs">
                No level found.
              </CommandEmpty>
              <CommandGroup className="px-1 py-1">
                {OFFLOAD_LEVELS.map((opt) => {
                  const isSelected = opt.value === value;
                  return (
                    <CommandItem
                      key={opt.value}
                      value={opt.value}
                      onSelect={(val) => {
                        const picked =
                          OFFLOAD_LEVELS.find(
                            (o) => String(o.value) === String(val),
                          ) || OFFLOAD_LEVELS[0];
                        onChange(picked.value);
                        setOpen(false);
                      }}
                      className="flex items-center gap-2 px-2 py-2 hover:bg-brand-light/5 rounded-sm "
                    >
                      <div className="h-4 w-4 mt-0.5">
                        {isSelected ? (
                          <Check className="h-4 w-4 text-brand-lighter" />
                        ) : (
                          <span className="inline-block h-4 w-4" />
                        )}
                      </div>
                      <div className="flex flex-col min-w-0 max-w-full overflow-hidden">
                        <span className="block text-brand-light text-[11.5px] font-medium truncate">
                          {opt.name}
                        </span>
                        <span className="block text-[10px] text-brand-light/70 truncate">
                          {opt.description}
                        </span>
                      </div>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
};

interface ToggleProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  ariaLabel?: string;
}

const Toggle: React.FC<ToggleProps> = ({ checked, onChange, disabled, ariaLabel }) => {
  return (
    <button
      type="button"
      aria-pressed={checked}
      aria-label={ariaLabel || (checked ? "Disable" : "Enable")}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none",
        checked ? "bg-blue-600" : "bg-brand-background border border-brand-light/10",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 transform rounded-full bg-brand-light shadow transition-transform",
          checked ? "translate-x-4.5" : "translate-x-0.5",
        )}
      />
    </button>
  );
};

const InfoTip: React.FC<{ text: string; disabled?: boolean }> = ({
  text,
  disabled,
}) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "text-brand-light/70 hover:text-brand-light focus:outline-none",
            disabled && "opacity-40 cursor-not-allowed hover:text-brand-light/70",
          )}
        >
          <LuInfo className="w-3 h-3" />
        </button>
      </TooltipTrigger>
      <TooltipContent
        sideOffset={6}
        className="max-w-xs whitespace-pre-wrap text-[10px] font-poppins bg-brand-background border border-brand-light/10"
      >
        {text}
      </TooltipContent>
    </Tooltip>
  );
};

const OffloadProperties: React.FC<OffloadPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | ModelClipProps
    | undefined;

  if (!clip || clip.type !== "model") return null;

  const manifestId = String(clip.manifest?.metadata?.id || "");
  const { data: manifest } = useManifestQuery(manifestId);
  if (!manifest) return null;

  const components = useMemo(() => {
    const comps = (manifest?.spec?.components || []) as ManifestComponent[];
    // Show the same “component list” as architecture (model_path-based), but exclude scheduler explicitly.
    return comps.filter((c) => (c as any)?.type !== "scheduler" && !!(c as any)?.model_path);
  }, [manifest]);

  const getComponentKey = useCallback((comp: ManifestComponent): string => {
    return String((comp as any).name || (comp as any).type || "component");
  }, []);

  const getLabel = useCallback((comp: ManifestComponent): string => {
    return formatComponentName(
      (comp as any).label || (comp as any).name || (comp as any).base || (comp as any).type,
    );
  }, []);

  const getOffloadState = useCallback(
    (key: string) => {
      const map = (clip.offload || {}) as NonNullable<ModelClipProps["offload"]>;
      const existing = map[key] || {};
      const enabled = !!existing.enabled;
      const level: OffloadLevel = (existing.level as OffloadLevel) || "leaf";
      // Defaults: stream + record + low CPU memory ON unless explicitly disabled
      const use_stream = existing.use_stream ?? true;
      const record_stream = existing.record_stream ?? true;
      const low_cpu_mem_usage = existing.low_cpu_mem_usage ?? true;
      const num_blocks_raw = typeof existing.num_blocks === "number" ? existing.num_blocks : 1;
      const num_blocks = Math.max(1, Math.floor(Number.isFinite(num_blocks_raw) ? num_blocks_raw : 1));

      return { enabled, level, num_blocks, use_stream, record_stream, low_cpu_mem_usage };
    },
    [clip.offload],
  );

  const patchOffload = useCallback((key: string, patch: Partial<NonNullable<ModelClipProps["offload"]>[string]>) => {
    const store = useClipStore.getState();
    const current = store.getClipById(clipId) as ModelClipProps | undefined;
    if (!current || current.type !== "model") return;

    const prev = (current.offload || {}) as NonNullable<ModelClipProps["offload"]>;
    const prevEntry = prev[key] || {};

    let nextEntry = { ...prevEntry, ...patch };

    // Normalize/clamp
    const useStream = nextEntry.use_stream ?? true;
    const level = (nextEntry.level as OffloadLevel) || "leaf";
    let numBlocks = typeof nextEntry.num_blocks === "number" ? nextEntry.num_blocks : 1;
    numBlocks = Math.max(1, Math.floor(Number.isFinite(numBlocks) ? numBlocks : 1));
    const lowCpuMemUsage = nextEntry.low_cpu_mem_usage ?? true;

    if (useStream) {
      numBlocks = 1;
      // Default record_stream ON when streaming is enabled (unless explicitly disabled)
      nextEntry.record_stream = nextEntry.record_stream ?? true;
    }

    nextEntry.level = level;
    nextEntry.num_blocks = numBlocks;
    nextEntry.low_cpu_mem_usage = lowCpuMemUsage;

    // If stream is disabled, keep record_stream but it will be ignored by the engine unless stream is on
    nextEntry.use_stream = useStream;

    const next = { ...prev, [key]: nextEntry };
    store.updateClip(clipId, { offload: next } as any);
  }, [clipId]);

  if (!components.length) return null;

  return (
    <div>
        <h3 className="text-brand-light text-[12px] font-medium text-start ml-4 mt-4">
            Offload
        </h3>
        <p className="text-brand-light/60 text-[10px] font-normal text-start ml-4 mt-1">
            Configure offloading for each component in the model.
        </p>
    <div className="flex flex-col p-4 justify-start items-stretch w-full">
      <div className="flex flex-col gap-3 w-full">
        {components.map((comp, idx) => {
          const key = getComponentKey(comp);
          const label = getLabel(comp);
          const state = getOffloadState(key);

          return (
            <div
              key={`${key}-${idx}`}
              className="flex flex-col gap-y-2 p-3 rounded-[6px] bg-brand/70 border border-brand-light/5"
            >
              <div className="flex flex-row items-center justify-between w-full gap-x-3 min-w-0">

                  <div className="text-brand-light text-[11px] font-medium truncate">
                    {label}
                  </div>
                <Toggle
                  checked={state.enabled}
                  onChange={(next) => patchOffload(key, { enabled: next })}
                  ariaLabel={state.enabled ? "Disable offloading" : "Enable offloading"}
                />
              </div>

              {state.enabled && (
                <div className="flex flex-col gap-y-3">
                  <div className="flex flex-col gap-y-1 w-full min-w-0">
                    
                    <OffloadLevelDropdown
                      value={state.level}
                      onChange={(next) => patchOffload(key, { level: next })}
                    />
                  </div>

                  {state.level === "block" && (
                    <div className={cn("flex flex-col gap-1", state.use_stream && "opacity-70")}>
                      <Input
                        className="bg-brand-background h-7"
                        stepClassName="h-7 bg-brand-background-light/60"
                        label="Number of blocks"
                        description="How many blocks to split the component into for block-level offloading. More blocks can reduce peak memory per block, but may increase overhead."
                        startLogo="N"
                        value={String(state.num_blocks)}
                        onChange={(v) => {
                          if (state.use_stream) {
                            // Hard constraint: streaming requires a single block
                            patchOffload(key, { num_blocks: 1 });
                            return;
                          }
                          const parsed = Math.max(1, Math.floor(Number(v)));
                          patchOffload(key, { num_blocks: Number.isFinite(parsed) ? parsed : 1 });
                        }}
                        canStep
                        step={1}
                        min={1}
                      />
                      {state.use_stream && (
                        <div className="text-[10px] text-brand-light/60 text-start">
                          When <span className="text-brand-light/80 font-medium">Use stream</span> is enabled,{" "}
                          <span className="text-brand-light/80 font-medium">Number of blocks</span> is fixed to 1.
                        </div>
                      )}
                    </div>
                  )}

                  <div className="flex flex-col gap-y-2">
                    <div className="flex flex-row items-center justify-between w-full gap-x-3">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <div className="text-brand-light text-[11px] font-medium">
                          Use stream
                        </div>
                        <InfoTip text="Enable streaming mode for offloading. When enabled, the engine streams data instead of staging multiple blocks; this requires Number of blocks to be 1." />
                      </div>
                      <Toggle
                        checked={state.use_stream}
                        onChange={(next) =>
                          patchOffload(key, {
                            use_stream: next,
                            num_blocks: next ? 1 : state.num_blocks,
                            record_stream: next ? (state.record_stream ?? true) : state.record_stream,
                          })
                        }
                        ariaLabel={state.use_stream ? "Disable use stream" : "Enable use stream"}
                      />
                    </div>

                    <div className="flex flex-row items-center justify-between w-full gap-x-3">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <div
                          className={cn(
                            "text-[11px] font-medium",
                            state.use_stream ? "text-brand-light" : "text-brand-light/40",
                          )}
                        >
                          Record stream
                        </div>
                        <InfoTip
                          disabled={!state.use_stream}
                          text="Record stream events for debugging/inspection. Requires Use stream to be enabled."
                        />
                      </div>
                      <Toggle
                        checked={state.record_stream}
                        disabled={!state.use_stream}
                        onChange={(next) => patchOffload(key, { record_stream: next })}
                        ariaLabel={
                          state.record_stream
                            ? "Disable record stream"
                            : "Enable record stream"
                        }
                      />
                    </div>

                    <div className="flex flex-row items-center justify-between w-full gap-x-3">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <div className="text-brand-light text-[11px] font-medium">
                          Low CPU Memory
                        </div>
                        <InfoTip text="Reduce CPU RAM usage during offloading by using a lower-memory pathway. This can be slightly slower, but helps avoid CPU OOMs on large models." />
                      </div>
                      <Toggle
                        checked={state.low_cpu_mem_usage}
                        onChange={(next) => patchOffload(key, { low_cpu_mem_usage: next })}
                        ariaLabel={
                          state.low_cpu_mem_usage
                            ? "Disable low CPU memory"
                            : "Enable low CPU memory"
                        }
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
    </div>
  );
};

export default OffloadProperties;

